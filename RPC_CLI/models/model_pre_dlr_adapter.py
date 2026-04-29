"""SurvPath-aligned MRI/WSI adapters (PRE / DLR / DLP).

IMPORTANT — manuscript provenance
---------------------------------
Of the three classes below, only ``SurvPathDLPAdapter`` corresponds to a
manuscript-adopted model (DLP).

  - PRE manuscript model = ``SeqVITRadCli`` from ``PRE_DLR/train_dl_mri_full.py``
  - DLR manuscript model = ``SeqVIT``       from ``PRE_DLR/test_DLR_external_split.py``
  - DLP manuscript model = ``SurvPathDLPAdapter`` (this file)

``SurvPathPREAdapter`` and ``SurvPathDLRAdapter`` here are SurvPath-framework
candidate variants that share the same training pipeline as DLP, but were
*not* the adopted PRE / DLR results in the manuscript. They are kept so that
this unified pipeline can still run pre / dlr modalities for users who want
to retrain a SurvPath-style PRE / DLR model from scratch.
"""

import torch
import torch.nn as nn
import timm


class _SeqResNetBackbone(nn.Module):
    """Encode 3-phase MRI input arranged as 9 channels (3xRGB)."""

    def __init__(self, sequence_length: int = 3, pretrained: bool = True):
        super().__init__()
        self.sequence_length = sequence_length
        self.backbone = timm.create_model("resnet50", pretrained=pretrained, num_classes=0)
        self.out_dim = self.backbone.num_features * sequence_length

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        # Accept [B, H, W, 9], [H, W, 9], or [B, 9, H, W].
        if x_img.dim() == 3:
            x_img = x_img.unsqueeze(0)

        if x_img.dim() != 4:
            raise ValueError(f"x_img must be 4D tensor, got shape {tuple(x_img.shape)}")

        if x_img.shape[-1] == 9:
            x = x_img.permute(0, 3, 1, 2).contiguous()
        elif x_img.shape[1] == 9:
            x = x_img
        else:
            raise ValueError(f"x_img must have 9 channels, got shape {tuple(x_img.shape)}")

        x = x.float()
        if x.max() > 1.5:
            x = x / 255.0

        feats = []
        for t in range(self.sequence_length):
            start = t * 3
            end = (t + 1) * 3
            feats.append(self.backbone(x[:, start:end, :, :]))

        return torch.cat(feats, dim=1)


def _load_optional_encoder_weights(encoder: nn.Module, weights_path=None, freeze: bool = False) -> None:
    if weights_path:
        state_dict = torch.load(weights_path, map_location="cpu")
        encoder.load_state_dict(state_dict, strict=False)
    if freeze:
        for param in encoder.parameters():
            param.requires_grad = False


class SurvPathPREAdapter(nn.Module):
    """PRE adapter: MRI + Clinical (no WSI dependency)."""

    def __init__(self, clinical_dim: int = 19, dropout: float = 0.1, pretrained_backbone: bool = False, mri_encoder_weights=None, freeze_mri_encoder: bool = False):
        super().__init__()
        self.encoder = _SeqResNetBackbone(sequence_length=3, pretrained=pretrained_backbone)
        _load_optional_encoder_weights(self.encoder, mri_encoder_weights, freeze_mri_encoder)

        self.img_head = nn.Sequential(
            nn.Linear(self.encoder.out_dim, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 256),
            nn.ReLU(),
        )

        self.clinical_head = nn.Sequential(
            nn.Linear(clinical_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.to_logits = nn.Sequential(
            nn.Linear(320, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, **kwargs) -> torch.Tensor:
        x_img = kwargs["x_img"]
        clinical_data = kwargs.get("clinical_data", None)

        img_feat = self.img_head(self.encoder(x_img))

        if clinical_data is None:
            raise ValueError("PRE adapter requires clinical_data")

        cli = clinical_data.float()
        if cli.dim() == 1:
            cli = cli.unsqueeze(0)

        cli_feat = self.clinical_head(cli)
        fused = torch.cat([img_feat, cli_feat], dim=1)
        return self.to_logits(fused)


class SurvPathDLRAdapter(nn.Module):
    """DLR adapter: MRI-only (no WSI, no clinical branch)."""

    def __init__(self, dropout: float = 0.1, pretrained_backbone: bool = False, mri_encoder_weights=None, freeze_mri_encoder: bool = False):
        super().__init__()
        self.encoder = _SeqResNetBackbone(sequence_length=3, pretrained=pretrained_backbone)
        _load_optional_encoder_weights(self.encoder, mri_encoder_weights, freeze_mri_encoder)
        self.to_logits = nn.Sequential(
            nn.Linear(self.encoder.out_dim, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, **kwargs) -> torch.Tensor:
        x_img = kwargs["x_img"]
        feat = self.encoder(x_img)
        return self.to_logits(feat)


class SurvPathDLPAdapter(nn.Module):
    """DLP adapter: WSI-only gated attention MIL head under SurvPath training flow."""

    def __init__(
        self,
        wsi_input_dim: int = 1024,
        hidden_dim: int = 256,
        attn_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(wsi_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.attn_a = nn.Linear(hidden_dim, attn_dim)
        self.attn_b = nn.Linear(hidden_dim, attn_dim)
        self.attn_c = nn.Linear(attn_dim, 1)

        self.to_logits = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, **kwargs) -> torch.Tensor:
        x_wsi = kwargs.get("x_wsi", kwargs.get("data_WSI", None))
        if x_wsi is None:
            raise ValueError("DLP adapter requires x_wsi/data_WSI input.")

        if x_wsi.dim() == 2:
            x_wsi = x_wsi.unsqueeze(0)
        if x_wsi.dim() != 3:
            raise ValueError(f"x_wsi must be [B, N, D] or [N, D], got shape {tuple(x_wsi.shape)}")

        x_wsi = x_wsi.float()
        h = self.encoder(x_wsi)

        a = torch.tanh(self.attn_a(h))
        b = torch.sigmoid(self.attn_b(h))
        attn_scores = self.attn_c(a * b).squeeze(-1)

        mask = kwargs.get("mask", None)
        if mask is not None and isinstance(mask, torch.Tensor):
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            if mask.shape == attn_scores.shape:
                valid = mask <= 0.5
                all_masked = ~valid.any(dim=1)
                attn_scores = attn_scores.masked_fill(~valid, -1e9)
                if all_masked.any():
                    attn_scores[all_masked] = 0.0

        attn = torch.softmax(attn_scores, dim=1)
        pooled = torch.bmm(attn.unsqueeze(1), h).squeeze(1)
        return self.to_logits(pooled)
