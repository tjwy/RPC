import torch
import torch.nn as nn

from models.model_SurvPath_latest1 import FeedForward, MMAttentionLayer, SeqVIT


class SurvPathDLRP(nn.Module):
    """DLRP variant: MRI + WSI only (no clinical branch)."""

    def __init__(
        self,
        wsi_embedding_dim=1024,
        img_embedding_dim=2048,
        dropout=0.25,
        num_classes=2,
        wsi_projection_dim=256,
        image_size=224,
        sequence_length=3,
    ):
        super().__init__()

        self.dropout = dropout
        self.num_classes = num_classes
        self.wsi_embedding_dim = wsi_embedding_dim
        self.wsi_projection_dim = wsi_projection_dim
        self.img_embedding_dim = img_embedding_dim
        self.num_slices = 3

        self.wsi_projection_net = nn.Sequential(
            nn.Linear(self.wsi_embedding_dim, self.wsi_projection_dim),
        )

        self.seq_vit_model = SeqVIT(
            sequence_length=sequence_length,
            image_size=image_size,
            drop_rate=0.1,
        )
        if torch.cuda.is_available():
            self.seq_vit_model = self.seq_vit_model.to("cuda")

        self.img_projection_net = nn.Sequential(
            nn.Linear(self.img_embedding_dim, self.wsi_projection_dim),
        )

        self.identity = nn.Identity()
        self.cross_attender = MMAttentionLayer(
            dim=self.wsi_projection_dim,
            dim_head=self.wsi_projection_dim // 2,
            heads=1,
            residual=False,
            dropout=0.1,
        )

        self.feed_forward = FeedForward(self.wsi_projection_dim // 2, dropout=dropout)
        self.layer_norm = nn.LayerNorm(self.wsi_projection_dim // 2)

        # 128 (WSI pooled) + 128 (MRI pooled) = 256
        self.to_logits = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, **kwargs):
        wsi_features = kwargs["x_wsi"]
        img_data = kwargs["x_img"]

        wsi_proj = self.wsi_projection_net(wsi_features)

        img_features = self.seq_vit_model(img_data)
        img_proj = self.img_projection_net(img_features)

        tokens = torch.cat([img_proj, wsi_proj], dim=1)
        tokens = self.identity(tokens)

        mm_embed = self.cross_attender(x=tokens, mask=None, return_attention=False)
        mm_embed = self.feed_forward(mm_embed)
        mm_embed = self.layer_norm(mm_embed)

        paths_postsa_embed = mm_embed[:, : self.num_slices, :]
        paths_postsa_embed = torch.mean(paths_postsa_embed, dim=1)

        wsi_postsa_embed = mm_embed[:, self.num_slices :, :]
        wsi_postsa_embed = torch.mean(wsi_postsa_embed, dim=1)

        embedding = torch.cat([wsi_postsa_embed, paths_postsa_embed], dim=1)
        logits = self.to_logits(embedding)

        return logits
