import argparse
import json
import os
import random
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


def seed_torch(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class DlMriDataset(Dataset):
    def __init__(
        self,
        images: List[np.ndarray],
        labels: List[int],
        case_ids: List[str],
        transform,
        cli: Optional[List[np.ndarray]] = None,
    ) -> None:
        self.images = images
        self.labels = labels
        self.case_ids = case_ids
        self.transform = transform
        self.cli = cli

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        label = self.labels[idx]
        case_id = self.case_ids[idx]

        img1 = img[:, :, :3]
        img2 = img[:, :, 3:6]
        img3 = img[:, :, 6:]

        img1_pil = Image.fromarray((img1 * 255).astype(np.uint8))
        img2_pil = Image.fromarray((img2 * 255).astype(np.uint8))
        img3_pil = Image.fromarray((img3 * 255).astype(np.uint8))

        img1_t = self.transform(img1_pil)
        img2_t = self.transform(img2_pil)
        img3_t = self.transform(img3_pil)
        img_t = torch.cat([img1_t, img2_t, img3_t], dim=0)

        if self.cli is None:
            return img_t, torch.tensor(label, dtype=torch.float32), case_id

        cli_t = torch.tensor(self.cli[idx].astype(np.float32), dtype=torch.float32)
        return img_t, torch.tensor(label, dtype=torch.float32), case_id, cli_t


def make_balanced_weights(labels: List[int]) -> torch.DoubleTensor:
    arr = np.array(labels)
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(arr), y=arr
    )
    weights = [class_weights[int(y)] for y in arr]
    return torch.DoubleTensor(weights)


class SeqVITRad(nn.Module):
    def __init__(self, sequence_length: int = 3):
        super().__init__()
        self.sequence_length = sequence_length
        self.backbone = timm.create_model("resnet50", pretrained=True, num_classes=0)
        self.dropout = nn.Dropout(0.1)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(
            self.backbone.num_features * sequence_length, self.backbone.num_features
        )
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(
            self.backbone.num_features, self.backbone.num_features // 8
        )
        self.classifier = nn.Linear(self.backbone.num_features // 8, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = []
        for t in range(self.sequence_length):
            chunk = inputs[:, t * self.sequence_length : (t + 1) * self.sequence_length, :, :]
            outputs.append(self.backbone(chunk))
        x = torch.cat(outputs, dim=-1)
        x = self.flatten(x)
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.relu(self.linear2(x))
        x = self.dropout(x)
        return torch.sigmoid(self.classifier(x))


class SeqVITRadCli(nn.Module):
    def __init__(self, sequence_length: int = 3, clinical_feature_dim: int = 11):
        super().__init__()
        self.sequence_length = sequence_length
        self.backbone = timm.create_model("resnet50", pretrained=True, num_classes=0)
        self.dropout = nn.Dropout(0.1)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(
            self.backbone.num_features * sequence_length, self.backbone.num_features
        )
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(
            self.backbone.num_features, self.backbone.num_features // 8
        )
        self.clinical_fc = nn.Sequential(
            nn.Linear(clinical_feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
        )
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.classifier = nn.Linear(self.backbone.num_features // 8 + 64, 1)

    def forward(self, inputs: torch.Tensor, cli: torch.Tensor) -> torch.Tensor:
        outputs = []
        for t in range(self.sequence_length):
            chunk = inputs[:, t * self.sequence_length : (t + 1) * self.sequence_length, :, :]
            outputs.append(self.backbone(chunk))
        x = torch.cat(outputs, dim=-1)
        x = self.flatten(x)
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.relu(self.linear2(x))
        x = self.dropout(x)

        cli_emb = self.alpha * self.clinical_fc(cli)
        x = torch.cat((x, cli_emb), dim=1)
        return self.classifier(x)


def preprocess_case_triplet(
    case_id: str,
    data_dir: str,
    img_size: int = 224,
    outline: int = 10,
) -> Optional[np.ndarray]:
    mask_dir = os.path.join(data_dir, "mask")
    ori_dir = os.path.join(data_dir, "ori")
    images = []

    for suffix in ["Ap", "PVP", "DP"]:
        x_name = f"{case_id}_{suffix}.npy"
        m_name = f"{case_id}_{suffix}.npy"
        x_path = os.path.join(ori_dir, x_name)
        m_path = os.path.join(mask_dir, m_name)
        if not (os.path.exists(x_path) and os.path.exists(m_path)):
            return None

        x = np.load(x_path)
        m = np.load(m_path)

        x = np.uint8(cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX))
        if len(x.shape) == 3 and x.shape[2] == 3:
            x_gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        elif len(x.shape) == 2:
            x_gray = x
        else:
            return None

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        x_eq = clahe.apply(x_gray)
        x_eq = (x_eq - np.min(x_eq)) / (np.max(x_eq) - np.min(x_eq) + 1e-8)

        pos = np.where(m > 0)
        if len(pos[0]) == 0:
            return None

        w0, h0 = m.shape
        x_min = max(0, int(np.min(pos[0]) - outline))
        x_max = min(w0, int(np.max(pos[0]) + outline))
        y_min = max(0, int(np.min(pos[1]) - outline))
        y_max = min(h0, int(np.max(pos[1]) + outline))

        x_crop = x_eq[x_min:x_max, y_min:y_max]
        if x_crop.shape[0] != img_size or x_crop.shape[1] != img_size:
            x_crop = cv2.resize(x_crop, (img_size, img_size), interpolation=cv2.INTER_CUBIC)

        x_crop = (x_crop - np.min(x_crop)) / (np.max(x_crop) - np.min(x_crop) + 1e-8)
        x_crop = np.expand_dims(x_crop, axis=-1)
        x_crop = np.concatenate([x_crop, x_crop, x_crop], axis=-1)
        images.append(x_crop)

    return np.concatenate(images, axis=-1)


def load_clinical_features(raw_cli_path: str, case_ids: List[str]) -> Tuple[pd.DataFrame, int]:
    raw = pd.read_excel(raw_cli_path)
    raw = raw[raw["case_id"].astype(str).isin(case_ids)].copy()

    categorical_columns = ["BCLC", "AFP400"]
    numerical_columns = [
        "age",
        "PLT",
        "tumor_dm",
        "Gamma-Glutamyltransferase",
        "Alkaline_Phosphatase",
        "Cirrhosis",
        "NLR",
    ]

    case_id_data = raw["case_id"].astype(str)
    categorical_df = pd.get_dummies(raw[categorical_columns], columns=categorical_columns)

    scaler = StandardScaler()
    numerical_df = pd.DataFrame(
        scaler.fit_transform(raw[numerical_columns]),
        columns=numerical_columns,
        index=raw.index,
    )

    processed = pd.concat([numerical_df, categorical_df], axis=1)
    processed.fillna(processed.mean(numeric_only=True), inplace=True)
    processed = pd.concat([case_id_data.rename("case_id"), processed], axis=1)
    return processed, processed.shape[1] - 1


def build_dataset(
    labels_csv: str,
    data_dir: str,
    raw_cli_path: Optional[str],
    use_cli: bool,
) -> Tuple[List[np.ndarray], List[int], List[str], Optional[List[np.ndarray]], int]:
    df = pd.read_csv(labels_csv)
    df["case_id"] = df["case_id"].astype(str)
    case_ids = df["case_id"].tolist()

    clinical_df = None
    clinical_dim = 0
    if use_cli:
        if raw_cli_path is None:
            raise ValueError("raw_cli_path is required for PRE mode")
        clinical_df, clinical_dim = load_clinical_features(raw_cli_path, case_ids)

    images = []
    labels = []
    kept_case_ids = []
    cli_list: List[np.ndarray] = []

    for _, row in df.iterrows():
        cid = str(row["case_id"])
        sample = preprocess_case_triplet(cid, data_dir)
        if sample is None:
            continue
        images.append(sample)
        labels.append(int(row["class_label"]))
        kept_case_ids.append(cid)

        if use_cli:
            cli_row = clinical_df[clinical_df["case_id"].astype(str) == cid]
            if cli_row.empty:
                images.pop()
                labels.pop()
                kept_case_ids.pop()
                continue
            cli_list.append(cli_row.values[0][1:])

    if use_cli:
        return images, labels, kept_case_ids, cli_list, clinical_dim
    return images, labels, kept_case_ids, None, 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Full training for dl-mri PRE/DLR models")
    parser.add_argument("--model", choices=["pre", "dlr"], required=True)
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--raw_cli_path", default=None)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    seed_torch(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_cli = args.model == "pre"
    images, labels, case_ids, cli_list, clinical_dim = build_dataset(
        labels_csv=args.labels_csv,
        data_dir=args.data_dir,
        raw_cli_path=args.raw_cli_path,
        use_cli=use_cli,
    )

    if len(images) == 0:
        raise RuntimeError("No valid training samples found after preprocessing.")

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=224, scale=(0.9, 1.1)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = DlMriDataset(images, labels, case_ids, train_transform, cli=cli_list)
    weights = make_balanced_weights(labels)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=args.num_workers,
    )

    if use_cli:
        model = SeqVITRadCli(sequence_length=3, clinical_feature_dim=clinical_dim)
    else:
        model = SeqVITRad(sequence_length=3)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss()

    os.makedirs(args.output_dir, exist_ok=True)
    history = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            optimizer.zero_grad()
            if use_cli:
                inputs, labels_t, _, cli_t = batch
                inputs = inputs.to(device)
                labels_t = labels_t.to(device).view(-1, 1)
                cli_t = cli_t.to(device)
                logits = model(inputs, cli_t)
                outputs = torch.sigmoid(logits)
            else:
                inputs, labels_t, _ = batch
                inputs = inputs.to(device)
                labels_t = labels_t.to(device).view(-1, 1)
                outputs = model(inputs)

            loss = criterion(outputs, labels_t)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        history.append({"epoch": epoch, "train_loss": avg_loss})
        print(f"Epoch {epoch + 1}/{args.epochs} - train_loss: {avg_loss:.6f}", flush=True)

    ckpt_name = (
        f"checkpoint_full_{args.model}_lr{args.lr}_wd{args.weight_decay}_epochs{args.epochs}.pt"
    )
    ckpt_path = os.path.join(args.output_dir, ckpt_name)
    torch.save(model.state_dict(), ckpt_path)

    with open(os.path.join(args.output_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    summary = {
        "model": args.model,
        "samples": len(dataset),
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "checkpoint": ckpt_path,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Training completed.")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
