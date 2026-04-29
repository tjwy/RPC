#!/usr/bin/env python3
"""External validation for DLR full-training checkpoint (SeqVIT, MRI-only).

This is the canonical DLR inference module used to produce the manuscript
external probability vector. SeqVIT is an MRI-only model (no clinical
features). It is the same architecture as the dl-mri framework's classifier
(`SeqVITRad`) used during training.

Sanitized for public release:
  - All private absolute paths removed; everything is CLI-driven.
  - The internal label-override block (external_labels_final.csv) was
    removed; raw labels in the input CSV are used directly. If you need
    label override at inference time, supply a pre-merged CSV.
"""

import argparse
import os
import random
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def seed_torch(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class SeqVIT(nn.Module):
    """MRI-only model used by the canonical DLR pipeline."""

    def __init__(self, sequence_length: int = 3):
        super().__init__()
        self.sequence_length = sequence_length
        self.vit_base_model = timm.create_model("resnet50", pretrained=False, num_classes=0)
        self.dropout = nn.Dropout(0.1)
        self.flatten = nn.Flatten()
        self.linear_layer = nn.Linear(
            self.vit_base_model.num_features * sequence_length,
            self.vit_base_model.num_features,
        )
        self.activation = nn.ReLU()
        self.linear_layer2 = nn.Linear(
            self.vit_base_model.num_features,
            self.vit_base_model.num_features // 8,
        )
        self.classifier = nn.Linear(self.vit_base_model.num_features // 8, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = []
        for t in range(self.sequence_length):
            chunk = inputs[:, t * 3 : (t + 1) * 3, :, :]
            outputs.append(self.vit_base_model(chunk))
        x = torch.cat(outputs, dim=-1)
        x = self.flatten(x)
        x = self.activation(self.linear_layer(x))
        x = self.dropout(x)
        x = self.activation(self.linear_layer2(x))
        x = self.dropout(x)
        x = self.classifier(x)
        return torch.sigmoid(x)


class MRIDataset(Dataset):
    def __init__(self, external_df: pd.DataFrame, mri_base_dirs: Dict[str, str]):
        self.df = external_df.reset_index(drop=True)
        self.mri_base_dirs = mri_base_dirs

        self.phase_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def _read_phase_png(self, path: str) -> np.ndarray:
        if not os.path.exists(path):
            return np.zeros((224, 224, 3), dtype=np.float32)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros((224, 224, 3), dtype=np.float32)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

        if np.max(img) - np.min(img) > 0:
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
        else:
            img = np.zeros_like(img, dtype=np.float32)

        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)

        if np.max(img) - np.min(img) > 0:
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
        else:
            img = np.zeros_like(img, dtype=np.float32)

        img = np.expand_dims(img, axis=-1)
        img = np.concatenate([img, img, img], axis=-1)
        return img.astype(np.float32)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        case_id = str(row["case_id"]).strip()
        label = int(row["label"])
        group = str(row["group"]).strip()
        cohort = str(row["cohort"]).strip() if "cohort" in row.index else ""
        slide_id = (
            str(row["slide_id"]).strip()
            if "slide_id" in row.index and pd.notna(row["slide_id"])
            else case_id
        )

        if group == "test":
            npy_path = os.path.join(self.mri_base_dirs["test_npy"], f"{slide_id}_rad_stacked.npy")
            if os.path.exists(npy_path):
                arr = np.load(npy_path).astype(np.float32)
                if arr.shape == (224, 224, 9):
                    arr = np.transpose(arr, (2, 0, 1))
                if arr.shape == (9, 224, 224):
                    mri = torch.from_numpy(arr)
                    normalize = transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    )
                    for i in range(3):
                        mri[i * 3 : (i + 1) * 3] = normalize(mri[i * 3 : (i + 1) * 3])
                    return mri, label, case_id, group, cohort
            return torch.zeros(9, 224, 224), label, case_id, group, cohort

        if group == "TCGA":
            mri_dir = self.mri_base_dirs["TCGA"]
            file_id = case_id
        else:
            mri_dir = self.mri_base_dirs["1"]
            if "MR_ID" in row.index and pd.notna(row["MR_ID"]):
                try:
                    file_id = str(int(float(row["MR_ID"])))
                except Exception:
                    file_id = str(row["MR_ID"]).strip()
            else:
                file_id = case_id

        phase_arrays: List[np.ndarray] = []
        for phase in ["AP", "PVP", "DP"]:
            path = os.path.join(mri_dir, f"{file_id}_{phase}.png")
            phase_arrays.append(self._read_phase_png(path))

        tensors = []
        for phase_img in phase_arrays:
            img_pil = Image.fromarray((phase_img * 255).astype(np.uint8))
            tensors.append(self.phase_transform(img_pil))
        mri = torch.cat(tensors, dim=0)
        return mri, label, case_id, group, cohort


def ppv(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def npv(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true, y_prob))
    if len(y_true) == 0:
        return {"AUC": auc, "Accuracy": float("nan"), "Precision": float("nan"),
                "Recall": float("nan"), "Specificity": float("nan"),
                "PPV": float("nan"), "NPV": float("nan"),
                "TN": 0, "FP": 0, "FN": 0, "TP": 0}
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "AUC": auc,
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "Specificity": float(specificity(y_true, y_pred)),
        "PPV": float(ppv(y_true, y_pred)),
        "NPV": float(npv(y_true, y_pred)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def summarize_subset(df: pd.DataFrame, subset_name: str) -> Dict[str, float]:
    y_true = df["true_label"].to_numpy(dtype=int)
    y_pred = df["predicted_label"].to_numpy(dtype=int)
    y_prob = df["probability"].to_numpy(dtype=float)
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    out = {
        "subset": subset_name,
        "n_samples": int(len(df)),
        "n_positive": int((y_true == 1).sum()),
        "n_negative": int((y_true == 0).sum()),
    }
    out.update(metrics)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="External inference for DLR full-training checkpoint")
    parser.add_argument("--external_csv", required=True,
                        help="External CSV with columns: case_id, group, label, slide_id (optional), MR_ID (optional), cohort (optional)")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--group1_mri_dir", required=True,
                        help="Directory containing {MR_ID}_{AP|PVP|DP}.png for group=='1'")
    parser.add_argument("--tcga_mri_dir", default="",
                        help="Directory containing {case_id}_{AP|PVP|DP}.png for group=='TCGA'")
    parser.add_argument("--test_npy_dir", default="",
                        help="Directory containing {slide_id}_rad_stacked.npy for group=='test'")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_torch(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("DLR (SeqVIT, MRI-only) external inference")
    print("=" * 80)

    ext_df = pd.read_csv(args.external_csv)
    if "group" not in ext_df.columns:
        ext_df["group"] = "test"
    ext_df = ext_df[ext_df["group"].astype(str).isin(["1", "test", "TCGA"])].copy()
    ext_df["case_id"] = ext_df["case_id"].astype(str)
    ext_df["label"] = ext_df["label"].astype(int)
    print(f"Loaded external samples: {len(ext_df)} (label pos={int(ext_df['label'].sum())})")

    mri_base_dirs = {
        "1": args.group1_mri_dir,
        "TCGA": args.tcga_mri_dir,
        "test_npy": args.test_npy_dir,
    }

    dataset = MRIDataset(ext_df, mri_base_dirs)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SeqVIT(sequence_length=3).to(device)
    ckpt = torch.load(args.checkpoint_path, map_location=device)

    mapped_ckpt = {}
    for k, v in ckpt.items():
        if k.startswith("backbone."):
            mapped_ckpt[k.replace("backbone.", "vit_base_model.")] = v
        elif k.startswith("linear1."):
            mapped_ckpt[k.replace("linear1.", "linear_layer.")] = v
        elif k.startswith("linear2."):
            mapped_ckpt[k.replace("linear2.", "linear_layer2.")] = v
        else:
            mapped_ckpt[k] = v

    load_res = model.load_state_dict(mapped_ckpt, strict=False)
    print(f"Loaded checkpoint: {args.checkpoint_path}")
    print(f"missing={len(load_res.missing_keys)} unexpected={len(load_res.unexpected_keys)}")

    model.eval()

    all_rows = []
    with torch.no_grad():
        done = 0
        total = len(dataset)
        for mri, labels, case_ids, groups, cohorts in dataloader:
            mri = mri.to(device)
            probs = model(mri).detach().cpu().numpy().reshape(-1)
            for i in range(len(probs)):
                prob = float(probs[i])
                pred = int(prob >= args.threshold)
                all_rows.append({
                    "case_id": str(case_ids[i]),
                    "true_label": int(labels[i].item()),
                    "predicted_label": pred,
                    "probability": prob,
                    "group": str(groups[i]),
                    "cohort": str(cohorts[i]),
                })
            done += len(probs)
            if done % 20 == 0 or done == total:
                print(f"Inferenced {done}/{total}", flush=True)

    pred_df = pd.DataFrame(all_rows)
    pred_df.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)

    subset_rows = [summarize_subset(pred_df, "all")]
    for cohort_name in sorted(pred_df["cohort"].unique()):
        if not cohort_name:
            continue
        sub = pred_df[pred_df["cohort"] == cohort_name]
        if len(sub) > 0:
            subset_rows.append(summarize_subset(sub, f"cohort_{cohort_name}"))
    metrics_df = pd.DataFrame(subset_rows)
    metrics_df.to_csv(os.path.join(args.output_dir, "metrics_by_subset.csv"), index=False)

    print("=" * 80)
    print("Inference completed.")
    print(f"Saved: {os.path.join(args.output_dir, 'predictions.csv')}")
    print(f"Saved: {os.path.join(args.output_dir, 'metrics_by_subset.csv')}")


if __name__ == "__main__":
    main()
