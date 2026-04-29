#!/usr/bin/env python3
"""Canonical PRE external inference (SeqVITRadCli + MRI + 11 clinical).

This is the inference path that produced the adopted PRE manuscript
probability vector (verified 295/295 exact match against the archived
April-10 result). It is based on the dl-mri framework:
  - Model:    SeqVITRadCli (defined in train_dl_mri_full.py)
  - Dataset:  MRIClinicalDataset (defined in mri_clinical_dataset.py)
  - Output:   sigmoid(model(mri, cli)). The model already applies sigmoid
              internally; the additional sigmoid here intentionally
              reproduces the original April-10 inference behavior.

Sanitized for public release: no private absolute paths, all I/O is CLI-driven.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Allow running as a script from PRE_DLR/.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from train_dl_mri_full import SeqVITRadCli  # noqa: E402
from mri_clinical_dataset import MRIClinicalDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PRE external inference (SeqVITRadCli + MRI + 11 clinical)"
    )
    p.add_argument("--checkpoint_path", required=True,
                   help="Path to a trained SeqVITRadCli checkpoint (.pt state_dict)")
    p.add_argument("--external_csv", required=True,
                   help="External clinical CSV with required columns: "
                        "case_id, group, label, age, NLR, PLT, tumor_dm, "
                        "Gamma-Glutamyltransferase, Alkaline_Phosphatase, "
                        "Cirrhosis, BCLC, AFP400 (and optionally slide_id, MR_ID, cohort)")
    p.add_argument("--scaler_path", required=True,
                   help="Pickled sklearn StandardScaler fit on the 7 numerical clinical features")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--label_col", default="label")
    p.add_argument("--group1_mri_dir", required=True,
                   help="Directory containing {MR_ID}_{AP|PVP|DP}.png for group=='1'")
    p.add_argument("--tcga_mri_dir", default="",
                   help="Directory containing {case_id}_{AP|PVP|DP}.png for group=='TCGA'")
    p.add_argument("--test_npy_dir", default="",
                   help="Directory containing {slide_id}_rad_stacked.npy for group=='test'")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("PRE external inference (SeqVITRadCli + 11-clinical)")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"External CSV: {args.external_csv}")
    print(f"Scaler: {args.scaler_path}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 80)

    if not os.path.isfile(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    if not os.path.isfile(args.scaler_path):
        raise FileNotFoundError(f"Scaler not found: {args.scaler_path}")

    df = pd.read_csv(args.external_csv)
    df["case_id"] = df["case_id"].astype(str).str.strip()
    if "group" not in df.columns:
        df["group"] = "test"
    if "slide_id" not in df.columns:
        df["slide_id"] = df["case_id"]
    if "MR_ID" not in df.columns:
        df["MR_ID"] = df["case_id"]

    print(f"Loaded {len(df)} cases (label pos={int(df[args.label_col].sum())})")

    mri_base_dirs = {
        "1": args.group1_mri_dir,
        "TCGA": args.tcga_mri_dir,
        "test_npy": args.test_npy_dir,
    }

    dataset = MRIClinicalDataset(df, mri_base_dirs, args.scaler_path)

    model = SeqVITRadCli(sequence_length=3, clinical_feature_dim=11)
    sd = torch.load(args.checkpoint_path, map_location="cpu")
    load_result = model.load_state_dict(sd, strict=True)
    assert (
        not load_result.missing_keys and not load_result.unexpected_keys
    ), f"Unexpected checkpoint mismatch: {load_result}"
    print(f"Checkpoint loaded.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_case_ids: List[str] = []
    all_probs: List[float] = []
    all_labels: List[float] = []
    all_groups: List[str] = []
    all_cohorts: List[str] = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            mri_t, cli_t, label, case_id = dataset[idx]
            mri_t = mri_t.unsqueeze(0).to(device)
            cli_t = cli_t.unsqueeze(0).float().to(device)

            output = model(mri_t, cli_t)
            prob = torch.sigmoid(output).item()

            row = df[df["case_id"] == str(case_id)].iloc[0]
            cohort_val = str(row["cohort"]) if "cohort" in df.columns else ""

            all_case_ids.append(str(case_id))
            all_probs.append(prob)
            all_labels.append(float(label))
            all_groups.append(str(row.get("group", "test")))
            all_cohorts.append(cohort_val)

            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(dataset)}")

    print(f"Inference done: {len(all_probs)} cases (MRI missing: {dataset.missing_count})")

    probs_arr = np.array(all_probs)
    preds_arr = (probs_arr >= args.threshold).astype(int)
    labels_arr = np.array(all_labels)

    out_df = pd.DataFrame({
        "case_id": all_case_ids,
        "true_label": labels_arr.astype(int),
        "predicted_label": preds_arr,
        "probability": probs_arr,
        "group": all_groups,
        "cohort": all_cohorts,
    })

    predictions_csv = os.path.join(args.output_dir, "predictions.csv")
    out_df.to_csv(predictions_csv, index=False)
    print(f"Saved: {predictions_csv}")

    if len(np.unique(labels_arr)) >= 2:
        auc_all = roc_auc_score(labels_arr, probs_arr)
        print(f"AUC (all {len(labels_arr)}): {auc_all:.4f}")
        for cohort_id in sorted(set(all_cohorts)):
            if not cohort_id:
                continue
            mask = np.array(all_cohorts) == cohort_id
            if mask.sum() > 0 and len(np.unique(labels_arr[mask])) >= 2:
                auc_c = roc_auc_score(labels_arr[mask], probs_arr[mask])
                print(f"AUC (cohort={cohort_id}): {auc_c:.4f}  (n={mask.sum()})")

    meta = {
        "script": __file__,
        "run_at": datetime.now().isoformat(),
        "checkpoint": args.checkpoint_path,
        "external_csv": args.external_csv,
        "scaler": args.scaler_path,
        "model_class": "SeqVITRadCli",
        "threshold": args.threshold,
        "n_cases": len(out_df),
    }
    with open(os.path.join(args.output_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("=" * 80)


if __name__ == "__main__":
    main()
