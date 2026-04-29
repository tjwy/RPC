# PRE_DLR — Canonical PRE / DLR pipelines (dl-mri framework)

This folder contains the **dl-mri-based** scripts that produced the manuscript
PRE and DLR external probability vectors. These are the *adopted* manuscript
versions, separate from the SurvPath-aligned candidate adapters under
`RPC_CLI/models/model_pre_dlr_adapter.py`.

## Files

| File | Purpose | Provides |
|------|---------|----------|
| `train_dl_mri_full.py` | End-to-end full-training entry for PRE / DLR | `SeqVITRadCli` (PRE), `SeqVITRad` (DLR-equivalent classifier), `DlMriDataset` |
| `mri_clinical_dataset.py` | Inference-time dataset for PRE (3-phase MRI + 11 clinical) | `MRIClinicalDataset` |
| `test_DLR_external_split.py` | Canonical DLR external inference (MRI-only) | `SeqVIT`, `MRIDataset`, CLI |
| `predict_PRE_external.py` | Canonical PRE external inference (MRI + clinical) | CLI |

## Manuscript-adopted models

| Model | Architecture | Inputs | Sigmoid policy at inference |
|-------|--------------|--------|-----------------------------|
| **PRE** | `SeqVITRadCli` | 3-phase MRI (9 ch) + 11 clinical features | model returns sigmoid; an extra `torch.sigmoid()` is applied (intentional, for exact reproduction of April-10 result) |
| **DLR** | `SeqVIT` | 3-phase MRI only (9 ch) | model returns sigmoid; no extra sigmoid |

## Training (example)

```bash
# PRE — MRI + 11 clinical
python train_dl_mri_full.py \
    --model pre \
    --labels_csv /path/to/labels.csv \
    --data_dir /path/to/processed_npy_root \
    --raw_cli_path /path/to/clinical.xlsx \
    --epochs 10 --lr 0.001 --weight_decay 0.01 \
    --output_dir ./runs/pre_full

# DLR — MRI only
python train_dl_mri_full.py \
    --model dlr \
    --labels_csv /path/to/labels.csv \
    --data_dir /path/to/processed_npy_root \
    --epochs 14 --lr 0.0001 --weight_decay 0.001 \
    --output_dir ./runs/dlr_full
```

## External inference (example)

```bash
# PRE
python predict_PRE_external.py \
    --checkpoint_path /path/to/checkpoint_full_pre_*.pt \
    --external_csv /path/to/external_clinical.csv \
    --scaler_path /path/to/scaler_7features.pkl \
    --output_dir ./out_pre \
    --group1_mri_dir /path/to/group1_pngs \
    --tcga_mri_dir /path/to/tcga_pngs \
    --test_npy_dir /path/to/test_npy

# DLR
python test_DLR_external_split.py \
    --checkpoint_path /path/to/checkpoint_full_dlr_*.pt \
    --external_csv /path/to/external_clinical.csv \
    --output_dir ./out_dlr \
    --group1_mri_dir /path/to/group1_pngs \
    --tcga_mri_dir /path/to/tcga_pngs \
    --test_npy_dir /path/to/test_npy
```

## Privacy & data notes

- No real labels, splits, MRI volumes, or clinical CSVs are shipped.
- The training pipeline's `load_clinical_features` expects the user's own
  Excel file with the documented columns (see `train_dl_mri_full.py:load_clinical_features`).
- The `--scaler_path` for PRE inference must be a `pickle`d sklearn
  `StandardScaler` fit on the same 7 numerical features at training time.
- Pretrained backbones come from `timm` (`resnet50`); no proprietary
  weights are required.
