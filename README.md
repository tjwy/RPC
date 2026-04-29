# RPC

Source code for Recurrence Prediction and Classification (RPC) models.

This public-release staging package contains the training and inference
entry points for the seven manuscript model families. Interpretability
scripts are intentionally not included in this first release and can be
added later as an optional module.

## Repository Layout

```text
RPC_CLI/   # Unified SurvPath-derived training / inference pipeline
           #   adopted manuscript provenance: POST, DLP, DLRP, PRE-CLI, POST-CLI
PRE_DLR/   # dl-mri-framework canonical scripts for PRE & DLR
           #   adopted manuscript provenance: PRE (SeqVITRadCli), DLR (SeqVIT)
```

> **Manuscript provenance summary.** The PRE and DLR manuscript probability
> vectors were produced by the dl-mri framework (`PRE_DLR/`), not by the
> SurvPath PRE / DLR adapter classes inside `RPC_CLI/`. The latter are kept
> as candidate variants for users who want to retrain PRE / DLR within the
> unified SurvPath pipeline. See `PRE_DLR/README.md` for details.

## Included Model Families

| Model | Adopted source | Modality flag |
|-------|----------------|---------------|
| POST | `RPC_CLI/` (SurvPath, MRI + WSI + clinical) | `--modality survpath` |
| PRE | **`PRE_DLR/predict_PRE_external.py`** (`SeqVITRadCli`, MRI + 11 clinical) | n/a (dl-mri) |
| DLP | `RPC_CLI/` (SurvPath, WSI only) | `--modality dlp` |
| DLR | **`PRE_DLR/test_DLR_external_split.py`** (`SeqVIT`, MRI only) | n/a (dl-mri) |
| DLRP | `RPC_CLI/` (SurvPath, MRI + WSI) | `--modality dlrp` |
| PRE-CLI | `RPC_CLI/` (clinical-only, preoperative) | `--modality pre_cli` |
| POST-CLI | `RPC_CLI/` (clinical-only, postoperative) | `--modality post_cli` |

## Data Policy

No raw clinical data, real split files, MRI scans, WSI images, WSI feature
bags, checkpoints, or private pretrained MRI encoder weights are included.
The CSV files in `RPC_CLI/datasets_csv/metadata/` and `RPC_CLI/splits/` are
synthetic examples only.

MRI encoder weights are exposed only as runtime parameters:

```bash
--mri_encoder_weights /path/to/private_encoder_weights.pt
--mri_encoder_pretrained
--freeze_mri_encoder
```

Do not commit private weights or patient-level data to GitHub.

## Minimal Inputs (RPC_CLI)

Training expects de-identified local paths supplied at runtime:

- `--label_file`: CSV with `case_id`, `slide_id`, and binary label column.
- `--clinical_file`: CSV with `case_id`, `group`, and the clinical variables used by the model.
- `--data_root_dir`: directory containing WSI feature HDF5 files named `<slide_id>_features.h5`.
- `--rad_data_dir`: directory containing preprocessed MRI data, either stacked `.npy` files or phase PNG files.
- `--which_splits` and `--study`: together resolve split files under `RPC_CLI/splits/<which_splits>/<study>/splits_<fold>.csv`.

The synthetic example layout is:

```text
RPC_CLI/splits/example_5fold/rpc_example/splits_0.csv ... splits_4.csv
RPC_CLI/datasets_csv/metadata/example_labels.csv
RPC_CLI/datasets_csv/metadata/example_clinical.csv
```

## Example Training (RPC_CLI)

```bash
cd RPC_CLI
bash scripts/run_train_7models_examples.sh
```

For real de-identified data, override paths with environment variables:

```bash
LABEL_FILE=/path/to/labels.csv \
CLINICAL_FILE=/path/to/clinical.csv \
WSI_FEATURE_DIR=/path/to/wsi_features \
MRI_DATA_DIR=/path/to/mri_data \
MRI_ENCODER_WEIGHTS=/path/to/private_encoder.pt \
FREEZE_MRI_ENCODER=1 \
bash scripts/run_train_7models_examples.sh
```

## Example Inference

```bash
# RPC_CLI (POST / DLP / DLRP / PRE-CLI / POST-CLI)
CHECKPOINT_PATH=/path/to/s_0_checkpoint.pt \
MODALITY=survpath \
bash RPC_CLI/scripts/run_inference_example.sh

# PRE  (canonical dl-mri pipeline)
python PRE_DLR/predict_PRE_external.py \
    --checkpoint_path /path/to/checkpoint_full_pre_*.pt \
    --external_csv /path/to/external_clinical.csv \
    --scaler_path /path/to/scaler_7features.pkl \
    --group1_mri_dir /path/to/group1_pngs \
    --output_dir ./out_pre

# DLR  (canonical dl-mri pipeline)
python PRE_DLR/test_DLR_external_split.py \
    --checkpoint_path /path/to/checkpoint_full_dlr_*.pt \
    --external_csv /path/to/external_clinical.csv \
    --group1_mri_dir /path/to/group1_pngs \
    --output_dir ./out_dlr
```

See `PRE_DLR/README.md` for the complete dl-mri training and inference recipe.

## Notes

- This release keeps the original training style close to the internal SurvPath-derived workflow, but removes private absolute paths.
- The previous public RPC repository used separate top-level modules for DLP, PRE/DLR, and RPC_CLI. This release preserves that split: PRE / DLR are canonical under `PRE_DLR/`, while POST / DLP / DLRP / clinical-only models are unified under `RPC_CLI/`.
- The example CSVs are schema templates, not real data.
- Clinical feature dimension is controlled by `--pre_adapter_clinical_dim` or the `CLINICAL_DIM` environment variable in the example scripts.

## Acknowledgements

The `RPC_CLI/` pipeline (POST / DLP / DLRP / PRE-CLI / POST-CLI) is adapted
from the SurvPath framework released by the Mahmood Lab:

- SurvPath: https://github.com/mahmoodlab/SurvPath

We thank the SurvPath authors for open-sourcing their code. The
multimodal dataset class, training loop, and transformer-based fusion
modules in `RPC_CLI/` are derived from their implementation, with
modifications for binary recurrence classification, MRI integration, and
clinical-only variants. Please cite the original SurvPath paper if you
build on this part of the codebase.

