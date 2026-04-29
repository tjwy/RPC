# Release Manifest

## Public-Release Source Files

### `RPC_CLI/` — unified SurvPath-derived training pipeline

- `RPC_CLI/main_train.py`: five-fold training entry point.
- `RPC_CLI/main-latest.py`: compatibility copy matching the existing RPC_CLI entry-point naming style.
- `RPC_CLI/main_inference.py`: checkpoint-based inference entry point.
- `RPC_CLI/models/`: model definitions for POST, DLP, DLRP, PRE-CLI, POST-CLI, and SurvPath-aligned PRE / DLR candidate adapters.
- `RPC_CLI/models/layers/cross_attention_rad.py`: multimodal attention layer used by POST/DLRP.
- `RPC_CLI/datasets/dataset_survival_latest.py`: dataset factory and dataloader logic with public path arguments.
- `RPC_CLI/utils/`: training, metrics, argument parsing, file utilities, and loss functions.
- `RPC_CLI/custom_optims/`: RAdam and LAMB optimizers retained from the internal workflow.
- `RPC_CLI/scripts/run_train_7models_examples.sh`: example shell workflow for training the seven model families.
- `RPC_CLI/scripts/run_inference_example.sh`: example shell workflow for checkpoint inference.
- `RPC_CLI/splits/example_5fold/rpc_example/`: synthetic example splits only.
- `RPC_CLI/datasets_csv/metadata/`: synthetic example label and clinical templates only.

### `PRE_DLR/` — canonical PRE / DLR pipelines (dl-mri framework)

- `PRE_DLR/train_dl_mri_full.py`: full-training entry point producing the manuscript-adopted PRE (`SeqVITRadCli`) and DLR (`SeqVITRad`) checkpoints.
- `PRE_DLR/predict_PRE_external.py`: canonical PRE external inference (MRI + 11 clinical).
- `PRE_DLR/test_DLR_external_split.py`: canonical DLR external inference (MRI-only `SeqVIT`).
- `PRE_DLR/mri_clinical_dataset.py`: PRE-side inference dataset (3-phase PNG + 11 clinical).
- `PRE_DLR/README.md`: usage and provenance details.

> The `SurvPathPREAdapter` and `SurvPathDLRAdapter` classes in `RPC_CLI/models/model_pre_dlr_adapter.py` are SurvPath-aligned candidates only. The manuscript-adopted PRE and DLR probability vectors come from `PRE_DLR/`.

## Deliberately Excluded

- Real clinical CSV files.
- Real split CSV files.
- Patient identifiers beyond synthetic examples.
- Raw WSI/MRI data.
- WSI feature HDF5 bags.
- Training outputs, logs, and result tables.
- Checkpoints and model weights.
- Private MRI encoder pretrained weights and pickled clinical scalers.
- Interpretability scripts and heatmap generation workflows for the first release.
- Legacy baseline model definitions that are not required for the seven manuscript models.

## Release-Specific Cleaning Performed

- Replaced private absolute data paths with runtime arguments.
- Replaced hard-coded MRI encoder checkpoint loading with `--mri_encoder_weights`.
- Made MRI/ImageNet pretrained backbone usage opt-in via `--mri_encoder_pretrained`.
- Added PRE-CLI and POST-CLI clinical-only modalities to the staged training flow.
- Added synthetic split and metadata templates instead of real CSV files.
- Added a separate top-level `PRE_DLR/` module containing the dl-mri framework scripts that produced the adopted PRE and DLR manuscript results, and removed the embedded private `external_labels_final.csv` override block from those scripts.
