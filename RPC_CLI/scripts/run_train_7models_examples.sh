#!/usr/bin/env bash
set -euo pipefail

# Example only. Replace these paths with de-identified local files before running.
ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LABEL_FILE="${LABEL_FILE:-${ROOT_DIR}/datasets_csv/metadata/example_labels.csv}"
CLINICAL_FILE="${CLINICAL_FILE:-${ROOT_DIR}/datasets_csv/metadata/example_clinical.csv}"
WSI_FEATURE_DIR="${WSI_FEATURE_DIR:-/path/to/wsi_feature_h5}"
MRI_DATA_DIR="${MRI_DATA_DIR:-/path/to/mri_png_or_npy}"
RESULTS_PREFIX="${RESULTS_PREFIX:-rpc_release_example}"
STUDY="${STUDY:-rpc_example}"
SPLITS="${SPLITS:-example_5fold}"
GPU_ID="${GPU_ID:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
CLINICAL_DIM="${CLINICAL_DIM:-19}"

COMMON_ARGS=(
  --study "${STUDY}"
  --task survival
  --which_splits "${SPLITS}"
  --label_file "${LABEL_FILE}"
  --clinical_file "${CLINICAL_FILE}"
  --data_root_dir "${WSI_FEATURE_DIR}"
  --rad_data_dir "${MRI_DATA_DIR}"
  --label_col label
  --k 5
  --max_epochs "${MAX_EPOCHS}"
  --batch_size 1
  --lr 0.001
  --reg 0.0001
  --opt adamW
  --bag_loss bce_logits
  --n_classes 2
  --encoding_dim 1024
  --num_patches 4096
  --wsi_projection_dim 256
  --pre_adapter_clinical_dim "${CLINICAL_DIM}"
)

MRI_ARGS=()
if [[ -n "${MRI_ENCODER_WEIGHTS:-}" ]]; then
  MRI_ARGS+=(--mri_encoder_weights "${MRI_ENCODER_WEIGHTS}")
fi
if [[ "${MRI_ENCODER_PRETRAINED:-0}" == "1" ]]; then
  MRI_ARGS+=(--mri_encoder_pretrained)
fi
if [[ "${FREEZE_MRI_ENCODER:-0}" == "1" ]]; then
  MRI_ARGS+=(--freeze_mri_encoder)
fi

run_model() {
  local name="$1"
  local modality="$2"
  shift 2
  echo "Running ${name} (${modality})"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${ROOT_DIR}/main_train.py" \
    "${COMMON_ARGS[@]}" \
    --modality "${modality}" \
    --results_dir "${RESULTS_PREFIX}_${name}" \
    "$@"
}

# Seven model families used in the manuscript.
run_model "POST" "survpath" "${MRI_ARGS[@]}"
run_model "PRE" "pre" "${MRI_ARGS[@]}"
run_model "DLP" "dlp"
run_model "DLR" "dlr" "${MRI_ARGS[@]}"
run_model "DLRP" "dlrp"
run_model "PRE_CLI" "pre_cli"
run_model "POST_CLI" "post_cli"
