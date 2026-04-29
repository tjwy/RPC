#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LABEL_FILE="${LABEL_FILE:-${ROOT_DIR}/datasets_csv/metadata/example_labels.csv}"
CLINICAL_FILE="${CLINICAL_FILE:-${ROOT_DIR}/datasets_csv/metadata/example_clinical.csv}"
WSI_FEATURE_DIR="${WSI_FEATURE_DIR:-/path/to/wsi_feature_h5}"
MRI_DATA_DIR="${MRI_DATA_DIR:-/path/to/mri_png_or_npy}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to a trained .pt checkpoint}"
MODALITY="${MODALITY:-survpath}"
STUDY="${STUDY:-rpc_example}"
SPLITS="${SPLITS:-example_5fold}"
GPU_ID="${GPU_ID:-0}"
CLINICAL_DIM="${CLINICAL_DIM:-19}"

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

CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${ROOT_DIR}/main_inference.py" \
  --study "${STUDY}" \
  --task survival \
  --which_splits "${SPLITS}" \
  --label_file "${LABEL_FILE}" \
  --clinical_file "${CLINICAL_FILE}" \
  --data_root_dir "${WSI_FEATURE_DIR}" \
  --rad_data_dir "${MRI_DATA_DIR}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --label_col label \
  --k 5 \
  --batch_size 1 \
  --bag_loss bce_logits \
  --n_classes 2 \
  --encoding_dim 1024 \
  --num_patches 4096 \
  --wsi_projection_dim 256 \
  --pre_adapter_clinical_dim "${CLINICAL_DIM}" \
  --modality "${MODALITY}" \
  --results_dir "inference_${MODALITY}" \
  "${MRI_ARGS[@]}"
