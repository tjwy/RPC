"""MRI + 11-clinical dataset used by the canonical PRE external inference.

Same preprocessing pipeline as the April-10 SeqVITRadCli script that produced
the adopted PRE manuscript probability vector:
  - 3-phase PNG (AP / PVP / DP) -> grayscale -> CLAHE -> minmax -> 224x224
    -> 3-channel replicate -> ImageNet normalize -> concat 9 channels
  - 7 numerical clinical features standardized with the saved scaler
  - 2 categorical clinical features (BCLC, AFP400) one-hot to 4 dims
  - Final clinical vector: 7 + 4 = 11 dims
"""

from __future__ import annotations

import os
import pickle

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MRIClinicalDataset(Dataset):
    def __init__(self, clinical_data: pd.DataFrame, mri_base_dirs: dict, scaler_path: str):
        self.clinical_data = clinical_data.reset_index(drop=True)
        self.mri_base_dirs = mri_base_dirs

        self.standardization_features = [
            "age", "NLR", "PLT", "tumor_dm",
            "Gamma-Glutamyltransferase", "Alkaline_Phosphatase", "Cirrhosis",
        ]
        self.extraction_categorical = ["BCLC", "AFP400"]

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        self.missing_count = 0
        self.total_count = 0

    def __len__(self) -> int:
        return len(self.clinical_data)

    def __getitem__(self, idx: int):
        self.total_count += 1
        row = self.clinical_data.iloc[idx]
        case_id = str(row["case_id"])
        label = row["label"]
        group = str(row["group"])

        if pd.isna(row.get("MR_ID", np.nan)):
            mr_id = ""
        else:
            try:
                mr_id = str(int(float(row["MR_ID"])))
            except (ValueError, TypeError):
                mr_id = str(row["MR_ID"])

        slide_id = (
            str(row["slide_id"])
            if "slide_id" in row.index and pd.notna(row["slide_id"])
            else case_id
        )

        # ---- MRI ------------------------------------------------------------
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
                else:
                    self.missing_count += 1
                    mri = torch.zeros(9, 224, 224)
            else:
                self.missing_count += 1
                mri = torch.zeros(9, 224, 224)
        else:
            if group == "TCGA":
                mri_dir = self.mri_base_dirs["TCGA"]
                target_id = case_id
            else:
                mri_dir = self.mri_base_dirs["1"]
                target_id = mr_id

            phases = ["AP", "PVP", "DP"]
            images = []
            missing_flag = False

            for phase in phases:
                img_path = os.path.join(mri_dir, f"{target_id}_{phase}.png")
                if os.path.exists(img_path):
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        img = clahe.apply(img)
                        if np.max(img) - np.min(img) != 0:
                            img = (img - np.min(img)) / (np.max(img) - np.min(img))
                        else:
                            img = np.zeros_like(img, dtype=np.float32)
                        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
                        if np.max(img) - np.min(img) != 0:
                            img = (img - np.min(img)) / (np.max(img) - np.min(img))
                        else:
                            img = np.zeros_like(img, dtype=np.float32)
                        img = np.expand_dims(img, axis=-1)
                        img = np.concatenate([img, img, img], axis=-1)
                        images.append(img)
                    else:
                        images.append(np.zeros((224, 224, 3)))
                else:
                    missing_flag = True
                    images.append(np.zeros((224, 224, 3)))

            if missing_flag:
                self.missing_count += 1

            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            tensors = []
            for img in images:
                img_pil = Image.fromarray((img * 255).astype(np.uint8))
                tensors.append(transform(img_pil))
            mri = torch.cat(tensors, dim=0)

        # ---- clinical -------------------------------------------------------
        numerical_features = []
        for feat in self.standardization_features:
            val = row[feat]
            if pd.isna(val):
                val = 0.0
            numerical_features.append(val)

        numerical_df = pd.DataFrame([numerical_features], columns=self.standardization_features)
        numerical_features = self.scaler.transform(numerical_df)[0]

        categorical_features = []
        for feat in self.extraction_categorical:
            val = row[feat]
            if pd.isna(val):
                val = 0
            if feat == "BCLC":
                if str(val) in ("0", "0.0"):
                    val = 0
                elif str(val) == "A":
                    val = 1
                else:
                    val = 1
            else:
                try:
                    val = int(float(val))
                except Exception:
                    val = 0
            one_hot = [0, 0]
            if val < 2:
                one_hot[val] = 1
            else:
                one_hot[1] = 1
            categorical_features.extend(one_hot)

        clinical_vector = np.concatenate([numerical_features, categorical_features])
        return mri, torch.FloatTensor(clinical_vector), label, case_id
