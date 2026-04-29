from __future__ import print_function, division
from cProfile import label
import os
import pdb
from unittest import case
import pandas as pd
import pickle
import networkx as nx
import numpy as np
import pandas as pd
import copy
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2
from utils.general_utils_fufa import _series_intersection
import time
from sklearn.preprocessing import StandardScaler
# ALL_MODALITIES = ['rna_clean.csv']  
from torchvision import transforms
from PIL import Image

data_transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(size=224, scale=(0.9, 1.1)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _normalize_case_id(value):
    text = str(value).strip()
    if text.endswith('.0'):
        text = text[:-2]
    return text

def gen_dataset_multi_images(data_dir, case_id, img_size=224, outline=10):
    mask_dir = data_dir + '/mask'
    ori_dir = data_dir + '/ori'

    image_dict = {}  # case_id

    for cid in case_id:
        stacked_image = None  # ID
        for suffix in ['Ap', 'PVP', 'DP']:
            X_file_name = f"{cid}_{suffix}.npy"
            m_file_name = f"{cid}_{suffix}.npy"

            if not (os.path.exists(os.path.join(ori_dir, X_file_name)) and
                    os.path.exists(os.path.join(mask_dir, m_file_name))):
                continue

            X = np.load(os.path.join(ori_dir, X_file_name))
            m = np.load(os.path.join(mask_dir, m_file_name))

            X = np.uint8(cv2.normalize(X, None, 0, 255, cv2.NORM_MINMAX))
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

            if len(X.shape) == 3 and X.shape[2] == 3:
                X = X[:,:,0]
            X = clahe.apply(X)
            X = (X - np.min(X)) / (np.max(X) - np.min(X))

            x, y = np.where(m > 0)

            w0, h0 = m.shape
            x_min = max(0, int(np.min(x) - outline))
            x_max = min(w0, int(np.max(x) + outline))
            y_min = max(0, int(np.min(y) - outline))
            y_max = min(h0, int(np.max(y) + outline))

            m = m[x_min:x_max, y_min:y_max]
            X = X[x_min:x_max, y_min:y_max]

            X_m_1 = X.copy()
            if X_m_1.shape[0] != img_size or X_m_1.shape[1] != img_size:
                X_m_1 = cv2.resize(X_m_1, (img_size, img_size), interpolation=cv2.INTER_CUBIC)

            X_m_1 = (X_m_1 - np.min(X_m_1)) / (np.max(X_m_1) - np.min(X_m_1))
            X_m_1 = np.expand_dims(X_m_1, axis=-1)
            X_m_1 = np.concatenate([X_m_1, X_m_1, X_m_1], axis=-1)
            
            # # numpyPIL
            # X_m_1_pil = Image.fromarray((X_m_1 * 255).astype(np.uint8))
            # # data_transform_test
            # X_m_1 = data_transform_test(X_m_1_pil)

            if stacked_image is None:
                stacked_image = X_m_1
            else:
                stacked_image = np.concatenate([stacked_image, X_m_1], axis=-1)

        # ID
        image_dict[cid] = stacked_image
        
    return image_dict
# ### transform normalize
    
class SurvivalDatasetFactory:

    def __init__(self,
        study,
        label_file, 
        clinical_file,
        rad_data_dir,  # omics_dirrad_dir
        external_pathology_dir=None,
        tcga_pathology_dir=None,
        external_mri_dir=None,
        tcga_mri_dir=None,
        seed=1, 
        print_info=True, 
        label_col="label", 
        eps=1e-6,
        num_patches=4096, # ##  patch
        need_rad_data=True,
        ):
        r"""
        Initialize the factory to store metadata, survival label, and slide_ids for each case id. 

        Args:
            - study : String 
            - label_file : String 
            - omics_dir : String
            - seed : Int
            - print_info : Boolean
            - n_bins : Int
            - label_col: String
            - eps Float
            - num_patches : Int 
            - is_mcat : Boolean
            - is_survapth : Boolean 
            - type_of_pathway : String

        Returns:
            - None
        """

        #---> self
        self.study = study
        self.label_file = label_file
        self.clinical_file = clinical_file
        # self.omics_dir = omics_dir
        
        self.seed = seed
        self.print_info = print_info
        self.train_ids, self.val_ids  = (None, None)
        self.data_dir = None
        self.label_col = label_col
        # self.n_bins = n_bins
        self.num_patches = num_patches
        self.rad_data_dir = rad_data_dir
        self.need_rad_data = bool(need_rad_data)
        
        self.group1_pathology_dir = external_pathology_dir
        self.tcga_pathology_dir = tcga_pathology_dir
        self.group1_mri_dir = external_mri_dir
        self.tcga_mri_dir = tcga_mri_dir

        #---> labels, metadata, patient_df
        self._setup_metadata_and_labels(eps)

        #---> prepare for weighted sampling
        self._cls_ids_prep()

        #---> load all clinical data 
        self._load_clinical_data()

        #---> summarize
        self._summarize()

        if self.need_rad_data:
            start_time = time.time()
            self._setup_rad_data()
            end_time = time.time()
            total_time = end_time - start_time
            print(f"Total time for _setup_rad_data: {total_time} seconds")
        else:
            self.rad_img_dict = {}
            if self.print_info:
                print("Skipping _setup_rad_data for non-MRI modality run.")

    def _load_clinical_data(self):
        r"""
        Load the clinical data for the patient which has grade, stage, etc.
        
        Args:
            - self 
        
        Returns:
            - None
            
        """
        if self.clinical_file is None:
            raise ValueError("--clinical_file is required for public-release runs.")
        path_to_data = self.clinical_file
        self.clinical_data = pd.read_csv(path_to_data)
        self.clinical_data.index = self.clinical_data["case_id"].astype(str)
            # Z-score

    def _setup_metadata_and_labels(self, eps):
        r"""
        Process the metadata required to run the experiment. Clean the data. Set up patient dicts to store slide ids per patient.
        Get label dict.
        
        Args:
            - self
            - eps : Float 
        
        Returns:
            - None 
        
        """

        #---> read labels 
        self.label_data = pd.read_csv(self.label_file, low_memory=False)

        # uncensored_df = self.label_data[self.label_data[self.censorship_var] < 1]

        # case_id
        self.patients_df = self.label_data.drop_duplicates(['case_id']).copy()

        #---> get patient info, labels, and metadata
        self._get_patient_dict()
        self._get_label_dict()
        self._get_patient_data()
        
    def _get_patient_data(self):
        r"""
        Final patient data is just the clinical metadata + label for the patient 
        
        Args:
            - self 
        
        Returns: 
            - None
        
        """
        patients_df = self.label_data[~self.label_data.index.duplicated(keep='first')] 
        patient_data = {'case_id': patients_df["case_id"].values, 'label': patients_df['label'].values} # only setting the final data to self 'label': patients_df['label'].values 
        self.patient_data = patient_data


    def _get_label_dict(self):
        r"""
        "class_label"

        Args:
            - self
        
        Returns:
            - self
        """

        # 01
        label_dict = {0: 0, 1: 1}
        self.num_classes = len(label_dict)
        # print(f'Number of classes: {self.num_classes}')
        self.label_dict = label_dict

        # label_data'label'
        self.label_data['label'] = self.label_data[self.label_col].astype(int)

    def _get_patient_dict(self):
        r"""
        For every patient store the respective slide ids

        Args:
            - self 
        
        Returns:
            - None
        """
    
        patient_dict = {}
        temp_label_data = self.label_data.set_index('case_id')
        for patient in self.patients_df['case_id']:
            slide_ids = temp_label_data.loc[patient, 'slide_id']
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            else:
                slide_ids = slide_ids.values
            patient_dict.update({str(patient):slide_ids})
        self.patient_dict = patient_dict    
        self.label_data = self.patients_df
        self.label_data.reset_index(drop=True, inplace=True)

    def _cls_ids_prep(self):
        r"""
        Find which patient/slide belongs to which label and store the label-wise indices of patients/ slides

        Args:
            - self 
        
        Returns:
            - None

        """
        self.patient_cls_ids = [[] for i in range(self.num_classes)]   
        # Find the index of patients for different labels
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0] 

        # Find the index of slides for different labels
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.label_data['label'] == i)[0]

    def _summarize(self):
        r"""
        Summarize which type of survival you are using, number of cases and classes
        
        Args:
            - self 
        
        Returns:
            - None 
        
        """
        if self.print_info:
            print("label column: {}".format(self.label_col))
            print("number of cases {}".format(len(self.label_data)))
            print("number of classes: {}".format(self.num_classes))

    def _load_stacked_npy(self, npy_path):
        if not os.path.isfile(npy_path):
            return None
        try:
            arr = np.load(npy_path)
        except Exception:
            return None

        if arr.ndim != 3:
            return None

        if arr.shape == (9, 224, 224):
            arr = np.transpose(arr, (1, 2, 0))
        elif arr.shape[2] != 9:
            return None

        if arr.shape[0] != 224 or arr.shape[1] != 224:
            try:
                p1 = cv2.resize(arr[:, :, 0:3], (224, 224), interpolation=cv2.INTER_CUBIC)
                p2 = cv2.resize(arr[:, :, 3:6], (224, 224), interpolation=cv2.INTER_CUBIC)
                p3 = cv2.resize(arr[:, :, 6:9], (224, 224), interpolation=cv2.INTER_CUBIC)
                arr = np.concatenate([p1, p2, p3], axis=2)
            except Exception:
                return None

        return torch.from_numpy(arr.astype(np.float32))

    def _load_png_triplet(self, p1, p2, p3):
        img1 = cv2.imread(p1)
        img2 = cv2.imread(p2)
        img3 = cv2.imread(p3)

        if img1 is None or img2 is None or img3 is None:
            return None

        return torch.cat([torch.tensor(img1), torch.tensor(img2), torch.tensor(img3)], dim=2)

    def _setup_rad_data(self):
        # rad_features_dict = {}
        label_case_ids = set(self.label_data['case_id'].astype(str).tolist())
        case_ids = [cid for cid in self.clinical_data['case_id'].astype(str).tolist() if cid in label_case_ids]
        img_dict = {}

        # auto: prefer preprocessed npy -> fallback png
        # npy: use npy only
        # png: use png only
        rad_load_mode = os.environ.get("SURVPATH_RAD_LOAD_MODE", "auto").strip().lower()
        stacked_subdir = os.environ.get("SURVPATH_RAD_STACKED_SUBDIR", "preprocessed").strip()

        npy_hits = 0
        png_hits = 0
        missing_hits = 0

        # img_dict = gen_dataset_multi_images(self.rad_data_dir, case_ids,img_size=224, outline=10)

        for case_id in case_ids:
            case_id_norm = _normalize_case_id(case_id)
            # Determine group and directory
            try:
                group = self.clinical_data.loc[self.clinical_data['case_id'] == case_id, 'group'].values[0]
            except Exception:
                group = 'train'

            if str(group) == '1':
                rad_dir = self.group1_mri_dir or self.rad_data_dir
                suffix_map = {'Ap': 'AP', 'PVP': 'PVP', 'DP': 'DP'}
                use_subdir = False
            elif str(group) == 'TCGA':
                rad_dir = self.tcga_mri_dir or self.rad_data_dir
                suffix_map = {'Ap': 'AP', 'PVP': 'PVP', 'DP': 'DP'}
                use_subdir = False
            else:
                rad_dir = self.rad_data_dir
                suffix_map = {'Ap': 'Ap', 'PVP': 'PVP', 'DP': 'DP'}
                use_subdir = True

            try:
                img_tensor = None

                if rad_load_mode in ["auto", "npy"]:
                    npy_candidates = [
                        os.path.join(rad_dir, stacked_subdir, f"{case_id_norm}_rad_stacked.npy"),
                        os.path.join(rad_dir, stacked_subdir, f"{case_id}_rad_stacked.npy"),
                    ]
                    for npy_path in npy_candidates:
                        img_tensor = self._load_stacked_npy(npy_path)
                        if img_tensor is not None:
                            npy_hits += 1
                            break

                if img_tensor is None and rad_load_mode in ["auto", "png"]:
                    if use_subdir:
                        p1 = os.path.join(rad_dir, 'png_images_224_10', f"{case_id}_{suffix_map['Ap']}.png")
                        p2 = os.path.join(rad_dir, 'png_images_224_10', f"{case_id}_{suffix_map['PVP']}.png")
                        p3 = os.path.join(rad_dir, 'png_images_224_10', f"{case_id}_{suffix_map['DP']}.png")
                    else:
                        p1 = os.path.join(rad_dir, f"{case_id}_{suffix_map['Ap']}.png")
                        p2 = os.path.join(rad_dir, f"{case_id}_{suffix_map['PVP']}.png")
                        p3 = os.path.join(rad_dir, f"{case_id}_{suffix_map['DP']}.png")

                    img_tensor = self._load_png_triplet(p1, p2, p3)
                    if img_tensor is not None:
                        png_hits += 1

                if img_tensor is None:
                    img_tensor = torch.zeros((224, 224, 9), dtype=torch.float32)
                    missing_hits += 1

                img_dict[str(case_id)] = img_tensor
            except Exception as e:
                print(f"Error loading images for {case_id}: {e}")
                img_dict[str(case_id)] = torch.zeros((224, 224, 9), dtype=torch.float32)
                missing_hits += 1

        self.rad_img_dict = img_dict
        if self.print_info:
            print(
                f"_setup_rad_data source stats: npy={npy_hits}, png={png_hits}, missing={missing_hits}, mode={rad_load_mode}"
            )
        

    def return_splits(self, args, csv_path, fold):
        r"""
        Create the train and val splits for the fold
        
        Args:
            - self
            - args : argspace.Namespace 
            - csv_path : String 
            - fold : Int 
        
        Return: 
            - datasets : tuple 
            
        """

        assert csv_path 
        all_splits = pd.read_csv(csv_path)
        print("Defining datasets...")
        # train_split, scaler = self._get_split_from_df(args, all_splits=all_splits, split_key='train', fold=fold, scaler=None)
        train_split, scaler = self._get_split_from_df(args, all_splits=all_splits, split_key='train', fold=fold, scaler=None)

        val_split = self._get_split_from_df(args, all_splits=all_splits, split_key='val', fold=fold, scaler=scaler)

        # args.omic_sizes = args.dataset_factory.omic_sizes
        datasets = (train_split, val_split)
        
        return datasets

    def _get_scaler(self, data):
        """
        Define the scaler for the training dataset using Z-score normalization (StandardScaler).
        The same scaler should be used for the validation set.
        
        Args:
            - data : np.array - The data to fit the scaler on.

        Returns:
            - scaler : StandardScaler - An instance of the scaler fitted to the data.
        """
        scaler = StandardScaler()
        scaler.fit(data)
        return scaler
    def _apply_scaler(self, data, scaler):
        r"""
        Given the datatype and a predefined scaler, apply it to the data 
        
        Args:
            - self
            - data : np.array 
            - scaler : MinMaxScaler 
        
        Returns:
            - data : np.array """
        
        # find out which values are missing
        zero_mask = data == 0

        # transform data
        transformed = scaler.transform(data)
        data = transformed

        # rna -> put back in the zeros 
        data[zero_mask] = 0.
        
        return data

    def _get_split_from_df(self, args, all_splits, split_key: str='train', fold = None, scaler=None, valid_cols=None):
        r"""
        Initialize SurvivalDataset object for the correct split and after normalizing the RNAseq data 
        # ## scalerRNAseq data 
        Args:
            - self 
            - args: argspace.Namespace 
            - all_splits: pd.DataFrame 
            - split_key : String 
            - fold : Int 
            - scaler : Z_SCORE-Scaler
            - valid_cols : List 

        Returns:
            - SurvivalDataset 
            - Optional: Z_SCORE scaler
        
        """
##
        if not scaler:
            scaler = {}
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        mask = self.label_data['case_id'].astype(str).isin(split.astype(str).tolist())
        df_metadata_slide = args.dataset_factory.label_data.loc[mask, :].reset_index(drop=True)
        
        # omics
        # print(self.clinical_data.columns)
        if 'case_id' not in self.clinical_data.columns:
            raise ValueError("Missing 'case_id' column in clinical data.")
        clinical_data_mask = self.clinical_data['case_id'].astype(str).isin(split.astype(str).tolist())
        clinical_data_for_split = self.clinical_data[clinical_data_mask].reset_index(drop=True)
        clinical_data_for_split = clinical_data_for_split.replace(np.nan, "N/A")

        # df_metadata_slideclinical_data_for_splitcase_id
        common_case_ids = list(set(clinical_data_for_split['case_id'].astype(str)).intersection(set(df_metadata_slide['case_id'].astype(str))))
        df_metadata_slide = df_metadata_slide[df_metadata_slide['case_id'].astype(str).isin(common_case_ids)].reset_index(drop=True)
        clinical_data_for_split = clinical_data_for_split[clinical_data_for_split['case_id'].astype(str).isin(common_case_ids)].reset_index(drop=True)
        # print(f"df_metadata_slide: {df_metadata_slide.columns.tolist()}")
        # print("\n")
        # print(f"df_metadata_slide:\n{df_metadata_slide.head()}")
        if split_key == "train":

            numerical_columns = [
            'age', 'NLR', 'PLT', 'tumor_dm', 'Gamma-Glutamyltransferase', 
            'Alkaline_Phosphatase', 'Cirrhosis'
        ]

            # StandardScalerZ-score
            # scaler = StandardScaler()
            data_to_scale = clinical_data_for_split[numerical_columns]
            scaler_for_data = self._get_scaler(data_to_scale)
            clinical_data_for_split[numerical_columns] =  self._apply_scaler(data = data_to_scale, scaler = scaler_for_data)

            scaler[split_key] = scaler_for_data
        elif split_key in ["val"]:

            numerical_columns = [
            'age', 'NLR', 'PLT', 'tumor_dm', 'Gamma-Glutamyltransferase', 
            'Alkaline_Phosphatase', 'Cirrhosis'
        ]
            scaler_for_data = scaler['train']

            # StandardScalerZ-score
            scaler = StandardScaler()
            data_to_scale = clinical_data_for_split[numerical_columns]

            # self.clinical_data[numerical_columns] = scaler.fit_transform(data_to_scale)
            clinical_data_for_split[numerical_columns] = self._apply_scaler(data = data_to_scale, scaler = scaler_for_data)
        if split_key == "train":
            sample=True
            # sample=False
        elif split_key == "val":
            sample=False
            # sample=True
            
        split_dataset = SurvivalDataset(
            split_key=split_key,
            fold=fold,
            study_name=args.study,
            modality=args.modality,
            patient_dict=args.dataset_factory.patient_dict,
            metadata=df_metadata_slide,
            rad_img_dict=self.rad_img_dict,
            # rad_features= self.rad_features,  
            # data_dir= os.path.join(args.data_root_dir, "{}_20x_features".format(args.combined_study)),
            data_dir= os.path.join(args.data_root_dir),
            pathology_dirs={
                'group1': args.dataset_factory.group1_pathology_dir,
                'tcga': args.dataset_factory.tcga_pathology_dir,
            },
            num_classes=self.num_classes,
            label_col = self.label_col,
            valid_cols = valid_cols,
            is_training=split_key=='train',
            clinical_data = clinical_data_for_split,
            num_patches = self.num_patches,
            sample=sample
            )
        # if split_key == "train":
        #     return split_dataset, scaler, None
        # else:
        #     return split_dataset
        if split_key == "train":
            return split_dataset, scaler
        else:
            return split_dataset
    
    def __len__(self):
        return len(self.label_data)
    
class SurvivalDataset(Dataset):

    def __init__(self,
        split_key,
        fold,
        study_name,
        modality,
        patient_dict,
        metadata, 
        # omics_data_dict,
        data_dir, 
        num_classes,
        # rad_features ,
        rad_img_dict,
        pathology_dirs=None,
        label_col="class_label",# ## OS
        # censorship_var = "censorship",
        valid_cols=None,
        is_training=True,
        clinical_data=-1,
        num_patches=4096,
        # omic_names=None,
        sample=True,
        ): 

        super(SurvivalDataset, self).__init__()

        #---> self
        self.split_key = split_key
        self.fold = fold
        self.study_name = study_name
        self.modality = modality
        self.patient_dict = patient_dict
        self.metadata = metadata 
        # self.omics_data_dict = omics_data_dict
        # self.rad_features = rad_features
        self.rad_img_dict = rad_img_dict
        self.data_dir = data_dir
        self.pathology_dirs = pathology_dirs
        self.num_classes = num_classes
        self.label_col = label_col
        # self.censorship_var = censorship_var
        self.valid_cols = valid_cols
        self.is_training = is_training
        self.clinical_data = clinical_data
        self.num_patches = num_patches
        # self.omic_names = omic_names
        # self.num_pathways = len(omic_names)
        self.sample = sample
        self.clinical_data.index = self.clinical_data["case_id"].astype(str)
        # for weighted sampling
        self.slide_cls_id_prep()
    
    def _get_valid_cols(self):
        r"""
        Getter method for the variable self.valid_cols 
        """
        return self.valid_cols

    def slide_cls_id_prep(self):
        r"""
        For each class, find out how many slides do you have
        
        Args:
            - self 
        
        Returns: 
            - None
        
        """
        self.slide_cls_ids = [[] for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.metadata['label'] == i)[0]

            
    def __getitem__(self, idx):
        r"""
        Given the modality, return the correctly transformed version of the data
        
        Args:
            - idx : Int 
        
        Returns:
            - variable, based on the modality 
        
        """
        
        label, slide_ids, clinical_data, case_id = self.get_data_to_return(idx)
        case_id = str(case_id)
        # print('clinical_data get', clinical_data)
                # case_id

        if self.modality in ["pre_cli", "post_cli", "cli"]:
            dummy_patch = torch.zeros((1, 1), dtype=torch.float32)
            dummy_img = torch.zeros((224, 224, 9), dtype=torch.float32)
            dummy_mask = torch.ones((1,), dtype=torch.float32)
            return (dummy_patch, dummy_img, label, clinical_data, dummy_mask)

        if self.modality in ["pre_adapter", "pre", "dlr_adapter", "dlr"]:
            # Adapter branch: strict PRE/DLR without WSI dependency.
            img_tensor_data = self.rad_img_dict.get(case_id, torch.zeros((224, 224, 9)))
            dummy_patch = torch.zeros((1, 1), dtype=torch.float32)
            dummy_mask = torch.ones((1,), dtype=torch.float32)

            if self.modality in ["dlr_adapter", "dlr"]:
                clinical_data = []

            return (dummy_patch, img_tensor_data, label, clinical_data, dummy_mask)

        elif self.modality in ["dlp_adapter", "dlp", "abmil_wsi"]:

            try:
                group = self.clinical_data.loc[case_id, 'group']
            except Exception:
                group = 'train'

            if str(group) == '1' and self.pathology_dirs and self.pathology_dirs.get('group1'):
                curr_data_dir = self.pathology_dirs['group1']
            elif str(group) == 'TCGA' and self.pathology_dirs and self.pathology_dirs.get('tcga'):
                curr_data_dir = self.pathology_dirs['tcga']
            else:
                curr_data_dir = self.data_dir

            patch_features, mask = self._load_wsi_embs_from_path(curr_data_dir, slide_ids)
            dummy_img = torch.zeros((1,), dtype=torch.float32)
            clinical_data = []

            return (patch_features, dummy_img, label, clinical_data, mask)

        elif self.modality in ["survpath", "dlrp"]:
            
            # Determine data_dir based on group
            try:
                group = self.clinical_data.loc[case_id, 'group']
            except Exception:
                group = 'train'
            
            if str(group) == '1' and self.pathology_dirs and self.pathology_dirs.get('group1'):
                curr_data_dir = self.pathology_dirs['group1']
            elif str(group) == 'TCGA' and self.pathology_dirs and self.pathology_dirs.get('tcga'):
                curr_data_dir = self.pathology_dirs['tcga']
            else:
                curr_data_dir = self.data_dir

            patch_features, mask = self._load_wsi_embs_from_path(curr_data_dir, slide_ids)

            img_tensor_data = self.rad_img_dict[case_id]
            # img_tensor_data = torch.tensor(self.rad_img_dict[case_id])

            return (patch_features,img_tensor_data, label,  clinical_data, mask)
        else:
            raise NotImplementedError('Model Type [%s] not implemented.' % self.modality)
    def get_data_to_return(self, idx):
        r"""
        Collect all metadata and slide data to return for this case ID 
        
        Args:
            - idx : Int 
        
        Returns: 
            - label : torch.Tensor
            - event_time : torch.Tensor
            - c : torch.Tensor
            - slide_ids : List
            - clinical_data : tuple
            - case_id : String
        
        """
        case_id = str(self.metadata['case_id'][idx])
        # print('case_id',case_id)
        label = torch.Tensor([self.metadata[self.label_col][idx]]) 

        slide_ids = self.patient_dict[case_id]
        # print('slide_ids',slide_ids)
        clinical_data = self.get_clinical_data(case_id)

        return label, slide_ids, clinical_data, case_id
  


    def _load_wsi_embs_from_path(self, data_dir, slide_ids):
        """
        Load all the patch embeddings from a list of slide IDs stored in HDF5 files.

        Args:
            - self
            - data_dir : String
            - slide_ids : List

        Returns:
            - patch_features : torch.Tensor
            - mask : torch.Tensor

        """
        patch_features = []
        # Modify the code to load HDF5 files
        for slide_id in slide_ids:
            # wsi_path = os.path.join(data_dir, 'h5_files', '{}.h5'.format(slide_id.rstrip('.svs')))
            wsi_path = os.path.join(data_dir, '{}_features.h5'.format(slide_id.rstrip('.svs')))
            with h5py.File(wsi_path, 'r') as hf:
                wsi_bag = torch.tensor(np.array(hf['features']))  # Assuming that the features are stored under 'features' key
                patch_features.append(wsi_bag)
        patch_features = torch.cat(patch_features, dim=0)

        if self.sample:
            max_patches = self.num_patches

            n_samples = min(patch_features.shape[0], max_patches)
            idx = np.sort(np.random.choice(patch_features.shape[0], n_samples, replace=False))
            patch_features = patch_features[idx, :]

            # Make a mask
            if n_samples == max_patches:
                mask = torch.zeros([max_patches])
            else:
                original = patch_features.shape[0]
                how_many_to_add = max_patches - original
                zeros = torch.zeros([how_many_to_add, patch_features.shape[1]])
                patch_features = torch.cat([patch_features, zeros], dim=0)
                mask = torch.cat([torch.zeros([original]), torch.ones([how_many_to_add])])

        else:
            mask = torch.ones([1])

        return patch_features, mask

    def get_clinical_data(self, case_id):
        case_id = str(case_id)
        clinical_data_list = []

        categorical_columns = [
 'major_resection', 
  'Poor_differentiation','MVI', 'Satellite_nodules',
 'BCLC',
  'AFP400',            
        ]
        numerical_columns = [
            'age', 'NLR',
            'PLT','tumor_dm', 'Gamma-Glutamyltransferase', 
            'Alkaline_Phosphatase',
 'Cirrhosis'
        ]

        for column_name in numerical_columns:
            value = self.clinical_data.loc[case_id, column_name]
            clinical_data_list.append(float(value))

        binary_columns = {
            'major_resection': [0, 1],
            'Poor_differentiation': [0, 1],
            'MVI': [0, 1],
            'Satellite_nodules': [0, 1],
            'AFP400': [0, 1],
        }
        categorical_levels = {
            **binary_columns,
            'BCLC': ['A', 'B'],
        }

        for column_name in categorical_columns:
            value = self.clinical_data.loc[case_id, column_name]
            levels = categorical_levels[column_name]
            clinical_data_list.extend([1.0 if str(value) == str(level) else 0.0 for level in levels])

        return clinical_data_list

    def getlabel(self, idx):
        r"""
        Use the metadata for this dataset to return the survival label for the case 
        
        Args:
            - idx : Int 
        
        Returns:
            - label : Int 
        
        """
        label = self.metadata['label'][idx]
        return label

    def __len__(self):
        return len(self.metadata) 