from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score

from sklearn.model_selection import train_test_split, StratifiedKFold
import torch.nn.functional as F
from torchsummary import summary
import cv2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torchsummary import summary

import timm
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import class_weight
from torchvision import transforms
from torch.nn import DataParallel
import os

import random

def seed_torch(device, seed=42):

    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
def bootstrap_metric(y_true, y_pred_or_scores, metric_func, n_bootstraps=1000):
    metrics = []
    for _ in range(n_bootstraps):
        resampled_indices = np.random.choice(len(y_true), len(y_true), replace=True)
        resampled_y_true = y_true[resampled_indices]
        # if metric_func == roc_auc_score:
        resampled_y_pred_or_scores = y_pred_or_scores[resampled_indices]  
        # else: 
        #     resampled_y_pred_or_scores = (y_pred_or_scores[resampled_indices] > 0.5).astype(int)
        metrics.append(metric_func(resampled_y_true, resampled_y_pred_or_scores))
    metrics = np.array(metrics)
    lower_bound, upper_bound = np.percentile(metrics, [2.5, 97.5])
    mean = np.mean(metrics)
    return mean, lower_bound, upper_bound

def ppv(y_true, y_pred):
    train_conf_matrix = confusion_matrix(y_true, y_pred)
    ppv = train_conf_matrix[1, 1] / (train_conf_matrix[1, 1]+train_conf_matrix[0, 1])
    return ppv

def npv(y_true, y_pred):
    train_conf_matrix = confusion_matrix(y_true, y_pred)
    npv = train_conf_matrix[0, 0] / (train_conf_matrix[0, 0] + train_conf_matrix[1, 0])
    return npv

def specificity(y_true, y_pred):
    train_conf_matrix = confusion_matrix(y_true, y_pred)
    specificity = train_conf_matrix[0, 0] / sum(train_conf_matrix[0, :])
    return specificity


metric_funcs = {
    'AUC': roc_auc_score,
    'Accuracy': accuracy_score,
    'Precision': precision_score,
    'Recall': recall_score,
    'Specificity': specificity,
    'PPV': ppv,
    'NPV': npv,
}

fold_df = pd.read_csv('demo_case.csv')
checkpoint_dir = './checkpoint/DLR/'

# case_id
label = 'class_label'
case_ids = fold_df['case_id'].tolist()
labels = fold_df[label].tolist()


### ， one-hot ，，，。
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). not Saving model ...')
        # torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

def gen_dataset_multi_images(data_dir, id_list, fold_df, img_size=224, outline=10):
    mask_dir = data_dir + '/mask'
    ori_dir = data_dir + '/ori'

    img_list = []
    label_list = []
    ids_list = []
    for id in id_list:
        images_stack = []  #  ID 

        for suffix in ['Ap', 'PVP', 'DP']:
            X_file_name = f"{id}_{suffix}.npy"
            m_file_name = f"{id}_{suffix}.npy"

            if not (os.path.exists(os.path.join(ori_dir, X_file_name)) and
                    os.path.exists(os.path.join(mask_dir, m_file_name))):
                continue

            X = np.load(os.path.join(ori_dir, X_file_name))
            m = np.load(os.path.join(mask_dir, m_file_name))

            # ...
            X = np.uint8(cv2.normalize(X, None, 0, 255, cv2.NORM_MINMAX))
            
            if len(X.shape) == 3 and X.shape[2] == 3:  # 3，BGR
                X_gray = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
            elif len(X.shape) == 2:  # ，
                X_gray = X
            else:
                raise ValueError("Unsupported image shape: {}".format(X.shape))

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            X = clahe.apply(X_gray)

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

            images_stack.append(X_m_1)

        # 
        stacked_image = np.concatenate(images_stack, axis=-1)
        img_list.append(stacked_image)
        label_list.append(fold_df[fold_df.case_id == id][label].values[0])
        ids_list.append(id)

    return img_list, label_list,ids_list

image_size = 224

class SeqVIT(nn.Module):
    def __init__(self, sequence_length=3, image_size=224):
    # def __init__(self, sequence_length=3):
        super(SeqVIT, self).__init__()
        self.image_size = image_size
        self.sequence_length = sequence_length
        self.vit_base_model = timm.create_model(
            # 'vit_small_patch16_224',
            # 'vit_base_patch16_224',
            # 'tf_efficientnet_b0',
            # 'tf_efficientnet_b0',
            'resnet50',
            # 'resnet18',
            ## 'vvgg16',
            pretrained=True,
            num_classes=0,  # Removing the classification head
            # img_size=image_size,
            drop_rate=0.1,   # dropout probability is 0.1 resnetCNN
        )
        self.dropout = nn.Dropout(0.1)
        self.flatten = nn.Flatten()
        self.linear_layer = nn.Linear(self.vit_base_model.num_features * sequence_length, self.vit_base_model.num_features)
        self.activation = nn.ReLU()  # Adding a ReLU activation function
        self.linear_layer2 = nn.Linear(self.vit_base_model.num_features, self.vit_base_model.num_features//8)
        self.classifier = nn.Linear(self.vit_base_model.num_features//8, 1)

    def forward(self, inputs):
        outputs = []
        inputs = inputs.type(torch.FloatTensor).to(device)
        for t in range(self.sequence_length):
            output_t = self.vit_base_model(inputs[:, t * self.sequence_length:(t + 1) * self.sequence_length, :, :])
            outputs.append(output_t)
        concatenated_outputs = torch.cat(outputs, dim=-1)
        flattened_output = self.flatten(concatenated_outputs)
        linear_output = self.linear_layer(flattened_output)
        activated_output = self.activation(linear_output)  # Applying ReLU activation
        dropout_output = self.dropout(activated_output)  # Applying dropout after ReLU
        linear_output2 = self.linear_layer2(dropout_output)  # Using dropout output here
        activated_output2 = self.activation(linear_output2)  # Applying ReLU activation
        dropout_output2 = self.dropout(activated_output2)  # Applying dropout after second ReLU
        final_output = self.classifier(dropout_output2)
        final_output = torch.sigmoid(final_output)
        return final_output

class CustomDataset(Dataset):
    def __init__(self, img_list, label_list, ids_list,cli_list, transform=None):
        self.img_list = img_list
        self.label_list = label_list
        self.transform = transform
        self.ids_list = ids_list
        self.classes = list(set(label_list))
        self.cli_list = cli_list
    def __len__(self):
        return len(self.img_list)
    def get_class_counts(self):
        # 
        unique_labels, counts = np.unique(self.label_list, return_counts=True)
        class_counts = np.array(counts).tolist()
        return class_counts
    def __getitem__(self, idx):
        #  img_list 
        img = self.img_list[idx]  #  img (224, 224, 9)
        label = self.label_list[idx]
        id = self.ids_list[idx]
        cli = self.cli_list[idx]
        cli = cli.astype(np.float32)
        clinical_tensor = torch.tensor(cli, dtype=torch.float32)

        img1 = img[:, :, :3]  # 
        img2 = img[:, :, 3:6]  # 
        img3 = img[:, :, 6:]  # 

        img1_pil = Image.fromarray((img1 * 255).astype(np.uint8))
        img2_pil = Image.fromarray((img2 * 255).astype(np.uint8))
        img3_pil = Image.fromarray((img3 * 255).astype(np.uint8))
        #  transform，
        if self.transform:
            img1_pil = self.transform(img1_pil)
            img2_pil = self.transform(img2_pil)
            img3_pil = self.transform(img3_pil)
        img_pil = torch.cat([img1_pil, img2_pil, img3_pil], dim=0) 
        return img_pil, label, id, clinical_tensor
# Define the data transformation
data_transform1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
data_transform2 = transforms.Compose([

    transforms.RandomResizedCrop(size=224, scale=(0.9, 1.1)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


all_external_auc = []
all_external_acc = []
all_external_precision = []
all_external_recall = []
all_external_specificity = []
all_external_ppv = []
all_external_npv = []
all_external_metrics = []

#  DataFrame  fold 、
all_predictions_df = pd.DataFrame(columns=['Fold', 'CaseID', 'Prediction', 'TrueLabel'])
formatted_results = []
# 
fold = 2
# for fold in range(5):
print(f"Fold {fold + 1}:")
        #  fold 、
all_predictions = []
all_true_labels = []
fold_case_ids = []
# foldtest
test_ids =  fold_df[fold_df[f'fold-{fold}'] == 'test']['case_id'].tolist()
# case_idscase_id
X_test_fold = [case_id for case_id in case_ids if case_id in test_ids]

test_img_list, test_label_list,test_ids_list , gen_dataset_multi_images(file_dir, X_test_fold, fold_df)

test_y_binary = np.array(test_label_list)
test_img = np.array(test_img_list)
#  DataLoader
test_dataset = CustomDataset(test_img_list, test_label_list, test_ids_list,transform=data_transform1)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)

# 
model = SeqVIT(sequence_length=3)

checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{fold+1}_0.0001_0.001.pt')
model.load_state_dict(torch.load(checkpoint_path))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
    # set the random seed
seed_torch(torch.device('cuda'), seed=42)
model.eval()
external_outputs = []
external_labels = []
external_ids = []

with torch.no_grad():
    for inputs, labels, ids, clinicals in test_loader:
        inputs = inputs.to(device)  # CUDA
        labels = labels.to(device)  # CUDA
        # clinicals = clinicals.to(device)
        # outputs = model(inputs,clinicals)
        outputs = model(inputs)
        labels1 = labels.to(torch.float32)  # 
        labels1 = labels1.view(-1, 1)  #  [batch_size, 1]

        external_outputs.append(outputs)
        external_labels.append(labels1)
        external_ids.extend(ids)

external_outputs = torch.cat(external_outputs).cpu().numpy()
external_labels = torch.cat(external_labels).cpu().numpy()
external_pred_class = (external_outputs > 0.5).astype(np.int32)
# ， AUC

external_auc = roc_auc_score(external_labels, external_outputs)
external_acc = accuracy_score(external_labels, external_pred_class)
external_precision = precision_score(external_labels, external_pred_class)
external_recall = recall_score(external_labels, external_pred_class)
external_conf_matrix = confusion_matrix(external_labels, external_pred_class)
external_specificity = external_conf_matrix[0, 0] / sum(external_conf_matrix[0, :])
external_ppv = external_conf_matrix[1, 1] / (external_conf_matrix[1, 1]+external_conf_matrix[0, 1])
external_npv = external_conf_matrix[0, 0] / (external_conf_matrix[0, 0] + external_conf_matrix[1, 0])
external_metrics = {
    'Fold': [fold+1],
    'AUC': [external_auc],
    'Accuracy': [external_acc],
    'Precision': [external_precision],
    'Recall': [external_recall],
    'Specificity': [external_specificity],
    'PPV': [external_ppv],
    'NPV': [external_npv]
}
if fold == 2:
    formatted_results.extend(external_metrics)
                        #  AUC （）
    external_auc_mean, external_auc_low, external_auc_up = bootstrap_metric(
        external_labels, external_outputs, roc_auc_score, n_bootstraps=1000
    )
    for metric_name, metric_func in metric_funcs.items():
        if metric_name != 'AUC':  #  AUC 
            external_mean, external_low, external_up = bootstrap_metric(
                external_labels, external_pred_class, metric_func, n_bootstraps=1000
            )
        else:
            #  AUC，，

            external_mean, external_low, external_up = external_auc_mean, external_auc_low, external_auc_up

        external_formatted = f"Fold {fold + 1}, {metric_name} (External): Mean = {external_mean:.3f} (95% CI: {external_low:.3f}, {external_up:.3f})"
        # 

        formatted_results.append(external_formatted)
        #  CSV 
        with open('final_metrics_formatted_rad_ex).csv', 'w') as f:
            for result in formatted_results:
                f.write(result + '\n')
        print(f'Metrics in the specified format saved to final_metrics_formatted.csv')

external_metrics_df = pd.DataFrame(external_metrics)
all_external_metrics.append(external_metrics_df)
print(f'External Validation AUC: {external_auc}')
    # scheduler.step()
all_predictions.extend(np.around(external_outputs, 3))
all_true_labels.extend(external_labels.tolist())
fold_case_ids.extend(external_ids)
#     
predicted_classes = (np.array(all_predictions) > 0.5).astype(np.int64)
    # 
fold_dataframe = pd.DataFrame({
    'Fold': [fold] * len(fold_case_ids),
    'CaseID': fold_case_ids,
    'Prediction': [None] * len(fold_case_ids),  # 
    'PredictionProb': all_predictions,
    'PredictedClass': predicted_classes.tolist(),
    'TrueLabel': all_true_labels,
    'IsCorrect': np.array(predicted_classes).ravel() == np.array(all_true_labels).ravel()
})
#  fold  DataFrame  all_predictions_df 
all_predictions_df = pd.concat([all_predictions_df, fold_dataframe], ignore_index=True)

# 
all_external_auc.append(external_auc)
all_external_acc.append(external_acc)
all_external_precision.append(external_precision)
all_external_recall.append(external_recall)
all_external_specificity.append(external_specificity)
all_external_ppv.append(external_ppv)
all_external_npv.append(external_npv)

external_metrics_df = pd.concat(all_external_metrics, ignore_index=True)

# DataFrameCSV
external_metrics_df.to_csv('external_metrics_rad.csv', index=False)

#  fold  CSV 
output_file = 'model_predictions_external_rad.csv'
all_predictions_df.to_csv(output_file, index=False)
print(f'Predictions, true labels, fold numbers, and case IDs saved to {output_file}')



