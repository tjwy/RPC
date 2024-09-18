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

fold_df = pd.read_csv('demo_case.csv')

checkpoint_dir = './checkpoint/PRE/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
# case_id
label = 'class_label'
case_ids = fold_df['case_id'].tolist()
labels = fold_df[label].tolist()

file_dir = 'output/'

raw_data = pd.read_excel('clinical_demo.xlsx')

case_id_data = raw_data['case_id']


categorical_columns = [
 'BCLC', 
 'AFP400', 
        ]

        # 
numerical_columns = [
            'age',  'PLT', 
            'tumor_dm', 'Gamma-Glutamyltransferase',
                        'Alkaline_Phosphatase',
            'Cirrhosis','NLR'
        ]

# one-hot
categorical_data_df = pd.get_dummies(raw_data[categorical_columns], columns=categorical_columns)

# one-hot
one_hot_columns = categorical_data_df.columns.tolist()
# # 
scaler = StandardScaler()
numerical_data = pd.DataFrame(scaler.fit_transform(raw_data[numerical_columns]), 
                              columns=numerical_columns, index=raw_data.index)

# 
processed_data = pd.concat([numerical_data, categorical_data_df], axis=1)

# 
processed_data.fillna(processed_data.mean(numeric_only=True), inplace=True)

# 'case_id'
processed_data = pd.concat([case_id_data, processed_data], axis=1)

clinical_df = processed_data

### 
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
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
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss
outline  = 10

def gen_dataset_multi_images(data_dir, id_list, fold_df,clinical_df, img_size=224, outline=outline):
    mask_dir = data_dir + '/mask'
    ori_dir = data_dir + '/ori'

    img_list = []
    label_list = []
    ids_list = []
    clinical_list = []
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

        clinical_list.append(clinical_df[clinical_df['case_id'].astype(str) == id].values[0][1:])
    return img_list, label_list,ids_list,clinical_list

image_size = 224

class SeqVIT(nn.Module):
    # def __init__(self, sequence_length=3, image_size=224,clinical_feature_dim=10):
    def __init__(self, sequence_length=3,clinical_feature_dim=27):
        super(SeqVIT, self).__init__()
        # self.image_size = image_size
        self.sequence_length = sequence_length
        self.vit_base_model = timm.create_model(
            # 'vit_small_patch16_224',
            # 'vit_base_patch16_224',
            # 'tf_efficientnet_b0',
            'resnet50',
            # 'resnet18',
            pretrained=True,
            num_classes=0,  # Removing the classification head
            # img_size=image_size,
            drop_rate=0.1,   # dropout probability is 0.1
        )
        self.dropout = nn.Dropout(0.1)
        self.flatten = nn.Flatten()
        self.linear_layer = nn.Linear(self.vit_base_model.num_features * sequence_length, self.vit_base_model.num_features)
        self.activation = nn.ReLU()  # Adding a ReLU activation function
        self.linear_layer2 = nn.Linear(self.vit_base_model.num_features, self.vit_base_model.num_features//8)
        self.classifier = nn.Linear(self.vit_base_model.num_features//8 + 64, 1)

                # 
        self.clinical_fc = nn.Sequential(
            nn.Linear(clinical_feature_dim, 128),  # 128
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)  # 
        )
        # alpha
        self.alpha = nn.Parameter(torch.tensor(1.0))
    def forward(self, inputs,clinical_features):
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
        clinical_embedding = self.clinical_fc(clinical_features)
        # # alpha
        clinical_embedding = self.alpha * clinical_embedding 
        combined_features = torch.cat((dropout_output2, clinical_embedding), dim=1)
        final_output = self.classifier(combined_features)
        # final_output = torch.sigmoid(final_output)
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
        img = self.img_list[idx]  #  img (224, 224, 9)
        label = self.label_list[idx]
        id = self.ids_list[idx]
        cli = self.cli_list[idx]
        cli = cli.astype(np.float32)
        clinical_tensor = torch.tensor(cli, dtype=torch.float32)
        # 
        img1 = img[:, :, :3]  # 
        img2 = img[:, :, 3:6]  # 
        img3 = img[:, :, 6:]  # 
                # numpyPIL
        img1_pil = Image.fromarray((img1 * 255).astype(np.uint8))
        img2_pil = Image.fromarray((img2 * 255).astype(np.uint8))
        img3_pil = Image.fromarray((img3 * 255).astype(np.uint8))
        #  transform，
        if self.transform:
            img1_pil = self.transform(img1_pil)
            img2_pil = self.transform(img2_pil)
            img3_pil = self.transform(img3_pil)
        # print('img1_pil.shape',img1_pil.shape)  #img1_pil.shape torch.Size([3, 224, 224])
        # 
        img_pil = torch.cat([img1_pil, img2_pil, img3_pil], dim=0) 
        # print('img_pil.shape',img_pil.shape) #img_pil.shape torch.Size([9, 224, 224])
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

def _make_weights_for_balanced_classes(dataset):
    # 
    class_counts = np.bincount(dataset.label_list)
    
    # 
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(dataset.label_list),
        y=dataset.label_list
    )
    
    # ，0
    weights = [0] * len(dataset)
    
    # 
    for i, label in enumerate(dataset.label_list):
        weights[i] = class_weights[label]
    weights = torch.DoubleTensor(weights)
    # print(weights)
    return weights

# # Training loop
# 
all_train_auc = []
all_test_auc = []
all_train_sensitivity = []
all_test_sensitivity = []
all_train_specificity = []
all_test_specificity = []
all_train_ppv = []
all_test_ppv = []
all_train_npv = []
all_test_npv = []
all_train_precision = []
all_test_precision = []
all_train_recall = []
all_test_recall = []


lr = 0.0001
weight_decay= 0.001

#  DataFrame  fold 、
all_predictions_df = pd.DataFrame(columns=['Fold', 'CaseID', 'Prediction', 'TrueLabel'])
# 
for fold in range(5):
    print(f"Fold {fold + 1}:")
        #  fold 、
    all_predictions = []
    all_true_labels = []
    fold_case_ids = []
    # fold
    train_ids = fold_df[fold_df[f'fold-{fold}'] == 'training']['case_id'].tolist()
    val_ids = fold_df[fold_df[f'fold-{fold}'] == 'validation']['case_id'].tolist()

    X_train_fold = [case_id for case_id in case_ids if case_id in train_ids]
    X_val_fold = [case_id for case_id in case_ids if case_id in val_ids]

    train_img_list, train_label_list,train_ids_list,train_clinical_list = gen_dataset_multi_images(file_dir, X_train_fold, fold_df,clinical_df)
    val_img_list, val_label_list,test_ids_list ,test_clinical_list= gen_dataset_multi_images(file_dir, X_val_fold, fold_df,clinical_df)

    train_img = np.array(train_img_list)
    val_img = np.array(val_img_list)
    train_y_binary = np.array(train_label_list)
    val_y_binary = np.array(val_label_list)
    #  DataLoader
    train_dataset = CustomDataset(train_img_list, train_label_list, train_ids_list,train_clinical_list,transform=data_transform2)
    val_dataset = CustomDataset(val_img_list, val_label_list, test_ids_list,test_clinical_list,transform=data_transform1)
    # 
    weights = _make_weights_for_balanced_classes(train_dataset)
    # print('weights',weights)
    
    train_loader = DataLoader(train_dataset, batch_size=64, sampler = WeightedRandomSampler(weights, len(weights),replacement=True), drop_last=False,num_workers=8)
    test_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=8)
    # 
    model = SeqVIT(sequence_length=3)
    # model = SeqVIT(sequence_length=3, image_size=224)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
        # set the random seed
    seed_torch(torch.device('cuda'), seed=42)
    # 
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=5, verbose=True)
    criterion = nn.BCELoss()
    epochs = 50
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        model.train()
        for inputs, labels,ids,clinicals in train_loader:
            optimizer.zero_grad()
            inputs = inputs.to(device)  # CUDA
            labels = labels.to(device).long()  #  torch.long
            clinicals = clinicals.to(device)
            outputs = model(inputs,clinicals)
            outputs = torch.sigmoid(outputs)

            labels = labels.to(torch.float32)  # 
            labels = labels.view(-1, 1)  #  [batch_size, 1]
            loss = criterion(outputs, labels)  # 
            loss.backward()
            optimizer.step()
            # epoch
        # print(f"Epoch {epoch} completed")

        model.eval()
        train_outputs = []
        train_labels = []
        with torch.no_grad():
            for inputs, labels,ids,clinicals in train_loader:
                inputs = inputs.to(device)  
                labels = labels.to(device)  
                clinicals = clinicals.to(device)
                outputs = model(inputs,clinicals)
                outputs = torch.sigmoid(outputs)
                train_outputs.append(outputs)
                train_labels.append(labels)
            train_outputs = torch.cat(train_outputs).cpu().numpy()
            train_labels = torch.cat(train_labels).cpu().numpy()
        test_outputs = []
        test_labels = []
        test_ids = []
        val_loss_total = 0
        with torch.no_grad():
            for inputs, labels,ids,clinicals in test_loader:
                inputs = inputs.to(device)  
                labels = labels.to(device)  
                clinicals = clinicals.to(device)
                outputs = model(inputs,clinicals)
                outputs = torch.sigmoid(outputs)
                labels1 = labels.to(torch.float32)  
                labels1 = labels1.view(-1, 1)  
                val_loss = criterion(outputs, labels1)  # 
                val_loss_total += val_loss.item()
                test_outputs.append(outputs)
                test_labels.append(labels)
                test_ids.extend(ids)
            test_outputs = torch.cat(test_outputs).cpu().numpy()
            test_labels = torch.cat(test_labels).cpu().numpy()

        train_pred_class = (train_outputs > 0.5).astype(np.int32)
        test_pred_class = (test_outputs > 0.5).astype(np.int32)

        train_acc = accuracy_score(train_labels, train_pred_class)
        test_acc = accuracy_score(test_labels, test_pred_class)

        train_precision = precision_score(train_labels, train_pred_class)
        test_precision = precision_score(test_labels, test_pred_class)

        train_recall = recall_score(train_labels, train_pred_class)
        test_recall = recall_score(test_labels, test_pred_class)
        #          
        train_conf_matrix = confusion_matrix(train_labels, train_pred_class)
        test_conf_matrix = confusion_matrix(test_labels, test_pred_class)


        train_specificity = train_conf_matrix[0, 0] / sum(train_conf_matrix[0, :])
        test_specificity = test_conf_matrix[0, 0] / sum(test_conf_matrix[0, :])

        train_ppv = train_conf_matrix[1, 1] / (train_conf_matrix[1, 1]+train_conf_matrix[0, 1])
        test_ppv = test_conf_matrix[1, 1] / (test_conf_matrix[1, 1]+test_conf_matrix[0, 1])

        train_npv = train_conf_matrix[0, 0] / (train_conf_matrix[0, 0] + train_conf_matrix[1, 0])
        test_npv = test_conf_matrix[0, 0] / (test_conf_matrix[0, 0] + test_conf_matrix[1, 0])

        train_auc = roc_auc_score(train_labels, train_outputs)
        test_auc = roc_auc_score(test_labels, test_outputs)

        print('Train Confusion Matrix:')
        print(train_conf_matrix)

        print('Test Confusion Matrix:')
        print(test_conf_matrix)
        val_loss_avg = val_loss_total / len(test_loader)  # Calculate average validation loss

    # EarlyStopping check
        early_stopping(val_loss_avg, model)

        if early_stopping.early_stop:
            print("Early stopping")
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'checkpoint_fold_{fold+1}.pt'))
            break
        # scheduler.step()
    print(f"Fold {fold + 1} Metrics:")
    print(f"Train AUC: {train_auc:.3f}, Test AUC: {test_auc:.3f}")
    print(f"Train Sensitivity: {train_recall:.3f}, Test Sensitivity: {test_recall:.3f}")
    print(f"Train Specificity: {train_specificity:.3f}, Test Specificity: {test_specificity:.3f}")
    print(f"Train PPV: {train_ppv:.3f}, Test PPV: {test_ppv:.3f}")
    print(f"Train NPV: {train_npv:.3f}, Test NPV: {test_npv:.3f}")
    print(f"Train Precision: {train_precision:.3f}, Test Precision: {test_precision:.3f}")
    print(f"Train Recall: {train_recall:.3f}, Test Recall: {test_recall:.3f}\n")
    print(f'Train AUC: {train_auc}')
    print(f'Test AUC: {test_auc}')
    all_predictions.extend(test_outputs.tolist())
    all_true_labels.extend(test_labels.tolist())
    fold_case_ids.extend(test_ids)
    # 
    predicted_classes = (np.array(all_predictions) > 0.5).astype(np.int64)

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
    all_train_auc.append(train_auc)  #  fold  AUC 
    all_test_auc.append(test_auc)  #  fold  AUC 
    
    all_train_sensitivity.append(train_recall)
    all_test_sensitivity.append(test_recall)
    all_train_specificity.append(train_specificity)
    all_test_specificity.append(test_specificity)
    all_train_ppv.append(train_ppv)
    all_test_ppv.append(test_ppv)
    all_train_npv.append(train_npv)
    all_test_npv.append(test_npv)
    all_train_precision.append(train_precision)
    all_test_precision.append(test_precision)
    all_train_recall.append(train_recall)
    all_test_recall.append(test_recall)

#  fold  CSV 
output_file = 'model_predictions_rad_cli.csv'
all_predictions_df.to_csv(output_file, index=False)
print(f'Predictions, true labels, fold numbers, and case IDs saved to {output_file}')
# 

print("Final results for each fold:")
for i in range(5):
    print(f"Fold {i + 1} - Train AUC: {all_train_auc[i]:.3f}, Test AUC: {all_test_auc[i]:.3f}")

print('lr',lr)
print('weight decay',weight_decay)
print('label',label)
print('outline',outline)
