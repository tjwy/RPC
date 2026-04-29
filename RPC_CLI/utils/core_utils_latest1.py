import numpy as np
import pdb
import os
from custom_optims.radam import RAdam
from custom_optims.lamb import Lamb
# from models.model_SurvPath_latest11_cli import SurvPath
from models.model_SurvPath_freeze_RPC import SurvPath
from models.model_DLRP import SurvPathDLRP
from models.model_pre_dlr_adapter import SurvPathPREAdapter, SurvPathDLRAdapter, SurvPathDLPAdapter
from models.model_19CLI import ClinicalOnlyModel

# from models.model_TMIL import TMIL
# from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, brier_score, integrated_brier_score, cumulative_dynamic_auc
# from sksurv.util import Surv
import torch.nn as nn
from transformers import (
    get_constant_schedule_with_warmup, 
    get_linear_schedule_with_warmup, 
    get_cosine_schedule_with_warmup
)
import csv
#----> pytorch imports
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from utils.general_utils_fufa import _get_split_loader, _print_network, _save_splits
from utils.loss_func import NLLSurvLoss

import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score


def _resolve_cls_threshold(args_or_threshold):
    """Resolve and clamp the classification threshold to [0, 1]."""
    if isinstance(args_or_threshold, (int, float)):
        threshold = float(args_or_threshold)
    else:
        threshold = float(getattr(args_or_threshold, "cls_threshold", 0.5))
    return min(1.0, max(0.0, threshold))

def _get_splits(datasets, cur, args):
    r"""
    Summarize the train and val splits and return them individually
    
    Args:
        - datasets : tuple
        - cur : Int 
        - args: argspace.Namespace
    
    Return:
        - train_split : SurvivalDataset
        - val_split : SurvivalDataset
    
    """

    print('\nTraining Fold {}!'.format(cur))
    print('\nInit train/val splits...', end=' ')
    train_split, val_split = datasets
    # _save_splits(datasets, ['train', 'val'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    train_dataset, val_dataset = datasets  # train_dataset  val_dataset train_split  val_split
    _save_splits([train_dataset, val_dataset], ['train', 'val'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    return train_split,val_split


def _init_loss_function(args):
    r"""
    Init the binary classification loss function
    
    Args:
        - args : argspace.Namespace 
    
    Returns:
        - loss_fn : nn.BCEWithLogitsLoss or nn.CrossEntropyLoss
    
    """
    print('\nInit loss function...', end=' ')

    if args.bag_loss == 'bce_logits':  # Sigmoid logits 
        loss_fn = nn.BCEWithLogitsLoss( reduction='mean')
    elif args.bag_loss == 'cross_entropy':  # one-hot 
        loss_fn = nn.CrossEntropyLoss( reduction='mean')
    elif args.bag_loss == 'BCEloss':
        loss_fn = nn.BCELoss()
    else:
        raise ValueError("Invalid value for binary_loss argument. Choose between 'bce_logits' and 'cross_entropy'.")
    print('Done!')
    return loss_fn

def _init_optim(args, model):
    r"""
    Init the optimizer 
    
    Args: 
        - args : argspace.Namespace 
        - model : torch model 
    
    Returns:
        - optimizer : torch optim 
    """
    print('\nInit optimizer ...', end=' ')

    if args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    elif args.opt == "adamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.opt == "radam":
        optimizer = RAdam(model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.opt == "lamb":
        optimizer = Lamb(model.parameters(), lr=args.lr, weight_decay=args.reg)
    else:
        raise NotImplementedError

    return optimizer

def _init_model(args):
    
    print('\nInit Model...', end=' ')

    if args.modality == "survpath":

        model_dict = {'num_classes': args.n_classes}
        model_dict['mri_encoder_weights'] = getattr(args, 'mri_encoder_weights', None)
        model_dict['mri_encoder_pretrained'] = bool(getattr(args, 'mri_encoder_pretrained', False))
        model_dict['freeze_mri_encoder'] = bool(getattr(args, 'freeze_mri_encoder', False))
        model = SurvPath(**model_dict)

    elif args.modality == "dlrp":

        model_dict = {'num_classes': args.n_classes}
        model = SurvPathDLRP(**model_dict)

    elif args.modality in ["pre_cli", "post_cli", "cli"]:

        clinical_dim = int(getattr(args, "pre_adapter_clinical_dim", 19))
        model_dropout = float(getattr(args, "encoder_dropout", 0.1))
        model = ClinicalOnlyModel(clinical_dim=clinical_dim, dropout=model_dropout)

    elif args.modality in ["pre_adapter", "pre"]:

        clinical_dim = int(getattr(args, "pre_adapter_clinical_dim", 19))
        model = SurvPathPREAdapter(
            clinical_dim=clinical_dim,
            pretrained_backbone=bool(getattr(args, 'mri_encoder_pretrained', False)),
            mri_encoder_weights=getattr(args, 'mri_encoder_weights', None),
            freeze_mri_encoder=bool(getattr(args, 'freeze_mri_encoder', False)),
        )

    elif args.modality in ["dlr_adapter", "dlr"]:

        model = SurvPathDLRAdapter(
            pretrained_backbone=bool(getattr(args, 'mri_encoder_pretrained', False)),
            mri_encoder_weights=getattr(args, 'mri_encoder_weights', None),
            freeze_mri_encoder=bool(getattr(args, 'freeze_mri_encoder', False)),
        )

    elif args.modality in ["dlp_adapter", "dlp", "abmil_wsi"]:

        wsi_input_dim = int(getattr(args, "encoding_dim", 1024))
        model_dropout = float(getattr(args, "encoder_dropout", 0.1))
        model = SurvPathDLPAdapter(wsi_input_dim=wsi_input_dim, dropout=model_dropout)


    else:
        raise NotImplementedError
    print(f"Initialized model: {model.__class__.__name__}")
    if torch.cuda.is_available():
        model = model.to(torch.device('cuda'))

    print('Done!')
    _print_network(args.results_dir, model)  # ## block grad-cam-fixed
    return model

def _init_loaders(args, train_split, val_split):
    r"""
    Init dataloaders for the train and val datasets 

    Args:
        - args : argspace.Namespace 
        - train_split : SurvivalDataset 
        - val_split : SurvivalDataset 
    
    Returns:
        - train_loader : Pytorch Dataloader 
        - val_loader : Pytorch Dataloader

    """

    print('\nInit Loaders...', end=' ')
    if train_split:
        train_loader = _get_split_loader(args, train_split, training=True, testing=False, weighted=args.weighted_sample, batch_size=args.batch_size)
    else:
        train_loader = None

    if val_split:
        val_loader = _get_split_loader(args, val_split,  testing=False, batch_size=1)
    else:
        val_loader = None
    print('Done!')

    return train_loader,val_loader

def _extract_survival_metadata(train_loader, val_loader,args):
    r"""
    Extract binary class labels from the train and val loader and combine to get labels for the fold.

    Args:
        - train_loader : Pytorch Dataloader
        - val_loader : Pytorch Dataloader
        - class_label_col : str, default='class_label'
            The name of the column containing binary class labels.

    Returns:
        - all_labels : np.array
            Combined binary class labels for both training and validation sets.

    """
    class_label_col= args.label_col
    all_labels = np.concatenate(
        [train_loader.dataset.metadata[class_label_col].to_numpy(),
        val_loader.dataset.metadata[class_label_col].to_numpy()],
        axis=0)
    return all_labels
def _unpack_data(modality, device, data):
    r"""
    Depending on the model type, unpack the data and put it on the correct device
    
    Args:
        - modality : String 
        - device : torch.device 
        - data : tuple 
    
    Returns:
        - data_WSI : torch.Tensor
        - mask : torch.Tensor
        - y_disc : torch.Tensor
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - data_omics : torch.Tensor
        - clinical_data_list : list
        - mask : torch.Tensor
    
    """
    
    if modality in ["pre_cli", "post_cli", "cli"]:

        data_WSI = data[0].to(device)
        data_rad = None
        y_disc = data[2]
        clinical_data_list = data[3]
        mask = None

    elif modality in ["dlp_adapter", "dlp", "abmil_wsi"]:

        data_WSI = data[0].to(device)
        y_disc = data[2]
        clinical_data_list = data[3]

        if data[4].dim() >= 2 and data[4][0, 0] == 1:
            mask = None
        else:
            mask = data[4].to(device)

        data_rad = None

    elif modality in ["survpath", "dlrp", "pre_adapter", "pre", "dlr_adapter", "dlr"]:

        data_WSI = data[0].to(device)

        rad_items = data[1]
        if isinstance(rad_items, list):
            if len(rad_items) == 0:
                raise ValueError("Empty MRI batch in collate output")
            if isinstance(rad_items[0], torch.Tensor):
                data_rad = torch.stack([item.float() for item in rad_items], dim=0).to(device)
            else:
                data_rad = torch.tensor(rad_items, dtype=torch.float32, device=device)
        elif isinstance(rad_items, torch.Tensor):
            data_rad = rad_items.to(device)
        else:
            data_rad = torch.tensor(rad_items, dtype=torch.float32, device=device)

        if data_rad.dim() == 3:
            data_rad = data_rad.unsqueeze(0)

        if data[4].dim() >= 2 and data[4][0, 0] == 1:
            mask = None
        else:
            mask = data[4].to(device)

        y_disc = data[2]
        clinical_data_list = data[3]
    else:
        raise ValueError('Unsupported modality:', modality)

    y_disc = y_disc.to(device)
    # return data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask
    return data_WSI, mask, y_disc,  data_rad, clinical_data_list, mask

def _process_data_and_forward(model, modality, device, data):
    r"""
    Depeding on the modality, process the input data and do a forward pass on the model 
    
    Args:
        - model : Pytorch model
        - modality : String
        - device : torch.device
        - data : tuple
    
    Returns:
        - out : torch.Tensor
        - y_disc : torch.Tensor
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - clinical_data_list : List
    
    """
    # data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask = _unpack_data(modality, device, data)
    data_WSI, mask, y_disc, data_rad, clinical_data_list, mask = _unpack_data(modality, device, data)

    # print(f"y_disc shape: {y_disc.shape}, type: {type(y_disc)}")  # 

        # WSI
    # print(f"WSI Features shape before passing to model: {data_WSI.shape}")
    # print(f"WSI Features shape before passing to model: {type(data_rad)}")
 
    if modality in ["pre_cli", "post_cli", "cli"]:
        clinical_data_tensor = torch.tensor(clinical_data_list, dtype=torch.float32, device=device)
        if clinical_data_tensor.dim() == 1:
            clinical_data_tensor = clinical_data_tensor.unsqueeze(0)
        out = model(clinical_data=clinical_data_tensor)

    elif modality in ["dlp_adapter", "dlp", "abmil_wsi"]:
        out = model(x_wsi=data_WSI.to(device), mask=mask)

    elif modality in ["survpath", "dlrp"]:
        input_args = {
            "x_wsi": data_WSI.to(device)
        }

        # assert isinstance(img_features, torch.Tensor), f"img_features should be a torch.Tensor but got {type(img_features)}"
        # print(f"img_features type after unpacking: {type(img_features)}")
        # input_args
        # print('data_rad',type(data_rad))
        input_args["x_img"] = data_rad.to(device)

        input_args["return_attn"] = False
        
        if modality == "dlrp":
            # DLRP explicitly removes the clinical branch.
            input_args["clinical_data"] = None
        else:
            assert isinstance(clinical_data_list, list) and all(isinstance(x, list) for x in clinical_data_list), ""
            flattened_clinical_data_list = [float(item) for sublist in clinical_data_list for item in sublist]
            clinical_data_tensors = [torch.tensor(item).to(device) for item in flattened_clinical_data_list]
            clinical_data_tensor = torch.stack(clinical_data_tensors)
            input_args["clinical_data"] = clinical_data_tensor
        out = model(**input_args)

    elif modality in ["pre_adapter", "pre", "dlr_adapter", "dlr"]:
        input_args = {
            "x_img": data_rad.to(device),
            "return_attn": False,
        }
        if modality in ["pre_adapter", "pre"]:
            clinical_data_tensor = torch.tensor(clinical_data_list, dtype=torch.float32, device=device)
            if clinical_data_tensor.dim() == 1:
                clinical_data_tensor = clinical_data_tensor.unsqueeze(0)
            input_args["clinical_data"] = clinical_data_tensor
        else:
            input_args["clinical_data"] = None
        out = model(**input_args)

    else:
        raise ValueError(f"Unsupported modality for forward: {modality}")

    if len(out.shape) == 1:
        out = out.unsqueeze(0)

    return out, y_disc, clinical_data_list


def _calculate_risk(h):
    r"""
    Take the logits of the model and calculate the risk for the patient 
    
    Args: 
        - h : torch.Tensor 
    
    Returns:
        - risk : torch.Tensor 
    
    """
    hazards = torch.sigmoid(h)
    survival = torch.cumprod(1 - hazards, dim=1)
    risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
    return risk, survival.detach().cpu().numpy()

def _update_arrays(all_risk_scores,  all_clinical_data,  risk, clinical_data_list):
    r"""
    Update the arrays with new values 
    
    Args:
        - all_risk_scores : List
        - all_censorships : List
        - all_event_times : List
        - all_clinical_data : List
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - risk : torch.Tensor
        - clinical_data_list : List
    
    Returns:
        - all_risk_scores : List
        - all_censorships : List
        - all_event_times : List
        - all_clinical_data : List
    
    """
    all_risk_scores.append(risk)
    all_clinical_data.append(clinical_data_list)
    return all_risk_scores, all_clinical_data
    
def bootstrap_metric(y_true, y_pred_or_scores, metric_func, n_bootstraps=1000):
    n_bootstraps = int(n_bootstraps)

    # Fast path: point estimate only (used for intermediate epochs).
    if n_bootstraps <= 0:
        try:
            value = metric_func(y_true, y_pred_or_scores)
        except Exception:
            value = np.nan
        return float(value), float(value), float(value)

    metrics = []
    for _ in range(n_bootstraps):
        resampled_indices = np.random.choice(len(y_true), len(y_true), replace=True)
        resampled_y_true = y_true[resampled_indices]
        # AUC
        # if metric_func == roc_auc_score:
        resampled_y_pred_or_scores = y_pred_or_scores[resampled_indices]  
        # else: 
        #     resampled_y_pred_or_scores = (y_pred_or_scores[resampled_indices] > 0.5).astype(int)
        try:
            metrics.append(metric_func(resampled_y_true, resampled_y_pred_or_scores))
        except Exception:
            # Skip invalid bootstrap samples (e.g., one-class AUC resamples).
            continue
    if len(metrics) == 0:
        return np.nan, np.nan, np.nan
    metrics = np.array(metrics)
    lower_bound, upper_bound = np.percentile(metrics, [2.5, 97.5])
    mean = np.mean(metrics)
    return mean, lower_bound, upper_bound

def ppv(y_true, y_pred):
    train_conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    den = train_conf_matrix[1, 1] + train_conf_matrix[0, 1]
    return train_conf_matrix[1, 1] / den if den > 0 else 0.0

def npv(y_true, y_pred):
    train_conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    den = train_conf_matrix[0, 0] + train_conf_matrix[1, 0]
    return train_conf_matrix[0, 0] / den if den > 0 else 0.0

def specificity(y_true, y_pred):
    train_conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    den = sum(train_conf_matrix[0, :])
    return train_conf_matrix[0, 0] / den if den > 0 else 0.0

def calculate_metrics_all(all_labels, all_probs, threshold=0.5, n_bootstraps=1000):
    """
    Calculate various metrics including AUC, Accuracy, Sensitivity, Specificity, PPV, NPV.
    
    Args:
        - all_labels : np.array
            True binary labels.
        - all_probs : np.array
            Predicted probabilities for the positive class.
    
    Returns:
        - metrics_dict : dict
            A dictionary containing calculated metrics.
    """
    metrics_dict = {}
    all_labels = np.asarray(all_labels).reshape(-1)
    all_probs = np.asarray(all_probs).reshape(-1)
    threshold = _resolve_cls_threshold(threshold)
    all_preds = (all_probs > threshold).astype(int)

    # Calculate AUC
    metrics_dict['AUC'] = bootstrap_metric(all_labels, all_probs, roc_auc_score, n_bootstraps=n_bootstraps)

    # Calculate Accuracy
    metrics_dict['ACC'] = bootstrap_metric(all_labels, all_preds, accuracy_score, n_bootstraps=n_bootstraps)

    # Calculate Sensitivity, Specificity, PPV, NPV using a custom bootstrap function
    metrics_dict['recall'] = bootstrap_metric(all_labels, all_preds, recall_score, n_bootstraps=n_bootstraps)
    metrics_dict['SPECIFICITY'] = bootstrap_metric(all_labels, all_preds, specificity, n_bootstraps=n_bootstraps)
    metrics_dict['PPV'] = bootstrap_metric(all_labels, all_preds, ppv, n_bootstraps=n_bootstraps)
    metrics_dict['NPV'] = bootstrap_metric(all_labels, all_preds, npv, n_bootstraps=n_bootstraps)
    return metrics_dict


def _train_loop_survival(epoch, model, modality, loader, optimizer, scheduler, loss_fn, cls_threshold=0.5, n_bootstraps=0):
    r"""
    Perform one epoch of training 

    Args:
        - epoch : Int
        - model : Pytorch model
        - modality : String 
        - loader : Pytorch dataloader
        - optimizer : torch.optim
        - loss_fn : custom loss function class 
    
    Returns:
        - c_index : Float
        - total_loss : Float 
    
    """
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    total_loss = 0.
    cls_threshold = _resolve_cls_threshold(cls_threshold)

    running_corrects = 0
    all_probs = []
    all_labels = []
    all_risk_scores = []

    all_clinical_data = []
    all_case_ids = []  # <- case_id
    # one epoch
    for batch_idx, data in enumerate(loader):
        
        optimizer.zero_grad()

        h, y_disc,clinical_data_list = _process_data_and_forward(model, modality, device, data)

        # if y_disc.dim() == 2 and y_disc.size(1) == 1:
        #     y_disc = y_disc.squeeze(1)
        # y_disc = y_disc.long()

        y_disc = y_disc.type(torch.float)

        loss = loss_fn(input=h, target=y_disc)
        loss_value = loss.item()
        batch_size = y_disc.shape[0]
        
        risk, _ = _calculate_risk(h)

        all_risk_scores, all_clinical_data = _update_arrays(all_risk_scores, all_clinical_data, risk, clinical_data_list)

        total_loss += loss_value * batch_size

        probs = torch.sigmoid(h)  # Convert logits to probabilities
        # all_probs.append(probs[:, 1].detach().cpu().numpy())
        ###
        all_probs.append(probs.detach().cpu().numpy())
        all_labels.append(y_disc.detach().cpu().numpy())
        loss.backward()

        optimizer.step()
        scheduler.step()
                # === case_idslide_id ===
        case_ids = loader.dataset.metadata.iloc[batch_idx : (batch_idx+1)]['case_id']
        all_case_ids.extend(case_ids)
        # if (batch_idx % 20) == 0:
        #     print("batch: {}, loss: {:.3f}".format(batch_idx, loss.item()))
    
    total_loss /= len(loader.dataset)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)

    all_probs = np.concatenate(all_probs).reshape(-1)
    all_labels = np.concatenate(all_labels).reshape(-1)


    # Calculate AUC and accuracy
    train_auc = roc_auc_score(all_labels, all_probs)

    train_acc = accuracy_score(all_labels, (all_probs > cls_threshold).astype(int))

    all_predictions = (all_probs > cls_threshold).astype(np.int32)
    
    cm = confusion_matrix(all_labels, all_predictions)

    # TP, FP, FN, TN
    TP = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TN = cm[0, 0]

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    PPV = TP / (TP + FP) if (TP + FP) > 0 else 0
    NPV = TN / (TN + FN) if (TN + FN) > 0 else 0
    print('Epoch: {}, train_loss: {:.4f}, train_auc: {:.4f}, train_acc: {:.4f}, Sensitivity: {:.3f}, Specificity: {:.3f}, PPV: {:.3f}, NPV: {:.3f}, Thresh: {:.2f}'.format(epoch, total_loss, train_auc, train_acc, sensitivity, specificity, PPV, NPV, cls_threshold))
    # print('Epoch: {}, train_loss: {:.4f}, train_auc: {:.4f}, train_acc: {:.4f}'.format(epoch, total_loss, train_auc, train_acc))
    train_metrics_dict = calculate_metrics_all(
        all_labels,
        all_probs,
        threshold=cls_threshold,
        n_bootstraps=n_bootstraps,
    )
            # ===  ===
        # all_probs  all_labels
    # all_probsall_labels
    all_probs = all_probs.reshape(-1)
    all_labels = all_labels.reshape(-1)
    
    if len(all_case_ids) != len(all_probs):
        # Random/weighted samplers break simple index slicing; keep output lengths aligned.
        all_case_ids = [f"train_case_{i}" for i in range(len(all_probs))]

    train_results = {
        'case_id': all_case_ids,
        'probs': all_probs,
        'labels': all_labels,
        'predictions': (all_probs > cls_threshold).astype(int)
    }

    return train_auc, train_acc, total_loss ,train_metrics_dict,train_results
    # return train_auc, train_acc, total_loss ,train_metrics_dict,cm

def _calculate_metrics(loader, dataset_factory, all_labels, all_probs, cls_threshold=0.5):
    r"""
    Calculate various survival metrics 
    
    Args:
        - loader : Pytorch dataloader
        - dataset_factory : SurvivalDatasetFactory
        - survival_train : np.array
        - all_risk_scores : np.array
        
    Returns:
        - iauc : Float
        - acc : Float
    """
    
    # ... ()

    iauc, acc = 0., 0.

    all_labels = np.asarray(all_labels).reshape(-1)
    all_probs = np.asarray(all_probs).reshape(-1)
    cls_threshold = _resolve_cls_threshold(cls_threshold)

    # Calculate AUC and accuracy for validation set
    val_auc = roc_auc_score(all_labels, all_probs)
    val_acc = accuracy_score(all_labels, (all_probs > cls_threshold).astype(int))

    # print('val_auc: {:.4f}, val_acc: {:.4f}'.format(val_auc, val_acc))

    return val_auc, val_acc


def _summary(dataset_factory, model, modality, loader, loss_fn, survival_train=None, cls_threshold=0.5, n_bootstraps=0):
    r"""
    Run a validation loop on the trained model 
    
    Args: 
        - dataset_factory : SurvivalDatasetFactory
        - model : Pytorch model
        - modality : String
        - loader : Pytorch loader
        - loss_fn : custom loss function clas
        - survival_train : np.array
    
    Returns:
        - patient_results : dictionary
        - c_index : Float
        - c_index_ipcw : Float
        - BS : List
        - IBS : Float
        - iauc : Float
        - total_loss : Float

    """
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    total_loss = 0.
    cls_threshold = _resolve_cls_threshold(cls_threshold)
    incorrect_cases = [] ## add
    all_case_ids = [] ###add

    all_risk_scores = []
    all_risk_by_bin_scores = []
    all_censorships = []
    all_event_times = []
    all_clinical_data = []
    all_logits = []
    all_slide_ids = []
    all_labels = []
    all_probs = []
    slide_ids = loader.dataset.metadata['slide_id']
    case_ids = loader.dataset.metadata['case_id']
    count = 0
    with torch.no_grad():

        for data in loader:

            data_WSI, mask, y_disc, data_rad, clinical_data_list, mask = _unpack_data(modality, device, data)

            if modality in ["pre_cli", "post_cli", "cli"]:
                clinical_data_tensor = torch.tensor(clinical_data_list, dtype=torch.float32, device=device)
                if clinical_data_tensor.dim() == 1:
                    clinical_data_tensor = clinical_data_tensor.unsqueeze(0)
                h = model(clinical_data=clinical_data_tensor)

            elif modality in ["dlp_adapter", "dlp", "abmil_wsi"]:
                h = model(x_wsi=data_WSI.to(device), mask=mask)

            elif modality in ["survpath", "dlrp"]:

                input_args = {"x_wsi": data_WSI.to(device)}
                # for i in range(len(data_omics)):
                #     input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device)

                # input_args
                input_args["x_img"] = data_rad.to(device)
                flattened_clinical_data_list = [float(item) for sublist in clinical_data_list for item in sublist]
                
                clinical_data_tensors = [torch.tensor(item).to(device) for item in flattened_clinical_data_list]

                clinical_data_tensor = torch.stack(clinical_data_tensors)
                # print(clinical_data_tensor.shape)
                # input_argsclinical_data
                if modality == "dlrp":
                    input_args["clinical_data"] = None
                else:
                    input_args["clinical_data"] = clinical_data_tensor

                # input_args["return_attn"] = False                
                h = model(**input_args)

            elif modality in ["pre_adapter", "pre", "dlr_adapter", "dlr"]:
                input_args = {
                    "x_img": data_rad.to(device),
                    "return_attn": False,
                }
                if modality in ["pre_adapter", "pre"]:
                    clinical_data_tensor = torch.tensor(clinical_data_list, dtype=torch.float32, device=device)
                    if clinical_data_tensor.dim() == 1:
                        clinical_data_tensor = clinical_data_tensor.unsqueeze(0)
                    input_args["clinical_data"] = clinical_data_tensor
                else:
                    input_args["clinical_data"] = None
                h = model(**input_args)
                
            else:
                raise ValueError(f"Unsupported modality in summary: {modality}")
                    
            if len(h.shape) == 1:
                h = h.unsqueeze(0)


            
            # y_disc 
            # if y_disc.dim() == 2 and y_disc.size(1) == 1:
            #     y_disc = y_disc.squeeze(1)
            # y_disc = y_disc.long()
            y_disc = y_disc.type(torch.float)
            loss = loss_fn(input=h, target=y_disc)
            loss_value = loss.item()
            batch_size = y_disc.shape[0]


            risk, risk_by_bin = _calculate_risk(h)
            all_risk_by_bin_scores.append(risk_by_bin)
            all_risk_scores,  clinical_data_list = _update_arrays(all_risk_scores, all_clinical_data,  risk, clinical_data_list)
            probs = torch.sigmoid(h)
            # probs = probs[:, 1]
            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(y_disc.detach().cpu().numpy())
            total_loss += loss_value * batch_size
            all_slide_ids.append(slide_ids.values[count])
            count += 1

    total_loss /= len(loader.dataset)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)

    all_probs = np.concatenate(all_probs, axis=0).reshape(-1)
    all_labels = np.concatenate(all_labels, axis=0).reshape(-1)
    all_predictions = (all_probs > cls_threshold).astype(np.int32)

    cm = confusion_matrix(all_labels, all_predictions)

    patient_results = {}

    for i in range(len(all_slide_ids)):
        slide_id = slide_ids.values[i]
        case_id = slide_id[:12]
        patient_results[case_id] = {}
        # patient_results[case_id]["risk"] = all_risk_scores[i]
        # patient_results[case_id]["clinical"] = all_clinical_data[i]
        patient_results[case_id]["probs"] = all_probs[i]
        patient_results[case_id]['preds'] = all_predictions[i]
        patient_results[case_id]['labels'] = all_labels[i]
    iauc, acc = _calculate_metrics(loader, dataset_factory, all_labels, all_probs, cls_threshold=cls_threshold)

    val_metrics_dict = calculate_metrics_all(
        all_labels,
        all_probs,
        threshold=cls_threshold,
        n_bootstraps=n_bootstraps,
    )
    return patient_results,iauc, acc,total_loss, val_metrics_dict
    # return patient_results,iauc, acc,total_loss, val_metrics_dict,cm

def _get_lr_scheduler(args, optimizer, dataloader):
    scheduler_name = args.lr_scheduler
    warmup_epochs = args.warmup_epochs
    epochs = args.max_epochs if hasattr(args, 'max_epochs') else args.epochs

    if warmup_epochs > 0:
        warmup_steps = warmup_epochs * len(dataloader)
    else:
        warmup_steps = 0
    if scheduler_name=='constant':
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps
        )
    elif scheduler_name=='cosine':
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(dataloader) * epochs,
        )
    elif scheduler_name=='linear':
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(dataloader) * epochs,
        )
    return lr_scheduler


def _step(cur, args, loss_fn, model, optimizer, scheduler,train_loader, val_loader):
    r"""
    Trains the model for the set number of epochs and validates it.
    
    Args:
        - cur
        - args
        - loss_fn
        - model
        - optimizer
        - lr scheduler 
        - train_loader
        - val_loader
        
    Returns:
        - results_dict : dictionary
        - val_cindex : Float
        - val_cindex_ipcw  : Float
        - val_BS : List
        - val_IBS : Float
        - val_iauc : Float
        - total_loss : Float
    """

    all_survival = _extract_survival_metadata(train_loader, val_loader,args)
    cls_threshold = _resolve_cls_threshold(args)
    bootstrap_n = int(os.environ.get("SURVPATH_BOOTSTRAP_N", "1000"))
    bootstrap_final_only = str(os.environ.get("SURVPATH_BOOTSTRAP_FINAL_ONLY", "1")).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    epoch_bootstrap_n = 0 if bootstrap_final_only else bootstrap_n

    print('Using classification threshold for metrics: {:.2f}'.format(cls_threshold))
    if bootstrap_final_only:
        print('Bootstrap CI policy: final validation only (n_bootstraps={})'.format(bootstrap_n))
    else:
        print('Bootstrap CI policy: every epoch (n_bootstraps={})'.format(bootstrap_n))
    
    for epoch in range(args.max_epochs):
        # _,_,_,train_metrics_dict,train_cm = _train_loop_survival(epoch, model, args.modality, train_loader, optimizer,scheduler, loss_fn)
        _,_,_,train_metrics_dict,train_results = _train_loop_survival(
            epoch,
            model,
            args.modality,
            train_loader,
            optimizer,
            scheduler,
            loss_fn,
            cls_threshold=cls_threshold,
            n_bootstraps=epoch_bootstrap_n,
        )
        
        # evaluate the model
        # val_patient_results, val_iauc, val_acc, val_loss = _summary(val_loader, loss_fn, model, args.modality)
        # results_dict, val_iauc,acc, total_loss,val_metrics_dict,val_cm = _summary(args.dataset_factory, model, args.modality, val_loader, loss_fn, all_survival)
        results_dict, val_iauc,acc, total_loss,val_metrics_dict = _summary(
            args.dataset_factory,
            model,
            args.modality,
            val_loader,
            loss_fn,
            all_survival,
            cls_threshold=cls_threshold,
            n_bootstraps=epoch_bootstrap_n,
        )
        # print('epoch val_iauc: {:.4f} | epoch Val acc: {:.4f} '.format(
        # val_iauc, 
        # acc
        # ))
        
    # save the trained model
    torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    # _,_,_,train_metrics_dict = _train_loop_survival(epoch, model, args.modality, train_loader, optimizer,scheduler, loss_fn)
    # results_dict, val_iauc, acc, total_loss, val_metrics_dict,val_cm = _summary(args.dataset_factory, model, args.modality, val_loader, loss_fn, all_survival)
    results_dict, val_iauc, acc, total_loss, val_metrics_dict  = _summary(
        args.dataset_factory,
        model,
        args.modality,
        val_loader,
        loss_fn,
        all_survival,
        cls_threshold=cls_threshold,
        n_bootstraps=bootstrap_n,
    )
    print('Final val_iauc: {:.3f} | Final Val acc: {:.3f} '.format(
        val_iauc, 
        acc
        ))

    # return results_dict, (train_metrics_dict, val_metrics_dict, val_iauc, acc,  total_loss, (train_cm, val_cm))
    return results_dict, (train_metrics_dict, val_metrics_dict, val_iauc, acc,  total_loss),train_results 

def _train_val(datasets, cur, args):
    """   
    Performs train val test for the fold over number of epochs

    Args:
        - datasets : tuple
        - cur : Int 
        - args : argspace.Namespace 
    
    Returns:
        - results_dict : dict
        - val_cindex : Float
        - val_cindex2 : Float
        - val_BS : Float
        - val_IBS : Float
        - val_iauc : Float
        - total_loss : Float
    """

    #----> gets splits and summarize
    train_split, val_split = _get_splits(datasets, cur, args)
    
    #----> init loss function
    loss_fn = _init_loss_function(args)

    #----> init model
    # model = _init_model(args, cur)
    model = _init_model(args)  # args  _init_model 
    
    #---> init optimizer
    optimizer = _init_optim(args, model)
    
    #---> init loaders
    train_loader, val_loader = _init_loaders(args, train_split, val_split)
    # lr scheduler 
    lr_scheduler = _get_lr_scheduler(args, optimizer, train_loader)
    # #---> do train val
    # results_dict, (train_metrics_dict,val_metrics_dict,val_iauc,acc, total_loss, confusion_matrices) = _step(cur, args, loss_fn, model, optimizer, lr_scheduler,train_loader, val_loader)

    # return results_dict, (train_metrics_dict,val_metrics_dict, val_iauc, acc,total_loss, confusion_matrices)

    #---> do train val
    results_dict, (train_metrics_dict,val_metrics_dict,val_iauc,acc, total_loss),train_results  = _step(cur, args, loss_fn, model, optimizer, lr_scheduler,train_loader, val_loader)

    return results_dict, (train_metrics_dict,val_metrics_dict, val_iauc, acc,total_loss),train_results 