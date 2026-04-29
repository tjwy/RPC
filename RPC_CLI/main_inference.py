import os
from timeit import default_timer as timer

import pandas as pd
import torch

from datasets.dataset_survival_latest import SurvivalDatasetFactory
from utils.core_utils_latest1 import _get_splits, _init_loaders, _init_loss_function, _init_model, _summary
from utils.general_utils_fufa import _get_start_end, _prepare_for_experiment
from utils.process_args import _process_args


def _write_predictions(patient_results, output_path):
    rows = []
    for case_id, values in patient_results.items():
        rows.append({
            "case_id": case_id,
            "probability": values.get("probs"),
            "predicted_label": values.get("preds"),
            "label": values.get("labels"),
        })
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main(args):
    if not args.checkpoint_path:
        raise ValueError("--checkpoint_path is required for inference.")

    folds = _get_start_end(args)
    model = _init_model(args)
    state_dict = torch.load(args.checkpoint_path, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)

    loss_fn = _init_loss_function(args)
    all_prediction_files = []

    for fold in folds:
        datasets = args.dataset_factory.return_splits(
            args,
            csv_path=os.path.join(args.split_dir, f"splits_{fold}.csv"),
            fold=fold,
        )
        train_split, val_split = _get_splits(datasets, fold, args)
        train_loader, val_loader = _init_loaders(args, train_split, val_split)
        patient_results, val_auc, val_acc, val_loss, _ = _summary(
            args.dataset_factory,
            model,
            args.modality,
            val_loader,
            loss_fn,
            survival_train=None,
            cls_threshold=args.cls_threshold,
            n_bootstraps=0,
        )

        output_path = os.path.join(args.results_dir, f"split_{fold}_predictions.csv")
        _write_predictions(patient_results, output_path)
        all_prediction_files.append(output_path)
        print(f"Fold {fold}: AUC={val_auc:.4f}, ACC={val_acc:.4f}, loss={val_loss:.4f}")
        print(f"Saved predictions to {output_path}")

    return all_prediction_files


if __name__ == "__main__":
    start = timer()
    args = _process_args()
    args = _prepare_for_experiment(args)
    args.dataset_factory = SurvivalDatasetFactory(
        study=args.study,
        label_file=args.label_file,
        clinical_file=args.clinical_file,
        rad_data_dir=args.rad_data_dir,
        external_pathology_dir=args.external_pathology_dir,
        tcga_pathology_dir=args.tcga_pathology_dir,
        external_mri_dir=args.external_mri_dir,
        tcga_mri_dir=args.tcga_mri_dir,
        seed=args.seed,
        print_info=True,
        label_col=args.label_col,
        eps=1e-6,
        num_patches=args.num_patches,
        need_rad_data=(args.modality not in ["dlp_adapter", "dlp", "abmil_wsi", "pre_cli", "post_cli", "cli"]),
    )
    main(args)
    end = timer()
    print("finished!")
    print("Script Time: %f seconds" % (end - start))
