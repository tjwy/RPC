# Adapted from https://github.com/mahmoodlab/SurvPath/blob/main/main.py
# @article{jaume2023modeling,
#   title={Modeling Dense Multimodal Interactions Between Biological Pathways and Histology for Survival Prediction},
#   author={Jaume, Guillaume and Vaidya, Anurag and Chen, Richard and Williamson, Drew and Liang, Paul and Mahmood, Faisal},
#   journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
#   year={2024}
# }

#----> pytorch imports
import torch
import warnings
#----> general imports
import pandas as pd
import numpy as np
import numpy
import pdb
import os
from timeit import default_timer as timer
from datasets.dataset_survival_latest import SurvivalDatasetFactory
from utils.core_utils_latest1 import _train_val
from utils.file_utils import _save_pkl
from utils.general_utils import _get_start_end, _prepare_for_experiment
import random
from utils.process_args import _process_args

numpy.random.seed(42)
random.seed(42)
torch.manual_seed(42)

def main(args):

    #----> prep for 5 fold cv study
    folds = _get_start_end(args)
    
    #----> storing the val and test cindex for 5 fold cv
    all_val_iauc = []
    all_val_loss = []
    all_acc = []
    # all_train_metrics = []
    all_val_metrics = []
    #  final_metrics_df
    final_metrics_df = pd.DataFrame()
    for i in folds:
        
        datasets = args.dataset_factory.return_splits(
            args,
            csv_path='{}/splits_{}.csv'.format(args.split_dir, i),
            fold=i
        )
        
        print("Created train and val datasets for fold {}".format(i))

        results, (val_metrics_dict,val_iauc, acc,total_loss) = _train_val(datasets, i, args)

        all_val_iauc.append(val_iauc)
        all_acc.append(acc)
        all_val_loss.append(total_loss)

        all_val_metrics.append(val_metrics_dict)

        # Assuming `results` is a dictionary
        df = pd.DataFrame(results)
        # Save DataFrame to CSV
        filename = os.path.join(args.results_dir, 'split_{}_results.csv'.format(i))
        df.to_csv(filename, index=False)
        
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        print("Saving results...")
        _save_pkl(filename, results)
        # DataFrame
        # train_metrics_series = pd.Series({f'train_{key}': f"{train_metrics_dict[key][0]:.3f} ({train_metrics_dict[key][1]:.3f}, {train_metrics_dict[key][2]:.3f})" for key in train_metrics_dict})
        val_metrics_series = pd.Series({f'val_{key}': f"{val_metrics_dict[key][0]:.3f} (95% CI: {val_metrics_dict[key][1]:.3f}, {val_metrics_dict[key][2]:.3f})" for key in val_metrics_dict})
        # metrics_df = pd.DataFrame({
        #     'folds': [f'Fold {i+1}'],
        #     **{f'val_{key}': [val_metrics_dict.get(key, np.nan)] for key in val_metrics_dict.keys()},
        # })
        metrics_df = pd.DataFrame({
            'folds': [f'Fold {i+1}'],
            **val_metrics_series.to_dict(),
        })
        #  final_metrics_df
        final_metrics_df = pd.concat([final_metrics_df, metrics_df], ignore_index=True)
    final_df = pd.DataFrame({
        'folds': folds,
        'val_iauc': all_val_iauc,
        "val_loss": all_val_loss,
        "val_acc" : all_acc,
    })
            # Calculate the mean of val_iauc
    val_iauc_mean = np.mean(all_val_iauc)
    print("Mean val_iauc:", val_iauc_mean)
            # Add mean val_iauc to the final_df
    final_df.loc[len(final_df)] = ['mean', val_iauc_mean, np.nan, np.nan]
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
        
    final_df.to_csv(os.path.join(args.results_dir, save_name))

    print(type(final_metrics_df))

    save_name = 'summary_metrics.csv' if len(folds) == args.k else 'summary_metrics_partial.csv'
    final_metrics_df.to_csv(os.path.join(args.results_dir, save_name))

    print("Metrics summary saved to", os.path.join(args.results_dir, save_name))
 
warnings.filterwarnings("ignore", category=FutureWarning)


if __name__ == "__main__":
    start = timer()


    #----> read the args
    args = _process_args()
    print(f"Use Nystrom approximation: {args.use_nystrom}")
    #----> Prep
    args = _prepare_for_experiment(args)
    args.dataset_factory = SurvivalDatasetFactory(
        study=args.study,
        label_file=args.label_file,
        rad_data_dir=args.rad_data_dir,  # 
        seed=args.seed, 
        print_info=True, 
        # n_bins=args.n_classes, 
        label_col=args.label_col, 
        eps=1e-6,
        num_patches=args.num_patches,
        # is_mcat = True if args.modality == "coattn" else False,
        # is_survpath = True if args.modality == "survpath" else False,
        # type_of_pathway=args.type_of_path,
    )

    #---> perform the experiment
    results = main(args)

    #---> stop timer and print
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))