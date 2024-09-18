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
# 
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered in scalar divide.*')
#----> general imports
import pandas as pd
import numpy as np
import numpy
import pdb
import os
from timeit import default_timer as timer
from datasets.dataset_survival_pre import SurvivalDatasetFactory
from utils.core_utils_latest1 import _train_val
from utils.file_utils import _save_pkl
from utils.general_utils import _prepare_for_experiment
import random
from utils.process_args import _process_args

numpy.random.seed(42)
random.seed(42)
torch.manual_seed(42)

def main(args):

    #  final_metrics_df
    final_metrics_df = pd.DataFrame()

    datasets = args.dataset_factory.return_splits(
        args,
        csv_path='{}/splits.csv'.format(args.split_dir),
    )
    
    print("Created train and val datasets")

    results, ( train_metrics_dict,val_metrics_dict,val_iauc, acc,total_loss) = _train_val(datasets, args)


    # Assuming `results` is a dictionary
    df = pd.DataFrame(results)
    # Save DataFrame to CSV
    filename = os.path.join(args.results_dir, 'split_results.csv')
    df.to_csv(filename, index=False)
    
    #write results to pkl
    filename = os.path.join(args.results_dir, 'split_results.pkl')
    print("Saving results...")
    _save_pkl(filename, results)
    # DataFrame
    
    train_metrics_series = pd.Series({f'train_{key}': f"{train_metrics_dict[key][0]:.3f} (95% CI: {train_metrics_dict[key][1]:.3f}, {train_metrics_dict[key][2]:.3f})" for key in train_metrics_dict})
    val_metrics_series = pd.Series({f'val_{key}': f"{val_metrics_dict[key][0]:.3f} (95% CI: {val_metrics_dict[key][1]:.3f}, {val_metrics_dict[key][2]:.3f})" for key in val_metrics_dict})

    #  final_metrics_df
    final_metrics_df  = pd.DataFrame({
        **train_metrics_series.to_dict(),
        **val_metrics_series.to_dict(),
    })

    save_name = 'summary_metrics.csv'
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
        rad_data_dir=args.rad_data_dir, 
        seed=args.seed, 
        print_info=True, 
        label_col=args.label_col, 
        eps=1e-6,
        num_patches=args.num_patches,
    )

    #---> perform the experiment
    results = main(args)    

    #---> stop timer and print
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))