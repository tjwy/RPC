import argparse

def _process_args():
    r"""
    Function creates a namespace to read terminal-based arguments for running the experiment

    Args
        - None 

    Return:
        - args : argparse.Namespace

    """

    parser = argparse.ArgumentParser(description='Configurations for SurvPath Survival Prediction Training')

    #---> study related
    parser.add_argument('--study', type=str, help='study name')
    parser.add_argument('--task', type=str, choices=['survival'])
    parser.add_argument('--n_classes', type=int, default=4, help='number of classes (4 bins for survival)')
    parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
    parser.add_argument("--type_of_path", type=str, default="hallmarks", choices=["xena", "hallmarks", "combine"])
    parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')

    #----> data related
    parser.add_argument('--data_root_dir', type=str, default=None, help='data directory')
    parser.add_argument('--label_file', type=str, default=None, help='Path to csv with labels')
    parser.add_argument('--clinical_file', type=str, default=None, help='Path to csv with clinical variables')
    parser.add_argument('--omics_dir', type=str, default=None, help='Path to dir with omics csv for all modalities')
    parser.add_argument('--rad_data_dir', type=str, default=None, help='Path to dir with rad data')
    parser.add_argument('--external_pathology_dir', type=str, default=None, help='Optional external WSI feature directory')
    parser.add_argument('--tcga_pathology_dir', type=str, default=None, help='Optional TCGA WSI feature directory')
    parser.add_argument('--external_mri_dir', type=str, default=None, help='Optional external MRI image directory')
    parser.add_argument('--tcga_mri_dir', type=str, default=None, help='Optional TCGA MRI image directory')
    parser.add_argument('--num_patches', type=int, default=4096, help='number of patches')
    parser.add_argument('--label_col', type=str, default="milan_2")
    parser.add_argument("--wsi_projection_dim", type=int, default=1)
    ### add
    parser.add_argument("--encoding_layer_1_dim", type=int, default=8)
    parser.add_argument("--encoding_layer_2_dim", type=int, default=16)
    parser.add_argument("--encoder_dropout", type=float, default=0.25)

    #----> split related 
    parser.add_argument('--k', type=int, default=5, help='number of folds (default: 10)')
    parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
    parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
    parser.add_argument('--split_dir', type=str, default=None, help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
    parser.add_argument('--custom_split_file', type=str, default=None, help='Path to a single split file for ensemble/inference')
    parser.add_argument('--ensemble', action='store_true', default=False, help='Run in ensemble mode')
    parser.add_argument('--which_splits', type=str, default="10foldcv", help='where are splits')
        
    #----> training related 
    parser.add_argument('--max_epochs', type=int, default=20, help='maximum number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--opt', type=str, default="adam", help="Optimizer")
    parser.add_argument('--reg_type', type=str, default="None", help="regularization type [None, L1, L2]")
    parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--bag_loss', type=str, choices=['BCEloss','bce_logits','ce_surv', 'cross_entropy',"nll_surv", "nll_rank_surv", "rank_surv", "cox_surv"], default='bce_logits',
                        help='survival loss function (default: ce)')
    parser.add_argument('--alpha_surv', type=float, default=0.0, help='weight given to uncensored patients')
    parser.add_argument('--reg', type=float, default=1e-5, help='weight decay / L2 (default: 1e-5)')
    ##add
    parser.add_argument('--lr_scheduler', type=str, default='cosine')
    parser.add_argument('--warmup_epochs', type=int, default=1)
    parser.add_argument('--cls_threshold', type=float, default=0.5,
                        help='classification threshold for ACC/Sensitivity/Specificity/PPV/NPV metrics')
    parser.add_argument('--adapter_num_workers', type=int, default=2,
                        help='num_workers for adapter modalities (pre_adapter/pre/dlr_adapter/dlr/dlp_adapter/dlp)')
    #---> model related
    parser.add_argument('--fusion', type=str, default=None)
    parser.add_argument('--modality', type=str, default="wsi")
    parser.add_argument('--model_load_dir', type=str, default=None, help='Directory containing model checkpoints')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to a trained model checkpoint for inference')
    parser.add_argument('--mri_encoder_weights', type=str, default=None,
                        help='Optional private MRI encoder checkpoint path. Weights are not included in this public release.')
    parser.add_argument('--mri_encoder_pretrained', action='store_true', default=False,
                        help='Use timm/ImageNet pretrained MRI backbone weights if available locally or downloadable.')
    parser.add_argument('--freeze_mri_encoder', action='store_true', default=False,
                        help='Freeze MRI encoder parameters after optional weight loading')
    parser.add_argument('--encoding_dim', type=int, default=768, help='WSI encoding dim')
    parser.add_argument('--pre_adapter_clinical_dim', type=int, default=19, help='Clinical feature dimension for pre_adapter modality')
    # _model related  
    # parser.add_argument('--encoding_layer_1_dim', type=int, help='Dimension of the first encoding layer')
    # parser.add_argument('--encoding_layer_2_dim', type=int, help='Dimension of the second encoding layer')
    
    # parser.add_argument('--encoder_dropout', type=float, help='Dropout rate for the encoder layers')
    parser.add_argument('--label_frac', type=float, default=1, help='The fraction of labels to use for the experiment')
    parser.add_argument('--use_nystrom', action='store_false', default=False,help='Whether to use Nystrom approximation or not')
    args = parser.parse_args()

    if not (args.task == "survival"):
        print("Task and folder does not match")
        exit()

    return args