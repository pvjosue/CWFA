# 2022/2023 Josue Page Vizcaino pv.josue@gmail.com

# Third party libraries imports
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from datetime import datetime
import os, argparse, glob

# Import test script
from CWFA import run_CWFA
# from main_OOD import evaluate_OOD_prediction
from XLFMDataset import *
from utils import *
from networks import *

parser = argparse.ArgumentParser()

# Arguments
parser.add_argument('--main_data_path', nargs='?', default='XLFM_data/Datasets/')
parser.add_argument('--data_folder', nargs='?', default=[])
parser.add_argument('--data_folder_test', nargs='?', default=[])
parser.add_argument('--dataset_ids', nargs='?', default=[])
parser.add_argument('--dataset_ids_test', nargs='?', default=[])
parser.add_argument('--cross_validation_nFold', type=int, default=1)
parser.add_argument('--use_sparse_for_all', type=int, default=1)
parser.add_argument('--lenslet_file', nargs='?', default= "XLFM_data/lenslet_centers_python.txt")
parser.add_argument('--images_to_use', nargs='+', type=int, default=10)#list(range(50,55)))
parser.add_argument('--images_to_use_test', nargs='+', type=int, default=[0,250])
parser.add_argument('--images_to_use_fine_tune_val', nargs='+', type=int, default=5)

parser.add_argument('--seed', type=int, default=364898)
parser.add_argument('--use_half_precision', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=221)
parser.add_argument('--learning_rate_first_step', type=float, default=80)
parser.add_argument('--loss_func_first_step', type=str, default='L2', help="L1, L2 or wL2 or LL")
parser.add_argument('--loss_func_reg', type=str, default='L2', help="L1, L2 or wL2 or LL")
parser.add_argument('--learning_rate_cond', type=float, default=845)
parser.add_argument('--learning_weight_decay', type=float, default=1e-2)
parser.add_argument('--add_noise', type=int, default=1, help='Apply noise to imag=es? 0 or 1')

# Logging configuration
parser.add_argument('--eval_every', type=int, default=25)
parser.add_argument('--save_every', type=int, default=25)
parser.add_argument('--save_model', type=int, default=1)
parser.add_argument('--save_tiff_volumes', type=int, default=1, help="Save volumes and image logs")
parser.add_argument('--save_images', type=int, default=0, help="Save images")
parser.add_argument('--files_to_store', nargs='+', default='*.py')

parser.add_argument('--load_pretrained_networks', type=int, default=0, help='If loading older networks is desired, select here the number of iteration.') 
parser.add_argument('--output_testing_path', type=str, default=f'output/WF_XLFM_23/cleanup/CWF/')


# Volume loading arguments
parser.add_argument('--volume_norm_func', nargs='?', default= None)
parser.add_argument('--volume_ths', type=float, default=[0.0,20000])
parser.add_argument('--images_ths', type=float, default=[0.01,1])
parser.add_argument('--quantile_ths', type=float, default=[0,0.99999])
parser.add_argument('--n_depths', type=int, default=96)
parser.add_argument('--volume_side_size', type=int, default=512)


# For comparison against other methods
parser.add_argument('--evaluation_dataset', type=str, default='train')
parser.add_argument('--neural_activation_filter_width', nargs='+', type=float, default=10)

parser.add_argument('--evaluation_prefix', type=str, default=f'')
parser.add_argument('--main_gpu', nargs='+', type=int, default=-2)
parser.add_argument('--n_threads', nargs='+', type=int, default=8)


# OOD settings
parser.add_argument('--step_LL_to_use', type=int, default=0)
parser.add_argument('--step_LL_ths_to_use', type=float, default=-1.33)

# Create OOD plots? 
parser.add_argument('--create_dist_plots', type=int, default=0)

# Use pretrained networks?
parser.add_argument('--pretrain_models_path', type=str, default='')
parser.add_argument('--fine_tune_optimize_steps', type=int, nargs='+', default=[1,2,3,4,5], help="Which steps to train: [1,2,3,4,5]")
parser.add_argument('--fine_tune_load_checkpoints', nargs='?', default=[], help="Load pretrained models [1,2,3,4,5] to load all steps")
parser.add_argument('--max_test_load_epoch', type=int, default=25000)
parser.add_argument('--fine_tune_use_model_args', type=int, default=0, help="Flag to use the loaded arguments, instead of the general args")


parser.add_argument('--force_all_steps_NF', type=int, default=0)
parser.add_argument('--force_last_step_NF', type=int, default=0)
parser.add_argument('--disable_low_res_input', type=int, default=0)
parser.add_argument('--train_with_gt_low_res', type=int, default=0)

# INN settings
parser.add_argument('--INN_net_type', type=int, default=1, help='0: regular INN with condition, 1: conditional Wavelet flow net, 2: XLFMNet')
parser.add_argument('--INN_down_steps', type=int, default=5, help='Number of downsample operations: 1 to INN_max_down_steps, where INN_max_down_steps corresponds to computing z')
parser.add_argument('--INN_max_down_steps', type=int, default=5, help='Number of downsample operations')
parser.add_argument('--INN_use_perm', type=int, default=1)
parser.add_argument('--INN_use_bias', type=int, default=1)
parser.add_argument('--INN_n_blocks', type=int, default=4)
parser.add_argument('--INN_internal_chans', type=int, default=64)
parser.add_argument('--INN_cond_chans', type=int, default=32)
parser.add_argument('--INN_cond_weight', type=float, default=0.40984)
parser.add_argument('--INN_block_type', type=str, default='CAT', help='RNVP, GLOW, AI1, CAT see networks.py')
parser.add_argument('--INN_z_temperature', type=float, default=0.0, help='0: conv network \n 1: Preprocess psf and image and conv.')
parser.add_argument('--INN_n_samples', type=int, default=1, help="Number of samples to average inside NF")

args = parser.parse_args()

# Pretrained networks to evaluate
if args.load_pretrained_networks==1:
    if args.INN_net_type==1:
        runs_dir = 'pretrained_networks/'
        if args.force_all_steps_NF==0:
            if args.cross_validation_nFold==0:
                args.pretrain_models_path = f'{runs_dir}/2023_04_19__10:38:41_501E___Slurm0_CV0_0.9T_50Timgs_L2_Optim_train_all__lrTrain_/_train___'
            elif args.cross_validation_nFold==1:
                args.pretrain_models_path = f'{runs_dir}/2023_04_19__15:09:15_501E___Slurm0_CV1_0.9T_50Timgs_L2_Optim_train_all__lrTrain_/_train___'
            elif args.cross_validation_nFold==2:
                args.pretrain_models_path = f'{runs_dir}/2023_04_19__10:38:40_501E___Slurm0_CV2_0.9T_50Timgs_L2_Optim_train_all__lrTrain_/_train___'
            elif args.cross_validation_nFold==3:
                args.pretrain_models_path = f'{runs_dir}/2023_04_19__10:38:40_501E___Slurm0_CV3_0.9T_50Timgs_L2_Optim_train_all__lrTrain_/_train___'
            elif args.cross_validation_nFold==4:
                args.pretrain_models_path = f'{runs_dir}/2023_04_19__14:29:18_501E___Slurm0_CV4_0.9T_50Timgs_L2_Optim_train_all__lrTrain_/_train___'
            elif args.cross_validation_nFold==5:
                args.pretrain_models_path = f'{runs_dir}/2023_04_19__10:38:34_501E___Slurm0_CV5_0.9T_50Timgs_L2_Optim_train_all__lrTrain_/_train___'
        else:
            pass


################### Lets find the datasets present in the main directory
# Create cross validation sets for single fish train/test
datasets_to_cross_validate = glob.glob(args.main_data_path+'/*')
# Select raw or SLNet_preprocessed deppending if we want sparse images
dataset_paths = {ds.split('/')[-1] : f"{ds}/{'SLNet_preprocessed/' if args.use_sparse_for_all else 'raw/'}"  for ds in datasets_to_cross_validate}
datasets_to_cross_validate = sorted(dataset_paths.keys())

# Create cross validation sets again
n_datasets = len(datasets_to_cross_validate)
all_nums = list(range(n_datasets))
cross_validation_groups = {}
cv_nums = {}
for nn in range(n_datasets):
    train_ids = list(all_nums[:nn]) + list(all_nums[(nn+1):])
    train_sets = [datasets_to_cross_validate[ix]for ix in train_ids]
    test_sets = datasets_to_cross_validate[nn]
    cross_validation_groups[nn] = {'train' : train_sets, 'val' : train_sets, 'test' : [test_sets]}
    cv_nums[nn] = {'train' : train_ids, 'val' : train_ids, 'test' : [nn]}

# In case that we want to use a single fish for training we can use CV sets > 30
for fish_ix,curr_fish in enumerate(datasets_to_cross_validate):
    try:
        cross_validation_groups[30+fish_ix] = {'train' : [curr_fish], 'test' : [cross_validation_groups[fish_ix]['train'][0]]}
    except:
        pass

cv = args.cross_validation_nFold
args.dataset_ids = cross_validation_groups[cv]['train']
args.dataset_ids_test = cross_validation_groups[cv]['test']

if args.evaluation_prefix == '':
    args.evaluation_prefix = f'CV{cv}_{args.INN_z_temperature}T'

print(f'Runing with prefix: {args.evaluation_prefix}')


args.data_folder = [dataset_paths[dd] for dd in args.dataset_ids]
args.data_folder_test = [dataset_paths[dd] for dd in args.dataset_ids_test]


if isinstance(args.dataset_ids_test,str):
    args.dataset_ids_test = args.dataset_ids_test.split(',')
    if len(args.dataset_ids_test)==1 and len(args.dataset_ids_test[0])==0:
        args.dataset_ids_test = None

# If cross validation is requested, set dataset-id's
if args.cross_validation_nFold is not None:
    if not args.dataset_ids or len(args.dataset_ids)==0:
        args.dataset_ids = cross_validation_groups[args.cross_validation_nFold]['train']

    # If one id is provided use that one instead
    if not args.dataset_ids_test or len(args.dataset_ids_test)==0:
        args.dataset_ids_test = cross_validation_groups[args.cross_validation_nFold]['test']

    # We add two more augmented datasets, so lets reduce the number of images comming from the other datasets
    if isinstance(args.images_to_use, list):
        if len(args.images_to_use)==1:
            args.images_to_use = args.images_to_use[0]
        #else:
         #   args.images_to_use = len(args.images_to_use)
    if args.cross_validation_nFold>=5:
        args.images_to_use = args.images_to_use*len(cross_validation_groups[0]['train']) // len(cross_validation_groups[args.cross_validation_nFold]['train'])

    if isinstance(args.images_to_use_test, list):
        if len(args.images_to_use_test)==1:
            args.images_to_use_test = args.images_to_use_test[0]
            args.images_to_use_test = args.images_to_use_test*len(cross_validation_groups[0]['train']) // len(args.dataset_ids_test)

# Adjust number of samples deppending on provided input
start_sample = 0
n_samples = 500
n_samples_test = 500
if isinstance(args.images_to_use,list) and len(args.images_to_use)!=1:
    args.images_to_use = [img_ix+start_sample for img_ix in args.images_to_use]
elif isinstance(args.images_to_use,list) and len(args.images_to_use)==1:
    args.images_to_use = args.images_to_use[0]
elif isinstance(args.images_to_use,int):
    if cv<30:
        args.images_to_use = args.images_to_use//len(args.dataset_ids)
    n_samples = np.max([n_samples,args.images_to_use])
    args.images_to_use = list(range(start_sample, start_sample+n_samples,n_samples//int(args.images_to_use)))[:args.images_to_use]

# For test and validation, we start loading after 500 images, by using n_samples as tarting point
if isinstance(args.images_to_use_test,list) and len(args.images_to_use_test)==1:
    args.images_to_use_test = int(args.images_to_use_test[0])
if isinstance(args.images_to_use_test, int):
    args.images_to_use_test = list(range(n_samples, n_samples+args.images_to_use_test))[:args.images_to_use_test]

if isinstance(args.images_to_use_fine_tune_val,list) and len(args.images_to_use_fine_tune_val)==1:
    args.images_to_use_fine_tune_val = int(args.images_to_use_fine_tune_val[0])
if isinstance(args.images_to_use_fine_tune_val, int):
    args.images_to_use_fine_tune_val = list(range(n_samples, n_samples+args.images_to_use_fine_tune_val))[:args.images_to_use_fine_tune_val]


# If dataset_ids are provided, use those instead
if len(args.dataset_ids):
    args.data_folder = [dataset_paths[dd] for dd in args.dataset_ids]
if len(args.dataset_ids_test):
    args.data_folder_test = [dataset_paths[dd] for dd in args.dataset_ids_test]


################### Set parameters
# Allow int numbers to be passed as arguments, for example when using with Guild.ai or Slurm
if args.learning_rate >=1:
    args.learning_rate /= 1e7
if args.learning_rate_first_step >=1:
    args.learning_rate_first_step /= 1e7
if args.learning_rate_cond >=1:
    args.learning_rate_cond /= 1e7

# Set random seed
set_all_seeds(args.seed)

# Fetch Device to use
if args.main_gpu==-1:
    device = torch.device('cuda')
elif args.main_gpu==-2:
    device = torch.device(
        "cuda:"+str(get_free_gpu()) if torch.cuda.is_available() else "cpu"
    )
else:
    device = torch.device(
        "cuda:"+str(args.main_gpu) if torch.cuda.is_available() else "cpu"
    )

torch.set_num_threads(args.n_threads)

n_threads = 0
print(F'Torch version: {torch.__version__} {device}')

# Set volume size
vol_shape = 2*[args.volume_side_size] + [args.n_depths]
inn_input_shape = [args.n_depths] + 2*[args.volume_side_size]

# Lateral size of PSF in pixels
psf_size_real = 2160
# Load Dataset and lenslet coordinates
mla_coordinates = get_lenslet_centers(args.lenslet_file)
condition_shape = [1, len(mla_coordinates), args.volume_side_size, args.volume_side_size]



####################### Load datasets
dataset = []
dataset_test_array = []
datasets_val_finetunning = []

# Load training data
for ix_dp,curr_path in enumerate(args.data_folder):
    print('Loading training/validation data from ' + curr_path)
    ds = load_XLFM_data(dataset_path=curr_path, lenslet_coords_file=args.lenslet_file, vol_shape=vol_shape,
        img_shape=2*[psf_size_real], images_to_use=args.images_to_use, n_depths_to_fill=args.n_depths, 
        ds_id=args.dataset_ids[ix_dp], volume_ths=args.volume_ths if args.volume_ths else [], 
        volume_quantiles=args.quantile_ths, img_ths=args.images_ths, norm=args.volume_norm_func)
    dataset.append(ds)


# Load testing data from same datasets
for ix_dp,curr_path in enumerate(args.data_folder if args.evaluation_dataset=='train' else args.data_folder_test):
    print('Loading fine tunning validation data from ' + curr_path)
    ds = load_XLFM_data(dataset_path=curr_path, lenslet_coords_file=args.lenslet_file, vol_shape=vol_shape,
        img_shape=2*[psf_size_real], images_to_use=args.images_to_use_fine_tune_val, n_depths_to_fill=args.n_depths, 
        ds_id=args.dataset_ids[ix_dp], volume_ths=args.volume_ths if args.volume_ths else [], 
        volume_quantiles=args.quantile_ths, img_ths=args.images_ths, norm=args.volume_norm_func)
    datasets_val_finetunning.append(ds)

    
# Load testing data from different dataset
for ix_dp, curr_path in enumerate(args.data_folder_test):
    print('Loading testing data from ' + curr_path)
    with torch.no_grad():
        ds = load_XLFM_data(dataset_path=curr_path, lenslet_coords_file=args.lenslet_file, vol_shape=vol_shape,
            img_shape=2*[psf_size_real], images_to_use=args.images_to_use_test, n_depths_to_fill=args.n_depths, 
            ds_id=args.dataset_ids[ix_dp], volume_ths=args.volume_ths if args.volume_ths else [], 
            volume_quantiles=args.quantile_ths, img_ths=args.images_ths, norm=args.volume_norm_func)
        dataset_test_array.append(ds)

# Creating data indices for training and validation splits:
full_dataset = ConcatDataset(*dataset)
full_dataset_val_finetunning = ConcatDataset(*datasets_val_finetunning)  
full_dataset_test = ConcatDataset(*dataset_test_array)

dataset_size = len(full_dataset)
indices = list(range(dataset_size))

# Create dataloaders
train_sampler = SequentialSampler(indices)
val_sampler = SequentialSampler(list(range(len(args.images_to_use_fine_tune_val)*len(datasets_val_finetunning))))
test_sampler = SequentialSampler(np.concatenate([args.images_to_use_test[:len(full_dataset_test)] for dd in range(len(dataset_test_array))]))

data_loaders = \
    {'train' : \
            data.DataLoader(full_dataset, batch_size=args.batch_size, 
                                sampler=train_sampler, pin_memory=False, num_workers=0), \
    'val'   : \
            data.DataLoader(full_dataset_val_finetunning, batch_size=args.batch_size, 
                                sampler=val_sampler, pin_memory=False, num_workers=0, shuffle=False), \
    'test'  : \
            data.DataLoader(full_dataset_test, batch_size=args.batch_size, 
                                sampler=test_sampler, pin_memory=False, num_workers=0, shuffle=False)
    }

# Get training data statistics
train_statistics = list(full_dataset.get_statistics())


################### Prepare information for networks
# Load neural coordinates for each dataset
neural_coordinates_filenames = {kk : [] for kk in ['train','test','val']}
for k,v in cross_validation_groups[args.cross_validation_nFold].items():
    for vv in v:
        ds_path = f'{dataset_paths[vv]}/Neural_activity_coordinates.csv'
        neural_coordinates_filenames[k].append(glob.glob(ds_path)[0])
# Prepare info for network
results_storage = {}
args.fine_tune = len(args.fine_tune_optimize_steps) > 0
network_settings = {'mla_coordinates':mla_coordinates, 'input_volume_shape': inn_input_shape, 'condition_shape': condition_shape, 'vol_shape':vol_shape,
                        'subnetwork': wavelet_flow_subnetwork2D, 'device':device}


dataloders_to_use = data_loaders[args.evaluation_dataset]
args.output_testing_path += F"{datetime.now().strftime('%Y_%m_%d__%H:%M:%S')}_{'test_set__' if args.evaluation_dataset=='test' else ''}{args.epochs}E_{args.evaluation_prefix}_/"


################### Train CWFA
print('\n\n' + 40*'#' + '  Train  ' + 40*'#')
conv_inn, cond_nets, args_nets, train_statistics, output_path, tb_writer, results_storage['train'] = run_CWFA(args,  
    network_settings=network_settings, dataloader=dataloders_to_use, dataloader_validation=data_loaders['val'], dataloader_test=None, train_statistics=train_statistics, 
    eval_every=args.eval_every, pretrain_models_path=args.pretrain_models_path, output_path=args.output_testing_path, output_posfix= F'_{args.evaluation_dataset}__', neural_coordinates_filename=neural_coordinates_filenames)

# Save mean volumes
volumes_cache_path_out = args.output_testing_path + 'mean_volumes/'
with torch.no_grad():
    if not os.path.exists(volumes_cache_path_out):
        os.mkdir(volumes_cache_path_out)
    for k,dl in data_loaders.items():
        if not dl:
            continue
        n_datasets = len(dl.dataset.datasets)
        for ds in dl.dataset.datasets:
            # if k=='test':
            #     ds.gt_cache = [vol_cache*0 for vol_cache in ds.gt_cache]
            torch.save({'mean_vol_gt_cache' : [vol_cache.cpu() for vol_cache in ds.gt_cache]}, f'{volumes_cache_path_out}/mean_vol_{len(ds)}Imgs_ds_{ds.dataset_id}_{k}')

# Clean up
with torch.cuda.device(device):
    torch.cuda.empty_cache() 

################### Perform evaluation on validation and testing
datasets_to_evaluate = ['val','test']
data_loaders_to_compare = ['train','test']
data_loaders_to_compare = {k : data_loaders[k] for k in data_loaders_to_compare}
with torch.no_grad():
    pre_trained_networks = {'conv_inn' : conv_inn, 'cond_nets' : cond_nets, 'args_nets' : args_nets}
    for args.evaluation_dataset in datasets_to_evaluate:
        print('\n\n' + 40*'#' + f'  {args.evaluation_dataset}  ' + 40*'#')
        args.fine_tune = 0
        conv_inn, cond_nets, args_nets, train_statistics,output_path, tb_writer,results_storage[args.evaluation_dataset] = run_CWFA(args, pre_trained_networks=pre_trained_networks, 
                network_settings=network_settings, dataloader=data_loaders[args.evaluation_dataset], train_statistics=train_statistics, 
            output_path=args.output_testing_path, output_posfix= F'_{args.evaluation_dataset}__', neural_coordinates_filename={'train':neural_coordinates_filenames[args.evaluation_dataset]})
    
    # Compute distributions
    if args.evaluation_dataset=='test':
            print('\n\n' + 40*'#' + f'  Out of distribution detection  ' + 40*'#')
            posfix = f'__test_{args.dataset_ids_test[0]}'
            # todo
            # evaluate_OOD_prediction(args, conv_inn, cond_nets, args_nets, data_loaders_to_compare, network_settings, train_statistics, step_LL_to_use=args.step_LL_to_use,  ths_to_use=args.step_LL_ths_to_use,
            #                 output_path=output_path,  posfix=posfix, tb_writer=tb_writer, results_storage=results_storage)
