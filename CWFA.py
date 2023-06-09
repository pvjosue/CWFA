# 2022/2023 Josue Page Vizcaino pv.josue@gmail.com

# Third party libraries imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast,GradScaler
from torch.profiler import profile, record_function, ProfilerActivity
import torch.autograd.profiler as profiler

import copy
import pathlib
from tifffile import imsave
from datetime import datetime
from tqdm import tqdm
import re, glob, time, sys, os, zipfile
from sklearn.metrics import mean_absolute_error
from lion_pytorch import Lion

# My files
import losses as Losses
from utils import *
from XLFMDataset import XLFMDatasetFull
from networks import *



# getting the original colormap using cm.get_cmap() function
orig_map=plt.cm.get_cmap('inferno')
orig_map.colors[-1] = [1.,1.,1.]
orig_map.colors[-2] = [1.,1.,1.]
orig_map.colors[-3] = [1.,1.,1.]
orig_map.colors[-4] = [1.,1.,1.]
orig_map.colors[-5] = [1.,1.,1.]
# reversing the original colormap using reversed() function
cm = orig_map.reversed()

## Neuron coords
neural_coords = [[190,200],[250,200],[379,271], [465,200], [65,260], [100,210], [640,190],[180,600],[112,595],[465,630],[461,275],[493,202],[415,195]]

def sample_z_truncated(x, device="cpu", temperature=1):
    """
    This function generates a truncated normal distribution with mean 0 and standard deviation `temperature`. The distribution is truncated between `-temperature` and `temperature`. If `x` is a tensor, the function returns a tensor of the same shape as `x`. If `x` is an integer, the function returns a tensor of shape `(x,)`.
    @param x - an integer or a tensor
    @param device - the device to use for the tensor
    @param temperature - the standard deviation of the truncated normal distribution
    @return a tensor of the same shape as `x` or `(x,)`
    """
    if torch.is_tensor(x):
        if temperature==0:
            return torch.zeros_like(x, device=device)
        else:
            return _no_grad_trunc_normal_(torch.zeros_like(x, device=device), a=-temperature, b=temperature)
    else:
        if temperature==0:
            return torch.zeros(x, device=device)
        else:
            return _no_grad_trunc_normal_(torch.zeros(x, device=device), a=-temperature, b=temperature)
    
def sample_z_rev_like(x, device="cpu", temperature=0, same_size=False):
    """
    This function generates a tensor of random values with the same shape as the input tensor x.
    @param x - the input tensor
    @param device - the device to use for the tensor
    @param temperature - the temperature to use for the tensor
    @param same_size - whether to use the same size as the input tensor
    @return a tensor of random values with the same shape as the input tensor x.
    """
    f = [torch.randn,torch.torch.randn_like] if temperature != 0 else [torch.zeros, torch.zeros_like]
    if isinstance(x,tuple) or isinstance(x,list):
        d = f[0](x, device=device)
    elif same_size:
        d = f[1](x, device=device)
    else:
        d = f[0]([x.shape[0], 3*x.shape[-1], x.shape[2], x.shape[3]], device=device)
    return d * temperature

def check_empty_depths(gt_volume):
    """
    Check if any of the depths in a volume are empty. If any are empty, add a small amount of noise to the empty depths.
    @param gt_volume - the ground truth volume
    @return the ground truth volume with added noise to empty depths
    """
    empty_depths = gt_volume.std(dim=1)==0
    device = gt_volume.device
    n_depths = gt_volume.shape[1]
    empty_depths = empty_depths.view(gt_volume.shape[0], empty_depths.shape[0]//gt_volume.shape[0], empty_depths.shape[1], empty_depths.shape[2])
    if (empty_depths).any():
        gt_volume[empty_depths.repeat(1,n_depths,1,1)] += torch.normal(0, 0.001, gt_volume[empty_depths.repeat(1,n_depths,1,1)].size(), device=device) 
    return gt_volume

def compute_INN_step_performance(gt_volume_in, pred_volume_in, step, mean, std, normaliaze_before_metrics=False, ths=0.05):
    """
    Compute the performance of the INN step by comparing the ground truth volume to the predicted volume.
    @param gt_volume_in - the ground truth volume
    @param pred_volume_in - the predicted volume
    @param step - the step of the INN
    @param mean - the mean value
    @param std - the standard deviation
    @param normalize_before_metrics - whether to normalize before computing metrics
    @param ths - the threshold value
    @return the PSNR, masked PSNR, ground truth volume, and predicted volume
    """
    # Step is zero-based
    # Unnormalize
    gt_volume_raw = gt_volume_in.clone() / 2**step
    gt_volume_raw = gt_volume_raw*std - mean
    
    # Unnormalize prediction
    pred_volume_raw = pred_volume_in.clone() / 2**step
    pred_volume_raw = pred_volume_raw*std - mean

    if normaliaze_before_metrics:
        gt_volume_raw -= gt_volume_raw.min()
        pred_volume_raw -= pred_volume_raw.min()
    
    try:
        if ths!=0:
            # Create mask for MAPE (Mean absolute percentage error)
            p = pred_volume_raw.clone() ; p[p<p.abs().max()*ths] = 0
            masked_psnr = mean_absolute_error(gt_volume_raw.cpu().view(-1).detach().numpy(), p.cpu().view(-1).detach().numpy()) * 100
        curr_psnr = psnr(gt_volume_raw, pred_volume_raw).item()
    except:
        return 0, 0, 1, gt_volume_raw, pred_volume_raw

    return curr_psnr, masked_psnr, gt_volume_raw, pred_volume_raw

def evaluate_INN_forward(conv_inn, cond_nets, args_general, args_nets, gt_volume, input_views, train_statistics, extra_cond_in=None):
    """
    Evaluate the forward pass of an Invertible Neural Network (INN) on a given input volume.
    @param conv_inn - the INN model
    @param cond_nets - the conditional networks
    @param args_general - general arguments
    @param args_nets - arguments for the networks
    @param gt_volume - the ground truth volume
    @param input_views - the input views
    @param train_statistics - the training statistics
    @param extra_cond_in - additional conditional inputs
    @return losses, gt_cache, prior_errors, log_jacobians
    """
    device = gt_volume.device
    gt_volume = check_empty_depths(gt_volume)
    mean_imgs, std_imgs, mean_imgs_s, std_imgs_s, mean_vols, std_vols = train_statistics
    losses = []
    prior_errors = []
    log_jacobians = []
    gt_cache = args_general.INN_max_down_steps * [None]
    gt_cache[0] = gt_volume
    cond_input = (input_views-mean_imgs)/std_imgs
    for n_net in range(len(conv_inn)):
        force_all_steps_NF = args_general.force_all_steps_NF 
        # Compute condition?
        is_last_step = n_net==args_general.INN_max_down_steps-1
        if is_last_step:
            if force_all_steps_NF:
                cond_in = []
            else:
                curr_cond = cond_nets[n_net](cond_input)[-1]
                cond_in = [curr_cond]
        else:
            if len(conv_inn[n_net].dims_c) > 0:
                cond_in = [torch.zeros((gt_volume.shape[0],)+conv_inn[n_net].dims_c[0], device=device)]
            else:
                cond_in = []

        if len(conv_inn[n_net].dims_c) > 1:
            if extra_cond_in is None:
                extra_cond = torch.zeros((gt_volume.shape[0],) + conv_inn[n_net].dims_c[1], device=device) 
                cond_in.append(extra_cond)
            else:
                cond_in.append(extra_cond_in[n_net].clone())
               
        
        # Run again with correct conditions
        Z, log_jac_det = conv_inn[n_net](gt_volume, c=cond_in)
        Z_current = Z[0]
        error_on_prior = torch.norm(Z_current)**2 
        if torch.isinf(error_on_prior):
            print(F'Inf on error_on_prior, step {n_net+1}')
            if torch.isinf(Z_current.max()) or torch.isneginf(Z_current.min()):
                print(F'Inf on Z, step {n_net+1}')

        curr_LL_loss = (0.5*error_on_prior - log_jac_det) / Z[-1].numel()
        losses.append(curr_LL_loss.mean())
        prior_errors.append(0.5*error_on_prior.mean()/ Z[-1].numel())
        log_jacobians.append(log_jac_det.mean()/ Z[-1].numel())
        if not is_last_step:
            gt_volume = Z[1]
            gt_cache[n_net+1] = gt_volume
    return losses, gt_cache, prior_errors, log_jacobians

def plot_distributions(x1, x2, n_std=5):
    """
    This function plots the distributions of two input arrays, x1 and x2, and returns the plot.
    @param x1 - the first input array
    @param x2 - the second input array
    @param n_std - the number of standard deviations to use for clipping the data (default is 5)
    @return the plot of the distributions of x1 and x2
    """
    plt.clf()
    x1_vec = 1.0*x1.reshape(np.prod(x1.shape))
    x2_vec = 1.0*x2.reshape(np.prod(x2.shape))
    # Clamp by n_std
    if n_std!=0:
        x1_std,x1_mean = x1_vec.std(), x1_vec.mean()
        x2_std,x2_mean = x2_vec.std(), x2_vec.mean()
        x1_vec[x1_vec > x1_mean+n_std*x1_std] = x1_mean+n_std*x1_std
        x2_vec[x2_vec > x2_mean+n_std*x2_std] = x2_mean+n_std*x2_std
        x1_vec[x1_vec < x1_mean-n_std*x1_std] = x1_mean-n_std*x1_std
        x2_vec[x2_vec < x2_mean-n_std*x2_std] = x2_mean-n_std*x2_std
    plt.hist([x1_vec,x2_vec], color=['red','blue'], bins=256, alpha=0.5)
    plt.axvline(x1.mean(), color='red', linestyle='--', label='x1 mean', linewidth=0.75)
    plt.axvline(x2.mean(), color='blue', linestyle='--', label='x2 mean', linewidth=0.75)
    plt.legend()
    return plt.gcf()

def read_neural_coordinates_from_file(filename):
    """
    Reads neural coordinates from a file and returns them as a list of lists.
    @param filename - the name of the file containing the neural coordinates
    @return A list of lists containing the neural coordinates
    """
    if isinstance(filename, str):
        dataframe = pd.read_csv(filename)
    else: # Is a list
        dataframe = pd.read_csv(filename[0])
        for F in filename[1:]:
            dataframe.append(pd.read_csv(F))
    coords = [[],[],[]] # x,y,z
    for ix,axis in enumerate(['coord_x','coord_y','coord_z']):
        coords[ix] = dataframe[dataframe['is_gt']==1][axis].tolist()
    return [[coords[0][ix], coords[1][ix], coords[2][ix]] for ix in range(len(coords[0]))]

def corr_coeff_3D(stack_gt, pred_3D, coords, r12, r3, n_time_steps=None, start_plane_offset=-25//2, output_path=None, n_show=20, minmax_ths=50, filter_width=10):
    """
    Calculate the correlation coefficient between the ground truth and predicted data for a given set of 3D coordinates.
    @param stack_gt - the ground truth data
    @param pred_3D - the predicted data
    @param coords - the coordinates to calculate the correlation coefficient for
    @param r12 - the radius of the patch in the x and y directions
    @param r3 - the radius of the patch in the z direction
    @param n_time_steps - the number of time steps
    @param start_plane_offset - the starting plane offset
    @param output_path - the output path
    @param n_show - the number of patches to show
    @param minmax_ths - the minimum threshold
    @param filter_width - the filter width
    @return all_corr_coeffs - the correlation coefficients
    @return neural_activity_dataframe
    """
    if n_time_steps is None:
        n_time_steps = stack_gt.shape[0]
    all_corr_coeffs = []
    # Create storage dataframe
    neural_activity_dataframe = pd.DataFrame(columns=['patch_n','coord_x','coord_y','coord_z','corr_coeff','is_gt']
                                        +[f't{t}' for t in range(n_time_steps)])
    neural_activity_dataframe = neural_activity_dataframe.astype(float)

    plot = output_path is not None

    stack_gt /= stack_gt.max()
    pred_3D /= pred_3D.max()
    n_divisions = 0
    if plot:
        result_vol = np.zeros(pred_3D.shape[1:] + (3,))
        colors = [0,0.25,0]
        coords_to_paint = []
        plt.clf()
        plt.figure(figsize=(10,10))
    required_coords = int(len(coords)*0.2)
    while len(all_corr_coeffs) <= required_coords and n_divisions<5:
        img_ths = stack_gt[stack_gt>0].median() * minmax_ths
            
        for ix,(x_coord,y_coord,z_coord) in enumerate(coords):
            # this coordinates came from the central 25 slices of a stack
            z_coord += stack_gt.shape[1]//2 + start_plane_offset 
            xpix = list(range(max(0,int(x_coord)-r12), min(stack_gt.shape[2], int(x_coord)+r12)))
            ypix = list(range(max(0,int(y_coord)-r12), min(stack_gt.shape[3], int(y_coord)+r12)))
            zpix = list(range(max(0,int(z_coord)-r3), min(stack_gt.shape[1], int(z_coord)+r3)))
            xpix,ypix,zpix = np.meshgrid(xpix, ypix, zpix)
            
            corr_coeff = 0
            try:
                gt_data = stack_gt[:,zpix, ypix, xpix]
                # print('range')
                GT_signal,minmax = norm_data(gt_data.mean(1).mean(1).mean(1).cpu(), min(filter_width, gt_data.shape[-1]))
                if minmax < img_ths:
                    # print(minmax)
                    continue
                pred_rois_data = pred_3D[:,zpix, ypix, xpix]
                # pred_rois_data = stack_prediction[zpix, ypix, xpix]
                pred_signal,minmax = norm_data(pred_rois_data.mean(1).mean(1).mean(1).cpu(), min(filter_width, gt_data.shape[-1]))
                if GT_signal.max()==0 or pred_signal.max()==0:
                    corr_coeff = 0
                else:
                    corr_coeff = np.corrcoef(GT_signal, pred_signal)[0][1]
                
                if plot:
                    for ix_color in range(3):
                        result_vol[zpix, ypix, xpix, ix_color] += colors[ix_color]
                    n_result = len(all_corr_coeffs)
                    if n_result<n_show:
                        plt.subplot(n_show,2,(n_result*2)+1)
                        # Draw
                        plt.plot(GT_signal, label='GT')
                        plt.gca().axes.xaxis.set_ticklabels([])
                        plt.gca().axes.yaxis.set_ticklabels([])
                        plt.plot(pred_signal, label='pred')
                        plt.gca().axes.xaxis.set_ticklabels([])
                        plt.gca().axes.yaxis.set_ticklabels([])
                        plt.grid()
            except:
                corr_coeff = 0
                print(f'roi_outside {ix}')
            all_corr_coeffs.append(corr_coeff)


            # Update dataframe
            data_in = {'patch_n' : ix, 'coord_x' : x_coord, 'coord_y' : y_coord, 'coord_z' : z_coord, 'corr_coeff' : corr_coeff, 'is_gt':1}
            data_in.update({f't{t}' : GT_signal[t] for t in range(len(GT_signal))})
            neural_activity_dataframe = pd.concat([neural_activity_dataframe, pd.DataFrame(data_in, index=[ix])])
            data_in = {'patch_n' : ix, 'coord_x' : x_coord, 'coord_y' : y_coord, 'coord_z' : z_coord, 'corr_coeff' : corr_coeff, 'is_gt':0}
            data_in.update({f't{t}' : pred_signal[t] for t in range(len(pred_signal))})
            neural_activity_dataframe = pd.concat([neural_activity_dataframe, pd.DataFrame(data_in, index=[ix])])

        if len(all_corr_coeffs) <= required_coords:
            minmax_ths /= 2
            n_divisions += 1
            print(f'CC ths: {minmax_ths}', end=' ')
    if len(all_corr_coeffs) == 0:
        return all_corr_coeffs, neural_activity_dataframe
        
    if plot:
        GT_3D_mean = stack_gt[0].clone().detach().cpu().numpy()
        result_vol_gt = result_vol.copy()
        result_vol_gt[...,0] += (GT_3D_mean/GT_3D_mean.max())
        result_vol_gt[...,2] += result_vol_gt[...,0]

        pred_3D_mean = pred_3D[0].clone().detach().cpu().numpy()
        result_vol_pred = result_vol
        result_vol_pred[...,0] += (pred_3D_mean/pred_3D_mean.max())
        result_vol_pred[...,2] += result_vol_pred[...,0]

        img_out = composite_projection(result_vol_gt)
        img_out = np.clip(img_out, 0, 1.0)
        plt.subplot(3,2,2)
        plt.imshow(img_out)
        plt.gca().axes.xaxis.set_ticklabels([])
        plt.gca().axes.yaxis.set_ticklabels([])
        plt.tight_layout()
        plt.grid(b=None)
        
        
        img_out = composite_projection(result_vol_pred)
        img_out = np.clip(img_out, 0, 1.0)
        plt.subplot(3,2,4)
        plt.imshow(img_out)
        plt.gca().axes.xaxis.set_ticklabels([])
        plt.gca().axes.yaxis.set_ticklabels([])
        plt.tight_layout()
        plt.grid(b=None)
        
        plt.subplot(3,2,6)
        plt.plot(all_corr_coeffs)
        plt.hlines([np.mean(all_corr_coeffs)], 0, len(all_corr_coeffs), ['red'], 'dashed')
        plt.grid()

        plt.tight_layout()
        # plt.savefig(f'{output_path}', dpi=300)


        print(f'CC={np.mean(all_corr_coeffs)} from {len(all_corr_coeffs)} coords')
    return all_corr_coeffs, neural_activity_dataframe

def run_CWFA(args, pre_trained_networks=None, network_settings={'input_volume_shape': [11, 512, 512], 'condition_shape': [], 'cond_constructor': None,'subnetwork': None, 'block_settings':{}, 'device':'cuda:0'}, pretrain_models_path='', dataloader=None, dataloader_validation=None, dataloader_test=None, eval_every=100, train_statistics=[], output_path='final_runs_tests', output_posfix='train', neural_coordinates_filename=None, mean_volumes_cache_path='', opt_to_use=Lion):
    """
    This function runs a conditional wavelet flow algorithm. It takes in a variety of arguments, including pre-trained networks, data loaders, and output paths. It then initializes the network and trains it using the provided data. The function logs the training process and saves the trained model to the output path. Finally, it returns the trained model and other relevant information.
    @param args - the arguments for the CWFA algorithm
    @param pre_trained_networks - pre-trained networks to use
    @param network_settings - settings for the network
    @param pretrain_models_path - path to pre-trained models
    @param dataloader - the training data loader
    @param dataloader_validation - the validation data loader
    @param dataloader_test - the test data loader
    @param eval_every - when to performe an evaluation, use -1 to only evaluate at the start and end of the training
    """

    # Fetch device to run computations
    device = network_settings['device']

    ### Load network information from a directory.
    runs_info = args.INN_max_down_steps*[None]
    fine_tune_checkpoints = None
    start_epoch = 0
    force_last_step_NF = 0
    # As there are n upsampling steps, we train each step for epochs/n_steps epochs
    epochs_per_step = int(np.floor(args.epochs/(args.INN_max_down_steps)))
    if eval_every==-1:
        eval_every = args.epochs-1
    else:
        eval_every = min([epochs_per_step, eval_every])
    # Maybe we want to load some pretrained upsampling steps and train others. So we start at the correct epoch
    if len(args.fine_tune_optimize_steps):
        start_epoch = int(args.INN_max_down_steps-np.max(args.fine_tune_optimize_steps)) * epochs_per_step
    else:
        start_epoch = args.epochs-1
        
    # If the user passed as arguments pretrained models, use those        
    if pre_trained_networks:
        conv_inn = pre_trained_networks['conv_inn']
        cond_nets = pre_trained_networks['cond_nets']
        args_nets = pre_trained_networks['args_nets']
    else:
        # Check if runs where found
        runs_found = False
                 
        # If a path of previous runs is provided, load the checkpoint files
        if len(pretrain_models_path) > 0:
            fine_tune_checkpoints = glob.glob(f'{pretrain_models_path}/model_step_*', recursive=True)
            assert len(fine_tune_checkpoints)>0 , f'Error: pretrain_models_path was provided but no models where found in: {pretrain_models_path}'
            runs_found = True
            print(f"Checkpoints found at {pretrain_models_path}")

        for n_step in range(1,args.INN_max_down_steps+1):

            config = {}
            # Fetch the current step being loaded
            config['INN_down_steps'] = n_step
            
            if runs_found:
                # If a path was provided instead of a guild tag, overwrite runs info
                if fine_tune_checkpoints is not None and len(pretrain_models_path):
                    model_highest_epoch = ''
                    max_checkpoint_epoch = 0
                    for cp in fine_tune_checkpoints:
                        max_checkpoint = ''
                        if F'model_step_{n_step}' in cp:
                            model_finetune_epoch = re.match(F'.*model_step_{n_step}__ep_(\d*).*',cp)
                            if model_finetune_epoch:
                                model_epoch = int(model_finetune_epoch[1])
                                if model_epoch>max_checkpoint_epoch:
                                    max_checkpoint_epoch = model_epoch
                                    model_highest_epoch = cp
                                    max_epoch_model_allowed = model_epoch
                    if model_highest_epoch != '':
                        print(f'Found pre-trained model step{n_step}')

                        # Store the one with more iterations
                        config['model_path'] = model_highest_epoch
                        config['model_epoch'] = max_epoch_model_allowed

            
            # Store it in the correct order
            if runs_info[n_step-1] is None:
                runs_info[n_step-1] = config
            else:
                # Grab the run with the highest iteration
                if runs_info[n_step-1]['model_epoch'] < config['model_epoch']:
                    runs_info[n_step-1] = config
                else:
                    print(F'Ignoring due to already loaded model: ',end='  ')
            
        assert all([r is not None for r in runs_info]), f'Some runs are missing... {runs_info}'


        ### Create networks based on loaded info
        n_steps = len(runs_info)
        conv_inn = []
        cond_nets = []
        args_nets = []

        for ix, curr_net_info in enumerate(runs_info):
            if runs_found and ix+1 in args.fine_tune_load_checkpoints:
                # Load model, arguments, etc
                data = torch.load(curr_net_info['model_path'],map_location=device)
                args_model = data['args']
                # args_model.INN_n_blocks = 8
            else:
                args_model = copy.deepcopy(args)
            args_model.INN_down_steps = ix+1

            is_last_step = ix==(n_steps-1)

            force_last_step_NF = 0
            if 'force_last_step_NF' in args_model and args_model.force_last_step_NF:
                force_last_step_NF = 1

            if is_last_step:
                cond_net_temp = Encoder(29, int(network_settings['vol_shape'][-1]/ (2**(args.INN_max_down_steps-1))), 
                                n_steps, args_model.INN_internal_chans, args_model.INN_use_bias).to(device)
            else:
                condition_n_chans = args.n_depths//(2**(ix+1))
                # Create networks for condition and Normalizing flow
                
                cond_shape = [1,29, args.volume_side_size, args.volume_side_size]
                cond_constructor = lambda : cond_network(29, condition_n_chans, ix+1, args_model.INN_max_down_steps, [], args.INN_cond_chans)

                cond_net_temp, conv_inn_temp = conditional_wavelet_flow(input_volume_shape=network_settings['input_volume_shape'], condition_shape=cond_shape, 
                                        st_subnet=network_settings['subnetwork'], conditional_network=cond_constructor, n_internal_ch=args_model.INN_internal_chans,
                                        n_down_steps=ix+1, use_permutations=args_model.INN_use_perm==1, block_type=args_model.INN_block_type, 
                                        n_blocks=args_model.INN_n_blocks, disable_low_res_input=args.disable_low_res_input, device=network_settings['device'])
            
            if not is_last_step or force_last_step_NF:
                conv_inn.append(conv_inn_temp[ix])

                if ix+1 in args.fine_tune_load_checkpoints:
                    try:
                        conv_inn[ix].load_state_dict(data['INN_state_dict'])
                        print(f'Loaded state dic: {ix+1}')
                    except:
                        print(f'Failed to load state dic: {ix}')
                    cond_net_temp.load_state_dict(data['condition_state_dict'], strict=True)
                if train_statistics is None:
                    train_statistics = data['training_statistics']
            elif ix+1 in args.fine_tune_load_checkpoints:# and not ix+1 in args.fine_tune_optimize_steps:
                cond_net_temp.load_state_dict(data['condition_state_dict'], strict=True)
                print(f'Loaded state dic: {ix+1}')
                
            cond_nets.append(cond_net_temp)
            args_nets.append(args_model)
    
    conv_inn = [c.eval().to(device) for c in conv_inn]
    cond_nets = [c.eval().to(device) for c in cond_nets]

    # Let's set the last net to train, important due to the dropout/batchnorm
    cond_nets[-1].train()

    ## reset actNorms with all the data!
    for n_net in args.fine_tune_optimize_steps:
        if n_net<args.INN_max_down_steps-1:
            conv_inn[n_net],_ = reset_ActNorm(conv_inn[n_net])
            conv_inn[n_net] = reset_perm(conv_inn[n_net])
    

    force_last_step_NF =  hasattr(args_nets[-1], 'force_last_step_NF') and args_nets[-1].force_last_step_NF
    force_all_steps_NF = args.force_all_steps_NF #or (hasattr(args_nets[0], 'force_all_steps_NF') and args_nets[0].force_all_steps_NF)
    disable_low_res_input = args.disable_low_res_input #hasattr(args_nets[0], 'disable_low_res_input') and args_nets[0].disable_low_res_input
    
    ## Create log directory and copy files for reproducibility 
    args_nets[0].seed = args.seed
    ############################ Setup outputs
    set_all_seeds(args_nets[0].seed)
    output_path = F"{output_path}/{output_posfix}_/"
    writer = SummaryWriter(log_dir=output_path)
    writer.add_text('arguments_general',str(vars(args)),0)
    writer.add_scalar('sampling_temperature', args.INN_z_temperature)
    for ix,A in enumerate(args_nets):
        writer.add_text(F'arguments_step_{ix}',str(vars(A)),0)
    print(F'Logging directory: {output_path}')

    # Store files for backup
    zf = zipfile.ZipFile(output_path + "/files.zip", "w")
    file_path = pathlib.Path(__file__).parent.absolute()
    files_to_store = glob.glob(str(file_path) + '/' + args.files_to_store)
    for ff in files_to_store:
        zf.write(ff, os.path.split(ff)[-1])
    zf.close()
    
    ############################ Loaded, ready to run
    if dataloader is None:# or len(args.fine_tune_optimize_steps)==0:
        return conv_inn, cond_nets, args_nets, train_statistics, output_path, writer, 0
    
    params = 0
    params_cond = 0
    total_n_params = 0
    for n in range(args.INN_down_steps-1):
        try:
            params += sum([np.prod(p.size()) for p in conv_inn[n].parameters()])
        except:
            pass
        try:
            params_cond += sum([np.prod(p.size()) for p in cond_nets[n].parameters()])
        except:
            pass
        total_n_params += params_cond + params
    params_LRNN = sum([np.prod(p.size()) for p in cond_nets[-1].parameters()])
    total_n_params += params_LRNN
    print(f'nParameters: WF: {params}\tOmega: {params_cond}\tLRNN:{params_LRNN}\t\ttotal: {total_n_params}')
    # Fine-tunning all toguether?
    if args.fine_tune:
        optimizers = args.INN_max_down_steps * [None]
        optimizers_cond = (args.INN_max_down_steps-1) * [None]
        steps_to_optimize = args.fine_tune_optimize_steps
        for curr_step in args.fine_tune_optimize_steps:
            curr_step -= 1
            print(f'Training step {curr_step+1}')
            params_to_use_t = []
            is_last_step = curr_step==(n_steps-1)
            lr = args.learning_rate_first_step if is_last_step else args.learning_rate
            lr_cond = args.learning_rate_cond

            # Lets decide if we use the lr from the args or from each loaded model
            if args.fine_tune_use_model_args:
                lr = args_nets[curr_step].learning_rate
            if is_last_step:
                lr = args.learning_rate_first_step
                params_to_use_t = [{'params': cond_nets[-1].parameters(), 'lr':lr, 'weight_decay':args.learning_weight_decay}]
            else:
                params_to_use_t.append({'params': conv_inn[curr_step].parameters(), 'lr':lr, 'weight_decay':args.learning_weight_decay})#args_nets[curr_step].learning_rate, 'weight_decay':args.learning_weight_decay})#
                # params_to_use_t.append({'params': cond_nets[curr_step].parameters(), 'lr':lr_cond, 'weight_decay':args.learning_weight_decay})
                
                optimizers_cond[curr_step] = opt_to_use(cond_nets[curr_step].parameters(), lr=lr_cond)#, num_epochs=501//4, num_batches_per_epoch=1)
            # Store this optimizer
            optimizers[curr_step] = opt_to_use(params_to_use_t, lr=lr)#, num_epochs=501//4, num_batches_per_epoch=1)


        scaler = GradScaler(init_scale=2.**2)

        if not dataloader_validation:
            dataloader_validation = dataloader
        if not dataloader_test:
            dataloader_test = dataloader

    # Fetch statistics for normalization 
    mean_imgs, std_imgs, mean_imgs_s, std_imgs_s, mean_vols, std_vols = train_statistics
    mla_coordinates = network_settings['mla_coordinates']

    # Compute volume means per dataset
    with torch.no_grad():
        for k,dl in zip(['train','val','test'],[dataloader, dataloader_validation, dataloader_test]):
            if not dl:
                continue
            n_datasets = len(dl.dataset.datasets)
            input_views = torch.rand([1]+list(dataloader.dataset.datasets[0].stacked_views[0].unsqueeze(0).shape))
            input_views = XLFMDatasetFull.extract_views(input_views, mla_coordinates, network_settings['vol_shape'], debug=False).to(device) 
            for ds in dl.dataset.datasets:
                # Check if the volumes exist already
                if len(ds.gt_cache) == 0:
                    available_volume_cache = []
                    # Try to load from disk
                    if mean_volumes_cache_path is not None:
                        available_volume_cache = glob.glob(f'{mean_volumes_cache_path}/mean_vol_*ds_{ds.dataset_id}_{k}')
                    if len(available_volume_cache):
                        ds.gt_cache = [vol.to(device) for vol in torch.load(available_volume_cache[0])['mean_vol_gt_cache']]
                    else: # Compute them
                        if k=='test':
                            ds.mean_vols_stack = ds.vols[0].unsqueeze(0)
                        else:
                            ds.mean_vols_stack = ds.vols.mean(0).unsqueeze(0)
                        gt_volume = ((ds.mean_vols_stack-mean_vols)/std_vols).to(device)

                        # Generate low resolution gt volumes by running the NF forward
                        with autocast(enabled=args.use_half_precision==1):
                            # Add some noise to avoid problems with ActNorm
                            gt_volume += torch.normal(0, 1e-3, gt_volume.size(), device=device) 
                            # Run INN forward, generating all low res volumes and Z's
                            _, ds.gt_cache, _,_ = evaluate_INN_forward(conv_inn, cond_nets, args, args_nets, gt_volume, input_views, train_statistics, extra_cond_in=None)
                            # ds.gt_cache = [gt[0,...] for gt in ds.gt_cache]
                            ds.gt_cache = [gt[0,::2,...]-gt[0,1::2,...] for gt in ds.gt_cache]

    # Use the mean from the training set
    if False:#dataloader_validation:
        for ix,ds in enumerate(dataloader_validation.dataset.datasets):
            if args.evaluation_dataset == 'train':
                ds.gt_cache = dataloader.dataset.datasets[ix].gt_cache
            elif args.evaluation_dataset == 'test':
                ds.gt_cache = dataloader_test.dataset.datasets[ix].gt_cache

    with torch.cuda.device(device):
        torch.cuda.empty_cache()
        
    # Storage for projections, times, certainities, likelehoods, etc
    neural_activity_dataframe = {}
    results_storage_template = {
                        'projections_gt'         : [],
                        F'projections_diff'      : [],
                        F'projections_predicted' : [],
                        F'psnr'                  : [],
                        F'MAPE'                  : [],
                        F'CC'                    : [],
                        'times'                  : [],
                        'log-likelihoods'        : [],
                        'prior-errors'           : []
                        }
    results_storage_template = {'train' : copy.deepcopy(results_storage_template), 'val' : copy.deepcopy(results_storage_template), 'test'  : copy.deepcopy(results_storage_template)}
    # Create timers
    starter, ender = torch.cuda.Event(enable_timing=True),          torch.cuda.Event(enable_timing=True)
    # Find neural coordinates through std
    neural_coords_dic = { tag : [read_neural_coordinates_from_file(ds_path) for ds_path in neural_coordinates_filename[tag]] for tag in neural_coordinates_filename.keys()}

    add_gt_after = [3]
    steps_to_optimize = []
    curr_condition = None
    global_it = -1 if start_epoch==0 else start_epoch*len(dataloader)
    gt_cache_global = {'train':{}, 'val':{}, 'test' :{}}
    
    epoch = start_epoch-1 if len(args.fine_tune_optimize_steps)>0 and args.fine_tune else args.epochs -1
    optimizer = None
    optimizer_cond = None
  
    upsampled_cache = len(dataloader) * [None]
    
    # Create storage for volumes
    vol_shape = [network_settings['vol_shape'][-1]] + network_settings['vol_shape'][0:2]
    all_volumes_gt = {'val' : [False, torch.zeros([len(dataloader_validation)] + vol_shape) if dataloader_validation is not None else None], 
        'train' : [False, torch.zeros([len(dataloader)] + vol_shape) if dataloader is not None else None], 
        'test' : [False, torch.zeros([len(dataloader_test)] + vol_shape) if dataloader_test is not None else None]}
    all_volumes_pred = { k : [False, v[1].clone() if v[1] is not None else None] for k,v in all_volumes_gt.items()}
    results_storage = copy.deepcopy(results_storage_template)
    is_eval_train_mode = False
    is_val_mode = False
    is_test_mode = False
    perform_evaluation = False
    
    while epoch <= args.epochs:
        # Set current stage tag
        stage_tag = 'train'
        # Pick dataloader
        curr_dataloader = dataloader

        if is_test_mode:
            is_test_mode = False
            perform_evaluation = False
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        if is_val_mode:
            is_val_mode = False
            is_test_mode = True
            stage_tag = 'test'
            curr_dataloader = dataloader_test
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        if is_eval_train_mode:
            is_eval_train_mode = False
            is_val_mode = True
            stage_tag = 'val'
            curr_dataloader = dataloader_validation
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

        if ((epoch+1)%(eval_every)==0 or epoch+1==args.epochs) and not perform_evaluation:
            results_storage = copy.deepcopy(results_storage_template)
            corr_coeff_all_ds = {tag : [] for tag in ['train','test','val']}
            is_eval_train_mode = True
            perform_evaluation = True
            epoch += 1

        if curr_dataloader is None:
            continue
        # Reset cache for training, store the last iteration of the current step, to use as input for the next step
        capture_cache = False
        if (epoch+1) % epochs_per_step==0 and len(steps_to_optimize)>0 and steps_to_optimize[0] > 0:
            capture_cache = True
            upsampled_cache = len(dataloader) * [None]
        if args.fine_tune and epoch%epochs_per_step==0 and (epoch+1)<(args.epochs):
            steps_to_optimize = [int(args.INN_max_down_steps-epoch/epochs_per_step)-1]
            optimizer = optimizers[steps_to_optimize[0]]
            if steps_to_optimize[0] < len(optimizers)-1:
                optimizer_cond = optimizers_cond[steps_to_optimize[0]]
            norm_factor = None
            writer.add_scalar('step_to_optimize', steps_to_optimize[0], global_it)
            # Delete previous optimizers
            if epoch!=0:
                n_net = steps_to_optimize[0]+1
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
                try:
                    del optimizers[n_net]
                    conv_inn[n_net].eval()
                except:
                    pass
                cond_nets[n_net].train()
                if n_net not in args.fine_tune_optimize_steps and n_net > 0:
                    epoch += (epochs_per_step-1)
                    continue

        
        if args.fine_tune:  
            print(40*'#' + F" epoch: {epoch+1}/{args.epochs} {stage_tag}")  
            if is_val_mode:
                print(output_path)   
        

        # Do we need to go through all the steps?
        steps_to_reconstruct = range(len(conv_inn) - force_last_step_NF,-1,-1)

        # Reset volumes
        for k in all_volumes_gt.keys():
            all_volumes_gt[k][0] = False
            all_volumes_pred[k][0] = False
        
######## Iterate samples
        for ix,(raw_views, gt_volume_,_, mean_vols_cache) in enumerate(tqdm(curr_dataloader, desc='Optimizing images')): 
            if stage_tag == 'train' and not perform_evaluation:
                global_it += raw_views.shape[0]


            views = raw_views.to(device)
            # Compute Condition B: Crop the 29 lenslets
            input_views = XLFMDatasetFull.extract_views(views, mla_coordinates, network_settings['vol_shape'], debug=False).to(device) 
            cond_input = (input_views-mean_imgs)/std_imgs
            # if args.add_noise==1 and stage_tag=='train':
            #    cond_input += torch.normal(0, 0.05, cond_input.size(), device=device) 
            
            ####################### Compute low resolution GTs and cache
            torch.set_grad_enabled(False)
            # is this volume already computed? as it might be a low resolution if its the second or another step
            if (ix in gt_cache_global[stage_tag]):
                gt_cache = [v.to(device) for v in gt_cache_global[stage_tag][ix]]
            else:

                gt_volume = gt_volume_.to(device)
                # Normalize volume and image
                gt_volume = (gt_volume-mean_vols)/std_vols

                # Generate low resolution gt volumes by running the NF forward
                with autocast(enabled=args.use_half_precision==1):
                    gt_cache = args.INN_max_down_steps * [None]
                    # Add some noise to avoid problems with ActNorm
                    gt_volume += torch.normal(0, 1e-3, gt_volume.size(), device=device) 
                    gt_cache[0] = gt_volume 
                    loss_cond = 0
                
                    # Run INN forward, generating all low res volumes and Z's
                    losses, gt_cache, prior_errors,_ = evaluate_INN_forward(conv_inn, cond_nets, args, args_nets, gt_volume, input_views, train_statistics)
                    
                gt_cache_global[stage_tag][ix] = [v.detach().cpu() for v in gt_cache]

                if any([torch.isinf(L) or torch.isnan(L) for L in losses]):
                    losses = [torch.tensor([1e15],device=device) if torch.isinf(L) or torch.isnan(L) else L for L in losses]
                    prior_errors = [torch.tensor([1e15],device=device) if torch.isnan(L) else L for L in prior_errors]
                    print('Inf found')
                # Store GT piramid
                # results_storage[stage_tag]['projections_gt'].append(projections_piramid) #create_image_piramid(projections_piramid[::-1])
            torch.set_grad_enabled(args.fine_tune==1 and perform_evaluation==0)
            
            stored_volumes = args.INN_max_down_steps*[None]
            
            projections = []
            log_jacobians = []
            if args.fine_tune and optimizer:
                optimizer.zero_grad()
                
            start_time = time.time()
            curr_time = 0
            upsampled_vol = 0


            with autocast(enabled=args.use_half_precision==1):
        ####################### Compute all previous downsamplig steps, 
            
                if not perform_evaluation and stage_tag == 'train':
                    if upsampled_cache[ix] is not None:
                        upsampled_vol = upsampled_cache[ix].to(device)
                        steps_to_reconstruct = steps_to_optimize
                    if optimizer:
                        optimizer.zero_grad()
                    if optimizer_cond is not None:
                        optimizer_cond.zero_grad()

                if not perform_evaluation:
                    try:
                        cond_nets[steps_to_optimize[0]].train()
                    except:
                        pass

                full_loss = 0
                LL_loss = 0
                for n_net in steps_to_reconstruct:
                    is_last_step = n_net==(args.INN_max_down_steps-1)
                    # Use low resolution GT as input?
                    if not is_last_step and ((args.train_with_gt_low_res==1 and not perform_evaluation) or (args.train_with_gt_low_res==2 and n_net in add_gt_after )): 
                        upsampled_vol = gt_cache[n_net+1].clone()
                    
                    # We don't want to compute all the steps if we are calibrating one by one
                    if args.fine_tune and not perform_evaluation and n_net<np.max(steps_to_optimize):
                        continue
                    
                    
                    starter.record()
                
                    # Lowest resolution step, just run the cond_net to generate lowest resolution volume
                    if is_last_step and not force_last_step_NF:
                        # with profiler.profile(with_stack=True, profile_memory=True) as prof:
                        views_input = cond_input + (torch.normal(0, 0.5, cond_input.size(), device=device) if args.add_noise==1 and stage_tag=='train' else 0)
                        upsampled_vol = cond_nets[n_net](views_input, mean_vols_cache[n_net-1])[-1]
                        if args.use_half_precision == 0:
                            upsampled_vol = upsampled_vol.float()
                        log_jac = torch.zeros([1], device=device) 
                        # prof.export_chrome_trace(F'{output_path}_profiler_trace.json')
                        # print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=10))
                        # sys.exit(0)
                    else:
                    # Other upsampling NFs using condition and NF
                        # condition 1 is processed LF views
                        if force_all_steps_NF:
                            cond_processed = [torch.zeros([gt_cache[n_net].shape[0], gt_cache[n_net].shape[1],gt_cache[n_net].shape[2]//2,gt_cache[n_net].shape[3]//2], device=device)]
                        else:
                            cond_processed = [cond_nets[n_net](cond_input)[-1].float()]
                        
                        if not disable_low_res_input:
                            # condition 2 is the low resolution volume as well
                            cond_processed.append(mean_vols_cache[n_net])
                        else:
                            cond_processed = [upsampled_vol]

                        # Sample multiple times?
                        n_samples = args.INN_n_samples if args.batch_size == 1 else 1
                        # input 1 is Z gaussian distributed
                        inputs = [sample_z_truncated((n_samples,)+tuple(conv_inn[n_net].global_out_shapes[0]), 
                                    device=device, temperature=args.INN_z_temperature)]
                        if not is_last_step: # In case that we are forcing a NF on the last step
                            # input 2 is the low resolution volume n_step+1
                            inputs.append(upsampled_vol.repeat(n_samples,1,1,1))
                        # Compute reconstruction for current step
                        upsampled_vol, log_jac = conv_inn[n_net](inputs, c=[cc.repeat(n_samples,1,1,1) for cc in cond_processed], rev=True)
                        if args.batch_size == 1:
                            upsampled_vol = upsampled_vol.mean(0).unsqueeze(0)
                        

                    # Lets store this in the cache
                    if args.fine_tune and stage_tag == 'train' and capture_cache:
                        upsampled_cache[ix] = upsampled_vol.detach().cpu()
                    
                    ender.record()
                    torch.cuda.current_stream().synchronize()
                    t = (starter.elapsed_time(ender)/1000) / gt_volume.shape[0]
                    curr_time += t
                    # print(f'Step{n_net} time: {t}')

                    # Do we train?
                    if args.fine_tune and n_net in steps_to_optimize and epoch>0:
                        # Fetch GT for forward pass
                        curr_gt = gt_cache[n_net].clone()
                        # Add gaussian noise?
                        # if args.add_noise and args.fine_tune and not perform_evaluation and n_net in steps_to_optimize:
                        #     curr_gt += torch.normal(0, 1/np.max([2*n_net,1]), gt_cache[n_net].size(), device=device) 
                        
                        if is_last_step:
                            # We normalize as the intensity of the volumes double as they shrink due to Haar transform
                            if args.loss_func_first_step == 'L1':
                                loss_cond = F.l1_loss(curr_gt, upsampled_vol)
                            elif args.loss_func_first_step == 'L2':
                                loss_cond = F.mse_loss(curr_gt, upsampled_vol) 
                            elif args.loss_func_first_step == 'wL2':
                                loss_cond = Losses.weighted_mse_loss(curr_gt, upsampled_vol)
                            elif args.loss_func_first_step == 'LL':
                                loss_cond = ((upsampled_vol-upsampled_vol.min()) - (curr_gt-curr_gt.min()) * torch.log(1e-8+upsampled_vol-upsampled_vol.min())).mean()
                            # loss /= np.max([2*n_net,1])
                            full_loss += loss_cond
########################### Optimize Log Likelihood
                        else:
                            if not perform_evaluation:
                                if args.loss_func_reg == 'L1':
                                    loss_cond = F.l1_loss(curr_gt, upsampled_vol)
                                elif args.loss_func_reg == 'L2':
                                    loss_cond = F.mse_loss(curr_gt, upsampled_vol) 
                                elif args.loss_func_reg == 'wL2':
                                    loss_cond = Losses.weighted_mse_loss(curr_gt, upsampled_vol)
                                full_loss += loss_cond * args.INN_cond_weight
                            if True: #else:
                                # If training we reset the ActNorm of the current step before starting training
                                if norm_factor==None and stage_tag=='train':
                                    # conv_inn[n_net],_ = reset_ActNorm(conv_inn[n_net])
                                    # Reset attention from condition
                                    # cond_nets[n_net].global_attention.reset()
                                    norm_factor = 1
                                    
                                # Run NF backwards
                                Z, log_jac_det = conv_inn[n_net](curr_gt, c=cond_processed)
                                Z_current = Z[0]

                                # Compute loss
                                error_on_prior = torch.norm(Z_current)**2 
                                ## Check for instabilities
                                if torch.isinf(error_on_prior):
                                    print(F'Inf on error_on_prior, step {n_net+1}')
                                    if torch.isinf(Z_current.max()) or torch.isneginf(Z[0].min()):
                                        print(F'Inf on Z, step {n_net+1}')
                                    raise Exception()

                                curr_LL_loss = (0.5*error_on_prior - log_jac_det.mean()) / upsampled_vol.numel()
                                # loss = curr_LL_loss
                                
                                LL_loss += curr_LL_loss.item()

                                if args.save_images and perform_evaluation:
                                #     # store conditions for display
                                    curr_condition = cond_processed[0].detach().clone() #todo: clone needed?
                                full_loss += curr_LL_loss * (1 - args.INN_cond_weight)

                        if stage_tag == 'train':
                            if torch.isnan(full_loss):
                                writer.add_scalar(F"fine_tune/psnr/{stage_tag}/step_{n_step}", 0, global_it)
                                writer.add_scalar('corr_coeff_mean_train/pred',0, global_it)
                                writer.add_scalar('corr_coeff_mean_test/pred',0, global_it)
                                writer.add_scalar('corr_coeff_mean_val/pred',0, global_it)
                                raise ValueError(f'Nan loss found in {stage_tag}')
                        
                            
                    # Store volumes for display and evaluation
                    if perform_evaluation:
                        stored_volumes[n_net] = upsampled_vol.float().detach()

####################### Optimize
                    if not perform_evaluation and n_net in steps_to_optimize and args.fine_tune and epoch>0:
                        half_possible = True
                        try:
                            scaler.scale(full_loss).backward()
                        except:
                            continue
                            half_possible = False
                        try:
                            if half_possible:
                                if optimizer_cond:#epoch % 2 == 0:
                                    scaler.step(optimizer_cond)
                                if True:#else:
                                    scaler.step(optimizer)
                                scaler.update()
                            else:
                                if optimizer_cond:
                                    optimizer_cond.step()
                                else:
                                    optimizer.step()
                            # scheduler.step()
                            upsampled_vol = upsampled_vol.detach()
                        except:
                            if optimizer_cond:#epoch % 2 == 0:
                                optimizer_cond.step()
                            if True:#else:
                                optimizer.step()

            # Store runtime
            results_storage[stage_tag]['times'].append(curr_time)

################# Compute evaluation metrics
            if perform_evaluation:
                # Fetch GT volume
                vol_out = gt_cache[0][0]*std_vols + mean_vols
                vol_out -= vol_out.min()
                
                all_volumes_gt[stage_tag][1][ix] = vol_out.detach().cpu()
                all_volumes_gt[stage_tag][0] = True

                vol_out_pred = (stored_volumes[0][0] * 2**len(stored_volumes[0]))*std_vols + mean_vols
                # vol_out -= vol_out.min()

                all_volumes_pred[stage_tag][1][ix] = vol_out_pred.detach().cpu()

                # Store if is last epoch
                if args.save_tiff_volumes and (epoch<=1 or epoch>=args.epochs-1 or (epoch+1)%epochs_per_step==0):
                    try:
                        os.makedirs(F'{output_path}/stacks/gt/')
                        os.makedirs(F'{output_path}/stacks/pred/')
                    except:
                        pass
                    
                    imsave(F'{output_path}/stacks/gt/stack_{"{:03d}".format(ix)}.tif', F.relu(vol_out).detach().cpu().numpy())
                    imsave(F'{output_path}/stacks/pred/stack_{"{:03d}".format(ix)}.tif', F.relu(vol_out_pred).detach().cpu().numpy())

                
                # MAPE and PSNR
                curr_psnrs = []
                curr_masked_psnrs = []
                with autocast(enabled=args.use_half_precision==1):          
                    projections = []
                    projections_diff = []
                    projections_gt = []
                    for n_step,v in enumerate(stored_volumes):
                        # Normalize with respect to GT, so we have range max 1
                        gt_volume = gt_cache[n_step].float().clone()
                        
                        curr_recon = v.float().clone()
                        if args.save_images and args.create_dist_plots:
                            my_plot = plot_distributions(gt_volume.cpu().detach().numpy(), curr_recon.cpu().detach().numpy())
                            writer.add_figure(f'posterior/{stage_tag}/step{n_step}', my_plot, global_it)
                        curr_psnr, masked_psnr, gt_volume, curr_recon = compute_INN_step_performance(gt_volume_in=gt_volume, pred_volume_in=curr_recon, step=n_step, mean=mean_vols, std=std_vols)
                        
                        # Compute psnr
                        curr_psnrs.append(curr_psnr)
                        # Masked psnr (only measure where one of both gt and prediciton is non-zero)
                        curr_masked_psnrs.append(masked_psnr)

                        if ix==0 or args.fine_tune==0 or epoch>=args.epochs-1:
                            projections_diff.append(volume_2_projections((curr_recon-gt_volume).abs().permute(0,2,3,1).unsqueeze(1))[0,0,...].float().detach().cpu().numpy())
                            # Store reconstruction projection
                            projections.append(volume_2_projections(curr_recon.clone().permute(0,2,3,1).unsqueeze(1), add_scale_bars=False)[0,0,...].float().detach().cpu().numpy())
                            # Store gt projection
                            projections_gt.append(volume_2_projections((gt_volume).permute(0,2,3,1).unsqueeze(1), add_scale_bars=False)[0,0,...].float().detach().cpu().numpy())
            
                    # Store projections and statistics
                    results_storage[stage_tag][F'psnr'].append(curr_psnrs)
                    results_storage[stage_tag][F'MAPE'].append(curr_masked_psnrs)
                    
                    results_storage[stage_tag][F'projections_predicted'].append(projections)
                    results_storage[stage_tag][F'projections_gt'].append(projections_gt)
                    results_storage[stage_tag][F'projections_diff'].append(projections_diff)

        if perform_evaluation and all_volumes_gt[stage_tag][0]:
            # Do we have multiple datasets? 
            # Lets compute the cc individually per dataset
            neur_act_all_df = pd.DataFrame()
            n_datasets = len(neural_coords_dic[stage_tag])
            n_vols = len(all_volumes_pred[stage_tag][1])
            vols_per_ds = n_vols // n_datasets
            for ix,ds in enumerate(range(n_datasets)):
                cc, neur_act_df = corr_coeff_3D(all_volumes_gt[stage_tag][1][ix*vols_per_ds:(ix+1)*vols_per_ds], 
                                                        all_volumes_pred[stage_tag][1][ix*vols_per_ds:(ix+1)*vols_per_ds], 
                                                        neural_coords_dic[stage_tag][ix], 
                                                        5, 3, n_time_steps=None,
                                                        output_path=f'{output_path}/NN3D_{stage_tag}_ds{ix}_{epoch}e.pdf' if args.save_images else None,
                                                        filter_width=args.neural_activation_filter_width)
                
                # Add column with data_path 
                neur_act_df = neur_act_df.assign(sample_id=curr_dataloader.dataset.datasets[ix].dataset_id)
                # Accumulate in singl dataset
                neur_act_all_df = pd.concat([neur_act_all_df, neur_act_df])
                corr_coeff_all_ds[stage_tag].append(np.mean(cc))
                if args.save_images:
                    writer.add_figure(f'corr_coeff/CC3D_{stage_tag}_ds{ix}', plt.gcf(), global_it)
            neural_activity_dataframe[stage_tag] = neur_act_all_df
            
            if args.fine_tune:
                writer.add_scalar('corr_coeff_mean_'+ stage_tag +'/pred', np.mean(corr_coeff_all_ds[stage_tag]), global_it)
        if args.fine_tune:
            print(F'Full Loss step-{steps_to_optimize[0]}: {full_loss}')
            if not perform_evaluation or is_val_mode or is_test_mode:
                # if full_loss != 0:
                #     
                if full_loss != 0:
                    writer.add_scalar(f"fine_tune/loss/{stage_tag}", full_loss, global_it)
                    if True:#epoch % 2 == 0:
                        writer.add_scalar(f"fine_tune/loss_LL/{stage_tag}", LL_loss, global_it)
                    if True:#else:
                        writer.add_scalar(f"fine_tune/loss_cond/{stage_tag}", loss_cond, global_it)
                writer.add_scalar(f"epochs/{stage_tag}", epoch, global_it)
                writer.add_scalar('fine_tune/learning_rate', args.learning_rate)
            if perform_evaluation:
                
                # Store psnr and MAPE of current run.
                n_images = len(results_storage[stage_tag][F'psnr'])
                for n_step in range(len(results_storage[stage_tag][F'psnr'][0])):
                    curr_computation =np.mean([results_storage[stage_tag][F'psnr'][n_img][n_step] for n_img in range(n_images)])
                    writer.add_scalar(F"fine_tune/psnr/{stage_tag}/step_{n_step}", curr_computation, global_it)
                    curr_computation =np.mean([results_storage[stage_tag][F'MAPE'][n_img][n_step] for n_img in range(n_images)])
                    writer.add_scalar(F"fine_tune/masked_psnr/{stage_tag}/step_{n_step}", curr_computation, global_it)
                
                if curr_condition is not None:
                        imshow3D(curr_condition, color_map=cm, add_scale_bars=True)
                        writer.add_figure(f'condition/{stage_tag}_step{steps_to_optimize[0]}', plt.gcf(), global_it)

                add_projection_borders = True
                if args.save_images: 
                    for n_step in steps_to_optimize:
                        imshow3D(stored_volumes[n_step], color_map=cm, add_scale_bars=add_projection_borders)
                        writer.add_figure(f'fine_tune/recon_{stage_tag}_step{n_step}', plt.gcf(), global_it)
                        imshow3D(gt_cache[n_step].float().clone(), color_map=cm, add_scale_bars=add_projection_borders)
                        writer.add_figure(f'fine_tune/GT_{stage_tag}_step{n_step}', plt.gcf(), global_it)
                        
                    # writer.add_scalar('gpu_usage', get_current_gpu_usage(device), global_it)

                n_images = 1
                if n_images > 0:
                    mip_size = results_storage[stage_tag]['projections_gt'][0][0].shape
                    all_projections_gt = np.concatenate([np.array(results_storage[stage_tag]['projections_gt'][i][0]).reshape(1,mip_size[0],mip_size[1]) for i in range(n_images)], axis=0).astype(np.float16)
                    all_projections_prediction = np.concatenate([np.array(results_storage[stage_tag]['projections_predicted'][i][0]).reshape(1,mip_size[0],mip_size[1]) for i in range(n_images)], axis=0).astype(np.float16)

                    if args.save_images:
                        writer.add_image('projections_gt/'+ stage_tag, tv.utils.make_grid(volume_2_projections(torch.from_numpy(all_projections_gt).unsqueeze(0), add_scale_bars=False, depths_in_ch=True)[0,0,...].float().unsqueeze(0).cpu().data.detach(), normalize=True, scale_each=True), global_it)
                    writer.add_image('projections_pred/'+ stage_tag, tv.utils.make_grid(volume_2_projections(torch.from_numpy(all_projections_prediction).unsqueeze(0), add_scale_bars=False, depths_in_ch=True)[0,0,...].float().unsqueeze(0).cpu().data.detach(), normalize=True, scale_each=True), global_it)
                    if epoch>=args.epochs-1:
                        save_image(torch.from_numpy(all_projections_gt).unsqueeze(0),f'output_gt_{stage_tag}.png')
                        save_image(torch.from_numpy(all_projections_prediction).unsqueeze(0),f'output_pred_{stage_tag}.png')
        
        if args.fine_tune and epoch%epochs_per_step==0:
            for step in list(range(args.INN_max_down_steps)):
                    serialize_INN_step(conv_inn[step] if step<args.INN_max_down_steps-1 else None, 
                                    cond_nets[step], None, train_statistics, args_nets[step], epoch , output_path)
        # prof.export_chrome_trace(F'{output_path}_profiler_trace.json')
        # print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=10))

        if not perform_evaluation:
            epoch += 1

#######################  Compute results
    stage_tag = 'train'
    n_images = len(results_storage[stage_tag][F'psnr'])
    print('\n' + 40*'#' + '  Results  ' + 40*'#')

    print(40*'#'+40*'#')
    print(40*'-' + F'  Per Layer  ' + 40*'-')
    print('metric',end='\t\t')
    for n_step in range(len(results_storage[stage_tag][F'psnr'][0])):
        print(n_step+1, end='\t')

    for metric in ['psnr','MAPE']:
        print(f'\nMean {metric} ',end='\t')
        for n_step in range(len(results_storage[stage_tag][metric][0])):
            curr_computation =np.mean([results_storage[stage_tag][metric][n_img][n_step] for n_img in range(n_images)])
            print('{:.3f}'.format(curr_computation), end='\t')
            writer.add_scalar(F'{metric}/step_{n_step}', curr_computation)
     
    results_storage[stage_tag]['CC'] = np.mean(corr_coeff_all_ds['train'])
    # Print timings
    print('\n\n\t Mean CC: \t\t{:.4f}'.format(results_storage[stage_tag]['CC']))
    print('\t Mean runtime: \t\t{:.4f}'.format(np.mean(results_storage[stage_tag]['times'])))
    print('\t Min runtime: \t\t{:.4f}'.format(np.min(results_storage[stage_tag]['times'])))



    ## Calculate final coorrelation coefficient
    for tag,v in corr_coeff_all_ds.items():
        writer.add_scalar(F'corr_coeff_mean/{tag}', np.mean(v) if len(v)>0 else 0)
    writer.add_scalar('time/mean', np.mean(results_storage[stage_tag]['times']))
    writer.add_scalar('time/min', np.min(results_storage[stage_tag]['times']))
    


    # Draw outputs
    # For now only the last output
    norm = np.max
    if args.save_images:
        for img_ix in tqdm(range(np.min([10,len(results_storage[stage_tag]['projections_predicted'])])), "Saving images..."): #len(results_storage[stage_tag]['projections_predicted']))
            images_predicted = results_storage[stage_tag]['projections_predicted'][img_ix]
            images_gt = results_storage[stage_tag]['projections_gt'][img_ix]
            if len(images_predicted) == 0:
                continue

            img_out_predicted = create_image_piramid(images_predicted, norm)
            img_out_gt = create_image_piramid(images_gt, norm)

            # Save individual HD images
            plt.clf() ; plt.figure(figsize=(5,5)) ; plt.imshow(images_predicted[0]/images_predicted[0].max(), cmap=cm) ; plt.axis('off') ; plt.tight_layout() ; plt.rcParams['axes.xmargin'] = 0 ; plt.rcParams['axes.ymargin'] = 0 ; plt.savefig(F'{output_path}/_output_image_pred{img_ix}.png')
            plt.clf() ; plt.figure(figsize=(5,5)) ; plt.imshow(images_gt[0]/images_gt[0].max(), cmap=cm) ; plt.axis('off') ; plt.tight_layout() ; plt.rcParams['axes.xmargin'] = 0 ; plt.rcParams['axes.ymargin'] = 0 ; plt.savefig(F'{output_path}/_output_image_gt{img_ix}.png')
            n_cols = 3

            images_diff = results_storage[stage_tag]['projections_diff'][img_ix]
            img_out_diff = create_image_piramid(images_diff, norm)
            n_rows = 1
            fig1 = plt.figure(figsize=(35,24))
            plt.subplot(n_rows,n_cols,1)
            plt.imshow(img_out_gt, cmap=cm)
            # plt.colorbar()
            plt.title('GT')
            plt.subplot(n_rows,n_cols,2)
            plt.imshow(img_out_predicted, cmap=cm)
            # plt.colorbar()
            plt.title('Prediction')
            plt.subplot(n_rows,n_cols,3)
            plt.imshow(img_out_diff, cmap=cm)
            # plt.colorbar()
            plt.title('Diff')

            plt.tight_layout()
            
            fig1.tight_layout(pad=0)
            fig1.canvas.draw()

            data = np.frombuffer(fig1.canvas.tostring_rgb(), dtype=np.uint8)
            # print(F"Saving image {img_ix+1}/{len(results_storage[stage_tag]['projections_predicted'])}")
            data = data.reshape(fig1.canvas.get_width_height()[::-1] + (3,))
            writer.add_image('Output', data, dataformats='HWC', global_step=img_ix)
            fig1.savefig(F'{output_path}/_output_{output_posfix}_image_{img_ix}.png')
            plt.close(fig=fig1)
        # writer.close()

    # Save tif MIP of reconstructions and GT
    mip_size = results_storage[stage_tag]['projections_gt'][0][0].shape
    n_images = np.sum([len(results_storage[stage_tag]['projections_predicted'][ii])>0 for ii in range(len(results_storage[stage_tag]['projections_predicted']))])
    all_projections_gt = np.concatenate([np.array(results_storage[stage_tag]['projections_gt'][i][0]).reshape(1,mip_size[0],mip_size[1]) for i in range(n_images)], axis=0).astype(np.float16)
    all_projections_prediction = np.concatenate([np.array(results_storage[stage_tag]['projections_predicted'][i][0]).reshape(1,mip_size[0],mip_size[1]) for i in range(n_images)], axis=0).astype(np.float16)
    
    
    
    # plt.savefig(F'{output_path}/Corr_plots_predicition.pdf',dpi=300)
    for k,v in neural_activity_dataframe.items():
        v.to_csv(F'{output_path}/Neural_activity_{k}.csv')

    mean_projections_gt = all_projections_gt.max(axis=0)
    mean_projections_prediction = all_projections_prediction.max(axis=0)

    print(F'Saving directory: {output_path}')
    
    if args.fine_tune:
        conv_inn.append(None)
        for n_step in range(args.INN_max_down_steps):
            serialize_INN_step(conv_inn[n_step], cond_nets[n_step], None, train_statistics, args_nets[n_step], epoch , output_path)
        conv_inn.pop()
    elif args.save_tiff_volumes==1:
        imsave(F'{output_path}/stack_MIP_gt.tif', all_projections_gt)
        imsave(F'{output_path}/stack_MIP_prediction.tif', all_projections_prediction)
    return conv_inn, cond_nets, args_nets, train_statistics, output_path, writer, results_storage['train']
