# 2022/2023 Josue Page Vizcaino pv.josue@gmail.com
"""
This script processes XLFM data by deconvolving the images and saving the resulting volumes to disk.
It takes in several command line arguments, including the path to the data folder, the path to the PSF file, and the indices of the images to use. It also takes in several optional arguments, including the path to a background file, the path to a lenslet file, and the number of iterations to use in the deconvolution process.
"""
import numpy as np
import torch
from torch.utils import data

from torch.utils.data.sampler import SequentialSampler
from datetime import datetime
import os
import sys
import argparse
from tifffile import imsave

from utils import *
from XLFMDataset import *

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', nargs='?', default='XLFM_data/Datasets/GCaMP6s_NLS_1/SLNet_preprocessed/')
parser.add_argument('--psf_file', nargs='?', default= "XLFM_data/PSF_241depths_16bit.tif")
parser.add_argument('--bkg_file', nargs='?', default= "")#XLFM_data/background.tif")
parser.add_argument('--lenslet_file', nargs='?', default= "XLFM_data/lenslet_centers_python.txt")
parser.add_argument('--images_to_use', nargs='+', type=int, default=list(range(0,2)))
parser.add_argument('--n_it', nargs='+', type=int, default=50)

parser.add_argument('--main_gpu', nargs='+', type=int, default=5)
parser.add_argument('--posfix', type=str, default='') 
parser.add_argument('--n_depths', type=int, default=241//2)
parser.add_argument('--vol_xy_size', type=int, default=600)
parser.add_argument('--n_split_fourier', nargs='+', type=int, default=1, help='How to split the OTF for convolution, \
                                                    use 1 to process all at once, but it might not fit in you gpu')
parser.add_argument('--dark_current', type=int, default=0, help='there is no zero, just dark current')

args = parser.parse_args()

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
print(F'Torch version: {torch.__version__}')
print(F'Using {device}')
torch.set_num_threads(8)


# Configure output
output_path = args.data_folder + "/"
stack_path = output_path + '/XLFM_stack_' + datetime.now().strftime('%Y_%m_%d__%H:%M:%S')
os.makedirs(stack_path, exist_ok=True)


# Set volume size
n_depths = args.n_depths
vol_xy_size = args.vol_xy_size
vol_shape = [vol_xy_size, vol_xy_size, n_depths]

# Lateral size of PSF in pixels
psf_size_real = 2160

# Load images
dataset = XLFMDatasetFull(args.data_folder, args.lenslet_file, img_shape=2*[psf_size_real],
        images_to_use=args.images_to_use, load_vols=False)
dataset.stacked_views = dataset.stacked_views.float()
indices = list(range(len(dataset)))
# Create dataloaders
sampler = SequentialSampler(indices)
data_loader = data.DataLoader(dataset, batch_size=1, 
                                sampler=sampler, pin_memory=True, num_workers=0)

# Load PSF and OTF
n_split = args.n_split_fourier
OTF,psf_shape = load_PSF_OTF(args.psf_file, vol_shape, n_split=n_split, compute_OTF=True)

# Load background image
try:
    background = torch.from_numpy(imread(args.bkg_file).mean(axis=0)).unsqueeze(0).unsqueeze(0).to(device)
    background = center_crop(background,[psf_size_real,psf_size_real])
except:
    background = 0

# Save arguments used
with open(stack_path + "/arguments.txt", "w") as file1:
    file1.writelines(str(vars(args)))

with torch.no_grad():
    ####################### Start iterating through the samples in the current dataset
    for ix,(views) in tqdm(enumerate(data_loader), total=len(args.images_to_use), position=0, desc="img", leave=False, colour='green'):
        curr_im_ix = args.images_to_use[ix]
        views = views.to(device)

        # Remove background
        views -= background
        # Deconvolve image
        deconv = XLFMDeconv(OTF, views, args.n_it, ObjSize=2*[vol_xy_size], ROIsize=[vol_xy_size,vol_xy_size,90], n_split_fourier=args.n_split_fourier, max_allowed=10*65535, device=device, all_in_device=args.n_split_fourier==n_depths)
        volume_out = deconv[0]
        # Save volume
        imsave(stack_path + '/XLFM_stack_'+ "%03d" % curr_im_ix + '.tif', volume_out.cpu().numpy())
    # Save a preview of the dataset
    save_image(volume_2_projections(volume_out, depths_in_ch=True),stack_path + '/preview_XLFM_image_stack.png')
    print(F'Output path: {output_path}')