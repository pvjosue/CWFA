import torch
from torch.utils import data
import torch.nn.functional as F
import csv
import glob, os
from PIL import Image
import numpy as np
import re
from tifffile import imread
import sys
import torch
from tqdm import tqdm
import multipagetiff as mtif

def pad_img_to_min(image):
    """
    Pad an image to the minimum size of its dimensions.
    @param image - the image to pad
    @return the padded image
    """
    min_size = min(image.shape[-2:])
    img_pad = [min_size-image.shape[-1], min_size-image.shape[-2]]
    img_pad = [img_pad[0]//2, img_pad[0]//2, img_pad[1],img_pad[1]]
    image = F.pad(image.unsqueeze(0).unsqueeze(0), img_pad)[0,0]
    return image

def center_crop(layer, target_size, pad=0):
    """
    Crop the given layer to the target size by taking the center of the layer.
    @param layer - the layer to crop
    @param target_size - the size of the target crop
    @param pad - the amount of padding to add to the crop
    @return The cropped layer
    """
    _, _, layer_height, layer_width = layer.size()
    diff_y = (layer_height - target_size[0]) // 2
    diff_x = (layer_width - target_size[1]) // 2
    return layer[
        :, :, (diff_y - pad) : (diff_y + target_size[0] - pad), (diff_x - pad) : (diff_x + target_size[1] - pad)
    ]

def get_lenslet_centers(filename):
    """
    Given a filename, read in the data and return the lenslet coordinates.
    @param filename - the name of the file containing the lenslet coordinates
    @return The lenslet coordinates as a tensor.
    """
    x,y = [], []
    with open(filename,'r') as f:
        reader = csv.reader(f,delimiter='\t')
        for row in reader:
            x.append(int(row[0]))
            y.append(int(row[1]))
    lenslet_coords = torch.cat((torch.IntTensor(x).unsqueeze(1),torch.IntTensor(y).unsqueeze(1)),1)
    return lenslet_coords

class XLFMDatasetFull(data.Dataset):
    """
    This is a PyTorch dataset class for loading XLFM data. It loads the image and volume data from the specified paths and provides methods for normalization and data retrieval.
    @param data_path - The path to the data directory
    @param lenslet_coords_path - The path to the lenslet coordinates file
    @param img_shape - The shape of the image data
    @param n_depths_to_fill - The number of depths to fill
    @param border_blanking - The number of pixels to blank at the border
    @param images_to_use - The indices of the images to use
    @param lenslets_offset - The offset of the lenslets
    @param load_vols - Whether to load the volumes
    @param maxWorkers - The maximum number of workers to use
    @param
    """
    def __init__(self, data_path, lenslet_coords_path, img_shape, n_depths_to_fill=120, border_blanking=0, images_to_use=None, lenslets_offset=50,
     load_vols=True, maxWorkers=10, ds_id=''):
        # Load lenslets coordinates
        self.lenslet_coords = get_lenslet_centers(lenslet_coords_path) + torch.tensor(lenslets_offset)
        self.n_lenslets = self.lenslet_coords.shape[0]
        self.data_path = data_path
        self.load_vols = load_vols
        self.vol_type = torch.float16
        self.dataset_id = ds_id
        self.gt_cache = []
        self.last_sample = 0

        self.img_shape = img_shape

        # Tiff images are stored in single tiff stack
        # Volumes are stored in individual tiff stacks
        imgs_path = data_path + '/XLFM_image/XLFM_image_stack.tif'
        imgs_path_sparse = data_path + '/XLFM_image/XLFM_image_stack_S.tif'
        vols_path = data_path + '/XLFM_stack/'

        try:
            self.img_dataset = imread(imgs_path, maxworkers=maxWorkers, key=images_to_use)
        except:
            self.img_dataset = imread(imgs_path, maxworkers=maxWorkers)
            try:
                self.img_dataset = self.img_dataset[images_to_use]
            except:
                self.img_dataset = self.img_dataset[:len(images_to_use)]
                images_to_use = list(range(len(images_to_use)))

        # Lets clamp the images in case of infinites or large numbers
        self.img_dataset = np.nan_to_num(self.img_dataset)
        self.img_dataset = np.clip(self.img_dataset, 0, 50000)
        self.img_dataset[np.isinf(self.img_dataset)] = 50000

        n_frames,h,w = np.shape(self.img_dataset)

        if images_to_use is None:
            images_to_use = list(range(n_frames))
        self.n_images = min(len(images_to_use), n_frames)

        self.all_files = sorted(glob.glob(vols_path))
        if len(self.all_files)>0 and load_vols:
            # If more images where requested than available
            # if max(images_to_use) > len(self.all_files):
            #     # if more images requested than available use all the images
            #     if len(images_to_use) > len(self.all_files):
            #         images_to_use = list(range(len(self.all_files)))
            #     # if there are enough images, then only resample all the images
            #     else:
            #         images_to_use = list(range(0,len(self.all_files), len(self.all_files)//len(images_to_use)))[:len(images_to_use)]
            self.all_files = sorted([glob.glob(f'{vols_path}*{images_to_use[i]:03d}.tif')[0] for i in range(self.n_images)])

        if load_vols:
            # read single volume
            currVol = self.read_tiff_stack(self.all_files[0])
            odd_size = [int(currVol.shape[0]),int(currVol.shape[1])]
            half_volume_shape = [odd_size[0]//2,odd_size[1]//2]
            self.volStart = [currVol.shape[0]//2-half_volume_shape[0], currVol.shape[1]//2-half_volume_shape[1]]
            self.volEnd = [odd_size[n] + self.volStart[n] for n in range(len(self.volStart))]
            self.vols = torch.zeros(self.n_images, n_depths_to_fill, odd_size[0], odd_size[1], dtype=self.vol_type)
            
        else:
            odd_size = self.img_shape
            self.vols = 255*torch.ones(1)
        
        # Create storage
        self.stacked_views = torch.zeros(self.n_images, self.img_shape[0], self.img_shape[1],dtype=torch.float32)
    
        for nImg in tqdm (range(self.n_images), desc="Loading Img..."):
            if load_vols:
                currVol = self.read_tiff_stack(self.all_files[nImg])
                currVol[torch.isinf(currVol)] = 0
                assert not torch.isinf(currVol).any()
                if border_blanking>0:
                    currVol[:border_blanking,...] = 0
                    currVol[-border_blanking:,...] = 0
                    currVol[:,:border_blanking,...] = 0
                    currVol[:,-border_blanking:,...] = 0
                    currVol[:,:,:border_blanking] = 0
                    currVol[:,:,-border_blanking:] = 0
                # Do we need less depths?
                depths_to_use = min(n_depths_to_fill, self.vols.shape[1])
                depths_to_use = list(range(currVol.shape[2]//2-depths_to_use//2,currVol.shape[2]//2+depths_to_use//2))
                # Store the volume
                self.vols[nImg,:currVol.shape[2],:,:] = currVol.permute(2,0,1)\
                    [depths_to_use,self.volStart[0]:self.volEnd[0],self.volStart[1]:self.volEnd[1]]

            # Load image
            image = torch.from_numpy(np.array(self.img_dataset[nImg,:,:]).astype(np.float16)).type(torch.float32)
            image = pad_img_to_min(image)
            self.stacked_views[nImg,...] = center_crop(image.unsqueeze(0).unsqueeze(0), self.img_shape)[0,0,...]
                
            
        if self.img_dataset is not None:
            del self.img_dataset
        print('Loaded ' + str(self.n_images))  

    def __len__(self):
        'Denotes the total number of samples'
        return int(self.n_images)

    def get_n_depths(self):
        return self.vols.shape[1]

    def get_max(self):
        'Get max intensity from volumes and images for normalization'
        return self.stacked_views.float().max().type(self.stacked_views.type()),\
            self.stacked_views.float().max().type(self.stacked_views.type()),\
            self.vols.float().max().type(self.vols.type())

    def get_statistics(self):
        'Get mean and standard deviation from volumes and images for normalization'
        return  self.stacked_views.float().mean().type(self.stacked_views.type()), self.stacked_views.float().std().type(self.stacked_views.type()), \
                self.vols.float().mean().type(self.vols.type()), self.vols.float().std().type(self.vols.type())

    def standarize(self, stats):
        mean_imgs, std_imgs, mean_imgs_s, std_imgs_s, mean_vols, std_vols = stats
        self.stacked_views[...] = self.standarize_sample(self.stacked_views[...], mean_imgs, std_imgs)
        self.vols = self.standarize_sample(self.vols, mean_vols, std_vols)

    @staticmethod
    def standarize_sample(sample, mean, std):
        return (sample-mean) / std
    def len_lenslets(self):
        'Denotes the total number of lenslets'
        return self.n_lenslets
    def get_lenslets_coords(self):
        'Returns the 2D coordinates of the lenslets'
        return self.lenslet_coords
    

    def __getitem__(self, index):
        views_out = self.stacked_views[[index],...]
        if self.load_vols is False:
            return views_out,0,0,0
        vol_out = self.vols[index,...]
        
        return views_out,vol_out, index, self.gt_cache


    @staticmethod
    def extract_views(image, lenslet_coords, subimage_shape, debug=False):
        """
        Given an image, lenslet coordinates, and a subimage shape, extract views of the image
        centered around each lenslet. 
        @param image - the input image
        @param lenslet_coords - the coordinates of the lenslets
        @param subimage_shape - the shape of the subimages to extract
        @param debug - whether to output debug information
        @return stacked_views - the extracted views of the image
        """
        # print(str(image.shape))
        half_subimg_shape = [subimage_shape[0]//2,subimage_shape[1]//2]
        n_lenslets = len(lenslet_coords)
        stacked_views = torch.zeros(size=[image.shape[0], n_lenslets, subimage_shape[0], subimage_shape[1]], device=image.device, dtype=image.dtype)
        
        if debug:
            debug_image = image.detach().clone()
            max_img = image.float().cpu().max()
        for nLens in range(n_lenslets):
            # Fetch coordinates
            currCoords = lenslet_coords[nLens]
            if debug:
                debug_image[:,:,currCoords[0]-2:currCoords[0]+2,currCoords[1]-2:currCoords[1]+2] = max_img
            # Grab patches
            lower_bounds = [currCoords[0]-half_subimg_shape[0], currCoords[1]-half_subimg_shape[1]]
            lower_bounds = [max(lower_bounds[kk],0) for kk in range(2)]
            currPatch = image[:,0,lower_bounds[0] : currCoords[0]+half_subimg_shape[0], lower_bounds[1] : currCoords[1]+half_subimg_shape[1]]
            stacked_views[:,nLens,-currPatch.shape[1]:,-currPatch.shape[2]:] = currPatch
        
        return stacked_views
    
    @staticmethod
    def read_tiff_stack(filename, out_datatype=torch.float16):
        tiffarray = mtif.read_stack(filename, units='voxels')
        return torch.from_numpy(tiffarray.raw_images).permute(1,2,0).type(out_datatype)



class ConcatDataset(torch.utils.data.Dataset):
    """
    This is a class definition for a custom dataset that concatenates multiple datasets. It has several methods to get statistics, normalize and standardize the data, and add random shot noise to the dataset. 
    """
    def __init__(self, *datasets):
        self.datasets = datasets
        self.max_values = None
    def __getitem__(self, input):
        n_dataset = 0
        i = input
        for d in self.datasets:
            if i>=len(d):
                n_dataset += 1
                i -= len(d)
            else:
                break
        return tuple(self.datasets[n_dataset][i])
    
    def getSamplePath(self, input):
        """
        Given an input, return the path of the dataset that contains the input.
        @param self - the object instance
        @param input - the input to find the dataset for
        @return The path of the dataset that contains the input.
        """
        n_dataset = 0
        i = input
        for d in self.datasets:
            if i>=len(d):
                n_dataset += 1
                i -= len(d)
            else:
                break
        return self.datasets[n_dataset].data_path

    def __len__(self):
        """
        Return the total number of samples in the dataset by summing the length of each dataset in the list of datasets.
        @return The total number of samples in the dataset.
        """
        return sum(len(d) for d in self.datasets)

    def std(self, dim=0):
        """
        Calculate the standard deviation of the volumes in the dataset along a given dimension.
        @param self - the object instance
        @param dim - the dimension along which to calculate the standard deviation (default is 0)
        @return The standard deviation of the volumes in the dataset along the given dimension.
        """
        all_vols = torch.cat(tuple([d.vols.float().unsqueeze(-1) for d in self.datasets]), dim=0)
        std_vols = all_vols.std(dim=dim).type(self.datasets[0].vols.type())
        return std_vols.permute(3,0,1,2)
    
    def mean(self, dim=0):
        """
        Calculate the mean of the volumes in the dataset along a specified dimension.
        @param self - the dataset object
        @param dim - the dimension along which to calculate the mean (default is 0)
        @return The mean volumes
        """
        all_vols = torch.cat(tuple([d.vols.float().unsqueeze(-1) for d in self.datasets]), dim=0)
        mean_vols = all_vols.mean(dim=dim).type(self.datasets[0].vols.type())
        return mean_vols.permute(3,0,1,2)

    def get_statistics(self):
        """
        Calculate the mean and standard deviation of the images and volumes in the dataset.
        If the images are 4D, concatenate the first channel of each image in the dataset.
        @param self - the object instance
        @return The mean and standard deviation of the images and volumes in the dataset.
        """
        if len(self.datasets[0].stacked_views.shape)==4: # It has sparse images as well
            all_images = torch.cat(tuple([d.stacked_views[...,0].float().unsqueeze(-1) for d in self.datasets]), dim=-1)
            all_images_s = torch.cat(tuple([d.stacked_views[...,1].float().unsqueeze(-1) for d in self.datasets]), dim=-1)
        else:
            all_images = torch.cat(tuple([d.stacked_views.float().unsqueeze(-1) for d in self.datasets]), dim=0)
            all_images_s = all_images

        mean_imgs = all_images.mean().type(self.datasets[0].stacked_views.type())
        std_imgs = all_images.std().type(self.datasets[0].stacked_views.type())

        mean_imgs_s = all_images_s.mean().type(self.datasets[0].stacked_views.type())
        std_imgs_s = all_images_s.std().type(self.datasets[0].stacked_views.type())

        all_vols = torch.cat(tuple([d.vols.float().unsqueeze(-1) for d in self.datasets]), dim=0)
        mean_vols = all_vols.mean().type(self.datasets[0].vols.type())
        std_vols = all_vols.std().type(self.datasets[0].vols.type())

        return mean_imgs, std_imgs, mean_imgs_s, std_imgs_s, mean_vols, std_vols

    def get_max(self):
        """
        This method returns the maximum values of the stacked views and volumes in the datasets.
        If the maximum values have not been computed yet, it concatenates the stacked views and volumes
        from all the datasets and computes the maximum values. It then returns the maximum values.
        @return The maximum values of the stacked views and volumes in the datasets.
        """
        if self.max_values is None:
            if len(self.datasets[0].stacked_views.shape)==4: # It has sparse images as well
                all_images = torch.cat(tuple([d.stacked_views[...,0].float().unsqueeze(-1) for d in self.datasets]), dim=-1)
                all_images_s = torch.cat(tuple([d.stacked_views[...,1].float().unsqueeze(-1) for d in self.datasets]), dim=-1)
            else:
                all_images = torch.cat(tuple([d.stacked_views.float().unsqueeze(-1) for d in self.datasets]), dim=-1)
                all_images_s = all_images
            
            all_vols = torch.cat(tuple([d.vols.float().unsqueeze(-1) for d in self.datasets]), dim=-1)

            # Store for later use
            self.max_values = [all_images.max(), all_images_s.max(), all_vols.max()]
        return self.max_values
    
    def normalize_datasets(self):
        """
        This method normalizes the datasets by dividing each element by the maximum value of the dataset.
        If the maximum values have not been set, it will calculate them using the get_max() method.
        If the dataset has sparse data, it will normalize each channel separately.
        @param self - the object instance
        @return None
        """
        if self.max_values is None:
            self.max_values = self.get_max()
        has_sparse = len(self.datasets[0].stacked_views.shape)==4 # It has sparse images as well

        # Normalize datasets
        for d in self.datasets:
            if has_sparse:
                d.stacked_views[...,0] = (d.stacked_views[...,0]/d.stacked_views[...,0].max() * self.max_values[1]).type(d.stacked_views.type())
                d.stacked_views[...,1] = (d.stacked_views[...,1]/d.stacked_views[...,1].max() * self.max_values[0]).type(d.stacked_views.type())
            else:
                d.stacked_views = (d.stacked_views.float()/d.stacked_views.float().max() * self.max_values[0]).type(d.stacked_views.type())
            d.vols = (d.vols.float() / d.vols.float().max() * self.max_values[2]).type(d.vols.type())
        
    def standarize_datasets(self,stats=None):
        """
        Standardize the datasets using the statistics provided. If no statistics are provided, calculate them first.
        @param self - the object instance
        @param stats - the statistics to use for standardization
        @return None
        """
        if stats is None:
            stats = self.get_statistics()
        
        # Normalize datasets
        for d in self.datasets:
            d.standarize(stats)
    
    def add_random_shot_noise_to_dataset(self, signal_power_range=[32**2,32**2]):
        """
        Add random shot noise to the dataset.
        @param self - the dataset
        @param signal_power_range - the range of signal power
        @return None
        """
        for d in self.datasets:
            d.add_random_shot_noise_to_dataset(signal_power_range=signal_power_range)