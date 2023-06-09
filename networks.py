# 2022/2023 Josue Page Vizcaino pv.josue@gmail.com
# Third party libraries imports
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.functional import split
import torch.nn as nn
import torch.nn.functional as F
import pathlib, glob, re
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from torch.nn.modules import module
from torch.cuda.amp import autocast,GradScaler

from INN_utils import *
from unet import *
from utils import *

def subnet_initialization(m):
    """
    Initialize the weights of the convolutional and linear layers using the Kaiming uniform initialization method. If the layer has a bias, set it to 0.1.
    @param m - the layer to initialize.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data *= 0.1

def subnet_initialization_small(m):
    """
    Initialize the weights of a small subnet using Xavier initialization.
    @param m - the module to initialize
    @return None
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data,0.01)
        if m.bias is not None:
            m.bias.data *= 0.01

def zero_initialization(m):
    """
    This function initializes the weights of a convolutional or linear layer to zero.
    @param m - the layer to initialize
    @return None
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        nn.init.constant_(m.weight.data, 0.0)
        if m.bias is not None:
            m.bias.data *= 0.0

def subnet_initialization_positive(m):
    """
    Initialize the weights of a convolutional or linear layer using Xavier initialization. 
    Then, take the absolute value of the weights and multiply the bias by 0.1.
    @param m - the layer to initialize.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data,0.1)
        m.weight.data = m.weight.data.abs()
        if m.bias is not None:
            m.bias.data *= 0.1
            
def subnet_identity_initializaiton(m):
    """
    Initialize the weights and biases of a convolutional or linear layer to identity or zero values.
    @param m - the layer to initialize
    @return None
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.dirac_(m.weight.data)
        if m.bias is not None:
            m.bias.data *= 0.1
    if isinstance(m, nn.Conv3d):
        nn.init.constant_(m.weight.data,0)
        m.weight.data[:,:,m.weight.data.shape[2]//2,m.weight.data.shape[3]//2, m.weight.data.shape[4]//2] = 1/float(m.weight.data.shape[1])

        if m.bias is not None:
            m.bias.data *= 0.1

def subnet_ones_initializaiton(m):
    """
    Initialize the weights of a neural network with a subnet of ones. 
    @param m - the neural network
    @return None
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.constant_(m.weight.data,1/m.weight.numel())
        m.weight.data /= m.weight.data.sum()
        if m.bias is not None:
            m.bias.data *= 0.0
    if isinstance(m, nn.Conv3d):
        nn.init.constant_(m.weight.data,0)
        m.weight.data[:,:,m.weight.data.shape[2]//2,m.weight.data.shape[3]//2, m.weight.data.shape[4]//2] = 1/float(m.weight.data.shape[1])
        # m.weight.data /= m.weight.data.sum()
        if m.bias is not None:
            m.bias.data *= 0.1
    
def subnet_conv(c_in, c_internal, c_out, do=False, use_bias=True):
    """
    Create a convolutional neural network with convolutional layers and ReLU activation functions.
    @param c_in - the number of input channels
    @param c_internal - the number of internal channels
    @param c_out - the number of output channels
    @param do - whether to use dropout or not given a percentage 0-1.0
    @param use_bias - whether to use bias or not
    @return The convolutional neural network
    """
    net = nn.Sequential(nn.Conv2d(c_in, c_internal, 3, padding=1, bias=use_bias),
                        nn.ReLU(),
                        nn.Conv2d(c_internal, c_out, 3, padding=1, bias=use_bias),
                        nn.ReLU(),
                        nn.Conv2d(c_out, c_out, 1, bias=use_bias),
                        nn.ReLU(),
                        nn.Conv2d(c_out, c_out, 1, bias=use_bias))
    net.apply(subnet_initialization)
    # net[-1].apply(zero_initialization)
    return net

def subnet_conv_half(c_in, c_med, c_out, do=False, use_bias=True):
    """
    Create a convolutional neural network that halves the input size.
    @param c_in - the number of input channels
    @param c_med - the number of output channels for the first convolutional layer
    @param c
    """
    modules = [
                    nn.Conv2d(c_in, c_med, 4, stride=2, padding=1, bias=use_bias),
                        nn.LeakyReLU(),
                        nn.Conv2d(c_med, c_out, 3, padding=1, bias=use_bias),
                        ]
    if do:
        modules.append(nn.Dropout2d(0.01))
    net = nn.Sequential(*modules)
    net.apply(subnet_initialization)
    return net

def reset_ActNorm(network, n_to_reset=50):
    """
    Reset the activation normalization layers in an invertible neural network.
    @param network - the network we are resetting
    @param n_to_reset - the number of activation normalization layers to reset
    @return the network and the number of activation normalization layers reset
    """
    act_norms_reseted = 0
    for INN_module in next(network.named_children())[1]:
        if isinstance(INN_module, Fm.ActNorm):
            INN_module.init_on_next_batch = True
            act_norms_reseted += 1
            if n_to_reset and act_norms_reseted>=n_to_reset:
                break
    return network,act_norms_reseted
    
def reset_perm(network):
    """
    Reset the permutation modules in an invertible neural network.
    @param network - the network to reset
    @return the reset network
    """
    for INN_module in next(network.named_children())[1]: 
        if isinstance(INN_module, PermuteDim):
            INN_module = PermuteDim(INN_module.dims_in, INN_module.dims_c, seed=1234)
            INN_module = INN_module.to(INN_module.perm.device)
    return network

class cond_network(nn.Module):
    """
    This is a PyTorch module that defines a conditional network. The network takes in a light field image and outputs a list of tensors. 
    @param c_in - number of input channels
    @param c_out - number of output channels
    @param n_steps - number of steps in the network
    @param max_steps - maximum number of steps in the network
    @param n_channels - list of number of channels in each step of the network
    @param cond_chans - number of channels in the conditional network
    @param net_constructor - constructor for the subnet_conv
    @return a list of tensors
    """
    def __init__(self, c_in, c_out, n_steps, max_steps=7, n_channels=[], cond_chans=32,net_constructor=subnet_conv):
        super(cond_network, self).__init__()
        self.n_steps = n_steps
        if len(n_channels) == 0:
            n_channels = n_steps * [64]
            n_channels = [int(1*f) for f in [c_in, 48, 64, 80, 128, 128, 128, 90]]

        if n_steps==max_steps:
            n_channels[n_steps-2] = c_out
        else:
            n_channels[n_steps-1] = c_out
        c_in_curr = c_in
        self.subnetworks = []
        self.global_attention = None#GlobalAttention(c_in)

        self.subnetworks.append(ResidualBlock(c_in, c_out, chans_3D=cond_chans))
        self.subnetworks = nn.Sequential(*self.subnetworks)
    
    def forward(self, lf_img):
        return [self.subnetworks(lf_img * (self.global_attention(lf_img) if self.global_attention else 1))]

class ResidualBlock(nn.Module):
    """
    This class defines a Residual Block for a neural network. It takes in an input tensor and applies two convolutional layers with a residual connection. It also applies a 3D convolutional layer to the output of the residual connection. 
    @param in_channels - the number of input channels
    @param out_channels - the number of output channels
    @param chans_3D - the number of channels for the 3D convolutional layer
    @param stride - the stride for the first convolutional layer
    @param downsample - a downsampling layer to be applied to the input tensor
    @param activation - the activation function to be used
    @return the output tensor
    """
    def __init__(self, in_channels, out_channels, chans_3D=32, stride = 1, downsample = None, activation=nn.PReLU()):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        activation)
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        )
        self.downsample = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        )
        self.relu = activation
        self.conv3d = nn.Sequential(
                        nn.Conv3d(1, chans_3D, kernel_size = 3, stride = stride, padding = 1),
                        activation,
                        nn.Dropout3d(),
                        nn.Conv3d(chans_3D, 1, kernel_size = 3, stride = stride, padding = 1))
        self.bn_out = None
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.conv2:
            out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        out = self.conv3d(out.permute(0,2,3,1).unsqueeze(1))[:,0,...].permute(0,3,1,2)
        if self.bn_out:
            out = self.bn_out(out)
        return out

class GlobalAttention(nn.Module):
    """
    This is a PyTorch module that implements a global attention mechanism. The module takes as input a tensor of shape (batch_size, n_chans, seq_len) and applies a convolutional neural network to it. The output of the network is a tensor of shape (batch_size, n_chans, seq_len), where each element in the tensor represents the attention weight for the corresponding element in the input tensor. The attention weights are then used to compute a weighted sum of the input tensor, which is returned as the output of the module.
    """
    def __init__(self, n_chans):
        super(GlobalAttention, self).__init__()
        self.m = nn.Sequential(nn.Conv1d(n_chans, n_chans, 3, 1, 1),
                                    nn.ReLU(),
                                    nn.Conv1d(n_chans, n_chans, 1, 1, 0),
                                    nn.Sigmoid()
        )
    def weights_init(self,m):
        if isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform(m.weight.data)
    def reset(self):
        self.apply(self.weights_init)
    # @autocast
    def forward(self, input):
        return self.m(input.view(input.shape[0],input.shape[1],-1)).view(input.shape)

def conditional_wavelet_flow(input_volume_shape, condition_shape, st_subnet, conditional_network, n_down_steps=2, use_permutations=False, block_type='RNVP', n_internal_ch=128 , n_blocks=1, disable_low_res_input=False, device='cpu'):
    """
    This function creates a conditional wavelet flow model with a given input volume shape, condition shape, and other hyperparameters. 
    The model is constructed using a series of subnetworks, each of which is a graph of nodes representing the flow of data through the network. 
    The subnetworks are constructed using a combination of Haar wavelet transforms, conditional affine transforms, and various types of coupling blocks. 
    The function returns the conditional network and the subnetworks. 
    """
    
    # We use this global variable to inform the subnetworks about the ammount of parameters to use
    global networks_n_chans
    networks_n_chans = n_internal_ch
    # Generate dummy condition
    # This could also just be the shape of the current condition
    if conditional_network is None:
        cond_processed = [torch.zeros(condition_shape, device=device)]
        cond_net = None
    else:
        cond_net = conditional_network().to(device)
        with torch.no_grad():
            cond_processed = cond_net(torch.rand(condition_shape, device=device))
    # Define constructors for Normalizing flows subnetworks
    args_conv_block = {'subnet_constructor':st_subnet}
    
    
    # Select block type
    INN_block = Fm.RNVPCouplingBlock
    if block_type=='GLOW':
        INN_block = Fm.GLOWCouplingBlock
    if block_type=='GIN':
        INN_block = Fm.GINCouplingBlock
    if block_type=='AI1':
        INN_block = Fm.AllInOneBlock
    if block_type=='CAT':
        INN_block = Fm.ConditionalAffineTransform

    # Which permutation function to use.
    permute_function = PermuteDim# Fm.PermuteRandom #


    # We create a subnetwork for every downsampling operation
    subnetworks = []
    for k in range(n_down_steps):
        nodes = [Ff.InputNode(*input_volume_shape, name=F'input {k}')]
        # Haar downampling
        nodes.append(Ff.Node(nodes[-1],
                            HaarTransform1D,
                            module_args={'order_by_wavelet':True, },
                            name=F'down_sampling_{k}'))
        
        # nodes.append(Ff.Node(nodes[-1],
        #                     Fm.HaarDownsampling,
        #                     module_args={'order_by_wavelet':True, },
        #                     name=F'down_sampling_{k}'))
        
        # Split 1 forth of the channels as output to the next network
        # And use the other 3/4 for the normalizing flow
        input_split_n_dims = nodes[-1].output_dims[0][0]
        output0_split_size = int(input_split_n_dims * 0.5)
        output1_split_size = input_split_n_dims - output0_split_size
        split1 =  Ff.Node(nodes[-1], Fm.Split, {'section_sizes':(output0_split_size, output1_split_size), 'dim':0}, name=F'Split {k}')
        nodes.append(split1)

        # Process the Haar coefficients through the NF
        if k==n_down_steps-1:
            # The second split goes to a conditioned normalizing flow
            curr_condition_shape = list(cond_processed[-1].shape[1:])
            cond = [Ff.ConditionNode(*curr_condition_shape, name=F'Condition {k-1}')]
            if not disable_low_res_input:
                
                cond.append(Ff.ConditionNode(*curr_condition_shape, name=F'Condition I {k-1}'))
                nodes.append(cond[1])
            nodes.append(cond[0])
            nodes.append(Ff.Node(split1.out1,
                                Fm.ConditionalAffineTransform,
                                {'subnet_constructor': wavelet_flow_subnetwork2D if disable_low_res_input else wavelet_flow_subnetwork2D_first}, conditions=cond,
                                name=F'Block_net{k}_input'))

            for nn in range(1,n_blocks+1):
                curr_args = args_conv_block
                nodes.append(Ff.Node(nodes[-1],
                    permute_function if nn%2==0 else Fm.PermuteRandom,
                    {'seed':k+nn},
                    name=F'Permute_net{k}_{nn}'))
                nodes.append(Ff.Node(nodes[-1],
                                    INN_block,
                                    curr_args, conditions=[cond[-1]],
                                    name=F'Block_net{k}_{nn}'))

            
            if use_permutations:
                nodes.append(Ff.Node(nodes[-1],
                                Fm.PermuteRandom,
                                {},
                                name=F'Permute_final2'))
        
        nodes.append(Ff.OutputNode(nodes[-1] if k==n_down_steps-1 else nodes[-1].out1, name=F'Output WVF{k}'))
        # The first split is the output low resolution image
        # Add norm only if the dimensions are larger than 1, as there's no std for a tensor size 1
        nodes.append(Ff.OutputNode(split1.out0, name=F'Output_net{k}'))
        
        # Update input for next step
        input_volume_shape = split1.output_dims[0]
        subnetworks.append(Ff.GraphINN(nodes))

    return cond_net,subnetworks

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Dropout is a regularization technique that randomly drops out (i.e. set to zero) a fraction of the input units during training. This function implements drop path, which is a variant of dropout that drops entire channels of feature maps instead of individual units. 
    @param x - the input tensor
    @param drop_prob - the probability of dropping a channel
    @param training - whether the model is currently training or not
    @return the output tensor after applying drop path
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class LayerNorm(nn.Module):
    """
    This is a custom implementation of Layer Normalization in PyTorch. It normalizes the input tensor along a specified axis. 
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        """
        This is the constructor for a normalization layer. It initializes the weight and bias parameters, as well as the epsilon value and data format.
        @param normalized_shape - the shape of the normalization layer
        @param eps - the epsilon value
        @param data_format - the data format of the normalization layer
        @return None
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    @autocast
    def forward(self, x):
        """
        This function is the forward pass of a Layer Normalization module. It normalizes the input tensor x along the specified dimensions using the mean and variance of the tensor. The normalization is done using the weight and bias parameters of the module. The normalization is done differently depending on the data format specified. If the data format is "channels_last", the normalization is done using the F.layer_norm function. If the data format is "channels_first", the normalization is done manually by computing the mean and variance of the tensor along the channel dimension, normalizing the tensor using these values, and then applying the weight and bias parameters. The normalized tensor is then returned. 
        @param x - the input tensor
        @return the normalized tensor
        """
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Block(nn.Module):
    """
    This is a class that defines a block in a neural network. The block consists of a series of convolutional and linear layers, with normalization and activation functions in between. The block also includes a dropout function and a layer scaling parameter.
    """
    def __init__(self, c_in, dim, drop_path=0., layer_scale_init_value=1e-6):
        """
        This code defines a class that implements a MobileNetV3 block. The block consists of a depthwise convolution, followed by a pointwise convolution, and a skip connection. The class constructor initializes the block's layers and parameters.
        @param c_in - number of input channels
        @param dim - number of output channels
        @param drop_path - dropout probability for the skip connection
        @param layer_scale_init_value - initial value for the layer scale parameter
        @return None
        """
        super().__init__()
        self.drop_prob = drop_path
        self.input = nn.Conv2d(c_in, dim, 1, 1)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        """
        This function defines the forward pass of a neural network module. It takes an input tensor and applies a series of operations to it, returning the output tensor.
        @param x - the input tensor
        @return the output tensor after applying the series of operations.
        """
        x = self.input(x)
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + drop_path(x, self.drop_prob, training=self.training)
        return x

class ConvNeXt(nn.Module):
    """
    This is a class definition for a ConvNeXt neural network module. It takes in an input channel size, output channel size, and optional dropout probability and size. The module consists of a 1x1 convolutional layer followed by a sequence of 3 convolutional layers with kernel size 7, layer normalization, 1x1 convolutional layer, and GELU activation. The forward method takes in an input tensor and returns the output tensor after passing it through the module. The output tensor is the result of adding the output of the sequence of convolutional layers to the output of the 1x1 convolutional layer after applying dropout with the specified probability.
    """
    def __init__(self, c_in, c_out, drop_prob=0.1, size=512):
        """
        This is a ConvNeXt class. It initializes the ConvNeXt object with the following parameters:
        @param c_in - number of input channels
        @param c_out - number of output channels
        @param drop_prob - dropout probability
        @param size - size of the input image
        The class has the following layers:
        - input layer: a 2D convolutional layer with kernel size 1 and stride 1
        - m: a sequential module consisting of:
          - a 2D convolutional layer with kernel size 7, stride 1, and padding 3
          - a layer normalization layer with input shape [c_out, size, size]
          - a 2D convolutional layer with kernel
        """
        super(ConvNeXt, self).__init__()
        self.drop_prob = drop_prob
        self.input = nn.Conv2d(c_in, c_out, 1, 1)
        self.m = nn.Sequential(nn.Conv2d(c_out, c_out, 7, 1, 3),
                    nn.LayerNorm([c_out, size, size]),
                    nn.Conv2d(c_out, c_out, 1, 1),
                    nn.GELU())

    def forward(self, input):
        """
        This is a forward pass through a neural network. The input is first passed through an input layer, which upsamples the input. The upsampled input is then passed through a module `m`. The output of the module is added to the upsampled input after applying dropout with a certain probability `drop_prob`. The final output is returned. 
        @param self - the neural network
        @param input - the input tensor
        @return The output tensor after passing through the neural network.
        """
        upsampled = self.input(input)
        out = self.m(upsampled) + drop_path(upsampled, self.drop_prob, training=self.training)
        return out

class LRNN(nn.Module):
    """
    This is a PyTorch module that implements a 3D convolutional neural network with a U-Net decoder and a global attention mechanism. The network takes in a 3D input tensor and an optional mean volume tensor, and outputs a 3D tensor of the same shape as the input.
    """
    def __init__(self, ch_in, n_depths, use_bias=False, activation=nn.Softplus()):
        """
        This is a class constructor for a LRNN (Low-resolution Neural Network) model. The model takes in a number of input channels, a number of depths, and other optional parameters. The model consists of a 3D convolutional layer, a ConvNeXt layer, a global attention layer, and a deconvolutional layer. 
        @param ch_in - number of input channels
        @param n_depths - number of depths
        @param use_bias - whether to use bias in the layers
        @param activation - activation function to use
        @return None
        """
        super(LRNN, self).__init__()

        n_ch = 3
        self.conv3d = nn.Sequential(nn.Conv3d(1,n_ch, 3, 1, 1),
                                    activation,
                                    nn.Conv3d(n_ch,n_ch, 3, 1, 1),
                                    activation,
                                    nn.Conv3d(n_ch,1, 3, 1, 1)
                                    )
        self.conv3d = nn.Sequential(
                        ConvNeXt(n_depths, 64, 0.05),
                        ConvNeXt(64, n_depths, 0.05)
                        )

        self.attention_3d = GlobalAttention(n_depths)

        # 3D reconstruction net
        unet_settings = {'depth': 3, 'wf': 8, 'drop_out': 0.005, 'use_bias':use_bias, 'skip_conn': True, 'up_mode': 'upconv', 'batch_norm': True}
        self.deconv = nn.Sequential(
                        nn.Conv2d(ch_in, n_depths, 1, stride=1, padding=0, bias=use_bias),
                        UNet(n_depths, n_depths, depth=unet_settings['depth'], wf=unet_settings['wf'], 
                            drop_out=unet_settings['drop_out'], use_bias=unet_settings['use_bias'], 
                            skip_conn=unet_settings['skip_conn'], up_mode=unet_settings['up_mode'], batch_norm=unet_settings['batch_norm']),
                        )
        self.deconv[0].apply(subnet_initialization_positive)

    def forward(self, x_in, mean_vol=None):
        """
        This function is a forward pass through a 3D convolutional neural network. 
        @param x_in - the input tensor
        @param mean_vol - the mean volume tensor
        @return The output tensor after the forward pass.
        """
        x = self.deconv(x_in)
        if mean_vol is not None:
            mean_processed = self.conv3d(mean_vol)
            x += mean_processed * 2 * (self.attention_3d(mean_vol)-0.5)
        return x

class Encoder(nn.Module):
    """
    This is a PyTorch module that defines an Encoder class. The Encoder class is used to encode input images using a LRNN (Locally Recurrent Neural Network) model. 
    """
    def __init__(self, c_in, c_out, n_steps, n_channels=[], use_bias=False):
        """
        This is a constructor for an Encoder class that inherits from the nn.Module class. The Encoder class takes in the following parameters:
        @param c_in - the number of input channels
        @param c_out - the number of output channels
        @param n_steps - the number of time steps
        @param n_channels - a list of the number of channels for each layer
        @param use_bias - a boolean indicating whether to use bias in the layers
        """
        super(Encoder, self).__init__()
        self.net = LRNN(c_in, c_out, use_bias)

    def forward(self, im_in, mean_vol=None):
        """
        This function is a forward pass through a neural network. It takes an input image and an optional mean volume. If the mean volume is not None, it is passed to the network along with the input image. The output of the network is returned as a list. 
        @param self - the neural network
        @param im_in - the input image
        @param mean_vol - the optional mean volume
        @return a list containing the output of the network.
        """
        if mean_vol is None:
            return [self.net(im_in)]
        else:
            return [self.net(im_in, mean_vol)]

class wavelet_flow_subnetwork(nn.Module):
    """
    This is a PyTorch module that implements a wavelet flow subnetwork. The wavelet flow network is described in the paper "Wavelet Flow: Fast Training of High Resolution Normalizing Flows" (arxiv 2010.13821). 
    """
    '''Wavelet flow network: arxiv 2010.13821'''
    def __init__(self, c_in, c_out, c_internal=32):
        """
        This is the constructor for a wavelet flow subnetwork. It initializes the
        parameters of the network.
        @param c_in - the number of input channels
        @param c_out - the number of output channels
        @param c_internal - the number of internal channels
        @return None
        """
        super(wavelet_flow_subnetwork, self).__init__()
        self.c_in=c_in
        self.c_out = c_out
        self.n_ch = c_internal
        self.n_ch = networks_n_chans
        self.init_blocks(nn.Conv3d, nn.BatchNorm3d)
        self.normal = True

    def init_blocks(self, conv_type, bn=nn.BatchNorm2d, use_bias=True):
        """
        This function initializes the blocks of a neural network. It takes in the type of convolutional layer to use, whether to use batch normalization, and whether to use bias. It initializes the blocks of the network using the given parameters.
        @param self - the object instance
        @param conv_type - the type of convolutional layer to use
        @param bn - whether to use batch normalization
        @param use_bias - whether to use bias
        @return None
        """
        self.conv_type = conv_type
        kernel_size = 3
        self.bn = None#bn(self.c_in)
        self.act = nn.ELU
        self.block_grad_up = self.conv_type(self.c_in//2, self.c_in, 3, padding=1, bias=use_bias)
        self.block1 = self.conv_type(self.c_in//2, self.n_ch, 1, bias=use_bias)
        self.block12 = self.conv_type(self.c_in, self.n_ch, 1, bias=use_bias)
        self.block2 = nn.Sequential( self.conv_type(self.n_ch, self.n_ch, kernel_size, kernel_size//2, 1, bias=use_bias),
                                self.act(),
                                self.conv_type(self.n_ch, self.n_ch, 1, bias=use_bias))
        self.block3 = self.act()
        self.block4 = nn.Sequential( self.conv_type(self.n_ch, self.n_ch, kernel_size, kernel_size//2, 1, bias=use_bias),
                                self.act(),
                                self.conv_type(self.n_ch, self.n_ch, 1, bias=use_bias))
        self.block5 = self.act()
        self.block6 = nn.Sequential( self.conv_type(self.n_ch, self.n_ch, kernel_size, kernel_size//2, 1, bias=use_bias),
                                self.act(),
                                self.conv_type(self.n_ch, self.n_ch, 1, bias=use_bias))
        self.block7 = nn.Sequential( self.act(), 
                                self.conv_type(self.n_ch, self.c_out//2, kernel_size, kernel_size//2, 1, bias=use_bias))
        self.block72 = nn.Sequential( self.act(), 
                                self.conv_type(self.n_ch, self.c_out, kernel_size, kernel_size//2, 1, bias=use_bias))

        
    def forward(self, input):
        """
        This is the forward pass of a neural network. It takes an input and passes it through a series of blocks to produce an output.
        @param self - the neural network object
        @param input - the input tensor
        @return the output tensor
        """

        if self.normal:
            b1 = self.block12(input)
        else:
            # Conditions are the processed cond C and the mean volumes at this resolution
            n = self.c_in//2
            if self.bn:
                input = self.bn(input)
            low_res_up_grad,cond = input[:,:-n,...], input[:,-n:,...]
            b1 = self.block1(cond)
        
        b2 = self.block2(b1) + b1
        b3 = self.block3(b2)
        b4 = self.block4(b3) + b3
        b5 = self.block5(b4)
        b6 = self.block6(b5) + b5

        if self.normal:
            b7 = self.block72(b6)
            return b7
        else:
            b7 = self.block7(b6)

            return torch.cat((b7, -low_res_up_grad / math.sqrt(2)), 1)

class wavelet_flow_subnetwork2D(wavelet_flow_subnetwork):
    """
    This is a class that inherits from `wavelet_flow_subnetwork` and defines a 2D version of the wavelet flow subnetwork. It takes in the number of input channels, output channels, and internal channels as arguments. It initializes the parent class with these arguments and then initializes the blocks using `nn.Conv2d`.
    """
    def __init__(self, c_in, c_out, c_internal=[]):
        super(wavelet_flow_subnetwork2D, self).__init__(c_in, c_out, c_internal)
        self.c_in=c_in
        self.c_out = c_out
        self.c_internal = c_internal
        self.init_blocks(nn.Conv2d)

class wavelet_flow_subnetwork2D_first(wavelet_flow_subnetwork):
    """
    This is a class definition for a 2D wavelet flow subnetwork. It inherits from the `wavelet_flow_subnetwork` class. The constructor takes in the number of input channels, output channels, and internal channels. It initializes the input and output channels, internal channels, and the blocks using `nn.Conv2d`. It sets the `normal` attribute to `False`. Finally, it applies a `subnet_initialization_small` function to the last block in the network.
    """
    def __init__(self, c_in, c_out, c_internal=[]):
        """
        This is a class constructor for a 2D wavelet flow subnetwork. It initializes the
        class with the given input and output channels, as well as any internal channels.
        It then initializes the blocks with a 2D convolutional layer. Finally, it sets
        the normal flag to False and applies a subnet initialization function to the
        last block. 
        @param c_in - the number of input channels
        @param c_out - the number of output channels
        @param c_internal - the number of internal channels
        @return None
        """
        super(wavelet_flow_subnetwork2D_first, self).__init__(c_in, c_out, c_internal)
        self.c_in=c_in
        self.c_out = c_out
        self.c_internal = c_internal
        self.init_blocks(nn.Conv2d)
        self.normal = False
        self.block7[-1].apply(subnet_initialization_small)
                
def serialize_INN_step(INN, cond, optimizer, std_train_stats, args, epoch, path, posfix=''):
    """
    Serialize the INN step by saving the epoch, arguments, INN state dictionary, condition state dictionary, optimizer state dictionary, training statistics, and path.
    @param INN - the INN model
    @param cond - the condition model
    @param optimizer - the optimizer
    @param std_train_stats - the training statistics
    @param args - the arguments
    @param epoch - the epoch
    @param path - the path to save the serialized INN step
    @param posfix - the postfix to add to the path
    @return None
    """
    path += '/model_step_' + str(args.INN_down_steps)  + '__ep_' + str(epoch) + posfix
    torch.save({
        'epoch': epoch,
        'args' : args,
        'INN_state_dict': INN.state_dict() if INN else None,
        'condition_state_dict': cond.state_dict() if cond else None,
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'training_statistics': std_train_stats},
        path)
    return

def load_INN_steps(path, prefix='model_step_*__ep_*', epoch=-1):
    """
    Load the INN model from a given path and return the available steps.
    @param path - the path to the INN model
    @param prefix - the prefix of the model
    @param epoch - the epoch to load
    @return A dictionary of available steps
    """
    models = glob.glob(path + '/' + prefix)
    steps_available = {}
    for m in models:
        step,it = map(int, re.findall(r'\d+', m.split('/')[-1]))

        # if a specific epoch was requested, store only if that is found
        if epoch==-1:
            # Check if this is the highes iteration
            if step in steps_available.keys() and it < steps_available[step][0]:  
                pass
            else:
                steps_available[step] = [it, m]
        else:
            if it==epoch:
                steps_available[step] = [it, m]
    
    return steps_available
            
class XLFMNet(nn.Module):
    """
    This is a PyTorch neural network class called XLFMNet. It inherits from the nn.Module class. The constructor takes in several parameters:
    """
    def __init__(self, in_views, output_shape, use_bias=False, unet_settings={'depth':5, 'wf':6, 'drop_out':1.0, 'batch_norm':True, 'skip_conn':False}):
        """
        This is the constructor for the XLFMNet class. The XLFMNet is a neural network that takes in multiple views of an object and outputs a single 3D reconstruction of the object. 
        """
        super(XLFMNet, self).__init__()
        self.output_shape = output_shape

        out_depths = output_shape[2]
        
        # 3D reconstruction net
        self.deconv = nn.Sequential(
                        nn.Conv2d(in_views,out_depths, 3, stride=1, padding=1, bias=use_bias),
                        nn.BatchNorm2d(out_depths),
                        nn.LeakyReLU(),
                        UNet(out_depths, out_depths, depth=unet_settings['depth'], wf=unet_settings['wf'], drop_out=unet_settings['drop_out'], use_bias=use_bias, activation=nn.ELU))
        

    @autocast()
    def forward(self, input):
        """
        This is a forward pass function for a neural network. It takes an input tensor and applies a deconvolution operation to it. The `@autocast()` decorator is used to automatically cast the input tensor to the appropriate data type for the operation. The function returns the output tensor after the deconvolution operation has been applied.
        """
        # Run 3D reconstruction network
        out = self.deconv(input)
        return out
