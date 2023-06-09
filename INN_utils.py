# Third party libraries imports
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.arraysetops import isin
import torch
from torch.functional import split
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
import pathlib, glob, re
import logging
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from scipy import linalg as la
from typing import Union
import math

class Inv2Dto3D(Fm.InvertibleModule):
    """
    This is a class definition for an invertible module that converts 2D images to 3D images.
    @param input_shape - the shape of the input tensor
    @param split_factor - the factor by which to split the input tensor
    @return an instance of the Inv2Dto3D class
    """
    def __init__(self, input_shape, split_factor=3):
        """
        This is the constructor for a class that inherits from another class. It takes in an input shape and a split factor, which defaults to 3 if not provided. The constructor then calls the constructor of the parent class.
        """
        super().__init__(input_shape)
        self.split_factor = split_factor
        
    def forward(self, x, rev=False, jac=True): 
        """
        This is a forward function for a neural network. It takes in an input tensor `x`, and two optional boolean parameters `rev` and `jac`. 
        """
        if rev:
            out = turn_3D_to_2D(x[0])
        else:
            out = torch.cat(tuple([t.permute(0,2,3,1).unsqueeze(1) for t in torch.split(x[0], x[0].shape[1]//self.split_factor, 1)]), 1)
        return (out,), 0

    def output_dims(self, input_dims):
        '''See docstring of base class (FrEIA.modules.InvertibleModule).'''
        return tuple([self.split_factor, i[1], i[2], i[0]//self.split_factor] for i in input_dims)

class PermuteDim(Fm.InvertibleModule):
    """
    This is a class that inherits from the InvertibleModule class. It is used to permute dimensions of the input tensor.
    @param dims_in - the input dimensions
    @param dims_c - the conditional dimensions
    @param dims_to_permute - the dimensions to permute
    @param seed - the seed for the random number generator
    @return None
    """
    def __init__(self, dims_in, dims_c=None, dims_to_permute=[1,2], seed: Union[int, None] = None):
        super().__init__(dims_in, dims_c)

        possible_dims_permute = [[1,2],[1,3]]

        self.in_channels = dims_in[0][0]
        self.dims_to_permute = possible_dims_permute[np.random.randint(0,len(possible_dims_permute))]
        if seed is not None:
            np.random.seed(seed)
        self.perm = np.random.permutation(dims_in[0][self.dims_to_permute[1]-1])

        self.perm_inv = np.zeros_like(self.perm)
        for i, p in enumerate(self.perm):
            self.perm_inv[p] = i

        self.perm = nn.Parameter(torch.LongTensor(self.perm), requires_grad=False)
        self.perm_inv = nn.Parameter(torch.LongTensor(self.perm_inv), requires_grad=False)

    def forward(self, x, rev=False, jac=True):
        """
        This function is a forward pass of a permutation layer. It takes an input tensor `x` and applies a permutation to it. The permutation is defined by the `perm` attribute of the layer. If `rev` is True, it applies the inverse permutation instead. The `jac` argument is not used in this implementation and is ignored.
        """
        input_t = x[0].transpose(self.dims_to_permute[0],self.dims_to_permute[1])
        if not rev:
            return [input_t[:, self.perm].transpose(self.dims_to_permute[0],self.dims_to_permute[1])], 0.
        else:
            return [input_t[:, self.perm_inv].transpose(self.dims_to_permute[0],self.dims_to_permute[1])], 0.

    def output_dims(self, input_dims):
        '''See docstring of base class (FrEIA.modules.InvertibleModule).'''
        if len(input_dims) != 1:
            raise ValueError(f"{self.__class__.__name__} can only use 1 input")
        return input_dims

class Inv3Dto2D(Fm.InvertibleModule):
    """
    This is a class definition for an invertible module that converts 3D input to 2D output.
    @param input_shape - the shape of the input tensor
    @returns an instance of the Inv3Dto2D class
    """
    def __init__(self, input_shape):
        super().__init__(input_shape)
        
    def forward(self, x, rev=False, jac=True):
        """
        This function takes an input tensor `x` and converts it from 3D to 2D or vice versa depending on the value of the `rev` parameter. If `rev` is False, the input tensor is converted from 3D to 2D using the `turn_3D_to_2D` function. If `rev` is True, the input tensor is converted from 2D to 3D by concatenating the tensor along the channel dimension. The `jac` parameter is not used in this function. The function returns a tuple containing the output tensor and a scalar value of 0. 
        @param x - the input tensor
        @param rev - a boolean indicating whether to convert from 3D to 
        """
        if not rev:
            out = turn_3D_to_2D(x[0])
        else:
            out = torch.cat(tuple([t.permute(0,2,3,1).unsqueeze(1) for t in torch.split(x[0], x[0].shape[1]//3, 1)]), 1)
        return (out,), 0

    def output_dims(self, input_dims):
        '''See docstring of base class (FrEIA.modules.InvertibleModule).'''
        return tuple([3*i[-1], i[1], i[2]] for i in input_dims)

def turn_2D_to_3D(x):
    return x.permute(0,2,3,1).unsqueeze(1)

def turn_3D_to_2D(x):
    if x.ndim == 4: # Already a 2D tensor
        return x
    assert x.ndim == 5, "This isn't a 3D tensor"
    out = torch.zeros(x.shape[0], x.shape[-1]*x.shape[1], x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
    for nD in range(x.shape[1]):
        out[:,nD*x.shape[-1]:(nD+1)*x.shape[-1],...] = x[:,nD,...].permute(0,3,1,2)
    return out

class HaarTransform1D(Fm.InvertibleModule):
    """
    This is a class that implements the 1D Haar wavelet transform. It inherits from the InvertibleModule class. The constructor takes in the input dimensions, the conditioning dimensions, a boolean flag indicating whether to order by wavelet, and a rebalance factor. The forward method takes in the input tensor and conditioning tensor, and returns the output tensor and the log determinant of the Jacobian. The output dimensions are calculated based on the input dimensions.
    """
    def __init__(self, dims_in, dims_c = None,
                 order_by_wavelet: bool = False,
                 rebalance: float = 1.):
        
        super().__init__(dims_in, dims_c)
        self.fac_fwd = 0.5 * rebalance
        self.fac_rev = 0.5 / rebalance

        self.jac_fwd = (np.log(16.) + 4 * np.log(self.fac_fwd)) / 4.

        self.jac_rev = (np.log(16.) + 4 * np.log(self.fac_rev)) / 4.

    def forward(self, x_in, c=None, jac=True, rev=False):
        """
        This function is a forward pass of a neural network layer that performs a Haar wavelet transform on the input tensor. The Haar wavelet transform is a mathematical operation that decomposes a signal into a set of wavelets, which can be used to analyze and compress the signal.
        """
        x = x_in[0]
        n = x.shape[1]
        h = n // 2
        ndims = x[0].numel()
        
        output = torch.zeros_like(x)
        fac = 1.0 / math.sqrt(2)
        if not rev:
            output[:, :h, ...] = (x[:, ::2, ...] + x[:, 1::2, ...])
            output[:, h:, ...] = (x[:, ::2, ...] - x[:, 1::2, ...]) 
            logdet = ndims * self.jac_fwd
        else:
            output[:, ::2, ...] = (x[:, :h, ...] + x[:, h:, ...])
            output[:, 1::2, ...] = (x[:, :h, ...] - x[:, h:, ...])
            logdet = -ndims * self.jac_rev
        return (output * fac,), logdet

    def output_dims(self, input_dims):
        '''See docstring of base class (FrEIA.modules.InvertibleModule).'''

        if len(input_dims) != 1:
            raise ValueError("HaarDownsampling must have exactly 1 input")
        if len(input_dims[0]) != 3:
            raise ValueError("HaarDownsampling can only transform 2D images"
                             "of the shape CxWxH (channels, width, height)")

        c2, w2, h2 = input_dims[0]
        
        return ((c2, w2, h2),)
