
import torch
from torch import nn
import torchvision
import torch.nn.functional as F
import math
from torch.cuda.amp import autocast,GradScaler

class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        depth=5,
        wf=6,
        padding=True,
        batch_norm=True,
        up_mode='upsample',
        drop_out=0,
        use_bias=False,
        skip_conn=False,
        activation=nn.PReLU # elu for unet
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        self.skip_conn = skip_conn
        self.drop_out = drop_out
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm, use_bias=use_bias, activation=activation)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm, use_bias=use_bias, skip_conn=skip_conn, activation=activation)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Sequential(nn.Conv2d(prev_channels, n_classes, kernel_size=1, bias=use_bias),
                    activation()
        )

    @autocast()
    def forward(self, x, store_activations=False):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                # x = F.max_pool2d(x, 2)
                x = F.adaptive_max_pool2d_with_indices(x, x.shape[-1]//2)[0]
                x = F.dropout2d(x, self.drop_out)
        if store_activations:
            activations = x.clone().detach()
            
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])
            x = F.dropout2d(x, self.drop_out)

        if store_activations:
            return self.last(x),activations
        else:
            return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm, kernel_size=3, use_bias=False, stride=1, activation=nn.LeakyReLU):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=int(padding), bias=use_bias))
        block.append(activation())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=kernel_size, stride=1, padding=int(padding), bias=use_bias))
        block.append(activation())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class UNetConv3DBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm, kernel_size=3, use_bias=False, stride=1, activation=nn.LeakyReLU):
        super(UNetConv3DBlock, self).__init__()
        block = []

        block.append(nn.Conv3d(in_size, out_size, kernel_size=kernel_size, stride=[stride,stride,1], padding=kernel_size//2, bias=use_bias))
        block.append(activation())
        if batch_norm:
            block.append(nn.BatchNorm3d(out_size))

        block.append(nn.Conv3d(out_size, out_size, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=use_bias))
        block.append(activation())
        if batch_norm:
            block.append(nn.BatchNorm3d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        # convert to 3D 
        if x.ndim==4:
            x = x.permute(0,2,3,1).unsqueeze(1)
        out = self.block(x)
        return out

class UNetPullBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm=False, kernel_size=3, use_bias=False, stride=1, activation=nn.LeakyReLU):
        super(UNetPullBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=kernel_size//2, bias=use_bias))
        block.append(activation())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.MaxPool2d(kernel_size, stride=stride, padding=1))
        block.append(activation())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm, use_bias=False, skip_conn=True, activation=nn.Softplus):
        super(UNetUpBlock, self).__init__()
        self.skip_conn = skip_conn
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=use_bias)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )
        if self.skip_conn == False:
            in_size = out_size
        else:
            in_size = in_size//2
        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm, use_bias=use_bias, activation=activation)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        if self.skip_conn:
            crop1 = self.center_crop(bridge, up.shape[2:])
            out = up + crop1#torch.cat([up, crop1], 1)
            out = self.conv_block(out)
        else:
            out = self.conv_block(up)

        return out
