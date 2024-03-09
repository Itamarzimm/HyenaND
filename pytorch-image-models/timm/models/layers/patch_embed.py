""" Image to Patch Embedding using Conv2d

A convolution based approach to patchifying a 2D image w/ embedding projection.

Based on the impl in https://github.com/google-research/vision_transformer

Hacked together by / Copyright 2020 Ross Wightman
"""

from torch import nn as nn
import torch
from .helpers import to_2tuple
from einops.layers.torch import Rearrange
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

# from einops.layers.torch import Rearrange

class EarlyConvs(nn.Module):
    def __init__(self, channels, dim):
        super(EarlyConvs, self).__init__()

        n_filter_list = (channels, 48, 96, 192, 384)

        # Define the convolutional layers in a Sequential container
        conv_layers = []
        for i in range(len(n_filter_list) - 1):
            conv_layers.append(
                nn.Conv2d(
                    in_channels=n_filter_list[i],
                    out_channels=n_filter_list[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1
                )
            )

        self.conv_layers = nn.Sequential(*conv_layers)

        # Add the final 1x1 convolution layer
        self.conv_layers.add_module("conv_1x1",
                                    torch.nn.Conv2d(
                                        in_channels=n_filter_list[-1],
                                        out_channels=dim,
                                        stride=1,
                                        kernel_size=1,
                                        padding=0
                                    ))

        # Add the flatten image layer using einops
        self.conv_layers.add_module("flatten_image",
                                    Rearrange('batch channels height width -> batch (height width) channels'))

    def forward(self, x):

        return self.conv_layers(x)
    
    