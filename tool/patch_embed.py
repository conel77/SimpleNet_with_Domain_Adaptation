""" Image to Patch Embedding using Conv2d

A convolution based approach to patchifying a 2D image w/ embedding projection.

Based on code in:
  * https://github.com/google-research/vision_transformer
  * https://github.com/google-research/big_vision/tree/main/big_vision

Hacked together by / Copyright 2020 Ross Wightman
"""
import logging
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import nn as nn
import torch.nn.functional as F

from tool.format import Format, nchw_to
#from .helpers import to_2tuple
#from .trace_utils import _assert

_logger = logging.getLogger(__name__)

import torch
from torch import nn
import torch.nn.functional as F

class PatchEmbedDecoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 norm_layer=None, flatten=True, output_fmt='NCHW', bias=True,
                 strict_img_size=True, dynamic_img_pad=False, device='cuda'):
        super(PatchEmbedDecoder, self).__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.flatten = flatten
        self.output_fmt = output_fmt
        self.dynamic_img_pad = dynamic_img_pad
        self.device = device
        self.proj = nn.ConvTranspose2d(embed_dim, in_chans, kernel_size=patch_size, stride=patch_size, bias=bias).to(self.device)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity().to(self.device)

    def forward(self, x):
        print('start', x.shape)

        # Transpose NHWC to NCHW
        x = x.permute(0, 2, 1).reshape(x.size(0), self.embed_dim, 1, -1)  # NHWC -> NCHW
        # Inverse ConvTranspose2d operation to upsample the patches to the original image size
        x = self.proj(x)

        # Remove any padding applied during the embedding process
        if self.dynamic_img_pad:
            img_size = (x.size(2) * self.patch_size, x.size(3) * self.patch_size)
            x = x[:, :, :img_size[0], :img_size[1]]

        # Transpose NCHW to NHWC
        x = x.permute(0, 2, 1, 3).reshape(x.size(0), self.embed_dim, -1)  # NCHW -> NHWC -> NLC

        #print('1', x.shape)

        # If the model used flattening, transpose the tensor back
        if self.flatten:
            x = x.permute(0, 2, 1)  # NLC -> NHWC

        #print('2', x.shape)
        # If the output format is not NCHW, convert it to NCHW
        if self.output_fmt != 'NCHW':
            x = self._nchw_to(x, 'NCHW')

        #print('3', x.shape)
        return x
    # def forward(self, x):
    #     print('start', x.shape)
    #     # Normalize the embedded patches
    #     x = self.norm(x)

    #     # If the output format is not NCHW, convert it to NCHW
    #     if self.output_fmt != 'NCHW':
    #         x = self._nchw_to(x, 'NCHW')

    #     print('1',x.shape)

    #     # If the model used flattening, transpose the tensor back
    #     if self.flatten:
    #         x = x.reshape(x.size(0), self.embed_dim, -1)  # NLC -> NHWC
    #         x = x.permute(0, 2, 1)  # NHWC -> NCHW

    #     print('2', x.shape)


    #     # Transpose NHWC to NCHW
    #     #x = x.permute(0, 2, 1).reshape(x.size(0), -1, 1, 1)
    #     x = x.permute(0, 2, 1).reshape(x.size(0), self.embed_dim, 1, -1)  # NHWC -> NCHW

    #     print('3', x.shape)

    #     # Inverse ConvTranspose2d operation to upsample the patches to the original image size
    #     x = self.proj(x)


    #     # Remove any padding applied during the embedding process
    #     if self.dynamic_img_pad:
    #         img_size = (x.size(2) * self.patch_size, x.size(3) * self.patch_size)
    #         x = x[:, :, :img_size[0], :img_size[1]]

    #     return x

    def _nchw_to(self, x, target_format):
        if target_format == 'NCHW':
            return x
        elif target_format == 'NHWC':
            return x.permute(0, 2, 3, 1)
        else:
            raise ValueError(f"Unsupported target format: {target_format}")



