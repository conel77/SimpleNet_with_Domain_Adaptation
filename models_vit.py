# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer

from tool.patch_embed import PatchEmbedDecoder


patch_embed_params = {
    'img_size': 224,
    'patch_size': 16,
    'in_chans': 3,
    'embed_dim': 768,
    'norm_layer': None,
    'flatten': True,
    'output_fmt': 'NCHW',
    'bias': True,
    'strict_img_size': True,
    'dynamic_img_pad': False,
}


# PatchEmbedDecoder 초기화
patchembeddecoder = PatchEmbedDecoder(**patch_embed_params)


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        self.intermediate_output = None
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        #print('image', x.shape)
        B = x.shape[0]
        x = self.patch_embed(x)

        intermediate_outputs = []  # 리스트로 intermediate_output 저장

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
            intermediate_outputs.append(patchembeddecoder(x.clone().detach()))

        #intermediate_outputs = patchembeddecoder(intermediate_outputs)
        self.intermediate_output = intermediate_outputs

            
        # intermediate_output = x.clone().detach()
        # print('xxxx', x.shape)
        # intermediate_output = patchembeddecoder(intermediate_output)


        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        return outcome, intermediate_outputs


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

#-------------------------original ver----------------------------------------#
# class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
#     """ Vision Transformer with support for global average pooling
#     """
#     def __init__(self, global_pool=False, **kwargs):
#         super(VisionTransformer, self).__init__(**kwargs)

#         self.global_pool = global_pool
#         if self.global_pool:
#             norm_layer = kwargs['norm_layer']
#             embed_dim = kwargs['embed_dim']
#             self.fc_norm = norm_layer(embed_dim)

#             del self.norm  # remove the original norm

#     def forward_features(self, x):
#         B = x.shape[0]
#         x = self.patch_embed(x)

#         cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
#         x = torch.cat((cls_tokens, x), dim=1)
#         x = x + self.pos_embed
#         x = self.pos_drop(x)

#         for blk in self.blocks:
#             x = blk(x)

#         if self.global_pool:
#             x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
#             outcome = self.fc_norm(x)
#         else:
#             x = self.norm(x)
#             outcome = x[:, 0]

#         return outcome


# def vit_base_patch16(**kwargs):
#     model = VisionTransformer(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


# def vit_large_patch16(**kwargs):
#     model = VisionTransformer(
#         patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


# def vit_huge_patch14(**kwargs):
#     model = VisionTransformer(
#         patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model