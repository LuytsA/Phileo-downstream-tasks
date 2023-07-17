# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from itertools import repeat
import collections.abc

import torch
import torch.nn as nn

import timm.models.vision_transformer
from utils.transformer_utils import get_2d_sincos_pos_embed
from timm.models.vision_transformer import PatchEmbed, Block

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)


class ViT_backbone(nn.Module):
    """VisionTransformer backbone
    """
    def __init__(self, chw:tuple=(10, 64, 64), patch_size:int=16,
                 embed_dim:int=768, depth:int=3, num_heads:int=16,
                 mlp_ratio:float=4., norm_layer:nn.Module=nn.LayerNorm):
        super().__init__()

        # Attributes
        self.chw = chw  # (C, H, W)
        self.in_c = chw[0]
        self.img_size = chw[1]
        self.emded_dim = embed_dim

        self.patch_embed = PatchEmbed(self.img_size, patch_size, self.in_c, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

class ViT_decoder(ViT_backbone):
    """ Autoencoder with VisionTransformer backbone
    """
    def __init__(self, chw:tuple=(10, 64, 64), out_chans:int=1, patch_size:int=16, input_embed_dim:int=768,
                 embed_dim:int=512,  depth:int=3, num_heads:int=16,
                 mlp_ratio:float=4., norm_layer:nn.Module=nn.LayerNorm):

        super().__init__(chw=chw, patch_size=patch_size,
                         embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                         norm_layer=norm_layer)


        self.decoder_embed = nn.Linear(input_embed_dim, self.emded_dim, bias=True)
        self.decoder_pred = nn.Linear(self.emded_dim, patch_size ** 2 * out_chans, bias=True)  # dec
        self.initialize_weights()

    def forward(self, x):
        # embed tokens
        x = self.decoder_embed(x)

        # add pos embed
        x = x + self.pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x



class Vit_ae(nn.Module):
    def __init__(self, chw:tuple=(10, 64, 64), out_chans:int=1, patch_size:int=16,
                 embed_dim:int=512, depth:int=3, num_heads:int=16,
                 mlp_ratio:float=4., norm_layer:nn.Module=nn.LayerNorm,
                 decoder_embed_dim:int=128, decoder_depth:int=8, decoder_num_heads:int=16,
                 ):
        super().__init__()

        self.vit_encoder = ViT_backbone(chw, patch_size, embed_dim, depth, num_heads, mlp_ratio, norm_layer)

        self.vit_decoder = ViT_decoder(chw, out_chans, patch_size,  input_embed_dim=embed_dim, embed_dim=decoder_embed_dim,
                                       depth=decoder_depth, num_heads=decoder_num_heads, mlp_ratio=mlp_ratio,
                                       norm_layer=norm_layer)

    def forward(self, x):
        x = self.vit_encoder(x)
        x = self.vit_decoder(x)

        return x

class Vit_basic(nn.Module):
    def __init__(self, chw:tuple=(10, 64, 64), out_chans:int=1, patch_size:int=16,
                 embed_dim:int=512, depth:int=1, num_heads:int=16,
                 mlp_ratio:float=4., norm_layer:nn.Module=nn.LayerNorm):
        super().__init__()

        self.vit_encoder = ViT_backbone(chw, patch_size, embed_dim, depth, num_heads, mlp_ratio, norm_layer)
        # decoder to patch
        self.decoder_pred = nn.Linear(embed_dim, int(out_chans * patch_size ** 2), bias=True)

    def forward(self, x):
        x = self.vit_encoder(x)
        x = self.decoder_pred(x)
        # remove cls token
        x = x[:, 1:, :]
        return x

if __name__ == '__main__':
    m = vit_base_patch16()
    x = torch.rand((32, 10, 64, 64))
    m(x)