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


class VitEncoder(nn.Module):
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

class VitDecoder(VitEncoder):
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

class VitMaskedEncoder(VitEncoder):
    def __init__(self, chw:tuple=(10, 64, 64), patch_size:int=16, mask_ratio:float=0.75,
                 embed_dim:int=768, depth:int=3, num_heads:int=16,
                 mlp_ratio:float=4., norm_layer:nn.Module=nn.LayerNorm):

        super().__init__(chw=chw, patch_size=patch_size, embed_dim=embed_dim,
                         depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                         norm_layer=norm_layer)

        self.mask_ratio = mask_ratio

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [B, L, D], sequence
        """
        B, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(B, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

class VitMaskedDecoder(VitDecoder):
    def __init__(self, chw:tuple=(10, 64, 64), out_chans:int=1, patch_size:int=16, input_embed_dim:int=768,
                 embed_dim:int=512,  depth:int=3, num_heads:int=16,
                 mlp_ratio:float=4., norm_layer:nn.Module=nn.LayerNorm):

        super().__init__(chw=chw, patch_size=patch_size, out_chans=out_chans, input_embed_dim=input_embed_dim,
                         embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                         norm_layer=norm_layer)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

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

        self.vit_encoder = VitEncoder(chw, patch_size, embed_dim, depth, num_heads, mlp_ratio, norm_layer)

        self.vit_decoder = VitDecoder(chw, out_chans, patch_size, input_embed_dim=embed_dim, embed_dim=decoder_embed_dim,
                                      depth=decoder_depth, num_heads=decoder_num_heads, mlp_ratio=mlp_ratio,
                                      norm_layer=norm_layer)

    def forward(self, x):
        x = self.vit_encoder(x)
        x = self.vit_decoder(x)

        return x


class mae_vit(VitEncoder):
    def __init__(self, chw:tuple=(10, 64, 64), out_chans:int=10, patch_size:int=16, mask_ratio:float=0.75,
                 embed_dim:int=768, depth:int=12, num_heads:int=12,
                 mlp_ratio:float=4., norm_layer:nn.Module=partial(nn.LayerNorm, eps=1e-6),
                 decoder_embed_dim:int=512, decoder_depth:int=8, decoder_num_heads:int=16,
                 ):
        super().__init__()

        self.vit_encoder = VitMaskedEncoder(chw, patch_size, mask_ratio,
                                            embed_dim, depth, num_heads,
                                            mlp_ratio, norm_layer)

        self.vit_decoder = VitMaskedDecoder(chw, out_chans, patch_size, embed_dim,
                                            decoder_embed_dim,  decoder_depth, decoder_num_heads,
                                            mlp_ratio, norm_layer)

    def patchify(self, imgs, p, c):
        """
        imgs: (N, C, H, W)
        p: Patch embed patch size
        c: Num channels
        x: (N, L, patch_size**2 *C)
        """
        # p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        # c = self.in_c
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * c))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *C)
        p: Patch embed patch size
        c: Num channels
        imgs: (N, C, H, W)
        """
        c = self.in_c
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def loss(self, imgs, pred, mask):
        """
        imgs: [N, C, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        # target = imgs[:, :3, :, :]
        # pred = self.unpatchify(pred, self.patch_embed.patch_size[0], self.in_c)
        # pred = self.patchify(pred[:, :3, :, :], self.patch_embed.patch_size[0], 3)
        # target = self.patchify(target, self.patch_embed.patch_size[0], 3)
        target = self.patchify(imgs, self.patch_embed.patch_size[0], self.in_c)
        # if self.norm_pix_loss:
        #     mean = target.mean(dim=-1, keepdim=True)
        #     var = target.var(dim=-1, keepdim=True)
        #     target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss


    def forward(self, x):
        latent, mask, ids_restore = self.vit_encoder(x)
        pred = self.vit_decoder(latent, ids_restore)
        loss = self.loss(x, pred, mask)

        return loss, pred, mask

class Vit_basic(nn.Module):
    def __init__(self, chw:tuple=(10, 64, 64), out_chans:int=1, patch_size:int=16,
                 embed_dim:int=512, depth:int=1, num_heads:int=16,
                 mlp_ratio:float=4., norm_layer:nn.Module=nn.LayerNorm):
        super().__init__()

        self.vit_encoder = VitEncoder(chw, patch_size, embed_dim, depth, num_heads, mlp_ratio, norm_layer)
        # decoder to patch
        self.decoder_pred = nn.Linear(embed_dim, int(out_chans * patch_size ** 2), bias=True)

    def forward(self, x):
        x = self.vit_encoder(x)
        x = self.decoder_pred(x)
        # remove cls token
        x = x[:, 1:, :]
        return x


