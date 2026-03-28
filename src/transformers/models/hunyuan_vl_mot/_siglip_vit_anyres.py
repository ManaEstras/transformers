# coding=utf-8
# Copyright (C) 2025 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Ported from hy_embodied_dev/siglip_vit_anyres.py

"""SigLIP ViT with any-resolution support for HunYuanVL-MoT."""

import math
import warnings
from dataclasses import dataclass
from functools import partial
from typing import (
    Callable,
    Dict,
    Final,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from timm.layers import (
        DropPath,
        LayerType,
        Mlp,
        PatchDropout,
        PatchEmbed,
        to_2tuple,
        resample_abs_pos_embed,
    )
    from timm.models._manipulate import checkpoint_seq, named_apply
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False

from flash_attn import flash_attn_func, flash_attn_varlen_func


# ---------------------------------------------------------------------------
# Feature flags from environment variables (kept for checkpoint compatibility)
# ---------------------------------------------------------------------------
import os

USE_QWEN_MLP = "USE_QWEN_MLP" in os.environ
ENABLE_CONVSTEM = "ENABLE_CONVSTEM" in os.environ
ENABLE_LARGE = "ENABLE_LARGE" in os.environ
VIT_WITH_GRAD = "VIT_WITH_GRAD" in os.environ
FIX_SIZE = "FIX_SIZE" in os.environ


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x.pow(2), dim=(2, 3), keepdim=True) + self.eps)
        return self.gamma.view(1, self.dim, 1, 1) * (x / rms)


class ConvStem(nn.Module):
    """Dual-stem patch embedding: one large stride conv + one multi-stage conv."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=5,
        norm_layer=None,
        checkpointing=False,
    ):
        super().__init__()
        assert embed_dim % 8 == 0, "Embed dimension must be divisible by 8 for ConvStem"

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.depth = depth

        # Stem 1: single large-kernel conv
        self.depth1 = depth1 = 1
        output_dims = [3072] if ENABLE_LARGE else [2048]
        kernal_sizes = [32]
        strides = [32]
        paddings = [0]
        stem1 = []
        input_dim = in_chans
        for idx in range(depth1):
            stage_list = [
                nn.Conv2d(input_dim, output_dims[idx],
                          kernel_size=kernal_sizes[idx], stride=strides[idx],
                          padding=paddings[idx], bias=False),
            ]
            if idx == depth1 - 1 and output_dims[idx] != embed_dim:
                stage_list.append(nn.Conv2d(output_dims[idx], embed_dim, kernel_size=1))
            stem1.append(nn.Sequential(*stage_list))
            input_dim = output_dims[idx]
        self.proj1 = nn.ModuleList(stem1)

        # Stem 2: multi-stage progressively downsampling conv
        if ENABLE_LARGE:
            self.depth2 = depth2 = 6
            output_dims_2 = [64, 64, 128, 512, 512, 512]
            kernal_sizes_2 = [4, 3, 3, 3, 3, 3]
            strides_2 = [4, 2, 2, 2]
            paddings_2 = [0, 1, 1, 1]
        else:
            self.depth2 = depth2 = 4
            output_dims_2 = [64, 64, 128, 512]
            kernal_sizes_2 = [4, 3, 3, 3]
            strides_2 = [4, 2, 2, 2, 1, 1]
            paddings_2 = [0, 1, 1, 1, 1, 1]

        stem2 = []
        input_dim = in_chans
        for idx in range(depth2):
            if idx in (4, 5):
                stage_list = [
                    depthwise_separable_conv(input_dim, output_dims_2[idx]),
                    nn.GroupNorm(1, output_dims_2[idx], eps=1e-6),
                    nn.GELU(),
                ]
            else:
                stage_list = [
                    nn.Conv2d(input_dim, output_dims_2[idx],
                              kernel_size=kernal_sizes_2[idx],
                              stride=strides_2[idx], padding=paddings_2[idx], bias=False),
                    RMSNorm(output_dims_2[idx]),
                    nn.GELU(),
                ]
            if idx == depth2 - 1:
                stage_list.append(nn.Conv2d(output_dims_2[idx], embed_dim, kernel_size=1))
            stem2.append(nn.Sequential(*stage_list))
            input_dim = output_dims_2[idx]
        self.proj2 = nn.ModuleList(stem2)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x1 = x
        x2 = x
        for i, stage in enumerate(self.proj1):
            x1 = stage(x1)
            if i == len(self.proj1) - 1:
                x1 = x1.flatten(2).transpose(1, 2)
                x1 = self.norm(x1)
        for i, stage in enumerate(self.proj2):
            x2 = stage(x2)
            if i == len(self.proj2) - 1:
                x2 = x2.flatten(2).transpose(1, 2)
                x2 = self.norm(x2)
        return x1 + x2


# ---------------------------------------------------------------------------
# Weight init helpers
# ---------------------------------------------------------------------------

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    with torch.no_grad():
        dtype = tensor.dtype
        tensor_fp32 = tensor.float()
        tensor_fp32 = _no_grad_trunc_normal_(tensor_fp32, mean, std, a, b)
        tensor.copy_(tensor_fp32.to(dtype=dtype))


def init_weights_vit_timm(module, name: str = "") -> None:
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()


# ---------------------------------------------------------------------------
# Attention / Block
# ---------------------------------------------------------------------------

class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        seperate_qv_bias: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0.0 else nn.Identity()

        if seperate_qv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

    def forward(self, x: torch.Tensor, cu_slens=None) -> torch.Tensor:
        B, N, C = x.shape
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias,
                                   torch.zeros_like(self.v_bias, requires_grad=False),
                                   self.v_bias))
            qkv = F.linear(x, self.qkv.weight, qkv_bias)
        else:
            qkv = self.qkv(x)

        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Rearrange to (B, N, heads, head_dim) for flash_attn
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if cu_slens is not None:
            max_seqlen = torch.max(cu_slens[1:] - cu_slens[:-1]).item()
            x = flash_attn_varlen_func(
                q.squeeze(0), k.squeeze(0), v.squeeze(0),
                cu_seqlens_q=cu_slens, cu_seqlens_k=cu_slens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                softmax_scale=self.scale, causal=False,
            )
            x = x.reshape(B, N, -1)
        else:
            x = flash_attn_func(q, k, v, softmax_scale=self.scale)
            x = x.reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.SiLU, drop=0., norm_layer=nn.LayerNorm, subln=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        return self.drop(x)


class BlockEVA(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = None,
    ) -> None:
        super().__init__()
        if mlp_layer is None:
            mlp_layer = Mlp
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=False, qk_norm=qk_norm,
            attn_drop=attn_drop, proj_drop=proj_drop, norm_layer=norm_layer,
            seperate_qv_bias=True,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        if USE_QWEN_MLP:
            self.mlp = SwiGLU(in_features=dim, hidden_features=int(dim * mlp_ratio),
                               subln=False, norm_layer=norm_layer)
        else:
            self.mlp = mlp_layer(in_features=dim, hidden_features=int(dim * mlp_ratio),
                                  act_layer=act_layer, drop=proj_drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, cu_slens=None) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), cu_slens=cu_slens)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = None,
    ) -> None:
        super().__init__()
        if mlp_layer is None:
            mlp_layer = Mlp
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
                               attn_drop=attn_drop, proj_drop=proj_drop, norm_layer=norm_layer)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(in_features=dim, hidden_features=int(dim * mlp_ratio),
                              act_layer=act_layer, drop=proj_drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, cu_slens=None) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), cu_slens=cu_slens)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# ---------------------------------------------------------------------------
# VisionTransformer
# ---------------------------------------------------------------------------

class VisionTransformer(nn.Module):
    """Vision Transformer (SigLIP variant with variable-resolution support)."""

    dynamic_img_size: Final[bool]

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: Literal["", "avg", "token", "map"] = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        class_token: bool = True,
        no_embed_class: bool = False,
        reg_tokens: int = 0,
        pre_norm: bool = False,
        fc_norm: Optional[bool] = None,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        weight_init: Literal["skip", "jax", "jax_nlhb", "moco", ""] = "",
        embed_layer: Callable = None,
        norm_layer: Optional[LayerType] = None,
        act_layer: Optional[LayerType] = None,
        strict_img_size: bool = False,
        block_fn: Type[nn.Module] = Block,
        mlp_layer: Type[nn.Module] = None,
        ignore_head: bool = False,
    ) -> None:
        super().__init__()
        if embed_layer is None:
            embed_layer = PatchEmbed
        if mlp_layer is None:
            mlp_layer = Mlp

        assert global_pool in ("", "avg", "token", "map")
        assert class_token or global_pool != "token"
        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False
        self.ignore_head = ignore_head

        embed_args = {}
        if dynamic_img_size:
            embed_args.update(dict(strict_img_size=False, output_fmt="NHWC"))
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, bias=not pre_norm,
            dynamic_img_pad=dynamic_img_pad, strict_img_size=strict_img_size,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(patch_drop_rate, num_prefix_tokens=self.num_prefix_tokens)
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_norm=qk_norm, init_values=init_values,
                    proj_drop=proj_drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                    mlp_layer=mlp_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = None
        self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != "skip":
            self.init_weights(weight_init)

    def init_weights(self, mode: Literal["jax", "jax_nlhb", "moco", ""] = "") -> None:
        trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable

    def rescale_positional_embedding(self, out_size):
        h, w = out_size
        pos_embed_shape = int(self.pos_embed.shape[1] ** 0.5)
        if (h, w) == (pos_embed_shape, pos_embed_shape):
            return self.pos_embed
        rescaled = self.pos_embed.new_zeros(1, h * w, self.pos_embed.shape[2])
        pe_2d = self.pos_embed[0].T.contiguous().view(1, -1, pos_embed_shape, pos_embed_shape)
        if torch.__version__ == "2.0.0":
            dtype = pe_2d.dtype
            pe_2d = F.interpolate(pe_2d.float(), out_size, mode="bilinear", align_corners=False).to(dtype).view(-1, h * w)
        else:
            pe_2d = F.interpolate(pe_2d, out_size, mode="bilinear", align_corners=False).view(-1, h * w)
        rescaled[0] = pe_2d.T.contiguous()
        return rescaled

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.dynamic_img_size:
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.pos_embed, (H, W),
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.pos_drop(x)

    def sample_positional_embedding(self, grid):
        pos_embed_shape = int(self.pos_embed.shape[1] ** 0.5)
        pe_2d = self.pos_embed[0].T.contiguous().view(1, -1, pos_embed_shape, pos_embed_shape)
        n, _ = grid.shape
        grid = grid.view(1, n, 1, 2)
        pos_embedding = F.grid_sample(
            pe_2d.float(), grid.float(), mode="bilinear", align_corners=False, padding_mode="border"
        )
        return pos_embedding.view(1, -1, n).bfloat16().transpose(1, 2)

    def forward_get_embedding_list(self, x_list, use_grid_sampling):
        x_all = []
        image_sizes = []
        slen = []

        if use_grid_sampling:
            image_grids = []
            image_patches = []
            for x in x_list:
                _, _, h, w = x.shape
                pad_h = (self.patch_embed.patch_size[0] - h % self.patch_embed.patch_size[0]) % self.patch_embed.patch_size[0]
                pad_w = (self.patch_embed.patch_size[1] - w % self.patch_embed.patch_size[1]) % self.patch_embed.patch_size[1]
                x = F.pad(x, (0, pad_w, 0, pad_h))
                _, _, h, w = x.shape
                h = h // self.patch_embed.patch_size[0]
                w = w // self.patch_embed.patch_size[1]
                x = x.view(1, 3, h, self.patch_embed.patch_size[0], w, self.patch_embed.patch_size[1]).permute(0, 2, 4, 1, 3, 5).reshape(-1, 3, self.patch_embed.patch_size[0], self.patch_embed.patch_size[1])
                margin_h = 1.0 / h
                margin_w = 1.0 / w
                dh = torch.linspace(-1 + margin_h, 1 - margin_h, steps=h, device=x.device)
                dw = torch.linspace(-1 + margin_w, 1 - margin_w, steps=w, device=x.device)
                meshx, meshy = torch.meshgrid((dh, dw))
                grid = torch.stack((meshy, meshx), 2).reshape(-1, 2)
                image_patches.append(x)
                image_grids.append(grid)
                image_sizes.append((h, w))
                slen.append(h * w)

            image_patches = torch.cat(image_patches, dim=0)
            image_grids = torch.cat(image_grids, dim=0)
            x = self.patch_embed(image_patches)
            pos_embedding = self.sample_positional_embedding(image_grids)
            c = pos_embedding.size(-1)
            x = x.reshape(1, -1, c) + pos_embedding
            x = self.patch_drop(x)
            x = self.norm_pre(x)
        else:
            for x in x_list:
                _, _, h, w = x.shape
                pad_h = (self.patch_embed.patch_size[0] - h % self.patch_embed.patch_size[0]) % self.patch_embed.patch_size[0]
                pad_w = (self.patch_embed.patch_size[1] - w % self.patch_embed.patch_size[1]) % self.patch_embed.patch_size[1]
                x = F.pad(x, (0, pad_w, 0, pad_h))
                _, _, h, w = x.shape
                h = h // self.patch_embed.patch_size[0]
                w = w // self.patch_embed.patch_size[1]
                x = self.patch_embed(x)
                x = x + self.rescale_positional_embedding(out_size=(h, w))
                x = self.patch_drop(x)
                x = self.norm_pre(x)
                x_all.append(x)
                image_sizes.append((h, w))
            slen = [xi.size(1) for xi in x_all]
            x = torch.cat(x_all, dim=1)

        cu_indices = [0]
        for i in slen:
            cu_indices.append(cu_indices[-1] + i)
        cu_slens = torch.tensor(cu_indices, dtype=torch.int32, device=x.device)
        return x, cu_slens, slen, image_sizes

    def forward_features_list(self, x_list, use_grid_sampling=True):
        x, cu_slens, slen, image_sizes = self.forward_get_embedding_list(x_list, use_grid_sampling=use_grid_sampling)
        for idx, blk in enumerate(self.blocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, cu_slens, use_reentrant=True)
            else:
                x = blk(x, cu_slens=cu_slens)
        x_return = x.split(slen, dim=1)
        return x_return, image_sizes

    def forward_features(self, x: torch.Tensor):
        _, _, h, w = x.shape
        h = h // self.patch_embed.patch_size[0]
        w = w // self.patch_embed.patch_size[1]
        x = self.patch_embed(x)
        x = x + self.rescale_positional_embedding(out_size=(h, w))
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x, (h, w)

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.norm(x)
        if self.attn_pool is not None:
            x = self.attn_pool(x)
        elif self.global_pool == "avg":
            x = x[:, self.num_prefix_tokens:].mean(dim=1)
        elif self.global_pool:
            x = x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x, cal_attn_pool=False):
        if isinstance(x, list):
            x, image_sizes = self.forward_features_list(x)
            if not cal_attn_pool:
                return x, image_sizes, None
            cls_tokens = torch.cat([self.forward_head(cur_x) for cur_x in x], dim=0)
            return x, image_sizes, cls_tokens
        else:
            x, image_sizes = self.forward_features(x)
            if not cal_attn_pool:
                return x, image_sizes, None
            return x, image_sizes, self.forward_head(x)


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

@dataclass
class SigLIPVisionCfg:
    width: int = 1152
    layers: Union[Tuple[int, int, int, int], int] = 27
    heads: int = 16
    patch_size: int = 14
    image_size: Union[Tuple[int, int], int] = 336
    global_pool: str = "map"
    mlp_ratio: float = 3.7362
    class_token: bool = False
    num_classes: int = 0
    use_checkpoint: bool = False


SigLIP_MODEL_CONFIG = {
    "siglip_so400m_patch14_384": {
        "image_size": 384, "patch_size": 14, "width": 1152, "layers": 27,
        "heads": 16, "mlp_ratio": 3.7362, "global_pool": "map", "use_checkpoint": False,
    },
    "siglip_so400m_patch16_384": {
        "image_size": 384, "patch_size": 16, "width": 1152, "layers": 27,
        "heads": 16, "mlp_ratio": 3.7362, "global_pool": "map", "use_checkpoint": False,
    },
    "siglip_so400m_patch14_224": {
        "image_size": 224, "patch_size": 14, "width": 1152, "layers": 27,
        "heads": 16, "mlp_ratio": 3.7362, "global_pool": "map", "use_checkpoint": False,
    },
    "siglip_large_patch16_384": {
        "image_size": 384, "patch_size": 16, "width": 1024, "layers": 24,
        "heads": 16, "mlp_ratio": 4, "global_pool": "map", "use_checkpoint": False,
    },
    "siglip2_giant_patch16_384": {
        "image_size": 384, "patch_size": 16, "width": 1536, "layers": 40,
        "heads": 16, "mlp_ratio": 4, "global_pool": "map", "use_checkpoint": False,
    },
    "siglip2_so400m_patch16_384": {
        "image_size": 384, "patch_size": 16, "width": 1152, "layers": 27,
        "heads": 16, "mlp_ratio": 3.7362, "global_pool": "map", "use_checkpoint": False,
    },
}


def create_siglip_vit(
    model_name: str = "siglip_so400m_patch14_384",
    image_size: int = 384,
    select_layer: int = -1,
    ckpt_path: str = "",
    teacher: bool = False,
    gradient_checkpointing: bool = False,
    **kwargs,
) -> VisionTransformer:
    assert model_name in SigLIP_MODEL_CONFIG, f"model name should be in {list(SigLIP_MODEL_CONFIG.keys())}"
    vision_cfg = SigLIPVisionCfg(**SigLIP_MODEL_CONFIG[model_name])

    if select_layer <= 0:
        layers = min(vision_cfg.layers, vision_cfg.layers + select_layer + 1)
    else:
        layers = min(vision_cfg.layers, select_layer)

    model = VisionTransformer(
        img_size=2048, patch_size=vision_cfg.patch_size, embed_dim=vision_cfg.width,
        depth=layers, num_heads=vision_cfg.heads, mlp_ratio=vision_cfg.mlp_ratio,
        class_token=vision_cfg.class_token, global_pool=vision_cfg.global_pool,
        dynamic_img_pad=False, strict_img_size=teacher,
        ignore_head=kwargs.get("ignore_head", False),
        weight_init=kwargs.get("weight_init", "skip"), num_classes=0,
    )

    if ckpt_path:
        state_dict = torch.load(ckpt_path, map_location="cpu")
        new_state_dict = {}
        if ckpt_path.endswith(".pth"):
            for k, v in state_dict.items():
                prefix = "base_model.model.model.vision_tower.vision_tower."
                if k.startswith(prefix):
                    new_state_dict[k[len(prefix):]] = v
        else:
            for k, v in state_dict.items():
                if k.startswith("visual.trunk."):
                    new_state_dict[k[13:]] = v

        if not teacher:
            model.pos_embed = nn.Parameter(
                _resize_pos_embed(model.pos_embed, new_state_dict.get("pos_embed"), target_size=128)
            )
        incompatible = model.load_state_dict(new_state_dict, strict=False)
        import logging as _log
        _log.getLogger(__name__).info(f"SigLIP-ViT loaded from {ckpt_path}; incompatible_keys: {incompatible}")

    if gradient_checkpointing:
        model.set_grad_checkpointing(True)
    return model


def _resize_pos_embed(current_embed, new_embed, target_size=128):
    if new_embed is None or new_embed.shape[1] == current_embed.shape[1]:
        return current_embed
    embed_dim = new_embed.shape[-1]
    src_size = int(math.sqrt(new_embed.shape[1]))
    pe = new_embed.reshape(1, src_size, src_size, embed_dim).permute(0, 3, 1, 2)
    pe = F.interpolate(pe, size=(target_size, target_size), mode="bicubic", align_corners=False)
    return pe.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)


# ---------------------------------------------------------------------------
# Projection / pooling
# ---------------------------------------------------------------------------

class NormalizedDwPooler(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.predictor = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x, forward_type="2x"):
        B, H, W, C = x.shape
        if forward_type == "2x":
            new_x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 2, 4, 5).reshape(B, H // 2, W // 2, 4, C)
            pooled_x = new_x.mean(-2, keepdim=True).expand(-1, -1, -1, 4, -1)
        elif forward_type == "1x":
            new_x = x.reshape(B, H, W, 1, C)
            fused_x = torch.cat([new_x, new_x], dim=-1)
            score = self.predictor(fused_x)
            return (new_x * F.softmax(score, dim=-2)).sum(dim=-2)
        elif forward_type == "4x":
            new_x = x.reshape(B, H // 4, 4, W // 4, 4, C).permute(0, 1, 3, 2, 4, 5).reshape(B, H // 4, W // 4, 16, C)
            pooled_x = new_x.mean(-2, keepdim=True).expand(-1, -1, -1, 16, -1)
        else:
            raise ValueError(f"Unknown forward_type: {forward_type}")
        fused_x = torch.cat([new_x, pooled_x], dim=-1)
        score = self.predictor(fused_x)
        return (new_x * F.softmax(score, dim=-2)).sum(dim=-2)


class OryxMLPv2(nn.Module):
    def __init__(self, in_channels, out_channels, twoview=False):
        super().__init__()
        self.proj1 = nn.Linear(in_channels, out_channels)
        self.proj2 = nn.Linear(out_channels, out_channels)
        self.act = nn.GELU()
        self.pooler = NormalizedDwPooler(out_channels)
        self.out_channels = out_channels
        embed_std = 1 / math.sqrt(out_channels)
        if twoview:
            self.image_sep = nn.Parameter(torch.randn(out_channels) * embed_std)

    def _forward_list(self, x, size):
        split_lens = [h // 2 * w // 2 for h, w in size]
        dtype = x[0].dtype
        all_x = []
        for i, (h, w) in enumerate(size):
            now_x = x[i].reshape(1, h // 2, 2, w // 2, 2, -1).permute(0, 1, 3, 2, 4, 5).reshape(h // 2 * w // 2, 2, 2, -1)
            all_x.append(now_x)
        x = torch.cat(all_x, dim=0)
        x = self.proj1(x)
        x = self.pooler(x, forward_type="2x")
        x = self.act(x)
        x = self.proj2(x)
        c = x.shape[-1]
        x = torch.split(x, split_lens, dim=0)
        xs = []
        for i, (h, w) in enumerate(size):
            now_x = x[i].reshape(1, h // 2, w // 2, -1)
            now_x = now_x.reshape(1, -1, c)
            xs.append(now_x)
        return xs

    def forward(self, x, size=(16, 16), x2=None, size2=(16, 16)):
        if isinstance(x, list):
            xs = self._forward_list(x, size)
            if x2 is not None:
                xs2 = self._forward_list(x2, size2)
                dtype = xs[0].dtype
                sep = self.image_sep.reshape(1, 1, -1).expand(1, 1, self.out_channels).to(dtype)
                xs = [torch.cat([xi, sep, x2i], dim=1) for xi, x2i in zip(xs, xs2)]
            return xs
        else:
            h, w = size
            x = x.reshape(x.shape[0], h, w, -1)
            x = self.proj1(x)
            x = self.pooler(x, forward_type="2x")
            x = self.act(x)
            x = self.proj2(x)
            b, h, w, c = x.shape
            x = x.reshape(b, -1, c)
            return x


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

class SigLIPViTAnysizeWrapper(nn.Module):
    """
    Any-resolution SigLIP ViT wrapper.
    Loads the vision tower from a pre-configured checkpoint (weights are bundled
    with the model directory) and projects features to the language model dimension.
    """

    def __init__(self, vision_tower: str, delay_load: bool = False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.merger = OryxMLPv2(in_channels=1152, out_channels=2048, twoview=False)
        self.select_layer = -1
        self.load_model()

    def load_model(self, device_map=None):
        # Image preprocessing (normalization with mean/std=[0.5,0.5,0.5]) is handled
        # by HunYuanVLMoTProcessor before the model forward pass.
        self.vision_tower = create_siglip_vit(
            ckpt_path=None, model_name="siglip2_so400m_patch16_384",
            gradient_checkpointing=VIT_WITH_GRAD,
        )
        if not VIT_WITH_GRAD:
            for p in self.vision_tower.parameters():
                p.requires_grad = False
            self.vision_tower.eval()
        self.is_loaded = True

    def train(self, mode=True):
        self.training = mode
        if self.is_loaded and not VIT_WITH_GRAD:
            self.vision_tower.eval()

    def _forward_func(self, images, cal_attn_pool=False):
        if isinstance(images, list):
            if FIX_SIZE:
                xs = [x.to(self.dtype) for x in images]
                xs = torch.cat([F.interpolate(x, size=(384, 384), mode="bilinear", align_corners=False) for x in xs], dim=0)
                image_features, img_size, cls_token = self.vision_tower(xs, cal_attn_pool=cal_attn_pool)
                image_features = torch.split(image_features, 1, dim=0)
                img_size = [img_size] * len(images)
            else:
                xs = [x.to(self.dtype) for x in images]
                image_features, img_size, cls_token = self.vision_tower(xs, cal_attn_pool=cal_attn_pool)
        else:
            image_forward_outs, img_size, cls_token = self.vision_tower(images.to(self.dtype), cal_attn_pool=cal_attn_pool)
            image_features = image_forward_outs.to(images.dtype)
        return image_features, img_size, cls_token

    def forward(self, images, cal_attn_pool=False):
        if VIT_WITH_GRAD:
            image_features, img_size, cls_token = self._forward_func(images, cal_attn_pool=cal_attn_pool)
        else:
            with torch.no_grad():
                image_features, img_size, cls_token = self._forward_func(images, cal_attn_pool=cal_attn_pool)

        if isinstance(images, list):
            image_features = [self.merger(x, s).squeeze(0) for x, s in zip(image_features, img_size)]
        else:
            image_features = self.merger(image_features, img_size)
            C = image_features.shape[-1]
            image_features = [image_features.reshape(-1, C)]

        return image_features

    @property
    def dtype(self):
        return self.vision_tower.pos_embed.dtype

    @property
    def device(self):
        return self.vision_tower.pos_embed.device

    @property
    def hidden_size(self):
        return 1152

    @property
    def config(self):
        return type("SigLIPConfigWrapper", (), {"patch_size": 16})()
