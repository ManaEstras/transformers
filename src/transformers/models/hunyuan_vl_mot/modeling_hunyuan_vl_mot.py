# coding=utf-8
# Copyright (C) 2025 THL A29 Limited, a Tencent company and the HuggingFace Inc. team. All rights reserved.
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

"""HunYuanVL-MoT (Mixture of Tokens) multimodal model."""

import logging
from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, ModelOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import check_model_inputs

from .configuration_hunyuan_vl_mot import HunYuanVLMoTConfig

logger = logging.getLogger(__name__)

IMAGE_TOKEN_ID = 120687
VIDEO_TOKEN_ID = 120688
LATENT_TOKEN_ID = 120690


# ============================================================================
# Flash Attention — required for MoT varlen mechanism
# ============================================================================
try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    _FLASH_ATTN_AVAILABLE = False


def _check_flash_attn():
    if not _FLASH_ATTN_AVAILABLE:
        raise ImportError(
            "flash-attn is required for HunYuanVL-MoT. The Mixture of Tokens mechanism uses "
            "variable-length flash attention with per-modality causal masking.\n"
            "Install it with:  pip install flash-attn>=2.0"
        )


# ============================================================================
# Output dataclass
# ============================================================================

@dataclass
@auto_docstring(custom_intro="Base class for HunYuanVLMoT outputs.")
class HunYuanVLMoTModelOutputWithPast(ModelOutput):
    r"""
    past_key_values (`Cache`, *optional*):
        Pre-computed hidden-states for fast sequential decoding.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


# ============================================================================
# Utility helpers
# Copied from hy_embodied_dev.hunyuan_v1_dense_mot
# ============================================================================

def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def _apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


# ============================================================================
# RMSNorm
# Copied from transformers.models.hunyuan_v1_dense.modeling_hunyuan_v1_dense.HunYuanDenseV1RMSNorm
# ============================================================================

class HunYuanVLMoTRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# ============================================================================
# MLP
# Copied from transformers.models.hunyuan_v1_dense.modeling_hunyuan_v1_dense.HunYuanDenseV1MLP
# ============================================================================

class HunYuanVLMoTMLP(nn.Module):
    def __init__(self, config: HunYuanVLMoTConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ============================================================================
# Rotary Embedding
# Copied from transformers.models.hunyuan_v1_dense.modeling_hunyuan_v1_dense.HunYuanDenseV1RotaryEmbedding
# ============================================================================

class HunYuanVLMoTRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: HunYuanVLMoTConfig, device=None):
        super().__init__()
        self.config = config
        self.rope_type = config.rope_scaling.get("type", "default") if config.rope_scaling else "default"

        if self.rope_type == "dynamic" and config.rope_scaling and config.rope_scaling.get("alpha"):
            alpha = config.rope_scaling["alpha"]
            base = config.rope_theta * alpha ** (config.head_dim / (config.head_dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, config.head_dim, 2).float().to(device) / config.head_dim))
            self.attention_scaling = 1.0
        else:
            if self.rope_type in ROPE_INIT_FUNCTIONS:
                inv_freq, self.attention_scaling = ROPE_INIT_FUNCTIONS[self.rope_type](config, device)
            else:
                base = config.rope_theta
                dim = config.head_dim
                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32).to(device) / dim))
                self.attention_scaling = 1.0

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = inv_freq.clone()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ============================================================================
# MoT (Mixture of Tokens) helpers
# Copied from hy_embodied_dev.hunyuan_v1_dense_mot
# ============================================================================

def _mask_apply(hidden_states: torch.Tensor, mask: torch.Tensor, text_funcs, vision_funcs, out_dims=None):
    """
    Routes tokens to modality-specific functions.
    hidden_states: (B, S, D), mask: (B, S) bool — True = vision token.
    """
    B, S, D = hidden_states.size()
    flat = hidden_states.reshape(B * S, D)
    mask_flat = mask.reshape(B * S).bool()

    if out_dims is None:
        out_flat = [torch.empty_like(flat) for _ in text_funcs]
    else:
        out_flat = [torch.empty(B * S, od, device=flat.device, dtype=flat.dtype) for od in out_dims]

    placeholder = hidden_states[0:1, 0:1, :]
    zero_feature = 0

    text_idx = ~mask_flat
    if text_idx.any():
        hs_t = flat[text_idx]
        for i, fn in enumerate(text_funcs):
            out_flat[i][text_idx] = fn(hs_t)
    else:
        for fn in text_funcs:
            zero_feature = zero_feature + fn(placeholder).mean() * 0

    vis_idx = mask_flat
    if vis_idx.any():
        hs_v = flat[vis_idx]
        for i, fn in enumerate(vision_funcs):
            out_flat[i][vis_idx] = fn(hs_v)
    else:
        for fn in vision_funcs:
            zero_feature = zero_feature + fn(placeholder).mean() * 0

    result = [o.view(B, S, -1) for o in out_flat]
    result[0] = result[0] + zero_feature
    return result


def _flash_attention_forward_mot(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    """
    Varlen flash attention supporting per-modality causal masking.
    Text tokens use causal=True; vision tokens use causal=False.
    Copied from hy_embodied_dev.hunyuan_v1_dense_mot.flash_attention_forward_mot
    """
    _check_flash_attn()

    if kwargs.get("output_attentions", False):
        logger.warning_once("`flash_attention_2` does not support `output_attentions=True`.")

    # Transpose from (B, heads, S, D) → (B, S, heads, D) → squeeze batch
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(module.config, "_pre_quantization_dtype"):
            target_dtype = module.config._pre_quantization_dtype
        else:
            target_dtype = next(m for m in module.modules() if isinstance(m, nn.Linear)).weight.dtype
        query, key, value = query.to(target_dtype), key.to(target_dtype), value.to(target_dtype)

    query = query.squeeze(0)
    key = key.squeeze(0)
    value = value.squeeze(0)

    cu_seqlens_q = torch.tensor([0, query.shape[0]], dtype=torch.int32, device=query.device)
    cu_seqlens_k = torch.tensor([0, key.shape[0]], dtype=torch.int32, device=query.device)
    v_seqlens = attention_mask["v_seqlens"]

    with torch.no_grad():
        max_seqlen_q = max(cu_seqlens_q[i + 1] - cu_seqlens_q[i] for i in range(cu_seqlens_q.size(0) - 1)).item()
        max_seqlen_k = max(cu_seqlens_k[i + 1] - cu_seqlens_k[i] for i in range(cu_seqlens_k.size(0) - 1)).item()

    # Text path: causal attention
    attn_output = flash_attn_varlen_func(
        query, key, value,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        causal=True,
    )

    # Vision path: non-causal attention over visual segments
    if not (v_seqlens == 0).all():
        fake_visual = len(v_seqlens) == 0
        if fake_visual:
            v_seqlens = [(0, 2)]

        visual_query, visual_key, visual_value = [], [], []
        visual_mask = torch.zeros(query.shape[0], dtype=torch.bool, device=query.device)
        cu_v = [0]
        max_v_len = 0
        for s, e in v_seqlens:
            visual_query.append(query[s:e])
            visual_key.append(key[s:e])
            visual_value.append(value[s:e])
            visual_mask[s:e] = True
            cu_v.append(cu_v[-1] + (e - s))
            max_v_len = max(max_v_len, e - s)

        vq = torch.cat(visual_query, dim=0)
        vk = torch.cat(visual_key, dim=0)
        vv = torch.cat(visual_value, dim=0)
        cu_v_seqlens = torch.tensor(cu_v, device=query.device, dtype=torch.int32)

        visual_attn_out = flash_attn_varlen_func(
            vq, vk, vv,
            cu_seqlens_q=cu_v_seqlens, cu_seqlens_k=cu_v_seqlens,
            max_seqlen_q=max_v_len, max_seqlen_k=max_v_len,
            causal=False,
        )
        if fake_visual:
            attn_output = attn_output + visual_attn_out.mean() * 0
        else:
            attn_output = attn_output.clone()
            attn_output[visual_mask] = visual_attn_out

    return attn_output.unsqueeze(0), None


def _modality_mask_to_segments(mask: torch.Tensor) -> torch.Tensor:
    """
    Convert a boolean modality mask to (start, end) visual segment pairs.
    Copied from hy_embodied_dev.hunyuan_v1_dense_mot.modality_mask_to_segments
    """
    if mask.size(1) == 1:
        return torch.tensor([[0, 0]], device=mask.device)
    if mask.dim() == 2:
        if mask.size(0) != 1:
            raise ValueError("Batch size > 1 not supported")
        mask = mask[0]
    mask = mask.to(torch.int64)
    slen = mask.numel()
    is_zero = (mask == 0).to(torch.int64)
    padded = torch.cat([torch.zeros(1, device=mask.device, dtype=torch.int64),
                        is_zero,
                        torch.zeros(1, device=mask.device, dtype=torch.int64)])
    diff = padded[1:] - padded[:-1]
    zero_starts = (diff == 1).nonzero(as_tuple=True)[0]
    zero_ends = (diff == -1).nonzero(as_tuple=True)[0] - 1

    separators = [(s.item(), e.item()) for s, e in zip(zero_starts, zero_ends) if e - s + 1 >= 2]
    segments = []
    seg_start = 0
    for s, e in separators:
        seg_end = s - 1
        if seg_end >= seg_start:
            segments.append([seg_start, seg_end])
        seg_start = e + 1
    if seg_start < slen:
        segments.append([seg_start, slen - 1])

    for i in range(len(segments)):
        segments[i][1] = segments[i][1] + 2  # make end exclusive

    return torch.tensor(segments, device=mask.device) if segments else torch.zeros((0, 2), dtype=torch.long, device=mask.device)


# ============================================================================
# MoT Attention
# Ported from hy_embodied_dev.hunyuan_v1_dense_mot.HunYuanDenseV1MoTAttention
# ============================================================================

class HunYuanVLMoTAttention(nn.Module):
    """Multi-headed attention with per-modality text/vision projection paths."""

    def __init__(self, config: HunYuanVLMoTConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        # Text projections
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.query_layernorm = HunYuanVLMoTRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.key_layernorm = HunYuanVLMoTRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Vision projections (separate path, _v suffix matches checkpoint keys)
        self.q_proj_v = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj_v = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj_v = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj_v = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

    def _mask_apply(self, hidden_states, modality_mask, text_funcs, vision_funcs, out_dims=None):
        if modality_mask is None:
            return [text_funcs[0](hidden_states)]
        return _mask_apply(hidden_states, modality_mask, text_funcs, vision_funcs, out_dims)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        modality_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, key_states, value_states = self._mask_apply(
            hidden_states, modality_mask,
            [self.q_proj, self.k_proj, self.v_proj],
            [self.q_proj_v, self.k_proj_v, self.v_proj_v],
            out_dims=[
                self.config.num_attention_heads * self.head_dim,
                self.config.num_key_value_heads * self.head_dim,
                self.config.num_key_value_heads * self.head_dim,
            ],
        )

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_states = self.query_layernorm(query_states)
        key_states = self.key_layernorm(key_states)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_output, attn_weights = _flash_attention_forward_mot(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self._mask_apply(
            attn_output, modality_mask,
            [self.o_proj], [self.o_proj_v],
        )[0]
        return attn_output, attn_weights


# ============================================================================
# Decoder Layer
# Ported from hy_embodied_dev.hunyuan_v1_dense_mot.HunYuanDenseV1MoTDecoderLayer
# ============================================================================

class HunYuanVLMoTDecoderLayer(GradientCheckpointingLayer):
    """A single transformer decoder layer with per-modality norm and MLP paths."""

    def __init__(self, config: HunYuanVLMoTConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = HunYuanVLMoTAttention(config=config, layer_idx=layer_idx)

        # Text paths
        self.mlp = HunYuanVLMoTMLP(config)
        self.input_layernorm = HunYuanVLMoTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = HunYuanVLMoTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Vision paths (_v suffix matches checkpoint keys)
        self.mlp_v = HunYuanVLMoTMLP(config)
        self.input_layernorm_v = HunYuanVLMoTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_v = HunYuanVLMoTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.layer_idx = layer_idx

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        modality_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = _mask_apply(
            hidden_states, modality_mask,
            [self.input_layernorm], [self.input_layernorm_v],
        )[0]

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            modality_mask=modality_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = _mask_apply(
            hidden_states, modality_mask,
            [lambda x: self.mlp(self.post_attention_layernorm(x))],
            [lambda x: self.mlp_v(self.post_attention_layernorm_v(x))],
        )[0]
        hidden_states = residual + hidden_states
        return hidden_states


# ============================================================================
# Inner language model (text-only decoder with MoT)
# Ported from hy_embodied_dev.hunyuan_v1_dense_mot.HunYuanDenseV1MoTModel / ForCausalLM
# ============================================================================

class _HunYuanVLMoTInnerPreTrainedModel(PreTrainedModel):
    """Internal base for the text decoder that lives inside HunYuanVLMoTModel."""
    config_class = HunYuanVLMoTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HunYuanVLMoTDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    _supports_sdpa = False

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class _HunYuanVLMoTTextModel(_HunYuanVLMoTInnerPreTrainedModel):
    """Pure text decoder (embed_tokens + layers + norm + rotary)."""

    def __init__(self, config: HunYuanVLMoTConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [HunYuanVLMoTDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = HunYuanVLMoTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = HunYuanVLMoTRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        modality_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        text_position_ids = position_ids

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        if modality_mask is None:
            modality_mask = torch.zeros(inputs_embeds.shape[:-1], dtype=torch.bool, device=inputs_embeds.device)

        visual_segs = _modality_mask_to_segments(modality_mask)
        causal_mask = {"cu_seqlens": causal_mask, "v_seqlens": visual_segs}

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, text_position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                modality_mask=modality_mask,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class _HunYuanVLMoTTextForCausalLM(_HunYuanVLMoTInnerPreTrainedModel, GenerationMixin):
    """Text decoder + lm_head for generation (inner component)."""
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: HunYuanVLMoTConfig):
        super().__init__(config)
        self.model = _HunYuanVLMoTTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        modality_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            modality_mask=modality_mask,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        if labels is not None:
            flat_hs = hidden_states.reshape(-1, hidden_states.size(-1))
            flat_labels = labels.reshape(-1)
            valid = flat_labels >= 0
            if valid.sum() == 0:
                flat_hs = flat_hs[:1]
                flat_labels = flat_labels[:1]
            else:
                flat_hs = flat_hs[valid]
                flat_labels = flat_labels[valid]
            logits = self.lm_head(flat_hs)
            loss = self.loss_function(logits=logits, labels=flat_labels, vocab_size=self.config.vocab_size, **kwargs)
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])
            loss = None

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )


# ============================================================================
# Vision Encoder — wraps SigLIPViTAnysizeWrapper from hy_embodied_dev
# ============================================================================

class _HunYuanVLMoTVisionEncoder(nn.Module):
    """
    Thin wrapper that loads the SigLIPViTAnysizeWrapper.
    The vision encoder is loaded from a separate config file bundled with the checkpoint.
    """

    def __init__(self, siglip_config_name: str = "siglip_vit_anyres"):
        super().__init__()
        # Import lazily to avoid hard dependency when SigLIP is not used
        try:
            from transformers.models.hunyuan_vl_mot._siglip_vit_anyres import SigLIPViTAnysizeWrapper
        except ImportError:
            raise ImportError(
                "SigLIPViTAnysizeWrapper not found. "
                "Ensure siglip_vit_anyres.py is available in the model directory."
            )
        self._encoder = SigLIPViTAnysizeWrapper(siglip_config_name)

    @property
    def dtype(self):
        return next(self._encoder.parameters()).dtype

    def forward(self, images):
        return self._encoder(images)


# ============================================================================
# Top-level PreTrainedModel
# ============================================================================

@auto_docstring
class HunYuanVLMoTPreTrainedModel(PreTrainedModel):
    config_class = HunYuanVLMoTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HunYuanVLMoTDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    _supports_sdpa = False
    _can_compile_fullgraph = False

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


# ============================================================================
# HunYuanVLMoTModel — core model combining vision + language
# ============================================================================

@auto_docstring
class HunYuanVLMoTModel(HunYuanVLMoTPreTrainedModel):
    """
    The HunYuanVL-MoT model: SigLIP vision encoder + MoT language decoder.
    Token slots marked with IMAGE_TOKEN_ID / VIDEO_TOKEN_ID are replaced
    by the corresponding visual embeddings before being passed to the decoder.
    """

    base_model_prefix = ""
    config: HunYuanVLMoTConfig
    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False

    def __init__(self, config: HunYuanVLMoTConfig):
        super().__init__(config)
        _check_flash_attn()
        self.language_model = _HunYuanVLMoTTextForCausalLM._from_config(config)
        # Use SigLIPViTAnysizeWrapper directly (no wrapper) so weight keys
        # match the checkpoint: model.visual.vision_tower.*, model.visual.merger.*
        try:
            from transformers.models.hunyuan_vl_mot._siglip_vit_anyres import SigLIPViTAnysizeWrapper
        except ImportError:
            raise ImportError(
                "SigLIPViTAnysizeWrapper not found. "
                "Ensure _siglip_vit_anyres.py is available in the model directory."
            )
        self.visual = SigLIPViTAnysizeWrapper("siglip_vit_anyres")
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    # ------------------------------------------------------------------
    # Vision feature extraction helpers
    # ------------------------------------------------------------------

    def _reshape_pixel_values(self, pixel_values: torch.Tensor, grid_thw: torch.LongTensor):
        """Reshape flat patch tensor back into (T, H*patch, W*patch, C) images."""
        pixel_values = pixel_values.reshape(-1, 3, 16, 16)
        num_patches = grid_thw.prod(dim=-1).tolist()
        patches_list = torch.split(pixel_values, num_patches, dim=0)
        images = []
        for idx, pv in enumerate(patches_list):
            T, H, W = grid_thw[idx][0].item(), grid_thw[idx][1].item(), grid_thw[idx][2].item()
            pv = (pv
                  .reshape(T, H // 2, W // 2, 2, 2, 3, 16, 16)
                  .permute(0, 1, 3, 2, 4, 6, 7, 5)
                  .reshape(T, H, W, 16, 16, 3)
                  .permute(0, 1, 3, 2, 4, 5)
                  .reshape(T, H * 16, W * 16, 3)
                  .permute(0, 3, 1, 2))
            images.append(pv)
        return images

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        pixel_values = pixel_values.type(self.visual.dtype)
        images = self._reshape_pixel_values(pixel_values, image_grid_thw)
        return self.visual(images)

    def get_video_features(self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None):
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        frames = []
        for img_list in self._reshape_pixel_values(pixel_values_videos, video_grid_thw):
            # Each video's T frames are treated as independent images
            frames.extend(img_list.split(1, dim=0))
        return self.visual(frames)

    def get_image_video_features(self, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw, device, dtype):
        recon_images, recon_videos = [], []
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.dtype)
            recon_images = self._reshape_pixel_values(pixel_values, image_grid_thw)
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
            for pv in self._reshape_pixel_values(pixel_values_videos, video_grid_thw):
                recon_videos.extend(pv.split(1, dim=0))

        fake_image = torch.zeros(1, 3, 64, 64, dtype=dtype, device=device)
        all_embeds = self.visual([fake_image] + recon_images + recon_videos)
        split_point = len(recon_images) + 1
        image_embeds = all_embeds[1:split_point]
        video_embeds = all_embeds[split_point:]
        zero_feature = all_embeds[0].mean() * 0
        return image_embeds, video_embeds, zero_feature

    def get_placeholder_mask(self, input_ids, inputs_embeds, image_features=None, video_features=None):
        """Find where IMAGE_TOKEN_ID / VIDEO_TOKEN_ID appear and validate counts."""
        if input_ids is None:
            embed_fn = self.get_input_embeddings()
            img_embed = embed_fn(torch.tensor(IMAGE_TOKEN_ID, dtype=torch.long, device=inputs_embeds.device))
            vid_embed = embed_fn(torch.tensor(VIDEO_TOKEN_ID, dtype=torch.long, device=inputs_embeds.device))
            special_image_mask = (inputs_embeds == img_embed).all(-1)
            special_video_mask = (inputs_embeds == vid_embed).all(-1)
        else:
            special_image_mask = input_ids == IMAGE_TOKEN_ID
            special_video_mask = input_ids == VIDEO_TOKEN_ID

        union_mask = special_image_mask | special_video_mask
        n_img = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(f"Image feature/token count mismatch: tokens={n_img}, features={image_features.shape[0]}")
        n_vid = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(f"Video feature/token count mismatch: tokens={n_vid}, features={video_features.shape[0]}")
        return special_image_mask, special_video_mask, union_mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @auto_docstring
    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> HunYuanVLMoTModelOutputWithPast:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            Temporal, height, and width of each image's patch grid.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            Temporal, height, and width of each video's patch grid.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        union_mask = None
        if not self.training and pixel_values is None and pixel_values_videos is None:
            pass  # inference with KV cache — skip vision encoding
        else:
            image_embeds, video_embeds, zero_feature = self.get_image_video_features(
                pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw,
                inputs_embeds.device, inputs_embeds.dtype,
            )
            if len(image_embeds) > 0:
                image_embeds_cat = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                image_mask, _, union_mask = self.get_placeholder_mask(
                    input_ids, inputs_embeds, image_features=image_embeds_cat
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds_cat)
            if len(video_embeds) > 0:
                video_embeds_cat = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                _, video_mask, union_mask = self.get_placeholder_mask(
                    input_ids, inputs_embeds, video_features=video_embeds_cat
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds_cat)
            inputs_embeds = inputs_embeds + zero_feature

        if union_mask is not None:
            kwargs["modality_mask"] = union_mask
        else:
            kwargs["modality_mask"] = torch.zeros(inputs_embeds.shape[:-1], dtype=torch.bool, device=inputs_embeds.device)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            labels=labels,
            **kwargs,
        )

        return HunYuanVLMoTModelOutputWithPast(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )


# ============================================================================
# HunYuanVLMoTForConditionalGeneration — public generation entry point
# ============================================================================

@auto_docstring
class HunYuanVLMoTForConditionalGeneration(HunYuanVLMoTPreTrainedModel, GenerationMixin):
    """
    HunYuanVL-MoT with a language modelling head for multimodal conditional generation.

    Supports images, videos, and text input with Flash Attention 2 and Mixture of Tokens.
    """

    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = {"model.language_model.lm_head.weight": "model.language_model.model.embed_tokens.weight"}
    accepts_loss_kwargs = False
    config: HunYuanVLMoTConfig

    def __init__(self, config: HunYuanVLMoTConfig):
        super().__init__(config)
        self.model = HunYuanVLMoTModel(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_image_features(self, pixel_values, image_grid_thw=None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    def get_video_features(self, pixel_values_videos, video_grid_thw=None):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> HunYuanVLMoTModelOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modelling loss.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            Temporal, height, and width of each image patch grid.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            Temporal, height, and width of each video patch grid.
        """
        return self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            labels=labels,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            **kwargs,
        )
        # Position IDs are generated from rope_deltas inside forward
        model_inputs["position_ids"] = None
        # Only pass pixel_values on the first forward pass (position 0)
        _cp = model_inputs.get("cache_position")
        if _cp is not None and len(_cp) > 0 and _cp[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["image_grid_thw"] = None
            model_inputs["video_grid_thw"] = None
        return model_inputs


__all__ = [
    "HunYuanVLMoTPreTrainedModel",
    "HunYuanVLMoTModel",
    "HunYuanVLMoTForConditionalGeneration",
]
