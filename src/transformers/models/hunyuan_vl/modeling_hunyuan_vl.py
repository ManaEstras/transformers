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
"""
Unified HunYuanVL model supporting both Dense and MoE text models.
"""

from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...integrations import use_kernel_forward_from_hub
from ...masking_utils import create_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.deprecation import deprecate_kwarg
from .configuration_hunyuan_vl import HunYuanVLConfig, HunYuanVLVisionConfig


# ============================================================================
# Common Utilities
# ============================================================================

@use_kernel_forward_from_hub("RMSNorm")
class HunYuanVLRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        HunYuanVLRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    is_full_attention: bool = False,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if attention_mask is not None:
        if is_full_attention:
            # Vision Full Attention: use complete mask without truncation
            attn_weights = attn_weights + attention_mask
        else:
            # Non-Full Attention: keep original truncation logic
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_xdrope(q, k, cos, sin, position_ids, xdrope_section, output_size=None):
    """Applies XD Rotary Position Embedding to the query and key tensors."""
    x_dim = len(xdrope_section)
    cos = cos[position_ids, ...].permute(0, 2, 1, 3).reshape(output_size[0], output_size[2], x_dim, -1).contiguous()
    sin = sin[position_ids, ...].permute(0, 2, 1, 3).reshape(output_size[0], output_size[2], x_dim, -1).contiguous()

    xdrope_section = xdrope_section * 2

    assert sum(xdrope_section) == cos.shape[-1], "Illegal partition for xd rope"
    cos = torch.cat([m[:, :, i % x_dim, :] for i, m in enumerate(cos.split(xdrope_section, dim=-1))], dim=-1)
    sin = torch.cat([m[:, :, i % x_dim, :] for i, m in enumerate(sin.split(xdrope_section, dim=-1))], dim=-1)

    cos = cos.view(output_size[0], 1, output_size[2], -1)
    sin = sin.view(output_size[0], 1, output_size[2], -1)

    origin_dtype = q.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.float(), sin.float()
    q_out, k_out = (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

    return q_out.to(origin_dtype), k_out.to(origin_dtype)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
):
    """Applies Rotary Position Embedding to the query and key tensors."""
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    else:
        cos = cos.unsqueeze(0).unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(0).unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ============================================================================
# Vision Components (Shared between Dense and MoE)
# ============================================================================

class HunYuanVLVisionMLP(nn.Module):
    def __init__(self, config: HunYuanVLVisionConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]
        self.dense_h_to_4h = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.dense_4h_to_h = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)

    def forward(self, x):
        intermediate = self.dense_h_to_4h(x)
        intermediate = self.act_fn(intermediate)
        output = self.dense_4h_to_h(intermediate)
        return output


class HunYuanVLVisionPatchEmbed(nn.Module):
    def __init__(self, config: HunYuanVLVisionConfig):
        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels
        self.spatial_merge_size = config.spatial_merge_size
        self.interpolate_mode = config.interpolate_mode

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

        self.max_num_patches = (config.max_image_size // self.patch_size) ** 2
        self.num_positions = self.max_num_patches + 1
        self.position_edge = int(self.num_positions**0.5)
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

        self.patch_pos_embed = None

    def forward(self, pixel_values: torch.Tensor, grid_thw: list[list[int]]) -> torch.Tensor:
        num_patches, hidden_size = pixel_values.shape
        pixel_values = pixel_values.reshape(num_patches, self.num_channels, self.patch_size, self.patch_size)

        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.squeeze(-1).squeeze(-1).unsqueeze(0)

        if self.patch_pos_embed is None:
            patch_pos_shape = (1, self.position_edge, self.position_edge, self.embed_dim)
            if self.position_embedding.weight.device.type == 'meta':
                position_indices = torch.arange(1, self.num_positions, device=patch_embeds.device)
                position_embeddings = self.position_embedding(position_indices)
            else:
                position_embeddings = self.position_embedding.weight[1:, :]
            
            self.patch_pos_embed = (
                position_embeddings.reshape(patch_pos_shape).permute(0, 3, 1, 2).float()
            )

        patch_pos_embed_list = []
        for grid in grid_thw:
            t, h0, w0 = grid
            # Convert to values for interpolation scale calculation
            t_val = t.item() if hasattr(t, 'item') else t
            h_val = h0.item() if hasattr(h0, 'item') else h0
            w_val = w0.item() if hasattr(w0, 'item') else w0
            
            h_float, w_float = h_val + 0.1, w_val + 0.1
            patch_pos_embed = nn.functional.interpolate(
                self.patch_pos_embed,
                scale_factor=((h_float / self.position_edge), (w_float / self.position_edge)),
                mode=self.interpolate_mode,
                align_corners=False,
            )

            # For a single frame (image), shape is [1, embed_dim, h, w]
            # Need to flatten to [1, h*w, embed_dim]
            single_frame_pos_embed = (
                patch_pos_embed.reshape(self.embed_dim, -1).transpose(0, 1).unsqueeze(0).to(patch_embeds.dtype)
            )
            
            # For video (t > 1), repeat the position embedding for each frame
            # But note: now we process frame by frame, so t should always be 1 here
            if t_val > 1:
                # This branch should not be hit when processing frame by frame
                single_frame_pos_embed = single_frame_pos_embed.repeat(1, t_val, 1)
            
            patch_pos_embed_list.append(single_frame_pos_embed)

        patch_pos_embed = torch.cat(patch_pos_embed_list, dim=1)
        embeddings = patch_embeds + patch_pos_embed

        return embeddings


class HunYuanVLVisionPatchMerger(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        spatial_merge_size,
        rms_norm_eps,
        **kwargs,
    ):
        super().__init__()

        embed_std = out_channels**-0.5
        self.spatial_merge_size = spatial_merge_size
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=spatial_merge_size, stride=spatial_merge_size),
            nn.GELU(),
            nn.Conv2d(in_channels * 2, in_channels * 4, kernel_size=1),
        )
        self.mlp = nn.Linear(in_channels * 4, out_channels)
        self.image_newline = nn.Parameter(torch.randn(in_channels * 4) * embed_std)
        self.image_begin = nn.Parameter(torch.randn(out_channels) * embed_std)
        self.image_end = nn.Parameter(torch.randn(out_channels) * embed_std)
        self.image_sep = nn.Parameter(torch.randn(out_channels) * embed_std)

        self.before_rms = HunYuanVLRMSNorm(in_channels, eps=rms_norm_eps)
        self.after_rms = HunYuanVLRMSNorm(out_channels, eps=rms_norm_eps)

    def forward(self, x, size=(16, 16)):
        x = self.before_rms(x)
        h, w = size
        dtype = x.dtype
        x = x.permute(0, 2, 1).reshape(x.shape[0], -1, int(h.item()), int(w.item()))
        x = self.proj(x)
        b, c, h, w = x.shape
        x = torch.cat(
            [x, self.image_newline.reshape(1, c, 1, 1).expand(b, c, h, 1).to(dtype, non_blocking=True)], dim=-1
        )
        x = x.reshape(b, c, -1).permute(0, 2, 1)
        x = self.mlp(x)

        begin = self.image_begin.reshape(1, 1, -1).expand(b, 1, x.shape[-1]).to(dtype, non_blocking=True)
        end = self.image_end.reshape(1, 1, -1).expand(b, 1, x.shape[-1]).to(dtype, non_blocking=True)
        x = torch.cat([begin, x, end], dim=1)

        return self.after_rms(x)


class HunYuanVLVisionAttention(nn.Module):
    def __init__(self, config: HunYuanVLVisionConfig):
        super().__init__()
        self.config = config
        self.is_causal = False
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=True)
        self.vision_full_attention: bool = getattr(config, "vision_full_attention", False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            is_full_attention=self.vision_full_attention,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class HunYuanVLVisionBlock(GradientCheckpointingLayer):
    def __init__(self, config: HunYuanVLVisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = HunYuanVLVisionAttention(config)
        self.mlp = HunYuanVLVisionMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class HunYuanVLVisionTransformer(nn.Module):
    config: HunYuanVLVisionConfig
    _no_split_modules = ["HunYuanVLVisionBlock"]

    def __init__(self, config: HunYuanVLVisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = HunYuanVLVisionPatchEmbed(config)
        self.layers = nn.ModuleList([HunYuanVLVisionBlock(config) for _ in range(config.num_hidden_layers)])
        self.perceive = HunYuanVLVisionPatchMerger(
            self.config.hidden_size,
            self.config.text_hidden_size,
            self.config.spatial_merge_size,
            self.config.rms_norm_eps,
        )

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: list[list[int]],
    ) -> torch.Tensor:
        r"""
        grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
            The temporal, height and width dimensions of feature shape for each image.
            For images, t=1. For videos, t=num_frames.
        """
        processed_items = []
        patch_offset = 0
        
        for grid in grid_thw:
            t, h, w = grid
            t_val = t.item() if hasattr(t, 'item') else t
            h_val = h.item() if hasattr(h, 'item') else h
            w_val = w.item() if hasattr(w, 'item') else w
            
            patches_per_frame = h_val * w_val
            total_patches = t_val * patches_per_frame
            
            # Extract patches for this item
            item_patches = x[patch_offset:patch_offset + total_patches]
            patch_offset += total_patches
            
            # For video (t > 1), process each frame separately to avoid OOM
            if t_val > 1:
                frame_outputs = []
                for frame_idx in range(t_val):
                    start_idx = frame_idx * patches_per_frame
                    end_idx = start_idx + patches_per_frame
                    frame_patches = item_patches[start_idx:end_idx]  # [h*w, hidden]
                    
                    # Create a fake grid_thw for single frame
                    single_frame_grid = [[1, h_val, w_val]]
                    
                    # Embed single frame
                    frame_hidden = self.embeddings(frame_patches, single_frame_grid)  # [1, h*w, hidden]
                    
                    # Process through transformer layers
                    for layer in self.layers:
                        frame_hidden = layer(frame_hidden)
                    
                    # Process through perceive
                    frame_processed = self.perceive(frame_hidden, size=(h, w))
                    frame_outputs.append(frame_processed)
                
                # Concatenate all frame outputs
                processed = torch.cat(frame_outputs, dim=1)
            else:
                # For image, process normally
                single_frame_grid = [[1, h_val, w_val]]
                hidden_states = self.embeddings(item_patches, single_frame_grid)
                
                for layer in self.layers:
                    hidden_states = layer(hidden_states)
                
                processed = self.perceive(hidden_states, size=(h, w))
            
            processed_items.append(processed)
        
        hidden_states = torch.cat(processed_items, dim=1)
        return hidden_states


# ============================================================================
# Text Components - Dense MLP
# ============================================================================

class HunYuanVLMLP(nn.Module):
    """Dense MLP layer."""
    def __init__(
        self,
        config: HunYuanVLConfig,
        layer_idx=None,
        is_shared_mlp=False,
        is_moe=False,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]
        self.layer_idx = layer_idx
        
        if is_shared_mlp or is_moe:
            # Use moe_intermediate_size for MoE
            if config.moe_intermediate_size is not None:
                self.intermediate_size = (
                    config.moe_intermediate_size
                    if isinstance(config.moe_intermediate_size, int)
                    else config.moe_intermediate_size[layer_idx % config.num_hidden_layers]
                )

            if is_shared_mlp:
                num_shared_expert = (
                    config.num_shared_expert
                    if isinstance(config.num_shared_expert, int)
                    else config.num_shared_expert[layer_idx % config.num_hidden_layers]
                )
                self.intermediate_size *= num_shared_expert
                
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


# ============================================================================
# Text Components - MoE (Mixture of Experts)
# ============================================================================

class HunYuanVLGate(nn.Module):
    """Gate for MoE routing."""
    def __init__(self, config: HunYuanVLConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        num_experts = config.num_experts if isinstance(config.num_experts, int) else config.num_experts[layer_idx]
        self.wg = nn.Linear(config.hidden_size, num_experts, bias=False, dtype=torch.float32)

    def forward(self, hidden_states):
        bsz, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_size)
        if self.wg.weight.dtype == torch.float32:
            hidden_states = hidden_states.float()
        logits = self.wg(hidden_states)
        return logits


class HunYuanVLMoe(nn.Module):
    """Mixture of Experts layer."""
    def __init__(self, config: HunYuanVLConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_experts = (
            config.num_experts
            if isinstance(config.num_experts, int)
            else config.num_experts[layer_idx % config.num_hidden_layers]
        )
        self.top_k = (
            config.moe_topk
            if isinstance(config.moe_topk, int)
            else config.moe_topk[layer_idx % config.num_hidden_layers]
        )
        self.gate = HunYuanVLGate(config, layer_idx=layer_idx)
        self.experts = nn.ModuleList(
            [
                HunYuanVLMLP(config, layer_idx=layer_idx, is_shared_mlp=False, is_moe=True)
                for _ in range(self.num_experts)
            ]
        )

        self.shared_mlp = HunYuanVLMLP(config, layer_idx=layer_idx, is_shared_mlp=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_mlp = self.shared_mlp(hidden_states)
        router_logits = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_dim)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states + hidden_states_mlp


# ============================================================================
# Text Components - Common (Attention, Decoder Layer)
# ============================================================================

class HunYuanVLRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: HunYuanVLConfig, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type if self.rope_type != "xdrope" else "dynamic"]
        if self.rope_type in ["xdrope", "dynamic"] and config.rope_scaling["alpha"]:
            self.dim = config.head_dim
            base = config.rope_theta * config.rope_scaling.get("alpha") ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.attention_scaling = 1.0
        else:
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq
        self._set_cos_sin_cache(
            seq_len=config.max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).float()
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len: Optional[int] = None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class HunYuanVLAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_causal = True
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        self.query_layernorm = HunYuanVLRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.key_layernorm = HunYuanVLRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rotary_emb = HunYuanVLRotaryEmbedding(config=config)
        self.xdrope_section = config.rope_scaling["xdrope_section"]

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        origin_kv_seq_len = key_states.shape[-2]
        # Get the cache length for THIS layer specifically
        layer_cache_len = 0
        if past_key_values is not None:
            layer_cache_len = past_key_values.get_seq_length(self.layer_idx)
            kv_seq_len += layer_cache_len

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        if self.xdrope_section is not None:
            # Use layer_cache_len to check if this specific layer has cached KV
            if past_key_values is None or layer_cache_len == 0:
                output_size = (
                    query_states.size(0),
                    query_states.size(1),
                    query_states.size(2),
                    key_states.size(2),
                )
                query_states, key_states = apply_rotary_pos_emb_xdrope(
                    query_states, key_states, cos, sin, position_ids, self.xdrope_section, output_size
                )
            else:
                position_ids = (
                    torch.ones(position_ids.shape[0], 1, dtype=torch.long, device=position_ids.device)
                    * layer_cache_len
                )
                cos, sin = cos[-origin_kv_seq_len:, :], sin[-origin_kv_seq_len:, :]
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        else:
            position_ids = torch.ones(
                position_ids.shape[0], 1, dtype=torch.long, device=position_ids.device
            ) * layer_cache_len
            cos, sin = cos[-origin_kv_seq_len:, :], sin[-origin_kv_seq_len:, :]
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        query_states = self.query_layernorm(query_states)
        key_states = self.key_layernorm(key_states)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class HunYuanVLDecoderLayer(GradientCheckpointingLayer):
    """Unified decoder layer supporting both Dense and MoE architectures."""
    def __init__(self, config: HunYuanVLConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = HunYuanVLAttention(config=config, layer_idx=layer_idx)
        self.input_layernorm = HunYuanVLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = HunYuanVLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx
        
        # Determine whether to use MoE or Dense MLP
        # Dense config does NOT have num_experts attribute
        # MoE config HAS num_experts attribute
        is_moe_layer = False
        num_experts = getattr(config, "num_experts", None)
        if num_experts is not None:
            # MoE config - check if this layer should use MoE
            if isinstance(num_experts, list):
                layer_num_experts = num_experts[layer_idx % config.num_hidden_layers]
            else:
                layer_num_experts = num_experts
            
            moe_layer_num_skipped = getattr(config, "moe_layer_num_skipped", 0)
            if layer_num_experts > 1 and layer_idx >= moe_layer_num_skipped:
                is_moe_layer = True
        
        if is_moe_layer:
            self.mlp = HunYuanVLMoe(config, layer_idx=layer_idx)
        else:
            self.mlp = HunYuanVLMLP(config, layer_idx=layer_idx, is_shared_mlp=False, is_moe=False)

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
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# ============================================================================
# Model Classes
# ============================================================================

def modify_causal_mask_for_image_bidirectional(
    causal_mask: torch.Tensor,
    image_mask: torch.Tensor,
    grid_thw: torch.Tensor
) -> torch.Tensor:
    """
    Modify causal mask to enable bidirectional attention within image token regions.
    """
    modified_mask = causal_mask.clone()
    device = modified_mask.device
    image_mask = image_mask[:, :, 0]
    batch_size, seq_len = image_mask.shape
    
    image_token_positions = []
    for b in range(batch_size):
        img_pos = torch.where(image_mask[b])[0]
        image_token_positions.append(img_pos)
    
    if len(image_token_positions[0]) == 0:
        return modified_mask
    
    img_token_counts = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()
    
    img_intervals = []
    current_pos = 0
    for count in img_token_counts:
        if current_pos + count > len(image_token_positions[0]):
            break
        img_interval = image_token_positions[0][current_pos:current_pos+count]
        img_intervals.append(img_interval)
        current_pos += count
    
    for interval in img_intervals:
        if len(interval) == 0:
            continue
        img_start = interval[0]
        img_end = interval[-1] + 1
        img_bidirectional_mask = torch.zeros(
            (img_end - img_start, img_end - img_start),
            device=device
        )
        modified_mask[:, :, img_start:img_end, img_start:img_end] = img_bidirectional_mask

    return modified_mask


@auto_docstring
class HunYuanVLPreTrainedModel(PreTrainedModel):
    config_class = HunYuanVLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HunYuanVLDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": HunYuanVLDecoderLayer,
        "attentions": HunYuanVLAttention,
    }

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


@auto_docstring
class HunYuanVLModel(HunYuanVLPreTrainedModel):
    """
    HunYuanVL Text Model supporting both Dense and MoE architectures.
    """
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [HunYuanVLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = HunYuanVLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.vision_full_attention: bool = getattr(config, "vision_full_attention", False)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        image_mask: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        
        if self.vision_full_attention and past_seen_tokens == 0:
            if image_mask is not None and image_grid_thw is not None:
                causal_mask = modify_causal_mask_for_image_bidirectional(
                    causal_mask, image_mask, image_grid_thw
                )
                
        hidden_states = inputs_embeds
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class HunYuanVLForConditionalGeneration(HunYuanVLPreTrainedModel, GenerationMixin):
    """
    Unified HunYuanVL model for conditional generation supporting both Dense and MoE text models.
    """
    _tied_weights_keys = ["lm_head.weight"]
    config: HunYuanVLConfig

    def __init__(self, config: HunYuanVLConfig):
        super().__init__(config)
        self.model = HunYuanVLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vit = HunYuanVLVisionTransformer(config.vision_config)
        self.config = config
        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @auto_docstring
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
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoProcessor, HunYuanVLForConditionalGeneration
        >>> from PIL import Image
        >>> import torch

        >>> model_name_or_path = "tencent/HunYuanVL"
        >>> processor = AutoProcessor.from_pretrained(model_name_or_path, use_fast=False)
        >>> model = HunYuanVLForConditionalGeneration.from_pretrained(
        ...     model_name_or_path,
        ...     attn_implementation="eager",
        ...     torch_dtype=torch.bfloat16,
        ...     device_map="auto",
        ... )

        >>> img_path = "path/to/your/image.jpg"
        >>> image = Image.open(img_path).convert("RGB")

        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "image", "image": img_path},
        ...             {"type": "text", "text": "Describe this image."},
        ...         ],
        ...     }
        ... ]
        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)

        >>> with torch.no_grad():
        ...     generated_ids = model.generate(**inputs, max_new_tokens=1024)
        >>> generated_ids_trimmed = generated_ids[0][len(inputs["input_ids"][0]):]
        >>> output = processor.decode(generated_ids_trimmed, skip_special_tokens=True)

        >>> print(output)
        ```"""
        # Pop video-related kwargs if present (they're handled separately)
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        
        # Process image and video inputs if provided
        image_mask = None
        orig_input_ids = input_ids  # Save original for video mask
        
        if inputs_embeds is None and input_ids is not None:
            # Process image inputs if provided
            if pixel_values is not None and image_grid_thw is not None:
                inputs_embeds = self.model.embed_tokens(input_ids)
                image_embeds = self.get_image_features(pixel_values, image_grid_thw)
                image_mask = self.get_placeholder_mask(
                    orig_input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                input_ids = None
            
            # Process video inputs if provided
            if pixel_values_videos is not None and video_grid_thw is not None:
                if inputs_embeds is None:
                    inputs_embeds = self.model.embed_tokens(input_ids)
                    input_ids = None
                    
                video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
                
                # Get video token ID from config
                video_token_id = getattr(self.config, 'video_token_id', None)
                if video_token_id is None:
                    video_token_id = self.config.image_token_id
                
                video_mask = self.get_placeholder_mask(
                    orig_input_ids, inputs_embeds=inputs_embeds, image_features=video_embeds,
                    token_id=video_token_id
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
                
                if image_mask is None:
                    image_mask = video_mask
                    image_grid_thw = video_grid_thw

        # Remove image_mask from kwargs if present
        kwargs.pop("image_mask", None)
        
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            image_mask=image_mask,
            image_grid_thw=image_grid_thw,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.
        """
        vit_dtype = next(self.vit.parameters()).dtype
        pixel_values = pixel_values.type(vit_dtype)
        image_embeds = self.vit(pixel_values, grid_thw=image_grid_thw)
        return image_embeds

    def get_video_features(self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model.
        Video frames are treated as images with temporal dimension.
        """
        vit_dtype = next(self.vit.parameters()).dtype
        pixel_values_videos = pixel_values_videos.type(vit_dtype)
        # Video features go through the same ViT as images
        video_embeds = self.vit(pixel_values_videos, grid_thw=video_grid_thw)
        return video_embeds

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        token_id: Optional[int] = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`.
        """
        if token_id is None:
            token_id = self.config.image_token_id
            
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        return special_image_mask

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        imgs: Optional[list[torch.FloatTensor]] = None,
        imgs_pos: Optional[list[int]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[list[int]] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        inputs_embeds = self.model.embed_tokens(input_ids)
        image_mask = None
        
        # Process image inputs
        if self.vit is not None and pixel_values is not None:
            pixel_values = pixel_values.to(self.dtype)
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = image_embeds.to(input_ids.device, non_blocking=True)

            image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds,
                token_id=self.config.image_token_id
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        
        # Process video inputs
        if self.vit is not None and pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.to(self.dtype)
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = video_embeds.to(input_ids.device, non_blocking=True)
            
            # Get video token ID from config
            video_token_id = getattr(self.config, 'video_token_id', None)
            if video_token_id is None:
                # Fallback: use image token for video as well
                video_token_id = self.config.image_token_id
            
            video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=video_embeds,
                token_id=video_token_id
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            
            # Use video mask as image_mask for attention
            if image_mask is None:
                image_mask = video_mask

        # Determine grid_thw for attention mask modification
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw

        return super().generate(
            inputs=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            image_mask=image_mask,
            image_grid_thw=grid_thw,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        image_mask=None,
        image_grid_thw=None,
        token_type_ids=None,
        imgs_pos=None,
        **kwargs,
    ):
        kwargs.pop("imgs", None)
        kwargs.pop("imgs_pos", None)

        if not hasattr(self, "mtp") or self.mtp is None or not hasattr(self, "mtp_index") or self.mtp_index is None:
            inputs = super().prepare_inputs_for_generation(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
            inputs["image_mask"] = image_mask
            inputs["image_grid_thw"] = image_grid_thw if past_key_values.get_seq_length() == 0 else None
            return inputs

        # MTP handling code (same as MoE version)
        inputs = {}
        position_ids = kwargs.get("position_ids")

        if hasattr(self, "mtp_index"):
            offseted_input_ids, offseted_attn_mask = (
                input_ids[:, self.mtp_index :],
                attention_mask[:, self.mtp_index :],
            )
            if self.mtp_index != 0:
                input_ids, attention_mask = offseted_input_ids, offseted_attn_mask

        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.get_seq_length()
                max_cache_length = past_key_values.get_max_cache_shape()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            if past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
                if hasattr(self, "mtp_index") and self.mtp_index >= 1 and past_length == 0:
                    input_ids = input_ids[:, -self.mtp_index :]
                    inputs_embeds = inputs_embeds[:, past_length + self.mtp_index :]

            if (
                max_cache_length is not None
                and max_cache_length > 0
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids")
        if position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        if hasattr(self, "mtp_index") and self.mtp_index is not None:
            if position_ids is not None:
                if self.mtp_index >= 1:
                    for i in range(4):
                        position_ids[:, i, :] = torch.cat(
                            [
                                position_ids[:, i, 1:],
                                (position_ids[:, i, -1] + 1).unsqueeze(1),
                            ],
                            dim=1,
                        )
                else:
                    position_ids = position_ids + self.mtp_index

        if inputs_embeds is not None and past_length == 0:
            inputs = {"inputs_embeds": inputs_embeds}
        if hasattr(self, "mtp_index") and self.mtp_index >= 1 and past_length == 0:
            inputs = {"inputs_embeds": inputs_embeds, "input_ids": input_ids}
        if past_length > 0:
            inputs = {"input_ids": input_ids}

        inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )

        return inputs


@auto_docstring
class HunYuanVLForCausalLM(HunYuanVLPreTrainedModel, GenerationMixin):
    """HunYuanVL model for causal language modeling (text-only)."""
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = HunYuanVLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @can_return_tuple
    @auto_docstring
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
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "HunYuanVLForConditionalGeneration",
    "HunYuanVLForCausalLM",
    "HunYuanVLModel",
    "HunYuanVLPreTrainedModel",
]
