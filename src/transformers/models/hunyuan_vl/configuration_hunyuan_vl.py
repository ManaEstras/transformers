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
Unified HunYuanVL configuration supporting both Dense and MoE text models.
"""

from typing import Union

from transformers.configuration_utils import PretrainedConfig


class HunYuanVLVisionConfig(PretrainedConfig):
    """
    Configuration class for HunYuanVL Vision Transformer.
    
    This configuration is shared between Dense and MoE variants.
    """
    model_type = "hunyuan_vl"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_act="gelu",
        hidden_size=1152,
        intermediate_size=4304,
        interpolate_mode="bilinear",
        rms_norm_eps=1e-05,
        attention_dropout=0.0,
        learnable_mlp_pooling_size=0,
        num_attention_heads=16,
        num_key_value_heads=None,
        num_channels=3,
        num_hidden_layers=27,
        out_hidden_size=4096,
        patch_size=16,
        remove_prenorm=True,
        spatial_merge_size=2,
        spatial_patch_size=1,
        temporal_patch_size=1,
        text_hidden_size=4096,
        anyres_vit_max_image_size=2048,
        cat_extra_token=1,
        img_max_token_num=4096,
        max_image_size=2048,
        max_vit_seq_len=16384,
        min_image_size=512,
        resize_resolution=2048,
        vision_full_attention=False,
        video_max_image_size=768,
        video_min_image_size=256,
        perceive_pre_norm=True,
        perceive_post_norm=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.interpolate_mode = interpolate_mode
        self.learnable_mlp_pooling_size = learnable_mlp_pooling_size
        self.num_attention_heads = num_attention_heads
        if not num_key_value_heads:
            self.num_key_value_heads = num_attention_heads
        else:
            self.num_key_value_heads = num_key_value_heads
        self.num_channels = num_channels
        self.num_hidden_layers = num_hidden_layers
        self.out_hidden_size = out_hidden_size
        self.patch_size = patch_size
        self.remove_prenorm = remove_prenorm
        self.rms_norm_eps= rms_norm_eps
        self.spatial_merge_size = spatial_merge_size
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.text_hidden_size = text_hidden_size

        self.anyres_vit_max_image_size = anyres_vit_max_image_size
        self.attention_dropout = attention_dropout
        self.cat_extra_token = cat_extra_token
        self.img_max_token_num = img_max_token_num
        self.max_image_size = max_image_size
        self.max_vit_seq_len = max_vit_seq_len
        self.min_image_size = min_image_size
        self.resize_resolution = resize_resolution
        self.vision_full_attention = vision_full_attention
        self.video_max_image_size = video_max_image_size
        self.video_min_image_size = video_min_image_size
        self.perceive_pre_norm = perceive_pre_norm
        self.perceive_post_norm = perceive_post_norm


class HunYuanVLTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`HunYuanVLTextModel`]. 
    It supports both Dense and MoE (Mixture of Experts) architectures.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 290943):
            Vocabulary size of the HunYuan model.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations or shared MLP representations.
        moe_intermediate_size (`int` or `List`, *optional*):
            Dimension of the MLP representations in MoE. Use a list for different sizes per layer.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*):
            Number of key_value heads for Grouped Query Attention. Defaults to `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to return the last key/values attentions.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        num_experts (`int` or `List`, *optional*, defaults to 1):
            The number of experts for MoE. If > 1, the model uses MoE architecture.
        num_shared_expert (`int` or `List`, *optional*, defaults to 1):
            The number of shared experts for MoE.
        moe_topk (`int` or `List`, *optional*, defaults to 1):
            The topk value for MoE routing.
        moe_layer_num_skipped (`int`, *optional*, defaults to 0):
            First moe_layer_num_skipped layers do not use MoE.
    """

    model_type = "hunyuan_vl_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=290943,
        org_vocab_size=290943,
        hidden_size=4096,
        intermediate_size: int = 11008,
        moe_intermediate_size: Union[int, list] = None,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        attention_head_dim=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        eod_token_id=3,
        sep_token_id=4,
        text_start_id=7,
        text_end_id=8,
        mask_init_id=13,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
        use_qk_norm=False,
        use_rotary_pos_emb=True,
        cla_share_factor=1,
        norm_type="hf_rms",
        # MoE parameters
        num_experts: Union[int, list] = 1,
        use_mixed_mlp_moe=False,
        num_shared_expert: Union[int, list] = 1,
        moe_topk: Union[int, list] = 1,
        moe_drop_tokens=False,
        moe_random_routing_dropped_token=False,
        moe_layer_num_skipped=0,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        group_limited_greedy=False,
        n_group=None,
        topk_group=None,
        # MLA parameters
        use_mla=False,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        v_head_dim=128,
        qk_nope_head_dim=128,
        # Classification head
        add_classification_head=False,
        class_num=0,
        pool_type="last",
        pad_id=-1,
        head_dim=None,
        # MTP parameters
        num_nextn_predict_layers=1,
        num_predictor_layers=0,
        mtp_loss_factor=0.1,
        mtp_no_bias=True,
        # Vision full attention
        vision_full_attention=False,
        qk_norm=False,
        expert_hidden_dim: Union[int, list] = None,
        first_k_dense_replace=0,
        num_experts_per_tok: Union[int, list] = 1,
        num_shared_experts: Union[int, list] = 1,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.org_vocab_size = org_vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        
        # MoE parameters
        self.num_experts = num_experts
        self.use_mixed_mlp_moe = use_mixed_mlp_moe
        self.num_shared_expert = num_shared_expert
        self.moe_topk = moe_topk
        self.moe_drop_tokens = moe_drop_tokens
        self.moe_random_routing_dropped_token = moe_random_routing_dropped_token
        self.moe_layer_num_skipped = moe_layer_num_skipped
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.group_limited_greedy = group_limited_greedy
        self.n_group = n_group
        self.topk_group = topk_group

        self.head_dim = head_dim
        if attention_head_dim is not None:
            self.attention_head_dim = attention_head_dim
        else:
            self.attention_head_dim = self.hidden_size // num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.attention_dropout = attention_dropout
        self.use_qk_norm = use_qk_norm
        self.use_rotary_pos_emb = use_rotary_pos_emb
        self.norm_type = norm_type
        
        # MLA args
        self.use_mla = use_mla
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim

        # Classification head
        self.add_classification_head = add_classification_head
        self.class_num = class_num
        self.pool_type = pool_type
        self.pad_id = pad_id

        if self.class_num is not None:
            self.dense_list = [self.hidden_size, self.class_num]

        # MTP args
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.num_predictor_layers = num_predictor_layers
        self.mtp_loss_factor = mtp_loss_factor
        self.mtp_no_bias = mtp_no_bias

        # Token IDs
        self.eod_token_id = eod_token_id
        self.text_start_id = text_start_id
        self.text_end_id = text_end_id
        self.mask_init_id = mask_init_id

        self.vision_full_attention = vision_full_attention

        self.qk_norm = qk_norm
        self.expert_hidden_dim = expert_hidden_dim
        self.first_k_dense_replace = first_k_dense_replace
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            sep_token_id=sep_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def is_moe(self) -> bool:
        """
        Check if the model uses MoE architecture.
        
        Dense config does NOT have `num_experts` attribute.
        MoE config HAS `num_experts` attribute (default=1, but typically >1 for actual MoE).
        
        Returns True if:
        - `num_experts` attribute exists AND
        - `num_experts` > 1 (int) or max(num_experts) > 1 (list)
        """
        # Dense config doesn't have num_experts at all
        if not hasattr(self, 'num_experts') or self.num_experts is None:
            return False
        
        # Check actual value
        if isinstance(self.num_experts, int):
            return self.num_experts > 1
        elif isinstance(self.num_experts, list):
            return max(self.num_experts) > 1
        return False


class HunYuanVLConfig(PretrainedConfig):
    """
    Unified configuration class for HunYuanVL supporting both Dense and MoE text models.
    
    This configuration automatically determines whether to use Dense or MoE architecture
    based on the `num_experts` parameter in the text configuration.
    """
    model_type = "hunyuan_vl"
    sub_configs = {"vision_config": HunYuanVLVisionConfig, "text_config": HunYuanVLTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_start_token_id=120118,
        image_end_token_id=120119,
        image_token_id=120120,
        video_start_token_id=120122,
        video_end_token_id=120123,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            # For BC use all kwargs to init `TextConfig`
            self.text_config = self.sub_configs["text_config"](**kwargs)
        else:
            self.text_config = text_config

        self.image_token_id = image_token_id
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id

        self.vision_config.text_hidden_size = self.text_config.hidden_size

        # Attention implementation
        self._attn_implementation = kwargs.pop("attn_implementation", None)

    @property
    def is_moe(self) -> bool:
        """Check if the model uses MoE architecture."""
        return self.text_config.is_moe

    def __setattr__(self, key, value):
        if (
            (text_config := super().__getattribute__("__dict__").get("text_config")) is not None
            and key not in ["dtype", "_attn_implementation_internal"]
            and key in text_config.__dict__
        ):
            setattr(text_config, key, value)
        else:
            super().__setattr__(key, value)

    def __getattribute__(self, key):
        if "text_config" in super().__getattribute__("__dict__") and key not in [
            "_name_or_path",
            "model_type",
            "dtype",
            "_attn_implementation_internal",
        ]:
            text_config = super().__getattribute__("text_config")
            if key in text_config.__dict__:
                return getattr(text_config, key)

        return super().__getattribute__(key)


__all__ = ["HunYuanVLConfig", "HunYuanVLVisionConfig", "HunYuanVLTextConfig"]
