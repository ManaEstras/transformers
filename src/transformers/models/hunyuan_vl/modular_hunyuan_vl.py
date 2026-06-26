# Copyright (C) 2026 THL A29 Limited, a Tencent company and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch HunYuanVL model."""

from collections.abc import Callable

import numpy as np
import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...generation import GenerationMixin
from ...image_utils import SizeDict
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling, CausalLMOutputWithPast
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TensorType, TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import maybe_autocast, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..hunyuan_v1_dense.configuration_hunyuan_v1_dense import HunYuanDenseV1Config
from ..hunyuan_v1_dense.modeling_hunyuan_v1_dense import (
    HunYuanDenseV1Attention,
    HunYuanDenseV1DecoderLayer,
    HunYuanDenseV1Model,
    HunYuanDenseV1PreTrainedModel,
    HunYuanDenseV1RotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,  # noqa: F401  - re-exported for downstream tooling
    rotate_half,
)
from ..llama.modeling_llama import LlamaRMSNorm
from ..mllama.modeling_mllama import MllamaVisionAttention
from ..qwen2_vl.image_processing_qwen2_vl import (
    Qwen2VLImageProcessor,
    Qwen2VLImageProcessorKwargs,
    smart_resize,
)
from ..qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
from ..siglip.modeling_siglip import SiglipEncoderLayer, SiglipMLP


@auto_docstring(
    custom_intro="""
    Vision backbone configuration for the dense-only, image-text HunYuanVL open-source variant.
    """,
    checkpoint="tencent/HunyuanOCR",
)
@strict
class HunYuanVLVisionConfig(PreTrainedConfig):
    r"""
    interpolate_mode (`str`, *optional*, defaults to `"bilinear"`):
        Interpolation mode used when resizing learned patch positional embeddings to match the current image grid.
    learnable_mlp_pooling_size (`int`, *optional*, defaults to 0):
        Optional learnable pooling size for the vision tower.
    out_hidden_size (`int`, *optional*, defaults to 4096):
        Output hidden size produced by the vision tower before it is consumed by the text backbone.
    remove_prenorm (`bool`, *optional*, defaults to `True`):
        Whether to remove the pre-normalization behavior used by some internal vision variants.
    resize_resolution (`int`, *optional*, defaults to 2048):
        Reference resolution used when deriving image resizing and tokenization behavior.
    img_max_token_num (`int`, *optional*, defaults to 4096):
        Maximum image token count expected by the vision stack.
    max_image_size (`int`, *optional*, defaults to 2048):
        Maximum supported image size for the current open-source vision configuration.
    min_image_size (`int`, *optional*, defaults to 512):
        Minimum supported image size for the current open-source vision configuration.
    anyres_vit_max_image_size (`int`, *optional*, defaults to 2048):
        Maximum image size supported by the any-resolution vision preprocessing path.
    max_vit_seq_len (`int`, *optional*, defaults to 16384):
        Maximum sequence length produced by the vision transformer.
    text_hidden_size (`int`, *optional*, defaults to 3072):
        Hidden size expected by the text backbone when consuming visual embeddings.

    Example:

    ```python
    >>> from transformers import HunYuanVLVisionConfig
    >>>
    >>> configuration = HunYuanVLVisionConfig()
    >>> configuration.hidden_size
    1152
    ```"""

    model_type = "hunyuan_vl_vision"
    base_config_key = "vision_config"

    hidden_act: str = "gelu"
    hidden_size: int = 1152
    intermediate_size: int = 4304
    interpolate_mode: str = "bilinear"
    rms_norm_eps: float = 1e-5
    learnable_mlp_pooling_size: int = 0
    attention_dropout: float = 0.0
    num_attention_heads: int = 16
    num_key_value_heads: int | None = None
    num_channels: int = 3
    num_hidden_layers: int = 27
    out_hidden_size: int = 4096
    patch_size: int = 16
    remove_prenorm: bool = True
    spatial_merge_size: int = 2
    temporal_patch_size: int = 1
    resize_resolution: int = 2048
    img_max_token_num: int = 4096
    max_image_size: int = 2048
    min_image_size: int = 512
    anyres_vit_max_image_size: int = 2048
    max_vit_seq_len: int = 16384
    text_hidden_size: int = 3072

    def __post_init__(self, **kwargs):
        if not self.num_key_value_heads:
            self.num_key_value_heads = self.num_attention_heads
        super().__post_init__(**kwargs)


@auto_docstring(
    custom_intro="""
    Text backbone configuration for the dense-only, image-text HunYuanVL open-source variant.

    Inherits the standard fields from [`HunYuanDenseV1Config`]. A few legacy field names that some Tencent checkpoints
    persist on disk (`pad_id`, `attention_head_dim`, `org_vocab_size`) are exposed as `attribute_map` aliases that
    transparently redirect to the canonical `pad_token_id` / `head_dim` / `vocab_size` slots. The legacy `rope_scaling`
    / `rope_theta` keys are folded into the standardized `rope_parameters` dict by `convert_rope_params_to_dict`.
    """,
    checkpoint="tencent/HunyuanOCR",
)
@strict
class HunYuanVLTextConfig(HunYuanDenseV1Config):
    r"""
    eod_token_id (`int`, *optional*, defaults to 3):
        Token id representing the end-of-document marker. Inherited from [`HunYuanDenseV1Config`] and re-documented
        here so the auto-generated docstring stays in sync.
    sep_token_id (`int`, *optional*, defaults to 4):
        Token id used as a separator marker by HunYuan tokenizers.
    tie_word_embeddings (`bool`, *optional*, defaults to `True`):
        Whether to tie the input and output word embeddings.
    use_qk_norm (`bool`, *optional*, defaults to `False`):
        Legacy flag preserved for checkpoint compatibility. Has no runtime effect in the open-source variant.
    use_cla (`bool`, *optional*, defaults to `False`):
        Legacy flag preserved for checkpoint compatibility. Has no runtime effect in the open-source variant.
    enable_lm_head_fp32 (`bool`, *optional*, defaults to `False`):
        Legacy flag preserved for checkpoint compatibility. Has no runtime effect in the open-source variant.
    """

    model_type = "hunyuan_vl_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    # Legacy Tencent-export config keys -> canonical field names. ``attribute_map`` is a pure key alias, so reading or
    # writing the legacy name transparently redirects to the canonical attribute (see `PreTrainedConfig.__setattr__`).
    attribute_map = {
        "pad_id": "pad_token_id",
        "attention_head_dim": "head_dim",
        "org_vocab_size": "vocab_size",
    }
    # ``xdrope`` carries scaling knobs (alpha/beta/mscale) and a section layout that the generic rope validator does
    # not know about; whitelist them so validation does not raise (the rope type is standardized to ``dynamic``).
    ignore_keys_at_rope_validation = {
        "alpha",
        "beta_fast",
        "beta_slow",
        "mscale",
        "mscale_all_dim",
        "xdrope_section",
    }

    sep_token_id: int | None = 4
    tie_word_embeddings: bool = True
    use_qk_norm: bool = False
    use_cla: bool = False
    enable_lm_head_fp32: bool = False

    def __post_init__(self, **kwargs):
        # Some Tencent checkpoints persist both ``pad_id`` (the real id) and ``pad_token_id: -1`` (a sentinel). Because
        # ``pad_id`` is now an ``attribute_map`` alias of ``pad_token_id`` the two collide, so explicitly prefer the
        # real value when the canonical slot still holds the ``-1`` sentinel.
        pad_id = kwargs.pop("pad_id", None)
        if self.pad_token_id == -1 and pad_id not in (None, -1):
            self.pad_token_id = pad_id

        super().__post_init__(**kwargs)

    def convert_rope_params_to_dict(self, **kwargs):
        # Fold the legacy ``rope_scaling`` / ``rope_theta`` keys into the standardized ``rope_parameters`` dict and
        # rewrite HunYuan's ``xdrope`` rope type to the generic ``dynamic`` type the runtime understands. Always
        # produce a populated ``rope_parameters`` (defaulting to a plain ``default`` rope) so the rotary embedding
        # always has a ``rope_type`` to read, matching the base config behavior.
        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = rope_scaling or self.rope_parameters or {}
        self.rope_parameters = dict(self.rope_parameters)
        rope_type = self.rope_parameters.get("rope_type", self.rope_parameters.get("type", "default"))
        if rope_type == "xdrope":
            rope_type = "dynamic"
        self.rope_parameters["rope_type"] = rope_type
        self.rope_parameters.setdefault("rope_theta", kwargs.pop("rope_theta", self.default_theta))
        self.standardize_rope_params()
        return kwargs

    @property
    def xdrope_num_sections(self) -> int:
        """Number of position-id channels used by xdrope (e.g. 4 for `[text, w, h, t]`)."""
        rope_parameters = getattr(self, "rope_parameters", None) or {}
        xdrope_section = rope_parameters.get("xdrope_section")
        return len(xdrope_section) if xdrope_section else 4


@auto_docstring(
    custom_intro="""
    Top-level configuration for the open-source HunYuanVL integration.

    This configuration describes the dense-only, image-text-only variant used for OCR and document-understanding style
    workloads. It mirrors the `Qwen2_5_VL` / `Qwen3_VL` family layout: the top-level config simply composes a
    [`HunYuanVLTextConfig`] (text backbone) and a [`HunYuanVLVisionConfig`] (vision tower) plus a few token ids that
    delimit image spans in multimodal prompts.
    """,
    checkpoint="tencent/HunyuanOCR",
)
@strict
class HunYuanVLConfig(PreTrainedConfig):
    r"""
    text_config (`HunYuanVLTextConfig` or `dict`, *optional*):
        Configuration of the text backbone. When `None`, default values are used.
    vision_config (`HunYuanVLVisionConfig` or `dict`, *optional*):
        Configuration of the vision tower. When `None`, default values are used.
    image_token_id (`int`, *optional*, defaults to 120120):
        Token id used as the visual placeholder in multimodal prompts.
    im_start_id (`int`, *optional*, defaults to 120118):
        Token id marking the beginning of an image span in multimodal prompts.
    im_end_id (`int`, *optional*, defaults to 120119):
        Token id marking the end of an image span in multimodal prompts.
    im_newline_id (`int`, *optional*, defaults to 120121):
        Token id used for newline-style separators inserted inside serialized image regions.
    tie_word_embeddings (`bool`, *optional*, defaults to `True`):
        Whether to tie the input and output word embeddings.

    Example:

    ```python
    >>> from transformers import HunYuanVLConfig, HunYuanVLForConditionalGeneration
    >>>
    >>> configuration = HunYuanVLConfig()
    >>> model = HunYuanVLForConditionalGeneration(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "hunyuan_vl"
    sub_configs = {"vision_config": HunYuanVLVisionConfig, "text_config": HunYuanVLTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]
    # Legacy Tencent-export config keys -> canonical field names (released config.json uses the longer spellings).
    attribute_map = {
        "image_start_token_id": "im_start_id",
        "image_end_token_id": "im_end_id",
        "image_newline_token_id": "im_newline_id",
    }

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 120120
    im_start_id: int = 120118
    im_end_id: int = 120119
    im_newline_id: int = 120121
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        # When loading legacy "flat" Tencent checkpoints (where text fields live at the top level instead of inside a
        # nested `text_config` block) we fold the recognized text-side keys into the text config payload. This keeps
        # ``HunYuanVLConfig.from_pretrained(...)`` working with both the upstream nested layout and the existing
        # public OCR checkpoints.
        text_kwargs = self._extract_text_kwargs(kwargs)

        if isinstance(self.vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(self.text_config, dict):
            self.text_config = self.sub_configs["text_config"](**{**self.text_config, **text_kwargs})
        elif self.text_config is None:
            self.text_config = self.sub_configs["text_config"](**text_kwargs)

        # Keep the vision tower in sync with the consuming text backbone size.
        self.vision_config.text_hidden_size = self.text_config.hidden_size

        super().__post_init__(**kwargs)

    @classmethod
    def _extract_text_kwargs(cls, kwargs: dict) -> dict:
        """
        Pop and return the subset of ``kwargs`` that should be forwarded to [`HunYuanVLTextConfig`].

        Required to support legacy Tencent checkpoints whose ``config.json`` stores the text-backbone fields at the
        top level instead of inside a nested ``text_config`` block. Besides the canonical text fields, this also
        captures the legacy alias keys (`pad_id` / `attention_head_dim` / `org_vocab_size`) and the legacy rope keys
        (`rope_scaling` / `rope_theta`) so they reach the text config where they are normalized.
        """
        text_cfg = cls.sub_configs["text_config"]
        text_keys = (
            set(text_cfg.__dataclass_fields__)
            | set(getattr(text_cfg, "attribute_map", {}))
            | {"rope_scaling", "rope_theta"}
        )
        return {key: kwargs.pop(key) for key in list(kwargs) if key in text_keys}


class HunYuanVLImageProcessorKwargs(Qwen2VLImageProcessorKwargs):
    r"""
    min_pixels (`int`, *optional*, defaults to `512 * 512`):
        The min pixels of the image to resize the image.
    max_pixels (`int`, *optional*, defaults to `2048 * 2048`):
        The max pixels of the image to resize the image.
    patch_size (`int`, *optional*, defaults to 16):
        The spatial patch size of the vision encoder.
    temporal_patch_size (`int`, *optional*, defaults to 1):
        The temporal patch size of the vision encoder (the open-source variant is image-only).
    merge_size (`int`, *optional*, defaults to 2):
        The merge size of the vision encoder to the LLM encoder.
    """


class HunYuanVLImageProcessorPil(Qwen2VLImageProcessorPil):
    """
    PIL-backend HunYuanVL image processor. Inherits the resize + patchify pipeline from [`Qwen2VLImageProcessorPil`]
    (the patch reshape/transpose is identical); only the patch-grid defaults differ, and
    ``get_number_of_image_patches`` returns the `(grid_h, grid_w)` tuple expected by the HunYuanVL processor.
    """

    size = {"shortest_edge": 512 * 512, "longest_edge": 2048 * 2048}
    patch_size = 16
    temporal_patch_size = 1
    merge_size = 2
    valid_kwargs = HunYuanVLImageProcessorKwargs

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None) -> tuple[int, int]:
        """Return the `(grid_h, grid_w)` patch counts that the processor would produce for a `height x width` image."""
        images_kwargs = images_kwargs or {}
        min_pixels = images_kwargs.get("min_pixels", self.size["shortest_edge"])
        max_pixels = images_kwargs.get("max_pixels", self.size["longest_edge"])
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)

        factor = patch_size * merge_size
        resized_height, resized_width = smart_resize(
            height, width, factor, min_pixels=min_pixels, max_pixels=max_pixels
        )
        return resized_height // patch_size, resized_width // patch_size


class HunYuanVLImageProcessor(Qwen2VLImageProcessor):
    """
    Torchvision-backend HunYuanVL image processor. Inherits the public API from [`Qwen2VLImageProcessor`]
    (`TorchvisionBackend`) and overrides ``_preprocess`` because the open-source HunYuanVL vision stack is image-only
    (`temporal_patch_size = 1`) and flattens patches without the temporal expansion used by Qwen2-VL.
    """

    size = {"shortest_edge": 512 * 512, "longest_edge": 2048 * 2048}
    patch_size = 16
    temporal_patch_size = 1
    merge_size = 2
    valid_kwargs = HunYuanVLImageProcessorKwargs

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        from ...image_processing_utils_fast import group_images_by_shape, reorder_images

        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            height, width = stacked_images.shape[-2:]
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=size.shortest_edge,
                    max_pixels=size.longest_edge,
                )
                stacked_images = self.resize(
                    image=stacked_images,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                )
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        processed_grids = {}
        for shape, stacked_images in grouped_images.items():
            resized_height, resized_width = stacked_images.shape[-2:]
            patches = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            batch_size, channel = patches.shape[:2]
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
            patches = patches.reshape(
                batch_size,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            # Group grid and patch dimensions, then flatten. The open-source variant is image-only
            # (``temporal_patch_size == 1``), so unlike Qwen2-VL there is no temporal expansion.
            patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
            flatten_patches = patches.reshape(
                batch_size,
                grid_h * grid_w,
                channel * patch_size * patch_size,
            )
            processed_images_grouped[shape] = flatten_patches
            processed_grids[shape] = [[1, grid_h, grid_w]] * batch_size

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_grids_ordered = reorder_images(processed_grids, grouped_images_index)
        pixel_values = torch.cat(processed_images, dim=0)
        image_grid_thw = torch.tensor(processed_grids_ordered, dtype=torch.long)

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}, tensor_type=return_tensors
        )

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None) -> tuple[int, int]:
        """Return the `(grid_h, grid_w)` patch counts that the processor would produce for a `height x width` image."""
        images_kwargs = images_kwargs or {}
        min_pixels = images_kwargs.get("min_pixels", self.size["shortest_edge"])
        max_pixels = images_kwargs.get("max_pixels", self.size["longest_edge"])
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)

        factor = patch_size * merge_size
        resized_height, resized_width = smart_resize(
            height, width, factor, min_pixels=min_pixels, max_pixels=max_pixels
        )
        return resized_height // patch_size, resized_width // patch_size


def _normalize_xdrope_section(xdrope_section, head_dim: int) -> list[int] | None:
    """
    Normalize the ``xdrope_section`` config field to a list of integer half-head sizes.

    Tencent checkpoints store absolute half-head partition sizes (e.g. ``[16, 16, 16, 16]`` for ``head_dim=128``,
    i.e. four channels whose doubled sum equals ``head_dim``).
    """
    if xdrope_section is None:
        return None
    return [int(section) for section in xdrope_section]


def apply_rotary_pos_emb_xdrope(q, k, cos, sin, xdrope_section, unsqueeze_dim=1):
    """
    Apply HunYuan's multi-channel ("xdrope") rotary embedding to ``q`` and ``k``.

    This follows the same section-selection scheme as Qwen2-VL's `apply_multimodal_rotary_pos_emb`, generalized to an
    arbitrary number of channels. ``cos`` / ``sin`` have shape `(num_channels, batch, seq, head_dim)` and
    ``xdrope_section`` lists the per-channel half-head sizes (so the doubled sum equals ``head_dim``). For each rotary
    sub-band ``i`` we pick the cos/sin from channel ``i % num_channels`` and concatenate, then rotate in fp32.
    """
    num_channels = len(xdrope_section)
    section = [s * 2 for s in xdrope_section]
    cos = torch.cat(
        [m[i % num_channels] for i, m in enumerate(cos.split(section, dim=-1))], dim=-1
    ).unsqueeze(unsqueeze_dim)
    sin = torch.cat(
        [m[i % num_channels] for i, m in enumerate(sin.split(section, dim=-1))], dim=-1
    ).unsqueeze(unsqueeze_dim)

    origin_dtype = q.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.float(), sin.float()
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out.to(origin_dtype), k_out.to(origin_dtype)


class HunYuanVLRMSNorm(LlamaRMSNorm):
    pass


class HunYuanVLRotaryEmbedding(HunYuanDenseV1RotaryEmbedding):
    """
    Multi-channel rotary embedding for HunYuanVL.

    Identical inverse-frequency initialization to [`HunYuanDenseV1RotaryEmbedding`] (it keeps the DynamicNTKAlpha
    `inv_freq` used by the released checkpoints), but `forward` consumes a multi-channel ``position_ids`` of shape
    `(num_channels, batch, seq)` and returns ``cos`` / ``sin`` of shape `(num_channels, batch, seq, head_dim)`. A 2D
    ``position_ids`` of shape `(batch, seq)` (text-only / decode steps) is broadcast across the channels.
    """

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        if position_ids.dim() == 2:
            position_ids = position_ids[None, :, :].expand(self.config.xdrope_num_sections, -1, -1)

        num_channels = position_ids.shape[0]
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].float().expand(num_channels, position_ids.shape[1], -1, 1).to(x.device)
        )
        position_ids_expanded = position_ids[:, :, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@auto_docstring
class HunYuanVLVisionPreTrainedModel(HunYuanDenseV1PreTrainedModel):
    """Base class for the HunYuanVL vision tower.

    Subclassing [`PreTrainedModel`] (rather than a plain ``nn.Module``) lets the vision tower participate in the
    standard output-recording machinery, so ``output_attentions`` / ``output_hidden_states`` work out of the box.
    """

    config: HunYuanVLVisionConfig
    base_model_prefix = "vit"
    main_input_name = "pixel_values"
    input_modalities = "image"
    _no_split_modules = ["HunYuanVLVisionBlock"]


class HunYuanVLVisionMLP(SiglipMLP):
    """Vision MLP. Identical to [`SiglipMLP`] (`fc1` / `fc2`); released checkpoints use the legacy `dense_h_to_4h` /
    `dense_4h_to_h` names which are remapped to `fc1` / `fc2` in `conversion_mapping.py`."""

    pass


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
        # The first token is the cls token; the remaining tokens form the learnable patch positional grid.
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.LongTensor) -> torch.Tensor:
        num_patches, _ = pixel_values.shape
        pixel_values = pixel_values.reshape(num_patches, self.num_channels, self.patch_size, self.patch_size)

        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.squeeze(-1).squeeze(-1).unsqueeze(0)

        # Reshape the learnable patch positional grid (dropping the leading cls token) so it can be interpolated to
        # each image's grid size. Recomputed every call (cheaply) to avoid caching a tensor on the module, which keeps
        # the module export/compile friendly.
        patch_pos_shape = (1, self.position_edge, self.position_edge, self.embed_dim)
        base_pos_embed = self.position_embedding.weight[1:, :].reshape(patch_pos_shape).permute(0, 3, 1, 2).float()

        patch_pos_embed_list = []
        for grid in grid_thw:
            _, h0, w0 = grid
            # Interpolate the positional grid to ``(h0, w0)`` patches. Using an explicit output ``size`` (instead of
            # ``scale_factor=(h0/edge).item()``) avoids a ``.item()`` host sync that would break graph capture.
            patch_pos_embed = nn.functional.interpolate(
                base_pos_embed,
                size=(int(h0), int(w0)),
                mode=self.interpolate_mode,
                align_corners=False,
            )
            patch_pos_embed = (
                patch_pos_embed.reshape(self.embed_dim, -1).transpose(0, 1).unsqueeze(0).to(patch_embeds.dtype)
            )
            patch_pos_embed_list.append(patch_pos_embed)

        patch_pos_embed = torch.cat(patch_pos_embed_list, dim=1)
        return patch_embeds + patch_pos_embed


class HunYuanVLVisionPatchMerger(nn.Module):
    def __init__(self, config: HunYuanVLVisionConfig):
        super().__init__()

        in_channels = config.hidden_size
        out_channels = config.text_hidden_size
        spatial_merge_size = config.spatial_merge_size
        rms_norm_eps = config.rms_norm_eps
        embed_std = out_channels**-0.5
        self.spatial_merge_size = spatial_merge_size
        # The original implementation used an ``nn.Sequential`` (``proj.0`` / ``proj.2``); we expose the conv layers as
        # named attributes (the GELU is applied functionally), and the legacy ``proj.0`` / ``proj.2`` checkpoint keys
        # are remapped in `conversion_mapping.py`.
        self.proj_conv1 = nn.Conv2d(
            in_channels, in_channels * 2, kernel_size=spatial_merge_size, stride=spatial_merge_size
        )
        self.proj_conv2 = nn.Conv2d(in_channels * 2, in_channels * 4, kernel_size=1)
        self.mlp = nn.Linear(in_channels * 4, out_channels)
        self.image_newline = nn.Parameter(torch.randn(in_channels * 4) * embed_std)
        self.image_begin = nn.Parameter(torch.randn(out_channels) * embed_std)
        self.image_end = nn.Parameter(torch.randn(out_channels) * embed_std)
        self.image_sep = nn.Parameter(torch.randn(out_channels) * embed_std)

        self.before_rms = HunYuanVLRMSNorm(in_channels, eps=rms_norm_eps)
        self.after_rms = HunYuanVLRMSNorm(out_channels, eps=rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, size: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        hidden_states = self.before_rms(hidden_states)
        h, w = size
        dtype = hidden_states.dtype
        hidden_states = hidden_states.permute(0, 2, 1).reshape(hidden_states.shape[0], -1, int(h), int(w))
        hidden_states = self.proj_conv2(nn.functional.gelu(self.proj_conv1(hidden_states)))
        b, c, h, w = hidden_states.shape
        hidden_states = torch.cat(
            [
                hidden_states,
                self.image_newline.reshape(1, c, 1, 1).expand(b, c, h, 1).to(dtype, non_blocking=True),
            ],
            dim=-1,
        )
        hidden_states = hidden_states.reshape(b, c, -1).permute(0, 2, 1)
        hidden_states = self.mlp(hidden_states)

        begin = self.image_begin.reshape(1, 1, -1).expand(b, 1, hidden_states.shape[-1]).to(dtype, non_blocking=True)
        end = self.image_end.reshape(1, 1, -1).expand(b, 1, hidden_states.shape[-1]).to(dtype, non_blocking=True)
        hidden_states = torch.cat([begin, hidden_states, end], dim=1)

        return self.after_rms(hidden_states)


class HunYuanVLVisionAttention(MllamaVisionAttention):
    """
    Vision self-attention. Inherits the projection naming (`q_proj` / `k_proj` / `v_proj` / `o_proj`) from
    [`MllamaVisionAttention`], but uses biased projections and HunYuanVL's grouped-query head configuration.
    """

    def __init__(self, config: HunYuanVLVisionConfig):
        nn.Module.__init__(self)
        self.config = config
        self.is_causal = False
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

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


class HunYuanVLVisionBlock(SiglipEncoderLayer):
    """
    Vision transformer block. Reuses the pre-norm residual `forward` of [`SiglipEncoderLayer`]; only the layernorm
    attribute names differ (`input_layernorm` / `post_attention_layernorm`), so they are remapped to `layer_norm1` /
    `layer_norm2` in `conversion_mapping.py`. The eps source is HunYuanVL's `rms_norm_eps`.
    """

    def __init__(self, config: HunYuanVLVisionConfig):
        nn.Module.__init__(self)
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = HunYuanVLVisionAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = HunYuanVLVisionMLP(config)


class HunYuanVLVisionTransformer(HunYuanVLVisionPreTrainedModel):
    """
    HunYuanVL vision tower: patch embedding -> transformer blocks -> per-image patch merger.

    Inputs are flat per-patch pixel tensors plus an ``image_grid_thw`` tensor describing the spatial layout of every
    image in the batch. The forward returns a [`BaseModelOutputWithPooling`] whose ``last_hidden_state`` holds the
    concatenation of merged image embeddings, ready to be scattered into the language-model embedding stream. The
    ``@capture_outputs`` decorator together with ``_can_record_outputs`` collects per-block attentions / hidden states
    when ``output_attentions`` / ``output_hidden_states`` are requested.
    """

    config: HunYuanVLVisionConfig
    _can_record_outputs = {
        "hidden_states": HunYuanVLVisionBlock,
        "attentions": HunYuanVLVisionAttention,
    }

    def __init__(self, config: HunYuanVLVisionConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = HunYuanVLVisionPatchEmbed(config)
        self.layers = nn.ModuleList([HunYuanVLVisionBlock(config) for _ in range(config.num_hidden_layers)])
        self.perceive = HunYuanVLVisionPatchMerger(config)
        self.post_init()

    @capture_outputs
    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.LongTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.Tensor` of shape `(num_patches, num_channels * patch_size * patch_size)`):
            Flat per-patch pixel features produced by the image processor.
        grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
            The temporal, height and width dimensions for each image. Each row contains `[t, h, w]` patch counts.
        """
        hidden_states = self.embeddings(pixel_values, grid_thw)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=None)

        split_sizes = grid_thw.prod(dim=-1).tolist()
        split_items = torch.split(hidden_states, split_sizes, dim=1)

        processed_items = []
        for grid, item in zip(grid_thw, split_items):
            _, h, w = grid
            processed_items.append(self.perceive(item, size=(h, w)))

        image_embeds = torch.cat(processed_items, dim=1)
        return BaseModelOutputWithPooling(last_hidden_state=image_embeds)


class HunYuanVLDenseV1Attention(HunYuanDenseV1Attention):
    """
    HunYuan dense attention with optional xdrope rotary embeddings.

    On prefill, when ``rope_parameters['xdrope_section']`` is set and ``position_ids`` carries a 4-channel
    multimodal layout, queries and keys are rotated through [`apply_rotary_pos_emb_xdrope`]. Otherwise the layer
    behaves exactly like [`HunYuanDenseV1Attention`].
    """

    def __init__(self, config: HunYuanVLTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        self.xdrope_section = _normalize_xdrope_section(rope_parameters.get("xdrope_section"), self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # ``position_embeddings`` are the multi-channel ``(cos, sin)`` of shape `(num_channels, bs, seq, head_dim)`,
        # computed once in the model forward. When xdrope is configured we apply the per-channel section selection;
        # otherwise we collapse to the text channel for plain rotary embedding.
        cos, sin = position_embeddings
        if self.xdrope_section is not None:
            query_states, key_states = apply_rotary_pos_emb_xdrope(
                query_states, key_states, cos, sin, self.xdrope_section
            )
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos[0], sin[0])

        query_states = self.query_layernorm(query_states)
        key_states = self.key_layernorm(key_states)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
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


class HunYuanVLDenseV1DecoderLayer(HunYuanDenseV1DecoderLayer):
    def __init__(self, config: HunYuanVLTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = HunYuanVLDenseV1Attention(config=config, layer_idx=layer_idx)


@auto_docstring
class HunYuanVLPreTrainedModel(HunYuanDenseV1PreTrainedModel):
    config: HunYuanVLConfig
    input_modalities = ("image", "text")
    _no_split_modules = ["HunYuanVLDenseV1DecoderLayer", "HunYuanVLVisionBlock"]
    _can_record_outputs = {
        "hidden_states": HunYuanVLDenseV1DecoderLayer,
        "attentions": HunYuanVLDenseV1Attention,
    }


class HunYuanVLTextModel(HunYuanDenseV1Model):
    """Dense text backbone used inside [`HunYuanVLModel`]."""

    config: HunYuanVLTextConfig

    def __init__(self, config: HunYuanVLTextConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [HunYuanVLDenseV1DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.rotary_emb = HunYuanVLRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            ).unsqueeze(0)

        # ``position_ids`` may carry a multi-channel xdrope layout. The processor emits `(batch, num_channels, seq)`;
        # normalize it to `(num_channels, batch, seq)` for the rotary embedding, and use the text channel for masking.
        if position_ids.dim() == 3:
            position_ids = position_ids.permute(1, 0, 2)
            causal_position_ids = position_ids[0]
        else:
            causal_position_ids = position_ids

        causal_mask = create_causal_mask(
            self.config,
            inputs_embeds,
            attention_mask,
            past_key_values=past_key_values,
            position_ids=causal_position_ids,
        )

        hidden_states = inputs_embeds
        # Compute the rotary embeddings once and share them across every decoder layer.
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring(
    custom_intro="""
    The bare HunYuanVL model that composes the vision tower and the dense text backbone, returning raw hidden states
    (no language-modeling head). Use [`HunYuanVLForConditionalGeneration`] for text generation.
    """
)
class HunYuanVLModel(HunYuanVLPreTrainedModel):
    def __init__(self, config: HunYuanVLConfig):
        super().__init__(config)
        self.model = HunYuanVLTextModel(config.text_config)
        self.vit = HunYuanVLVisionTransformer(config.vision_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        Encode images into continuous embeddings that can be scattered into the language-model token stream.

        pixel_values (`torch.FloatTensor`):
            Flat per-patch pixel features produced by the image processor.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.

        Returns a [`BaseModelOutputWithPooling`]; the merged image embeddings are in ``last_hidden_state``.
        """
        vit_dtype = next(self.vit.parameters()).dtype
        pixel_values = pixel_values.to(vit_dtype)
        return self.vit(pixel_values, grid_thw=image_grid_thw, **kwargs)

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor | None,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor | None = None,
    ) -> torch.BoolTensor:
        """
        Compute a boolean mask over ``inputs_embeds`` selecting the positions that hold the visual placeholder
        token, and validate that the placeholder count matches the number of provided image features.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, "
                f"features {image_features.shape[0]}"
            )
        return special_image_mask

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: torch.LongTensor | None = None,
    ) -> torch.LongTensor:
        r"""
        Build the multi-channel `(text, width, height, temporal)` position ids used by HunYuanVL's xdrope.

        Text tokens use a flat sequence index in every channel. Vision tokens belonging to an image placeholder span
        overwrite the `width` and `height` channels with the per-image 2D grid coordinates; the temporal channel is
        kept aligned with the text index. Returns a `(batch, 4, seq)` tensor.
        """
        merge_size = self.config.vision_config.spatial_merge_size
        batch_position_ids = []
        image_index = 0
        for sample_ids in input_ids:
            seq_len = sample_ids.shape[-1]
            device = sample_ids.device
            base = torch.arange(seq_len, dtype=torch.int64, device=device)
            position_ids = base.clone()
            position_ids_w = base.clone()
            position_ids_h = base.clone()
            position_ids_t = base.clone()

            if image_grid_thw is not None:
                image_start_positions = torch.where(sample_ids == self.config.im_start_id)[0]
                for start_pos in image_start_positions:
                    _, grid_h, grid_w = (int(value) for value in image_grid_thw[image_index])
                    patch_h = grid_h // merge_size
                    patch_w = grid_w // merge_size
                    # Skip the image-start token and the merger's leading ``image_begin`` token before writing the
                    # 2D grid coordinates for the ``patch_h * (patch_w + 1)`` merged patch tokens.
                    token_start = int(start_pos) + 2
                    replace_num = patch_h * (patch_w + 1)
                    position_ids_w[token_start : token_start + replace_num] = torch.tensor(
                        list(range(patch_w + 1)) * patch_h, dtype=torch.int64, device=device
                    )
                    patch_h_list: list[int] = []
                    for h_idx in range(patch_h):
                        patch_h_list += [h_idx] * (patch_w + 1)
                    position_ids_h[token_start : token_start + replace_num] = torch.tensor(
                        patch_h_list, dtype=torch.int64, device=device
                    )
                    image_index += 1

            batch_position_ids.append(torch.stack([position_ids, position_ids_w, position_ids_h, position_ids_t]))

        return torch.stack(batch_position_ids)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        pixel_values (`torch.FloatTensor`, *optional*):
            Flat per-patch pixel features produced by the image processor.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        if pixel_values is not None and image_grid_thw is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw).last_hidden_state
            image_embeds = image_embeds.to(inputs_embeds.device, dtype=inputs_embeds.dtype, non_blocking=True)
            image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # When position ids are not supplied (e.g. on the prefill step of generation) build the multi-channel xdrope
        # layout in the model so the rotary logic stays centralized here rather than in the processor.
        if position_ids is None and input_ids is not None and (past_key_values is None or past_key_values.get_seq_length() == 0):
            position_ids = self.get_rope_index(input_ids, image_grid_thw)

        return self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )


@auto_docstring
class HunYuanVLForConditionalGeneration(HunYuanVLPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    config: HunYuanVLConfig

    def __init__(self, config: HunYuanVLConfig):
        super().__init__(config)
        self.model = HunYuanVLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        return self.model.get_image_features(pixel_values, image_grid_thw, **kwargs)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        pixel_values (`torch.FloatTensor`, *optional*):
            Flat per-patch pixel features produced by the image processor.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.

        Example:

        ```python
        >>> from transformers import AutoProcessor, HunYuanVLForConditionalGeneration

        >>> model_id = "tencent/HunyuanOCR"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = HunYuanVLForConditionalGeneration.from_pretrained(model_id, device_map="auto")

        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "image", "image": "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/image_ocr.jpg"},
        ...             {"type": "text", "text": "Extract the text from the image."},
        ...         ],
        ...     }
        ... ]
        >>> inputs = processor.apply_chat_template(
        ...     messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ... ).to(model.device)

        >>> generated_ids = model.generate(**inputs, max_new_tokens=128)
        >>> generated_trimmed = generated_ids[0][inputs["input_ids"].shape[-1]:]
        >>> print(processor.decode(generated_trimmed, skip_special_tokens=True))
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        pixel_values=None,
        image_grid_thw=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **kwargs,
        )

        # Vision features are only consumed on the prefill step; drop them once we are decoding.
        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            model_inputs["pixel_values"] = None
            model_inputs["image_grid_thw"] = None

        # The multi-channel xdrope position ids are rebuilt by the model on prefill (``get_rope_index``); for decode
        # steps we let the model derive them from the cache length, so do not forward a stale ``position_ids``.
        model_inputs["position_ids"] = None

        return model_inputs


__all__ = [
    "HunYuanVLConfig",
    "HunYuanVLVisionConfig",
    "HunYuanVLTextConfig",
    "HunYuanVLImageProcessor",
    "HunYuanVLImageProcessorPil",
    "HunYuanVLPreTrainedModel",
    "HunYuanVLModel",
    "HunYuanVLTextModel",
    "HunYuanVLForConditionalGeneration",
]
