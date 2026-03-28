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
Processor class for HunYuanVL-MoT model.
Inherits from HunYuanVLProcessor to handle image/video + text input.
"""

from transformers.models.hunyuan_vl.processing_hunyuan_vl import HunYuanVLProcessor


class HunYuanVLMoTProcessor(HunYuanVLProcessor):
    r"""
    Processor that combines image/video processing and text tokenization for HunYuanVL-MoT.
    
    Inherits from HunYuanVLProcessor. Handles different token IDs for MoT variant:
    - IMAGE_TOKEN_ID: 120687 (vs 120120 in base hunyuan_vl)
    - VIDEO_TOKEN_ID: 120688
    - LATENT_TOKEN_ID: 120690
    
    Args:
        image_processor (`HunYuanVLImageProcessor`):
            The image processor is in charge of resizing, normalizing and converting images into pixel values.
        video_processor (`HunYuanVLVideoProcessor`):
            The video processor is in charge of sampling frames, resizing, normalizing and converting videos into pixel values.
        tokenizer (`PreTrainedTokenizer`):
            A `PreTrainedTokenizer` is in charge of encoding text input.
    """
    
    # MoT-specific token IDs (different from base hunyuan_vl)
    IMAGE_TOKEN_ID = 120687
    VIDEO_TOKEN_ID = 120688
    LATENT_TOKEN_ID = 120690
    
    # Note: Parent class (HunYuanVLProcessor) defines:
    # IMAGE_START_TOKEN_ID = 120118
    # IMAGE_END_TOKEN_ID = 120119
    # IMAGE_TOKEN_ID = 120120  (override in MoT)
    # VIDEO_START_TOKEN_ID = 120122
    # VIDEO_END_TOKEN_ID = 120123
    
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        **kwargs,
    ):
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            **kwargs,
        )
    
    # Inherits all processing logic from parent:
    # - __call__(images, text, videos)
    # - Automatic token replacement with image/video counts
    # - Position ID generation with spatial layout
    # - ProcessorMixin features
    #
    # The parent implementation already handles:
    # 1. Image preprocessing via HunYuanVLImageProcessor
    # 2. Video preprocessing via HunYuanVLVideoProcessor
    # 3. Text tokenization
    # 4. Token count calculation (_get_num_multimodal_tokens)
    # 5. Image position tracking (get_imgs_pos)
    # 6. 4D position ID generation (absolute, width, height, temporal)
    #
    # MoT variant overrides token IDs but logic remains the same.
    # If additional MoT-specific processing is needed (e.g., modality masks),
    # override __call__ in this class.
