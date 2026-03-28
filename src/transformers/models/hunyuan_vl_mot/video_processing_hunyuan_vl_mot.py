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
Video processor for HunYuanVL-MoT model.
Reuses HunYuanVL video processor without modifications.
"""

from transformers.models.hunyuan_vl.video_processing_hunyuan_vl import HunYuanVLVideoProcessor

# HunYuanVL-MoT uses the same video processing as base hunyuan_vl
# (frame sampling, temporal patch encoding)
HunYuanVLMoTVideoProcessor = HunYuanVLVideoProcessor

__all__ = ["HunYuanVLMoTVideoProcessor"]
