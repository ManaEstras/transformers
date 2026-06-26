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
"""Testing suite for the PyTorch HunYuanVL model."""

import unittest

from transformers import (
    AutoProcessor,
    HunYuanVLConfig,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import cleanup, require_torch, slow, torch_device

from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        HunYuanVLForConditionalGeneration,
        HunYuanVLModel,
    )
    from transformers.models.hunyuan_vl.configuration_hunyuan_vl import (
        HunYuanVLTextConfig,
        HunYuanVLVisionConfig,
    )

if is_vision_available():
    from transformers.image_utils import load_image


class HunYuanVLVisionText2TextModelTester(VLMModelTester):
    base_model_class = HunYuanVLModel if is_torch_available() else None
    config_class = HunYuanVLConfig
    conditional_generation_class = HunYuanVLForConditionalGeneration if is_torch_available() else None
    text_config_class = HunYuanVLTextConfig if is_torch_available() else None
    vision_config_class = HunYuanVLVisionConfig if is_torch_available() else None

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("image_size", 64)
        kwargs.setdefault("patch_size", 16)
        kwargs.setdefault("num_channels", 3)
        kwargs.setdefault("num_image_tokens", (64 // 16) * (64 // 16 + 1) + 2)  # grid_hw * (grid_hw + 1) + 2
        kwargs.setdefault("image_token_id", 5)
        kwargs.setdefault("hidden_size", 64)
        kwargs.setdefault("intermediate_size", 128)
        kwargs.setdefault("num_hidden_layers", 2)
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("num_key_value_heads", 4)
        kwargs.setdefault("hidden_act", "silu")
        kwargs.setdefault("max_position_embeddings", 128)
        kwargs.setdefault("pad_token_id", 0)
        kwargs.setdefault("bos_token_id", 1)
        kwargs.setdefault("eos_token_id", 2)
        kwargs.setdefault("head_dim", 16)
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(parent, **kwargs)

    def get_vision_config(self):
        return self.vision_config_class(
            num_channels=self.num_channels,
            patch_size=self.patch_size,
            temporal_patch_size=1,
            spatial_merge_size=1,
            num_hidden_layers=2,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            hidden_act="silu",
            out_hidden_size=64,
            text_hidden_size=64,
            max_image_size=self.image_size,
            min_image_size=self.image_size,
            anyres_vit_max_image_size=self.image_size,
            max_vit_seq_len=(self.image_size // self.patch_size) ** 2,
        )

    def get_config(self):
        return self.config_class(
            text_config=self.get_text_config(),
            vision_config=self.get_vision_config(),
            image_token_id=self.image_token_id,
        )

    def create_pixel_values(self):
        from ...test_modeling_common import floats_tensor

        grid_hw = self.image_size // self.patch_size
        num_patches = grid_hw * grid_hw
        # HunYuanVL uses flattened pixel values: [batch * num_patches, channels * patch_size * patch_size]
        return floats_tensor(
            [self.batch_size * num_patches, self.num_channels * self.patch_size * self.patch_size], scale=1.0
        )

    def _prepare_modality_inputs(self, input_ids, config):
        pixel_values = self.create_pixel_values()
        input_ids = self.place_image_tokens(input_ids, config)

        grid_hw = self.image_size // self.patch_size
        image_grid_thw = torch.tensor(
            [[1, grid_hw, grid_hw]] * self.batch_size,
            device=torch_device,
        )
        # The model builds the 4-channel (text, width, height, temporal) xdrope position ids internally via
        # ``HunYuanVLModel.get_rope_index``, so the tester only needs to provide pixel_values + image_grid_thw.
        return input_ids, {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


@require_torch
class HunYuanVLModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = HunYuanVLVisionText2TextModelTester
    test_all_params_have_gradient = False
    test_torch_exportable = False
    # `get_image_features` returns the merged per-image embeddings concatenated along dim=1 (shape
    # `(1, total_merged_tokens, out_hidden_size)`), so the generic `(batch, ..., hidden)` shape assertion does not
    # apply. The return type / hidden_states / attentions are still validated by the other image-feature tests.
    skip_test_image_features_output_shape = True

    @unittest.skip(
        reason="HunYuanVL uses custom position_ids (4-channel) and image_grid_thw, "
        "making standard SDPA flash dispatch test incompatible."
    )
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(
        reason="HunYuanVL vision transformer uses absolute position embeddings "
        "that can cause device mismatch during CPU offloading."
    )
    def test_cpu_offload(self):
        pass

    @unittest.skip(
        reason="HunYuanVL vision transformer uses absolute position embeddings "
        "that can cause device mismatch during disk offloading."
    )
    def test_disk_offload_bin(self):
        pass

    @unittest.skip(
        reason="HunYuanVL vision transformer uses absolute position embeddings "
        "that can cause device mismatch during disk offloading."
    )
    def test_disk_offload_safetensors(self):
        pass


@require_torch
class HunYuanVLIntegrationTest(unittest.TestCase):
    """Slow integration tests with actual model weights."""

    model_id = "tencent/HunyuanOCR"

    def setUp(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_small_model_integration_test_ocr(self):
        """Test OCR generation with a real image."""
        model = HunYuanVLForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, device_map=torch_device
        )
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/image_ocr.jpg"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Extract the text from the image."},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, padding=True, return_tensors="pt"
        ).to(model.device)

        generate_ids = model.generate(**inputs, do_sample=False, max_new_tokens=50)
        decoded = self.processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Verify that the model generates non-empty text
        self.assertGreater(len(decoded.strip()), 0)

    @slow
    def test_text_only_generation(self):
        """Test that the model can do text-only generation without images."""
        model = HunYuanVLForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, device_map=torch_device
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello, what is 1+1?"},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, padding=True, return_tensors="pt"
        ).to(model.device)

        generate_ids = model.generate(**inputs, do_sample=False, max_new_tokens=20)
        decoded = self.processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        self.assertGreater(len(decoded.strip()), 0)
