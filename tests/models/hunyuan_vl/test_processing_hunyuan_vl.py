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

import unittest

import numpy as np

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_torchvision_available

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch

if is_torchvision_available():
    from transformers.models.hunyuan_vl.image_processing_hunyuan_vl import HunYuanVLImageProcessor

from transformers.models.hunyuan_vl.processing_hunyuan_vl import HunYuanVLProcessor


@require_torch
@require_vision
class HunYuanVLProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = HunYuanVLProcessor

    @classmethod
    def _setup_image_processor(cls):
        if is_torchvision_available():
            return HunYuanVLImageProcessor(
                min_pixels=32 * 32,
                max_pixels=32 * 32,
                patch_size=16,
                temporal_patch_size=1,
                merge_size=1,
            )
        # Fallback to PIL processor if torchvision not available
        from transformers.models.hunyuan_vl.image_processing_pil_hunyuan_vl import HunYuanVLImageProcessorPil

        return HunYuanVLImageProcessorPil(
            min_pixels=32 * 32,
            max_pixels=32 * 32,
            patch_size=16,
            temporal_patch_size=1,
            merge_size=1,
        )

    @classmethod
    def _setup_tokenizer(cls):
        tokenizer_class = cls._get_component_class_from_processor("tokenizer")
        tokenizer = tokenizer_class.from_pretrained("tencent/HunyuanOCR")
        return tokenizer

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = getattr(processor, "image_token", "<image>")

    def test_model_input_names(self):
        processor = self.get_processor()
        expected_names = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
        self.assertSetEqual(set(processor.model_input_names), expected_names)

    @require_torch
    def _test_apply_chat_template(
        self,
        modality: str,
        batch_size: int,
        return_tensors: str,
        input_name: str,
        processor_name: str,
        input_data: list[str],
    ):
        # HunYuanVL flattens vision features into per-patch rows, so ``pixel_values`` does not scale with batch size
        # the same way as standard models. This override mirrors Qwen2-VL's: it derives the expected ``pixel_values``
        # length from ``image_grid_thw`` instead of asserting it equals ``batch_size``.
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")
        if processor_name not in self.processor_class.get_attributes():
            self.skipTest(f"{processor_name} attribute not present in {self.processor_class}")

        batch_messages = [
            [{"role": "user", "content": [{"type": "text", "text": "Describe this."}]}]
        ] * batch_size

        formatted_prompt = processor.apply_chat_template(batch_messages, add_generation_prompt=True, tokenize=False)
        self.assertEqual(len(formatted_prompt), batch_size)

        out_dict_text = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors=return_tensors,
        )
        self.assertTrue(all(key in out_dict_text for key in ["input_ids", "attention_mask"]))
        self.assertEqual(len(out_dict_text["input_ids"]), batch_size)
        self.assertEqual(len(out_dict_text["attention_mask"]), batch_size)

        for idx, url in enumerate(input_data[:batch_size]):
            batch_messages[idx][0]["content"] = [batch_messages[idx][0]["content"][0], {"type": modality, "url": url}]

        out_dict = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors=return_tensors,
        )
        resolved_input_name = getattr(self, input_name)
        self.assertTrue(resolved_input_name in out_dict)
        self.assertEqual(len(out_dict["input_ids"]), batch_size)
        self.assertEqual(len(out_dict["attention_mask"]), batch_size)

        # ``pixel_values`` length equals the total number of patch rows across all images.
        expected_patch_rows = 0
        for thw in out_dict["image_grid_thw"]:
            expected_patch_rows += int(thw[0] * thw[1] * thw[2])
        self.assertEqual(len(out_dict[resolved_input_name]), expected_patch_rows)

        return_tensor_to_type = {"pt": torch.Tensor, "np": np.ndarray, None: list}
        for value in out_dict.values():
            self.assertIsInstance(value, return_tensor_to_type[return_tensors])

    def test_get_num_multimodal_tokens(self):
        processor = self.get_processor()
        output = processor._get_num_multimodal_tokens(image_sizes=[(32, 32)])

        self.assertEqual(len(output["num_image_tokens"]), 1)
        self.assertEqual(len(output["num_image_patches"]), 1)
        self.assertGreater(output["num_image_tokens"][0], 0)
