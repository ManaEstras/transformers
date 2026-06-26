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
"""Processor class for HunYuanVL."""

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring


# Sentinel marker used internally while expanding image placeholders so that already-expanded spans are not matched
# again when searching for the next image token. It never reaches the tokenizer (it is restored to the image token
# before tokenization).
_PLACEHOLDER_SENTINEL = "\x00<hunyuanvl_image_placeholder>\x00"


class HunYuanVLProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {}


@auto_docstring
class HunYuanVLProcessor(ProcessorMixin):
    r"""
    HunYuanVL processor that wraps an image processor and a tokenizer for image-text-to-text generation.

    The processor expands every image placeholder token in the prompts into a span of placeholder tokens whose length
    is inferred from the corresponding `image_grid_thw`, wrapping each span with the image start / end tokens.
    """

    valid_processor_kwargs = HunYuanVLProcessorKwargs

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, cat_extra_token: bool = True, **kwargs):
        r"""
        cat_extra_token (`bool`, *optional*, defaults to `True`):
            Whether each expanded image span includes the two extra (begin / end) delimiter tokens when counting the
            number of placeholder tokens to insert.
        """
        self.tokenizer = tokenizer

        # HunYuan-style tokenizers expose the special image tokens via attributes; preserve a useful error message
        # if a caller forgot to register them.
        for attr in ("image_token", "image_start_token", "image_end_token"):
            if not hasattr(tokenizer, attr):
                raise ValueError(
                    f"Tokenizer is missing required attribute '{attr}'. "
                    "Add the corresponding mapping to `extra_special_tokens` in `tokenizer_config.json` or set the "
                    "attribute manually before constructing the processor."
                )

        self.image_token = tokenizer.image_token
        self.image_token_id = tokenizer.image_token_id
        self.image_start_token = tokenizer.image_start_token
        self.image_start_token_id = tokenizer.image_start_token_id
        self.image_end_token = tokenizer.image_end_token
        self.image_end_token_id = tokenizer.image_end_token_id
        self.pad_id = tokenizer.pad_token_id
        self.cat_extra_token = cat_extra_token

        # The processor owns the chat template (v5 no longer routes chat templating through the tokenizer). Released
        # HunYuan checkpoints persist the template on the tokenizer config, so fall back to it when the processor was
        # not given an explicit template.
        if chat_template is None:
            chat_template = getattr(tokenizer, "chat_template", None)

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def _get_image_token_count(self, grid_h: int, grid_w: int) -> int:
        patch_h = grid_h // self.image_processor.merge_size
        patch_w = grid_w // self.image_processor.merge_size
        return patch_h * (patch_w + 1) + (2 if self.cat_extra_token else 0)

    @staticmethod
    def _has_wrappers(prompt: str, token_start: int, start_token: str, token: str, end_token: str) -> bool:
        start_index = token_start - len(start_token)
        end_index = token_start + len(token)
        return (
            start_index >= 0
            and prompt[start_index:token_start] == start_token
            and prompt[end_index : end_index + len(end_token)] == end_token
        )

    def __call__(
        self,
        images: ImageInput = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        **kwargs: Unpack[HunYuanVLProcessorKwargs],
    ) -> BatchFeature:
        if images is None and text is None:
            raise ValueError(f"You need to provide at least one input to call {self.__class__.__name__}")

        output_kwargs = self._merge_kwargs(
            HunYuanVLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        image_inputs: dict = {}
        image_grid_thw = None
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]

        if text is None:
            return BatchFeature(data={**image_inputs})

        if not isinstance(text, list):
            text = [text]
        text = text.copy()

        if images is not None and any(not isinstance(prompt, str) for prompt in text):
            raise ValueError(
                "`HunYuanVLProcessor` expects string prompts when multimodal inputs are provided so that multimodal "
                "placeholder tokens can be expanded before tokenization."
            )

        if images is not None:
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    token_start = text[i].index(self.image_token)
                    has_wrappers = self._has_wrappers(
                        text[i], token_start, self.image_start_token, self.image_token, self.image_end_token
                    )
                    _, grid_h, grid_w = (int(value) for value in image_grid_thw[index])
                    num_image_tokens = self._get_image_token_count(grid_h, grid_w)
                    replacement = _PLACEHOLDER_SENTINEL * num_image_tokens
                    if not has_wrappers:
                        replacement = self.image_start_token + replacement + self.image_end_token
                    text[i] = text[i].replace(self.image_token, replacement, 1)
                    index += 1
                text[i] = text[i].replace(_PLACEHOLDER_SENTINEL, self.image_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        # Multimodal placeholders are expanded as raw special tokens above, so special tokens must not be re-added.
        output_kwargs["text_kwargs"].pop("add_special_tokens", None)
        text_inputs = self.tokenizer(text, add_special_tokens=False, **output_kwargs["text_kwargs"])

        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])

        if return_mm_token_type_ids:
            text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(text_inputs["input_ids"])

        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        """Compute the number of placeholder tokens needed for the given list of image sizes."""
        vision_data: dict = {}
        if image_sizes is not None:
            merge_size = kwargs.get("merge_size") or self.image_processor.merge_size

            num_image_patches_size = [
                self.image_processor.get_number_of_image_patches(*image_size, kwargs) for image_size in image_sizes
            ]
            num_image_tokens = [
                patch_hw[0] // merge_size * (patch_hw[1] // merge_size + 1) + (2 if self.cat_extra_token else 0)
                for patch_hw in num_image_patches_size
            ]
            num_image_patches = [(patch_hw[0] * patch_hw[1]) for patch_hw in num_image_patches_size]
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        return MultiModalData(**vision_data)


__all__ = ["HunYuanVLProcessor"]
