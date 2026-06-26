<!--Copyright 2026 The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# HunYuanVL

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white">
</div>

## Overview

*This model was contributed to Hugging Face Transformers on 2026-06-09.*

HunYuanVL is a vision-language model developed by Tencent for image-text understanding and generation. The model is described in the paper [Tencent Hunyuan: A Comprehensive Technical Report](https://huggingface.co/papers/2511.19575).

*The abstract from the paper is the following:*

*Tencent Hunyuan is a series of large language and multimodal models designed for a wide range of applications including text understanding, image recognition, video analysis, code generation, and mathematical reasoning. The vision-language variant (HunYuanVL) combines a powerful dense text backbone with a vision transformer to enable OCR and document understanding tasks.*

The open-source `hunyuan_vl` integration in Transformers is a dense-only image-text variant tailored for OCR and document understanding style workloads such as [`tencent/HunyuanOCR`](https://huggingface.co/tencent/HunyuanOCR).

## Usage Tips

- The current open-source variant supports: dense-only text backbone, image-text prompting, OCR/document-understanding style generation.
- Not supported in this open-source variant: video inputs and runtime MoE execution paths.
- Some legacy Tencent-export configuration fields are still accepted so existing checkpoints can be loaded, but those fields do not imply that the open-source implementation enables extra runtime capabilities.
- When batching variable-length prompts, pass `padding=True` if you need tensor outputs from the processor.
- The current open-source variant is not a drop-in replacement for internal full-capability HunYuanVL stacks.
- Both image processor backends are available: the default `torchvision`-backed fast processor and a `pil` backend
  (load with `AutoProcessor.from_pretrained(..., backend="pil")`). The two backends produce equivalent patch-aligned
  features; pick `pil` only when torchvision is unavailable.
- If you are extending the model family upstream, make changes in `modular_hunyuan_vl.py` and regenerate the derived files instead of editing generated modeling/configuration files directly.

## Recommended checkpoints

- [`tencent/HunyuanOCR`](https://huggingface.co/tencent/HunyuanOCR) for OCR and document extraction workloads.

## Usage

```python
from transformers import AutoProcessor, HunYuanVLForConditionalGeneration

model_name_or_path = "tencent/HunyuanOCR"
processor = AutoProcessor.from_pretrained(model_name_or_path)
model = HunYuanVLForConditionalGeneration.from_pretrained(
    model_name_or_path,
    device_map="auto",
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "path/to/image.jpg"},
            {"type": "text", "text": "Extract the text from the image."},
        ],
    }
]
inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, padding=True, return_tensors="pt"
).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = generated_ids[0][len(inputs["input_ids"][0]) :]
output = processor.decode(generated_ids_trimmed, skip_special_tokens=True)
print(output)
```

## HunYuanVLConfig

[[autodoc]] HunYuanVLConfig

## HunYuanVLVisionConfig

[[autodoc]] HunYuanVLVisionConfig

## HunYuanVLTextConfig

[[autodoc]] HunYuanVLTextConfig

## HunYuanVLProcessor

[[autodoc]] HunYuanVLProcessor
    - __call__

## HunYuanVLImageProcessor

[[autodoc]] HunYuanVLImageProcessor

## HunYuanVLImageProcessorPil

[[autodoc]] HunYuanVLImageProcessorPil

`HunYuanVLForConditionalGeneration` is the main public entrypoint for image-text generation. `HunYuanVLModel` returns
the raw hidden states (vision tower + text backbone) without the language-modeling head, and `HunYuanVLTextModel`
exposes the text backbone for lower-level text-only workflows.

## HunYuanVLTextModel

[[autodoc]] HunYuanVLTextModel
    - forward

## HunYuanVLModel

[[autodoc]] HunYuanVLModel
    - forward

## HunYuanVLForConditionalGeneration

[[autodoc]] HunYuanVLForConditionalGeneration
    - forward
    - get_image_features
