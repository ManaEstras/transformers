# Qwen2‑VL 开发指南：简短示例（推理与生成）

以下示例演示完整调用链：Processor → Model/ForConditionalGeneration，覆盖图像与视频的推理与生成。示例基于标准 `transformers` 使用方式，便于快速验证。

## 前置条件

- 已安装 `transformers` 与其依赖（PyTorch）。
- 图像示例依赖 `Pillow`；视频示例可选使用 `decord` 或自行提取帧为 `PIL.Image` 列表。

## 示例一：图像 → 文本生成

```python
from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
import torch

model_id = "Qwen/Qwen2-VL-7B-Instruct"  # 或本地权重路径
processor = Qwen2VLProcessor.from_pretrained(model_id)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16
).eval()

image = Image.open("dog.jpg").convert("RGB")
prompt = "请用一句话描述这张图片：<|image_pad|>"

inputs = processor(text=[prompt], images=[image], return_tensors="pt")
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=64)

# 将生成的 token 解码为文本（使用 Processor 的后处理）
text = processor.post_process_image_text_to_text(generated_ids)[0]
print(text)
```

要点：
- 文本中包含 `<|image_pad|>`，Processor 会根据图像 patch 数展开占位符并对齐视觉特征。
- 视觉张量仅在 pre‑fill 阶段传入；后续步进阶段不再重复传入以提升性能。

## 示例二：视频 → 文本生成（使用 decord）

```python
from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
import torch

# 可选：读取视频并抽取若干帧为 PIL.Image 列表
from decord import VideoReader, cpu

model_id = "Qwen/Qwen2-VL-7B-Instruct"
processor = Qwen2VLProcessor.from_pretrained(model_id)
model = Qwen2VLForConditionalGeneration.from_pretrained(model_id).eval()

vr = VideoReader("sample.mp4", ctx=cpu())
# 均匀采样最多 64 帧（示例），Processor 内部仍会做整除 temporal_patch_size 的约束
idxs = list(range(0, len(vr), max(1, len(vr)//64)))
frames = [Image.fromarray(vr[i].asnumpy()).convert("RGB") for i in idxs]

prompt = "请总结这段视频的主要内容：<|video_pad|>"
inputs = processor(text=[prompt], videos=[frames], return_tensors="pt")
with torch.no_grad():
    gen_ids = model.generate(**inputs, max_new_tokens=128)

text = processor.post_process_image_text_to_text(gen_ids)[0]
print(text)
```

要点：
- 视频以帧序列形式传入；Processor 会进行均匀采样与批量 `smart_resize`，并保证采样帧数可被 `temporal_patch_size` 整除。
- `merge_size`/`patch_size`/`temporal_patch_size` 建议与模型 `vision_config` 保持一致。

## 示例三：仅提取视觉特征（推理）

```python
from transformers import Qwen2VLProcessor, Qwen2VLModel
from PIL import Image

model_id = "Qwen/Qwen2-VL-7B-Instruct"
processor = Qwen2VLProcessor.from_pretrained(model_id)
model = Qwen2VLModel.from_pretrained(model_id).eval()

image = Image.open("dog.jpg").convert("RGB")
proc = processor(text=["<|image_pad|>"], images=[image], return_tensors="pt")

# 仅编码图像特征（不经过语言头）
image_features = model.get_image_features(
    pixel_values=proc["pixel_values"],
    image_grid_thw=proc["image_grid_thw"],
)
print(image_features.shape)  # 例如：[num_patches, hidden_size]
```

说明：
- 若需视频特征，调用 `get_video_features(pixel_values_videos=..., video_grid_thw=...)`。
- 特征维度与 patch 数由预处理管线与 `merge_size` 共同决定。

## 常见排错

- 报占位符数量不匹配：检查 `merge_size` 与预处理器参数是否与模型对齐。
- 视频帧数不可整除：调整 `num_frames`/`fps` 或在采样后裁剪，确保 `num_frames % temporal_patch_size == 0`。
