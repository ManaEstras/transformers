# Qwen2.5-Omni 使用示例（Examples）

本指南给出端到端示例，涵盖 Thinker 推理、Talker 语音生成与 Token2Wav 波形合成。

## 1. 多模态推理（文本 + 图像 + 视频 + 音频）

```python
from transformers import (
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)

# 加载处理器与模型
processor = Qwen2_5OmniProcessor.from_pretrained("/path/to/thinker")
model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained("/path/to/thinker").to("cuda")

# 准备输入（示例占位）
text = "用户: 请描述这段视频中的场景，并结合音频内容。"
images = ["/path/to/image1.jpg"]
videos = ["/path/to/video1.mp4"]
audios = ["/path/to/audio1.wav"]

inputs = processor(
    text=text,
    images=images,
    videos=videos,
    audios=audios,
    return_tensors="pt",
)

inputs = {k: v.to("cuda") for k, v in inputs.items()}

# 生成结果
gen_ids = model.generate(**inputs, max_new_tokens=128)
print(gen_ids)
```

要点：
- 处理器自动替换 `<image>`/`<video>`/`<audio>` 占位符，并插入 start/end token。
- 输出字段与模型对齐，包括 `image_grid_thw`、`video_grid_thw`、`video_second_per_grid`、`input_features` 等。

## 2. 生成语音（Talker → Token2Wav）

```python
from transformers import (
    Qwen2_5OmniProcessor,
    Qwen2_5OmniTalkerForConditionalGeneration,
    Qwen2_5OmniToken2WavModel,
)
import soundfile as sf

# 加载处理器与模型
processor = Qwen2_5OmniProcessor.from_pretrained("/path/to/talker")
talker = Qwen2_5OmniTalkerForConditionalGeneration.from_pretrained("/path/to/talker").to("cuda")
token2wav = Qwen2_5OmniToken2WavModel.from_pretrained("/path/to/token2wav").to("cuda")

# 准备 TTS 文本与 thinker 回复片段（示例）
tts_text = "这是要合成的语音文本内容。"
thinker_reply_part = "好的，我来为你朗读："  # 可来自 Thinker 的输出或上下文

inputs = processor(
    text=f"{thinker_reply_part} <tts>{tts_text}</tts>",
    return_tensors="pt",
)
inputs = {k: v.to("cuda") for k, v in inputs.items()}

# 生成语音相关 token（或 codec）
codec_ids = talker.generate(**inputs, max_new_tokens=256)

# 将 token/codec 转为波形
audio = token2wav(codec_ids=codec_ids)
audio = audio.detach().cpu().numpy()
sf.write("output.wav", audio, samplerate=24000)
```

要点：
- Talker 会处理 TTS 特殊 token（如 `tts_text_start/end/pad`、`tts_codec_start` 等）。
- Token2Wav 聚合 `DiT` 与 `BigVGAN` 路径，将 codec 转换为最终波形。

## 3. 自定义视频尺寸与时序

```python
from transformers import Qwen2_5OmniProcessor, Qwen2_5_OmniVideosKwargs

videos_kwargs = Qwen2_5_OmniVideosKwargs(
    fps=24,
    seconds_per_chunk=2,
    position_id_per_seconds=25,
    patch_size=14,
    temporal_patch_size=2,
    merge_size=2,
    size={"shortest_edge": 448, "longest_edge": 1120},
)

processor = Qwen2_5OmniProcessor.from_pretrained(
    "/path/to/thinker",
    videos_kwargs=videos_kwargs,
)
```

要点：
- 调整 `size.shortest_edge/longest_edge` 控制分辨率；`patch_size/temporal_patch_size/merge_size` 控制网格与 token 数。
- `seconds_per_chunk` 与 `position_id_per_seconds` 要与模型训练配置一致，以保证时间位置编码正确。

## 4. 聊天模板与角色切换（System/User/Assistant）

```python
from transformers import Qwen2_5OmniProcessor

processor = Qwen2_5OmniProcessor.from_pretrained("/path/to/thinker")

# 构造标准聊天消息
messages = [
    {"role": "system", "content": "你是一名多模态助手，回答要简洁准确。"},
    {"role": "user", "content": "请看这张图并总结要点：<image>"},
]

# 应用聊天模板（如需强制启用可设置 force_enable_chat_template=True）
text = processor.apply_chat_template(messages, tokenize=False)

# 结合图像输入，生成最终模型输入
inputs = processor(text=text, images=["/path/to/image.jpg"], return_tensors="pt")
```

说明：
- 将 `messages` 传给 `apply_chat_template` 可插入系统/角色样式与特殊分隔符，兼容旧模板时可使用 `force_enable_chat_template=True`。
- 生成的 `text` 中若包含 `<image>`/`<video>`/`<audio>` 占位符，`processor(...)` 会自动替换为对应的序列并插入 start/end token。
