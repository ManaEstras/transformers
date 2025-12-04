# Qwen2.5-Omni 配置指南（Config）

本指南梳理 `qwen2_5_omni` 目录下的所有配置类、`model_type`、关键字段与注意点，帮助你快速定位需要调整的超参。

## 配置类总览

- `Qwen2_5OmniThinkerConfig`（`model_type="qwen2_5_omni_thinker"`）
  - 作用：多模态理解与生成的主干配置，聚合 `audio_config`、`vision_config`、`text_config`。
  - 别名映射：`attribute_map = { image_token_id → image_token_index, video_token_id → video_token_index, audio_token_id → audio_token_index }`
  - 子配置：`sub_configs = { audio_config: Qwen2_5OmniAudioEncoderConfig, vision_config: Qwen2_5OmniVisionEncoderConfig, text_config: Qwen2_5OmniTextConfig }`
  - 特殊 token：`audio_token_index`、`image_token_index`、`video_token_index`、`audio_start_token_id`、`audio_end_token_id`、`user_token_id`
  - 时序参数：`position_id_per_seconds`、`seconds_per_chunk`

- `Qwen2_5OmniTalkerConfig`（`model_type="qwen2_5_omni_talker"`）
  - 作用：说话侧（语音生成）配置，定义 TTS 相关特殊 token 与词表维度。
  - 关键字段：
    - Modal tokens：`audio_token_index`、`image_token_index`、`video_token_index`
    - 词表维度：`vocab_size`、`embedding_size`
    - TTS 特殊 token：`tts_text_start_token_id`、`tts_text_end_token_id`、`tts_text_pad_token_id`、`tts_codec_start_token_id` 等
  - RoPE/注意力：与文本侧参数一致（如 `rope_theta`、`use_sliding_window`、`sliding_window`、`max_window_layers`、`rope_scaling`）。

- `Qwen2_5OmniTextConfig`（`model_type="qwen2_5_omni_text"`）
  - 作用：文本干路的基础超参。
  - 关键字段：`vocab_size`、`hidden_size`、`num_hidden_layers`、`num_attention_heads`、`rope_theta`
  - 滑窗：`use_sliding_window`、`sliding_window`、`max_window_layers`、`layer_types`
  - RoPE：`rope_scaling`（支持 `mrope_section`、`rope_type`、`type` 以及 `long_factor/low_freq_factor/high_freq_factor`）

- `Qwen2_5OmniVisionEncoderConfig`（`model_type="qwen2_5_omni_vision_encoder"`）
  - 作用：图像/视频编码器的维度与网格参数。
  - 关键字段：`depth`、`hidden_size`、`num_heads`、`patch_size`、`temporal_patch_size`、`merge_size`
  - 位置编码：与 3D RoPE 相关的缩放与策略在 `rope_scaling` 中统一设置。

- `Qwen2_5OmniAudioEncoderConfig`（`model_type="qwen2_5_omni_audio_encoder"`）
  - 作用：音频特征编码器（如 mel）参数。
  - 关键字段：`num_mel_bins`、`encoder_layers`、`encoder_attention_heads`、`d_model` 等（具体以源码为准）。

- 语音后端链路：
  - `Qwen2_5OmniDiTConfig`（`model_type="qwen2_5_omni_dit"`）：DiT 结构相关维度与 RoPE 参数。
  - `Qwen2_5OmniBigVGANConfig`（`model_type="qwen2_5_omni_bigvgan"`）：BigVGAN 语音解码器参数。
  - `Qwen2_5OmniToken2WavConfig`（`model_type="qwen2_5_omni_token2wav"`）：聚合 DiT/BigVGAN 的顶层包装。

## 常见参数解释与建议

- RoPE 缩放（文本/视觉/语音通用）：
  - `rope_scaling = { mrope_section: [...], rope_type: "default", type: "default" }`：适用于长上下文与多频段策略；如需长文本能力，关注 `long_factor`。
- 滑窗注意力：
  - 若 `use_sliding_window=True`，确保 `sliding_window`、`max_window_layers` 与 `layer_types` 一致，并在推理端正确生成层级注意力掩码。
- 视觉网格：
  - `patch_size`、`temporal_patch_size`、`merge_size` 共同决定 THW 网格与 token 数；需与处理器 `size.shortest_edge/longest_edge`、`min_pixels/max_pixels` 协调。
- 音频特征：
  - 采样率、帧长与 mel 维度要与训练配置一致；推理端需正确提供 `feature_attention_mask` 与 `audio_feature_lengths`。
- 特殊 token：
  - 在 Processor 中统一插入与替换；通过 `attribute_map` 兼容旧字段名。

## 配置组合与继承建议

- Thinker -> 聚合 Audio/Vision/Text 子配置；优先从模型卡直接 `from_pretrained` 取得全套配置。
- Talker -> 单独加载用于生成语音 token；与 Thinker 输出的文本或中间回复一致对齐 TTS token 协议。
- Token2Wav -> 与 Talker 的 codec 维度匹配；DiT/BigVGAN 参数需与训练一致。

