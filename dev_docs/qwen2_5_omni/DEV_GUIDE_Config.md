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

## 类级属性与加载行为（Class Attributes & Loading）

- `model_type`
  - 每个配置类声明唯一 `model_type`（如 `qwen2_5_omni_thinker`、`qwen2_5_omni_talker`、`qwen2_5_omni_dit` 等），用于注册与 `AutoConfig/AutoModel` 的映射。
  - `from_pretrained` 会依据 `config.json` 中的 `model_type` 选择正确的模型实现类与权重装载路径。
- `attribute_map`
  - 在 `Qwen2_5OmniThinkerConfig` 中将通用别名映射到真实字段（如 `image_token_id → image_token_index`），提升向后兼容性。
  - 加载旧版本配置或模型卡时，`from_pretrained` 自动应用映射避免字段缺失。
- `sub_configs`
  - `ThinkerConfig` 支持以 dict 或具体子配置类传入 `audio_config`/`vision_config`/`text_config`，并在构造函数中统一实例化。
  - 当模型卡内嵌子配置时，`from_pretrained` 直接解析并构造对应子模块，确保多模态分支参数一致。
- RoPE 与滑窗相关类级参数
  - `rope_scaling`、`use_sliding_window`、`sliding_window`、`max_window_layers`、`layer_types` 等在加载时决定位置编码与注意力策略；
  - 若未显式提供，构造函数会根据 `num_hidden_layers` 自动填充 `layer_types` 并进行校验（`layer_type_validation`）。
- 初始化范围
  - `initializer_range` 等权重初始化超参在模型构建时生效；通过 `from_pretrained` 加载已训练权重时通常不影响权重值，但会影响随机初始化分支（例如新增未训练头）。
- 与 Processor 的交互
  - 配置中的特殊 token 索引（image/audio/video/TTS）需要与 `Qwen2_5OmniProcessor` 保持一致，处理器在构造多模态序列时依赖这些索引；
  - `ProcessorMixin.from_pretrained` 会加载 `preprocessor_config.json` 并与此处配置协同工作，确保 `model_input_names` 字段约定维持一致。
