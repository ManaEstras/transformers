# Qwen2.5-Omni 建模指南（Modeling）

本指南从实现角度说明 `modeling_qwen2_5_omni.py` 与 `modular_qwen2_5_omni.py` 的主要类、数据流、以及关键实现细节。

## 预训练基类与通用工具

- `Qwen2_5OmniPreTrainedModelForConditionalGeneration`
  - 提供通用的初始化、权重加载与一些辅助函数：
    - 4D 因果注意力掩码构造（匹配视觉/文本展开后的序列）
    - 3D RoPE 索引构造（视觉 THW 网格）

## 文本干路（Thinker）

- `Qwen2_5OmniThinkerTextModel`
  - 嵌入：文本嵌入 → 多层 `Qwen2_5OmniDecoderLayer`
  - 位置编码：`Qwen2_5OmniRotaryEmbedding`（支持长上下文的缩放与滑窗）
  - 输出：为条件生成（`ForConditionalGeneration`）提供 logits 与 past key values。
- `Qwen2_5OmniDecoderLayer`
  - 自注意力 + MLP；可在不同层应用 `sliding_attention` 或 `full_attention`，取决于配置 `layer_types`。

## 视觉分支

- `Qwen2_5OmniVisionBlock`
  - 注意力：`Qwen2_5OmniVisionAttention`（使用 3D RoPE，配合视觉专用 4D 掩码）
  - 前馈：`Qwen2_5OmniMLP`
  - 将 THW 网格编码为序列，供与文本融合或下游模块使用。

## 音频分支

- `Qwen2_5OmniAudioEncoderLayer` / `Qwen2_5OmniAudioAttention`
  - 将 mel/特征输入编码为与文本/视觉兼容的隐空间表示；支持 `feature_attention_mask` 与长度裁剪（`audio_feature_lengths`）。

## 说话侧（Talker）

- `Qwen2_5OmniTalkerModel` / `Qwen2_5OmniTalkerForConditionalGeneration`
  - 输入：`thinker_reply_part`（思考侧的回复片段）、`input_text_ids`（TTS 文本）、视觉/音频长度信息（`image_grid_thw`/`video_grid_thw`/`audio_feature_lengths`）
  - 位置编码：`Qwen2_5OmniRotaryEmbedding`
  - 输出：用于语音合成的 codec/token 序列（或中间语音相关 logits）。

## 语音合成后端（Token2Wav）

- `Qwen2_5OmniToken2WavModel`
  - 聚合：
    - `Qwen2_5OmniToken2WavDiTModel`（DiT 路径，支持 `Qwen2_5OmniDiTRotaryEmbedding`）
    - `Qwen2_5OmniToken2WavBigVGANModel`（BigVGAN 路径，做语音解码与重构）
  - 输入：来自 Talker 的 codec/token；输出：语音波形或中间特征。

## 位置编码家族

- `Qwen2_5OmniRotaryEmbedding`
  - 文本/视觉通用 RoPE；视觉侧支持 3D 索引（THW）构造与缩放策略。
- `Qwen2_5OmniDiTRotaryEmbedding`
  - 专用于 DiT 的位置编码，匹配语音后端的时空结构。

## 生成与掩码要点

- 因果性：
  - 文本与视觉融合序列必须使用正确的 4D 因果掩码，确保未来信息不被访问。
- 时间一致性：
  - 视频侧 `video_second_per_grid` 与 `position_id_per_seconds`、`seconds_per_chunk` 一致映射；音频侧需提供长度掩码。
- Past Key Values：
  - `ForConditionalGeneration` 类统一支持增量生成；注意在多模态下正确缓存与复用。

