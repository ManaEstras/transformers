# Qwen2.5-Omni 处理器指南（Processor）

`Qwen2_5OmniProcessor` 是多模态输入的统一入口，负责：
- 文本聊天模板与特殊 token 序列
- 图像/视频网格化预处理
- 音频特征提取与掩码
- 多模态占位符替换与拼接

## 结构与子参数

- `Qwen2_5OmniProcessor` 封装以下 kwargs 类：
  - `Qwen2_5_OmniVideosKwargs`
    - 关键字段：`fps`、`use_audio_in_video`、`seconds_per_chunk`、`position_id_per_seconds`、`min_pixels`、`max_pixels`、`patch_size`、`temporal_patch_size`、`merge_size`、`size.shortest_edge/longest_edge`
  - `Qwen2_5_OmniImagesKwargs`
    - 关键字段：`min_pixels`、`max_pixels`、`patch_size`、`merge_size`、`size.shortest_edge/longest_edge`
  - `Qwen2_5OmniProcessorKwargs`
    - 关键字段：`chat_template`、`force_enable_chat_template`、`text_max_length`、音频采样率与特征维度相关参数

## 关键方法与属性

- `__call__(...)`
  - 接受 `text`、`images`、`videos`、`audios`；内部调用 `replace_multimodal_special_tokens` 将 `<image>`/`<video>`/`<audio>` 占位符替换为真实序列，并插入 start/end token。
  - 输出字段与模型侧对齐（见下）。
- `replace_multimodal_special_tokens(...)`
  - 解析输入文本中的多模态占位符，计算需要插入的序列（图像/视频网格、音频特征），并拼接到文本 token 序列中。
  - 负责维护 `position_ids` 连续性（与 `seconds_per_chunk`、`position_id_per_seconds` 配合），确保时间维度位置编码正确。
- `apply_chat_template(...)`
  - 根据 `chat_template` 为对话输入生成规范的序列，如系统提示、用户/助手角色 token 等；`force_enable_chat_template=True` 时强制启用。
- `model_input_names`
  - 文本：`input_ids`、`attention_mask`、`position_ids`
  - 图像/视频：`image_grid_thw` / `video_grid_thw`、`video_second_per_grid`
  - 音频：`input_features`、`feature_attention_mask`、`audio_feature_lengths`

## 输入与形状约束

- 图像/视频：
  - `size.shortest_edge/longest_edge` 定义下采样目标；`min_pixels/max_pixels` 用于限制分辨率范围；最终以 `(T, H, W)` 网格表示并拼入序列。
  - `patch_size`、`temporal_patch_size`、`merge_size` 决定网格粒度与 token 数，需与模型配置一致。
- 音频：
  - 使用指定采样率提取特征（如 mel）；提供 `feature_attention_mask` 与 `audio_feature_lengths` 保证正确的帧有效性与掩码。
- 文本：
  - 模板化后生成 `input_ids`；`position_ids` 必须覆盖整个拼接序列（含多模态片段）。

## 常见用法建议

- 当需要视频时间位置对应：
  - 保持 `seconds_per_chunk` 与 `position_id_per_seconds` 一致，`video_second_per_grid` 将据此映射到 THW 网格时间步。
- 当使用聊天模板：
  - 确认模型卡的模板版本；必要时启用 `force_enable_chat_template`；确保占位符与模型支持的 special token 对应。
- 当调整分辨率：
  - 同步修改 `size.shortest_edge/longest_edge` 与 `min_pixels/max_pixels`，并评估 `merge_size` 对 token 数与显存的影响。

## 类级属性与加载行为（Class Attributes & Loading）

- `ProcessorMixin` 继承行为
  - `Qwen2_5OmniProcessor` 继承自 `ProcessorMixin`，拥有统一的 `from_pretrained` 加载机制：读取 `preprocessor_config.json`（或对应目录的处理器配置），并合并传入的 `kwargs`（如 `videos_kwargs`、`images_kwargs`）。
  - 通过 `save_pretrained` 可将当前处理器的参数写回到磁盘，供下次 `from_pretrained` 复用。
- `model_input_names`
  - 处理器声明并维护与模型加载一致的输入字段名集合（如 `input_ids`、`attention_mask`、`position_ids`、`image_grid_thw`、`video_grid_thw`、`video_second_per_grid`、`input_features`、`feature_attention_mask`、`audio_feature_lengths`）。
  - 模型侧在 `forward`/`generate` 中依赖这些字段，`Processor` 的设计确保 `from_pretrained` 后字段契约稳定。
- 模板与特殊 token 协议
  - `apply_chat_template` 的默认模板与是否强制启用（`force_enable_chat_template`）在处理器配置中体现；加载时若模板缺失或版本不一致，处理器将提供向后兼容策略。
  - `replace_multimodal_special_tokens` 在加载后保持与配置中的特殊 token 索引一致：例如 `image_token_index`、`video_token_index`、`audio_start/end_token_id`、`tts_*` 系列索引。
- `from_pretrained` 交互要点
  - 当模型与处理器来自同一模型卡时，推荐分别调用 `from_pretrained` 以确保配置与预处理同步；
  - 若自定义了 `videos_kwargs`/`images_kwargs`/`processor_kwargs`，传入的值会覆盖磁盘配置，适合做实验性调优但需与模型配置（如 `patch_size`、`temporal_patch_size`）保持一致。
