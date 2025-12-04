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

