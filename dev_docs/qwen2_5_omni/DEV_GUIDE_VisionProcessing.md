# Qwen2.5-Omni 视觉与视频处理（Vision Processing）

本指南说明图像/视频在处理器与模型中的网格化流程、3D RoPE、4D 注意力掩码、以及尺寸/像素约束的实操要点。

## THW 网格与尺寸约束

- 图像/视频在处理器中被映射到网格 `(T, H, W)`：
  - 图像：通常 `T=1`；由 `size.shortest_edge/longest_edge`、`patch_size`、`merge_size` 决定 `H, W`。
  - 视频：`T>1`；由 `fps` 与 `seconds_per_chunk`、`temporal_patch_size`、`merge_size` 决定时间维度；空间维度与图像一致。
- 像素约束：
  - `min_pixels` 与 `max_pixels` 在处理器侧生效，限制输入帧的有效分辨率范围，避免下游序列过长或过短。

## 3D RoPE 与 4D 因果掩码

- 3D RoPE：
  - 在视觉注意力中，位置编码基于 `(T, H, W)` 三维网格计算；`Qwen2_5OmniRotaryEmbedding` 支持对三维索引的构造与缩放。
  - 可配合 `rope_scaling` 做频段/跨度调节（如 `mrope_section`、`rope_type`）。
- 4D 因果掩码：
  - 在多头注意力中为 `(batch, heads, seq_q, seq_k)` 的掩码，视觉侧根据 THW 展平后的序列构造；
  - 保证时间维度的因果性与空间维度的局部性，防止未来帧信息泄漏。

## `video_second_per_grid` 与时序映射

- 处理器会输出 `video_second_per_grid`，用于将视频网格的时间步与原始秒数一致对齐：
  - 与 `seconds_per_chunk`、`position_id_per_seconds` 联动，确保 `position_ids` 沿时间维度递增。
  - 在模型侧，基于此信息构造 3D RoPE 索引与注意力掩码。

## 实操建议

- 统一配置：
  - 保持处理器的 `patch_size/temporal_patch_size/merge_size` 与模型配置一致；否则会出现网格形状不匹配。
- 分辨率调优：
  - `size.shortest_edge/longest_edge` 与 `merge_size` 决定 token 总数；显存不足时提高 `merge_size` 或降低 `longest_edge`。
- 时间分块：
  - `seconds_per_chunk` 在长视频场景尤为关键；应同时调整 `position_id_per_seconds` 以匹配时序位置编码步长。

