# Qwen2‑VL 开发指南：总览与架构

本文面向在 `transformers.models.qwen2_vl` 目录下开发与维护 Qwen2‑VL 的工程师，概述整体架构、关键组件关系与数据流。后续文档会分别深入 Config、Modeling、Processor、Image/Video Processing。

## 架构总览

- 组件分层
  - `configuration_qwen2_vl.py`：定义配置对象 `Qwen2VLConfig`，含 `text_config`(`Qwen2VLTextConfig`) 与 `vision_config`(`Qwen2VLVisionConfig`)；并声明多模态特殊 token id。
  - `modeling_qwen2_vl.py`：实现视觉 Transformer、文本解码器与组合模型：
    - 视觉侧：`Qwen2VisionTransformerPretrainedModel`（PatchEmbed / Rotary / Attention / MLP / PatchMerger）。
    - 文本侧：`Qwen2VLTextModel`（DecoderLayer / 多头注意力 / RoPE）。
    - 顶层：`Qwen2VLModel` 与 `Qwen2VLForConditionalGeneration`（融合视觉/文本，训练与生成）。
  - 预处理：
    - 图像：`image_processing_qwen2_vl.py`（常规）与 `image_processing_qwen2_vl_fast.py`（PyTorch/TorchVision 快速实现）。
    - 视频：`video_processing_qwen2_vl.py`（基于 `BaseVideoProcessor`）。
  - Processor：`processing_qwen2_vl.py` 将 tokenizer、image/video processor 打包为一个入口（`Qwen2VLProcessor`）。

- 数据流（推理）
  - 文本侧：`Qwen2VLProcessor.__call__` 先将文本中的 `<|image_pad|>`/`<|video_pad|>` 扩展为占位符数量（由图像/视频 patch 数决定），再调用 tokenizer 产出 `input_ids`、`attention_mask`。
  - 视觉侧：image/video processor 预处理为 `pixel_values`/`pixel_values_videos` 与对应 `grid_thw`（按 T/H/W patch 网格）。
  - 模型融合：`Qwen2VLModel.forward` 将视觉特征嵌入到 `inputs_embeds` 的占位位置，按输入/网格计算 3D RoPE `position_ids` 与 `rope_deltas`，调用文本模型解码，输出 hidden states。
  - 语言头：`Qwen2VLForConditionalGeneration.forward` 在顶层接入 `lm_head` 得到 `logits`，可选 `labels` 计算 loss。

- 生成（Generation）特殊逻辑
  - `prepare_inputs_for_generation`：在 pre‑fill 阶段计算一次 3D RoPE 与 `rope_deltas`；随后步进阶段基于 cache_position 和 `rope_deltas` 生成正确的 position ids；且在步进阶段不再传入视觉像素（提升性能）。

## 关键设计要点

- 多模态 3D RoPE
  - 视觉 patch 序列按 `(T, H, W)` 三个维度分别计算 RoPE 并拼接到 head_dim；文本部分仍为 1D RoPE，但在统一的 3D 索引空间中沿用同一索引（T/H/W 相同）。

- Patch 与占位替换
  - 图像/视频经预处理变为二维 patch 序列与 `grid_thw`。Processor 把 `<|image_pad|>`/`<|video_pad|>` 在原始文本里替换为等量占位符，保证 `input_ids` 中多模态 token 计数与视觉特征长度一致。

- 视觉特征注入
  - 在 `Qwen2VLModel.forward` 通过 `masked_scatter` 将视觉特征注入 `inputs_embeds` 的占位位置，随后走文本解码器流水线。

- 注意力实现切换
  - 文本侧支持 `eager`/SDPA/FA2；视觉侧块在 FA2 下使用 `cu_seqlens` 处理变长序列，否则按分块循环计算。

更多细节请参考本目录下其他开发指南：Config、Modeling、Processor、Image/Video Processing。

## 加载总览与最佳实践

- 统一加载入口
  - 使用 `AutoConfig.from_pretrained(repo_or_path)` 加载 `Qwen2VLConfig`（由 `model_type = "qwen2_vl"` 路由）。
  - 使用 `AutoProcessor.from_pretrained(repo_or_path)` 组装 `Qwen2VLProcessor`，其内部依次加载 tokenizer、`AutoImageProcessor` 与 `AutoVideoProcessor`。

- 关键类级属性
  - Config：`model_type`、属性委托（如注意力实现与 RoPE 到 `text_config`）。
  - Vision processors：`model_input_names` 与 `valid_kwargs` 声明输出键与参数白名单。
  - Processor：`attributes`/`*_processor_class`/`tokenizer_class` 用于子组件路由与组合。

- 字段与形状对齐
  - 保持 `vision_config` 的 `patch_size`/`temporal_patch_size`/`merge_size` 与各处理器参数一致；
  - 文本中的 `<|image_pad|>`/`<|video_pad|>` 占位符数量必须与视觉 patch 总数一致；
  - 模型前向按各处理器的 `model_input_names` 接收张量键（如 `pixel_values`, `image_grid_thw`, `pixel_values_videos`, `video_grid_thw`）。

- 兼容性与 BC 提示
  - 若仅提供 `size = {shortest_edge, longest_edge}`，处理器在加载时会映射到 `min_pixels`/`max_pixels`；
  - 缺失子组件配置时使用默认值并警告，建议仓库提供 `config.json` 与 `preprocessor_config.json` 以保证完整加载；
  - 扩展上下文长度优先通过 `rope_scaling`（`type`/`factor`）实现，并在文本侧正确读取。

- 发布与复现实践
  - 仓库建议包含：`config.json`、`tokenizer.json`、`preprocessor_config.json`；
  - 当修改处理器参数（如 `temporal_patch_size`/`merge_size`）时，同步更新 `vision_config` 与预处理器默认值；
  - 用以下最小示例验证加载与推理：
    - `config = AutoConfig.from_pretrained(repo)`
    - `processor = AutoProcessor.from_pretrained(repo)`
    - `model = AutoModelForCausalLM.from_pretrained(repo)`
    - `inputs = processor(images=..., text=..., videos=..., return_tensors="pt")`
    - `out = model(**inputs)` / 生成：`model.generate(**inputs)`。

