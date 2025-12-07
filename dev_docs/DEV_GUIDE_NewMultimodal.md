# 多模态模型开发指南（MyModelVL / MyModelOmni）

本指南面向普通算法/工程同学，目标是按本文档即可在 Transformers 仓库新增一个规范的多模态模型及其配套 Config、Processor，并能完成加载与端到端推理。以两个典型形态为例：
- MyModelVL：视觉 + 文本（可选音频），倾向于“VL”类的多模态理解与生成。
- MyModelOmni：多模态全集成（图像/视频/音频/文本），倾向于“Omni”类。

内容涵盖：必需定义项、加载原理、推理原理、类级属性与加载行为、字段与形状对齐、以及发布与复现的最佳实践。

## 文件与目录结构规范

- 源码路径：`src/transformers/models/mymodel_vl` 或 `src/transformers/models/mymodel_omni`
  - `configuration_mymodel_vl.py` / `configuration_mymodel_omni.py`
  - `modeling_mymodel_vl.py` / `modeling_mymodel_omni.py`（或 `modular_*.py`，若拆分子模块）
  - `processing_mymodel_vl.py` / `processing_mymodel_omni.py`
- 资源路径：`preprocessor_config.json`、`config.json`、权重文件（位于模型卡）
- 文档与示例：`dev_docs` 目录下可以放开发指南与参考示例；`examples/` 里可提供可运行脚本。

## 必需定义项（总览）

- Config设置：
  - 声明 `model_type`（唯一字符串）
  - 关键字段（维度、层数、heads、rope_scaling、滑窗、特殊 token 索引）
  - 对于 Omni：`sub_configs`（如 `audio_config`、`vision_config`、`text_config`）与 `attribute_map`（旧字段名兼容）
- 处理器（Processor）：
  - 继承 `ProcessorMixin`，提供 `from_pretrained/save_pretrained`
  - 定义多模态输入与占位符替换（`replace_multimodal_special_tokens`）
  - 声明 `model_input_names` 并与模型 `forward/generate` 对齐
  - 提供图像/视频/音频参数类（如 `MyModelVideosKwargs`、`MyModelImagesKwargs`、`MyModelProcessorKwargs`）
- 建模（Modeling）：
  - 继承 `PreTrainedModel` 或相应的 `*ForConditionalGeneration`
  - 定义文本/视觉/音频分支模块与融合逻辑
  - 构造 RoPE（文本/视觉/音频）与 4D 注意力掩码（视觉）、滑窗策略
  - 实现 `generate` 所需的 KV 缓存与增量推理

## 类级属性与加载行为（Class Attributes & Loading）

（说明：此处保留章节标题以便目录导航；详细内容已拆分并并入下方的各自设计章节。）

## 配置设计（Config）

- MyModelVL：
  - `MyModelVLConfig(PretrainedConfig)`：
    - 关键字段：`vocab_size`、`hidden_size`、`num_hidden_layers`、`num_attention_heads`、`rope_theta`、`rope_scaling`、`use_sliding_window`、`sliding_window`、`max_window_layers`、`layer_types`
    - 视觉相关：`patch_size`、`merge_size` 等；若支持视频：`temporal_patch_size`、时序参数
    - 特殊 token：图像/视频/audio 的起止/占位符索引（如需要）
    - `model_type = "mymodel_vl"`
- MyModelOmni：
  - `MyModelOmniConfig(PretrainedConfig)`：聚合子配置并声明：
    - `audio_config`、`vision_config`、`text_config`（可为 dict 或子配置类）
    - 特殊 token：`audio_token_index`、`image_token_index`、`video_token_index` 等；音频起止、用户 token；时序：`position_id_per_seconds`、`seconds_per_chunk`
    - `attribute_map`、`sub_configs` 维护别名与子配置类型
    - `model_type = "mymodel_omni"`

建议：
- 将 RoPE 与滑窗参数集中在 Text/Omni 主配置中；视觉/音频编码器在自身配置中只保留必要维度参数。

### 类级属性与加载行为要点（Config）：
- `model_type` 唯一且注册到 `AutoConfig/AutoModel`，`from_pretrained` 依据 `config.json` 选择实现类。
- `attribute_map` 提供老字段到新字段的兼容映射，旧模型卡加载时自动应用。
- `sub_configs` 统一实例化 `audio/vision/text` 子配置，确保多分支参数一致。
- RoPE/滑窗：`rope_scaling`（含 `mrope_section/rope_type/type/long_factor/low_freq_factor/high_freq_factor`）、`use_sliding_window/sliding_window/max_window_layers/layer_types` 决定位置编码与注意力形态，需校验层数一致性。
- 维度与词表：`vocab_size/hidden_size/num_hidden_layers/num_attention_heads/rope_theta` 等构成模型基础结构。
- 特殊 token 索引需与 Processor 保持一致（图像/视频/音频起止/占位符/用户 token）。
- `initializer_range` 等初始化超参仅影响新增或未加载权重的模块，不改变已加载权重。

## 处理器设计（Processor）

- 继承 `ProcessorMixin`，命名为 `MyModelVLProcessor` / `MyModelOmniProcessor`
- 提供可选的参数类：
  - `MyModelVideosKwargs`：`fps`、`use_audio_in_video`、`seconds_per_chunk`、`position_id_per_seconds`、`min_pixels`、`max_pixels`、`patch_size`、`temporal_patch_size`、`merge_size`、`size.shortest_edge/longest_edge`
  - `MyModelImagesKwargs`：`min_pixels`、`max_pixels`、`patch_size`、`merge_size`、`size.shortest_edge/longest_edge`
  - `MyModelProcessorKwargs`：`chat_template`、`force_enable_chat_template`、文本最大长度与音频采样率等
- 核心方法：
  - `apply_chat_template(messages, ...)` 生成规范对话文本；
  - `replace_multimodal_special_tokens(text, images, videos, audios, ...)` 将 `<image>/<video>/<audio>` 占位符替换为真实序列并插入 start/end token；维护连续的 `position_ids`；
  - 输出与模型契约对齐：`model_input_names` 中声明所有字段。

类级属性与加载行为要点（Processor）：
- `ProcessorMixin`：`from_pretrained` 读取并合并 `preprocessor_config.json` 与显式传参；`save_pretrained` 写回当前参数。
- `model_input_names`：与模型侧字段名集合一致（文本、视觉 THW/时序、音频特征与长度等），`forward/generate` 严格依赖。
- 聊天模板/特殊 token：`apply_chat_template` 与 `replace_multimodal_special_tokens` 保证角色格式与占位符替换一致，必要时可 `force_enable_chat_template=True`。
- 参数覆盖一致性：`videos_kwargs/images_kwargs/processor_kwargs` 可覆盖磁盘配置，需与 Config 的网格/时序/采样率等保持一致，避免形状不匹配。

### 开发细节（主要方法对照 Qwen2‑VL）

- `__call__(text=None, images=None, videos=None, audios=None, return_tensors="pt", **kwargs)`
  - 输入：
    - 文本序列（`text` 或 `input_texts`）、图像（`images`）、视频（`videos`）、可选音频（`audios`）。
    - 可选处理参数：`min_pixels/max_pixels`、`patch_size/temporal_patch_size/merge_size`、`fps/num_frames`、`seconds_per_chunk/position_id_per_seconds` 等。
  - 步骤：
    - 图像/视频预处理：调用相应处理器得到 `pixel_values`/`pixel_values_videos` 与 `image_grid_thw`/`video_grid_thw`。
    - 计算占位数量：依据每张图/每个视频的 patch 数（考虑 `merge_size` 与时序采样）确定替换长度。
    - 文本占位替换：将 `<image>`/`<video>`/`<audio>` 标记替换为等量占位符序列（使用相应 pad/占位 token），保证长度与特征一致。
    - 分词：调用 tokenizer 处理替换后的文本，得到 `input_ids`、`attention_mask` 等。
    - 打包输出：构造 `BatchEncoding/BatchFeature`，包含模型侧所需字段与预处理张量。
  - 输出：
    - 文本：`input_ids`、`attention_mask`、可选 `position_ids`。
    - 视觉：`pixel_values`、`image_grid_thw`；视频：`pixel_values_videos`、`video_grid_thw`；必要时提供 `video_second_per_grid`（时序网格到秒的映射）。
    - 音频：`input_features`、`feature_attention_mask`、`audio_feature_lengths`（如支持音频）。

- `_get_num_multimodal_tokens(image_grid_thw=None, video_grid_thw=None, patch_size=None, temporal_patch_size=None, merge_size=None, ...)`
  - 作用：根据处理后网格/参数，计算图像/视频对应需要替换的占位符数量，用于文本中占位符展开。
  - 说明：
    - 图像：依据 `H/W` 方向 patch 数与 `merge_size` 决定最终序列长度；
    - 视频：在图像基础上乘以时间维的 patch 数（由 `temporal_patch_size` 与帧采样策略决定）。
    - 对齐：返回值必须与视觉 encoder 输出的序列长度一致，否则抛错。

- `post_process_image_text_to_text(token_ids_or_logits, skip_special_tokens=True, **kwargs)`
  - 作用：将模型输出的 token id 序列或 logits 通过 tokenizer decode 为人类可读文本。
  - 细节：
    - 可选跳过特殊 token（图像/视频/音频占位、起止 token）；
    - 支持批量 decode 与拼接；
    - 与 `apply_chat_template` 的分隔符保持一致，确保对话输出结构清晰。

- `replace_multimodal_special_tokens(text, images=None, videos=None, audios=None, force_enable_chat_template=False, ...)`
  - 作用：解析 `<image>/<video>/<audio>` 标记，按各自 patch/帧数扩展占位序列，并插入 start/end 等特殊 token；
  - 一致性：
    - 计算并返回占位 mask，用于模型侧 `get_placeholder_mask` 校验；
    - 按 `seconds_per_chunk/position_id_per_seconds` 生成连续 `position_ids`（视频/音频），保持与模型时序对齐。

## 视觉/音频处理与位置编码

- 视觉（图像/视频）：
  - THW 网格：`(T, H, W)`，由 `patch_size/temporal_patch_size/merge_size` 与 `size.shortest_edge/longest_edge`、`min_pixels/max_pixels` 决定；
  - 3D RoPE：根据 THW 构造位置索引；
  - 4D 因果注意力掩码：保证时间因果与空间局部性；需与展平序列一致。
- 音频：
  - 特征（如 mel）与长度掩码：输出 `input_features`、`feature_attention_mask`、`audio_feature_lengths`；
  - 采样率/帧长与训练一致；长度掩码用于裁剪与注意力屏蔽。
- 时序映射：
  - 通过 `seconds_per_chunk` 与 `position_id_per_seconds` 保持视频/音频时间步的 `position_ids` 线性递增；处理器提供 `video_second_per_grid` 字段。

## 建模实现与推理（Modeling & Inference）

- 文本干路：
  - 嵌入 → 多层解码器（自注意力 + MLP），支持 `sliding_attention/full_attention`；
  - RoPE：`rope_scaling` 控制长上下文；
  - 输出：`ForConditionalGeneration` 提供 logits 与 KV 缓存。
- 视觉分支：
  - 将 THW 网格编码为序列；视觉注意力使用 3D RoPE 与 4D 掩码；
- 音频分支：
  - 编码 mel/特征并输出与文本/视觉兼容的隐空间表示；支持长度掩码；
- 增量生成与 KV 缓存：
  - `generate` 需正确缓存 past key values；多模态场景下，保持缓存与新步位置编码一致。

### 核心类与职责（建议命名）

- `MyModelVisionEncoderPretrainedModel`
  - 组成：`PatchEmbed`、视觉侧 `RotaryEmbedding`、若干视觉 `Attention+MLP` Block、可选 `PatchMerger`。
  - 作用：将图像/视频像素转换为连续视觉 embedding 序列，并提供与 THW 网格对应的 RoPE/索引参数。

- `MyModelTextModel`
  - 组成：`embed_tokens`、若干解码层（Attention+MLP）、规范化层、文本侧 `RotaryEmbedding`。
  - 作用：语言 backbone，接收融合后的 `inputs_embeds` 与统一 `position_ids`，输出文本隐藏态。

- `MyModelVLModel`
  - 作用：顶层融合模型，负责：
    - 计算视觉特征与网格索引（T/H/W）；
    - 按占位符将视觉特征注入 `inputs_embeds`；
    - 统一计算 3D/1D RoPE 的 `position_ids` 与可选 `rope_deltas`；
    - 调用文本模型进行解码，返回 `hidden_states` 与可选 `past_key_values`。

- `MyModelVLForConditionalGeneration`
  - 作用：在顶层融合模型基础上接 `lm_head` 产出 `logits` 与 `loss`；覆盖生成相关方法以支持多模态位置 id 的 pre‑fill 行为。

### 关键模块与属性（参考实现）

- Patch 提取与合并
  - `PatchEmbed`：2D/3D 卷积提取 `(T,H,W)` patch 并线性映射至 `embed_dim`。
  - `PatchMerger`：根据 `merge_size` 对序列做合并与降采样，可含 `LayerNorm` + MLP。

- RoPE（旋转位置编码）
  - 文本侧：`RotaryEmbedding`（支持 `rope_scaling`），在 Attention 中应用到打包后的 Q/K。
  - 视觉侧：`VisionRotaryEmbedding` 与 `apply_rotary_pos_emb_vision`，按 `(T,H,W)` 维度分片施加。

- Attention 实现与后端
  - 文本：`eager_attention_forward`/SDPA/FA2，支持 `sliding_window` 与全局注意力的切换。
  - 视觉：视觉 `Attention` 支持 FA2 或分块计算，需与展平序列和掩码匹配。

### 初始化与输入输出（接口约定）

- 视觉编码（`get_image_features`/`get_video_features`）
  - 输入：
    - `pixel_values` / `pixel_values_videos`：形如 `[B, C, H, W]` 或 `[B, C, T, H, W]`（图像视作 `T=1`），由处理器保证形状。
    - `image_grid_thw`/`video_grid_thw`：每个样本的 `(T,H,W)` 网格集合，用于 RoPE 与占位校验。
  - 输出：
    - 视觉 embedding `[sum(patches), hidden_size]` 或按批次打包后的表示；
    - 可能的中间网格信息与 RoPE 索引供后续使用。

- 顶层融合 forward（`MyModelVLModel.forward`）
  - 输入：
    - `input_ids` 或 `inputs_embeds`（二者其一必填）。
    - `pixel_values`, `pixel_values_videos`（仅在 pre‑fill 阶段传入）。
    - `image_grid_thw`, `video_grid_thw`（视觉 RoPE 索引计算所需）。
    - `attention_mask`、`past_key_values`（生成时缓存）。
  - 主要步骤：
    - 通过 Processor 提供的文本与占位符 mask，确定注入位置；
    - 编码视觉特征，并在占位位置注入到 `inputs_embeds`；
    - 统一计算 3D/1D `position_ids` 与可选 `rope_deltas`；
    - 传递到文本模型得到 `hidden_states` 与可选 `past_key_values`。
  - 输出：`last_hidden_state`、`past_key_values`、`rope_deltas`（供步进阶段位置更新）。

- 条件生成 forward（`MyModelVLForConditionalGeneration.forward`）
  - 输入几乎与顶层融合 forward 相同，末端接 `lm_head`；
  - 输出：`logits`（必有）、`loss`（若给定 `labels`）、`rope_deltas`（用于后续步进）。

### 生成流程（Generation，强制约定）

- `prepare_inputs_for_generation`
  - pre‑fill 阶段：计算并返回 `position_ids` 与（可选）`rope_deltas`；视觉张量仅在此阶段传入模型。
  - 步进阶段：不再传入视觉张量；基于 `cache_position` 与（可选）`rope_deltas` 计算新的 `position_ids`；维持 `past_key_values` 缓存。

- `_expand_inputs_for_generation`
  - 处理视觉张量的批维扩展：依据样本内 `image/video` 数量（由特殊 token 统计）重复视觉特征，保证与 `input_ids` 批次对齐。

### 校验与占位（一致性保障）

- `get_placeholder_mask`
  - 校验文本占位符数量与视觉特征长度一致；不一致直接抛错，避免注入错位。

### 开发建议（与 Processor/Config 对齐）

- 视觉/文本维度对齐：
  - 确保 Processor 生成的 `grid_thw` 与模型侧 `PatchEmbed/merge_size` 一致。

- RoPE 一致性：
  - 文本与视觉的 3D/1D RoPE 索引使用同一上界；在融合模型中统一计算（例如 `get_rope_index`），修改策略时集中在该入口与对应 `RotaryEmbedding`。

- 注意力性能：
  - 大批次或长上下文建议启用 SDPA/FA2；否则需在默认实现里注意分块与内存占用。

### 类级属性与加载行为要点（Modeling）：

- 基类与自动映射：继承 `PreTrainedModel`/`*ForConditionalGeneration`，并通过 `model_type` 与注册系统完成 `AutoModel` 映射与权重装载。
- 生成与缓存：`generate` 使用 `past_key_values` 做增量生成；可覆写 `prepare_inputs_for_generation` 定制新步输入（含 `position_ids` 与掩码）。
- 掩码与位置编码：文本滑窗/全局注意力；视觉 3D RoPE（THW）与 4D 因果掩码；音频长度掩码（`feature_attention_mask/audio_feature_lengths`）。
- 常用方法/属性：`tie_weights`、`get_input_embeddings`、`resize_token_embeddings`、`gradient_checkpointing_enable()`、`base_model_prefix` 等影响权重组织与训练性能。

## 加载原理（from_pretrained 交互）

- 配置加载：
  - `config = MyModelConfig.from_pretrained(repo_or_path)` 读取 `config.json`；依据 `model_type` 选择实现类；
  - Omni 子配置以 dict/类实例加载后统一实例化。
- 处理器加载：
  - `processor = MyModelProcessor.from_pretrained(repo_or_path)` 读取 `preprocessor_config.json`；合并传入的 `kwargs`；
  - `model_input_names` 字段契约在处理器与模型侧一致。
- 模型加载：
  - `model = MyModelForConditionalGeneration.from_pretrained(repo_or_path)`；
  - 注意权重初始化参数对新增头或未加载模块的影响。

## 端到端示例

- MyModelVL（文本+图像）
  - 加载：`processor = MyModelVLProcessor.from_pretrained(...)`；`model = MyModelVLForConditionalGeneration.from_pretrained(...)`
  - 输入：`text` 含 `<image>` 占位符；`images=[...]`；
  - 调用：`inputs = processor(text=text, images=images, return_tensors="pt")` → `model.generate(**inputs, max_new_tokens=128)`
- MyModelOmni（文本+图像+视频+音频）
  - 加载：`processor = MyModelOmniProcessor.from_pretrained(...)`；`model = MyModelOmniThinkerForConditionalGeneration.from_pretrained(...)`
  - 输入：`text` 含多模态占位符；`images/videos/audios` 同时提供；
  - 调用：`inputs = processor(..., return_tensors="pt")` → `model.generate(**inputs, ...)`

## Auto 注册与模型卡发布

- 注册：
  - 在 Transformers 的注册系统中为 `model_type` 绑定到 `AutoConfig`/`AutoModel`；确保 `from_pretrained` 能正确解析。
- 模型卡：
  - 明确 `model_type`、子配置、关键 RoPE/滑窗参数、特殊 token 索引；
  - 推荐 Processor 参数（视频 FPS、chunk 长度、最短/最长边、patch/merge、音频采样率）。

## 开发检查清单（Checklist）

- Config：`model_type` 唯一；子配置与别名映射正确；RoPE/滑窗参数完整；
- Processor：`model_input_names` 与模型侧一致；占位符替换逻辑完备；聊天模板可用且可强制启用；
- Modeling：RoPE/掩码正确；支持 KV 缓存与增量生成；多模态融合路径稳定；
- 示例：可运行端到端；
- 发布：模型卡参数齐全，便于他人复现。
