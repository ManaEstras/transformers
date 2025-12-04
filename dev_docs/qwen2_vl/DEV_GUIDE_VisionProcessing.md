# Qwen2‑VL 开发指南：图像与视频预处理

本指南涵盖 `image_processing_qwen2_vl.py`、`image_processing_qwen2_vl_fast.py` 与 `video_processing_qwen2_vl.py` 的主要逻辑、参数约束与与模型/Processor 的接口关系。

## 文件与类

- 图像（常规）
  - `Qwen2VLImageProcessor`
    - 参数：`min_pixels`, `max_pixels`, `patch_size`, `temporal_patch_size=1`, `merge_size`
    - 能力：分组批量 resize（`smart_resize`），rescale/normalize，patch/merge，输出 `pixel_values`, `image_grid_thw`

- 图像（fast 版本）
  - `Qwen2VLImageProcessorFast`
    - 参数：同上，同时包含默认的 `do_resize`, `do_rescale`, `do_normalize`, `do_convert_rgb`
    - 注意：不再处理视频（收到视频会发出 warning）；
    - 能力：分组按形状批量 `smart_resize`，随后 rescale/normalize，生成 patches 与 `image_grid_thw`

- 视频
  - `Qwen2VLVideoProcessor`
    - 参数：`min_pixels`, `max_pixels`, `patch_size`, `temporal_patch_size`, `merge_size`, `min_frames`, `max_frames`
    - 能力：帧采样（`sample_frames`），分组批量 resize（`smart_resize`），rescale/normalize，patch/merge，输出 `pixel_values_videos`, `video_grid_thw`

## 类级属性与加载行为（Class Attributes & Loading）

- `model_input_names`
  - 图像处理器（含 fast）：`["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]`
  - 视频处理器：`["pixel_values_videos", "video_grid_thw"]`
  - 作用：声明本处理器会产出的张量字段名；Processor/模型在打包与前向时依赖这些键进行对齐。

- `valid_kwargs` 与默认属性
  - 显式定义允许的关键字参数结构（如 `min_pixels`, `max_pixels`, `patch_size`, `temporal_patch_size`, `merge_size` 等）；
  - fast 版本还定义了 `DefaultFastImageProcessorKwargs` 相关属性以便批量加速；
  - 作用：`from_pretrained` 加载配置后进行参数校验与绑定，保证运行时行为稳定。

- 加载与兼容
  - `AutoImageProcessor.from_pretrained(repo_or_path)` / `AutoVideoProcessor.from_pretrained(repo_or_path)`：
    - 读取 `preprocessor_config.json` 并映射到具体处理器类；
    - 若仅提供 `size`，构造时会将 `min_pixels`/`max_pixels` 映射为 `size["shortest_edge"]`/`size["longest_edge"]`（BC）；
    - 参数缺失时使用类默认值，必要时发出警告。

- 与 Processor 的协同
  - `Qwen2VLProcessor` 在 `from_pretrained` 时会同时路由并加载图像/视频处理器；
  - 输出的 `model_input_names` 字段由各处理器提供，Processor 统一打包进 `BatchFeature`；
  - 建议：统一在模型仓库发布 `preprocessor_config.json`，确保 Auto 类在加载时选择到正确实现。

> 实战提示：当你新增或调整处理器参数（如 `temporal_patch_size`/`merge_size`），请同步更新模型侧 `vision_config`，并在处理器 `preprocessor_config.json` 中写入最新默认值，以便 `AutoImageProcessor`/`AutoVideoProcessor` 与 Processor 一致路由与加载。


## 共同概念与数据形状

- `smart_resize(height, width, min_pixels, max_pixels)`
  - 保持长宽比，约束像素面积在 `[min_pixels, max_pixels]` 区间；返回调整后的 `(height, width)`。

- patches 与网格
  - 输入形状（统一视图）：`[B, C, T, H, W]`（图像视作 `T=1`）。
  - patch 提取：
    - 时间维步长：`temporal_patch_size`
    - 空间步长：`patch_size`（高/宽）
  - 合并：`merge_size` 决定相邻 patch 的聚合比例（影响总 patch 数与序列长度）。
  - 输出网格：`grid_thw = (T_grid, H_grid, W_grid)`；用于 RoPE 与占位符替换。

- 归一化
  - `do_rescale` 与 `do_normalize` 控制是否将输入转为 `[0,1]` 并按均值/方差标准化；
  - RGB 转换：`do_convert_rgb` 统一通道顺序。

## 图像预处理流程（简要）

1) 按 `(H, W)` 分组；批量 `smart_resize` 至满足像素约束。
2) 转换为张量并按需 rescale/normalize。
3) 以 `temporal_patch_size=1` 与 `patch_size` 进行空间 patch；按 `merge_size` 合并。
4) 输出：`pixel_values`、`image_grid_thw`（每张图的网格集合）。

## 视频预处理与帧采样

- `sample_frames(num_frames=None, fps=None, ...)`
  - 支持两种方式：
    - 指定目标 `num_frames`，进行均匀采样；
    - 指定 `fps`，按视频时长采样帧数。
  - 约束：采样后的帧数需可被 `temporal_patch_size` 整除；必要时进行向下调整或内插补齐。
  - 最终帧数也受 `min_frames`/`max_frames` 限制。

- 视频 `_preprocess`
  1) 对采样后的每段视频根据 `(H, W)` 分组并批量 `smart_resize`；
  2) rescale/normalize；
  3) 按 `temporal_patch_size` 与 `patch_size` 切分；按 `merge_size` 合并；
  4) 输出：`pixel_values_videos`、`video_grid_thw`。

## Patch 数计算接口

- `get_number_of_image_patches(height, width, ...)`
  - 先做一次理论 `smart_resize` 得到 `(H', W')`；
  - 计算 `H_grid = H' // patch_size`, `W_grid = W' // patch_size`；
  - 考虑 `merge_size` 后返回总 patch 数。

- `get_number_of_video_patches(num_frames, height, width, ...)`
  - 对视频同理，但需计算 `T_grid = (num_frames // temporal_patch_size)`；
  - 结合 `merge_size` 返回总 patch 数。

## 与 Processor/Model 的契约

- Processor 使用 `get_number_of_*_patches` 决定占位符替换长度；
- Model 在 `get_rope_index` 使用 `grid_thw` 计算 3D RoPE 索引；
- 若 `merge_size`/`patch_size`/`temporal_patch_size` 不一致会导致占位符计数或 RoPE 索引错位，需保持处理器与模型一致性。

## 常见问题与建议

- 帧数不可整除：
  - 调整 `num_frames` 或 `fps`，确保 `num_frames % temporal_patch_size == 0`；
  - 或在采样后裁剪到可整除的帧数。

- 占位符数量不匹配：
  - 确认 `smart_resize` 与模型侧预期一致；
  - 检查 `merge_size` 是否改变了总 patch 数。

- Fast 版本差异：
  - `Qwen2VLImageProcessorFast` 不处理视频；仅保证图像处理一致性与速度。

## 开发建议

- 批量 `smart_resize`：
  - 按 `(H, W)` 分组批量插值可减少重复开销，提高吞吐。

- 参数一致性：
  - 对齐处理器与 `vision_config` 的 patch/merge 参数；
  - 测试多分辨率与多帧率场景，确保 `grid_thw` 与占位替换稳定。

