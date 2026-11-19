# 项目结构说明

本文档概述当前仓库的主要目录、关键入口和评测数据流。项目整体是 **4DWorldBench** 的评测代码，用于对 3D/4D world generation 模型进行感知质量、条件对齐、物理真实性和 4D 一致性等维度的评估。

## 顶层结构

```text
.
├── README.md                         # 项目介绍、安装方式、评测维度和数据 JSON 格式说明
├── requirements.txt                  # Python 依赖列表
├── scripts/                          # 单次、批量、冒烟和专项评测的 Shell 脚本
├── runner.py                         # 核心评测调度器：读取数据集、生成维度 JSON、动态加载指标模块
├── adaptive_qa_generator.py          # 缺失问答项时的自动问题生成逻辑
├── question_cache_manager.py         # 问题缓存管理工具
├── metric_compatibility.py           # 指标兼容性相关辅助逻辑
├── convert_videos_to_dataset.py      # 视频数据转换为评测数据集 JSON 的工具
├── metric/                           # 评测指标实现与第三方指标依赖
├── dimension_description_json/        # 各评测维度描述文件，以及部分模型评测结果 JSON
├── 4DWorldBench/                     # 额外的维度缓存数据
├── match_tools/                      # 结果维度匹配、平均分和最终得分计算工具
├── caption_method/                   # 视频/图像 caption 相关方法
├── qa_alignment_questions/           # 对齐类评测问题输出目录
├── checkpoints/                      # 模型权重或检查点目录
├── DROID-SLAM/                       # 外部 DROID-SLAM 代码，用于位姿/重投影等几何一致性评测
├── lietorch/                         # 外部 Lie group PyTorch 扩展，DROID-SLAM 等依赖会使用
└── droid.pth                         # DROID-SLAM 相关预训练权重
```

## 核心运行入口

### `scripts/evaluate.sh`

`scripts/evaluate.sh` 是推荐的单次评测入口。它接收以下主要参数：

- `--dataset_json`：输入数据集 JSON。
- `--model`：要评测的模型名称。
- `--dimension`：要运行的指标模块名称。
- `--prompt`：可选的 prompt 覆盖。
- `--device`：运行设备，如 `cuda:0`。

脚本会清理 SOCKS 代理相关环境变量，然后调用：

```bash
python runner.py --dataset_json ... --model ... --dimension ... --device ...
```

### `runner.py`

`runner.py` 是核心调度器，主要职责如下：

1. 初始化 import 路径，并将本地 `metric` 包兼容映射到部分旧代码使用的 `vbench2` 命名。
2. 读取数据集 JSON，将多模型、多条件、多 prompt 的结构展开为评测 item。
3. 按 `--model` 过滤评测数据。
4. 将当前维度的数据写入 `dimension_description_json/<dimension>.json`。
5. 对需要问答辅助信息的维度，调用 `AdaptiveQAGenerator` 自动补充缺失的 `auxiliary_info`。
6. 动态导入 `metric.<dimension>` 模块，并调用其中的 `compute_*` 函数完成评测。

除了数据集模式，`runner.py` 也支持通过 `--video_path` 进行单视频评测；`physics_realism` 还支持 `--retry_nulls` 对已有结果中的空回答重试。

## 批量运行脚本

`scripts/run_*.sh` 用于固定维度或批量评测：

- `scripts/run_all_perceptual.sh`：批量运行感知质量维度，例如 `perceptual_clip_aesthetic_metrics`、`perceptual_fastvqa`。
- `scripts/run_all_alignment.sh`：批量运行 Condition-4D Alignment 维度，例如事件、运动、关系、场景控制。
- `scripts/run_all_consistency.sh`：批量运行 4D Consistency 维度，例如 viewpoint consistency。
- `scripts/run_all_phygenbench.sh`：面向 PhyGenBench 数据集的批量评测入口。
- `scripts/run_clipiqa.sh`、`scripts/run_clipaesthetic.sh`、`scripts/run_fastvqa.sh`：单类感知指标脚本。
- `scripts/run_alignment_*.sh`：单个对齐维度脚本。
- `scripts/run_consistency_*.sh`：单个一致性维度脚本。
- `scripts/run_physics_realism.sh`：物理真实性评测脚本。

这些脚本通常会设置 `CUDA_VISIBLE_DEVICES`、`DEVICE`、`DATASET_JSON`、模型列表和维度列表，然后循环调用 `scripts/evaluate.sh`。

## 指标模块目录 `metric/`

`metric/` 是项目的主要实现目录。`runner.py` 会根据 `--dimension` 动态导入这里的同名模块，并寻找 `compute_*` 函数。

主要文件包括：

- `perceptual_clip_iqa_metrics.py`：CLIP-IQA / CLIP-IQA+ 感知图像质量评估。
- `perceptual_clip_aesthetic_metrics.py`：CLIP Aesthetic 美学质量评估。
- `perceptual_fastvqa.py`：FastVQA 视频质量评估。
- `alignment_attribute_control.py`：动态属性控制评估。
- `alignment_relationship_control.py`：动态空间关系控制评估。
- `alignment_motion_control.py`：运动顺序/运动控制评估。
- `alignment_event_control.py`：复杂事件或剧情对齐评估。
- `alignment_scene_control.py`：复杂场景/景观对齐评估。
- `alignment_camera_error_metrics.py`：相机运动误差评估。
- `physics_realism.py`：物理真实性推理评估。
- `consistency_viewpoint.py`：视角一致性/重投影误差评估，依赖 DROID-SLAM 和 lietorch。
- `consistency_motion_smoothness.py`：运动平滑性评估。
- `consistency_motion_qa.py`：基于问答的运动一致性评估。
- `consistency_style.py`：基于 VGG Gram matrix 的风格一致性评估。
- `base_metrics.py`：指标基类和通用图像预处理逻辑。
- `utils.py`、`distributed.py`、`hack_registry.py`：通用工具、分布式支持和注册兼容逻辑。

子目录：

- `metric/torchmetrics/`：封装 PSNR、SSIM、LPIPS、CLIP Score、CLIP-IQA 等 TorchMetrics 风格指标。
- `metric/third_party/`：项目集成的外部方法，包括 FAST-VQA、RAFT、DROID-SLAM、YOLO-World、Dense Match、VFIMamba、ViTDetector、arcface 等。

## 数据与结果目录

### `dimension_description_json/`

该目录同时承担两类职责：

1. 基础维度描述文件，例如：
   - `alignment_attribute_control.json`
   - `alignment_motion_control.json`
   - `physics_realism.json`
   - `perceptual_fastvqa.json`
   - `consistency_viewpoint.json`
2. 已生成的模型结果文件，命名通常类似：
   - `<dimension>__<model>__phygenbench_text_to_4d_results.json`

`runner.py` 在运行时也会把当前模型和维度的临时评测输入写到该目录下的 `<dimension>.json`。

### `4DWorldBench/dimension_description_json/cache/`

保存部分维度的缓存文件，例如 alignment、physics、consistency 相关缓存。它们通常用于减少重复生成问题或重复计算。

### `qa_alignment_questions/`

默认由 `runner.py --qa_dir` 使用，用于存放对齐类维度的问答文件。当前代码中 `ensure_questions_for_dimension()` 会保证目录存在，并按维度生成目标路径。

## 工具目录 `match_tools/`

`match_tools/` 用于评测结果后处理和最终分数计算：

- `cal_final_score.py`：从提交 zip 中读取各维度 JSON，计算 creativity、commonsense、controllability、human fidelity、physics 和总分。
- `compute_dimension_averages.py`：计算维度平均分。
- `i2a_dim_match.py`、`t2a_dim_match.py`、`v2a_dim_match.py`：不同任务类型到评测维度的匹配逻辑。
- `constant.py`：任务分组和维度列表常量。

## Caption 与 QA 相关模块

- `caption_method/VideoImg_caption.py`：视频/图像 caption 生成相关逻辑。
- `adaptive_qa_generator.py`：对缺少 `auxiliary_info` 的 JSON item 自动生成评测问题，支持并发处理，并通过环境变量控制并发和失败阈值：
  - `QA_NUM_WORKERS`
  - `QA_REQUEST_DELAY`
  - `QA_MAX_CONSECUTIVE_FAILURES`
- `question_cache_manager.py`、`scripts/manage_question_cache.sh`：问题缓存管理脚本和命令入口。

## 第三方依赖目录

### `DROID-SLAM/`

外部 DROID-SLAM 项目代码，包含：

- `droid_slam/`：核心 SLAM 模块。
- `src/`：CUDA/C++ 扩展。
- `calib/`、`data/`、`tools/`、`evaluation_scripts/`：标定、数据、工具和评估脚本。

本项目中的 `consistency_viewpoint.py` 和部分几何一致性指标会依赖这类能力。

### `lietorch/`

Lie group PyTorch 扩展，包含：

- `lietorch/`：Python 包和 C++/CUDA 扩展源码。
- `examples/`：PGO、registration、RGB-D SLAM 等示例。
- `setup.py`、`pyproject.toml`：安装配置。

它通常作为 DROID-SLAM 和 SE(3)/Sim(3) 几何计算的底层依赖。

## 评测数据流

典型运行链路如下：

```text
dataset_json
  -> scripts/evaluate.sh
  -> runner.py
  -> read_dataset_json()
  -> 按 model 过滤 item
  -> 写入 dimension_description_json/<dimension>.json
  -> 可选：adaptive_qa_generator.py 补充 auxiliary_info
  -> 动态导入 metric/<dimension>.py
  -> 调用 compute_<dimension>()
  -> 输出分数和 details
  -> 结果文件写回 dimension_description_json/ 或由指标模块内部保存
```

输入数据集 JSON 的核心结构是：

```json
{
  "dataset_info": {
    "base_path": "/path/to/dataset",
    "model_type": "text-to-3D | image-to-4D | video-to-4D",
    "condition_type": "text | image | video"
  },
  "models": [
    {
      "model_name": "your_model_name",
      "conditions": [
        {
          "condition_meta_info": "alignment_motion_control",
          "prompts": [
            {
              "prompt_id": "unique_id",
              "condition_caption": "condition description",
              "generated_videos": [
                "/path/to/generated/video.mp4"
              ],
              "auxiliary_info": []
            }
          ]
        }
      ]
    }
  ]
}
```

`runner.py` 会将其转换为指标模块更容易消费的列表格式：

```json
[
  {
    "dimension": ["alignment_motion_control"],
    "video_list": ["/path/to/generated/video.mp4"],
    "prompt_en": "condition description",
    "auxiliary_info": []
  }
]
```

## 开发和扩展新指标

新增一个评测维度通常需要：

1. 在 `metric/` 下新增 `<dimension>.py`。
2. 在模块中实现 `compute_<dimension>(json_dir, device, submodules_dict, **kwargs)`。
3. 让函数读取 `json_dir` 指向的维度 JSON，并返回 `(score, details)`。
4. 如需批量运行，在对应的 `scripts/run_all_*.sh` 或新增脚本中加入维度名称。
5. 如该指标不需要 QA 问题生成，可将维度名加入 `runner.py` 的 `SKIP_QUESTION_GENERATION_DIMENSIONS`。

注意：`runner.py` 的动态导入要求 `--dimension` 与 `metric/<dimension>.py` 文件名匹配。

## 环境和依赖注意事项

- README 和 `requirements.txt` 均建议先安装 CUDA 版 PyTorch，再安装项目依赖。
- 部分指标依赖外部权重、CUDA 扩展或大型第三方项目，例如 DROID-SLAM、lietorch、FAST-VQA、RAFT、YOLO-World。
- 对齐类和物理真实性评测可能调用 OpenAI 或其他多模态/语言模型 API，需要配置对应 API key。
- 批量脚本中当前硬编码了示例数据集路径 `/public/share/rogon_admin/4dworldbench/data/videos/dataset_jsons/phygenbench_text_to_4d.json`，迁移环境时需要修改或通过脚本参数传入。
