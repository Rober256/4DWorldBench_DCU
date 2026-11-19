# 4DWorldBench 交付版运行说明

在已配置的 DCU Python 环境中运行 `scripts/evaluate.sh` 或 `benchmark_sampled_videos.py`。如需指定解释器，设置 `FOURDWORLDBENCH_PYTHON` 后执行 shell 脚本。

```bash
source /opt/dtk-26.04/env.sh
bash scripts/evaluate.sh --model ours_14bs2_720p --dimension perceptual_clip_iqa_metrics
python benchmark_sampled_videos.py --model ours_14bs2_720p --output-dir ./results/seed42
```

两条入口接收相同的运行参数。四个 `scripts/run_all_*.sh` 脚本接受可选的首个设备参数（如 `cuda:0`），其余参数传给 `scripts/evaluate.sh`。普通评测直接运行 `runner.py`。使用 `--hipprof` 开启性能剖析，或使用 `--hipprof-output <路径>` 开启并指定输出位置。

| 命令行参数 | 环境变量 | 默认行为 |
|---|---|---|
| `--dataset-json`（兼容 `--dataset_json`） | `FOURDWORLDBENCH_DATASET_JSON` | 视频根目录下的 `dataset_jsons/phygenbench_text_to_4d.json` |
| `--video-root` | `FOURDWORLDBENCH_VIDEO_ROOT` | 项目相邻的 `data/videos/`。对放在 `dataset_jsons/` 中的显式清单，从清单所在目录的上一级推断视频根目录 |
| `--weights-root` | `FOURDWORLDBENCH_WEIGHTS_ROOT` | 源码目录。根目录应包含 `checkpoints/` 和 `pretrained_weights/` |
| `--api-model` | `OPENAI_MODEL` | `deepseek-v4-flash`，用于问题生成和 Physics 评分 |

命令行优先于同名环境配置。若清单记录了旧的 `dataset_info.base_path`，评测入口会把旧基准下的相对路径映射到当前视频根目录。独立清单也可使用相对于清单文件位置的路径。单视频模式使用 `--video_path <文件>`。

权重布局为 `checkpoints/Keye-VL-1_5-8B/`、`checkpoints/droid.pth`、`checkpoints/VFIMamba/model.pkl` 和 `pretrained_weights/FAST_VQA_B_1_4.pth`（FastVQA 实际文件名来自所选模型的 YAML）。单项覆盖变量 `KEYE_MODEL_PATH`、`FOURDWORLDBENCH_DROID_WEIGHTS`、`FOURDWORLDBENCH_VFIMAMBA_WEIGHTS`、`FOURDWORLDBENCH_FASTVQA_WEIGHTS` 优先于权重根目录。旧的 `FOURDWORLDBENCH_WEIGHTS_DIR` 仍可指向 FastVQA 权重目录。CLIP、VGG、LPIPS 等依赖沿用现有预训练模型缓存机制。

问答类评测的服务地址和认证通过 `OPENAI_BASE_URL`、`OPENAI_API_KEY` 配置。

Shell 脚本集中在 `scripts/`。从其他目录运行时，可用脚本的绝对路径调用。评测参数中传入的相对路径仍以调用者当前目录为准。Physics 专项可用 `PHYSICS_QUESTION_CACHE` 指定已有问题缓存，用 `PHYSICS_PROFILE_ROOT` 指定输出目录。
