"""Shared resource configuration for the runner, benchmark and metrics."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO_ROOT = PROJECT_ROOT.parent / 'data' / 'videos'
DEFAULT_DATASET_NAME = 'phygenbench_text_to_4d.json'
DEFAULT_API_MODEL = 'deepseek-v4-flash'


def add_runtime_arguments(parser):
    parser.add_argument('--video-root', type=Path, help='Root directory for videos in the dataset')
    parser.add_argument('--weights-root', type=Path, help='Directory containing checkpoints/ and pretrained_weights/')
    parser.add_argument('--api-model', help='API model used for question generation and scoring')


def configure_runtime(args):
    if args.video_root is not None:
        os.environ['FOURDWORLDBENCH_VIDEO_ROOT'] = str(args.video_root.expanduser().resolve())
    if args.weights_root is not None:
        os.environ['FOURDWORLDBENCH_WEIGHTS_ROOT'] = str(args.weights_root.expanduser().resolve())
    if args.api_model is not None:
        os.environ['OPENAI_MODEL'] = args.api_model


def default_dataset_path() -> Path:
    default_root = configured_path('FOURDWORLDBENCH_VIDEO_ROOT', DEFAULT_VIDEO_ROOT)
    return configured_path('FOURDWORLDBENCH_DATASET_JSON', default_root / 'dataset_jsons' / DEFAULT_DATASET_NAME)


def configured_path(variable: str, default: Path) -> Path:
    return Path(os.environ.get(variable) or default).expanduser().absolute()


def dataset_video_root(dataset_path: Path) -> tuple[Path, bool]:
    explicit_root = bool(os.environ.get('FOURDWORLDBENCH_VIDEO_ROOT'))
    default = dataset_path.parent.parent if dataset_path.parent.name == 'dataset_jsons' else DEFAULT_VIDEO_ROOT
    return configured_path('FOURDWORLDBENCH_VIDEO_ROOT', default), explicit_root


def resolve_video_path(raw_path: str, declared_base: Path | None, video_root: Path, dataset_dir: Path, explicit_root: bool) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        if path.is_file() and (not explicit_root or path.resolve().is_relative_to(video_root.resolve())):
            return path
        if declared_base is None or not path.is_relative_to(declared_base):
            return path
        return video_root / path.relative_to(declared_base)
    relative = path
    if declared_base is not None and not declared_base.is_absolute() and path.is_relative_to(declared_base):
        relative = path.relative_to(declared_base)
    mapped = video_root / relative
    if explicit_root or mapped.is_file():
        return mapped
    # Standalone manifests can use video paths relative to the manifest itself.
    sibling = dataset_dir / path
    if sibling.is_file():
        return sibling
    if declared_base is not None:
        if declared_base.is_absolute():
            legacy = declared_base / path
            if legacy.is_file():
                return legacy
        else:
            prefix = declared_base.parts
            for parent in (dataset_dir, *dataset_dir.parents):
                if prefix and parent.parts[-len(prefix):] == prefix:
                    legacy = parent.joinpath(*('..' for _ in prefix), path).resolve()
                    if legacy.is_file():
                        return legacy
    return mapped


def checkpoints_dir() -> Path:
    return configured_path('FOURDWORLDBENCH_WEIGHTS_ROOT', PROJECT_ROOT) / 'checkpoints'


def keye_model_path() -> Path:
    return configured_path('KEYE_MODEL_PATH', checkpoints_dir() / 'Keye-VL-1_5-8B')


def droid_weights_path() -> Path:
    return configured_path('FOURDWORLDBENCH_DROID_WEIGHTS', checkpoints_dir() / 'droid.pth')


def vfimamba_weights_path(name: str) -> Path:
    return configured_path('FOURDWORLDBENCH_VFIMAMBA_WEIGHTS', checkpoints_dir() / name / 'model.pkl')


def fastvqa_weights_path(filename: str) -> Path:
    root = configured_path('FOURDWORLDBENCH_WEIGHTS_ROOT', PROJECT_ROOT)
    legacy_dir = configured_path('FOURDWORLDBENCH_WEIGHTS_DIR', root / 'pretrained_weights')
    return configured_path('FOURDWORLDBENCH_FASTVQA_WEIGHTS', legacy_dir / Path(filename).name)


def api_model_name() -> str:
    return os.environ.get('OPENAI_MODEL') or DEFAULT_API_MODEL
