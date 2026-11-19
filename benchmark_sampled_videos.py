#!/usr/bin/env python3
"""Benchmark one deterministic sample from each of 16 ten-video intervals."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


from runtime_config import (
    PROJECT_ROOT, add_runtime_arguments, api_model_name, configure_runtime,
    dataset_video_root, default_dataset_path, droid_weights_path,
    keye_model_path, resolve_video_path,
)

DEFAULT_DATASET = default_dataset_path()
DEFAULT_MODEL = "ours_14bs2_720p"

LOCAL_DIMENSIONS = [
    "perceptual_clip_iqa_metrics",
    "perceptual_clip_aesthetic_metrics",
    "perceptual_fastvqa",
    "consistency_viewpoint",
    "consistency_motion_smoothness",
    "consistency_style",
]
QA_DIMENSIONS = [
    "alignment_attribute_control",
    "alignment_event_control",
    "alignment_motion_control",
    "alignment_relationship_control",
    "alignment_scene_control",
    "physics_realism",
    "consistency_motion_qa",
]
ALL_DIMENSIONS = LOCAL_DIMENSIONS + QA_DIMENSIONS

PHYSICS_PROTOCOL = {
    "PR_TARGET_QUESTIONS": "10",
    "KEYE_MAX_FRAMES": "48",
    "KEYE_VIDEO_TOKEN_BUDGET": "12288",
    "KEYE_VIDEO_MAX_PIXELS": "802816",
    "KEYE_MAX_NEW_TOKENS": "1536",
    "KEYE_CAPTION_THINKING": "0",
    "KEYE_USE_REFERENCE_PROMPT": "0",
    "KEYE_GUIDE_WITH_QUESTIONS": "0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", "--dataset_json", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--only", action="append", choices=ALL_DIMENSIONS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    add_runtime_arguments(parser)
    return parser.parse_args()


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def video_number(video_path: str) -> int:
    stem = Path(video_path).stem
    if not re.fullmatch(r"\d{4}", stem):
        raise ValueError(f"expected a four-digit video filename, received: {video_path}")
    return int(stem)


def load_model_rows(dataset_path: Path, model_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        root = json.loads(dataset_path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read dataset JSON {dataset_path}: {exc}") from exc

    dataset_info = root.get("dataset_info", {})
    declared_base_raw = dataset_info.get("base_path", "")
    declared_base = Path(declared_base_raw).expanduser() if declared_base_raw else None
    dataset_path = dataset_path.expanduser().absolute()
    video_root, explicit_root = dataset_video_root(dataset_path)

    model = next(
        (item for item in root.get("models", []) if item.get("model_name") == model_name),
        None,
    )
    if model is None:
        raise ValueError(f"model {model_name!r} was not found in {dataset_path}")

    rows: list[dict[str, Any]] = []
    for condition in model.get("conditions", []) or []:
        condition_name = condition.get("condition_meta_info", "")
        for prompt in condition.get("prompts", []) or []:
            caption = prompt.get("condition_caption", "")
            if not isinstance(caption, str) or not caption.strip():
                raise ValueError(f"empty prompt for prompt_id={prompt.get('prompt_id')!r}")
            for path in prompt.get("generated_videos", []) or []:
                resolved_video = resolve_video_path(path, declared_base, video_root, dataset_path.parent, explicit_root)
                item = {
                    "video_id": video_number(path),
                    "video": str(resolved_video),
                    "prompt": caption.strip(),
                    "prompt_id": prompt.get("prompt_id"),
                    "prompt_key": prompt.get("prompt_key", Path(path).stem),
                    "condition_meta_info": condition_name,
                    "condition_content": prompt.get("condition_content", ""),
                }
                rows.append(item)

    rows.sort(key=lambda item: item["video_id"])
    expected = list(range(1, 161))
    actual = [item["video_id"] for item in rows]
    if actual != expected:
        raise ValueError(
            "expected exactly one video for each numeric id 0001-0160; "
            f"received {len(rows)} rows with ids {actual[:5]}...{actual[-5:]}"
        )
    missing = [item["video"] for item in rows if not Path(item["video"]).is_file()]
    if missing:
        raise ValueError(f"{len(missing)} selected source video paths do not exist; first: {missing[0]}")
    return dataset_info, rows


def sample_rows(rows: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    intervals: list[dict[str, Any]] = []
    for interval_index in range(16):
        candidates = rows[interval_index * 10 : (interval_index + 1) * 10]
        choice = rng.choice(candidates).copy()
        choice["interval"] = interval_index + 1
        selected.append(choice)
        intervals.append(
            {
                "interval": interval_index + 1,
                "position_start": interval_index * 10 + 1,
                "position_end": (interval_index + 1) * 10,
                "candidate_video_ids": [f"{item['video_id']:04d}" for item in candidates],
                "selected_video_id": f"{choice['video_id']:04d}",
                "selected_video": choice["video"],
                "prompt": choice["prompt"],
            }
        )
    return selected, intervals


def build_sampled_dataset(
    dataset_info: dict[str, Any], selected: list[dict[str, Any]], model_name: str
) -> dict[str, Any]:
    prompts = []
    for item in selected:
        prompts.append(
            {
                "prompt_id": item["prompt_id"],
                "prompt_key": item["prompt_key"],
                "condition_content": item["condition_content"],
                "condition_caption": item["prompt"],
                "generated_videos": [item["video"]],
                "auxiliary_info": [],
                "sampling_interval": item["interval"],
            }
        )
    return {
        "dataset_info": {
            **dataset_info,
            "base_path": os.path.commonpath([str(Path(item["video"]).parent) for item in selected]),
            "sampling": {
                "order": "numeric_filename",
                "intervals": 16,
                "interval_size": 10,
                "samples_per_interval": 1,
            },
        },
        "models": [
            {
                "model_name": model_name,
                "conditions": [
                    {"condition_meta_info": "stratified_sample", "prompts": prompts}
                ],
            }
        ],
    }


def preflight(dimensions: list[str], device: str) -> dict[str, Any]:
    if Path(sys.prefix).name != "4dworldbench":
        raise RuntimeError(
            "activate the target 4dworldbench environment first; "
            f"current prefix is {sys.prefix}"
        )
    if any(dimension in QA_DIMENSIONS for dimension in dimensions):
        missing = [
            name
            for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL")
            if not os.environ.get(name)
        ]
        if missing:
            raise RuntimeError("missing required API environment variables: " + ", ".join(missing))
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError(f"required evaluation dependency cannot be imported: {exc}") from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is not available for requested device {device}")
    required_modules = {"decord"}
    if any(name.startswith("perceptual_clip_") for name in dimensions):
        required_modules.add("pyiqa")
    if "consistency_motion_smoothness" in dimensions:
        required_modules.add("mamba_ssm")
    if "consistency_viewpoint" in dimensions:
        required_modules.update({"lietorch", "droid_backends"})
    if any(name in QA_DIMENSIONS for name in dimensions):
        required_modules.add("flash_attn")
    failed_imports = []
    for module_name in sorted(required_modules):
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failed_imports.append(f"{module_name}: {type(exc).__name__}: {exc}")
    if failed_imports:
        raise RuntimeError("dependency preflight failed: " + "; ".join(failed_imports))
    if any(name in QA_DIMENSIONS for name in dimensions):
        try:
            from flash_attn import flash_attn_func  # noqa: F401
        except Exception as exc:
            raise RuntimeError(f"flash_attn_func cannot be imported: {exc}") from exc
        keye_checkpoint = keye_model_path()
        if not keye_checkpoint.is_dir():
            raise RuntimeError(f"Keye checkpoint directory is missing: {keye_checkpoint}")
    if "consistency_viewpoint" in dimensions:
        droid_checkpoint = droid_weights_path()
        if not droid_checkpoint.is_file():
            raise RuntimeError(f"DROID checkpoint is missing: {droid_checkpoint}")
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "environment": Path(sys.prefix).name,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "verified_modules": sorted(required_modules),
        "device": device,
        "openai_base_url": os.environ.get("OPENAI_BASE_URL") if any(d in QA_DIMENSIONS for d in dimensions) else None,
        "openai_model": api_model_name() if any(d in QA_DIMENSIONS for d in dimensions) else None,
    }


def find_result_file(dimension_dir: Path, dimension: str, model: str) -> Path | None:
    matches = sorted(dimension_dir.glob(f"{dimension}__{model}__*_results.json"))
    return matches[-1] if matches else None


def detail_score(detail: dict[str, Any]) -> float | None:
    for key in (
        "clip_iqa_score",
        "aesthetic_score",
        "fastvqa_score",
        "reprojection_error",
        "ssim",
        "style_score",
        "style_consistency",
        "video_results",
        "score",
    ):
        if key in detail:
            return safe_float(detail[key])
    return None


def load_scores(result_file: Path | None) -> tuple[float | None, dict[str, float | None]]:
    if result_file is None:
        return None, {}
    try:
        data = json.loads(result_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, {}
    average = safe_float(data.get("evaluation_summary", {}).get("average_score"))
    per_video = {}
    for detail in data.get("video_details", []) or []:
        path = detail.get("video_path")
        if isinstance(path, str):
            per_video[str(Path(path).resolve())] = detail_score(detail)
    return average, per_video


def write_reports(
    run_dir: Path,
    dimensions: list[str],
    selected: list[dict[str, Any]],
    dimension_records: dict[str, dict[str, Any]],
) -> None:
    timing_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for dimension in dimensions:
        record = dimension_records.get(dimension, {})
        timing = record.get("timing", {})
        by_path = {item.get("video_path"): item for item in timing.get("videos", [])}
        scores = record.get("per_video_scores", {})
        for item in selected:
            path = str(Path(item["video"]).resolve())
            video_timing = by_path.get(path, {})
            timing_rows.append(
                {
                    "interval": item["interval"],
                    "video_id": f"{item['video_id']:04d}",
                    "video_path": path,
                    "dimension": dimension,
                    "compute_sec": video_timing.get("compute_sec"),
                    "status": video_timing.get("status", record.get("status", "not_run")),
                    "score": scores.get(path),
                    "error": video_timing.get("error") or record.get("error"),
                }
            )
        summary_rows.append(
            {
                "dimension": dimension,
                "status": record.get("status", "not_run"),
                "module_import_sec": timing.get("module_import_sec"),
                "qa_prepare_sec": timing.get("qa_prepare_sec"),
                "model_load_sec": timing.get("model_load_sec"),
                "video_compute_total_sec": timing.get("video_compute_total_sec"),
                "average_video_compute_sec": timing.get("average_video_compute_sec"),
                "successful_videos": timing.get("successful_videos", 0),
                "failed_videos": timing.get("failed_videos", 16),
                "wall_sec": timing.get("wall_sec"),
                "average_score": record.get("average_score"),
                "result_file": record.get("result_file"),
            }
        )

    with (run_dir / "video_dimension_timings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(timing_rows[0]))
        writer.writeheader()
        writer.writerows(timing_rows)
    with (run_dir / "dimension_timing_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    atomic_json(
        run_dir / "timings.json",
        {"dimensions": {name: dimension_records.get(name, {}).get("timing") for name in dimensions}},
    )


def run_dimension(
    dimension: str,
    dataset_path: Path,
    model: str,
    device: str,
    run_dir: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    dimension_dir = run_dir / "dimension_results" / dimension
    dimension_dir.mkdir(parents=True, exist_ok=True)
    timing_file = dimension_dir / "timing.json"
    # A completed dimension is skipped by --resume before reaching this
    # function. Any run that does reach here is a fresh attempt: remove only
    # artifacts owned by this dimension so frame extraction remains cold and a
    # stale result cannot make a failed retry look successful.
    frames_cache = dimension_dir / "frames_cache"
    if frames_cache.exists():
        shutil.rmtree(frames_cache)
    if timing_file.exists():
        timing_file.unlink()
    for stale_result in dimension_dir.glob(f"{dimension}__{model}__*_results.json"):
        stale_result.unlink()
    log_file = run_dir / "logs" / f"{dimension}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "runner.py"),
        "--dataset_json",
        str(dataset_path),
        "--model",
        model,
        "--dimension",
        dimension,
        "--device",
        device,
        "--json_out_dir",
        str(dimension_dir),
        "--qa_dir",
        str(dimension_dir / "qa_alignment_questions"),
        "--timing_out",
        str(timing_file),
    ]
    started = time.perf_counter()
    with log_file.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    wall_sec = time.perf_counter() - started
    timing = json.loads(timing_file.read_text(encoding="utf-8")) if timing_file.is_file() else {}
    if timing:
        timing["wall_sec"] = round(wall_sec, 6)
    result_file = find_result_file(dimension_dir, dimension, model)
    average_score, per_video_scores = load_scores(result_file)
    if return_code != 0 or result_file is None:
        status = "failed"
    elif timing.get("failed_videos", 0):
        status = "partial"
    else:
        status = "ok"
    return {
        "status": status,
        "return_code": return_code,
        "error": None if return_code == 0 else f"runner exited with code {return_code}; see {log_file}",
        "timing": timing,
        "average_score": average_score,
        "per_video_scores": per_video_scores,
        "result_file": str(result_file) if result_file else None,
        "log_file": str(log_file),
    }


def main() -> int:
    args = parse_args()
    configure_runtime(args)
    args.dataset_json = args.dataset_json or default_dataset_path()
    dimensions = list(dict.fromkeys(args.only or ALL_DIMENSIONS))
    if args.resume and args.output_dir is None:
        raise SystemExit("Error: --resume requires an explicit --output-dir")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        args.output_dir
        or PROJECT_ROOT / "sampled_benchmark_results" / f"{args.model}_seed{args.seed}_{timestamp}"
    ).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_info, rows = load_model_rows(args.dataset_json, args.model)
    selected, intervals = sample_rows(rows, args.seed)
    manifest = {
        "source_dataset_json": str(args.dataset_json.expanduser().resolve()),
        "model": args.model,
        "seed": args.seed,
        "random_algorithm": "random.Random(seed).choice applied once per interval",
        "order": "numeric_filename",
        "source_video_count": len(rows),
        "interval_count": 16,
        "interval_size": 10,
        "samples_per_interval": 1,
        "intervals": intervals,
    }
    manifest_path = run_dir / "sample_manifest.json"
    if args.resume:
        if not manifest_path.is_file():
            raise SystemExit(f"Error: cannot resume without {manifest_path}")
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable_fields = ("source_dataset_json", "model", "seed", "order")
        mismatches = [
            field
            for field in comparable_fields
            if previous_manifest.get(field) != manifest.get(field)
        ]
        previous_ids = [
            item.get("selected_video_id") for item in previous_manifest.get("intervals", [])
        ]
        current_ids = [item.get("selected_video_id") for item in intervals]
        if previous_ids != current_ids:
            mismatches.append("selected_video_ids")
        previous_summary_path = run_dir / "run_summary.json"
        if previous_summary_path.is_file():
            previous_summary = json.loads(previous_summary_path.read_text(encoding="utf-8"))
            if previous_summary.get("device") != args.device:
                mismatches.append("device")
            if previous_summary.get("physics_protocol") != PHYSICS_PROTOCOL:
                mismatches.append("physics_protocol")
        if mismatches:
            raise SystemExit(
                "Error: resume configuration differs from the existing run: "
                + ", ".join(sorted(set(mismatches)))
            )
    atomic_json(manifest_path, manifest)
    dataset_path = run_dir / "sampled_dataset.json"
    atomic_json(dataset_path, build_sampled_dataset(dataset_info, selected, args.model))

    summary_path = run_dir / "run_summary.json"
    summary: dict[str, Any] = {
        "status": "dry_run" if args.dry_run else "running",
        "model": args.model,
        "seed": args.seed,
        "device": args.device,
        "source_dataset_json": manifest["source_dataset_json"],
        "sample_manifest": str(run_dir / "sample_manifest.json"),
        "sampled_dataset_json": str(dataset_path),
        "selected_video_ids": [f"{item['video_id']:04d}" for item in selected],
        "selected_dimensions": dimensions,
        "physics_protocol": PHYSICS_PROTOCOL,
        "dimensions": {},
    }

    if args.dry_run:
        atomic_json(summary_path, summary)
        write_reports(run_dir, dimensions, selected, {})
        print(f"Dry run complete: {run_dir}")
        print("Selected:", " ".join(summary["selected_video_ids"]))
        return 0

    try:
        summary["environment"] = preflight(dimensions, args.device)
    except Exception as exc:
        summary["status"] = "preflight_failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        atomic_json(summary_path, summary)
        print(f"Preflight failed: {exc}", file=sys.stderr)
        return 2

    env = os.environ.copy()
    env.pop("ALL_PROXY", None)
    env.pop("all_proxy", None)
    env.update(PHYSICS_PROTOCOL)
    failed = 0
    for index, dimension in enumerate(dimensions, start=1):
        dimension_dir = run_dir / "dimension_results" / dimension
        existing_timing = dimension_dir / "timing.json"
        existing_result = find_result_file(dimension_dir, dimension, args.model)
        if args.resume and existing_timing.is_file() and existing_result is not None:
            timing = json.loads(existing_timing.read_text(encoding="utf-8"))
            if timing.get("failed_videos", 0) == 0 and timing.get("successful_videos") == 16:
                average, scores = load_scores(existing_result)
                record = {
                    "status": "ok",
                    "return_code": 0,
                    "error": None,
                    "timing": timing,
                    "average_score": average,
                    "per_video_scores": scores,
                    "result_file": str(existing_result),
                    "resumed": True,
                }
                print(f"[{index}/{len(dimensions)}] Reusing completed {dimension}")
            else:
                print(f"[{index}/{len(dimensions)}] Re-running incomplete {dimension}")
                record = run_dimension(
                    dimension, dataset_path, args.model, args.device, run_dir, env
                )
        else:
            print(f"[{index}/{len(dimensions)}] Running {dimension}")
            record = run_dimension(
                dimension, dataset_path, args.model, args.device, run_dir, env
            )
        summary["dimensions"][dimension] = record
        if record["status"] != "ok":
            failed += 1
        atomic_json(summary_path, summary)
        write_reports(run_dir, dimensions, selected, summary["dimensions"])

    summary["status"] = "ok" if failed == 0 else "completed_with_failures"
    summary["completed_dimensions"] = len(dimensions)
    summary["failed_dimensions"] = failed
    atomic_json(summary_path, summary)
    write_reports(run_dir, dimensions, selected, summary["dimensions"])
    print(f"Benchmark complete: {run_dir}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
