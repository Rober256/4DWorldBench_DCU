"""Shared timing helpers for repeatable per-dimension benchmarks."""

from __future__ import annotations

import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


def _cuda_synchronize(device: str | None) -> None:
    """Synchronize CUDA when available without making CPU-only runs fail."""
    if not device or not str(device).startswith("cuda"):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
    except Exception:
        # Synchronization is a precision aid. The metric itself remains the
        # source of truth for whether the requested device is usable.
        return


@dataclass
class VideoTiming:
    video_path: str
    compute_sec: float | None = None
    status: str = "running"
    error: str | None = None
    _started: float | None = field(default=None, repr=False)


@dataclass
class EvaluationTimer:
    dimension: str
    device: str
    metadata: dict[str, Any] = field(default_factory=dict)
    module_import_sec: float = 0.0
    qa_prepare_sec: float = 0.0
    model_load_sec: float = 0.0
    postprocess_sec: float = 0.0
    videos: list[VideoTiming] = field(default_factory=list)
    _wall_started: float = field(default_factory=time.perf_counter)
    _active_video: VideoTiming | None = None

    @contextmanager
    def _phase(self, target: str, synchronize: bool = False) -> Iterator[None]:
        if synchronize:
            _cuda_synchronize(self.device)
        started = time.perf_counter()
        try:
            yield
        finally:
            if synchronize:
                _cuda_synchronize(self.device)
            setattr(self, target, getattr(self, target) + time.perf_counter() - started)

    def measure_module_import(self):
        return self._phase("module_import_sec")

    def measure_qa_prepare(self):
        return self._phase("qa_prepare_sec")

    def measure_model_load(self):
        return self._phase("model_load_sec", synchronize=True)

    def start_model_load(self) -> float:
        _cuda_synchronize(self.device)
        return time.perf_counter()

    def finish_model_load(self, started: float) -> None:
        _cuda_synchronize(self.device)
        self.model_load_sec += time.perf_counter() - started

    def measure_postprocess(self):
        return self._phase("postprocess_sec", synchronize=True)

    def start_postprocess(self) -> float:
        _cuda_synchronize(self.device)
        return time.perf_counter()

    def finish_postprocess(self, started: float) -> None:
        _cuda_synchronize(self.device)
        self.postprocess_sec += time.perf_counter() - started

    @contextmanager
    def measure_video(self, video_path: str, suppress: bool = False) -> Iterator[VideoTiming]:
        record = self.start_video(video_path)
        try:
            yield record
        except Exception as exc:
            self.finish_video(record, exc)
            if not suppress:
                raise
        else:
            self.finish_video(record)

    def start_video(self, video_path: str) -> VideoTiming:
        record = VideoTiming(video_path=video_path)
        self.videos.append(record)
        self._active_video = record
        _cuda_synchronize(self.device)
        record._started = time.perf_counter()
        return record

    def finish_video(self, record: VideoTiming, exc: BaseException | None = None) -> None:
        _cuda_synchronize(self.device)
        if record.compute_sec is None and record._started is not None:
            record.compute_sec = time.perf_counter() - record._started
        if exc is not None:
            record.status = "failed"
            record.error = f"{type(exc).__name__}: {exc}"
        elif record.status == "running":
            record.status = "ok"
        if self._active_video is record:
            self._active_video = None

    def fail_active(self, exc: BaseException) -> None:
        if self._active_video is None:
            return
        if self._active_video.status == "running":
            self._active_video.status = "failed"
            self._active_video.error = f"{type(exc).__name__}: {exc}"
        self.finish_video(self._active_video, exc)
        self._active_video = None

    def add_metadata(self, **values: Any) -> None:
        self.metadata.update(values)

    def to_dict(self, wall_sec: float | None = None) -> dict[str, Any]:
        successful = [
            item.compute_sec
            for item in self.videos
            if item.status == "ok" and item.compute_sec is not None
        ]
        failed = [item for item in self.videos if item.status != "ok"]
        return {
            "dimension": self.dimension,
            "device": self.device,
            "module_import_sec": round(self.module_import_sec, 6),
            "qa_prepare_sec": round(self.qa_prepare_sec, 6),
            "model_load_sec": round(self.model_load_sec, 6),
            "postprocess_sec": round(self.postprocess_sec, 6),
            "video_compute_total_sec": round(sum(successful), 6),
            "average_video_compute_sec": (
                round(sum(successful) / len(successful), 6) if successful else None
            ),
            "successful_videos": len(successful),
            "failed_videos": len(failed),
            "wall_sec": round(
                wall_sec if wall_sec is not None else time.perf_counter() - self._wall_started,
                6,
            ),
            "metadata": self.metadata,
            "videos": [
                {
                    "video_path": item.video_path,
                    "compute_sec": (
                        round(item.compute_sec, 6) if item.compute_sec is not None else None
                    ),
                    "status": item.status,
                    "error": item.error,
                }
                for item in self.videos
            ],
        }


def get_timer(
    submodules_dict: dict[str, Any] | None,
    dimension: str,
    device: str,
) -> EvaluationTimer:
    if submodules_dict:
        timer = submodules_dict.get("evaluation_timer")
        if isinstance(timer, EvaluationTimer):
            return timer
    return EvaluationTimer(dimension=dimension, device=device)


def exception_text(exc: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()
