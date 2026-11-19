"""Failure-isolated batching for Keye metrics that share one loaded model."""

from __future__ import annotations

from typing import Any, Callable


def run_keye_items(
    prompt_items: list[dict[str, Any]],
    evaluate_one: Callable[[list[dict[str, Any]]], tuple[float, list[dict[str, Any]]]],
    timer,
) -> tuple[float, list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    for item in prompt_items:
        try:
            _, item_details = evaluate_one([item])
        except Exception as exc:
            timer.fail_active(exc)
            videos = item.get("video_list", []) or []
            print(
                f"[keye-batch] Failed video item {videos}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        details.extend(item_details)
    scores = [
        float(detail["video_results"])
        for detail in details
        if detail.get("video_results") is not None
    ]
    return (sum(scores) / len(scores) if scores else 0.0), details
