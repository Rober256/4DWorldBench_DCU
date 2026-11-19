"""Shared memory-safe runtime settings for Keye video inference."""

import os
from typing import Any, Dict


def keye_video_content(video_path: str, fps: float = 1.0) -> Dict[str, Any]:
    """Build a Keye video content item with an explicit frame/token budget.

    Keye's preprocessing defaults to 2 FPS and a 65,536-token video budget.
    That can make the custom model materialize multi-gigabyte prompt logits for
    long videos. These defaults keep full-video temporal coverage while fitting
    inference on a 24 GiB RTX 4090. They remain overrideable for larger GPUs.
    """
    configured_fps = float(os.environ.get("KEYE_VIDEO_FPS", str(fps)))
    max_frames = int(os.environ.get("KEYE_MAX_FRAMES", "32"))
    total_tokens = int(os.environ.get("KEYE_VIDEO_TOKEN_BUDGET", "5120"))
    max_pixels = int(os.environ.get("KEYE_VIDEO_MAX_PIXELS", str(512 * 28 * 28)))
    return {
        "type": "video",
        "video": video_path,
        "fps": configured_fps,
        "max_frames": max_frames,
        "video_total_pixels": total_tokens * 28 * 28,
        "max_pixels": max_pixels,
    }


def keye_max_new_tokens(default: int = 768) -> int:
    return int(os.environ.get("KEYE_MAX_NEW_TOKENS", str(default)))


def keye_video_caption_prompt(reference_prompt: str = "", evaluation_questions=None) -> str:
    """Return a physics-oriented caption prompt with optional Keye thinking."""
    instruction = (
        "Provide one objective paragraph describing the complete video from start to finish. "
        "Identify visible people, robots, objects, and their actions in chronological order. "
        "For manipulated materials, use visible flexibility, folds, wrinkles, bending, and surface "
        "behavior to distinguish cloth or towel from paper or other rigid sheets. Distinguish an "
        "entity merely being visible from it physically interacting with the task; do not claim an "
        "interaction unless contact and its effect are clearly visible. Cover the initial state, key "
        "intermediate changes, and final state. Use only visual evidence and do not infer intent, "
        "off-screen events, or hidden causes. Return only the final objective paragraph."
    )
    if reference_prompt:
        instruction += (
            " The following intended-video description is vocabulary context only, not visual "
            "evidence. Verify every detail against the frames and explicitly avoid copying actions "
            f"that are not visible: {reference_prompt}"
        )
    if evaluation_questions:
        question_text = "\n".join(f"- {question}" for question in evaluation_questions)
        instruction += (
            " Ensure the paragraph records any clearly visible evidence relevant to the evaluation "
            "questions below, including contact timing, stability, deformation, lighting, and motion "
            "smoothness when those are observable. Do not answer the questions directly, and do not "
            f"invent evidence for anything that is not visible. Evaluation questions:\n{question_text}"
        )
    thinking = os.environ.get("KEYE_CAPTION_THINKING", "0").lower() in {"1", "true", "yes"}
    return f"{instruction}\n{'/think' if thinking else '/no_think'}"
