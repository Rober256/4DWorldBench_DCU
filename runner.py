import os
import sys
import json
import argparse
import importlib
import time
from pathlib import Path
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile

from runtime_config import add_runtime_arguments, configure_runtime, default_dataset_path, dataset_video_root, resolve_video_path

# Import adaptive question generator
try:
    from adaptive_qa_generator import AdaptiveQAGenerator
    QA_GENERATOR_AVAILABLE = True
except ImportError:
    QA_GENERATOR_AVAILABLE = False
    print("Warning: adaptive_qa_generator not available. Questions will not be auto-generated.")


def _bootstrap_paths_and_aliases():
    """Ensure imports for dimension.* and alias vbench2 -> phywordbench.metric.*"""
    base_dir = "/data1/luyt/phywordbench"
    phy_dir = os.path.join(base_dir, "phywordbench")
    for p in [base_dir, phy_dir]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # Local project path takes priority over remote copy.
    # Must remove-then-reinsert: when run as `python runner.py`, Python
    # already puts the script dir at sys.path[0]; a `not in` check would
    # silently skip re-insertion, leaving remote paths ahead of local.
    local_base = os.path.dirname(os.path.abspath(__file__))
    if local_base in sys.path:
        sys.path.remove(local_base)
    sys.path.insert(0, local_base)
    try:
        import metric as _metric
        sys.modules['vbench2'] = _metric
        # expose common submodules
        if hasattr(_metric, 'utils'):
            sys.modules['utils'] = _metric.utils
        if hasattr(_metric, 'distributed'):
            sys.modules['vbench2.distributed'] = _metric.distributed
    except Exception:
        pass


def read_dataset_json(path: str) -> List[Dict[str, Any]]:
    """
    Parse dataset JSON schema like:
    {
      "dataset_info": {...},
      "models": [
        {"model_name": "director3d", "conditions": [
           {"condition_meta_info": "Camera Motion", "prompts": [
              {"condition_caption": ..., "generated_videos": [...]}, ...
           ]}, ...
        ]}
      ]
    }
    Return a flattened list of items with keys: model, dimension, video_list, prompt, auxiliary_info(optional)
    """
    dataset_path = Path(path).expanduser().absolute()
    with dataset_path.open('r', encoding='utf-8') as f:
        root = json.load(f)

    base_path = root.get('dataset_info', {}).get('base_path', '') if isinstance(root, dict) else ''
    declared_base = Path(base_path).expanduser() if base_path else None
    video_root, explicit_root = dataset_video_root(dataset_path)

    items: List[Dict[str, Any]] = []
    models = root.get('models', []) if isinstance(root, dict) else []
    for m in models:
        model_name = m.get('model_name')
        for cond in m.get('conditions', []) or []:
            dim_raw = cond.get('condition_meta_info', '')
            dim_norm = _normalize_dim_name(dim_raw) if dim_raw else ''
            for p in cond.get('prompts', []) or []:
                videos = p.get('generated_videos', []) or []
                videos = [
                    str(resolve_video_path(v, declared_base, video_root, dataset_path.parent, explicit_root))
                    for v in videos
                ]
                prompt_text = p.get('condition_caption') or ''
                items.append({
                    'model': model_name,
                    'dimension': dim_norm,
                    'video_list': videos,
                    'prompt': prompt_text,
                    'auxiliary_info': p.get('auxiliary_info', []),
                })
    return items


def _normalize_dim_name(name: str) -> str:
    return name.strip().replace(' ', '_').replace('-', '_').lower()


def build_model_items(items: List[Dict[str, Any]], model_name: str) -> List[Dict[str, Any]]:
    """
    Filter items by model name, return all items for the specified model
    without filtering by dimension (condition_meta_info)
    """
    filtered_items = []
    for it in items:
        if model_name and it.get('model') == model_name:
            filtered_items.append(it)
    return filtered_items


def create_single_video_item(model_name: str, video_path: str, prompt: str = "") -> Dict[str, Any]:
    """
    Create a single item for evaluating a specific video
    """
    return {
        'model': model_name,
        'dimension': 'single_video',  # placeholder dimension
        'video_list': [video_path],
        'prompt': prompt or f"Evaluate video: {video_path}",
        'auxiliary_info': [],
    }


def ensure_questions_for_dimension(dimension: str, qa_dir: str, source_caption_json: str) -> str:
    os.makedirs(qa_dir, exist_ok=True)
    out_path = os.path.join(qa_dir, f"{dimension}.json")
    # If already exists, reuse
    if os.path.exists(out_path):
        return out_path
    # Build a minimal caption file for gpt4o generator if needed (or directly call it on the given json)
    # Here we simply copy or use the source json if it matches expected format for gpt4o.py
    # For now, just reuse source file path; gpt4o.py reads and writes in-place by dimension convention
    # You may customize gpt4o.py to accept inputs and write to out_path.
    return out_path


SKIP_QUESTION_GENERATION_DIMENSIONS = {
    "perceptual_clip_aesthetic_metrics",
    "perceptual_clip_iqa_metrics",
    "perceptual_fastvqa",
    # Viewpoint consistency is computed directly with DROID-SLAM and does not
    # consume auxiliary QA questions. Generating them wastes an API request.
    "consistency_viewpoint",
    "consistency_motion_smoothness",
    "consistency_style",
}


def auto_generate_questions_if_needed(json_file_path: str, dimension: str) -> bool:
    """
    Automatically detect if questions are missing in JSON file, and generate if needed
    Now changed to parallel generation: call generator in parallel for each entry missing auxiliary_info
    """
    if dimension in SKIP_QUESTION_GENERATION_DIMENSIONS:
        return False

    if not QA_GENERATOR_AVAILABLE:
        print("Question generator unavailable, skipping automatic question generation")
        return False

    # Read JSON and determine items needing question generation
    try:
        if not os.path.exists(json_file_path):
            print(f"JSON file does not exist: {json_file_path}")
            return False
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error during automatic question generation (failed to read file): {e}")
        return False

    if isinstance(data, dict):
        # Compatibility for non-list structure: convert to list view for processing
        items = list(data.values())
        key_list = list(data.keys())
        is_dict_mode = True
    elif isinstance(data, list):
        items = data
        key_list = None
        is_dict_mode = False
    else:
        print("JSON format not supported, must be list or dict")
        return False

    targets = []
    for idx, item in enumerate(items):
        aux = item.get('auxiliary_info', [])
        if not aux or len(aux) == 0:
            targets.append(idx)

    if not targets:
        print(f"JSON file {json_file_path} already has sufficient questions")
        return False

    print(f"Detected {len(targets)}/{len(items)} entries missing questions, starting parallel generation...")

    # Parallelism settings with rate-limiting controls
    try:
        max_workers = int(os.environ.get("QA_NUM_WORKERS", "0")) or (os.cpu_count() or 4)
    except Exception:
        max_workers = os.cpu_count() or 4
    max_workers = min(max_workers, 8)
    if max_workers < 1:
        max_workers = 1
    request_delay = float(os.environ.get("QA_REQUEST_DELAY", "1.0"))
    max_consecutive_failures = int(os.environ.get("QA_MAX_CONSECUTIVE_FAILURES", "5"))

    # Worker: use temporary file to call existing process_json_file interface for single entry
    def _process_one(index: int, item: Dict[str, Any]) -> tuple[int, List[Dict[str, Any]]]:
        try:
            with tempfile.TemporaryDirectory(prefix="qa_gen_") as td:
                tmp_path = os.path.join(td, "single_item.json")
                # Write single item list
                with open(tmp_path, "w", encoding="utf-8") as wf:
                    json.dump([item], wf, ensure_ascii=False, indent=2)
                # Instantiate generator independently to avoid shared state
                gen = AdaptiveQAGenerator()
                ok = gen.process_json_file(tmp_path, dimension)
                if not ok:
                    return index, item.get("auxiliary_info", [])
                # Read back result
                with open(tmp_path, "r", encoding="utf-8") as rf:
                    out_list = json.load(rf)
                if isinstance(out_list, list) and out_list:
                    new_aux = out_list[0].get("auxiliary_info", [])
                else:
                    new_aux = item.get("auxiliary_info", [])
                return index, new_aux
        except Exception as e:
            # On error return original (possibly empty), don't throw
            print(f"[QA Worker {index}] Failed: {type(e).__name__}: {e}")
            return index, item.get("auxiliary_info", [])

    changed = False
    consecutive_failures = 0
    # Execute parallel tasks with rate limiting and circuit breaker
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        # Stagger submissions to avoid flooding the API
        futures = {}
        for i, idx in enumerate(targets):
            if i > 0:
                time.sleep(request_delay)
            futures[ex.submit(_process_one, idx, items[idx])] = idx
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                i_ret, aux_new = fut.result()
                if i_ret != idx:
                    i_ret = idx
                if aux_new and not items[i_ret].get('auxiliary_info'):
                    items[i_ret]['auxiliary_info'] = aux_new
                    changed = True
                    consecutive_failures = 0
                else:
                    # aux_new is empty, treat as potential failure
                    consecutive_failures += 1
            except Exception:
                consecutive_failures += 1
                print(f"QA worker idx={idx} failed, consecutive failures: {consecutive_failures}")

            if consecutive_failures >= max_consecutive_failures:
                print(f"ERROR: {consecutive_failures} consecutive QA generation failures, aborting to avoid triggering risk controls")
                for f in futures:
                    f.cancel()
                raise RuntimeError(
                    f"Aborted: {consecutive_failures} consecutive QA generation failures. "
                    "Consider increasing QA_REQUEST_DELAY, reducing QA_NUM_WORKERS, or checking the API service."
                )

    if not changed:
        print("No new questions generated or generation failed.")
        return False

    # Merge and write back to original file
    try:
        if is_dict_mode:
            merged = {k: items[i] for i, k in enumerate(key_list)}
        else:
            merged = items
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"Parallel question generation complete, file updated: {json_file_path}")
        return True
    except Exception as e:
        print(f"Failed to write back file: {e}")
        return False


def dynamic_import_dimension(dimension_name: str):
    # Modules live in WorldModelBench/dimension/<name>.py
    module_name = f"metric.{dimension_name}"
    #try:
    return importlib.import_module(module_name)
    # except ModuleNotFoundError:
        # also try lowercase
    #    return importlib.import_module(f"dimension.{_normalize_dim_name(dimension_name)}")


def run_dimension(dimension: str, dim_items: List[Dict[str, Any]], json_dir: str, device: str = "cuda:0", model: str = "", dataset_json: str = "", timer=None) -> Dict[str, Any]:
    if timer is None:
        from metric.timing import EvaluationTimer
        timer = EvaluationTimer(dimension=dimension, device=device)
    with timer.measure_module_import():
        mod = dynamic_import_dimension(dimension)
    # dimension module should expose a compute_<dimension>(json_dir, device, submodules_dict, **kwargs)
    func_name = f"compute_{_normalize_dim_name(dimension)}"
    if not hasattr(mod, func_name):
        # fallback to generic name
        func_name = [n for n in dir(mod) if n.startswith("compute_")][0]
    compute_fn = getattr(mod, func_name)
    submodules_dict = {"evaluation_timer": timer}
    score, details = compute_fn(json_dir, device, submodules_dict, model=model, dataset_json=dataset_json)
    return {"dimension": dimension, "score": float(score), "details": details}


def write_timing_json(path: str, timer, **extra: Any) -> None:
    if not path:
        return
    output = timer.to_dict()
    output.update(extra)
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def write_dimension_json(json_dir_root: str, dimension: str, model_items: List[Dict[str, Any]]) -> str:
    os.makedirs(json_dir_root, exist_ok=True)
    # Transform to VBench-style list of dicts with keys: dimension, video_list, prompt_en, auxiliary_info
    out_items: List[Dict[str, Any]] = []
    for it in model_items:
        out_items.append({
            "dimension": [dimension],
            "video_list": it.get("video_list", []),
            "prompt_en": it.get("prompt", ""),
            "auxiliary_info": it.get("auxiliary_info", []),
        })
    out_path = os.path.join(json_dir_root, f"{dimension}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out_items, f, indent=2, ensure_ascii=False)
    return out_path


def main():
    _bootstrap_paths_and_aliases()
    parser = argparse.ArgumentParser(description="PhyWordBench runner")
    parser.add_argument("--dataset_json", "--dataset-json", help="Path to dataset JSON, e.g., condition_to_4D/text-to-any/text_to_3d_dataset.json")
    parser.add_argument("--model", required=True, help="Model name to filter items")
    parser.add_argument("--dimension", required=True, help="Dimension/metric script to run (e.g., dynamic_attribute, camera_motion, etc.)")
    parser.add_argument("--prompt", default="", help="Optional single prompt override for the specified dimension.")
    parser.add_argument("--video_path", help="Optional path to a single video file for evaluation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--qa_dir", default="qa_alignment_questions")
    parser.add_argument("--json_out_dir", default="dimension_description_json")
    parser.add_argument("--max_questions", type=int, default=0, help="Limit auxiliary questions per item (0 keeps all); useful for smoke tests")
    parser.add_argument("--retry_nulls", help="Path to existing physics_realism results JSON file; re-answers null entries in-place")
    parser.add_argument("--timing_out", help="Write standardized phase and per-video timings to this JSON file")
    add_runtime_arguments(parser)
    args = parser.parse_args()
    configure_runtime(args)

    # Dispatch: --retry_nulls mode (retry null answers in existing results)
    if args.retry_nulls:
        if args.dimension != 'physics_realism':
            print("Error: --retry_nulls is only supported with --dimension physics_realism")
            return
        from metric.physics_realism import retry_nulls_in_results_file
        max_rounds = int(os.environ.get("PR_MAX_NULL_RETRY_ROUNDS", "3"))
        retry_nulls_in_results_file(args.retry_nulls, args.device, max_rounds)
        return

    # Check if we're evaluating a single video or using dataset
    if args.video_path:
        # Single video evaluation mode
        if not os.path.exists(args.video_path):
            print(f"Error: Video file '{args.video_path}' does not exist.")
            return
        
        model_items = [create_single_video_item(args.model, args.video_path, args.prompt)]
        print(f"[phywordbench] Single video evaluation mode")
        print(f"  - Model: {args.model}")
        print(f"  - Video: {args.video_path}")
        print(f"  - Dimension: {args.dimension}")
        
    else:
        # Dataset evaluation mode
        args.dataset_json = args.dataset_json or str(default_dataset_path())
        items = read_dataset_json(args.dataset_json)
        model_items = build_model_items(items, args.model)
        
        if not model_items:
            print(f"No items found for model '{args.model}' in the dataset.")
            return

        # Apply prompt override if specified
        if args.prompt:
            for it in model_items:
                it["prompt"] = args.prompt

    dimension = args.dimension
    from metric.timing import EvaluationTimer
    timer = EvaluationTimer(dimension=dimension, device=args.device)
    result = None
    try:
        with timer.measure_qa_prepare():
            dim_json_path = write_dimension_json(args.json_out_dir, dimension, model_items)
            questions_generated = auto_generate_questions_if_needed(dim_json_path, dimension)
            if questions_generated:
                print(f"Automatically generated evaluation questions for dimension '{dimension}'")

            if args.max_questions > 0:
                with open(dim_json_path, 'r', encoding='utf-8') as f:
                    limited_items = json.load(f)
                for item in limited_items:
                    auxiliary_info = item.get('auxiliary_info', [])
                    if isinstance(auxiliary_info, list):
                        item['auxiliary_info'] = auxiliary_info[:args.max_questions]
                with open(dim_json_path, 'w', encoding='utf-8') as f:
                    json.dump(limited_items, f, indent=2, ensure_ascii=False)
                print(f"Limited auxiliary questions to {args.max_questions} per item")

            ensure_questions_for_dimension(dimension, args.qa_dir, dim_json_path)

        with open(dim_json_path, 'r', encoding='utf-8') as f:
            prepared_items = json.load(f)
        question_counts = [
            len(item.get('auxiliary_info', []))
            for item in prepared_items
            if isinstance(item, dict)
        ]
        timer.add_metadata(
            video_count=len(model_items),
            question_counts=question_counts,
            droid_load_policy=(
                "per_clip_existing_behavior; repeated DROID construction is included in video compute time"
                if dimension == "consistency_viewpoint" else None
            ),
        )

        print(f"[4dworldbench] Running dimension '{dimension}' with {len(model_items)} items from model '{args.model}'...")
        start = time.perf_counter()
        result = run_dimension(
            dimension,
            model_items,
            dim_json_path,
            device=args.device,
            model=args.model,
            dataset_json=args.dataset_json or "",
            timer=timer,
        )
        result["elapsed_sec"] = time.perf_counter() - start

        print("\n=== 4DWorldBench Summary ===")
        print(f"- {result['dimension']}: {result['score']:.4f} (items: {len(model_items)}, elapsed: {result['elapsed_sec']:.2f}s)")
        return 0
    except Exception as exc:
        timer.fail_active(exc)
        timer.add_metadata(fatal_error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_timing_json(
            args.timing_out,
            timer,
            status="ok" if result is not None else "failed",
            score=result.get("score") if result else None,
        )


if __name__ == "__main__":
    raise SystemExit(main())
