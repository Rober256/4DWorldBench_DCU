#!/usr/bin/env bash
# 4DWorldBench Physics random-3 performance profile.
# This is an isolated profiling entrypoint; it does not alter the formal run scripts.
set -o pipefail

ROOT="${FOURDWORLDBENCH_ROOT:-/public/home/lixin1/4dworldbench}"
CODE_ROOT="$ROOT/code"
PYTHON="${FOURDWORLDBENCH_PYTHON:-$ROOT/../envs/4dworldbench/bin/python}"
DATASET="$ROOT/data/videos/dataset_jsons/phygenbench_text_to_4d.json"
PHYSICS_CACHE="$CODE_ROOT/dimension_description_json/physics_realism.json"
SEED="${PHYSICS_PROFILE_SEED:-20260813}"
DEVICE="${DEVICE:-cuda:0}"
PROFILE_ROOT="${PHYSICS_PROFILE_ROOT:-$ROOT/profiling}"
RUN_ID="physics_random3_seed${SEED}_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$PROFILE_ROOT/$RUN_ID"
mkdir -p "$OUT_DIR"

# DTK env.sh uses an unset variable under nounset; load it safely.
set +u
source /opt/dtk-26.04/env.sh
set -u
source "$ROOT/use_isolated_cache.sh"

# HCU/Triton runtime required by Keye; keep it scoped to this profile.
export LD_PRELOAD="${LD_PRELOAD:-/opt/dtk-26.04/hip/lib/libamdhip64.so}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
unset HIP_VISIBLE_DEVICES

# ROCm/RCCL operation-size and communication logs requested for profiling.
export ROCFFT_LAYER=3
export ROCFFT_LOG_TRACE_PATH="$OUT_DIR/fft-trace.log"
export ROCFFT_LOG_BENCH_PATH="$OUT_DIR/fft-bench.log"
export HIPBLASLT_LOG_LEVEL=5
export HIPBLASLT_LOG_MASK=32
export HIPBLASLT_LOG_FILE="$OUT_DIR/hipblaslt.log"
export ROCSOLVER_LAYER=2
export ROCSOLVER_LEVELS=1
export ROCSOLVER_LOG_BENCH_PATH="$OUT_DIR/solver-bench.log"
export MIOPEN_ENABLE_LOGGING_CMD=1
export ROCBLAS_LAYER=3
export ROCBLAS_LOG_BENCH_PATH="$OUT_DIR/rocblas-bench.log"
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=coll
export NCCL_DEBUG_FILE="$OUT_DIR/rccl-log.log"

# Keep the profiling run bounded and small while exercising the real Physics path.
export KEYE_MAX_FRAMES="${KEYE_MAX_FRAMES:-32}"
export KEYE_VIDEO_TOKEN_BUDGET="${KEYE_VIDEO_TOKEN_BUDGET:-5120}"
export KEYE_VIDEO_MAX_PIXELS="${KEYE_VIDEO_MAX_PIXELS:-401408}"
export KEYE_MAX_NEW_TOKENS="${KEYE_MAX_NEW_TOKENS:-128}"
export PR_TARGET_QUESTIONS="${PR_TARGET_QUESTIONS:-1}"
export PR_MAX_WORKERS="${PR_MAX_WORKERS:-1}"
export PR_REQUEST_DELAY="${PR_REQUEST_DELAY:-0.5}"
export PR_AUTO_RETRY_NULLS=0
export FOURDWORLDBENCH_WEIGHTS_DIR="$CODE_ROOT/pretrained_weights"
unset ALL_PROXY all_proxy

# Use the existing generated Physics questions so this profile measures the
# model/evaluation path and does not add question-generation requests.
"$PYTHON" - "$PHYSICS_CACHE" "$ROOT/profiling/$RUN_ID/subset_random3.json" "$SEED" <<'PY'
import json, random, sys
src, dst, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
items = json.load(open(src, encoding="utf-8"))
if not isinstance(items, list) or len(items) < 3:
    raise SystemExit("physics question cache must contain at least 3 list items")
rng = random.Random(seed)
chosen = rng.sample(items, 3)
out = []
for item in chosen:
    row = dict(item)
    row["video_list"] = [p.replace("/public/share/rogon_admin/4dworldbench", "/public/home/lixin1/4dworldbench") for p in row.get("video_list", [])]
    out.append(row)
model = {
  "model_name": "ours_14bs2_720p",
  "conditions": [{"condition_meta_info": "physics_profile_random3", "prompts": [
      {"prompt_id": f"profile_{i:04d}", "prompt_key": f"profile_{i:04d}",
       "condition_caption": row.get("prompt_en", ""),
       "generated_videos": row.get("video_list", []),
       "auxiliary_info": row.get("auxiliary_info", [])}
      for i, row in enumerate(out, 1)
  ]}]
}
json.dump({"dataset_info":{"base_path":"/public/home/lixin1/4dworldbench/data/videos","sampling_seed":seed},"models":[model]}, open(dst,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
json.dump([{"video_list":row.get("video_list",[]),"prompt":row.get("prompt_en","")} for row in out], open(dst+".selection.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
print("SELECTED", [row.get("video_list",[]) for row in out])
PY

# Capture reproducibility and resource snapshots before execution.
{
  echo "timestamp=$(date -Is)"
  echo "host=$(hostname)"
  echo "pwd=$(pwd)"
  echo "python=$PYTHON"
  "$PYTHON" -c 'import sys,torch; print("python_version="+sys.version); print("torch_version="+torch.__version__); print("cuda_available="+str(torch.cuda.is_available())); print("device_count="+str(torch.cuda.device_count()))'
  echo "device=$DEVICE"
  echo "seed=$SEED"
  echo "out_dir=$OUT_DIR"
  command -v hipprof || true
  /opt/hyhal/bin/hy-smi 2>&1 || true
  rocminfo 2>&1 | head -120 || true
  env | grep -E '^(ROCFFT|HIPBLASLT|ROCSOLVER|MIOPEN|ROCBLAS|NCCL|LD_PRELOAD|CUDA_VISIBLE|KEYE_|PR_)' | sort
} > "$OUT_DIR/preflight.txt" 2>&1

# Do not issue an unauthenticated external request. The profile is rerunnable
# after OPENAI_API_KEY is set in the calling shell.
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  {
    echo "STATUS=blocked_api_config"
    echo "missing=OPENAI_API_KEY"
    echo "required=OPENAI_BASE_URL,OPENAI_API_KEY,OPENAI_MODEL"
    echo "selected_subset=$OUT_DIR/subset_random3.json"
  } | tee "$OUT_DIR/status.txt"
  echo "API key is not configured; Physics profile not started." | tee "$OUT_DIR/terminal.log"
  exit 78
fi
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.deepseek.com}"
export OPENAI_MODEL="${OPENAI_MODEL:-deepseek-v4-flash}"

cd "$CODE_ROOT"
{
  echo "START $(date -Is)"
  echo "command=hipprof --rccl-trace -o $OUT_DIR/hipprof"
  hipprof --rccl-trace -o "$OUT_DIR/hipprof" \
    "$PYTHON" "$CODE_ROOT/runner.py" \
    --dataset_json "$OUT_DIR/subset_random3.json" \
    --model ours_14bs2_720p \
    --dimension physics_realism \
    --device "$DEVICE" \
    --json_out_dir "$OUT_DIR/json" \
    --qa_dir "$OUT_DIR/qa" \
    --max_questions "$PR_TARGET_QUESTIONS" \
    --timing_out "$OUT_DIR/timing.json"
  rc=${PIPESTATUS[0]}
  echo "END $(date -Is) rc=$rc"
  exit "$rc"
} 2>&1 | tee "$OUT_DIR/terminal.log"
RC=${PIPESTATUS[0]}

# Collect post-run sizes and files, including profiler exports and requested logs.
{
  echo "status_rc=$RC"
  echo "completed=$(date -Is)"
  echo "--- files ---"
  find "$OUT_DIR" -maxdepth 3 -type f -printf '%s %p\n' | sort -n
  echo "--- du ---"
  du -ah "$OUT_DIR" | sort -h | tail -40
  echo "--- devices_after ---"
  /opt/hyhal/bin/hy-smi 2>&1 || true
} > "$OUT_DIR/postrun_sizes.txt" 2>&1
exit "$RC"
