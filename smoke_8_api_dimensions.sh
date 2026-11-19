#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_DATASET="$ROOT/smoke_single_cached_question.json"
PHYSICS_DATASET="$ROOT/smoke_single_cached_physics.json"
MODEL=ours_1_3bs2_720p
RUN_SECS=${RUN_SECS:-240}
OUT_DIR="$ROOT/smoke_8_api_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

GPU0_DIMS=(
  alignment_attribute_control
  alignment_motion_control
  alignment_scene_control
  physics_realism
)

GPU1_DIMS=(
  alignment_event_control
  alignment_relationship_control
  consistency_viewpoint
  consistency_motion_qa
)

run_one() {
  local gpu=$1
  local dim=$2
  local log="$OUT_DIR/${dim}.log"
  local rc_file="$OUT_DIR/${dim}.rc"
  local dataset="$SMOKE_DATASET"
  if [[ "$dim" == "physics_realism" ]]; then
    dataset="$PHYSICS_DATASET"
  fi

  printf 'START %s gpu=%s dimension=%s\n' "$(date -Is)" "$gpu" "$dim" | tee "$log"
  timeout --foreground --signal=TERM --kill-after=15s "${RUN_SECS}s" \
    env CUDA_VISIBLE_DEVICES="$gpu" \
    "${FOURDWORLDBENCH_PYTHON:-python}" -u "$ROOT/runner.py" \
      --dataset_json "$dataset" \
      --model "$MODEL" \
      --device cuda:0 \
      --dimension "$dim" >>"$log" 2>&1
  local rc=$?
  printf '%s\n' "$rc" >"$rc_file"
  printf 'END %s gpu=%s dimension=%s rc=%s\n' "$(date -Is)" "$gpu" "$dim" "$rc" | tee -a "$log"
}

worker() {
  local gpu=$1
  shift
  local dim
  for dim in "$@"; do
    run_one "$gpu" "$dim"
  done
}

cd "$ROOT"
printf '%s\n' "$OUT_DIR" > "$ROOT/smoke_8_api_latest_dir.txt"

worker 0 "${GPU0_DIMS[@]}" &
worker0=$!
worker 1 "${GPU1_DIMS[@]}" &
worker1=$!

while kill -0 "$worker0" 2>/dev/null || kill -0 "$worker1" 2>/dev/null; do
  {
    echo "===== $(date -Is) ====="
    ps -eo pid,ppid,etime,%cpu,%mem,rss,stat,cmd \
      | grep -E 'python .*runner.py|timeout .*runner.py' \
      | grep -v grep || true
    /opt/hyhal/bin/hy-smi 2>&1 || true
  } >> "$OUT_DIR/resource_monitor.log"
  sleep 10
done

wait "$worker0"
wait "$worker1"

echo "OUT_DIR=$OUT_DIR"
for dim in "${GPU0_DIMS[@]}" "${GPU1_DIMS[@]}"; do
  rc=$(cat "$OUT_DIR/${dim}.rc" 2>/dev/null || echo missing)
  tracebacks=$(grep -c 'Traceback (most recent call last)' "$OUT_DIR/${dim}.log" 2>/dev/null || true)
  printf '%-42s rc=%-7s tracebacks=%s\n' "$dim" "$rc" "$tracebacks"
done
