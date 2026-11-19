#!/usr/bin/env bash
set -euo pipefail

# Batch evaluation for 4D consistency dimensions
# Usage: bash scripts/run_all_consistency.sh [cuda:0] [options]

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEVICE="cuda:0"
if [[ ${1:-} == cuda:* ]]; then
    DEVICE="$1"
    shift
fi
# Extract GPU number from device string (e.g., "cuda:4" -> "4")
GPU_ID="${DEVICE#cuda:}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"


# Fix SOCKS proxy issue
unset ALL_PROXY all_proxy

export CAPTION_DEVICE="${CAPTION_DEVICE:-cuda:0}"
export DEVICE

MODELS=(
    # wisa
    # wan_1_3b
    # wan_14b
    # cogvideox
    # ours_1_3b
    ours_1_3bs2_720p
    # ours_14b
    # ours_14bs2_720p
    # videorepa
    # hunyuan
)

DIMENSIONS=(
    # consistency_viewpoint
    # consistency_motion_smoothness
    # consistency_motion_qa
    consistency_style
)

TOTAL_DIMS=${#DIMENSIONS[@]}
TOTAL_MODELS=${#MODELS[@]}
TOTAL=$((TOTAL_DIMS * TOTAL_MODELS))
CURRENT=0
FAILED=()

echo "============================================"
echo "4D Consistency Batch Evaluation"
echo "  Dimensions: ${TOTAL_DIMS}"
echo "  Models: ${TOTAL_MODELS}"
echo "  Total tasks: ${TOTAL}"
echo "  Device: ${DEVICE}"
echo "============================================"

for dim in "${DIMENSIONS[@]}"; do
    for model in "${MODELS[@]}"; do
        CURRENT=$((CURRENT + 1))
        echo ""
        echo "--- [${CURRENT}/${TOTAL}] ${model} / ${dim} ---"

        if bash "$SCRIPT_DIR/evaluate.sh" \
            --model "$model" \
            --dimension "$dim" \
            --device cuda:0 "$@"; then
            echo "--- [${CURRENT}/${TOTAL}] ${model}/${dim}: OK ---"
        else
            echo "--- [${CURRENT}/${TOTAL}] ${model}/${dim}: FAILED ---"
            FAILED+=("${model}/${dim}")
        fi
    done
done

echo ""
echo "============================================"
echo "Consistency evaluation complete."
echo "  Total: ${TOTAL}, Failed: ${#FAILED[@]}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  Failed: ${FAILED[*]}"
fi
echo "============================================"
