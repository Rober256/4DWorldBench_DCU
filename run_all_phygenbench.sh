#!/usr/bin/env bash
set -euo pipefail

# Batch evaluation script for all phygenbench text-to-4D models
# Usage: bash run_all_phygenbench.sh [--device cuda:6]

# --- Config ---
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEVICE="cuda:0"
if [[ ${1:-} == cuda:* ]]; then
    DEVICE="$1"
    shift
fi
# Extract GPU number from device string (e.g., "cuda:4" -> "4")
GPU_ID="${DEVICE#cuda:}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

DIMENSION="physics_realism"
CAPTION_DEVICE="${CAPTION_DEVICE:-cuda:0}"

# Fix SOCKS proxy issue
unset ALL_PROXY all_proxy

export CAPTION_DEVICE
export DEVICE

MODELS=(
    # wisa
    # wan_1_3b
    # wan_14b
    # cogvideox
    # ours_1_3b
    # ours_1_3bs2_720p
    # ours_14b
    ours_14bs2_720p
    # videorepa
    # hunyuan
)

TOTAL=${#MODELS[@]}
CURRENT=0
FAILED_MODELS=()

echo "============================================"
echo "PhyGenBench Batch Evaluation"
echo "  Models: ${TOTAL}"
echo "  Dimension: ${DIMENSION}"
echo "  Device: ${DEVICE}"
echo "============================================"

for model in "${MODELS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "--- [${CURRENT}/${TOTAL}] Evaluating: ${model} ---"

    if bash "$SCRIPT_DIR/evaluate.sh" \
        --model "$model" \
        --dimension "$DIMENSION" \
        --device cuda:0 "$@"; then
        echo "--- [${CURRENT}/${TOTAL}] ${model}: OK ---"
    else
        echo "--- [${CURRENT}/${TOTAL}] ${model}: FAILED ---"
        FAILED_MODELS+=("$model")
    fi
done

echo ""
echo "============================================"
echo "Batch evaluation complete."
echo "  Total: ${TOTAL}, Failed: ${#FAILED_MODELS[@]}"
if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo "  Failed models: ${FAILED_MODELS[*]}"
fi
echo "============================================"
