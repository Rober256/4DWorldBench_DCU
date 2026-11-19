#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUNNER=("${FOURDWORLDBENCH_PYTHON:-python}" "$PROJECT_ROOT/runner.py")
PROFILE=0
PROFILE_OUTPUT=""
ARGS=()

while (( $# )); do
    case "$1" in
        --hipprof)
            PROFILE=1
            shift
            ;;
        --hipprof-output)
            if [[ -z "${2:-}" || "$2" == --* ]]; then
                echo "--hipprof-output requires a path" >&2
                exit 2
            fi
            PROFILE=1
            PROFILE_OUTPUT="$2"
            shift 2
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

unset ALL_PROXY all_proxy
if (( PROFILE )); then
    PROFILE_OUTPUT="${PROFILE_OUTPUT:-./hiprof/evaluation_$(date +%Y%m%d_%H%M%S)_$$}"
    mkdir -p -- "$(dirname -- "$PROFILE_OUTPUT")"
    exec hipprof -o "$PROFILE_OUTPUT" "${RUNNER[@]}" "${ARGS[@]}"
fi
exec "${RUNNER[@]}" "${ARGS[@]}"
