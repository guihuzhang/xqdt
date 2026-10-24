#!/usr/bin/env bash
# Shared helpers used by the public WebNLG Control Prefix D2T repair runners.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${PROJECT_DIR}"

: "${MODEL_CACHE:=.cache}"
: "${JUDGE_TEMPERATURE:=0.3}"
: "${JUDGE_SEED:=2023}"

# Tensor-parallel defaults can be overridden by the caller. Individual wrappers
# intentionally reset these to single-card defaults when appropriate.
: "${QWEN14_TP:=2}"
: "${QWEN32_TP:=4}"
: "${GEMMA12_TP:=2}"
: "${GEMMA27_TP:=4}"
: "${LLAMA70_TP:=4}"

export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$PWD/.cache/huggingface/hub}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$PWD/.cache/modelscope}"
export VLLM_NO_USAGE_STATS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

resolve_model_spec() {
  local tag="$1"
  case "${tag}" in
    qwen3_0p6b_nothink) echo "Qwen/Qwen3-0.6B|qwen3|0.6b|bfloat16|1|--disable_thinking" ;;
    qwen3_1p7b_nothink) echo "Qwen/Qwen3-1.7B|qwen3|1.7b|bfloat16|1|--disable_thinking" ;;
    qwen3_4b_nothink) echo "Qwen/Qwen3-4B|qwen3|4b|bfloat16|1|--disable_thinking" ;;
    qwen3_8b_nothink) echo "Qwen/Qwen3-8B|qwen3|8b|bfloat16|1|--disable_thinking" ;;
    qwen3_14b_nothink) echo "Qwen/Qwen3-14B|qwen3|14b|bfloat16|${QWEN14_TP}|--disable_thinking" ;;
    qwen3_32b_nothink) echo "Qwen/Qwen3-32B|qwen3|32b|bfloat16|${QWEN32_TP}|--disable_thinking" ;;
    gemma3_270m_it) echo "google/gemma-3-270m-it|gemma3|270m|bfloat16|1|" ;;
    gemma3_1b_it) echo "google/gemma-3-1b-it|gemma3|1b|bfloat16|1|" ;;
    gemma3_4b_it) echo "google/gemma-3-4b-it|gemma3|4b|bfloat16|1|" ;;
    gemma3_12b_it) echo "google/gemma-3-12b-it|gemma3|12b|bfloat16|${GEMMA12_TP}|" ;;
    gemma3_27b_it) echo "google/gemma-3-27b-it|gemma3|27b|bfloat16|${GEMMA27_TP}|" ;;
    llama3_2_1b_instruct) echo "meta-llama/Llama-3.2-1B-Instruct|llama3|1b|bfloat16|1|" ;;
    llama3_2_3b_instruct) echo "meta-llama/Llama-3.2-3B-Instruct|llama3|3b|bfloat16|1|" ;;
    llama3_1_8b_instruct) echo "meta-llama/Llama-3.1-8B-Instruct|llama3|8b|bfloat16|1|" ;;
    llama3_8b_instruct) echo "meta-llama/Meta-Llama-3-8B-Instruct|llama3|8b|bfloat16|1|" ;;
    llama3_70b_awq) echo "casperhansen/llama-3-70b-instruct-awq|llama3|70b|auto|${LLAMA70_TP}|" ;;
    llama3_3_70b_awq) echo "casperhansen/llama-3.3-70b-instruct-awq|llama3|70b|auto|${LLAMA70_TP}|" ;;
    *) echo "Unknown model tag: ${tag}" >&2; return 1 ;;
  esac
}

sanitize_for_log() {
  printf '%s' "$1" | tr '/ ' '__'
}

find_compatible_eval_cache() {
  local baseline_json="$1"
  local baseline_base
  baseline_base="$(basename "${baseline_json}")"
  python3 - "$PROJECT_DIR" "$baseline_base" "$JUDGE_TEMPERATURE" "$JUDGE_SEED" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
baseline = sys.argv[2]
judge_temp = float(sys.argv[3])
judge_seed = int(sys.argv[4])
found = []
for summary_path in sorted(root.rglob("run_summary.json")):
    try:
        data = json.loads(summary_path.read_text())
    except Exception:
        continue
    if data.get("evaluator_model") != "qwen8b":
        continue
    if os.path.basename(str(data.get("baseline_json", ""))) != baseline:
        continue
    try:
        if abs(float(data.get("judge_temperature")) - judge_temp) > 1e-9:
            continue
        if int(data.get("judge_seed")) != judge_seed:
            continue
    except Exception:
        continue
    candidate = summary_path.parent / "round1_eval_active.json"
    if candidate.is_file():
        found.append(str(candidate))
print(found[0] if found else "")
PY
}
