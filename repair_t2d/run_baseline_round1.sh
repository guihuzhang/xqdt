#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

: "${USE_HF:=1}"
: "${VLLM_NO_USAGE_STATS:=1}"
: "${VLLM_DO_NOT_TRACK:=1}"
: "${CUDA_VISIBLE_DEVICES:=0}"
: "${DATASETS:=genwiki webnlg}"
: "${BASELINES:=bt5 regen}"
: "${REPAIR_MODELS:=gemma3_4b gemma3_12b gemma3_27b llama3_1_8b llama3_2_3b llama3_3_70b_awq qwen3_1.7b qwen3_4b qwen3_8b qwen3_14b qwen3_32b}"
: "${MODES:=pive_missing nli_binary factspotter_binary self_refine_v2}"
: "${OUTPUT_ROOT:=${SCRIPT_DIR}/t2d_repair_output/baseline/round1}"
: "${CACHE_ROOT:=${SCRIPT_DIR}/.cache_runtime/baseline_round1}"
: "${REPAIR_TEMPERATURE:=0.7}"
: "${REPAIR_SEED:=42}"
: "${REPAIR_TP_OVERRIDES:=}"
: "${MAX_SAMPLES:=}"
: "${NLI_THRESHOLD:=0.5}"
: "${NLI_GLOBAL_THRESHOLD:=0.5}"
: "${PIVE_SIGNAL_BATCH_SIZE:=4}"
: "${NLI_SIGNAL_BATCH_SIZE:=64}"
: "${FACTSPOTTER_SIGNAL_BATCH_SIZE:=64}"
: "${SELF_REFINE_FEEDBACK_BATCH_SIZE:=8}"
: "${FACTSPOTTER_BACKEND:=deberta_base}"
: "${FACTSPOTTER_THRESHOLD:=0.5}"
: "${PIVE_MAX_NEW_TOKENS:=256}"

read -r -a DATASET_ARR <<< "${DATASETS}"
read -r -a BASELINE_ARR <<< "${BASELINES}"
read -r -a MODEL_ARR <<< "${REPAIR_MODELS}"
read -r -a MODE_ARR <<< "${MODES}"

cmd=(
  python3 -u "${SCRIPT_DIR}/run_text_to_data_repair_baselines.py"
  --datasets "${DATASET_ARR[@]}"
  --baselines "${BASELINE_ARR[@]}"
  --repair-models "${MODEL_ARR[@]}"
  --modes "${MODE_ARR[@]}"
  --output-root "${OUTPUT_ROOT}"
  --model-cache "${CACHE_ROOT}"
  --repair-temperature "${REPAIR_TEMPERATURE}"
  --repair-seed "${REPAIR_SEED}"
  --nli-threshold "${NLI_THRESHOLD}"
  --nli-global-threshold "${NLI_GLOBAL_THRESHOLD}"
  --pive-signal-batch-size "${PIVE_SIGNAL_BATCH_SIZE}"
  --nli-signal-batch-size "${NLI_SIGNAL_BATCH_SIZE}"
  --factspotter-signal-batch-size "${FACTSPOTTER_SIGNAL_BATCH_SIZE}"
  --self-refine-feedback-batch-size "${SELF_REFINE_FEEDBACK_BATCH_SIZE}"
  --factspotter-backend "${FACTSPOTTER_BACKEND}"
  --factspotter-threshold "${FACTSPOTTER_THRESHOLD}"
  --pive-max-new-tokens "${PIVE_MAX_NEW_TOKENS}"
)
if [[ -n "${REPAIR_TP_OVERRIDES}" ]]; then
  cmd+=(--repair-tp-overrides ${REPAIR_TP_OVERRIDES})
fi
if [[ -n "${MAX_SAMPLES}" ]]; then
  cmd+=(--max-samples "${MAX_SAMPLES}")
fi

env \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  USE_HF="${USE_HF}" \
  VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS}" \
  VLLM_DO_NOT_TRACK="${VLLM_DO_NOT_TRACK}" \
  "${cmd[@]}"

python3 -u "${SCRIPT_DIR}/eval/eval_text_to_data_repair.py" \
  --repair-root "${OUTPUT_ROOT}" \
  --datasets "${DATASET_ARR[@]}" \
  --include-external
