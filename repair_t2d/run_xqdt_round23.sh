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
: "${REPAIR_MODELS:=gemma3_4b gemma3_12b gemma3_27b llama3_1_8b llama3_3_70b_awq qwen3_4b qwen3_8b qwen3_14b qwen3_32b}"
: "${ROUND1_ROOT:=${SCRIPT_DIR}/t2d_repair_output/xqdt_output/round1}"
: "${ROUND2_ROOT:=${SCRIPT_DIR}/t2d_repair_output/xqdt_output/round2}"
: "${ROUND3_ROOT:=${SCRIPT_DIR}/t2d_repair_output/xqdt_output/round3}"
: "${CACHE_ROOT:=${SCRIPT_DIR}/.cache_runtime/xqdt_round23}"
: "${VERIFIER_TEMPERATURE:=0.3}"
: "${VERIFIER_SEED:=2023}"
: "${VERIFIER_BATCH_SIZE:=1}"
: "${REPAIR_TEMPERATURE:=0.7}"
: "${REPAIR_SEED:=42}"
: "${REPAIR_TP_OVERRIDES:=}"
: "${MAX_SAMPLES:=}"

read -r -a DATASET_ARR <<< "${DATASETS}"
read -r -a BASELINE_ARR <<< "${BASELINES}"
read -r -a MODEL_ARR <<< "${REPAIR_MODELS}"

repair_batch_size_for_model() {
  case "$1" in
    llama3_3_70b_awq|qwen3_32b|gemma3_27b) echo 1 ;;
    qwen3_14b|qwen3_8b|llama3_1_8b|gemma3_12b) echo 2 ;;
    *) echo 4 ;;
  esac
}

run_one_round() {
  local prev_root="$1"
  local round_root="$2"
  local round_tag="$3"

  for repair_model in "${MODEL_ARR[@]}"; do
    local repair_batch_size
    repair_batch_size="$(repair_batch_size_for_model "${repair_model}")"
    local model_cache="${CACHE_ROOT}/${round_tag}/${repair_model}"

    for dataset in "${DATASET_ARR[@]}"; do
      for baseline in "${BASELINE_ARR[@]}"; do
        input_json="${prev_root}/markdown/${dataset}/vllm_text_to_data_${repair_model}_${baseline}_g_bridge_recover_guard_${dataset}_0shot_markdown.json"
        cmd=(
          python3 -u "${SCRIPT_DIR}/run_text_to_data_verifier_repair.py"
          --datasets "${dataset}"
          --baselines "${baseline}"
          --repair-models "${repair_model}"
          --prompt-variants g_bridge_recover_guard
          --stage repair
          --input-json "${input_json}"
          --only-previously-flagged
          --output-root "${round_root}"
          --model-cache "${model_cache}"
          --verifier-temperature "${VERIFIER_TEMPERATURE}"
          --verifier-seed "${VERIFIER_SEED}"
          --verifier-batch-size "${VERIFIER_BATCH_SIZE}"
          --repair-temperature "${REPAIR_TEMPERATURE}"
          --repair-seed "${REPAIR_SEED}"
          --repair-batch-size "${repair_batch_size}"
        )
        if [[ -n "${REPAIR_TP_OVERRIDES}" ]]; then
          cmd+=(--repair-tp-overrides ${REPAIR_TP_OVERRIDES})
        fi
        if [[ -n "${MAX_SAMPLES}" ]]; then
          cmd+=(--max-samples "${MAX_SAMPLES}")
        fi

        echo "[RUN] ${round_tag} dataset=${dataset} baseline=${baseline} repair_model=${repair_model}"
        env \
          CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
          USE_HF="${USE_HF}" \
          VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS}" \
          VLLM_DO_NOT_TRACK="${VLLM_DO_NOT_TRACK}" \
          "${cmd[@]}"
      done
    done
  done

  python3 -u "${SCRIPT_DIR}/eval/eval_text_to_data_repair.py" \
    --repair-root "${round_root}" \
    --datasets "${DATASET_ARR[@]}"

  for repair_model in "${MODEL_ARR[@]}"; do
    python3 -u "${SCRIPT_DIR}/eval/eval_text_to_data_repair.py" \
      --repair-root "${round_root}/${repair_model}" \
      --datasets "${DATASET_ARR[@]}"
  done
}

run_one_round "${ROUND1_ROOT}" "${ROUND2_ROOT}" "round2"
run_one_round "${ROUND2_ROOT}" "${ROUND3_ROOT}" "round3"
