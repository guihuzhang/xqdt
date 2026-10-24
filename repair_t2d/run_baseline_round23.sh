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
: "${ROUND1_ROOT:=${SCRIPT_DIR}/t2d_repair_output/baseline/round1}"
: "${ROUND2_BASELINE_ROOT:=${SCRIPT_DIR}/t2d_repair_output/baseline/round2_baseline}"
: "${ROUND2_SELF_REFINE_ROOT:=${SCRIPT_DIR}/t2d_repair_output/baseline/round2_self_refine}"
: "${ROUND3_BASELINE_ROOT:=${SCRIPT_DIR}/t2d_repair_output/baseline/round3_baseline}"
: "${ROUND3_SELF_REFINE_ROOT:=${SCRIPT_DIR}/t2d_repair_output/baseline/round3_self_refine}"
: "${CACHE_ROOT:=${SCRIPT_DIR}/.cache_runtime/baseline_round23}"
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
BASELINE_MODES=(pive_missing nli_binary factspotter_binary)
SELF_REFINE_GROUP_NAMES=(
  "gemma3_27b_qwen3_14b_qwen3_4b_gemma3_4b"
  "llama3_3_70b_awq_qwen3_8b_llama3_1_8b_qwen3_1.7b"
  "qwen3_32b_gemma3_12b_llama3_2_3b"
)
SELF_REFINE_GROUP_MODELS=(
  "gemma3_27b qwen3_14b qwen3_4b gemma3_4b"
  "llama3_3_70b_awq qwen3_8b llama3_1_8b qwen3_1.7b"
  "qwen3_32b gemma3_12b llama3_2_3b"
)

repair_batch_size_for_model() {
  case "$1" in
    llama3_3_70b_awq|qwen3_32b|gemma3_27b) echo 8 ;;
    qwen3_14b|qwen3_8b|llama3_1_8b|gemma3_12b) echo 16 ;;
    *) echo 32 ;;
  esac
}

repair_max_num_seqs_for_model() {
  case "$1" in
    llama3_3_70b_awq|qwen3_32b|gemma3_27b) echo 4 ;;
    qwen3_14b|qwen3_8b|llama3_1_8b|gemma3_12b) echo 8 ;;
    *) echo 16 ;;
  esac
}

mirror_baseline_model_artifacts() {
  local round_root="$1"
  local model="$2"
  local dataset="$3"
  local baseline="$4"
  local mode="$5"
  local model_root="${round_root}/${model}"

  mkdir -p "${model_root}/markdown/${dataset}"
  cp "${round_root}/markdown/${dataset}/vllm_text_to_data_${model}_${baseline}_${mode}_${dataset}_0shot_markdown.json" \
    "${model_root}/markdown/${dataset}/"

  mkdir -p "${model_root}/repair_records/${dataset}/${baseline}/${model}"
  cp -R "${round_root}/repair_records/${dataset}/${baseline}/${model}/${mode}" \
    "${model_root}/repair_records/${dataset}/${baseline}/${model}/"

  mkdir -p "${model_root}/signal_cache/${mode}/${dataset}/${baseline}"
  cp -R "${round_root}/signal_cache/${mode}/${dataset}/${baseline}/${model}/." \
    "${model_root}/signal_cache/${mode}/${dataset}/${baseline}/"
}

eval_per_model() {
  local round_root="$1"
  for model in "${MODEL_ARR[@]}"; do
    python3 -u "${SCRIPT_DIR}/eval/eval_text_to_data_repair.py" \
      --repair-root "${round_root}/${model}" \
      --datasets "${DATASET_ARR[@]}"
  done
}

run_baseline_signal_round() {
  local input_root="$1"
  local output_root="$2"

  for model in "${MODEL_ARR[@]}"; do
    local batch_size max_num_seqs model_cache
    batch_size="$(repair_batch_size_for_model "${model}")"
    max_num_seqs="$(repair_max_num_seqs_for_model "${model}")"
    model_cache="${CACHE_ROOT}/$(basename "${output_root}")/${model}"

    for dataset in "${DATASET_ARR[@]}"; do
      for baseline in "${BASELINE_ARR[@]}"; do
        for mode in "${BASELINE_MODES[@]}"; do
          input_json="${input_root}/markdown/${dataset}/vllm_text_to_data_${model}_${baseline}_${mode}_${dataset}_0shot_markdown.json"
          cmd=(
            python3 -u "${SCRIPT_DIR}/run_text_to_data_repair_baselines.py"
            --datasets "${dataset}"
            --baselines "${baseline}"
            --repair-models "${model}"
            --modes "${mode}"
            --input-json "${input_json}"
            --only-previously-flagged
            --output-root "${output_root}"
            --model-cache "${model_cache}"
            --repair-temperature "${REPAIR_TEMPERATURE}"
            --repair-seed "${REPAIR_SEED}"
            --repair-batch-size "${batch_size}"
            --repair-max-num-seqs "${max_num_seqs}"
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

          mirror_baseline_model_artifacts "${output_root}" "${model}" "${dataset}" "${baseline}" "${mode}"
        done
      done
    done
  done

  python3 -u "${SCRIPT_DIR}/eval/eval_text_to_data_repair.py" \
    --repair-root "${output_root}" \
    --datasets "${DATASET_ARR[@]}"
  eval_per_model "${output_root}"
}

run_self_refine_round() {
  local input_parent="$1"
  local output_parent="$2"
  local round_tag="$3"
  local group_summaries=()

  for idx in "${!SELF_REFINE_GROUP_NAMES[@]}"; do
    local group_name="${SELF_REFINE_GROUP_NAMES[$idx]}"
    local group_root="${output_parent}/${group_name}_${round_tag}"
    local group_models_str="${SELF_REFINE_GROUP_MODELS[$idx]}"
    read -r -a group_models <<< "${group_models_str}"
    local selected_group_models=()
    local requested_model
    for model in "${group_models[@]}"; do
      for requested_model in "${MODEL_ARR[@]}"; do
        if [[ "${model}" == "${requested_model}" ]]; then
          selected_group_models+=("${model}")
          break
        fi
      done
    done
    if [[ "${#selected_group_models[@]}" -eq 0 ]]; then
      continue
    fi
    local group_input_root="${input_parent}"
    if [[ "${round_tag}" == "r3" ]]; then
      group_input_root="${input_parent}/${group_name}_r2"
    fi

    for model in "${selected_group_models[@]}"; do
      local batch_size max_num_seqs model_cache
      batch_size="$(repair_batch_size_for_model "${model}")"
      max_num_seqs="$(repair_max_num_seqs_for_model "${model}")"
      model_cache="${CACHE_ROOT}/${group_name}_${round_tag}/${model}"

      for dataset in "${DATASET_ARR[@]}"; do
        for baseline in "${BASELINE_ARR[@]}"; do
          input_json="${group_input_root}/markdown/${dataset}/vllm_text_to_data_${model}_${baseline}_self_refine_v2_${dataset}_0shot_markdown.json"
          cmd=(
            python3 -u "${SCRIPT_DIR}/run_text_to_data_repair_baselines.py"
            --datasets "${dataset}"
            --baselines "${baseline}"
            --repair-models "${model}"
            --modes self_refine_v2
            --input-json "${input_json}"
            --only-previously-flagged
            --output-root "${group_root}"
            --model-cache "${model_cache}"
            --repair-temperature "${REPAIR_TEMPERATURE}"
            --repair-seed "${REPAIR_SEED}"
            --repair-batch-size "${batch_size}"
            --repair-max-num-seqs "${max_num_seqs}"
            --self-refine-feedback-batch-size "${SELF_REFINE_FEEDBACK_BATCH_SIZE}"
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
        done
      done
    done

    python3 -u "${SCRIPT_DIR}/eval/eval_text_to_data_repair.py" \
      --repair-root "${group_root}" \
      --datasets "${DATASET_ARR[@]}"
    group_summaries+=("${group_root}/eval_summary_text_to_data_repair.json")
  done

  python3 -u "${SCRIPT_DIR}/merge_eval_summaries.py" \
    --summary-files "${group_summaries[@]}" \
    --output "${output_parent}/eval_summary_text_to_data_repair.json"
}

run_baseline_signal_round "${ROUND1_ROOT}" "${ROUND2_BASELINE_ROOT}"
run_self_refine_round "${ROUND1_ROOT}" "${ROUND2_SELF_REFINE_ROOT}" "r2"
run_baseline_signal_round "${ROUND2_BASELINE_ROOT}" "${ROUND3_BASELINE_ROOT}"
run_self_refine_round "${ROUND2_SELF_REFINE_ROOT}" "${ROUND3_SELF_REFINE_ROOT}" "r3"
