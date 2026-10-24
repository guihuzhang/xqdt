#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Preserve any explicit TP overrides passed from the caller before common.sh sets defaults.
QWEN14_TP_WAS_SET="${QWEN14_TP+x}"
QWEN32_TP_WAS_SET="${QWEN32_TP+x}"
GEMMA12_TP_WAS_SET="${GEMMA12_TP+x}"
GEMMA27_TP_WAS_SET="${GEMMA27_TP+x}"
LLAMA70_TP_WAS_SET="${LLAMA70_TP+x}"
OUT_ROOT_WAS_SET="${OUT_ROOT+x}"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/repair_qwen8_fixed_judge_common.sh"

: "${GPU:=0}"
if [[ -z "${OUT_ROOT_WAS_SET}" ]]; then
  OUT_ROOT="xqdt_outputs"
fi
: "${PROMPT_VARIANTS:=p1c_p2b}"
: "${MODEL_TAGS:=gemma3_4b_it,gemma3_12b_it,gemma3_27b_it,llama3_1_8b_instruct,llama3_3_70b_awq,qwen3_4b_nothink,qwen3_8b_nothink,qwen3_14b_nothink,qwen3_32b_nothink}"
: "${MAX_ROUNDS:=3}"
: "${RUN_SCORE_AT_END:=1}"
: "${FORCE_RERUN:=0}"
: "${DRY_RUN:=0}"
: "${REPAIR_BATCH_SIZE:=2}"
: "${REPAIR_MAX_NEW_TOKENS:=1280}"
: "${GENERATOR_MAX_NUM_SEQS:=4}"
: "${GPU_MEMORY_UTILIZATION:=0.75}"
: "${MAX_MODEL_LEN:=4096}"
: "${METRICS:=bertscore,bleu,chrf++,meteor,parent,rouge_l,sbert}"

# Single-card defaults for this wrapper unless the caller explicitly overrides them.
[[ -n "${QWEN14_TP_WAS_SET}" ]] || QWEN14_TP=1
[[ -n "${QWEN32_TP_WAS_SET}" ]] || QWEN32_TP=1
[[ -n "${GEMMA12_TP_WAS_SET}" ]] || GEMMA12_TP=1
[[ -n "${GEMMA27_TP_WAS_SET}" ]] || GEMMA27_TP=1
[[ -n "${LLAMA70_TP_WAS_SET}" ]] || LLAMA70_TP=1

PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
cd "${PROJECT_DIR}"

export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$PWD/.cache/huggingface/hub}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$PWD/.cache/modelscope}"
export VLLM_NO_USAGE_STATS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOG_DIR="${OUT_ROOT}/logs"
MANIFEST_PATH="${OUT_ROOT}/selected_run_dirs.tsv"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"
: > "${MANIFEST_PATH}"

WEBNLG_CONTROL_JSON="${WEBNLG_CONTROL_JSON:-inputs/webnlg20_control_prefix_20_eval_input.json}"

run_complete() {
  local run_dir="$1"
  [[ -f "${run_dir}/run_summary.json" ]]
}

reference_json_for_dataset() {
  case "$1" in
    webnlg) echo "${WEBNLG_CONTROL_JSON}" ;;
    *) echo "Unknown dataset: $1" >&2; exit 1 ;;
  esac
}

variant_output_dir() {
  local dataset="$1"
  local baseline_tag="$2"
  local repair_tag="$3"
  local prompt_variant="$4"
  echo "${OUT_ROOT}/${dataset}_${baseline_tag}_repair_${repair_tag}_prompt_${prompt_variant}"
}

append_manifest_variant() {
  local dataset="$1"
  local baseline_tag="$2"
  local repair_tag="$3"
  local prompt_variant="$4"
  local run_dir="$5"
  local line
  line="${dataset}	${baseline_tag}	${repair_tag}	${prompt_variant}	${run_dir}"
  if [[ -f "${MANIFEST_PATH}" ]] && grep -Fqx "${line}" "${MANIFEST_PATH}"; then
    return 0
  fi
  printf '%b\n' "${line}" >> "${MANIFEST_PATH}"
}

run_variant_model() {
  local dataset="$1"
  local baseline_tag="$2"
  local baseline_json="$3"
  local reuse_eval_json="$4"
  local repair_tag="$5"
  local prompt_variant="$6"

  local spec model_id model_type model_size dtype tp extra_flags
  spec="$(resolve_model_spec "${repair_tag}")"
  IFS='|' read -r model_id model_type model_size dtype tp extra_flags <<< "${spec}"

  local output_dir baseline_name label log_file
  output_dir="$(variant_output_dir "${dataset}" "${baseline_tag}" "${repair_tag}" "${prompt_variant}")"
  baseline_name="${dataset}_${baseline_tag}__repair_${repair_tag}__prompt_${prompt_variant}__qwen8_fixed_judge_r${MAX_ROUNDS}"
  label="${dataset} ${baseline_tag} -> ${repair_tag} / ${prompt_variant}"
  log_file="${LOG_DIR}/$(sanitize_for_log "${baseline_name}").log"

  [[ -f "${baseline_json}" ]] || { echo "[ERROR] Missing baseline json: ${baseline_json}" >&2; exit 1; }
  if [[ -n "${reuse_eval_json}" && ! -f "${reuse_eval_json}" ]]; then
    echo "[ERROR] Missing reuse eval json: ${reuse_eval_json}" >&2
    exit 1
  fi

  if [[ "${FORCE_RERUN}" != "1" ]] && run_complete "${output_dir}"; then
    echo "[SKIP] ${label}" | tee -a "${LOG_DIR}/summary.log"
    append_manifest_variant "${dataset}" "${baseline_tag}" "${repair_tag}" "${prompt_variant}" "${output_dir}"
    return 0
  fi

  local -a cmd=(
    python3 -u repair/improve_vllm_with_eval_loop.py
    --baseline_json "${baseline_json}"
    --output_dir "${output_dir}"
    --evaluator_model qwen8b
    --generator_model_id_or_path "${model_id}"
    --generator_model_type "${model_type}"
    --generator_model_size "${model_size}"
    --model_cache .cache
    --dataset_name "${dataset}"
    --baseline_name "${baseline_name}"
    --max_rounds "${MAX_ROUNDS}"
    --judge_temperature 0.3
    --judge_seed 2023
    --evaluator_tensor_parallel_size 1
    --evaluator_gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}"
    --repair_temperature 0.7
    --repair_seed 42
    --repair_feedback_format json
    --repair_context_mode single
    --repair_prompt_variant "${prompt_variant}"
    --repair_batch_size "${REPAIR_BATCH_SIZE}"
    --repair_max_new_tokens "${REPAIR_MAX_NEW_TOKENS}"
    --generator_dtype "${dtype}"
    --generator_tensor_parallel_size "${tp}"
    --generator_gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}"
    --generator_max_model_len "${MAX_MODEL_LEN}"
    --generator_max_num_seqs "${GENERATOR_MAX_NUM_SEQS}"
  )

  # Only round 1 may reuse an external judge cache.
  # Round 2 and round 3 are always judged from the current run state.
  if [[ -n "${reuse_eval_json}" ]]; then
    cmd+=(--reuse_round1_eval_json "${reuse_eval_json}")
  fi

  if [[ -n "${extra_flags}" ]]; then
    # shellcheck disable=SC2206
    local extra_array=( ${extra_flags} )
    cmd+=("${extra_array[@]}")
  fi

  echo "[START] ${label}" | tee -a "${LOG_DIR}/summary.log"
  if [[ -n "${reuse_eval_json}" ]]; then
    echo "[REUSE_R1_JUDGE] ${reuse_eval_json}" | tee -a "${LOG_DIR}/summary.log"
  else
    echo "[JUDGE] fresh qwen8 verify from round1" | tee -a "${LOG_DIR}/summary.log"
  fi
  echo "[CMD] CUDA_VISIBLE_DEVICES=${GPU} ${cmd[*]}" | tee "${log_file}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY_RUN] command assembled only; not launching model" | tee -a "${log_file}" "${LOG_DIR}/summary.log"
    return 0
  fi

  if [[ "${repair_tag}" == qwen3_* ]]; then
    env USE_HF=1 CUDA_VISIBLE_DEVICES="${GPU}" "${cmd[@]}" 2>&1 | tee -a "${log_file}"
  else
    env CUDA_VISIBLE_DEVICES="${GPU}" "${cmd[@]}" 2>&1 | tee -a "${log_file}"
  fi

  append_manifest_variant "${dataset}" "${baseline_tag}" "${repair_tag}" "${prompt_variant}" "${output_dir}"
}

score_dataset() {
  local dataset="$1"
  local ref_json
  ref_json="$(reference_json_for_dataset "${dataset}")"

  mapfile -t run_dirs < <(find "${OUT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name "${dataset}_*" | sort)
  if [[ "${#run_dirs[@]}" -eq 0 ]]; then
    echo "[METRICS] skip ${dataset}: no run dirs" | tee -a "${LOG_DIR}/summary.log"
    return 0
  fi

  local -a cmd=(
    python3 -u scoring/score_multiround_prompt_ablation.py
    --reference_json "${ref_json}"
    --metrics "${METRICS}"
    --output "${OUT_ROOT}/${dataset}_prompt_selection_multimodel_r${MAX_ROUNDS}_metrics.json"
  )
  local run_dir
  for run_dir in "${run_dirs[@]}"; do
    cmd+=(--run_dir "${run_dir}")
  done

  echo "[METRICS] ${dataset}" | tee -a "${LOG_DIR}/summary.log"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY_RUN] ${cmd[*]}" | tee "${LOG_DIR}/${dataset}_metrics.log"
    return 0
  fi
  "${cmd[@]}" 2>&1 | tee "${LOG_DIR}/${dataset}_metrics.log"
}

run_baseline_group() {
  local dataset="$1"
  local baseline_tag="$2"
  local baseline_json="$3"
  local preferred_reuse="$4"

  local reuse_cache=""
  if [[ -n "${preferred_reuse}" && -f "${preferred_reuse}" ]]; then
    reuse_cache="${preferred_reuse}"
    echo "[REUSE] ${dataset} ${baseline_tag} round1 judge cache -> ${reuse_cache}" | tee -a "${LOG_DIR}/summary.log"
  else
    reuse_cache="$(find_compatible_eval_cache "${baseline_json}")"
    if [[ -n "${reuse_cache}" ]]; then
      echo "[REUSE] ${dataset} ${baseline_tag} compatible round1 judge cache -> ${reuse_cache}" | tee -a "${LOG_DIR}/summary.log"
    fi
  fi

  local seed_done=0
  local prompt_variant repair_tag seed_output_dir
  for repair_tag in "${model_tags[@]}"; do
    for prompt_variant in "${prompt_variants[@]}"; do
      if [[ -z "${reuse_cache}" && "${seed_done}" == "0" ]]; then
        seed_output_dir="$(variant_output_dir "${dataset}" "${baseline_tag}" "${repair_tag}" "${prompt_variant}")"
        run_variant_model "${dataset}" "${baseline_tag}" "${baseline_json}" "" "${repair_tag}" "${prompt_variant}"
        reuse_cache="${seed_output_dir}/round1_eval_active.json"
        seed_done=1
      else
        run_variant_model "${dataset}" "${baseline_tag}" "${baseline_json}" "${reuse_cache}" "${repair_tag}" "${prompt_variant}"
      fi
    done
  done
}

IFS=',' read -r -a prompt_variants <<< "${PROMPT_VARIANTS}"
IFS=',' read -r -a model_tags <<< "${MODEL_TAGS}"

echo "Project: ${PROJECT_DIR}"
echo "GPU: ${GPU}"
echo "Out root: ${OUT_ROOT}"
echo "Prompt variants: ${PROMPT_VARIANTS}"
echo "Model tags: ${MODEL_TAGS}"
echo "Max rounds: ${MAX_ROUNDS}"
echo "Dry run: ${DRY_RUN}"
echo "Judge: qwen8b temp=0.3 seed=2023"
echo "Repair: temp=0.7 seed=42 format=json context=single"
echo "Batch: repair_batch_size=${REPAIR_BATCH_SIZE} max_num_seqs=${GENERATOR_MAX_NUM_SEQS}"
echo "Single-card TP defaults: qwen14=${QWEN14_TP} qwen32=${QWEN32_TP} gemma12=${GEMMA12_TP} gemma27=${GEMMA27_TP} llama70=${LLAMA70_TP}"

run_baseline_group webnlg control_prefix_20 "${WEBNLG_CONTROL_JSON}" ""

if [[ "${RUN_SCORE_AT_END}" == "1" ]]; then
  score_dataset webnlg
fi

echo "Done. Outputs under ${OUT_ROOT}"
