#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_ROOT_WAS_SET="${OUT_ROOT+x}"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/repair_qwen8_fixed_judge_common.sh"

: "${GPU:=0}"
if [[ -z "${OUT_ROOT_WAS_SET}" ]]; then
  OUT_ROOT="baseline_outputs/self_refine_group_a"
fi
: "${MODEL_TAGS:=gemma3_4b_it,qwen3_4b_nothink,llama3_1_8b_instruct,qwen3_8b_nothink,gemma3_12b_it,qwen3_14b_nothink,gemma3_27b_it,qwen3_32b_nothink,llama3_3_70b_awq}"
: "${MAX_ROUNDS:=3}"
: "${RUN_SCORE_AT_END:=1}"
: "${FORCE_RERUN:=0}"
: "${REPAIR_BATCH_SIZE:=2}"
: "${REPAIR_MAX_NEW_TOKENS:=1280}"
: "${GENERATOR_MAX_NUM_SEQS:=4}"
: "${GPU_MEMORY_UTILIZATION:=0.75}"
: "${MAX_MODEL_LEN:=4096}"
: "${METRICS:=bertscore,bleu,chrf++,meteor,parent,rouge_l,sbert}"

# Single-card defaults.
: "${QWEN14_TP:=1}"
: "${QWEN32_TP:=1}"
: "${GEMMA12_TP:=1}"
: "${GEMMA27_TP:=1}"
: "${LLAMA70_TP:=1}"

PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
cd "${PROJECT_DIR}"

export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$PWD/.cache/huggingface/hub}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$PWD/.cache/modelscope}"
export VLLM_NO_USAGE_STATS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

WEBNLG_CONTROL_JSON="${WEBNLG_CONTROL_JSON:-inputs/webnlg20_control_prefix_20_eval_input.json}"

reference_json_for_dataset() {
  case "$1" in
    webnlg) echo "${WEBNLG_CONTROL_JSON}" ;;
    *) echo "Unknown dataset: $1" >&2; exit 1 ;;
  esac
}

run_complete() {
  local run_dir="$1"
  [[ -f "${run_dir}/run_summary.json" ]]
}

run_variant_model() {
  local dataset="$1"
  local baseline_tag="$2"
  local baseline_json="$3"
  local repair_tag="$4"

  local spec model_id model_type model_size dtype tp extra_flags
  spec="$(resolve_model_spec "${repair_tag}")"
  IFS='|' read -r model_id model_type model_size dtype tp extra_flags <<< "${spec}"

  local output_dir baseline_name label log_file
  output_dir="${OUT_ROOT}/${dataset}_${baseline_tag}_self_refine_${repair_tag}"
  baseline_name="${dataset}_${baseline_tag}__self_refine__repair_${repair_tag}__r${MAX_ROUNDS}"
  label="${dataset} ${baseline_tag} -> ${repair_tag} / self_refine"
  log_file="${LOG_DIR}/$(sanitize_for_log "${baseline_name}").log"

  if [[ "${FORCE_RERUN}" != "1" ]] && run_complete "${output_dir}"; then
    echo "[SKIP] ${label}" | tee -a "${LOG_DIR}/summary.log"
    return 0
  fi

  local -a cmd=(
    python3 -u repair/improve_vllm_with_self_refine_loop.py
    --baseline_json "${baseline_json}"
    --output_dir "${output_dir}"
    --dataset_name "${dataset}"
    --baseline_name "${baseline_tag}"
    --max_rounds "${MAX_ROUNDS}"
    --generator_model_id_or_path "${model_id}"
    --generator_model_type "${model_type}"
    --generator_model_size "${model_size}"
    --repair_temperature 0.7
    --repair_seed 42
    --repair_batch_size "${REPAIR_BATCH_SIZE}"
    --repair_max_new_tokens "${REPAIR_MAX_NEW_TOKENS}"
    --generator_dtype "${dtype}"
    --generator_tensor_parallel_size "${tp}"
    --generator_gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}"
    --generator_max_model_len "${MAX_MODEL_LEN}"
    --generator_max_num_seqs "${GENERATOR_MAX_NUM_SEQS}"
    --model_cache .cache
  )

  if [[ -n "${extra_flags}" ]]; then
    # shellcheck disable=SC2206
    local extra_array=( ${extra_flags} )
    cmd+=("${extra_array[@]}")
  fi

  echo "[START] ${label}" | tee -a "${LOG_DIR}/summary.log"
  echo "[CMD] CUDA_VISIBLE_DEVICES=${GPU} ${cmd[*]}" | tee "${log_file}"
  if [[ "${repair_tag}" == qwen3_* ]]; then
    env USE_HF=1 CUDA_VISIBLE_DEVICES="${GPU}" "${cmd[@]}" 2>&1 | tee -a "${log_file}"
  else
    env CUDA_VISIBLE_DEVICES="${GPU}" "${cmd[@]}" 2>&1 | tee -a "${log_file}"
  fi
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
    --output "${OUT_ROOT}/${dataset}_self_refine_multiround_metrics.json"
  )
  local run_dir
  for run_dir in "${run_dirs[@]}"; do
    cmd+=(--run_dir "${run_dir}")
  done

  echo "[METRICS] ${dataset}" | tee -a "${LOG_DIR}/summary.log"
  "${cmd[@]}" 2>&1 | tee "${LOG_DIR}/${dataset}_metrics.log"
}

IFS=',' read -r -a model_tags <<< "${MODEL_TAGS}"

echo "Project: ${PROJECT_DIR}"
echo "GPU: ${GPU}"
echo "Out root: ${OUT_ROOT}"
echo "Model tags: ${MODEL_TAGS}"
echo "Max rounds: ${MAX_ROUNDS}"
echo "Batch: repair_batch_size=${REPAIR_BATCH_SIZE} max_num_seqs=${GENERATOR_MAX_NUM_SEQS}"

for repair_tag in "${model_tags[@]}"; do
  run_variant_model webnlg control_prefix_20 "${WEBNLG_CONTROL_JSON}" "${repair_tag}"
done

if [[ "${RUN_SCORE_AT_END}" == "1" ]]; then
  score_dataset webnlg
fi

echo "Done. Outputs under ${OUT_ROOT}"
