#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"
cd "${ROOT_DIR}"

: "${RUN_ROOT:=xqdt_outputs}"
: "${MANIFEST_PATH:=}"
: "${FORCE_RERUN:=0}"
: "${LOG_DIR:=.run_logs/score_repair_rounds_7metrics_$(date +%Y%m%d_%H%M%S)}"
: "${METRICS:=bertscore,bleu,chrf++,meteor,parent,rouge_l,sbert}"

mkdir -p "${LOG_DIR}"

resolve_run_dir() {
  local raw="$1"
  if [[ "${raw}" = /* ]]; then
    printf '%s\n' "${raw}"
  else
    printf '%s\n' "${ROOT_DIR}/${raw}"
  fi
}

dataset_from_run_dir() {
  local run_dir="$1"
  local name
  name="$(basename "${run_dir}")"
  case "${name}" in
    webnlg_*) echo "webnlg" ;;
    ood_*) echo "ood" ;;
    genwiki_*) echo "genwiki" ;;
    *)
      case "${run_dir}" in
        */webnlg_*) echo "webnlg" ;;
        */ood_*) echo "ood" ;;
        */genwiki_*) echo "genwiki" ;;
        *)
          echo ""
          ;;
      esac
      ;;
  esac
}

dataset_reference_json() {
  local dataset="$1"
  case "${dataset}" in
    webnlg) echo "${ROOT_DIR}/inputs/webnlg20_control_prefix_20_eval_input.json" ;;
    *)
      return 1
      ;;
  esac
}

collect_run_dirs_from_manifest() {
  local manifest="$1"
  python3 - "$manifest" <<'PY'
import pathlib
import sys

manifest = pathlib.Path(sys.argv[1])
seen = set()
with manifest.open("r", encoding="utf-8") as handle:
    for raw in handle:
        line = raw.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        run_dir = parts[-1].strip()
        if run_dir and run_dir not in seen:
            seen.add(run_dir)
            print(run_dir)
PY
}

collect_run_dirs_by_scan() {
  find "${ROOT_DIR}/${RUN_ROOT}" -type f -name 'round0_baseline.json' -print \
    | sed 's#/round0_baseline\.json$##' \
    | sort -u
}

score_one_run() {
  local run_dir="$1"
  local dataset="$2"
  local reference_json="$3"
  local output_json="${run_dir}/text_metrics/round_metrics_full_7metrics.json"
  local log_file="${LOG_DIR}/$(basename "${run_dir}").log"

  if [[ "${FORCE_RERUN}" != "1" && -f "${output_json}" ]]; then
    echo "[SKIP] ${run_dir} -> ${output_json}" | tee -a "${LOG_DIR}/summary.log"
    return 0
  fi

  mkdir -p "$(dirname "${output_json}")"
  echo "[START] dataset=${dataset} run_dir=${run_dir}" | tee -a "${LOG_DIR}/summary.log"
  echo "[REF] ${reference_json}" | tee -a "${LOG_DIR}/summary.log"

  python3 "${ROOT_DIR}/scoring/score_improve_rounds_full.py" \
    --run_dir "${run_dir}" \
    --reference_json "${reference_json}" \
    --metrics "${METRICS}" \
    --output "${output_json}" \
    > "${log_file}" 2>&1

  echo "[OK] ${run_dir} -> ${output_json}" | tee -a "${LOG_DIR}/summary.log"
}

main() {
  local -a run_dirs=()

  if [[ -n "${MANIFEST_PATH}" ]]; then
    if [[ ! -f "${MANIFEST_PATH}" ]]; then
      echo "[ERROR] manifest not found: ${MANIFEST_PATH}" >&2
      exit 1
    fi
    while IFS= read -r run_dir; do
      [[ -n "${run_dir}" ]] || continue
      run_dirs+=("$(resolve_run_dir "${run_dir}")")
    done < <(collect_run_dirs_from_manifest "${MANIFEST_PATH}")
  else
    while IFS= read -r run_dir; do
      [[ -n "${run_dir}" ]] || continue
      run_dirs+=("${run_dir}")
    done < <(collect_run_dirs_by_scan)
  fi

  if [[ "${#run_dirs[@]}" -eq 0 ]]; then
    echo "[ERROR] no run directories found" >&2
    exit 1
  fi

  local run_dir dataset reference_json
  for run_dir in "${run_dirs[@]}"; do
    if [[ ! -d "${run_dir}" ]]; then
      echo "[SKIP] missing run_dir: ${run_dir}" | tee -a "${LOG_DIR}/summary.log"
      continue
    fi
    dataset="$(dataset_from_run_dir "${run_dir}")"
    if [[ -z "${dataset}" ]]; then
      echo "[SKIP] cannot infer dataset for ${run_dir}" | tee -a "${LOG_DIR}/summary.log"
      continue
    fi
    reference_json="$(dataset_reference_json "${dataset}")"
    if [[ ! -f "${reference_json}" ]]; then
      echo "[SKIP] missing reference_json for ${run_dir}: ${reference_json}" | tee -a "${LOG_DIR}/summary.log"
      continue
    fi
    score_one_run "${run_dir}" "${dataset}" "${reference_json}"
  done

  echo "[DONE] logs: ${LOG_DIR}" | tee -a "${LOG_DIR}/summary.log"
}

main "$@"
