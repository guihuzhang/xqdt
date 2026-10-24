#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
METRIC_DIR="$PACKAGE_DIR/metric_results"
MAIN_SCRIPT="$PACKAGE_DIR/corr_webnlg20_composed.py"

selected=()
while IFS= read -r line; do
  selected+=("$line")
done < <(find "$METRIC_DIR" -maxdepth 1 -mindepth 1 -type d -name 'dt_llama*' -exec basename {} \; | sort)

if [ "${#selected[@]}" -eq 0 ]; then
  echo "No dt_llama* folders found under $METRIC_DIR" >&2
  exit 1
fi

python "$MAIN_SCRIPT" --output-suffix llama "$@" --selected "${selected[@]}"
