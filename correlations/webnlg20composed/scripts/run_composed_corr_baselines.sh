#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MAIN_SCRIPT="$PACKAGE_DIR/corr_webnlg20_composed_baselines.py"

python "$MAIN_SCRIPT" "$@"
