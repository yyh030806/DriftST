#!/usr/bin/env bash
# Train DriftST on the Xenium COAD processed dataset.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/hest1k_datasets/xenium_coad/processed_data}"
export EXP_DIR="${EXP_DIR:-${REPO_DIR}/experiments/xenium_coad_$(date +%Y%m%d_%H%M%S)}"
export SVG_SLIDE="${SVG_SLIDE:-TENX111}"

bash "${REPO_DIR}/scripts/run_xenium.sh" "$@"
