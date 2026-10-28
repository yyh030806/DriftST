#!/usr/bin/env bash
# Prepare the default Xenium breast cancer processed dataset.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"
XENIUM_DIR="${XENIUM_DIR:-${REPO_DIR}/hest1k_datasets/xenium_janesick}"
OUTPUT_DIR="${OUTPUT_DIR:-${XENIUM_DIR}/processed_data}"
GENE_LIST="${GENE_LIST:-}"
SLIDES="${SLIDES:-TENX94 TENX95}"
TEST_SLIDE="${TEST_SLIDE:-TENX95}"
DEVICE="${DEVICE:-cuda}"

if [[ -z "${GENE_LIST}" || ! -f "${GENE_LIST}" ]]; then
    echo "Set GENE_LIST to an existing gene panel fixed independently of validation/test labels." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

"${PYTHON}" "${REPO_DIR}/process/preprocess_xenium_cell_level.py" \
    --data_dir "${XENIUM_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --gene_list "${GENE_LIST}" \
    --slides ${SLIDES} \
    --test_slide "${TEST_SLIDE}" \
    --neighbor_r "${NEIGHBOR_R:-300.0}" \
    --device "${DEVICE}"
