#!/usr/bin/env bash
# Prepare the default Xenium breast cancer processed dataset.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"
XENIUM_DIR="${XENIUM_DIR:-${REPO_DIR}/hest1k_datasets/xenium_janesick}"
OUTPUT_DIR="${OUTPUT_DIR:-${XENIUM_DIR}/processed_data}"
GENE_LIST="${GENE_LIST:-${OUTPUT_DIR}/selected_gene_list.txt}"
SLIDES="${SLIDES:-TENX94 TENX95}"
TEST_SLIDE="${TEST_SLIDE:-TENX95}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${OUTPUT_DIR}"

"${PYTHON}" "${REPO_DIR}/process/select_xenium_cell_level_genes.py" \
    --transcripts_dir "${XENIUM_DIR}/transcripts" \
    --out_file "${GENE_LIST}"

"${PYTHON}" "${REPO_DIR}/process/preprocess_xenium_cell_level.py" \
    --data_dir "${XENIUM_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --gene_list "${GENE_LIST}" \
    --slides ${SLIDES} \
    --test_slide "${TEST_SLIDE}" \
    --neighbor_r "${NEIGHBOR_R:-300.0}" \
    --device "${DEVICE}"
