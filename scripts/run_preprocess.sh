#!/bin/bash
# Xenium (TENX94) 预处理：生成基因列表 + 重建表达矩阵 + 提取图像特征
set -e

PYTHON="/data/buyonggan/miniconda3/envs/DriftST/bin/python"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROCESS_DIR="${REPO_DIR}/process"
XENIUM_DIR="/data/buyonggan/DriftST/hest1k_datasets/xenium_janesick"

# Step 1: 生成基因列表（黑名单过滤）
$PYTHON "${PROCESS_DIR}/select_xenium_genes.py"

# Step 2: 预处理（仅 TENX94）
CUDA_VISIBLE_DEVICES=3 $PYTHON "${PROCESS_DIR}/preprocess_xenium.py" \
    --data_dir   "${XENIUM_DIR}" \
    --output_dir "${XENIUM_DIR}/processed_data" \
    --gene_list  "${XENIUM_DIR}/processed_data/selected_gene_list.txt" \
    --slides     TENX94 \
    --test_slide TENX94 \
    --overlaps_nucleus \
    --min_counts 70 \
    --batch_size 256 \
    --device cuda \
    2>&1 | tee "${REPO_DIR}/preprocess_xenium.log"
