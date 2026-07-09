#!/usr/bin/env bash
# Preprocess and train DriftST on spot-level HEST-style datasets.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"

DATASET="her2st"
FOLD=0
SKIP_PREPROCESS=false
WANDB_OFFLINE=false
NO_WANDB=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --fold) FOLD="$2"; shift 2 ;;
        --skip-preprocess) SKIP_PREPROCESS=true; shift ;;
        --wandb-offline) WANDB_OFFLINE=true; shift ;;
        --no-wandb) NO_WANDB=true; shift ;;
        --gpu) export CUDA_VISIBLE_DEVICES="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

case "${DATASET}" in
    her2st)
        DATA_DIR="${DATA_DIR:-${REPO_DIR}/hest1k_datasets/her2st}"
        OUTPUT_DIR="${OUTPUT_DIR:-${DATA_DIR}/processed_data}"
        GENE_LIST="${GENE_LIST:-${OUTPUT_DIR}/selected_gene_list.txt}"
        WANDB_PROJECT="${WANDB_PROJECT:-DriftST}"
        N_GENES="${N_GENES:-300}"
        ;;
    prad)
        DATA_DIR="${DATA_DIR:-${REPO_DIR}/hest1k_datasets/PRAD}"
        OUTPUT_DIR="${OUTPUT_DIR:-${DATA_DIR}/processed_data}"
        GENE_LIST="${GENE_LIST:-${DATA_DIR}/select_gene_list.txt}"
        WANDB_PROJECT="${WANDB_PROJECT:-DriftST-PRAD}"
        N_GENES="${N_GENES:-200}"
        ;;
    kidney)
        DATA_DIR="${DATA_DIR:-${REPO_DIR}/hest1k_datasets/kidney}"
        OUTPUT_DIR="${OUTPUT_DIR:-${DATA_DIR}/processed_data}"
        GENE_LIST="${GENE_LIST:-${DATA_DIR}/select_gene_list.txt}"
        WANDB_PROJECT="${WANDB_PROJECT:-DriftST-Kidney}"
        N_GENES="${N_GENES:-200}"
        ;;
    *)
        echo "Unknown dataset: ${DATASET} (expected her2st, prad, or kidney)"
        exit 1
        ;;
esac

EXP_DIR="${EXP_DIR:-${REPO_DIR}/experiments/${DATASET}_$(date +%Y%m%d_%H%M%S)}"
FOLD_DIR="${EXP_DIR}/fold_${FOLD}"
mkdir -p "${FOLD_DIR}"

if [[ "${SKIP_PREPROCESS}" == false ]]; then
    "${PYTHON}" "${REPO_DIR}/process/preprocess.py" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --gene_list "${GENE_LIST}"
fi

TEST_SLIDE=$("${PYTHON}" -c "import json; s=json.load(open('${OUTPUT_DIR}/splits.json')); print(s[${FOLD}]['test_slide'])")
echo "Training ${DATASET} fold ${FOLD}; test slide: ${TEST_SLIDE}"

TRAIN_ARGS=(
    --data_dir "${OUTPUT_DIR}"
    --fold "${FOLD}"
    --output_dir "${FOLD_DIR}"
    --n_genes "${N_GENES}"
    --hidden_dim "${HIDDEN_DIM:-128}"
    --num_layers "${NUM_LAYERS:-2}"
    --num_heads "${NUM_HEADS:-4}"
    --dropout "${DROPOUT:-0.3}"
    --n_attn_layers "${N_ATTN_LAYERS:-2}"
    --gen_per_spot "${GEN_PER_SPOT:-8}"
    --R_list ${R_LIST:-0.02 0.05 0.2}
    --drift_step "${DRIFT_STEP:-1.0}"
    --bank_size "${BANK_SIZE:-4096}"
    --bank_sample_size "${BANK_SAMPLE_SIZE:-1024}"
    --drift_weight "${DRIFT_WEIGHT:-0.15}"
    --warm_epochs "${WARM_EPOCHS:-0}"
    --epochs "${EPOCHS:-250}"
    --batch_size "${BATCH_SIZE:-512}"
    --lr "${LR:-3e-4}"
    --wd "${WD:-1e-4}"
    --num_workers "${NUM_WORKERS:-8}"
    --device "${DEVICE:-cuda}"
    --wandb_project "${WANDB_PROJECT}"
    --wandb_name "${DATASET}-fold${FOLD}-$(date +%m%d_%H%M)"
    --use_gate
    --gate_entropy_weight "${GATE_ENTROPY_WEIGHT:-0.02}"
)

[[ "${WANDB_OFFLINE}" == true ]] && TRAIN_ARGS+=(--wandb_offline)
[[ "${NO_WANDB}" == true ]] && TRAIN_ARGS+=(--no_wandb)

PYTHONUNBUFFERED=1 "${PYTHON}" -u "${REPO_DIR}/train.py" "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${FOLD_DIR}/train.log"

echo "Experiment directory: ${FOLD_DIR}"
