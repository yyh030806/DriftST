#!/usr/bin/env bash
# Train DriftST on the Xenium breast cancer processed dataset.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"

N_GENES="${N_GENES:-280}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
NUM_LAYERS="${NUM_LAYERS:-2}"
NUM_HEADS="${NUM_HEADS:-4}"
DROPOUT="${DROPOUT:-0.3}"
N_ATTN_LAYERS="${N_ATTN_LAYERS:-2}"

BATCH_SIZE="${BATCH_SIZE:-512}"
LR="${LR:-3e-4}"
WD="${WD:-1e-4}"
EPOCHS="${EPOCHS:-250}"
WARM_EPOCHS="${WARM_EPOCHS:-0}"

GEN_PER_SPOT="${GEN_PER_SPOT:-8}"
R_LIST="${R_LIST:-0.02 0.05 0.2}"
DRIFT_STEP="${DRIFT_STEP:-1.0}"
BANK_SIZE="${BANK_SIZE:-4096}"
BANK_SAMPLE_SIZE="${BANK_SAMPLE_SIZE:-1024}"
DRIFT_WEIGHT="${DRIFT_WEIGHT:-0.15}"
GATE_ENTROPY_WEIGHT="${GATE_ENTROPY_WEIGHT:-0.02}"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/hest1k_datasets/xenium_janesick/processed_data}"
EXP_DIR="${EXP_DIR:-${REPO_DIR}/experiments/xenium_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-8}"
WANDB_PROJECT="${WANDB_PROJECT:-DriftST-Xenium}"
USE_NEIGHBOR=false
WANDB_OFFLINE=false
NO_WANDB=false
SNAPSHOT=false
SNAPSHOT_EPOCHS="${SNAPSHOT_EPOCHS:-0 1 2 3 4 5 7 9 12 15 19 24 30 38 48 60 75 95 120 150 190 249}"
FOLDS="${FOLDS:-0,1,2,3,4}"
SVG_SLIDE="${SVG_SLIDE:-TENX94}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --folds) FOLDS="$2"; shift 2 ;;
        --use-neighbor) USE_NEIGHBOR=true; shift ;;
        --wandb-offline) WANDB_OFFLINE=true; shift ;;
        --no-wandb) NO_WANDB=true; shift ;;
        --snapshot) SNAPSHOT=true; shift ;;
        --gpu) export CUDA_VISIBLE_DEVICES="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

log() { echo "[$(date '+%H:%M:%S')] $*"; }
hr() { echo "======================================================="; }

SVG_FILE="${OUTPUT_DIR}/svg_ranking.json"
if [[ ! -f "${SVG_FILE}" ]]; then
    log "Computing SVG ranking with Moran's I..."
    "${PYTHON}" "${REPO_DIR}/process/compute_svg.py" --data_dir "${OUTPUT_DIR}" --slide "${SVG_SLIDE}"
else
    log "Using existing SVG ranking: ${SVG_FILE}"
fi

N_SPLITS=$("${PYTHON}" -c "import json; print(len(json.load(open('${OUTPUT_DIR}/splits.json'))))")
log "splits.json contains ${N_SPLITS} folds"

IFS=',' read -ra FOLD_LIST <<< "${FOLDS}"
PCC_RESULTS=()

for FOLD in "${FOLD_LIST[@]}"; do
    FOLD_DIR="${EXP_DIR}/fold_${FOLD}"
    mkdir -p "${FOLD_DIR}"

    SPATIAL_INFO=$("${PYTHON}" -c "
import json
s = json.load(open('${OUTPUT_DIR}/splits.json'))
fold = s[${FOLD}]
info = fold.get('spatial_info', {})
print(f\"slide={fold['test_slide']} | band={info.get('coord_lo','?')}-{info.get('coord_hi','?')}\")
" 2>/dev/null || echo "fold ${FOLD}")

    hr
    log "Fold ${FOLD} | ${SPATIAL_INFO}"
    hr

    TRAIN_ARGS=(
        --data_dir "${OUTPUT_DIR}"
        --fold "${FOLD}"
        --output_dir "${FOLD_DIR}"
        --n_genes "${N_GENES}"
        --hidden_dim "${HIDDEN_DIM}"
        --num_layers "${NUM_LAYERS}"
        --num_heads "${NUM_HEADS}"
        --dropout "${DROPOUT}"
        --n_attn_layers "${N_ATTN_LAYERS}"
        --gen_per_spot "${GEN_PER_SPOT}"
        --R_list ${R_LIST}
        --drift_step "${DRIFT_STEP}"
        --bank_size "${BANK_SIZE}"
        --bank_sample_size "${BANK_SAMPLE_SIZE}"
        --drift_weight "${DRIFT_WEIGHT}"
        --warm_epochs "${WARM_EPOCHS}"
        --epochs "${EPOCHS}"
        --batch_size "${BATCH_SIZE}"
        --lr "${LR}"
        --wd "${WD}"
        --num_workers "${NUM_WORKERS}"
        --device "${DEVICE}"
        --wandb_project "${WANDB_PROJECT}"
        --wandb_name "xenium-fold${FOLD}-$(date +%m%d_%H%M)"
        --use_gate
        --gate_entropy_weight "${GATE_ENTROPY_WEIGHT}"
    )

    [[ "${USE_NEIGHBOR}" == true ]] && TRAIN_ARGS+=(--use_neighbor)
    [[ "${WANDB_OFFLINE}" == true ]] && TRAIN_ARGS+=(--wandb_offline)
    [[ "${NO_WANDB}" == true ]] && TRAIN_ARGS+=(--no_wandb)
    if [[ "${SNAPSHOT}" == true ]]; then
        SNAP_DIR="${REPO_DIR}/snapshots_xenium_fold${FOLD}"
        TRAIN_ARGS+=(--snapshot_epochs ${SNAPSHOT_EPOCHS})
        TRAIN_ARGS+=(--snapshot_dir "${SNAP_DIR}")
    fi

    PYTHONUNBUFFERED=1 "${PYTHON}" -u "${REPO_DIR}/train.py" "${TRAIN_ARGS[@]}" \
        2>&1 | tee "${FOLD_DIR}/train.log"

    BEST_PCC=$(grep "Val PCC" "${FOLD_DIR}/train.log" \
        | grep -o "Val PCC: [0-9.]*" | awk '{print $3}' \
        | sort -n | tail -1)
    PCC_RESULTS+=("fold${FOLD}=${BEST_PCC}")
    log "Fold ${FOLD} best Val PCC: ${BEST_PCC}"
done

hr
log "Fold summary"
for r in "${PCC_RESULTS[@]}"; do
    log "  ${r}"
done
AVG_PCC=$("${PYTHON}" -c "
vals = [float(r.split('=')[1]) for r in '${PCC_RESULTS[*]}'.split() if '=' in r]
print(f'{sum(vals)/len(vals):.4f}' if vals else 'N/A')
")
log "  mean PCC: ${AVG_PCC}"
log "Experiment directory: ${EXP_DIR}"
hr
