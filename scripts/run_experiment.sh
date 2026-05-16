# 用法：bash run_experiment.sh [--dataset her2st|prad|kidney] [--skip-preprocess] [--fold N] [--wandb-offline]

set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CODE_DIR="${REPO_DIR}"
PROCESS_DIR="${REPO_DIR}/process"

# ── 预处理超参 ────────────────────────────────────────────────────────────────
NEIGHBOR_R=150

# ── 模型超参 ──────────────────────────────────────────────────────────────────
N_GENES=200
HIDDEN_DIM=128
NUM_LAYERS=2
NUM_HEADS=4          # head_dim = hidden_dim / num_heads = 128 / 4 = 32
DROPOUT=0.3
N_ATTN_LAYERS=2

# ── 训练超参 ──────────────────────────────────────────────────────────────────
BATCH_SIZE=512
LR=1e-5
WD=1e-4
EPOCHS=500
WARM_EPOCHS=0

# ── Drift 超参 ────────────────────────────────────────────────────────────────
GEN_PER_SPOT=8
R_LIST="0.02 0.05 0.2"
DRIFT_STEP=1.0
BANK_SIZE=4096
BANK_SAMPLE_SIZE=1024
DRIFT_WEIGHT=0.05

DEVICE="cuda"
export CUDA_VISIBLE_DEVICES=2
NUM_WORKERS=16

# ── 参数解析 ──────────────────────────────────────────────────────────────────
SKIP_PREPROCESS=false
WANDB_OFFLINE=false
FOLD=0
DATASET="her2st"

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-preprocess) SKIP_PREPROCESS=true; shift ;;
        --wandb-offline) WANDB_OFFLINE=true; shift ;;
        --fold) FOLD=$2; shift 2 ;;
        --dataset) DATASET=$2; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ── 数据集相关路径与设置 ──────────────────────────────────────────────────────
case "$DATASET" in
    her2st)
        DATA_DIR="/data/buyonggan/DriftST/hest1k_datasets/her2st"
        OUTPUT_DIR="/data/buyonggan/DriftST/hest1k_datasets/her2st/processed_data"
        EXP_DIR="/data/buyonggan/DriftST/experiments/her2st_$(date +%Y%m%d_%H%M%S)"
        GENE_LIST="/data/buyonggan/DriftST/hest1k_datasets/her2st/processed_data/select_gene_list.txt"
        WANDB_PROJECT="DriftST"
        PREPROCESS_BATCH_SIZE="${BATCH_SIZE}"
        SLIDES=""
        WANDB_NAME_PREFIX="fold"
        ;;
    prad)
        DATA_DIR="/data/buyonggan/DriftST/hest1k_datasets/PRAD"
        OUTPUT_DIR="/data/buyonggan/DriftST/hest1k_datasets/PRAD/processed_data"
        EXP_DIR="/data/buyonggan/DriftST/experiments/prad_$(date +%Y%m%d_%H%M%S)"
        GENE_LIST="/data/buyonggan/DriftST/hest1k_datasets/PRAD/select_gene_list.txt"
        WANDB_PROJECT="DriftST-PRAD"
        PREPROCESS_BATCH_SIZE=64
        SLIDES="MEND139 MEND140 MEND141 MEND142 MEND143 MEND144 MEND145 MEND146 MEND147 MEND148 MEND149 MEND150 MEND151 MEND152 MEND153 MEND154 MEND156 MEND157 MEND158 MEND159 MEND160 MEND161 MEND162"
        TEST_SLIDE_ARG="MEND145"
        WANDB_NAME_PREFIX="prad-fold"
        ;;
    kidney)
        DATA_DIR="/data/buyonggan/DriftST/hest1k_datasets/kidney"
        OUTPUT_DIR="/data/buyonggan/DriftST/hest1k_datasets/kidney/processed_data"
        EXP_DIR="/data/buyonggan/DriftST/experiments/kidney_$(date +%Y%m%d_%H%M%S)"
        GENE_LIST="/data/buyonggan/DriftST/hest1k_datasets/kidney/select_gene_list.txt"
        WANDB_PROJECT="DriftST-Kidney"
        PREPROCESS_BATCH_SIZE=64
        SLIDES="NCBI692 NCBI693 NCBI694 NCBI695 NCBI696 NCBI697 NCBI698 NCBI699 NCBI700 NCBI701 NCBI702 NCBI703 NCBI704 NCBI705 NCBI706 NCBI707 NCBI708 NCBI709 NCBI710 NCBI711 NCBI712 NCBI713 NCBI714"
        TEST_SLIDE_ARG="NCBI697"
        WANDB_NAME_PREFIX="kidney-fold"
        ;;
    *)
        echo "未知数据集: $DATASET（支持 her2st / prad / kidney）"; exit 1 ;;
esac

log() { echo "[$(date '+%H:%M:%S')] $*"; }
hr()  { echo "═══════════════════════════════════════════════════════"; }

# ── Step 1: 预处理 ────────────────────────────────────────────────────────────
if [ "$SKIP_PREPROCESS" = false ]; then
    hr; log "Step 1: 预处理 ${DATASET}"; hr
    PREPROCESS_ARGS=(
        --data_dir   "${DATA_DIR}"
        --output_dir "${OUTPUT_DIR}"
        --gene_list  "${GENE_LIST}"
        --neighbor_r "${NEIGHBOR_R}"
        --batch_size "${PREPROCESS_BATCH_SIZE}"
        --device     "${DEVICE}"
    )
    if [ -n "${SLIDES}" ]; then
        PREPROCESS_ARGS+=(--slides ${SLIDES})
    fi
    if [ -n "${TEST_SLIDE_ARG}" ]; then
        PREPROCESS_ARGS+=(--test_slide "${TEST_SLIDE_ARG}")
    fi
    python "${PROCESS_DIR}/preprocess.py" "${PREPROCESS_ARGS[@]}"
else
    log "跳过预处理，使用已有数据: ${OUTPUT_DIR}"
fi

# ── 读取测试 slide ────────────────────────────────────────────────────────────
TEST_SLIDE=$(python -c "
import json; s=json.load(open('${OUTPUT_DIR}/splits.json')); print(s[${FOLD}]['test_slide'])
")

FOLD_DIR="${EXP_DIR}/fold_${FOLD}"
mkdir -p "${FOLD_DIR}"

TRAIN_WANDB_ARGS=(
    --wandb_project "${WANDB_PROJECT}"
    --wandb_name "${WANDB_NAME_PREFIX}${FOLD}-$(date +%m%d_%H%M)"
)

if [ "$WANDB_OFFLINE" = true ]; then
    TRAIN_WANDB_ARGS+=(--wandb_offline)
fi

# ── Step 2: 训练 ──────────────────────────────────────────────────────────────
hr; log "Fold ${FOLD} — 测试 slide: ${TEST_SLIDE}"; hr

python "${CODE_DIR}/train.py" \
    --data_dir           "${OUTPUT_DIR}"        \
    --fold               "${FOLD}"              \
    --output_dir         "${FOLD_DIR}"          \
    --n_genes            "${N_GENES}"           \
    --hidden_dim         "${HIDDEN_DIM}"        \
    --num_layers         "${NUM_LAYERS}"        \
    --num_heads          "${NUM_HEADS}"         \
    --dropout            "${DROPOUT}"           \
    --n_attn_layers      "${N_ATTN_LAYERS}"     \
    --gen_per_spot       "${GEN_PER_SPOT}"      \
    --R_list             ${R_LIST}              \
    --drift_step         "${DRIFT_STEP}"        \
    --bank_size          "${BANK_SIZE}"         \
    --bank_sample_size   "${BANK_SAMPLE_SIZE}"  \
    --drift_weight       "${DRIFT_WEIGHT}"      \
    --warm_epochs        "${WARM_EPOCHS}"       \
    --epochs             "${EPOCHS}"            \
    --batch_size         "${BATCH_SIZE}"        \
    --lr                 "${LR}"                \
    --wd                 "${WD}"                \
    --num_workers        "${NUM_WORKERS}"       \
    --device             "${DEVICE}"            \
    --use_gate                                 \
    "${TRAIN_WANDB_ARGS[@]}"                   \
    2>&1 | tee "${FOLD_DIR}/train.log"

hr; log "实验完成 → ${FOLD_DIR}"; hr