# DriftST 消融实验 — 每步负样本采样数 (bank_sample_size)
# 用法: bash run_coad_ablation_samplesize.sh --sample <n> --gpu <N>
# 数据集: xenium COAD (TENX111), 只跑 fold 0
# 基线 bank_sample_size=1024;本消融跑 256 / 64。
# bank_size 固定 4096(与基线一致), 只调采样数。
# 其余超参(含三核 R_list)与 run_xenium_coad.sh 完全一致。

set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CODE_DIR="${REPO_DIR}"
PROCESS_DIR="${REPO_DIR}/process"
PYTHON="/data/buyonggan/miniconda3/envs/DriftST/bin/python"

# ── 模型超参 ──────────────────────────────────────────────────────────────────
N_GENES=280
HIDDEN_DIM=128
NUM_LAYERS=2
NUM_HEADS=4
DROPOUT=0.3
N_ATTN_LAYERS=2

# ── 训练超参 ──────────────────────────────────────────────────────────────────
BATCH_SIZE=512
LR=3e-4
WD=1e-4
EPOCHS=250
WARM_EPOCHS=0

# ── Drift 超参 (与基线一致, 三核) ─────────────────────────────────────────────
GEN_PER_SPOT=8
R_LIST="0.02 0.05 0.2"
DRIFT_STEP=1.0
DRIFT_WEIGHT=0.15
BANK_SIZE=4096

# ── Gate 超参 ─────────────────────────────────────────────────────────────────
GATE_ENTROPY_WEIGHT=0.02

# ── 路径 ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR="/data/buyonggan/DriftST/hest1k_datasets/xenium_coad/processed_data"
GENE_LIST="${OUTPUT_DIR}/selected_gene_list.txt"

DEVICE="cuda"
NUM_WORKERS=16
WANDB_PROJECT="DriftST-Ablation-SampleSize"
FOLD=0

# ── 消融参数 ──────────────────────────────────────────────────────────────────
SAMPLE_SIZE=""
GPU=4

while [[ $# -gt 0 ]]; do
    case $1 in
        --sample) SAMPLE_SIZE=$2;  shift 2 ;;
        --gpu)    GPU=$2;          shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if [ -z "${SAMPLE_SIZE}" ]; then
    echo "必须指定 --sample <n>"; exit 1
fi
export CUDA_VISIBLE_DEVICES=${GPU}

EXP_DIR="/data/buyonggan/DriftST/experiments/coad_ablation_sample_${SAMPLE_SIZE}"
FOLD_DIR="${EXP_DIR}/fold_${FOLD}"
mkdir -p "${FOLD_DIR}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
hr()  { echo "═══════════════════════════════════════════════════════"; }

SVG_FILE="${OUTPUT_DIR}/svg_ranking.json"
if [ ! -f "${SVG_FILE}" ]; then
    log "计算 SVG ranking (Moran's I) ..."
    $PYTHON "${PROCESS_DIR}/compute_svg.py" --data_dir "${OUTPUT_DIR}" --slide TENX111
fi

hr
log "消融实验 [bank_sample_size=${SAMPLE_SIZE}, bank_size=${BANK_SIZE}] | GPU=${GPU} | fold=${FOLD} | 输出: ${EXP_DIR}"
hr

TRAIN_ARGS=(
    --data_dir           "${OUTPUT_DIR}"
    --fold               "${FOLD}"
    --output_dir         "${FOLD_DIR}"
    --n_genes            "${N_GENES}"
    --hidden_dim         "${HIDDEN_DIM}"
    --num_layers         "${NUM_LAYERS}"
    --num_heads          "${NUM_HEADS}"
    --dropout            "${DROPOUT}"
    --n_attn_layers      "${N_ATTN_LAYERS}"
    --gen_per_spot       "${GEN_PER_SPOT}"
    --R_list             ${R_LIST}
    --drift_step         "${DRIFT_STEP}"
    --bank_size          "${BANK_SIZE}"
    --bank_sample_size   "${SAMPLE_SIZE}"
    --drift_weight       "${DRIFT_WEIGHT}"
    --warm_epochs        "${WARM_EPOCHS}"
    --epochs             "${EPOCHS}"
    --batch_size         "${BATCH_SIZE}"
    --lr                 "${LR}"
    --wd                 "${WD}"
    --num_workers        "${NUM_WORKERS}"
    --device             "${DEVICE}"
    --wandb_project      "${WANDB_PROJECT}"
    --wandb_name         "coad-sample${SAMPLE_SIZE}-fold${FOLD}"
    --use_gate
    --gate_entropy_weight "${GATE_ENTROPY_WEIGHT}"
)

PYTHONUNBUFFERED=1 $PYTHON -u "${CODE_DIR}/train.py" "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${FOLD_DIR}/train.log"

BEST_PCC=$(grep "Val PCC" "${FOLD_DIR}/train.log" \
    | grep -o "Val PCC: [0-9.]*" | awk '{print $3}' \
    | sort -n | tail -1)
hr
log "bank_sample_size=${SAMPLE_SIZE} 最佳 Val PCC: ${BEST_PCC}"
log "实验目录: ${EXP_DIR}"
hr
