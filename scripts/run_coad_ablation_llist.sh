# DriftST 消融实验 — drift 三个尺度核 (R_list) 单独拿出来跑
# 用法: bash run_coad_ablation_llist.sh --r <R值> --gpu <N>
# 数据集: xenium COAD (TENX111), 只跑 fold 0
# 与 run_xenium_coad.sh 超参完全一致，唯一区别: R_list 只用单个尺度核。

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

# ── Drift 超参 ────────────────────────────────────────────────────────────────
GEN_PER_SPOT=8
DRIFT_STEP=1.0
BANK_SIZE=4096
BANK_SAMPLE_SIZE=1024
DRIFT_WEIGHT=0.15

# ── Gate 超参 ─────────────────────────────────────────────────────────────────
GATE_ENTROPY_WEIGHT=0.02

# ── 路径 ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR="/data/buyonggan/DriftST/hest1k_datasets/xenium_coad/processed_data"
GENE_LIST="${OUTPUT_DIR}/selected_gene_list.txt"

DEVICE="cuda"
NUM_WORKERS=16
WANDB_PROJECT="DriftST-Ablation-Lkernel"
FOLD=0

# ── 消融参数 ──────────────────────────────────────────────────────────────────
R_SINGLE=""
GPU=4

while [[ $# -gt 0 ]]; do
    case $1 in
        --r)    R_SINGLE=$2;                    shift 2 ;;
        --gpu)  GPU=$2;                         shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if [ -z "${R_SINGLE}" ]; then
    echo "必须指定 --r <R值>"; exit 1
fi
export CUDA_VISIBLE_DEVICES=${GPU}

# R 值标签 (0.02 -> r0p02)
R_TAG="r$(echo ${R_SINGLE} | tr '.' 'p')"
EXP_DIR="/data/buyonggan/DriftST/experiments/coad_ablation_llist_${R_TAG}"
FOLD_DIR="${EXP_DIR}/fold_${FOLD}"
mkdir -p "${FOLD_DIR}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
hr()  { echo "═══════════════════════════════════════════════════════"; }

# ── SVG 排序 (复用已有) ────────────────────────────────────────────────────────
SVG_FILE="${OUTPUT_DIR}/svg_ranking.json"
if [ ! -f "${SVG_FILE}" ]; then
    log "计算 SVG ranking (Moran's I) ..."
    $PYTHON "${PROCESS_DIR}/compute_svg.py" --data_dir "${OUTPUT_DIR}" --slide TENX111
fi

hr
log "消融实验 [单尺度核 R=${R_SINGLE}] | GPU=${GPU} | fold=${FOLD} | 输出: ${EXP_DIR}"
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
    --R_list             "${R_SINGLE}"
    --drift_step         "${DRIFT_STEP}"
    --bank_size          "${BANK_SIZE}"
    --bank_sample_size   "${BANK_SAMPLE_SIZE}"
    --drift_weight       "${DRIFT_WEIGHT}"
    --warm_epochs        "${WARM_EPOCHS}"
    --epochs             "${EPOCHS}"
    --batch_size         "${BATCH_SIZE}"
    --lr                 "${LR}"
    --wd                 "${WD}"
    --num_workers        "${NUM_WORKERS}"
    --device             "${DEVICE}"
    --wandb_project      "${WANDB_PROJECT}"
    --wandb_name         "coad-llist-${R_TAG}-fold${FOLD}"
    --use_gate
    --gate_entropy_weight "${GATE_ENTROPY_WEIGHT}"
)

PYTHONUNBUFFERED=1 $PYTHON -u "${CODE_DIR}/train.py" "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${FOLD_DIR}/train.log"

BEST_PCC=$(grep "Val PCC" "${FOLD_DIR}/train.log" \
    | grep -o "Val PCC: [0-9.]*" | awk '{print $3}' \
    | sort -n | tail -1)
hr
log "R=${R_SINGLE} 最佳 Val PCC: ${BEST_PCC}"
log "实验目录: ${EXP_DIR}"
hr
