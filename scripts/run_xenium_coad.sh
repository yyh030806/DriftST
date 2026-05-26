# 用法：bash run_xenium.sh [--folds 0,1,2,3,4] [--use-neighbor] [--wandb-offline] [--gpu N]
# 预处理已完成；先运行 generate_xenium_5fold_splits.py 生成 splits.json，再执行本脚本。
# 仅使用图像特征 + 基因表达矩阵（全局共表达 R 在 train.py 中自动计算）。

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
R_LIST="0.02 0.05 0.2"
DRIFT_STEP=1.0
BANK_SIZE=4096
BANK_SAMPLE_SIZE=1024
DRIFT_WEIGHT=0.15

# ── Gate 超参 ─────────────────────────────────────────────────────────────────
GATE_ENTROPY_WEIGHT=0.02

# ── 路径 ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR="/data/buyonggan/DriftST/hest1k_datasets/xenium_coad/processed_data"
EXP_DIR="/data/buyonggan/DriftST/experiments/xenium_coad_$(date +%Y%m%d_%H%M%S)"
GENE_LIST="${OUTPUT_DIR}/selected_gene_list.txt"

DEVICE="cuda"
export CUDA_VISIBLE_DEVICES=4
NUM_WORKERS=16

WANDB_PROJECT="DriftST-Xenium"
USE_NEIGHBOR=false
WANDB_OFFLINE=false
SNAPSHOT=false
SNAPSHOT_EPOCHS="0 1 2 3 4 5 7 9 12 15 19 24 30 38 48 60 75 95 120 150 190 249"  # 对数间隔，同 snapshots_xenium_d1
FOLDS="0,1,2,3,4"   # 逗号分隔，支持只跑部分折，例如 --folds 0,1

# ── 参数解析 ──────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --folds)              FOLDS=$2;                       shift 2 ;;
        --use-neighbor)       USE_NEIGHBOR=true;              shift ;;
        --wandb-offline)      WANDB_OFFLINE=true;             shift ;;
        --snapshot)           SNAPSHOT=true;                 shift ;;
        --gpu)                export CUDA_VISIBLE_DEVICES=$2; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

log() { echo "[$(date '+%H:%M:%S')] $*"; }
hr()  { echo "═══════════════════════════════════════════════════════"; }

# ── 计算 SVG 排序（若不存在则生成）─────────────────────────────────────────────
SVG_FILE="${OUTPUT_DIR}/svg_ranking.json"
if [ ! -f "${SVG_FILE}" ]; then
    log "计算 SVG ranking (Moran's I) ..."
    $PYTHON "${PROCESS_DIR}/compute_svg.py" --data_dir "${OUTPUT_DIR}" --slide TENX111
else
    log "SVG ranking 已存在，跳过计算: ${SVG_FILE}"
fi

# ── 验证 splits.json ──────────────────────────────────────────────────────────
N_SPLITS=$($PYTHON -c "import json; print(len(json.load(open('${OUTPUT_DIR}/splits.json'))))")
log "splits.json: ${N_SPLITS} 折"

# ── 依次训练各折 ──────────────────────────────────────────────────────────────
IFS=',' read -ra FOLD_LIST <<< "$FOLDS"

PCC_RESULTS=()

for FOLD in "${FOLD_LIST[@]}"; do
    FOLD_DIR="${EXP_DIR}/fold_${FOLD}"
    mkdir -p "${FOLD_DIR}"

    SPATIAL_INFO=$($PYTHON -c "
import json
s = json.load(open('${OUTPUT_DIR}/splits.json'))
fold = s[${FOLD}]
info = fold.get('spatial_info', {})
print(f\"slide={fold['test_slide']} | band={info.get('coord_lo','?'):.0f}-{info.get('coord_hi','?'):.0f}px\")
" 2>/dev/null || echo "fold ${FOLD}")

    hr
    log "Fold ${FOLD} / $(( ${#FOLD_LIST[@]} - 1 ))  —  ${SPATIAL_INFO}"
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
        --bank_sample_size   "${BANK_SAMPLE_SIZE}"
        --drift_weight       "${DRIFT_WEIGHT}"
        --warm_epochs        "${WARM_EPOCHS}"
        --epochs             "${EPOCHS}"
        --batch_size         "${BATCH_SIZE}"
        --lr                 "${LR}"
        --wd                 "${WD}"
        --num_workers        "${NUM_WORKERS}"
        --device             "${DEVICE}"
        --wandb_project          "${WANDB_PROJECT}"
        --wandb_name             "xenium-fold${FOLD}-$(date +%m%d_%H%M)"
        --use_gate
        --gate_entropy_weight    "${GATE_ENTROPY_WEIGHT}"
    )

    if [ "$USE_NEIGHBOR" = true ]; then
        TRAIN_ARGS+=(--use_neighbor)
    fi

    if [ "$WANDB_OFFLINE" = true ]; then
        TRAIN_ARGS+=(--wandb_offline)
    fi

    if [ "$SNAPSHOT" = true ]; then
        SNAP_DIR="${REPO_DIR}/snapshots_xenium_coad_zinb_fold${FOLD}"
        TRAIN_ARGS+=(--snapshot_epochs ${SNAPSHOT_EPOCHS})
        TRAIN_ARGS+=(--snapshot_dir "${SNAP_DIR}")
        log "snapshot 开启 → ${SNAP_DIR}  (epochs: ${SNAPSHOT_EPOCHS})"
    fi

    PYTHONUNBUFFERED=1 $PYTHON -u "${CODE_DIR}/train.py" "${TRAIN_ARGS[@]}" \
        2>&1 | tee "${FOLD_DIR}/train.log"

    # 提取当折最佳 PCC
    BEST_PCC=$(grep "Val PCC" "${FOLD_DIR}/train.log" \
        | grep -o "Val PCC: [0-9.]*" | awk '{print $3}' \
        | sort -n | tail -1)
    PCC_RESULTS+=("fold${FOLD}=${BEST_PCC}")
    log "Fold ${FOLD} 最佳 Val PCC: ${BEST_PCC}"
done

# ── 汇总 ─────────────────────────────────────────────────────────────────────
hr
log "5-fold 结果汇总"
for r in "${PCC_RESULTS[@]}"; do
    log "  ${r}"
done

# 计算均值
AVG_PCC=$($PYTHON -c "
vals = [float(r.split('=')[1]) for r in '${PCC_RESULTS[*]}'.split() if '=' in r]
print(f'{sum(vals)/len(vals):.4f}' if vals else 'N/A')
")
log "  平均 PCC: ${AVG_PCC}"
log "实验目录: ${EXP_DIR}"
hr
