# 用法：bash run_experiment.sh [--skip-preprocess] [--fold N]

set -e

# ── 路径 ──────────────────────────────────────────────────────────────────────
DATA_DIR="/data/buyonggan/DriftST/hest1k_datasets/her2st"
OUTPUT_DIR="/data/buyonggan/DriftST/hest1k_datasets/her2st/processed_data"
EXP_DIR="/data/buyonggan/DriftST/experiments/dit_$(date +%Y%m%d_%H%M%S)"
CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
GENE_LIST="/data/buyonggan/DriftST/hest1k_datasets/her2st/processed_data/select_gene_list.txt"

# ── 预处理超参 ────────────────────────────────────────────────────────────────
NEIGHBOR_R=150

# ── 模型超参 ──────────────────────────────────────────────────────────────────
N_GENES=300
HIDDEN_DIM=256
NUM_LAYERS=4
NUM_HEADS=8
DROPOUT=0.1

# ── 训练超参 ──────────────────────────────────────────────────────────────────
BATCH_SIZE=256
LR=3e-5
WD=1e-4
EPOCHS=1000
T_MAX=${EPOCHS} 
PATIENCE=20
WARM_EPOCHS=0

# ── Drift 超参 ────────────────────────────────────────────────────────────────
GEN_PER_SPOT=8
R_LIST="0.02 0.05 0.2"
DRIFT_STEP=1.0
BANK_SIZE=4096
BANK_SAMPLE_SIZE=1024
DRIFT_WEIGHT=0.15

DEVICE="cuda"
export CUDA_VISIBLE_DEVICES=4
NUM_WORKERS=16

# ── wandb 设置 ────────────────────────────────────────────────────────────────
WANDB_PROJECT="DriftST"

# ── 参数解析 ──────────────────────────────────────────────────────────────────
SKIP_PREPROCESS=false
SINGLE_FOLD=-1

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-preprocess) SKIP_PREPROCESS=true; shift ;;
        --fold) SINGLE_FOLD=$2; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

log() { echo "[$(date '+%H:%M:%S')] $*"; }
hr()  { echo "═══════════════════════════════════════════════════════"; }

# ── Step 1: 预处理 ────────────────────────────────────────────────────────────
if [ "$SKIP_PREPROCESS" = false ]; then
    hr; log "Step 1: 预处理"; hr
    python "${CODE_DIR}/preprocess_her2st.py" \
        --data_dir   "${DATA_DIR}"   \
        --output_dir "${OUTPUT_DIR}" \
        --gene_list  "${GENE_LIST}"  \
        --neighbor_r "${NEIGHBOR_R}" \
        --batch_size "${BATCH_SIZE}" \
        --device     "${DEVICE}"
else
    log "跳过预处理，使用已有数据: ${OUTPUT_DIR}"
fi

# ── 读取 fold 数量 ────────────────────────────────────────────────────────────
N_FOLDS=$(python -c "
import json; splits=json.load(open('${OUTPUT_DIR}/splits.json')); print(len(splits))
")
log "共 ${N_FOLDS} 个 fold"

if [ "$SINGLE_FOLD" -ge 0 ] 2>/dev/null; then
    FOLDS=("$SINGLE_FOLD")
else
    FOLDS=($(seq 0 $((N_FOLDS - 1))))
fi

# ── Step 2: 逐 fold 训练 ──────────────────────────────────────────────────────
hr; log "Step 2: 训练 (${#FOLDS[@]} 个 fold)"; hr
mkdir -p "${EXP_DIR}"

for FOLD in "${FOLDS[@]}"; do
    FOLD_DIR="${EXP_DIR}/fold_${FOLD}"
    mkdir -p "${FOLD_DIR}"

    TEST_SLIDE=$(python -c "
import json; s=json.load(open('${OUTPUT_DIR}/splits.json')); print(s[${FOLD}]['test_slide'])
")
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
        --patience           "${PATIENCE}"          \
        --num_workers        "${NUM_WORKERS}"       \
        --device             "${DEVICE}"            \
        --wandb_project      "${WANDB_PROJECT}"     \
        --wandb_name         "fold${FOLD}-$(date +%m%d_%H%M)" \
        2>&1 | tee "${FOLD_DIR}/train.log"

    log "Fold ${FOLD} 完成 → ${FOLD_DIR}"
done

# ── Step 3: 汇总结果 ──────────────────────────────────────────────────────────
hr; log "Step 3: 汇总结果"; hr

python - <<PYEOF
import json, numpy as np, torch
from pathlib import Path

exp_dir  = Path("${EXP_DIR}")
all_pcc, rows = [], []

for fd in sorted(exp_dir.glob("fold_*")):
    ckpt_path = fd / "best_model.pt"
    if not ckpt_path.exists():
        print(f"  [警告] {fd.name}: 无 best_model.pt，跳过"); continue
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    pcc   = ckpt.get("val_pcc", float("nan"))
    slide = ckpt.get("test_slide", fd.name)
    all_pcc.append(pcc)
    rows.append({"fold": fd.name, "test_slide": slide, "val_pcc": float(pcc)})
    print(f"  {fd.name} | slide={str(slide):6s} | PCC={pcc:.4f}")

print("─" * 45)
print(f"  Mean PCC = {float(np.nanmean(all_pcc)):.4f} ± {float(np.nanstd(all_pcc)):.4f}")

summary = {
    "mean_pcc": float(np.nanmean(all_pcc)),
    "std_pcc":  float(np.nanstd(all_pcc)),
    "per_fold": rows,
}
with open(exp_dir / "results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n结果已保存 → {exp_dir}/results_summary.json")
PYEOF

hr; log "实验完成 → ${EXP_DIR}"; hr