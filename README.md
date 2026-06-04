# DriftST: Drifting from H&E Histology to Spatial Transcriptomics

## 目录结构

```
DriftST/
├── train.py                  # 训练入口
├── test.py                   # 评估 / 导出入口（加载 ckpt 跑 val，默认含方差后处理）
├── src/                      # 核心库
│   ├── model.py              #   模型（drift + ZINB + gate + 共表达先验）
│   ├── dataset.py
│   ├── drift_step.py         #   drift 采样 + PredictionBank
│   ├── evaluation.py         #   evaluate() + metrics_from_arrays()
│   └── postprocess.py        #   方差后处理（逐基因仿射校准）
├── process/                  # 数据预处理脚本
│   ├── preprocess.py
│   ├── preprocess_xenium.py
│   ├── compute_svg.py
│   ├── generate_xenium_5fold_splits.py
│   ├── select_xenium_genes.py
│   └── select_xenium_hvg.py
└── scripts/                  # 一键运行脚本
    ├── run_preprocess.sh     #   Xenium 预处理
    ├── run_xenium.sh         #   Xenium janesick 单切片 5-fold 训练
    ├── run_xenium_coad.sh    #   Xenium coad 单切片 5-fold 训练
    └── run_experiment.sh     #   her2st / prad / kidney 预处理 + 训练
```

## 用法

### Xenium（单切片空间 5-fold）

```bash
# 1. 预处理
bash scripts/run_preprocess.sh

# 2. 训练（janesick 乳腺癌 / coad 结肠癌）
bash scripts/run_xenium.sh --folds 0
bash scripts/run_xenium_coad.sh --folds 0

# 3. 评估
python test.py \
    --data_dir hest1k_datasets/xenium_janesick/processed_data \
    --fold 0 \
    --ckpt experiments/xenium_xxxx/fold_0/best_model.pt
```

### her2st / PRAD / kidney（leave-one-slide-out）

```bash
bash scripts/run_experiment.sh --dataset her2st --fold 0
bash scripts/run_experiment.sh --dataset prad   --fold 0
bash scripts/run_experiment.sh --dataset kidney --fold 0
```

`run_experiment.sh` 一条命令包含预处理 + 训练；若已预处理可加 `--skip-preprocess`。
