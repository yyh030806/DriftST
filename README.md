# DriftST: Drifting from H&E Histology to Spatial Transcriptomics

## 目录结构

```
DriftST/
├── train.py                  # 训练入口
├── test.py                   # 评估入口（加载 ckpt 跑 val）
├── src/                      # 核心库
│   ├── model.py
│   ├── dataset.py
│   ├── drift_step.py
│   └── evaluation.py
├── process/                  # 数据预处理脚本
│   ├── preprocess.py
│   ├── preprocess_xenium.py
│   ├── compute_svg.py
│   ├── generate_xenium_5fold_splits.py
│   └── select_xenium_genes.py
└── scripts/                  # 一键运行脚本
    ├── run_xenium.sh
    ├── run_preprocess.sh
    └── run_experiment.sh
```

## 用法

### 预处理
```bash
bash scripts/run_preprocess.sh
```

### 训练（Xenium 5-fold）
```bash
bash scripts/run_xenium.sh --folds 0 --use-neighbor
```

### 评估
```bash
python test.py \
    --data_dir hest1k_datasets/xenium_janesick/processed_data \
    --fold 0 \
    --ckpt experiments/xenium_xxxx/fold_0/best.pt \
    --use_neighbor
```
