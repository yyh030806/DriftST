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
├── scripts/                  # 一键运行脚本
│   ├── run_preprocess.sh     #   Xenium 预处理
│   ├── run_xenium.sh         #   Xenium janesick 单切片 5-fold 训练
│   ├── run_xenium_coad.sh    #   Xenium coad 单切片 5-fold 训练
│   ├── run_experiment.sh     #   her2st / prad / kidney 预处理 + 训练
│   └── run_coad_ablation_*.sh#   消融实验脚本
├── eval/                     # 独立评估脚本
│   └── eval_dist_metrics.py  #   在 ckpt 上复算 PCC/SVG + JSD/SSIM（分布/结构指标）
├── plotting/                 # 绘图脚本
│   ├── plot_training_process.py
│   └── plot_attention_schematic.py
├── comparison/               # baseline 对比
│   ├── h5ad/                 #   各方法对齐后的预测 h5ad（X=pred, layers['gt']）
│   ├── metrics/              #   汇总指标 csv / json
│   ├── figures*/             #   可视化输出
│   ├── convert_*.py          #   各 baseline 原始预测 → 统一 h5ad
│   ├── visualize*.py         #   空间表达 / 分布对比图
│   ├── print_metrics_table.py#   打榜表（PCC / SVG）
│   ├── postprocess_driftst_coad.py  # DriftST 方差后处理（外部脚本版）
│   └── renorm_scellst_coad.py
├── logs/                     # 训练日志
└── archive/                  # 归档（训练快照等中间产物）
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

## 分布 / 结构指标

`eval/eval_dist_metrics.py` 在 best ckpt 上复算指标，除 PCC-10/50/200/all、SVG-20/50
外，额外给出抓「动态范围塌缩」与「空间结构」的两个指标：

- **JSD**：逐基因 预测值分布 vs 真实值分布 的 Jensen-Shannon 散度（越小越好）
- **SSIM**：把每基因表达按坐标栅格化成 2D 图，比预测图 vs 真实图的结构相似度（越大越好）

```bash
python eval/eval_dist_metrics.py --ckpt experiments/xxx/fold_0/best_model.pt --fold 0
```
