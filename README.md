# DriftST

DriftST is a one-step generative framework for inferring spatial transcriptomics
from H&E histology. It extracts pathology foundation model features from local
H&E patches, predicts spatial gene expression with an STransformer module, and
trains the predictor with a drifting objective that aligns stochastic prediction
distributions to matched transcriptomic profiles.

The code supports both cell-level Xenium data and spot-level HEST-style Visium
datasets. The current implementation uses UNI2 and CONCH features, a
co-expression attention bias, progressive gene residual gates, ZINB
reconstruction on raw counts, and optional per-gene variance calibration for
exported predictions.

## Repository Structure

```text
DriftST/
├── train.py                         # training entry point
├── test.py                          # checkpoint evaluation and h5ad export
├── src/
│   ├── model.py                     # DriftST / STransformer model
│   ├── drift_step.py                # one-step drifting target and bank
│   ├── dataset.py                   # processed_data dataset loader
│   ├── evaluation.py                # PCC, SVG-PCC, MSE, MAE metrics
│   └── postprocess.py               # per-gene variance calibration
├── process/
│   ├── preprocess.py                # HEST-style spot-level preprocessing
│   ├── preprocess_xenium.py         # Xenium transcript-to-cell preprocessing
│   ├── select_xenium_genes.py       # Xenium panel gene selection
│   ├── select_xenium_hvg.py         # optional Xenium HVG selection
│   ├── generate_xenium_5fold_splits.py
│   └── compute_svg.py
└── scripts/
    ├── run_preprocess.sh            # default Xenium preprocessing
    ├── run_xenium.sh                # Xenium breast cancer 5-fold training
    ├── run_xenium_coad.sh           # Xenium COAD 5-fold training
    └── run_experiment.sh            # HER2ST / PRAD / Kidney training
```

## Installation

Create a Python environment with PyTorch and install the package dependencies:

```bash
pip install -r requirements.txt
```

CONCH is not distributed through this repository. Install the CONCH package in
the same environment so that `conch.open_clip_custom` is importable before
running preprocessing.

The preprocessing scripts expect local UNI2 and CONCH checkpoints under:

```text
weights/
├── uni2/
└── conch/
```

You can also point to another location:

```bash
export DRIFTST_WEIGHTS_DIR=/path/to/weights
```

## Processed Data Format

Training consumes a `processed_data` directory with:

```text
gene_expression.npy      # raw counts, shape (N, G)
z_img_features.npy       # image features, shape (N, D) or (N, A, D)
barcodes.json
gene_names.json
neighbor_map.json
splits.json
summary.json
obs_metadata.csv
```

`src.dataset` converts raw counts to `log1p(count)` for the drifting target and
keeps raw counts for the ZINB reconstruction loss.

## Xenium Workflow

Prepare the default Xenium breast cancer dataset:

```bash
bash scripts/run_preprocess.sh
```

For spatial 5-fold cross-validation on a single slide:

```bash
python process/generate_xenium_5fold_splits.py \
  --data_dir hest1k_datasets/xenium_janesick/processed_data \
  --slide TENX94
```

Train DriftST:

```bash
bash scripts/run_xenium.sh --folds 0 --no-wandb
bash scripts/run_xenium_coad.sh --folds 0 --no-wandb
```

## Spot-Level Workflow

For HER2ST, PRAD, or Kidney Visium:

```bash
bash scripts/run_experiment.sh --dataset her2st --fold 0 --no-wandb
bash scripts/run_experiment.sh --dataset kidney --fold 0 --skip-preprocess --no-wandb
```

Most script defaults can be overridden with environment variables, for example:

```bash
EPOCHS=100 BATCH_SIZE=256 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_xenium.sh --folds 0 --no-wandb
```

## Evaluation and Export

Evaluate a checkpoint:

```bash
python test.py \
  --data_dir hest1k_datasets/xenium_janesick/processed_data \
  --fold 0 \
  --ckpt experiments/xenium_example/fold_0/best_model.pt \
  --hidden_dim 128 \
  --num_layers 2 \
  --num_heads 4 \
  --dropout 0.3 \
  --use_gate
```

Export predictions as AnnData:

```bash
python test.py \
  --data_dir hest1k_datasets/xenium_janesick/processed_data \
  --fold 0 \
  --ckpt experiments/xenium_example/fold_0/best_model.pt \
  --save_h5ad outputs/driftst_fold0.h5ad \
  --use_gate
```

By default, `test.py` applies per-gene affine variance calibration using
training-split statistics. Disable it with `--no-variance-postproc`.

## Citation

If you use this repository, please cite the DriftST preprint. A BibTeX entry
will be added after the paper metadata is finalized.

```bibtex
@article{yang2026driftst,
  title   = {DriftST: One-Step Generative Inference of Spatial Transcriptomics from H&E Histology},
  author  = {Yang, Yuhang and Bu, Yonggan and Zhou, Shengyuan and Luo, Yiming and Zhang, Kai},
  year    = {2026},
  note    = {Preprint}
}
```

## License

This repository is released for research use. See `LICENSE` for details.
