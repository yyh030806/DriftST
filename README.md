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
│   ├── preprocess_xenium_cell_level.py       # Xenium transcript-to-cell preprocessing
│   ├── select_xenium_cell_level_genes.py     # Xenium panel gene selection
│   ├── select_xenium_cell_level_hvg.py       # optional Xenium HVG selection
│   ├── generate_xenium_cell_level_splits.py  # cell-level spatial CV splits
│   ├── preprocess_spot_level.py              # HEST-style spot-level preprocessing
│   └── compute_spatial_variable_genes.py     # SVG ranking for evaluation
└── scripts/
    ├── run_xenium_cell_level_preprocess.sh   # default Xenium cell-level preprocessing
    ├── run_xenium_cell_level_train.sh        # Xenium breast cancer cell-level training
    ├── run_xenium_coad_cell_level_train.sh   # Xenium COAD cell-level training
    └── run_spot_level_train.sh               # HER2ST / PRAD / Kidney spot-level training
```

## Installation

The dependency versions in this repository are pinned from the development
`DriftST` conda environment (`Python 3.10.20`). You can create a matching
environment with:

```bash
conda env create -f environment.yml
conda activate DriftST
```

Alternatively, install the Python dependencies into an existing environment:

```bash
pip install -r requirements.txt
```

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
gene_expression.npy      # raw counts, shape (N cells/spots, G genes)
z_img_features.npy       # image features, shape (N cells/spots, D) or (N, A, D)
barcodes.json
gene_names.json
neighbor_map.json
splits.json
summary.json
obs_metadata.csv
```

`src.dataset` converts raw counts to `log1p(count)` for the drifting target and
keeps raw counts for the ZINB reconstruction loss.

## Xenium Cell-Level Workflow

Prepare the default Xenium breast cancer cell-level dataset. The preprocessing
step reconstructs a cell-by-gene matrix from transcript tables, extracts
cell-centered H&E features, and writes the shared `processed_data` format:

```bash
bash scripts/run_xenium_cell_level_preprocess.sh
```

For spatial 5-fold cross-validation on cells from a single slide:

```bash
python process/generate_xenium_cell_level_splits.py \
  --data_dir hest1k_datasets/xenium_janesick/processed_data \
  --slide TENX94
```

Train DriftST on cell-level Xenium data:

```bash
bash scripts/run_xenium_cell_level_train.sh --folds 0 --no-wandb
bash scripts/run_xenium_coad_cell_level_train.sh --folds 0 --no-wandb
```

## Spot-Level Workflow

For HER2ST, PRAD, or Kidney Visium:

```bash
bash scripts/run_spot_level_train.sh --dataset her2st --fold 0 --no-wandb
bash scripts/run_spot_level_train.sh --dataset kidney --fold 0 --skip-preprocess --no-wandb
```

Most script defaults can be overridden with environment variables, for example:

```bash
EPOCHS=100 BATCH_SIZE=256 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_xenium_cell_level_train.sh --folds 0 --no-wandb
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
