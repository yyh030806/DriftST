# DriftST

DriftST is a one-step generative framework for inferring spatial transcriptomics
from H&E histology. It extracts pathology foundation model features from local
H&E patches, predicts spatial gene expression with an STransformer module, and
trains the predictor with a drifting objective that aligns stochastic prediction
distributions to matched transcriptomic profiles.

The code supports both cell-level Xenium data and spot-level HEST-style Visium
datasets. The current implementation uses UNI2 and CONCH features, a
co-expression attention bias, progressive gene residual gates, ZINB
reconstruction on raw counts.

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

Each entry in `splits.json` must explicitly contain non-overlapping `train`,
`val`, and `test` barcode lists. New preprocessing runs write this format.
Training computes co-expression statistics from `train`, evaluates `val` at
each epoch, and selects `best_model.pt` using the mean per-gene validation PCC.
The held-out `test` split is evaluated separately with `test.py` after the
training configuration and checkpoint have been chosen. Checkpoints record
split and processed-data fingerprints, and `test.py` checks both before
evaluating. Calculating the data fingerprint reads the processed input files
in full and can add startup time for large datasets.

The preprocessing scripts choose validation partitions deterministically:

- In leave-one-slide-out experiments, one of the remaining slides is reserved
  for validation. If only one training slide remains, a contiguous region of
  that slide is used instead (20% by default, along the y axis).
- In single-slide spatial cross-validation, the next spatial band is reserved
  for validation, and the outer test band remains unchanged. If that would
  leave too few training cells, a contiguous part of the training band is used.
- Neighbor features are restricted to the same partition as their target cell
  or spot. Coordinates, fold membership, and validation method are recorded in
  the split file.

Spatial bands can still have correlated samples and overlapping image patches
at their boundaries; no exclusion buffer is applied. A slide-level split does
not automatically separate patients. Use patient-level splits or an external
test cohort when those are required by the evaluation goal.

Existing processed datasets with only `train` and `test` can be migrated
without recomputing image features or overwriting the original split file:

```bash
python process/generate_validation_splits.py --data_dir /path/to/processed_data
# The result is /path/to/processed_data/splits_with_validation.json.
OUTPUT_DIR=/path/to/processed_data \
SPLITS_FILE=/path/to/processed_data/splits_with_validation.json \
  bash scripts/run_xenium_cell_level_train.sh --folds 0 --no-wandb
```

Use the same file with `train.py` and `test.py` via `--splits_file`. The
generator accepts `--axis x` or `--axis y` and `--fraction` for the
single-training-slide case; it preserves each original test partition. Inspect
the resulting validation regions to ensure they suit the biological question.
The preferred `SVG_SLIDE` defaults to `TENX94`; when that slide has too few
training cells in a fold, the script chooses another eligible training slide.
Training uses `--seed 42` by default; hardware and library differences can still
affect exact reproducibility.

## Xenium Cell-Level Workflow

Prepare the default Xenium breast cancer cell-level dataset using an existing
gene panel fixed independently of validation and test labels. The preprocessing
step reconstructs a cell-by-gene matrix from transcript tables, extracts
cell-centered H&E features, and writes the shared `processed_data` format:

```bash
GENE_LIST=/path/to/independent_gene_panel.txt bash scripts/run_xenium_cell_level_preprocess.sh
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
  --hidden_dim 128 \
  --num_layers 2 \
  --num_heads 4 \
  --dropout 0.3 \
  --use_gate
```

Validation and test metrics use predictions clipped to a minimum of zero in
ln(1+x) space. H5AD exports contain the same clipped predictions.

SVG rankings used during training and testing are fold-specific and calculated
only from training cells. The cell-level training script writes one ranking in
each fold's experiment directory. An existing ranking without matching training
split provenance is rejected; the old global `svg_ranking.json` is no longer
loaded automatically. Gene selection and any other data-dependent preprocessing
must likewise be based on training data or a panel fixed independently of the
test labels.

`test.py` restores model and neighbor settings from current checkpoints and
rejects conflicting overrides. Other checkpoints, such as `min_loss_model.pt`
and `last_model.pt`, record their own selection criteria; their use must also be
decided before inspecting test results.

Checkpoints created by older versions may have selected the best epoch using
test labels and do not carry split provenance. `test.py` rejects them by
default; `--allow_legacy_checkpoint` permits explicit inspection but cannot make
their test results unbiased. Retrain with explicit validation splits. If model
or hyperparameter decisions have already been influenced by a test set, an
independent test cohort is needed for an untouched final evaluation.

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
