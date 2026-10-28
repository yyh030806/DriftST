"""Add validation partitions to existing outer splits without overwriting them."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.splits import add_validation_splits, validate_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--input", default=None, help="outer splits; defaults to data_dir/splits.json")
    parser.add_argument("--output", default=None, help="defaults to data_dir/splits_with_validation.json")
    parser.add_argument("--axis", choices=["x", "y"], default="y")
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true", help="allow replacing an existing output file")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    input_path = Path(args.input) if args.input else data_dir / "splits.json"
    output_path = Path(args.output) if args.output else data_dir / "splits_with_validation.json"
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output must be different files")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; choose another output or pass --overwrite")

    with open(input_path) as f:
        splits = json.load(f)
    obs = pd.read_csv(data_dir / "obs_metadata.csv", index_col=0, low_memory=False)
    result = add_validation_splits(splits, obs, axis=args.axis, fraction=args.fraction)
    with open(data_dir / "barcodes.json") as f:
        barcodes = json.load(f)
    for split in result:
        validate_split(split, barcodes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    for i, split in enumerate(result):
        print(f"fold {i}: train={len(split['train'])}, val={len(split['val'])}, "
              f"test={len(split['test'])}, method={split.get('validation_method', 'explicit')}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
