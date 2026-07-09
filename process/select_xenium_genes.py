"""Extract valid panel genes from Xenium transcript parquet files."""

import argparse
import re
from pathlib import Path

import pandas as pd


CONTROL_PATTERNS = [
    r"^NegControlProbe_",
    r"^antisense_",
    r"^NegControlCodeword_",
    r"^BLANK_",
    r"^Blank-",
    r"NegPrb",
    r"Unassigned",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--transcripts_dir", type=str, required=True)
    p.add_argument("--out_file", type=str, required=True)
    return p.parse_args()


def is_control(name: str) -> bool:
    return any(re.search(pattern, name) for pattern in CONTROL_PATTERNS)


def main():
    args = parse_args()
    transcripts_dir = Path(args.transcripts_dir)
    out_file = Path(args.out_file)

    parquet_files = sorted(transcripts_dir.glob("*_transcripts.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No *_transcripts.parquet files found in {transcripts_dir}")

    print(f"Found {len(parquet_files)} transcript parquet files")
    df = pd.read_parquet(str(parquet_files[0]), columns=["feature_name"])
    features = df["feature_name"].dropna().unique().tolist()
    features = [x.decode() if isinstance(x, bytes) else str(x) for x in features]
    features = sorted(set(features))

    controls = [g for g in features if is_control(g)]
    selected = [g for g in features if not is_control(g)]

    print(f"Panel features: {len(features)}")
    print(f"Excluded controls: {len(controls)}")
    print(f"Selected genes: {len(selected)}")
    print("Top 20:", selected[:20])

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        for gene in selected:
            f.write(gene + "\n")
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()
