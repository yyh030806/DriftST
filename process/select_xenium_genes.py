"""
从 Xenium 转录本数据提取全部真实基因（对齐 GHIST）。
策略：用黑名单排除控制探针（NegControlProbe_, antisense_, NegControlCodeword_,
BLANK_, Blank-, NegPrb, Unassigned），保留所有 Gene Expression 类型基因。

与原 HVG 策略的区别：不做主动筛选，保留全部有效基因（约 280 个）。

用法：
  python select_xenium_genes.py   # 默认 janesick
  python select_xenium_genes.py \
      --transcripts_dir .../xenium_coad/transcripts \
      --out_file        .../xenium_coad/processed_data/selected_gene_list.txt
"""

import argparse
import pandas as pd
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--transcripts_dir", type=str,
               default="/data/buyonggan/DriftST/hest1k_datasets/xenium_janesick/transcripts")
p.add_argument("--out_file", type=str,
               default="/data/buyonggan/DriftST/hest1k_datasets/xenium_janesick/processed_data/selected_gene_list.txt")
args = p.parse_args()

TRANSCRIPTS_DIR = Path(args.transcripts_dir)
OUT_FILE        = Path(args.out_file)

CONTROL_PREFIXES = [
    "NegControlProbe_",
    "antisense_",
    "NegControlCodeword_",
    "BLANK_",
    "Blank-",
    "NegPrb",
    "Unassigned",
]

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# 只需读一个 slide 的 feature_name 即可（两个 slide 用同一面板）
parquet_files = sorted(TRANSCRIPTS_DIR.glob("*.parquet"))
print(f"找到 {len(parquet_files)} 个 transcripts parquet")

df = pd.read_parquet(str(parquet_files[0]), columns=["feature_name"])
df["feature_name"] = df["feature_name"].apply(
    lambda x: x.decode() if isinstance(x, bytes) else x
)

all_features = sorted(df["feature_name"].unique())
print(f"面板总 feature 数: {len(all_features)}")

def is_control(name: str) -> bool:
    return any(name.startswith(p) or p in name for p in CONTROL_PREFIXES)

selected = [g for g in all_features if not is_control(g)]
controls = [g for g in all_features if is_control(g)]

print(f"控制探针排除: {len(controls)} 个")
print(f"保留真实基因: {len(selected)} 个")
print("前 20:", selected[:20])

with open(OUT_FILE, "w") as f:
    f.write("\n".join(selected) + "\n")
print(f"已保存: {OUT_FILE}")
