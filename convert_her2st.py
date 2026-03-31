"""
convert_her2st.py
=================
Convert HER2ST dataset to 10x Visium-compatible format.

Input structure:
    data/HER2ST/
    ├── count-matrices/   A1.tsv, A2.tsv, ...
    ├── images/HE/        A1.jpg, A2.jpg, ...
    ├── spot-selections/  A1_selection.tsv, ...
    └── meta/             A1_labeled_coordinates.tsv, ...  (optional)

Output structure:
    data/HER2ST/visium_format/
    └── A1/
        ├── filtered_feature_bc_matrix.h5
        └── spatial/
            ├── tissue_positions_list.csv
            ├── tissue_hires_image.png
            ├── tissue_lowres_image.png
            └── scalefactors_json.json

Usage:
    python convert_her2st.py

Spot selection columns (confirmed):
    x  y  new_x  new_y  pixel_x  pixel_y  selected
"""

import os
import json
import shutil
import numpy as np
import pandas as pd
import scipy.sparse as sp
import h5py
from PIL import Image
from pathlib import Path

# ─────────────────────────────────────────────
# Paths  (edit if needed)
# ─────────────────────────────────────────────
RAW_DIR   = Path("data/HER2ST")
OUT_DIR   = Path("data/HER2ST/visium_format")
CNTS_DIR  = RAW_DIR / "count-matrices"
IMGS_DIR  = RAW_DIR / "images" / "HE"
SPOTS_DIR = RAW_DIR / "spot-selections"
META_DIR  = RAW_DIR / "meta"          # optional, only A1/B1/.../H1 exist


def get_samples():
    """Get all sample names from count-matrices directory."""
    return sorted([f.stem for f in CNTS_DIR.glob("*.tsv")])


def convert_sample(sample: str):
    out = OUT_DIR / sample
    (out / "spatial").mkdir(parents=True, exist_ok=True)

    # ── 1. Read count matrix ─────────────────────────────────────────
    cnt_path = CNTS_DIR / f"{sample}.tsv"
    cnt = pd.read_csv(cnt_path, sep="\t", index_col=0)
    print(f"[{sample}] raw: {cnt.shape[0]} spots x {cnt.shape[1]} genes")

    # ── 2. Read spot selection file ──────────────────────────────────
    sel_path = SPOTS_DIR / f"{sample}_selection.tsv"
    sel_df = pd.read_csv(sel_path, sep="\t")

    # Count matrix index format: "row x col" e.g. "22x12"
    # spot-selection: x=col, y=row  ->  barcode = f"{y}x{x}"
    sel_df["barcode"] = sel_df["x"].astype(int).astype(str) + "x" + \
                        sel_df["y"].astype(int).astype(str)

    # Keep only selected spots (selected == 1)
    sel_df = sel_df[sel_df["selected"] == 1].reset_index(drop=True)
    print(f"[{sample}] selected spots: {len(sel_df)}")

    # ── 3. Intersect count matrix with selected spots ────────────────
    common = cnt.index.intersection(sel_df["barcode"])



    if len(common) == 0:
        print(f"[{sample}] WARNING: No matching barcodes!")
        print(f"  cnt index[:3]:     {cnt.index[:3].tolist()}")
        print(f"  sel barcode[:3]:   {sel_df['barcode'][:3].tolist()}")
        print(f"[{sample}] Skipping.\n")
        return

    cnt    = cnt.loc[common]
    sel_df = sel_df[sel_df["barcode"].isin(common)].reset_index(drop=True)
    print(f"[{sample}] after intersection: {len(cnt)} spots")

    # ── 4. Generate tissue_positions_list.csv ────────────────────────
    pos = pd.DataFrame({
        "barcode":            sel_df["barcode"],
        "in_tissue":          1,
        "array_row":          sel_df["y"].astype(int),
        "array_col":          sel_df["x"].astype(int),
        "pxl_col_in_fullres": sel_df["pixel_x"].astype(float),
        "pxl_row_in_fullres": sel_df["pixel_y"].astype(float),
    })
    pos.to_csv(out / "spatial" / "tissue_positions_list.csv", index=False)

    # ── 5. Generate filtered_feature_bc_matrix.h5 ───────────────────
    mat      = sp.csc_matrix(cnt.values.T.astype(np.float32))  # genes x spots
    genes    = cnt.columns.tolist()
    barcodes = cnt.index.tolist()

    with h5py.File(out / "filtered_feature_bc_matrix.h5", "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data",    data=mat.data.astype(np.float32))
        g.create_dataset("indices", data=mat.indices)
        g.create_dataset("indptr",  data=mat.indptr)
        g.create_dataset("shape",   data=np.array(mat.shape, dtype=np.int32))
        g.create_dataset("barcodes",data=np.array(barcodes, dtype="S"))
        ft = g.create_group("features")
        ft.create_dataset("id",           data=np.array(genes, dtype="S"))
        ft.create_dataset("name",         data=np.array(genes, dtype="S"))
        ft.create_dataset("feature_type", data=np.array(
            ["Gene Expression"] * len(genes), dtype="S"))
        ft.create_dataset("genome", data=np.array(["Unknown"] * len(genes), dtype="S"))
        # Required by scanpy read_visium
        g.attrs["library_ids"] = np.array([sample.encode("utf-8")])

    # ── 6. Convert image jpg -> png ──────────────────────────────────
    img_path = IMGS_DIR / f"{sample}.jpg"
    if not img_path.exists():
        img_path = IMGS_DIR / f"{sample}.JPG"

    if not img_path.exists():
        print(f"[{sample}] WARNING: image not found, skipping image conversion.")
    else:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # hires: original resolution
        img.save(out / "spatial" / "tissue_hires_image.png")

        # lowres: thumbnail (max 600px)
        img_low = img.copy()
        img_low.thumbnail((600, 600), Image.LANCZOS)
        img_low.save(out / "spatial" / "tissue_lowres_image.png")

        # scalefactors_json.json (required by scanpy)
        scale_lowres = 600 / max(w, h)
        with open(out / "spatial" / "scalefactors_json.json", "w") as f:
            json.dump({
                "spot_diameter_fullres":     200.0,
                "tissue_hires_scalef":       1.0,
                "tissue_lowres_scalef":      round(scale_lowres, 6),
                "fiducial_diameter_fullres": 200.0,
            }, f, indent=2)

    # ── 7. Copy meta label file if exists ────────────────────────────
    meta_path = META_DIR / f"{sample}_labeled_coordinates.tsv"
    if meta_path.exists():
        shutil.copy(meta_path, out / "spatial" / "labeled_coordinates.tsv")
        print(f"[{sample}] copied meta label file")

    print(f"[{sample}] ✓  spots={len(barcodes)}, genes={len(genes)}")
    print(f"[{sample}] -> {out}\n")


def verify_sample(sample: str):
    """Quick sanity check using scanpy."""
    try:
        import scanpy as sc
        adata = sc.read_visium(str(OUT_DIR / sample))
        print(f"[{sample}] scanpy verify OK: {adata}")
        print(f"  obs keys:  {list(adata.obs.columns)}")
        print(f"  obsm keys: {list(adata.obsm.keys())}")
    except Exception as e:
        print(f"[{sample}] scanpy verify FAILED: {e}")


if __name__ == "__main__":
    samples = get_samples()
    print(f"Found {len(samples)} samples: {samples}\n")

    for s in samples:
        try:
            convert_sample(s)
        except Exception as e:
            print(f"[{s}] ERROR: {e}\n")
            import traceback
            traceback.print_exc()

    print("=" * 50)
    print("Conversion complete. Verifying A1 with scanpy...")
    verify_sample("A1")