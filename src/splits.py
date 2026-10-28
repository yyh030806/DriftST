"""Deterministic train/validation/test split construction and validation."""

import hashlib
import json
from pathlib import Path

import numpy as np


def barcode_fingerprint(barcodes):
    """Identify a barcode set independently of its order."""
    payload = json.dumps(sorted(barcodes), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def processed_data_fingerprint(data_dir):
    """Hash processed inputs in full, including array content and gene order."""
    data_dir = Path(data_dir)
    names = ("barcodes.json", "gene_names.json", "gene_expression.npy", "z_img_features.npy",
             "neighbor_map.json", "summary.json", "obs_metadata.csv")
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8") + b"\0")
        with open(data_dir / name, "rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def split_fingerprint(split):
    payload = {name: sorted(split[name]) for name in ("train", "val", "test")}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_split(split, all_barcodes):
    """Reject missing, duplicate, overlapping or too-small partitions."""
    available = set(all_barcodes)
    partitions = {}
    for name in ("train", "val", "test"):
        if name not in split:
            raise ValueError(
                f"split is missing '{name}'; generate a validation split with "
                "process/generate_validation_splits.py"
            )
        items = split[name]
        if not isinstance(items, list) or len(items) < 2:
            raise ValueError(f"split '{name}' must contain at least two barcodes")
        selected = set(items)
        if len(selected) != len(items):
            raise ValueError(f"split '{name}' contains duplicate barcodes")
        missing = selected - available
        if missing:
            raise ValueError(f"split '{name}' contains missing barcodes: {sorted(missing)[:5]}")
        for other_name, other in partitions.items():
            overlap = selected & other
            if overlap:
                raise ValueError(f"split overlap between '{other_name}' and '{name}': {sorted(overlap)[:5]}")
        partitions[name] = selected
    assigned = set().union(*partitions.values())
    unassigned = available - assigned
    if unassigned:
        raise ValueError(f"split leaves processed barcodes unassigned: {sorted(unassigned)[:5]}")


def _spatial_validation_band(train_obs, coord_col, fraction, total_train):
    if coord_col not in train_obs:
        raise ValueError(f"spatial validation requires '{coord_col}' in obs metadata")
    coords = np.asarray(train_obs[coord_col], dtype=float)
    if not np.isfinite(coords).all():
        raise ValueError("spatial validation requires finite coordinates")
    values, counts = np.unique(coords, return_counts=True)
    cumulative = np.cumsum(counts)
    eligible = np.flatnonzero((cumulative >= 2) & (total_train - cumulative >= 2))
    if not len(eligible):
        raise ValueError("spatial coordinates cannot form non-empty validation and training regions")
    cut = eligible[np.argmin(np.abs(cumulative[eligible] - len(train_obs) * fraction))]
    threshold = values[cut]
    return (set(train_obs.index[coords <= threshold]),
            {"axis": coord_col, "coord_lo": float(values[0]), "coord_hi": float(threshold)})


def add_validation_splits(splits, obs, *, axis="y", fraction=0.2):
    """Reserve validation data from each outer training set, preserving its test set.

    For spatial outer folds, the next outer band is the validation band. For
    slide-held-out folds, hold out a whole training slide if possible; otherwise
    hold out a contiguous region of the only training slide.
    """
    if axis not in ("x", "y") or not 0 < fraction < 1:
        raise ValueError("validation axis must be x or y and fraction must be between 0 and 1")
    if not obs.index.is_unique or "slide_id" not in obs.columns or obs["slide_id"].isna().any():
        raise ValueError("obs metadata must have unique barcodes and non-null slide_id values")
    available = set(obs.index)
    result = []
    for fold_id, original in enumerate(splits):
        split = dict(original)
        if "val" in split:
            validate_split(split, available)
            result.append(split)
            continue
        train = split.get("train")
        test = split.get("test")
        if not isinstance(train, list) or not isinstance(test, list):
            raise ValueError("each split must have train and test barcode lists")
        if len(set(train)) != len(train) or len(set(test)) != len(test):
            raise ValueError("outer split contains duplicate barcodes")
        if set(train) & set(test):
            raise ValueError("outer train/test overlap")
        missing = (set(train) | set(test)) - available
        if missing:
            raise ValueError(f"outer split contains missing barcodes: {sorted(missing)[:5]}")
        if len(train) < 4:
            raise ValueError("outer training set needs at least four barcodes")

        train_set = set(train)
        val_set = None
        if "spatial_info" in split:
            for offset in range(1, len(splits)):
                candidate = splits[(fold_id + offset) % len(splits)]
                band = set(candidate.get("test", []))
                if (candidate.get("test_slide") == split.get("test_slide")
                        and "spatial_info" in candidate and len(band) >= 2
                        and band <= train_set and len(train_set - band) >= 2):
                    val_set = band
                    split["validation_method"] = "spatial_outer_band"
                    split["val_slide"] = candidate.get("test_slide")
                    split["val_spatial_info"] = candidate["spatial_info"]
                    break
            if val_set is None:
                train_obs = obs.loc[train]
                band_obs = train_obs[train_obs["slide_id"] == split.get("test_slide")]
                coord_col = split["spatial_info"].get("axis", f"pixel_{axis}")
                val_set, spatial_info = _spatial_validation_band(band_obs, coord_col, fraction, len(train))
                split["validation_method"] = "spatial_training_band"
                split["val_slide"] = split.get("test_slide")
                split["val_spatial_info"] = spatial_info
        else:
            train_obs = obs.loc[train]
            slides = sorted(train_obs["slide_id"].unique().tolist())
            if len(slides) >= 2:
                slide_counts = train_obs["slide_id"].value_counts()
                eligible_slides = [s for s in slides if 2 <= slide_counts[s] <= len(train) - 2]
                if not eligible_slides:
                    raise ValueError("no training slide can provide at least two validation and training barcodes")
                val_slide = eligible_slides[fold_id % len(eligible_slides)]
                val_set = set(train_obs.index[train_obs["slide_id"] == val_slide])
                split["validation_method"] = "held_out_slide"
                split["val_slide"] = val_slide
            else:
                coord_col = f"pixel_{axis}"
                val_set, spatial_info = _spatial_validation_band(train_obs, coord_col, fraction, len(train))
                split["validation_method"] = "spatial_training_band"
                split["val_slide"] = slides[0]
                split["val_spatial_info"] = spatial_info

        split["train"] = [bc for bc in train if bc not in val_set]
        split["val"] = [bc for bc in train if bc in val_set]
        validate_split(split, available)
        result.append(split)
    return result
