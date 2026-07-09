"""Per-gene affine variance calibration for DriftST predictions.

DriftST point predictions can be compressed toward the conditional mean. This
post-processing step rescales each predicted gene distribution to match
reference per-gene mean and standard deviation, typically computed on the
training split. It does not use test labels and preserves per-gene PCC because
the transform is monotonic.
"""
import numpy as np


def gene_stats(expr):
    """Return per-gene mean and standard deviation."""
    expr = np.asarray(expr, dtype=np.float64)
    return expr.mean(axis=0), expr.std(axis=0)


def variance_postprocess(pred, ref_mean, ref_std, alpha=1.2, clip_min=0.0):
    """Apply per-gene affine variance calibration.

    pred'[:, g] = ref_mean[g] + (pred[:, g] - pred_mean[g]) * (ref_std[g] / pred_std[g]) * alpha

    Parameters
    ----------
    pred      : (N, G) predictions in log1p space.
    ref_mean  : (G,) reference per-gene mean.
    ref_std   : (G,) reference per-gene standard deviation.
    alpha     : variance multiplier; 1.0 matches the reference variance.
    clip_min  : lower clipping bound; None disables clipping.
    """
    pred = np.asarray(pred, dtype=np.float64)
    ref_mean = np.asarray(ref_mean, dtype=np.float64)
    ref_std = np.asarray(ref_std, dtype=np.float64)

    pm = pred.mean(axis=0)
    ps = pred.std(axis=0)
    out = np.empty_like(pred)
    for g in range(pred.shape[1]):
        if ps[g] < 1e-8:
            out[:, g] = pred[:, g]
            continue
        out[:, g] = ref_mean[g] + (pred[:, g] - pm[g]) * (ref_std[g] / ps[g]) * alpha
    if clip_min is not None:
        out = np.clip(out, clip_min, None)
    return out.astype(np.float32)
