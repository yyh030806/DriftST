"""DriftST 预测的方差后处理（逐基因仿射校准）。

动机：DriftST 的点预测会向均值压缩（per-gene 方差通常只有真值的 ~25%），
点精度（PCC）不受影响，但分布层面（JSD / 动态范围）变差。此后处理把每个
基因的预测分布重新拉伸到参考分布的 per-gene 均值/标准差，可乘 alpha 略微过冲。

关键性质：
- 仅使用「参考」per-gene 统计量（默认取训练集），**不使用测试集真值**，可部署。
- 仿射变换单调，**不改变 per-gene PCC**；主要改善 JSD 与动态范围。
- alpha=1.0 即把方差对齐到参考；alpha>1 更大胆（过冲）。
"""
import numpy as np


def gene_stats(expr):
    """逐基因 (mean, std)。expr: (N, G) -> (G,), (G,)。"""
    expr = np.asarray(expr, dtype=np.float64)
    return expr.mean(axis=0), expr.std(axis=0)


def variance_postprocess(pred, ref_mean, ref_std, alpha=1.2, clip_min=0.0):
    """逐基因仿射校准。

    pred'[:, g] = ref_mean[g] + (pred[:, g] - pred_mean[g]) * (ref_std[g] / pred_std[g]) * alpha

    参数
    ----
    pred      : (N, G) 预测（log1p 空间）。
    ref_mean  : (G,) 参考 per-gene 均值（默认训练集）。
    ref_std   : (G,) 参考 per-gene 标准差。
    alpha     : 方差放大系数，1.0=对齐参考方差，>1 过冲。
    clip_min  : 下截断（log1p 空间应为 0）；None 表示不截断。
    """
    pred = np.asarray(pred, dtype=np.float64)
    ref_mean = np.asarray(ref_mean, dtype=np.float64)
    ref_std = np.asarray(ref_std, dtype=np.float64)

    pm = pred.mean(axis=0)
    ps = pred.std(axis=0)
    out = np.empty_like(pred)
    for g in range(pred.shape[1]):
        if ps[g] < 1e-8:                       # 常量预测，保持原样
            out[:, g] = pred[:, g]
            continue
        out[:, g] = ref_mean[g] + (pred[:, g] - pm[g]) * (ref_std[g] / ps[g]) * alpha
    if clip_min is not None:
        out = np.clip(out, clip_min, None)
    return out.astype(np.float32)
