"""模型评估：per-gene PCC、SVG-PCC、log2 空间 MSE/MAE。"""
import numpy as np
import torch
from scipy.stats import pearsonr


def metrics_from_arrays(all_pred, all_true, n_genes, svg_indices=None, verbose=True):
    """从 (pred, true) 数组直接算 12 个指标，与 evaluate() 内部口径一致。

    用于对后处理（方差校准）后的预测重新评估，无需再过模型。
    返回与 evaluate() 相同的 12 元组。
    """
    all_pred = np.asarray(all_pred)
    all_true = np.asarray(all_true)

    gene_pccs = []
    for i in range(n_genes):
        if all_pred[:, i].std() < 1e-8 or all_true[:, i].std() < 1e-8:
            gene_pccs.append(0.0)
            continue
        pcc, _ = pearsonr(all_pred[:, i], all_true[:, i])
        gene_pccs.append(float(pcc) if not np.isnan(pcc) else 0.0)

    pred_mean, pred_std = float(all_pred.mean()), float(all_pred.std())
    true_mean, true_std = float(all_true.mean()), float(all_true.std())

    pccs_sorted = sorted(gene_pccs, reverse=True)
    pcc_10  = np.mean(pccs_sorted[:10])
    pcc_50  = np.mean(pccs_sorted[:min(50,  len(pccs_sorted))])
    pcc_200 = np.mean(pccs_sorted[:min(200, len(pccs_sorted))])
    pcc_all = np.mean(gene_pccs)
    ln2 = np.log(2)
    mse_log2 = float(np.mean(((all_pred - all_true) / ln2) ** 2))
    mae_log2 = float(np.mean(np.abs((all_pred - all_true) / ln2)))

    svg_pcc20, svg_pcc50 = 0.0, 0.0
    if svg_indices is not None and len(svg_indices) > 0:
        svg_pccs = [gene_pccs[i] for i in svg_indices if i < n_genes]
        svg_pcc20 = float(np.mean(svg_pccs[:20]))
        svg_pcc50 = float(np.mean(svg_pccs[:min(50, len(svg_pccs))]))

    if verbose:
        print(f"PCC-10={pcc_10:.4f} | PCC-50={pcc_50:.4f} | "
              f"PCC-200={pcc_200:.4f} | PCC-all={pcc_all:.4f} | "
              f"SVG-20={svg_pcc20:.4f} | SVG-50={svg_pcc50:.4f}")
    return (pcc_all, pcc_10, pcc_50, pcc_200,
            pred_mean, pred_std, true_mean, true_std,
            mse_log2, mae_log2, svg_pcc20, svg_pcc50)


@torch.no_grad()
def evaluate(model, loader, n_genes, device, svg_indices=None, return_predictions=False):
    """
    svg_indices: list of gene indices sorted by Moran's I (descending).
                 If provided, reports PCC for top-20 and top-50 SVGs.
    return_predictions: 若 True，额外返回 (all_pred, all_true) numpy 数组。
    """
    model.eval()
    all_pred, all_true = [], []

    for batch in loader:
        img      = batch["z_img"].to(device)
        g_true   = batch["gene_expr"].to(device)
        nb_img   = batch["neighbor_zimg"].to(device)
        nb_valid = batch["neighbor_valid"].to(device)

        x0, _ = model(img, nb_img, nb_valid)
        all_pred.append(x0.cpu().numpy())
        all_true.append(g_true.cpu().numpy())

    all_pred = np.concatenate(all_pred, axis=0)
    all_true = np.concatenate(all_true, axis=0)

    print(f"pred mean={all_pred.mean():.4f}, std={all_pred.std():.4f}")
    print(f"true mean={all_true.mean():.4f}, std={all_true.std():.4f}")

    gene_pccs = []
    for i in range(n_genes):
        pcc, _ = pearsonr(all_pred[:, i], all_true[:, i])
        gene_pccs.append(float(pcc) if not np.isnan(pcc) else 0.0)

    pred_mean = float(all_pred.mean())
    pred_std  = float(all_pred.std())
    true_mean = float(all_true.mean())
    true_std  = float(all_true.std())

    if not any(p != 0.0 for p in gene_pccs):
        result = (0.0, 0.0, 0.0, 0.0, pred_mean, pred_std, true_mean, true_std, 0.0, 0.0, 0.0, 0.0)
        return (*result, all_pred, all_true) if return_predictions else result

    pccs_sorted = sorted(gene_pccs, reverse=True)
    pcc_10  = np.mean(pccs_sorted[:10])
    pcc_50  = np.mean(pccs_sorted[:min(50,  len(pccs_sorted))])
    pcc_200 = np.mean(pccs_sorted[:min(200, len(pccs_sorted))])
    pcc_all = np.mean(gene_pccs)
    ln2 = np.log(2)
    mse_log2 = float(np.mean(((all_pred - all_true) / ln2) ** 2))
    mae_log2 = float(np.mean(np.abs((all_pred - all_true) / ln2)))

    # SVG PCC（按 Moran's I 排序的 top-20 / top-50）
    svg_pcc20, svg_pcc50 = 0.0, 0.0
    if svg_indices is not None and len(svg_indices) > 0:
        svg_pccs = [gene_pccs[i] for i in svg_indices if i < n_genes]
        svg_pcc20 = float(np.mean(svg_pccs[:20]))
        svg_pcc50 = float(np.mean(svg_pccs[:min(50, len(svg_pccs))]))

    print(f"PCC-10={pcc_10:.4f} | PCC-50={pcc_50:.4f} | "
          f"PCC-200={pcc_200:.4f} | PCC-all={pcc_all:.4f} | "
          f"SVG-20={svg_pcc20:.4f} | SVG-50={svg_pcc50:.4f}")
    result = (pcc_all, pcc_10, pcc_50, pcc_200,
              pred_mean, pred_std, true_mean, true_std,
              mse_log2, mae_log2, svg_pcc20, svg_pcc50)
    return (*result, all_pred, all_true) if return_predictions else result
