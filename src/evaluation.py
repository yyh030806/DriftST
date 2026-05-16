"""模型评估：per-gene PCC、SVG-PCC、log2 空间 MSE/MAE。"""
import numpy as np
import torch
from scipy.stats import pearsonr


@torch.no_grad()
def evaluate(model, loader, n_genes, device, svg_indices=None):
    """
    svg_indices: list of gene indices sorted by Moran's I (descending).
                 If provided, reports PCC for top-20 and top-50 SVGs.
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
        return 0.0, 0.0, 0.0, 0.0, pred_mean, pred_std, true_mean, true_std, 0.0, 0.0, 0.0, 0.0

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
    return (pcc_all, pcc_10, pcc_50, pcc_200,
            pred_mean, pred_std, true_mean, true_std,
            mse_log2, mae_log2, svg_pcc20, svg_pcc50)
