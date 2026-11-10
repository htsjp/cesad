from typing import Dict, List

import numpy as np
import torch


def _effective_rank(x: torch.Tensor) -> float:
    x = x.float()
    if x.shape[0] < 2:
        return float("nan")
    x = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x)
    p = s / s.sum().clamp_min(1e-12)
    return float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()).item())


def _cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float() - x.float().mean(dim=0, keepdim=True)
    y = y.float() - y.float().mean(dim=0, keepdim=True)
    return float(((torch.linalg.norm(x.T @ y, ord="fro") ** 2) / (torch.linalg.norm(x.T @ x, ord="fro") * torch.linalg.norm(y.T @ y, ord="fro")).clamp_min(1e-12)).item())


def head_statistics(features: List[Dict[str, torch.Tensor]], labels: List[int], categories: List[str]) -> List[Dict]:
    rows = []
    labels_t = torch.tensor(labels)
    cats = np.asarray(categories)
    for layer in (12, 18):
        x = torch.cat([f[f"V{layer}_heads"].reshape(-1, f[f"V{layer}_heads"].shape[-2], f[f"V{layer}_heads"].shape[-1]).detach().cpu() for f in features], dim=0)
        heads = x.shape[1]
        per_head_flat = [x[:, h, :] for h in range(heads)]
        cka_mat = torch.zeros(heads, heads)
        for h in range(heads):
            for j in range(heads):
                cka_mat[h, j] = _cka(per_head_flat[h], per_head_flat[j])
        for h in range(heads):
            vals = per_head_flat[h]
            cat_means = []
            for cat in sorted(set(categories)):
                mask = torch.tensor(cats == cat)
                if mask.any():
                    cat_means.append(float(vals[mask].var(dim=0, unbiased=False).mean().item()))
            cv = float(np.std(cat_means) / (np.mean(cat_means) + 1e-12)) if cat_means else float("nan")
            normal = vals[labels_t == 0] if (labels_t == 0).any() else vals[:0]
            anomaly = vals[labels_t == 1] if (labels_t == 1).any() else vals[:0]
            sep = float((normal.mean(dim=0) - anomaly.mean(dim=0)).norm().item()) if normal.numel() and anomaly.numel() else float("nan")
            rows.append({
                "layer": layer,
                "head": h,
                "spatial_variance": float(vals.var(dim=0, unbiased=False).mean().item()),
                "channel_variance": float(vals.var(dim=-1, unbiased=False).mean().item()),
                "effective_rank": _effective_rank(vals),
                "source_category_stability": -cv,
                "mean_head_cka_redundancy": float((cka_mat[h].sum() - 1.0) / max(1, heads - 1)),
                "source_anomaly_normal_separation": sep,
            })
    return rows


def heterogeneity_weights(rows: List[Dict], tau: float = 1.0) -> Dict[int, torch.Tensor]:
    out = {}
    for layer in (12, 18):
        group = [r for r in rows if int(r["layer"]) == layer]
        var = torch.tensor([r["spatial_variance"] for r in group]).float().log()
        uniq = 1.0 - torch.tensor([r["mean_head_cka_redundancy"] for r in group]).float()
        stab = torch.tensor([r["source_category_stability"] for r in group]).float()
        def z(x):
            return (x - x.mean()) / x.std(unbiased=False).clamp_min(1e-6)
        score = z(var) + z(uniq) + z(stab)
        out[layer] = torch.softmax(score / float(tau), dim=0) * len(group)
    return out

