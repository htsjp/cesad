import numpy as np
import torch
import torch.nn.functional as F


def reconstruction_diagnostics(name, original, reconstructed, rank=None, energy_retention=None):
    o = F.normalize(original.float(), dim=-1)
    r = F.normalize(reconstructed.float(), dim=-1)
    cos = (o * r).sum(dim=-1)
    diff = (o - r).abs()
    return {
        "variant": name,
        "rank": rank,
        "energy_retention": float(energy_retention) if energy_retention is not None else float("nan"),
        "mse": float((o - r).pow(2).mean().item()),
        "mean_cosine_to_fullfps": float(cos.mean().item()),
        "worst_cosine_to_fullfps": float(cos.min().item()),
        "max_abs_diff_to_fullfps": float(diff.max().item()),
        "near_duplicate_ratio_cos_gt_095": near_duplicate_ratio(r, 0.95),
        "near_duplicate_ratio_cos_gt_099": near_duplicate_ratio(r, 0.99),
    }


def near_duplicate_ratio(x, threshold):
    g = x.reshape(-1, 4, x.shape[-1])
    total = 0
    hits = 0
    for block in g:
        sim = block @ block.T
        for i in range(4):
            for j in range(i + 1, 4):
                total += 1
                hits += int(float(sim[i, j].item()) > threshold)
    return float(hits / max(1, total))


def metric_value(metrics, key):
    return float(metrics.get("mean", {}).get(key, float("nan")))

