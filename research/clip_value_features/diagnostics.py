import math
from collections import Counter
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


def effective_rank(x: torch.Tensor) -> float:
    x = x.float()
    if x.ndim > 2:
        x = x.reshape(-1, x.shape[-1])
    x = x - x.mean(dim=0, keepdim=True)
    if x.shape[0] < 2:
        return float("nan")
    s = torch.linalg.svdvals(x)
    p = s / s.sum().clamp_min(1e-12)
    entropy = -(p * p.clamp_min(1e-12).log()).sum()
    return float(torch.exp(entropy).item())


def top_pc_energy(x: torch.Tensor) -> float:
    x = x.float()
    if x.ndim > 2:
        x = x.reshape(-1, x.shape[-1])
    x = x - x.mean(dim=0, keepdim=True)
    if x.shape[0] < 2:
        return float("nan")
    s = torch.linalg.svdvals(x)
    vals = s.square()
    return float((vals[0] / vals.sum().clamp_min(1e-12)).item())


def spatial_smoothness(patches: torch.Tensor) -> float:
    x = patches.float()
    if x.ndim == 3:
        x = x.reshape(-1, x.shape[-1])
    n = x.shape[0]
    side = int(math.sqrt(n))
    if side * side != n:
        return float("nan")
    grid = x.reshape(side, side, -1)
    diffs = []
    if side > 1:
        diffs.append(1.0 - (grid[:, 1:] * grid[:, :-1]).sum(dim=-1))
        diffs.append(1.0 - (grid[1:, :] * grid[:-1, :]).sum(dim=-1))
    return float(torch.cat([d.reshape(-1) for d in diffs]).mean().item()) if diffs else float("nan")


def representation_geometry(features: torch.Tensor, name: str) -> Dict:
    x = features.detach().float()
    flat = x.reshape(-1, x.shape[-1])
    sample = flat
    if sample.shape[0] > 2048:
        idx = torch.linspace(0, sample.shape[0] - 1, 2048, device=sample.device).long()
        sample = sample[idx]
    sim = sample @ sample.T
    mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    return {
        "variant": name,
        "mean_norm_before_normalization": float(flat.norm(dim=-1).mean().item()),
        "feature_variance": float(flat.var(dim=0, unbiased=False).mean().item()),
        "effective_rank": effective_rank(flat),
        "top_pc_energy": top_pc_energy(flat),
        "mean_pairwise_cosine": float(sim[mask].mean().item()) if sim.numel() > 1 else float("nan"),
        "patch_spatial_smoothness": spatial_smoothness(x[0] if x.ndim == 3 else x),
    }


def gini(counts: Counter, size: int) -> float:
    arr = np.zeros(size, dtype=np.float64)
    for idx, value in counts.items():
        if 0 <= idx < size:
            arr[idx] = value
    if arr.sum() == 0:
        return float("nan")
    arr.sort()
    n = arr.size
    return float((2 * np.arange(1, n + 1).dot(arr) / (n * arr.sum())) - (n + 1) / n)


def retrieval_summary(entropies, margins, top5_weights, top1_counter: Counter, top1_label_counter: Counter, memory_size: int) -> Dict:
    return {
        "retrieval_entropy": float(np.mean(entropies)) if entropies else float("nan"),
        "normalized_entropy": float(np.mean(entropies) / math.log(max(2, memory_size))) if entropies else float("nan"),
        "usage_gini": gini(top1_counter, memory_size),
        "never_top1_fraction": float((memory_size - len(top1_counter)) / max(1, memory_size)),
        "top1_margin": float(np.mean(margins)) if margins else float("nan"),
        "top5_cumulative_weight": float(np.mean(top5_weights)) if top5_weights else float("nan"),
        "unique_prototypes_used": int(len(top1_counter)),
        "normal_top1_usage": int(top1_label_counter.get(0, 0)),
        "anomaly_top1_usage": int(top1_label_counter.get(1, 0)),
    }
