import math
from collections import Counter

import numpy as np
import torch


def weight_rows(name, weights):
    rows = []
    for layer, w in weights.items():
        p = w.float() / w.float().sum().clamp_min(1e-12)
        entropy = float(-(p * p.clamp_min(1e-12).log()).sum().item())
        rows.append({
            "variant": name,
            "layer": layer,
            "head_weight_mean": float(w.float().mean().item()),
            "head_weight_std": float(w.float().std(unbiased=False).item()),
            "entropy": entropy,
            "effective_head_count": float(math.exp(entropy)),
            "top_heads": ",".join(map(str, torch.topk(w.float(), k=min(4, w.numel())).indices.tolist())),
            "bottom_heads": ",".join(map(str, torch.topk(-w.float(), k=min(4, w.numel())).indices.tolist())),
            "layer_total_mass": float(w.float().sum().item()),
        })
    return rows


def retrieval_summary(weights, sim, values):
    flat_w = weights.reshape(-1, weights.shape[-1]).detach()
    flat_s = sim.reshape(-1, sim.shape[-1]).detach()
    entropy = -(flat_w * flat_w.clamp_min(1e-12).log()).sum(dim=-1)
    vals, idx = torch.topk(flat_s, k=min(5, flat_s.shape[-1]), dim=-1)
    counts = Counter(idx[:, 0].detach().cpu().numpy().tolist())
    labels = values.argmax(dim=-1)
    return {
        "retrieval_entropy": float(entropy.mean().item()),
        "normalized_entropy": float((entropy / np.log(max(2, flat_w.shape[-1]))).mean().item()),
        "top1_margin": float((vals[:, 0] - vals[:, 1]).mean().item()) if vals.shape[1] > 1 else float(vals[:, 0].mean().item()),
        "unique_prototypes_used": int(len(counts)),
        "anomaly_memory_mass": float(flat_w[:, labels == 1].sum(dim=-1).mean().item()) if (labels == 1).any() else 0.0,
        "normal_memory_mass": float(flat_w[:, labels == 0].sum(dim=-1).mean().item()) if (labels == 0).any() else 0.0,
    }

