from collections import Counter
from typing import Dict

import numpy as np
import torch


def gate_statistics(name: str, seed: int, gate, sample_features: Dict[str, torch.Tensor]) -> list:
    rows = []
    for layer in (12, 18):
        with torch.no_grad():
            w = gate(sample_features, layer).detach().float().reshape(-1, sample_features[f"V{layer}_heads"].shape[-2])
        entropy = -(w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12) * (w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12)).clamp_min(1e-12).log()).sum(dim=-1)
        mean_w = w.mean(dim=0)
        rows.append({
            "seed": seed,
            "variant": name,
            "layer": layer,
            "mean_head_weight": float(mean_w.mean().item()),
            "weight_std": float(w.std(unbiased=False).item()),
            "gate_entropy": float(entropy.mean().item()),
            "effective_active_heads": float(torch.exp(entropy).mean().item()),
            "top_weighted_heads": ",".join(map(str, torch.topk(mean_w, k=min(4, mean_w.numel())).indices.cpu().tolist())),
            "bottom_weighted_heads": ",".join(map(str, torch.topk(-mean_w, k=min(4, mean_w.numel())).indices.cpu().tolist())),
            "fraction_near_uniform": float((w.sub(1.0).abs() < 0.05).float().mean().item()),
            "fraction_head_collapse": float((w.max(dim=-1).values > (0.75 * w.shape[-1])).float().mean().item()),
        })
    return rows


def retrieval_summary(weights: torch.Tensor, sim: torch.Tensor, values: torch.Tensor) -> Dict:
    flat_w = weights.reshape(-1, weights.shape[-1]).detach()
    flat_s = sim.reshape(-1, sim.shape[-1]).detach()
    entropy = -(flat_w * flat_w.clamp_min(1e-12).log()).sum(dim=-1)
    vals, idx = torch.topk(flat_s, k=min(5, flat_s.shape[-1]), dim=-1)
    top1 = idx[:, 0].cpu().numpy().tolist()
    counts = Counter(top1)
    labels = values.argmax(dim=-1)
    anomaly_mass = flat_w[:, labels == 1].sum(dim=-1) if (labels == 1).any() else torch.zeros(flat_w.shape[0], device=flat_w.device)
    return {
        "retrieval_entropy": float(entropy.mean().item()),
        "normalized_entropy": float((entropy / np.log(max(2, flat_w.shape[-1]))).mean().item()),
        "unique_prototypes_used": int(len(counts)),
        "never_top1_fraction": float(1.0 - len(counts) / max(1, flat_w.shape[-1])),
        "top1_margin": float((vals[:, 0] - vals[:, 1]).mean().item()) if vals.shape[1] > 1 else float(vals[:, 0].mean().item()),
        "anomaly_memory_mass": float(anomaly_mass.mean().item()),
    }

