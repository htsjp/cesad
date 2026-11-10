from collections import Counter
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F


def effective_rank(x: torch.Tensor) -> float:
    x = x.detach().float()
    x = x - x.mean(dim=0, keepdim=True)
    if min(x.shape) < 2:
        return float("nan")
    s = torch.linalg.svdvals(x)
    p = s / s.sum().clamp_min(1e-12)
    return float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()).item())


def cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().float() - x.detach().float().mean(dim=0, keepdim=True)
    y = y.detach().float() - y.detach().float().mean(dim=0, keepdim=True)
    kxy = torch.linalg.norm(x.T @ y, ord="fro") ** 2
    kxx = torch.linalg.norm(x.T @ x, ord="fro")
    kyy = torch.linalg.norm(y.T @ y, ord="fro")
    return float((kxy / (kxx * kyy).clamp_min(1e-12)).item())


def view_geometry_rows(seed: int, split: str, views: Dict[str, torch.Tensor], max_rows: int = 4096) -> List[Dict]:
    rows = []
    sampled = {k: v.reshape(-1, v.shape[-1])[:max_rows].detach().float().cpu() for k, v in views.items()}
    for key, x in sampled.items():
        rows.append({
            "seed": seed,
            "split": split,
            "view": key,
            "effective_rank": effective_rank(x),
            "variance": float(x.var(dim=0, unbiased=False).mean().item()),
            "mean_norm": float(x.norm(dim=-1).mean().item()),
            "pairwise_view_cosine_to_T24": float((F.normalize(x, dim=-1) * F.normalize(sampled["T24"], dim=-1)).sum(dim=-1).mean().item()) if key != "T24" else 1.0,
            "cka_to_T24": cka(sampled["T24"], x) if key != "T24" else 1.0,
        })
    return rows


def retrieval_overlap_rows(seed: int, variant: str, sims: Dict[str, torch.Tensor]) -> List[Dict]:
    rows = []
    flat = {k: v.reshape(-1, v.shape[-1]).detach() for k, v in sims.items()}
    pairs = [("T24", "V12"), ("T24", "V18"), ("V12", "V18")]
    for a, b in pairs:
        ia = torch.argsort(flat[a], dim=-1, descending=True)
        ib = torch.argsort(flat[b], dim=-1, descending=True)
        top1 = (ia[:, 0] == ib[:, 0]).float().mean().item()
        row = {"seed": seed, "variant": variant, "view_pair": f"{a}-{b}", "top1_agreement": float(top1)}
        for k in (5, 20, 100):
            kk = min(k, ia.shape[-1])
            vals = []
            for x, y in zip(ia[:, :kk].cpu().numpy(), ib[:, :kk].cpu().numpy()):
                sx, sy = set(x.tolist()), set(y.tolist())
                vals.append(len(sx & sy) / max(1, len(sx | sy)))
            row[f"top{k}_jaccard"] = float(np.mean(vals)) if vals else float("nan")
        rows.append(row)
    return rows


def distribution_summary(probs: torch.Tensor, labels: torch.Tensor) -> Dict:
    flat = probs.reshape(-1, probs.shape[-1]).detach()
    entropy = -(flat * flat.clamp_min(1e-12).log()).sum(dim=-1)
    top = torch.argmax(flat, dim=-1).detach().cpu().numpy().tolist()
    counts = Counter(top)
    label_ids = labels.argmax(dim=-1)
    anomaly_mass = flat[:, label_ids == 1].sum(dim=-1) if (label_ids == 1).any() else torch.zeros(flat.shape[0], device=flat.device)
    return {
        "retrieval_entropy": float(entropy.mean().item()),
        "normalized_entropy": float((entropy / np.log(max(2, flat.shape[-1]))).mean().item()),
        "unique_prototypes_used": int(len(counts)),
        "never_top1_fraction": float(1.0 - len(counts) / max(1, flat.shape[-1])),
        "anomaly_mass_mean": float(anomaly_mass.mean().item()),
    }


def memory_organization(seed: int, memory: Dict, storage_mb: float) -> Dict:
    def duplicate_ratio(x):
        rounded = torch.round(x.detach().float().cpu() * 10000).to(torch.int32).numpy()
        unique = {row.tobytes() for row in rounded}
        return 1.0 - len(unique) / max(1, rounded.shape[0])

    return {
        "seed": seed,
        "memory_entries": int(memory["T24"].shape[0]),
        "total_multiview_vectors": int(memory["T24"].shape[0] * 3),
        "t24_duplicate_ratio": duplicate_ratio(memory["T24"]),
        "v12_duplicate_ratio": duplicate_ratio(memory["V12"]),
        "v18_duplicate_ratio": duplicate_ratio(memory["V18"]),
        "memory_storage_mb": float(storage_mb),
        "compression_active": False,
        "graph_active": False,
        "spr_active": False,
    }

