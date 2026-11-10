from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F


@dataclass
class PrototypeBank:
    variant: str
    labels: torch.Tensor
    global_centroids: torch.Tensor
    subprototypes: Dict[int, torch.Tensor]
    core: Dict[int, torch.Tensor]
    boundary: Dict[int, torch.Tensor]
    prototype_count: int
    stats: List[Dict]


def deterministic_kmeans(x: torch.Tensor, k: int, steps: int = 30) -> torch.Tensor:
    x = x.detach().float()
    if x.shape[0] <= k:
        return x.clone()
    order = torch.argsort(x[:, 0])
    seeds = torch.linspace(0, x.shape[0] - 1, steps=k).long()
    centers = x[order[seeds]].clone()
    for _ in range(steps):
        dist = torch.cdist(x, centers)
        assign = dist.argmin(dim=1)
        new_centers = []
        for i in range(k):
            idx = torch.where(assign == i)[0]
            new_centers.append(x[idx].mean(dim=0) if idx.numel() else centers[i])
        new_centers = torch.stack(new_centers, dim=0)
        if torch.allclose(new_centers, centers, atol=1e-6):
            break
        centers = new_centers
    return centers


def effective_rank(x: torch.Tensor) -> float:
    centered = x.float() - x.float().mean(dim=0, keepdim=True)
    try:
        s = torch.linalg.svdvals(centered)
        if float(s.sum()) <= 0:
            return 0.0
        p = (s / s.sum()).clamp_min(1e-12)
        return float(torch.exp(-(p * p.log()).sum()).item())
    except Exception:
        return float("nan")


def pairwise_mean(x: torch.Tensor) -> float:
    if x.shape[0] < 2:
        return float("nan")
    sim = F.normalize(x, dim=-1) @ F.normalize(x, dim=-1).T
    off = sim[~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)]
    return float(off.mean().item()) if off.numel() else float("nan")


def build_projected_bank(variant: str, projected_keys: torch.Tensor, values: torch.Tensor) -> PrototypeBank:
    labels = values.argmax(dim=-1).long().cpu()
    keys = projected_keys.detach().float().cpu()
    global_centroids = []
    stats: List[Dict] = []
    subprototypes: Dict[int, torch.Tensor] = {}
    core: Dict[int, torch.Tensor] = {}
    boundary: Dict[int, torch.Tensor] = {}

    for cls in [0, 1]:
        idx = torch.where(labels == cls)[0]
        cls_keys = keys[idx]
        centroid = cls_keys.mean(dim=0)
        global_centroids.append(centroid)
    global_centroids_t = torch.stack(global_centroids, dim=0)

    if variant in {"D2_HQAP2", "D3_HQAP4"}:
        k = 2 if variant == "D2_HQAP2" else 4
        for cls in [0, 1]:
            idx = torch.where(labels == cls)[0]
            cls_keys = keys[idx]
            centers = deterministic_kmeans(cls_keys, k)
            subprototypes[cls] = centers
            dist = torch.cdist(cls_keys, centers)
            assign = dist.argmin(dim=1)
            for cid in range(centers.shape[0]):
                member = cls_keys[torch.where(assign == cid)[0]]
                stats.append({
                    "variant": variant,
                    "class_label": cls,
                    "prototype_kind": "subprototype",
                    "prototype_id": cid,
                    "cluster_size": int(member.shape[0]),
                    "cluster_radius": float(torch.cdist(member, centers[cid:cid + 1]).mean().item()) if member.numel() else 0.0,
                    "centroid_pairwise_similarity": pairwise_mean(centers),
                    "effective_rank": effective_rank(cls_keys),
                    "source_category_coverage": "official_cache_binary_only",
                })
        count = 2 * k
    elif variant == "D4_CoreBoundary":
        norm_global = F.normalize(global_centroids_t, dim=-1)
        for cls in [0, 1]:
            idx = torch.where(labels == cls)[0]
            cls_keys = keys[idx]
            sim_self = (F.normalize(cls_keys, dim=-1) @ norm_global[cls])
            sim_other = (F.normalize(cls_keys, dim=-1) @ norm_global[1 - cls])
            margin = sim_self - sim_other
            order = torch.argsort(margin, descending=True)
            core_n = max(1, int(round(0.50 * cls_keys.shape[0])))
            boundary_n = max(1, int(round(0.25 * cls_keys.shape[0])))
            core[cls] = cls_keys[order[:core_n]].mean(dim=0, keepdim=True)
            boundary[cls] = cls_keys[order[-boundary_n:]].mean(dim=0, keepdim=True)
            stats.append({
                "variant": variant,
                "class_label": cls,
                "prototype_kind": "core_boundary",
                "prototype_id": cls,
                "cluster_size": int(cls_keys.shape[0]),
                "core_size": core_n,
                "boundary_size": boundary_n,
                "margin_mean": float(margin.mean().item()),
                "margin_bottom25_mean": float(margin[order[-boundary_n:]].mean().item()),
                "effective_rank": effective_rank(cls_keys),
                "source_category_coverage": "official_cache_binary_only",
            })
        count = 4
    else:
        count = 2
        for cls in [0, 1]:
            idx = torch.where(labels == cls)[0]
            cls_keys = keys[idx]
            stats.append({
                "variant": variant,
                "class_label": cls,
                "prototype_kind": "global_centroid",
                "prototype_id": cls,
                "cluster_size": int(cls_keys.shape[0]),
                "cluster_radius": float(torch.cdist(cls_keys, global_centroids_t[cls:cls + 1]).mean().item()),
                "centroid_pairwise_similarity": pairwise_mean(global_centroids_t),
                "effective_rank": effective_rank(cls_keys),
                "source_category_coverage": "official_cache_binary_only",
            })

    return PrototypeBank(
        variant=variant,
        labels=labels,
        global_centroids=global_centroids_t,
        subprototypes=subprototypes,
        core=core,
        boundary=boundary,
        prototype_count=count,
        stats=stats,
    )
