from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .prototype_builder import PrototypeBank


def two_class_score(q_proj: torch.Tensor, protos: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
    values = F.one_hot(torch.tensor([0, 1], device=q_proj.device), num_classes=2).float()
    logits = q_proj @ protos.to(q_proj.device).T
    weights = torch.softmax(logits, dim=-1)
    score = (weights @ values)[:, 1]
    return score, {"logit_normal": float(logits[:, 0].mean().item()), "logit_anomaly": float(logits[:, 1].mean().item())}


def batched_two_class_score(q_proj: torch.Tensor, normal_proto: torch.Tensor, anomaly_proto: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
    logit_n = (q_proj * normal_proto).sum(dim=-1, keepdim=True)
    logit_a = (q_proj * anomaly_proto).sum(dim=-1, keepdim=True)
    logits = torch.cat([logit_n, logit_a], dim=-1)
    weights = torch.softmax(logits, dim=-1)
    return weights[:, 1], {"logit_normal": float(logit_n.mean().item()), "logit_anomaly": float(logit_a.mean().item())}


def score_prototype_variant(variant: str, q_proj: torch.Tensor, bank: PrototypeBank) -> Tuple[torch.Tensor, Dict]:
    if variant == "D1_ClassCentroid":
        return two_class_score(q_proj, bank.global_centroids)

    if variant in {"D2_HQAP2", "D3_HQAP4"}:
        protos = []
        diagnostics: Dict = {}
        for cls in [0, 1]:
            subs = bank.subprototypes[cls].to(q_proj.device)
            logits = q_proj @ subs.T
            weights = torch.softmax(logits, dim=-1)
            local = weights @ subs
            protos.append(local)
            diagnostics[f"class_{cls}_max_sub_weight"] = float(weights.max(dim=-1).values.mean().item())
            diagnostics[f"class_{cls}_shift_norm"] = float((local - bank.global_centroids[cls].to(q_proj.device)).norm(dim=-1).mean().item())
        score, base = batched_two_class_score(q_proj, protos[0], protos[1])
        diagnostics.update(base)
        return score, diagnostics

    if variant == "D4_CoreBoundary":
        core_n = bank.core[0].to(q_proj.device)
        core_a = bank.core[1].to(q_proj.device)
        boundary_n = bank.boundary[0].to(q_proj.device)
        boundary_a = bank.boundary[1].to(q_proj.device)
        core_logits = torch.cat([q_proj @ core_n.T, q_proj @ core_a.T], dim=-1)
        gate = torch.exp(-torch.abs(core_logits[:, 0:1] - core_logits[:, 1:2])).clamp(0.0, 1.0)
        p_n = (1.0 - gate) * core_n + gate * boundary_n
        p_a = (1.0 - gate) * core_a + gate * boundary_a
        score, base = batched_two_class_score(q_proj, p_n, p_a)
        base["boundary_gate"] = float(gate.mean().item())
        return score, base

    raise ValueError(f"Unsupported prototype variant: {variant}")
