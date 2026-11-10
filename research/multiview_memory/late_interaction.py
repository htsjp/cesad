from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _weights(spec: Dict) -> Tuple[float, float, float]:
    return tuple(float(x) for x in spec.get("weights", [1.0, 0.5, 0.5]))


def _temperatures(spec: Dict) -> Tuple[float, float, float]:
    return tuple(float(x) for x in spec.get("temperatures", [0.07, 0.07, 0.07]))


def early_concat(query_views: Dict[str, torch.Tensor], memory: Dict[str, torch.Tensor], weights) -> torch.Tensor:
    w_t, w_12, w_18 = weights
    q = F.normalize(torch.cat([w_t * query_views["T24"], w_12 * query_views["V12"], w_18 * query_views["V18"]], dim=-1), dim=-1)
    m = F.normalize(torch.cat([w_t * memory["T24"], w_12 * memory["V12"], w_18 * memory["V18"]], dim=-1), dim=-1)
    return q @ m.T


def early_fusion_matched(query_views: Dict[str, torch.Tensor], memory: Dict[str, torch.Tensor], weights) -> torch.Tensor:
    w_t, w_12, w_18 = weights
    q = F.normalize(w_t * query_views["T24"] + w_12 * query_views["V12"] + w_18 * query_views["V18"], dim=-1)
    m = F.normalize(w_t * memory["T24"] + w_12 * memory["V12"] + w_18 * memory["V18"], dim=-1)
    return q @ m.T


def value_mean(query_views: Dict[str, torch.Tensor], memory: Dict[str, torch.Tensor]) -> torch.Tensor:
    q = F.normalize(query_views["V12"] + query_views["V18"], dim=-1)
    m = F.normalize(memory["V12"] + memory["V18"], dim=-1)
    return q @ m.T


def rrf_distribution(sims: Dict[str, torch.Tensor], weights, c: float) -> torch.Tensor:
    scores = None
    for key, w in zip(("T24", "V12", "V18"), weights):
        order = torch.argsort(sims[key], dim=-1, descending=True)
        ranks = torch.empty_like(order, dtype=sims[key].dtype)
        ranks.scatter_(-1, order, torch.arange(order.shape[-1], device=order.device, dtype=sims[key].dtype).expand_as(order) + 1.0)
        part = float(w) / (float(c) + ranks)
        scores = part if scores is None else scores + part
    return scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def distribution_from_sims(sims: Dict[str, torch.Tensor], spec: Dict) -> torch.Tensor:
    family = spec["family"]
    weights = _weights(spec)
    taus = _temperatures(spec)
    if family == "aligned_logit":
        z = 0.0
        for key, w, tau in zip(("T24", "V12", "V18"), weights, taus):
            z = z + float(w) * sims[key] / float(tau)
        return F.softmax(z, dim=-1)
    if family == "mixture":
        norm = sum(weights)
        p = 0.0
        for key, w, tau in zip(("T24", "V12", "V18"), weights, taus):
            p = p + (float(w) / norm) * F.softmax(sims[key] / float(tau), dim=-1)
        return p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    if family == "product":
        logp = 0.0
        for key, w, tau in zip(("T24", "V12", "V18"), weights, taus):
            p = F.softmax(sims[key] / float(tau), dim=-1)
            logp = logp + float(w) * (p.clamp_min(float(spec.get("epsilon", 1e-12))).log())
        return F.softmax(logp, dim=-1)
    if family == "rrf":
        return rrf_distribution(sims, weights, float(spec.get("c", 60)))
    raise ValueError(f"Unsupported late-interaction family: {family}")


def logits_from_distribution(probs: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return probs @ values.float()

