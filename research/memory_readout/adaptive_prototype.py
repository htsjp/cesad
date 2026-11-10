import torch
import torch.nn.functional as F

from .kernel_evidence import safe_topk, topk_kernel_evidence


def weighted_topk_prototype(query: torch.Tensor, keys: torch.Tensor, k: int, tau: float, uniform: bool = False):
    sim = query @ keys.T
    vals, idx = safe_topk(sim, k)
    selected = keys[idx]
    if uniform:
        weights = torch.full_like(vals, 1.0 / max(1, vals.shape[-1]))
    else:
        weights = torch.softmax(vals / float(tau), dim=-1)
    proto = F.normalize((weights.unsqueeze(-1) * selected).sum(dim=-2), dim=-1)
    score = (query * proto).sum(dim=-1)
    return score, proto, vals, idx, weights


def query_adaptive_prototype_scores(query, normal_keys, anomaly_keys, k: int, tau: float, tau_cls: float, uniform: bool = False):
    rn, pn, vals_n, idx_n, w_n = weighted_topk_prototype(query, normal_keys, k, tau, uniform=uniform)
    ra, pa, vals_a, idx_a, w_a = weighted_topk_prototype(query, anomaly_keys, k, tau, uniform=uniform)
    logit = (ra - rn) / float(tau_cls)
    return {
        "logit": logit,
        "prob": torch.sigmoid(logit),
        "rn": rn,
        "ra": ra,
        "pn": pn,
        "pa": pa,
        "vals_n": vals_n,
        "vals_a": vals_a,
        "idx_n": idx_n,
        "idx_a": idx_a,
        "w_n": w_n,
        "w_a": w_a,
    }


def prototype_plus_evidence(query, normal_keys, anomaly_keys, k, tau, tau_cls, gamma, stats):
    base = query_adaptive_prototype_scores(query, normal_keys, anomaly_keys, k, tau, tau_cls)
    en, _, _ = topk_kernel_evidence(query @ normal_keys.T, k, tau)
    ea, _, _ = topk_kernel_evidence(query @ anomaly_keys.T, k, tau)
    mean = float(stats.get("evidence_mean", 0.0))
    std = float(stats.get("evidence_std", 1.0))
    hn = base["rn"] + float(gamma) * ((en - mean) / max(std, 1e-6))
    ha = base["ra"] + float(gamma) * ((ea - mean) / max(std, 1e-6))
    logit = (ha - hn) / float(tau_cls)
    base.update({"logit": logit, "prob": torch.sigmoid(logit), "en": en, "ea": ea})
    return base
