import math

import torch
import torch.nn.functional as F


def safe_topk(sim: torch.Tensor, k: int):
    k_eff = min(int(k), sim.shape[-1])
    return torch.topk(sim, k=k_eff, dim=-1)


def class_similarities(query: torch.Tensor, normal_keys: torch.Tensor, anomaly_keys: torch.Tensor):
    return query @ normal_keys.T, query @ anomaly_keys.T


def full_kernel_evidence(sim: torch.Tensor, tau: float):
    return torch.logsumexp(sim / float(tau), dim=-1) - math.log(max(1, sim.shape[-1]))


def topk_kernel_evidence(sim: torch.Tensor, k: int, tau: float):
    vals, idx = safe_topk(sim, k)
    return torch.logsumexp(vals / float(tau), dim=-1) - math.log(max(1, vals.shape[-1])), vals, idx


def batch_normalize(x: torch.Tensor, mean: float, std: float):
    return (x - float(mean)) / max(float(std), 1e-6)


def sigmoid_logits_to_probs(logit: torch.Tensor):
    p = torch.sigmoid(logit)
    return torch.stack([1.0 - p, p], dim=-1)


def original_full_bank_voting(query, memory, tau: float):
    sim = query @ memory["patch_keys"].T
    weights = F.softmax(sim / float(tau), dim=-1)
    return weights @ memory["patch_values"].float(), sim, weights
