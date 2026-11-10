import torch
import torch.nn.functional as F


def initial_read_context(q, memory_keys, memory_values, eps=1e-12):
    """Return patch-specific [normal, anomaly, contrast] contexts from exact full-bank weights."""
    q = F.normalize(q.float(), dim=-1)
    keys = F.normalize(memory_keys.float(), dim=-1)
    labels = memory_values.argmax(dim=-1).long().to(keys.device)
    sim = q @ keys.T
    weights = sim.softmax(dim=-1)
    normal_mask = (labels == 0).float()[None, :]
    anomaly_mask = (labels == 1).float()[None, :]
    s_n = (weights * normal_mask).sum(dim=-1, keepdim=True)
    s_a = (weights * anomaly_mask).sum(dim=-1, keepdim=True)
    c_n = (weights * normal_mask) @ keys / s_n.clamp_min(eps)
    c_a = (weights * anomaly_mask) @ keys / s_a.clamp_min(eps)
    c_n = F.normalize(c_n, dim=-1)
    c_a = F.normalize(c_a, dim=-1)
    c_d = F.normalize(c_a - c_n, dim=-1)
    context = torch.stack([c_n, c_a, c_d], dim=1)
    vals, idx = torch.topk(sim, k=min(2, sim.shape[-1]), dim=-1)
    entropy = -(weights * weights.clamp_min(eps).log()).sum(dim=-1)
    normalized_entropy = entropy / torch.log(torch.tensor(float(weights.shape[-1]), device=weights.device))
    margin = vals[:, 0] - vals[:, 1] if vals.shape[1] > 1 else torch.zeros_like(entropy)
    stats = torch.stack(
        [
            normalized_entropy,
            margin,
            s_n.squeeze(-1),
            s_a.squeeze(-1),
            (s_a - s_n).squeeze(-1),
            (q * c_n).sum(dim=-1),
            (q * c_a).sum(dim=-1),
            (c_n * c_a).sum(dim=-1),
        ],
        dim=-1,
    )
    diag = {
        "normalized_entropy": float(normalized_entropy.mean().detach().cpu()),
        "top1_margin": float(margin.mean().detach().cpu()),
        "normal_mass": float(s_n.mean().detach().cpu()),
        "anomaly_mass": float(s_a.mean().detach().cpu()),
        "context_cosine_na": float((c_n * c_a).sum(dim=-1).mean().detach().cpu()),
    }
    return context, stats, diag


def select_context_tokens(context, mode):
    if mode == "normal":
        return context[:, 0:1, :]
    if mode == "anomaly":
        return context[:, 1:2, :]
    if mode == "two":
        return context[:, :2, :]
    if mode == "contrast":
        return context[:, 2:3, :]
    if mode == "label_blind":
        blind = F.normalize((context[:, 0, :] + context[:, 1, :]) * 0.5, dim=-1)
        return torch.stack([blind, blind, torch.zeros_like(blind)], dim=1)
    if mode == "zero":
        return torch.zeros_like(context)
    return context

