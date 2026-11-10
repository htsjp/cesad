import math

import torch
import torch.nn.functional as F


def read_memory(q, keys, values, exclude_mask=None):
    sim = torch.matmul(q.float(), keys.float().T)
    if exclude_mask is not None:
        sim = sim.masked_fill(exclude_mask[None, :].to(sim.device), -1e4)
    weights = F.softmax(sim, dim=-1)
    logits = weights @ values.float()
    return logits, sim, weights


def patch_memory_context(q, keys, values, eps=1e-6, exclude_mask=None):
    logits, sim, weights = read_memory(q, keys, values, exclude_mask=exclude_mask)
    labels = values.argmax(dim=-1)
    normal = (labels == 0).float().to(q.device)
    anomaly = (labels == 1).float().to(q.device)
    s_n = weights @ normal
    s_a = weights @ anomaly
    c_n = (weights * normal[None, :]) @ keys.float()
    c_a = (weights * anomaly[None, :]) @ keys.float()
    c_n = F.normalize(c_n / (s_n[:, None] + eps), dim=-1)
    c_a = F.normalize(c_a / (s_a[:, None] + eps), dim=-1)
    r_m = F.normalize(c_a - c_n, dim=-1)
    prob = F.softmax(sim, dim=-1)
    entropy = -(prob * prob.clamp_min(1e-12).log()).sum(dim=-1) / max(1e-12, math.log(keys.shape[0]))
    top2 = torch.topk(sim, k=min(2, sim.shape[-1]), dim=-1).values
    margin = top2[:, 0] - (top2[:, 1] if top2.shape[1] > 1 else 0)
    stats = {
        "entropy": entropy,
        "margin": margin,
        "mass_diff": s_a - s_n,
        "cos_q_cn": (F.normalize(q, dim=-1) * c_n).sum(dim=-1),
        "cos_q_ca": (F.normalize(q, dim=-1) * c_a).sum(dim=-1),
        "cos_cn_ca": (c_n * c_a).sum(dim=-1),
        "s_n": s_n,
        "s_a": s_a,
    }
    return logits, sim, weights, r_m, stats


def pool_image_context(q, keys, values, k=10, mode="memory", exclude_mask=None, random_seed=None, swap=False, label_blind=False):
    logits, sim, weights, r_m, stats = patch_memory_context(q, keys, values, exclude_mask=exclude_mask)
    score = logits[:, 1]
    n = q.shape[0]
    kk = min(int(k), n)
    if random_seed is not None:
        gen = torch.Generator(device=q.device).manual_seed(int(random_seed))
        top_idx = torch.randperm(n, generator=gen, device=q.device)[:kk]
        bottom_idx = torch.randperm(n, generator=gen, device=q.device)[-kk:]
    else:
        top_idx = torch.topk(score, kk, largest=True).indices
        bottom_idx = torch.topk(score, kk, largest=False).indices
    if label_blind:
        ctx = F.normalize(weights @ keys.float(), dim=-1)
        r_m = ctx
    if swap:
        r_m = -r_m
    if mode == "image":
        r_m = F.normalize(q, dim=-1)
    r_top = r_m[top_idx].mean(dim=0)
    r_bottom = r_m[bottom_idx].mean(dim=0)
    r_img = F.normalize(r_top - r_bottom, dim=-1)
    scalar_parts = []
    for idx in (top_idx, bottom_idx, torch.arange(n, device=q.device)):
        for name in ("entropy", "margin", "mass_diff", "cos_q_cn", "cos_q_ca", "cos_cn_ca"):
            scalar_parts.append(stats[name][idx].mean())
    u_img = torch.stack(scalar_parts)[:18]
    patch_gate = torch.clamp(1.0 - stats["entropy"], 0.0, 1.0)
    diag = {
        "top_indices": top_idx.detach().cpu().tolist(),
        "bottom_indices": bottom_idx.detach().cpu().tolist(),
        "initial_scores": score.detach().cpu().tolist(),
        "context_norm": float(r_img.norm().detach().cpu()),
        "top_bottom_context_cosine": float(F.cosine_similarity(r_m[top_idx].mean(0), r_m[bottom_idx].mean(0), dim=0).detach().cpu()),
        "mean_entropy": float(stats["entropy"].mean().detach().cpu()),
        "mean_s_n": float(stats["s_n"].mean().detach().cpu()),
        "mean_s_a": float(stats["s_a"].mean().detach().cpu()),
        "mean_patch_gate": float(patch_gate.mean().detach().cpu()),
    }
    return logits, r_img, u_img, patch_gate, diag

