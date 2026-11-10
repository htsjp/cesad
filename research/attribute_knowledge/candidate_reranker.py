import torch
import torch.nn.functional as F


def _masked_cosine(q_profile, k_profile, q_idx, k_idx, top_idx, eps=1e-8):
    q = q_profile.reshape(-1, q_profile.shape[-1])
    k = k_profile
    q_top = q_idx.reshape(-1, q_idx.shape[-1])
    gather_k = k[top_idx.reshape(-1)].reshape(q.shape[0], top_idx.shape[1], -1)
    gather_k_idx = k_idx[top_idx.reshape(-1)].reshape(q.shape[0], top_idx.shape[1], -1)
    mask = torch.zeros(q.shape[0], top_idx.shape[1], q.shape[-1], device=q.device, dtype=torch.bool)
    mask.scatter_(2, q_top[:, None, :].expand(-1, top_idx.shape[1], -1), True)
    mask.scatter_(2, gather_k_idx, True)
    q_masked = q[:, None, :] * mask.float()
    k_masked = gather_k * mask.float()
    q_masked = q_masked / q_masked.norm(dim=-1, keepdim=True).clamp_min(eps)
    k_masked = k_masked / k_masked.norm(dim=-1, keepdim=True).clamp_min(eps)
    return (q_masked * k_masked).sum(dim=-1).clamp(0.0, 1.0)


def _overlap(q_idx, k_idx, q_profile, k_profile, top_idx, eps=1e-8):
    q_top = q_idx.reshape(-1, q_idx.shape[-1])
    gather_k_idx = k_idx[top_idx.reshape(-1)].reshape(q_top.shape[0], top_idx.shape[1], -1)
    q_weight = q_profile.reshape(-1, q_profile.shape[-1])
    k_weight = k_profile
    out = torch.zeros(q_top.shape[0], top_idx.shape[1], device=q_profile.device)
    for j in range(top_idx.shape[1]):
        kk = gather_k_idx[:, j, :]
        union = torch.cat([q_top, kk], dim=-1)
        numerator = torch.zeros(q_top.shape[0], device=q_profile.device)
        denominator = torch.zeros_like(numerator)
        for a in range(union.shape[1]):
            idx = union[:, a]
            qw = q_weight.gather(1, idx[:, None]).squeeze(1)
            kw = k_weight[top_idx[:, j]].gather(1, idx[:, None]).squeeze(1)
            numerator += torch.minimum(qw, kw)
            denominator += torch.maximum(qw, kw)
        out[:, j] = (numerator / denominator.clamp_min(eps)).clamp(0.0, 1.0)
    return out


def _cross(q_profile, k_profile, q_idx, top_idx):
    q = q_profile.reshape(-1, q_profile.shape[-1])
    q_top = q_idx.reshape(-1, q_idx.shape[-1])
    gather_k = k_profile[top_idx.reshape(-1)].reshape(q.shape[0], top_idx.shape[1], -1)
    q_selected = q.gather(1, q_top)
    k_selected = gather_k.gather(2, q_top[:, None, :].expand(-1, top_idx.shape[1], -1))
    return (q_selected[:, None, :] * k_selected).sum(dim=-1).clamp(0.0, 1.0)


def _full_cosine(q_profile, k_profile, top_idx):
    q = F.normalize(q_profile.reshape(-1, q_profile.shape[-1]), dim=-1)
    k = F.normalize(k_profile[top_idx.reshape(-1)], dim=-1).reshape(q.shape[0], top_idx.shape[1], -1)
    return (q[:, None, :] * k).sum(dim=-1).clamp(0.0, 1.0)


def candidate_reranked_logits(
    query,
    keys,
    values,
    q_profile,
    k_profile,
    top_l=20,
    lambda_attr=0.05,
    agreement="cross",
    q_indices=None,
    k_indices=None,
):
    sim = query @ keys.T
    flat = sim.reshape(-1, sim.shape[-1])
    top_l = min(int(top_l), flat.shape[-1])
    top_vals, top_idx = torch.topk(flat, k=top_l, dim=-1)
    if agreement == "full_cosine":
        agree = _full_cosine(q_profile, k_profile, top_idx)
    elif agreement == "cosine":
        agree = _masked_cosine(q_profile, k_profile, q_indices, k_indices, top_idx)
    elif agreement == "overlap":
        agree = _overlap(q_indices, k_indices, q_profile, k_profile, top_idx)
    elif agreement == "cross":
        agree = _cross(q_profile, k_profile, q_indices, top_idx)
    else:
        raise ValueError(f"Unknown agreement: {agreement}")
    reranked = flat.clone()
    reranked.scatter_(1, top_idx, top_vals + float(lambda_attr) * agree)
    reranked = reranked.reshape_as(sim)
    weights = F.softmax(reranked, dim=-1)
    logits = weights @ values.float()
    return logits, reranked, weights, top_idx.reshape(*sim.shape[:-1], top_l)
