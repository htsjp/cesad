import torch

from phase_xa.self_exclusion import apply_self_exclusion


def anomaly_margin(logits, values):
    labels = values[:, 1].bool().to(logits.device)
    anomaly = torch.logsumexp(logits[..., labels], dim=-1)
    normal = torch.logsumexp(logits[..., ~labels], dim=-1)
    return anomaly - normal


def posterior_from_logits(logits, values):
    return torch.softmax(logits, dim=-1) @ values.to(logits.device).float()


def teacher_targets(logits, values, provenance, sample_id, topk=8):
    excluded_logits, excluded = apply_self_exclusion(logits, provenance, sample_id)
    finite = torch.isfinite(excluded_logits)
    if not finite.any():
        raise RuntimeError("self-exclusion removed every teacher key")
    posterior = posterior_from_logits(excluded_logits, values)
    margin = anomaly_margin(excluded_logits, values)
    k = min(topk, excluded_logits.shape[-1])
    top_vals, top_idx = torch.topk(excluded_logits, k=k, dim=-1)
    return {
        "posterior": posterior,
        "anomaly_posterior": posterior[..., 1],
        "margin": margin,
        "top_values": top_vals,
        "top_indices": top_idx,
        "excluded_keys": excluded,
    }
