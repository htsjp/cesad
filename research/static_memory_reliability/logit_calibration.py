import torch
import torch.nn.functional as F


def expand_region_prior(region_prior):
    return region_prior.float().repeat_interleave(4)


def calibrated_retrieval_logits(query, keys, values, region_prior=None, alpha=0.0, s_logit=1.0, temperature=1.0):
    sim = query @ keys.T
    scaled = sim / float(temperature)
    if region_prior is not None and float(alpha) != 0.0:
        entry_prior = expand_region_prior(region_prior).to(device=scaled.device, dtype=scaled.dtype)
        scaled = scaled + float(alpha) * float(s_logit) * entry_prior
    weights = F.softmax(scaled, dim=-1)
    logits = weights @ values.float()
    return logits, sim, weights


def source_logit_scale(query, keys, batch_size=256, temperature=1.0):
    vals = []
    for start in range(0, query.shape[0], batch_size):
        sim = query[start : start + batch_size].float() @ keys.float().T
        vals.append((sim / float(temperature)).std(dim=-1).detach().cpu())
    return float(torch.cat(vals).median().item())

