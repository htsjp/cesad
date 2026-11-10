import torch
import torch.nn.functional as F


def reranked_logits(query, keys, values, variant, q_semantic=None, k_semantic=None, lambda_sem=0.1, tau_sem=0.5, eps=1e-6):
    sim = query @ keys.T
    if variant == "visual":
        reranked = sim
    else:
        if q_semantic is None or k_semantic is None:
            raise ValueError(f"{variant} requires query and prototype semantic scores")
        q = q_semantic.reshape(-1, 1)
        k = k_semantic.reshape(1, -1).to(q.device)
        if variant == "consistency":
            term = (1.0 - (q - k).abs() / 2.0).clamp(0.0, 1.0)
            reranked = sim.reshape(-1, sim.shape[-1]) + float(lambda_sem) * term
        elif variant == "signed":
            term = q * k
            reranked = sim.reshape(-1, sim.shape[-1]) + float(lambda_sem) * term
        elif variant == "gate":
            term = q * k
            gate = torch.sigmoid(term / float(tau_sem))
            reranked = sim.reshape(-1, sim.shape[-1]) + torch.log(gate + float(eps))
        else:
            raise ValueError(f"Unknown reranking variant: {variant}")
        reranked = reranked.reshape_as(sim)
    weights = F.softmax(reranked, dim=-1)
    logits = weights @ values.float()
    return logits, reranked, weights
