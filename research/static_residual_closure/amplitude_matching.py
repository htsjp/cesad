import torch
import torch.nn.functional as F


def residual_norm(memory, weights, a12=1.0, a18=1.0):
    v12 = memory["V12_heads"].float() * weights[12].to(memory["V12_heads"].device).view(1, -1, 1)
    v18 = memory["V18_heads"].float() * weights[18].to(memory["V18_heads"].device).view(1, -1, 1)
    r12 = F.normalize(v12.reshape(v12.shape[0], -1), dim=-1)
    r18 = F.normalize(v18.reshape(v18.shape[0], -1), dim=-1)
    residual = float(a12) * r12 + float(a18) * r18
    return float(residual.norm(dim=-1).mean().item()), float(residual.norm(dim=-1).std(unbiased=False).item())


def matched_alpha(base_norm, candidate_norm, base_alpha=0.5):
    if base_norm <= 1e-12:
        return float(base_alpha)
    return float(base_alpha) * float(candidate_norm) / float(base_norm)

