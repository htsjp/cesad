import torch
import torch.nn.functional as F


def shuffled_delta(mu, labels, seed):
    labels = torch.as_tensor(labels).long().clone()
    gen = torch.Generator().manual_seed(int(seed))
    labels = labels[torch.randperm(labels.numel(), generator=gen)]
    mu0 = mu.float().mean(dim=0)
    normal = mu[labels == 0].mean(dim=0) if bool((labels == 0).any()) else mu0
    abnormal = mu[labels == 1].mean(dim=0) if bool((labels == 1).any()) else mu0
    raw = abnormal - normal
    if not torch.isfinite(raw).all() or float(raw.norm().item()) < 1e-12:
        return torch.zeros_like(mu0)
    return F.normalize(raw, dim=0)
