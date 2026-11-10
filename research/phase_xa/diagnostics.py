import torch


def effective_rank(features):
    x = features.detach().float().reshape(-1, features.shape[-1])
    if x.shape[0] < 2:
        return 1.0
    x = x - x.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(x)
    probs = singular / singular.sum().clamp_min(1e-12)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
    return float(torch.exp(entropy).cpu())


def finite_dict(values):
    for value in values.values():
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
            return False
    return True
