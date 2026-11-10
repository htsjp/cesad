import torch
import torch.nn.functional as F


def visual_stats(mu, labels, eps=1e-12):
    mu = mu.float()
    labels = torch.as_tensor(labels).long()
    mu0 = mu.mean(dim=0)
    centered = mu - mu0[None, :]
    cov = centered.T @ centered / max(1, centered.shape[0])
    cov = cov / (torch.trace(cov).clamp_min(eps))
    normal_mask = labels == 0
    abnormal_mask = labels == 1
    normal = mu[normal_mask].mean(dim=0) if bool(normal_mask.any()) else mu0.clone()
    abnormal = mu[abnormal_mask].mean(dim=0) if bool(abnormal_mask.any()) else mu0.clone()
    delta_raw = abnormal - normal
    if not torch.isfinite(delta_raw).all() or float(delta_raw.norm().item()) < eps:
        delta = torch.zeros_like(mu0)
    else:
        delta = F.normalize(delta_raw, dim=0)
    return {
        "mu0": mu0,
        "centered": centered,
        "sigma_bar": cov,
        "mu_normal": normal,
        "mu_abnormal": abnormal,
        "delta_v": delta,
        "centroid_distance": float((abnormal - normal).norm().item()),
        "has_normal_centroids": bool(normal_mask.any()),
        "has_abnormal_centroids": bool(abnormal_mask.any()),
    }
