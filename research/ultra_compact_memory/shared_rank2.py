import torch
import torch.nn.functional as F

from shared_residual_memory.covariance import covariance_pca
from shared_residual_memory.shared_basis import coefficients


def decompose_fullfps(full_keys, group_size=4):
    if full_keys.shape[0] % group_size != 0:
        raise ValueError(f"FullFPS key count {full_keys.shape[0]} is not divisible by {group_size}")
    grouped = full_keys.float().reshape(-1, group_size, full_keys.shape[-1])
    mu = grouped.mean(dim=1)
    centered = grouped - mu[:, None, :]
    return grouped, mu, centered


def fit_residual_rank2(centered):
    flat = centered.reshape(-1, centered.shape[-1])
    basis, top, all_eigs = covariance_pca(flat, 2)
    coeff = coefficients(centered, basis)
    return {"basis": basis, "coeff": coeff, "eigvals": all_eigs, "top_eigvals": top}


def reconstruct(mu_hat, residual_basis, residual_coeff, normalize=True):
    residual = torch.matmul(residual_coeff.float(), residual_basis.float().T)
    out = mu_hat[:, None, :].float() + residual
    if normalize:
        out = F.normalize(out, dim=-1)
    return out.reshape(-1, out.shape[-1])


def centroid_repeated(mu):
    return F.normalize(mu[:, None, :].expand(-1, 4, -1), dim=-1).reshape(-1, mu.shape[-1])

