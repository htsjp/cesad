import torch

from shared_residual_memory.covariance import covariance_pca


def fit_centroid_pca(mu, max_rank):
    global_mean = mu.float().mean(dim=0)
    centered = mu.float() - global_mean
    basis, top, all_eigs = covariance_pca(centered, max_rank)
    coeff = centered @ basis
    return {"mean": global_mean, "basis": basis, "coeff": coeff, "eigvals": all_eigs, "top_eigvals": top}


def reconstruct_centroids(fit, rank, coeff_override=None, basis_override=None):
    basis = fit["basis"][:, :rank] if basis_override is None else basis_override[:, :rank]
    coeff = fit["coeff"][:, :rank] if coeff_override is None else coeff_override[:, :rank]
    return fit["mean"][None, :] + coeff.float() @ basis.float().T


def random_orthonormal(dim, rank, seed):
    gen = torch.Generator().manual_seed(int(seed))
    q, _ = torch.linalg.qr(torch.randn(dim, rank, generator=gen))
    return q[:, :rank].float()


def random_subspace_reconstruct(mu, rank, seed, energy_match=False):
    global_mean = mu.float().mean(dim=0)
    centered = mu.float() - global_mean
    basis = random_orthonormal(mu.shape[-1], rank, seed)
    coeff = centered @ basis
    recon = global_mean[None, :] + coeff @ basis.T
    if energy_match:
        true_energy = centered.norm(dim=1).clamp_min(1e-12)
        rec_delta = recon - global_mean[None, :]
        rec_energy = rec_delta.norm(dim=1).clamp_min(1e-12)
        recon = global_mean[None, :] + rec_delta * (true_energy / rec_energy)[:, None]
    return recon, basis, coeff


def same_label_shuffle_coeff(coeff, labels, seed):
    out = coeff.clone()
    gen = torch.Generator().manual_seed(int(seed))
    labels_t = torch.as_tensor(labels)
    for label in sorted(set(int(x) for x in labels)):
        idx = torch.nonzero(labels_t == label, as_tuple=False).squeeze(1)
        if idx.numel() > 1:
            out[idx] = out[idx[torch.randperm(idx.numel(), generator=gen)]]
    return out

