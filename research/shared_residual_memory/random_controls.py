import torch
import torch.nn.functional as F

from .shared_basis import coefficients, reconstruct_from_basis


def random_basis(dim: int, rank: int, seed: int):
    gen = torch.Generator().manual_seed(int(seed))
    q, _ = torch.linalg.qr(torch.randn(dim, rank, generator=gen), mode="reduced")
    return q[:, : int(rank)].contiguous()


def energy_matched_random(mu, centered_groups, pca_coeff, rank: int, seed: int, normalize=True):
    b = random_basis(centered_groups.shape[-1], rank, seed)
    c = coefficients(centered_groups, b)
    target = pca_coeff.float().pow(2).sum(dim=(1, 2), keepdim=True).sqrt()
    current = c.float().pow(2).sum(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-12)
    c = c / current * target
    c = c - c.mean(dim=1, keepdim=True)
    return reconstruct_from_basis(mu, c, b, normalize=normalize), c, b


def same_label_coefficient_shuffle(mu, coeff, basis, group_labels, seed: int, normalize=True):
    labels = torch.as_tensor(group_labels, dtype=torch.long)
    gen = torch.Generator().manual_seed(int(seed))
    shuffled = coeff.clone()
    for label in (0, 1):
        idx = torch.nonzero(labels == label, as_tuple=False).squeeze(1)
        perm = idx[torch.randperm(idx.numel(), generator=gen)]
        shuffled[idx] = coeff[perm]
        target = coeff[idx].float().pow(2).sum(dim=(1, 2), keepdim=True).sqrt()
        current = shuffled[idx].float().pow(2).sum(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-12)
        shuffled[idx] = shuffled[idx] / current * target
    return reconstruct_from_basis(mu, shuffled, basis, normalize=normalize), shuffled


def cross_label_coefficient_shuffle(mu, coeff, basis, group_labels, seed: int, normalize=True):
    labels = torch.as_tensor(group_labels, dtype=torch.long)
    gen = torch.Generator().manual_seed(int(seed))
    shuffled = coeff.clone()
    idx0 = torch.nonzero(labels == 0, as_tuple=False).squeeze(1)
    idx1 = torch.nonzero(labels == 1, as_tuple=False).squeeze(1)
    for dst, src in ((idx0, idx1), (idx1, idx0)):
        take = src[torch.randint(0, src.numel(), (dst.numel(),), generator=gen)]
        shuffled[dst] = coeff[take]
        target = coeff[dst].float().pow(2).sum(dim=(1, 2), keepdim=True).sqrt()
        current = shuffled[dst].float().pow(2).sum(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-12)
        shuffled[dst] = shuffled[dst] / current * target
    return reconstruct_from_basis(mu, shuffled, basis, normalize=normalize), shuffled


def fixed_simplex_coefficients(mu, centered_groups, basis, mode="region_norm", normalize=True):
    rank = basis.shape[1]
    if rank == 1:
        pattern = torch.tensor([[-1.5], [-0.5], [0.5], [1.5]], dtype=torch.float32)
    else:
        eye = torch.eye(rank, dtype=torch.float32)
        pts = []
        for i in range(4):
            pts.append(eye[i % rank])
        pattern = torch.stack(pts, dim=0)
        pattern = pattern - pattern.mean(dim=0, keepdim=True)
    pattern = pattern / pattern.pow(2).sum().sqrt().clamp_min(1e-12)
    coeff = pattern[None, :, :].expand(centered_groups.shape[0], -1, -1).clone()
    norms = centered_groups.float().pow(2).sum(dim=(1, 2), keepdim=True).sqrt()
    if mode == "label_median":
        scale = norms.median().view(1, 1, 1)
    else:
        scale = norms
    coeff = coeff * scale
    return reconstruct_from_basis(mu, coeff, basis, normalize=normalize), coeff

def gaussian_matched_coefficients(mu, coeff_ref, basis, seed: int, normalize=True):
    gen = torch.Generator().manual_seed(int(seed))
    flat = coeff_ref.reshape(-1, coeff_ref.shape[-1]).float()
    cov = torch.cov(flat.T) + 1e-6 * torch.eye(flat.shape[-1])
    chol = torch.linalg.cholesky(cov)
    z = torch.randn(coeff_ref.shape, generator=gen)
    coeff = z @ chol.T
    coeff = coeff - coeff.mean(dim=1, keepdim=True)
    target = coeff_ref.float().pow(2).sum(dim=(1, 2), keepdim=True).sqrt()
    current = coeff.float().pow(2).sum(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-12)
    coeff = coeff / current * target
    return reconstruct_from_basis(mu, coeff, basis, normalize=normalize), coeff

