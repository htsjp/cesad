import torch
import torch.nn.functional as F

from .covariance import covariance_pca
from .shared_basis import coefficients, reconstruct_from_basis


def fit_label_bases(centered_groups: torch.Tensor, group_labels, max_rank: int):
    labels = torch.as_tensor(group_labels, dtype=torch.long)
    bases, eigvals, counts = {}, {}, {}
    for label in (0, 1):
        idx = torch.nonzero(labels == label, as_tuple=False).squeeze(1)
        flat = centered_groups[idx].reshape(-1, centered_groups.shape[-1])
        basis, top, all_vals = covariance_pca(flat, max_rank)
        bases[label] = basis
        eigvals[label] = {"top": top, "all": all_vals}
        counts[label] = int(flat.shape[0])
    return bases, eigvals, counts


def reconstruct_label_conditioned(mu, centered_groups, group_labels, bases, rank: int, normalize=True):
    labels = torch.as_tensor(group_labels, dtype=torch.long)
    out = torch.empty_like(centered_groups)
    for label in (0, 1):
        idx = torch.nonzero(labels == label, as_tuple=False).squeeze(1)
        b = bases[label][:, : int(rank)]
        coeff = coefficients(centered_groups[idx], b)
        rec = mu[idx, None, :] + torch.matmul(coeff, b.T)
        out[idx] = rec
    if normalize:
        out = F.normalize(out, dim=-1)
    return out.reshape(-1, out.shape[-1])

