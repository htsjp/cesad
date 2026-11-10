import torch
import torch.nn.functional as F


def fit_global_basis(centered_groups: torch.Tensor, rank: int) -> torch.Tensor:
    flat = centered_groups.reshape(-1, centered_groups.shape[-1]).float()
    _, _, vh = torch.linalg.svd(flat, full_matrices=False)
    return vh[: int(rank)]


def reconstruct_global(mu: torch.Tensor, centered: torch.Tensor, basis: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    coeff = centered @ basis.T
    residual = coeff @ basis
    out = mu[:, None, :] + residual
    if normalize:
        out = F.normalize(out, dim=-1)
    return out.reshape(-1, out.shape[-1])

