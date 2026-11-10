import torch
import torch.nn.functional as F


def coefficients(centered_groups: torch.Tensor, basis_dk: torch.Tensor) -> torch.Tensor:
    return torch.matmul(centered_groups.float(), basis_dk.float())


def reconstruct_from_basis(mu: torch.Tensor, coeff: torch.Tensor, basis_dk: torch.Tensor, normalize=True) -> torch.Tensor:
    residual = torch.matmul(coeff.float(), basis_dk.float().T)
    out = mu[:, None, :].float() + residual
    if normalize:
        out = F.normalize(out, dim=-1)
    return out.reshape(-1, out.shape[-1])


def reconstruct_grouped(mu: torch.Tensor, coeff: torch.Tensor, basis_dk: torch.Tensor, normalize=True) -> torch.Tensor:
    residual = torch.matmul(coeff.float(), basis_dk.float().T)
    out = mu[:, None, :].float() + residual
    if normalize:
        out = F.normalize(out, dim=-1)
    return out


def retained_energy(centered_groups: torch.Tensor, coeff: torch.Tensor):
    original = centered_groups.float().pow(2).sum(dim=(1, 2)).clamp_min(1e-12)
    kept = coeff.float().pow(2).sum(dim=(1, 2))
    return kept / original

