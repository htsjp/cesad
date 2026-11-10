import torch
import torch.nn.functional as F


def reconstruction_row(name, original, reconstructed, family, rank, energy=None):
    o = F.normalize(original.float(), dim=-1)
    r = F.normalize(reconstructed.float(), dim=-1)
    cos = (o * r).sum(dim=-1)
    return {
        "variant": name,
        "family": family,
        "rank": rank,
        "energy_retention": float(energy) if energy is not None else float("nan"),
        "mse": float((o - r).pow(2).mean().item()),
        "mean_cosine_to_fullfps": float(cos.mean().item()),
        "worst_cosine_to_fullfps": float(cos.min().item()),
        "max_abs_diff_to_fullfps": float((o - r).abs().max().item()),
    }


def basis_rows(name, eigvals, rank, orth_error, label="global"):
    total = eigvals.sum().clamp_min(1e-12)
    return {
        "variant": name,
        "label": label,
        "rank": rank,
        "orthonormal_max_error": float(orth_error),
        "top_eigenvalue": float(eigvals[0].item()) if eigvals.numel() else float("nan"),
        "rank_explained_variance": float(eigvals[: int(rank)].sum().item() / total.item()),
    }

