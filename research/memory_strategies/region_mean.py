import torch

from .common import normalize


def select(tokens: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    return normalize(tokens.mean(dim=0, keepdim=True))
