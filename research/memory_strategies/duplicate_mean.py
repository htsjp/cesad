import torch

from .common import normalize


def select(tokens: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    mean = normalize(tokens.mean(dim=0, keepdim=True))
    return mean.repeat(k, 1)
