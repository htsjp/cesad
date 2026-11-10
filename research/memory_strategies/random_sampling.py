import torch

from .common import normalize


def select(tokens: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    n = tokens.shape[0]
    if n >= k:
        idx = torch.randperm(n, generator=generator, device="cpu")[:k].to(tokens.device)
    else:
        idx = torch.randint(0, n, (k,), generator=generator, device="cpu").to(tokens.device)
    return normalize(tokens[idx])
