import torch

from .common import normalize, repeat_to_k


def select(tokens: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    x = normalize(tokens)
    if x.shape[0] <= k:
        return normalize(repeat_to_k(x, k))
    selected = [0]
    dist = 1.0 - (x @ x[0:1].T).squeeze(-1)
    for _ in range(1, k):
        idx = torch.argmax(dist).item()
        selected.append(idx)
        dist = torch.minimum(dist, 1.0 - (x @ x[idx : idx + 1].T).squeeze(-1))
    return x[selected]
