import torch

from .common import normalize, repeat_to_k


def select(tokens: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    x = normalize(tokens)
    if x.shape[0] <= k:
        return normalize(repeat_to_k(x, k))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    first = int(torch.randint(0, x.shape[0], (1,), generator=generator).item())
    selected = [first]
    dist = 1.0 - (x @ x[first : first + 1].T).squeeze(-1)
    for _ in range(1, k):
        idx = torch.argmax(dist).item()
        selected.append(idx)
        dist = torch.minimum(dist, 1.0 - (x @ x[idx : idx + 1].T).squeeze(-1))
    return x[selected]
