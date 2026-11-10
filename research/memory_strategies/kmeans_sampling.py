import torch
import torch.nn.functional as F

from .common import normalize, repeat_to_k


def select(tokens: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    if tokens.shape[0] < k:
        return normalize(repeat_to_k(tokens, k))
    from sklearn.cluster import KMeans

    x_np = tokens.detach().float().cpu().numpy()
    km = KMeans(n_clusters=k, random_state=int(seed), n_init=10).fit(x_np)
    centers = torch.from_numpy(km.cluster_centers_).to(device=tokens.device, dtype=tokens.dtype)
    return F.normalize(centers, dim=-1)
