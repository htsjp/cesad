from typing import Dict

import torch
from torch import nn

from .residual_fusion import mean_one_softmax


def local_head_statistics(heads_bnhd: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = heads_bnhd.float()
    norm = x.norm(dim=-1)
    channel_std = x.std(dim=-1, unbiased=False)
    mean_head = x.mean(dim=1, keepdim=True)
    cosine_to_image_mean = torch.nn.functional.cosine_similarity(x, mean_head.expand_as(x), dim=-1)
    n = x.shape[1]
    side = int(n ** 0.5)
    if side * side == n:
        grid = x.reshape(x.shape[0], side, side, x.shape[2], x.shape[3])
        neigh = (
            torch.roll(grid, shifts=1, dims=1)
            + torch.roll(grid, shifts=-1, dims=1)
            + torch.roll(grid, shifts=1, dims=2)
            + torch.roll(grid, shifts=-1, dims=2)
        ) / 4.0
        neigh[:, 0] = (grid[:, 0] + grid[:, 1]) / 2.0
        neigh[:, -1] = (grid[:, -1] + grid[:, -2]) / 2.0
        neigh[:, :, 0] = (grid[:, :, 0] + grid[:, :, 1]) / 2.0
        neigh[:, :, -1] = (grid[:, :, -1] + grid[:, :, -2]) / 2.0
        neigh = neigh.reshape_as(x)
        contrast = (x - neigh).norm(dim=-1)
    else:
        contrast = torch.zeros_like(norm)
    return torch.stack([norm, channel_std, cosine_to_image_mean, contrast.clamp_min(eps)], dim=-1)


class RunningStats:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, rows: torch.Tensor):
        flat = rows.reshape(-1, rows.shape[-1]).float()
        self.mean = flat.mean(dim=0)
        self.std = flat.std(dim=0, unbiased=False).clamp_min(1e-6)
        return self

    def normalize(self, rows: torch.Tensor) -> torch.Tensor:
        return (rows.float() - self.mean.to(rows.device)) / self.std.to(rows.device)


class QueryAdaptiveHeadGate(nn.Module):
    trainable = True

    def __init__(self, heads: int, use_head_embedding: bool = False, head_embed_dim: int = 4):
        super().__init__()
        self.heads = int(heads)
        self.use_head_embedding = bool(use_head_embedding)
        in_dim = 4 + (head_embed_dim if use_head_embedding else 0)
        self.mlp12 = nn.Sequential(nn.Linear(in_dim, 16), nn.GELU(), nn.Linear(16, 1))
        self.mlp18 = nn.Sequential(nn.Linear(in_dim, 16), nn.GELU(), nn.Linear(16, 1))
        if use_head_embedding:
            self.head_embedding = nn.Embedding(heads, head_embed_dim)
        self.stats12 = RunningStats()
        self.stats18 = RunningStats()

    def fit_normalizer(self, features_list):
        rows12, rows18 = [], []
        for features in features_list:
            rows12.append(local_head_statistics(features["V12_heads"]).detach().cpu())
            rows18.append(local_head_statistics(features["V18_heads"]).detach().cpu())
        self.stats12.fit(torch.cat(rows12, dim=0))
        self.stats18.fit(torch.cat(rows18, dim=0))

    def _mlp(self, layer: int):
        return self.mlp12 if int(layer) == 12 else self.mlp18

    def _stats(self, layer: int):
        return self.stats12 if int(layer) == 12 else self.stats18

    def forward(self, features: Dict[str, torch.Tensor], layer: int) -> torch.Tensor:
        x = features[f"V{layer}_heads"]
        stats = self._stats(layer).normalize(local_head_statistics(x))
        if self.use_head_embedding:
            ids = torch.arange(x.shape[-2], device=x.device)
            emb = self.head_embedding(ids).view(*([1] * (stats.ndim - 2)), x.shape[-2], -1).expand(*stats.shape[:-1], -1)
            stats = torch.cat([stats, emb], dim=-1)
        logits = self._mlp(layer)(stats).squeeze(-1)
        return mean_one_softmax(logits)
