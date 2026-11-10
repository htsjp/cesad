import torch
from torch import nn


class ZeroResidualModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(()))
        self.direction = nn.Parameter(torch.randn(dim) * 1e-4)

    def forward(self, x):
        residual = self.scale * self.direction.to(dtype=x.dtype, device=x.device)
        return x + residual.view(1, 1, -1)


class IdentityInsertion(nn.Module):
    def forward(self, x):
        return x
