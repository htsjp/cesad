from typing import Dict

import torch
from torch import nn

from .residual_fusion import mean_one_softmax


class UniformHeadGate(nn.Module):
    trainable = False

    def __init__(self, heads: int):
        super().__init__()
        self.heads = int(heads)

    def forward(self, features: Dict[str, torch.Tensor], layer: int) -> torch.Tensor:
        x = features[f"V{layer}_heads"]
        return torch.ones((*x.shape[:-1],), device=x.device, dtype=x.dtype)


class LayerScalarGate(nn.Module):
    trainable = True

    def __init__(self, heads: int):
        super().__init__()
        self.heads = int(heads)
        self.logits = nn.Parameter(torch.zeros(2))

    def layer_weights(self):
        return torch.softmax(self.logits, dim=0) * 2.0

    def forward(self, features: Dict[str, torch.Tensor], layer: int) -> torch.Tensor:
        idx = 0 if int(layer) == 12 else 1
        x = features[f"V{layer}_heads"]
        return torch.ones((*x.shape[:-1],), device=x.device, dtype=x.dtype) * self.layer_weights()[idx]


class StaticHeadGate(nn.Module):
    trainable = True

    def __init__(self, heads: int, init=None):
        super().__init__()
        self.heads = int(heads)
        if init is None:
            init = torch.zeros(2, heads)
        self.logits = nn.Parameter(init.float().clone())

    def head_weights(self, layer: int) -> torch.Tensor:
        idx = 0 if int(layer) == 12 else 1
        return mean_one_softmax(self.logits[idx])

    def forward(self, features: Dict[str, torch.Tensor], layer: int) -> torch.Tensor:
        x = features[f"V{layer}_heads"]
        return self.head_weights(layer).to(x.device, x.dtype).view(*([1] * (x.ndim - 2)), -1).expand(*x.shape[:-1])


class FixedHeadWeights(nn.Module):
    trainable = False

    def __init__(self, weights_by_layer: Dict[int, torch.Tensor]):
        super().__init__()
        for layer, weights in weights_by_layer.items():
            self.register_buffer(f"w_{int(layer)}", weights.float())

    def forward(self, features: Dict[str, torch.Tensor], layer: int) -> torch.Tensor:
        x = features[f"V{layer}_heads"]
        w = getattr(self, f"w_{int(layer)}").to(x.device, x.dtype)
        return w.view(*([1] * (x.ndim - 2)), -1).expand(*x.shape[:-1])

