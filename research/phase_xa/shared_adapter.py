import math

import torch
from torch import nn


class SharedBlock18Adapter(nn.Module):
    def __init__(self, dim=1024, bottleneck=64, max_gate=0.20):
        super().__init__()
        self.pre_norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        initial = 0.05 / max_gate
        self.gate_raw = nn.Parameter(torch.tensor(math.log(initial / (1.0 - initial)), dtype=torch.float32))
        self.max_gate = float(max_gate)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.reset_batch_trace()

    @property
    def gate(self):
        return self.max_gate * torch.sigmoid(self.gate_raw)

    def reset_batch_trace(self):
        self.batch_trace = []

    def forward(self, x):
        residual = self.up(self.activation(self.down(self.pre_norm(x))))
        out = x + self.gate.to(dtype=x.dtype, device=x.device) * residual
        self.batch_trace.append({
            "input_shape": tuple(x.shape),
            "input_norm": float(x.detach().float().norm().cpu()),
            "residual_norm": float(residual.detach().float().norm().cpu()),
            "gate": float(self.gate.detach().cpu()),
        })
        return out


def trainable_parameter_count(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
