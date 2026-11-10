"""The fixed source transform; does not estimate anything from target data."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SourceTransform(nn.Module):
    def __init__(self, pc1=None, adapter_path=None):
        super().__init__()
        self.register_buffer(
            "pc1",
            None if pc1 is None else torch.as_tensor(np.load(pc1), dtype=torch.float32),
        )
        self.a = self.b = None
        if adapter_path is not None:
            payload = torch.load(adapter_path, map_location="cpu", weights_only=False)
            state = payload["model"]
            rank, dim = state["a.weight"].shape
            self.a = nn.Linear(dim, rank, bias=False)
            self.b = nn.Linear(rank, dim, bias=False)
            self.a.weight.data.copy_(state["a.weight"])
            self.b.weight.data.copy_(state["b.weight"])
        self.requires_grad_(False)
        self.eval()

    def forward(self, z):
        if self.pc1 is not None:
            z = z - (z @ self.pc1).unsqueeze(-1) * self.pc1
        z = F.normalize(z, dim=-1)
        if self.a is not None:
            z = F.normalize(z + self.b(self.a(z)), dim=-1)
        return z
