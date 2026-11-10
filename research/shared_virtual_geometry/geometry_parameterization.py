import torch
import torch.nn as nn

from shared_virtual_geometry.fixed_geometry import normalize_geometry, square_geometry


class SharedGeometry(nn.Module):
    def __init__(self, label_specific=True, seed=0, init="square"):
        super().__init__()
        base = square_geometry()
        gen = torch.Generator().manual_seed(int(seed))
        if init == "random":
            base = normalize_geometry(torch.randn(4, 2, generator=gen))
        h0 = base + 0.01 * torch.randn(4, 2, generator=gen)
        if label_specific:
            h1 = base + 0.01 * torch.randn(4, 2, generator=gen)
            raw = torch.stack([h0, h1], dim=0)
        else:
            raw = h0[None, :, :]
        self.raw = nn.Parameter(raw.float())
        self.label_specific = bool(label_specific)

    def geometries(self):
        geoms = torch.stack([normalize_geometry(g) for g in self.raw], dim=0)
        if not self.label_specific:
            geoms = geoms.expand(2, -1, -1)
        return geoms

    def extra_repr(self):
        return f"label_specific={self.label_specific}"


def trainable_parameter_count(module):
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))

