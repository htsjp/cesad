import math

import torch
import torch.nn.functional as F


def normalize_geometry(g, eps=1e-12):
    g = g.float()
    g = g - g.mean(dim=0, keepdim=True)
    rms = g.pow(2).sum(dim=-1).mean().sqrt().clamp_min(eps)
    return g / rms


def square_geometry():
    return normalize_geometry(torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]))


def cross_geometry():
    return normalize_geometry(torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]))


def random_geometry(seed):
    gen = torch.Generator().manual_seed(int(seed))
    return normalize_geometry(torch.randn(4, 2, generator=gen))


def rotated_square(seed):
    gen = torch.Generator().manual_seed(int(seed))
    theta = float(torch.rand((), generator=gen) * 6.283185307179586)
    q = torch.tensor([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=torch.float32)
    return square_geometry() @ q
