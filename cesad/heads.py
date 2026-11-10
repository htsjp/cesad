"""Exact source-extracted CES/NTR heads and losses for inspection.

No encoder, target data, asset fallback or training is run at import time.
Canonical historical implementations remain under research/.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import math
import numpy as np
import torch
import torch.nn.functional as F

SEED = 0
RHO = 0.10


def unit_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x) + eps)


def state_pair_from_m_alpha_v(
    m: np.ndarray, alpha: float, v: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    s_a = unit_np(m + float(alpha) * v)
    s_n = unit_np(m - float(alpha) * v)
    return s_n.astype(np.float32), s_a.astype(np.float32)


def ces_weight_np(m: np.ndarray, alpha: float, v: np.ndarray) -> np.ndarray:
    s_n, s_a = state_pair_from_m_alpha_v(m, alpha, v)
    return (s_a - s_n).astype(np.float32)


@dataclass
class HeadSpec:
    method: str
    params: Dict[str, np.ndarray | float]
    source: str
    seed: int = SEED

    def weight(self) -> np.ndarray:
        if self.method == "LinearHead":
            return np.asarray(self.params["w"], dtype=np.float32)
        if self.method == "NormalizedLinearHead":
            u = unit_np(np.asarray(self.params["u"], dtype=np.float32))
            return float(self.params["c0"]) * u
        if self.method == "CES":
            return ces_weight_np(
                np.asarray(self.params["m"], dtype=np.float32),
                float(self.params["alpha"]),
                unit_np(np.asarray(self.params["v"], dtype=np.float32)),
            )
        raise ValueError(self.method)

    def bias(self) -> float:
        if self.method == "LinearHead":
            return float(self.params.get("b", 0.0))
        return 0.0

    def logits_np(self, z: np.ndarray) -> np.ndarray:
        return z.astype(np.float32) @ self.weight().astype(np.float32) + self.bias()


class LinearHead(torch.nn.Module):
    def __init__(self, w0: np.ndarray):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(w0, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.w + self.b

    def to_spec(self, source: str) -> HeadSpec:
        return HeadSpec(
            "LinearHead",
            {"w": self.w.detach().cpu().numpy(), "b": float(self.b.detach().cpu())},
            source,
        )


class NormalizedLinearHead(torch.nn.Module):
    def __init__(self, w0: np.ndarray, c0: float):
        super().__init__()
        self.u = torch.nn.Parameter(torch.tensor(w0, dtype=torch.float32))
        self.register_buffer("c0", torch.tensor(float(c0), dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        u = F.normalize(self.u, dim=0)
        return self.c0 * (z @ u)

    def to_spec(self, source: str) -> HeadSpec:
        return HeadSpec(
            "NormalizedLinearHead",
            {"u": self.u.detach().cpu().numpy(), "c0": float(self.c0.detach().cpu())},
            source,
        )


class CESHead(torch.nn.Module):
    def __init__(self, m: np.ndarray, alpha: float, v0: np.ndarray):
        super().__init__()
        self.register_buffer("m", torch.tensor(m, dtype=torch.float32))
        self.register_buffer("alpha", torch.tensor(float(alpha), dtype=torch.float32))
        self.v = torch.nn.Parameter(torch.tensor(v0, dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        v = F.normalize(self.v, dim=0)
        s_a = F.normalize(self.m + self.alpha * v, dim=0)
        s_n = F.normalize(self.m - self.alpha * v, dim=0)
        return z @ (s_a - s_n)

    def to_spec(self, source: str) -> HeadSpec:
        return HeadSpec(
            "CES",
            {
                "m": self.m.detach().cpu().numpy(),
                "alpha": float(self.alpha.detach().cpu()),
                "v": F.normalize(self.v.detach(), dim=0).cpu().numpy(),
            },
            source,
        )


def initial_geometry(source_info: dict) -> Dict[str, np.ndarray | float]:
    m = np.asarray(source_info["m"], dtype=np.float32)
    alpha = float(source_info["alpha"])
    init_v = unit_np(np.asarray(source_info["init_v"], dtype=np.float32))
    s_n0, s_a0 = state_pair_from_m_alpha_v(m, alpha, init_v)
    w0 = (s_a0 - s_n0).astype(np.float32)
    c0 = float(np.linalg.norm(w0))
    return {
        "m": m,
        "alpha": alpha,
        "init_v": init_v,
        "s_n0": s_n0,
        "s_a0": s_a0,
        "w0": w0,
        "c0": c0,
    }


def ntr_loss_from_logits(
    logits: torch.Tensor, yb: torch.Tensor, rho: float = RHO
) -> torch.Tensor:
    neg = logits[yb < 0.5]
    if neg.numel() == 0:
        return logits.new_tensor(0.0)
    k = max(1, int(math.ceil(rho * int(neg.numel()))))
    vals, idx = torch.topk(neg.detach(), k)
    del vals
    return F.softplus(neg[idx]).mean()


def dice_loss(logits, y):
    p = torch.sigmoid(logits)
    y = y.float()
    return 1.0 - (2.0 * (p * y).sum() + 1.0) / (p.sum() + y.sum() + 1.0)


def balanced_loss(logits, y):
    y = y.float()
    pos = y.sum().clamp_min(1.0)
    neg = (1.0 - y).sum().clamp_min(1.0)
    w = torch.where(y > 0.5, 0.5 * (pos + neg) / pos, 0.5 * (pos + neg) / neg)
    bce = F.binary_cross_entropy_with_logits(logits, y, weight=w)
    dl = dice_loss(logits, y)
    return bce + dl, bce.detach(), dl.detach()
