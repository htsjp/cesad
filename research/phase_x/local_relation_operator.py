import math

import torch
from torch import nn
import torch.nn.functional as F


def neighbor_index(side: int, include_center: bool = False, device=None):
    rows = []
    masks = []
    for y in range(side):
        for x in range(side):
            cur = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if not include_center and dy == 0 and dx == 0:
                        continue
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < side and 0 <= xx < side:
                        cur.append(yy * side + xx)
            rows.append(cur)
    k = max(len(r) for r in rows)
    idx = torch.zeros((side * side, k), dtype=torch.long, device=device)
    mask = torch.zeros((side * side, k), dtype=torch.bool, device=device)
    for i, row in enumerate(rows):
        idx[i, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        mask[i, : len(row)] = True
    return idx, mask


class PlainTokenAdapter(nn.Module):
    accepts_list_output = True

    def __init__(self, dim: int, hidden: int = 32):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        nn.init.normal_(self.fc1.weight, std=0.01)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.last_stats = {}

    def adapt_nld(self, x):
        cls, patch = x[:, :1], x[:, 1:]
        delta = self.fc2(F.gelu(self.fc1(self.ln(patch))))
        self.last_stats = {
            "mean_entropy": float("nan"),
            "normalized_entropy": float("nan"),
            "top1_attention_probability": float("nan"),
            "effective_neighbor_count": float("nan"),
            "residual_ratio": (delta.float().norm(dim=-1) / (patch.float().norm(dim=-1) + 1e-6)).detach(),
        }
        return torch.cat([cls, patch + delta], dim=1)

    def forward(self, x):
        if isinstance(x, list):
            return [self.adapt_nld(x[0].permute(1, 0, 2)).permute(1, 0, 2), x[1]]
        return self.adapt_nld(x)


class LocalRelationOperator(nn.Module):
    accepts_list_output = True

    def __init__(self, dim: int, side: int, hidden: int = 32, relation_dim: int = 32, mode: str = "relative"):
        super().__init__()
        if mode not in {"relative", "absolute"}:
            raise ValueError(f"mode must be relative or absolute, got {mode}")
        self.mode = mode
        self.side = int(side)
        self.relation_dim = int(relation_dim)
        self.center_ln = nn.LayerNorm(dim)
        self.rel_ln = nn.LayerNorm(dim)
        self.wq = nn.Linear(dim, relation_dim)
        self.wk = nn.Linear(dim, relation_dim)
        self.wv = nn.Linear(dim, relation_dim)
        self.fc1 = nn.Linear(relation_dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        for layer in (self.wq, self.wk, self.wv, self.fc1):
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        idx, mask = neighbor_index(self.side, include_center=False)
        self.register_buffer("neighbor_idx", idx, persistent=False)
        self.register_buffer("neighbor_mask", mask, persistent=False)
        self.last_stats = {}

    def adapt_nld(self, x):
        cls, patch = x[:, :1], x[:, 1:]
        b, n, d = patch.shape
        if n != self.side * self.side:
            side = int(round(math.sqrt(n)))
            if side * side != n:
                raise ValueError(f"patch token count {n} is not a square grid")
            idx, mask = neighbor_index(side, include_center=False, device=patch.device)
        else:
            idx, mask = self.neighbor_idx.to(patch.device), self.neighbor_mask.to(patch.device)
        safe_idx = idx.clamp_min(0)
        neigh = patch[:, safe_idx.reshape(-1), :].reshape(b, idx.shape[0], idx.shape[1], d)
        center = patch.unsqueeze(2)
        if self.mode == "relative":
            kv_in = self.rel_ln(neigh - center)
        else:
            kv_in = self.rel_ln(neigh)
        q = self.wq(self.center_ln(patch)).unsqueeze(2)
        k = self.wk(kv_in)
        v = self.wv(kv_in)
        logits = (q * k).sum(dim=-1) / math.sqrt(float(self.relation_dim))
        logits = logits.masked_fill(~mask.unsqueeze(0), -1e4)
        attn = torch.softmax(logits, dim=-1)
        ctx = (attn.unsqueeze(-1) * v).sum(dim=2)
        delta = self.fc2(F.gelu(self.fc1(ctx)))
        valid_k = mask.sum(dim=-1).float().clamp_min(1.0).to(patch.device)
        entropy = -(attn * torch.log(attn.clamp_min(1e-8))).sum(dim=-1)
        top1 = attn.max(dim=-1).values
        self.last_stats = {
            "mean_entropy": entropy.detach(),
            "normalized_entropy": (entropy / torch.log(valid_k).unsqueeze(0).clamp_min(1e-6)).detach(),
            "top1_attention_probability": top1.detach(),
            "effective_neighbor_count": torch.exp(entropy).detach(),
            "residual_ratio": (delta.float().norm(dim=-1) / (patch.float().norm(dim=-1) + 1e-6)).detach(),
        }
        return torch.cat([cls, patch + delta], dim=1)

    def forward(self, x):
        if isinstance(x, list):
            return [self.adapt_nld(x[0].permute(1, 0, 2)).permute(1, 0, 2), x[1]]
        return self.adapt_nld(x)
