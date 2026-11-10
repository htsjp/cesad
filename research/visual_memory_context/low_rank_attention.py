import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankContextAdapter(nn.Module):
    def __init__(self, dim=768, rank=8, stat_dim=8, use_gate=True):
        super().__init__()
        self.rank = int(rank)
        self.use_gate = bool(use_gate)
        self.a_q = nn.Linear(dim, rank, bias=False)
        self.a_k = nn.Linear(dim, rank, bias=False)
        self.a_v = nn.Linear(dim, rank, bias=False)
        self.b_o = nn.Linear(rank, dim, bias=False)
        self.gate = nn.Sequential(nn.Linear(stat_dim, 16), nn.GELU(), nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, q0, context, stats, alpha=0.1, gate_override=None, permute_attention=False):
        q_lat = self.a_q(q0.float())
        k_lat = self.a_k(context.float())
        v_lat = self.a_v(context.float())
        attn = (q_lat[:, None, :] * k_lat).sum(dim=-1) / math.sqrt(float(self.rank))
        attn = attn.softmax(dim=-1)
        if permute_attention:
            attn = attn.roll(shifts=1, dims=-1)
        h = (attn[:, :, None] * v_lat).sum(dim=1)
        delta = self.b_o(h)
        if gate_override is None:
            gate = self.gate(stats.float()) if self.use_gate else torch.ones(q0.shape[0], 1, device=q0.device)
        else:
            gate = torch.full((q0.shape[0], 1), float(gate_override), device=q0.device)
        q1 = F.normalize(q0.float() + float(alpha) * gate * delta, dim=-1)
        return q1, delta, gate, attn


class DirectResidualMLP(nn.Module):
    def __init__(self, dim=768, rank=8):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)

    def forward(self, c_d):
        return self.up(self.down(c_d.float()))


def trainable_parameter_count(module):
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))

