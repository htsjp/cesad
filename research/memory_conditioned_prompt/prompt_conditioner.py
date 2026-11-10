import torch
import torch.nn as nn


class MemoryPromptConditioner(nn.Module):
    def __init__(self, visual_dim=768, text_width=768, scalar_dim=18, hidden=16):
        super().__init__()
        self.context_down = nn.Linear(visual_dim, hidden, bias=False)
        self.context_up = nn.Linear(hidden, text_width, bias=False)
        self.scalar_gate = nn.Sequential(nn.Linear(scalar_dim, hidden), nn.GELU(), nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, r_img, u_img, beta_max=0.1, force_zero=False):
        raw = self.context_up(self.context_down(r_img.float()))
        gate = self.scalar_gate(u_img.float()).reshape(())
        bias = gate * raw
        norm = bias.norm().clamp_min(1e-12)
        bias = bias / torch.maximum(torch.ones_like(norm), norm / float(beta_max))
        if force_zero:
            bias = torch.zeros_like(bias)
        return bias, gate


class DirectVisualResidual(nn.Module):
    def __init__(self, visual_dim=768, hidden=16):
        super().__init__()
        self.down = nn.Linear(visual_dim, hidden, bias=False)
        self.up = nn.Linear(hidden, visual_dim, bias=False)

    def forward(self, r_img, beta_max=0.1):
        out = self.up(self.down(r_img.float()))
        norm = out.norm().clamp_min(1e-12)
        return out / torch.maximum(torch.ones_like(norm), norm / float(beta_max))

