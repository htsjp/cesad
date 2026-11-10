import torch


class LowRankBranchAdapter(torch.nn.Module):
    def __init__(self, width=1024, bottleneck=32):
        super().__init__()
        self.norm = torch.nn.LayerNorm(width)
        self.down = torch.nn.Linear(width, bottleneck)
        self.act = torch.nn.GELU()
        self.up = torch.nn.Linear(bottleneck, width)
        self.gate_raw = torch.nn.Parameter(torch.tensor(-2.944439))
        torch.nn.init.zeros_(self.up.weight)
        torch.nn.init.zeros_(self.up.bias)

    def forward(self, x):
        gate = torch.clamp(torch.sigmoid(self.gate_raw) * 0.20, 0.0, 0.20)
        return x + gate * self.up(self.act(self.down(self.norm(x))))


class DualBranchAdapter(torch.nn.Module):
    def __init__(self, width=1024, bottleneck=32):
        super().__init__()
        self.local = LowRankBranchAdapter(width, bottleneck)
        self.global_adapter = LowRankBranchAdapter(width, bottleneck)

    def forward(self, branches):
        if not isinstance(branches, list) or len(branches) != 2:
            return branches
        return [self.local(branches[0]), self.global_adapter(branches[1])]
