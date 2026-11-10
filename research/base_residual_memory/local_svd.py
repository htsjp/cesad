from typing import Dict, List

import torch
import torch.nn.functional as F


def grouped_matrix(keys: torch.Tensor, groups: List[Dict]) -> torch.Tensor:
    mats = []
    for g in groups:
        mats.append(keys[torch.tensor(g["positions"], dtype=torch.long)])
    return torch.stack(mats, dim=0)


def decompose_groups(keys: torch.Tensor, groups: List[Dict]) -> Dict[str, torch.Tensor]:
    m = grouped_matrix(keys.float(), groups)
    mu = m.mean(dim=1)
    centered = m - mu[:, None, :]
    u, s, vh = torch.linalg.svd(centered, full_matrices=False)
    return {"M": m, "mu": mu, "centered": centered, "U": u, "S": s, "Vh": vh}


def reconstruct_rank(decomp: Dict[str, torch.Tensor], rank: int, normalize: bool = True) -> torch.Tensor:
    mu, u, s, vh = decomp["mu"], decomp["U"], decomp["S"], decomp["Vh"]
    if rank <= 0:
        out = mu[:, None, :].expand(-1, 4, -1).clone()
    else:
        uk = u[:, :, :rank]
        sk = s[:, :rank]
        vhk = vh[:, :rank, :]
        residual = torch.matmul(uk * sk[:, None, :], vhk)
        out = mu[:, None, :] + residual
    if normalize:
        out = F.normalize(out, dim=-1)
    return out.reshape(-1, out.shape[-1])


def energy_threshold_ranks(decomp: Dict[str, torch.Tensor], threshold: float) -> torch.Tensor:
    s2 = decomp["S"].pow(2)
    total = s2.sum(dim=1).clamp_min(1e-12)
    csum = s2.cumsum(dim=1) / total[:, None]
    ranks = torch.zeros((s2.shape[0],), dtype=torch.long)
    for i in range(s2.shape[0]):
        idx = torch.nonzero(csum[i] >= float(threshold), as_tuple=False)
        ranks[i] = int(idx[0].item() + 1) if idx.numel() else min(3, s2.shape[1])
    return ranks.clamp(max=3)


def reconstruct_energy_threshold(decomp: Dict[str, torch.Tensor], threshold: float, normalize: bool = True) -> torch.Tensor:
    ranks = energy_threshold_ranks(decomp, threshold)
    outs = []
    for i, r in enumerate(ranks.tolist()):
        sub = {k: v[i : i + 1] for k, v in decomp.items() if isinstance(v, torch.Tensor)}
        outs.append(reconstruct_rank(sub, int(r), normalize=normalize).reshape(1, 4, -1))
    out = torch.cat(outs, dim=0)
    return out.reshape(-1, out.shape[-1]), ranks

