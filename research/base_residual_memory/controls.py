import torch
import torch.nn.functional as F


def random_orthonormal(dim: int, rank: int, generator: torch.Generator) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(dim, rank, generator=generator))
    return q[:, :rank].T.contiguous()


def random_local_subspace(decomp, rank: int, seed: int, normalize: bool = True) -> torch.Tensor:
    gen = torch.Generator().manual_seed(int(seed))
    mu, centered, s = decomp["mu"], decomp["centered"], decomp["S"]
    groups, _, dim = centered.shape
    random = torch.randn(groups, dim, rank, generator=gen, dtype=centered.dtype)
    q, _ = torch.linalg.qr(random, mode="reduced")
    basis = q.transpose(1, 2).contiguous()
    coeff = torch.matmul(centered, basis.transpose(1, 2))
    norm = coeff.norm(dim=1, keepdim=True).clamp_min(1e-12)
    target = s[:, :rank].unsqueeze(1)
    coeff = coeff / norm * target
    residual = torch.matmul(coeff, basis)
    out = mu[:, None, :] + residual
    if normalize:
        out = F.normalize(out, dim=-1)
    return out.reshape(-1, out.shape[-1])


def basis_shuffle(decomp, labels, rank: int, seed: int, normalize: bool = True) -> torch.Tensor:
    gen = torch.Generator().manual_seed(int(seed))
    mu, u, s, vh = decomp["mu"], decomp["U"], decomp["S"], decomp["Vh"]
    labels = torch.as_tensor(labels)
    outs = []
    for label in sorted(set(int(x) for x in labels.tolist())):
        idx = torch.nonzero(labels == label, as_tuple=False).squeeze(1)
        perm = idx[torch.randperm(idx.numel(), generator=gen)]
        for src, basis_src in zip(idx.tolist(), perm.tolist()):
            coeff = u[src, :, :rank] * s[src, :rank][None, :]
            residual = coeff @ vh[basis_src, :rank, :]
            outs.append((src, mu[src : src + 1, None, :] + residual.unsqueeze(0)))
    outs = [x for _, x in sorted(outs, key=lambda z: z[0])]
    out = torch.cat(outs, dim=0)
    if normalize:
        out = F.normalize(out, dim=-1)
    return out.reshape(-1, out.shape[-1])
