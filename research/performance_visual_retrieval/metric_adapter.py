import torch
import torch.nn as nn
import torch.nn.functional as F


def random_orthonormal(dim, rank, seed=0):
    gen = torch.Generator().manual_seed(int(seed))
    q, _ = torch.linalg.qr(torch.randn(dim, rank, generator=gen), mode="reduced")
    return q[:, :rank].float()


class SymmetricMetricAdapter(nn.Module):
    def __init__(self, dim, rank=8, beta=0.1, seed=0, train_u=True):
        super().__init__()
        self.raw_u = nn.Parameter(random_orthonormal(dim, rank, seed), requires_grad=bool(train_u))
        self.a = nn.Parameter(torch.zeros(rank))
        self.beta = float(beta)

    def orthonormal_u(self):
        q, _ = torch.linalg.qr(self.raw_u.float(), mode="reduced")
        return q[:, : self.a.numel()]

    def eigen_delta(self):
        return self.beta * torch.tanh(self.a.float())

    def transform(self, x):
        u = self.orthonormal_u().to(x.device, x.dtype)
        delta = self.eigen_delta().to(x.device, x.dtype)
        coeff = x @ u
        return x + (coeff * delta) @ u.T

    def normalized(self, x):
        return F.normalize(self.transform(x.float()), dim=-1)

    def eigenvalues(self):
        eig = torch.ones(self.raw_u.shape[0])
        eig[: self.a.numel()] = 1.0 + self.eigen_delta().detach().cpu()
        return eig

    def regularization(self, lambda_identity=1e-3):
        delta = self.eigen_delta()
        eig = 1.0 + delta
        identity = delta.pow(2).sum()
        below = F.relu(0.8 - eig).pow(2).sum()
        above = F.relu(eig - 1.2).pow(2).sum()
        return float(lambda_identity) * identity + below + above


class DiagonalPositiveMetric(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.d = nn.Parameter(torch.zeros(dim))

    def transform(self, x):
        scale = torch.exp(self.d.float().clamp(-0.2, 0.2)).to(x.device, x.dtype)
        return x.float() * scale

    def normalized(self, x):
        return F.normalize(self.transform(x), dim=-1)

    def eigenvalues(self):
        return torch.exp(self.d.detach().float().clamp(-0.2, 0.2))

    def regularization(self, lambda_identity=1e-3):
        eig = torch.exp(self.d.float().clamp(-0.2, 0.2))
        return float(lambda_identity) * (eig - 1).pow(2).sum()


class IdentityMetric(nn.Module):
    def normalized(self, x):
        return F.normalize(x.float(), dim=-1)

    def transform(self, x):
        return x.float()

    def eigenvalues(self):
        return torch.ones(1)

    def regularization(self, lambda_identity=1e-3):
        return torch.tensor(0.0)


class RandomFrozenPSD(SymmetricMetricAdapter):
    def __init__(self, dim, rank=8, beta=0.1, seed=0):
        super().__init__(dim, rank=rank, beta=beta, seed=seed, train_u=False)
        gen = torch.Generator().manual_seed(int(seed) + 991)
        with torch.no_grad():
            self.a.copy_(torch.randn(rank, generator=gen) * 0.5)
        self.a.requires_grad_(False)


class RandomOrthogonalRotation(nn.Module):
    def __init__(self, dim, seed=0):
        super().__init__()
        q = random_orthonormal(dim, dim, seed)
        self.register_buffer("q", q)

    def transform(self, x):
        return x.float() @ self.q.to(x.device, x.dtype)

    def normalized(self, x):
        return F.normalize(self.transform(x), dim=-1)

    def eigenvalues(self):
        return torch.ones(self.q.shape[0])

    def regularization(self, lambda_identity=1e-3):
        return torch.tensor(0.0)


def diagnostics(adapter, sample):
    with torch.no_grad():
        eig = adapter.eigenvalues().float()
        z = adapter.normalized(sample.float())
        s = torch.linalg.svdvals(z[: min(512, z.shape[0])])
        eff_rank = (s.sum().pow(2) / s.pow(2).sum().clamp_min(1e-12)).item() if s.numel() else 0.0
        cos = z[: min(256, z.shape[0])] @ z[: min(256, z.shape[0])].T
        return {
            "eig_min": float(eig.min()),
            "eig_max": float(eig.max()),
            "condition_number": float(eig.max() / eig.min().clamp_min(1e-12)),
            "effective_rank": float(eff_rank),
            "mean_pairwise_cosine": float(cos.mean()) if cos.numel() else 0.0,
            "feature_collapse_rate": float((s < 1e-6).float().mean()) if s.numel() else 0.0,
        }

