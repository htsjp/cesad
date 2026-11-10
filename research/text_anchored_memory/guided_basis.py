import torch
import torch.nn.functional as F


def top_eig_symmetric(mat, rank):
    try:
        eigvals, eigvecs = torch.linalg.eigh(mat.double())
    except torch._C._LinAlgError:
        jitter = torch.eye(mat.shape[0], dtype=torch.double, device=mat.device) * 1e-8
        eigvals, eigvecs = torch.linalg.eigh(mat.double() + jitter)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order].float()
    eigvecs = eigvecs[:, order].float()
    q, _ = torch.linalg.qr(eigvecs[:, :rank], mode="reduced")
    return q[:, :rank].contiguous(), eigvals


def guided_basis(stats, rank, lam=0.0, rho=1.0, d_text=None, mode="pca", random_seed=None):
    sigma = stats["sigma_bar"].float()
    delta = stats["delta_v"].float()
    if mode == "pca":
        g = None
        scatter = sigma
    elif mode == "label":
        g = F.normalize(delta, dim=0)
        scatter = sigma + float(lam) * torch.outer(g, g)
    elif mode == "text":
        g = F.normalize(d_text.float().to(delta.device), dim=0)
        scatter = sigma + float(lam) * torch.outer(g, g)
    elif mode == "label_text":
        g = F.normalize(delta + float(rho) * d_text.float().to(delta.device), dim=0)
        scatter = sigma + float(lam) * torch.outer(g, g)
    elif mode == "random_text":
        gen = torch.Generator().manual_seed(int(random_seed or 0))
        g = F.normalize(torch.randn(delta.numel(), generator=gen), dim=0)
        scatter = sigma + float(lam) * torch.outer(g, g)
    elif mode == "random_label":
        gen = torch.Generator().manual_seed(int(random_seed or 0))
        rv = F.normalize(torch.randn(delta.numel(), generator=gen), dim=0)
        g = F.normalize(rv + float(rho) * d_text.float().to(delta.device), dim=0)
        scatter = sigma + float(lam) * torch.outer(g, g)
    else:
        raise ValueError(f"unknown basis mode {mode}")
    basis, eigvals = top_eig_symmetric(scatter, rank)
    coeff = stats["centered"].float() @ basis.float()
    mu_hat = stats["mu0"][None, :].float() + coeff @ basis.T
    return {"basis": basis, "eigvals": eigvals, "coeff": coeff, "mu_hat": mu_hat, "guidance": g}


def rotate_basis_equivalent(fit, seed=0):
    k = fit["basis"].shape[1]
    gen = torch.Generator().manual_seed(int(seed))
    rot, _ = torch.linalg.qr(torch.randn(k, k, generator=gen), mode="reduced")
    basis = fit["basis"] @ rot
    coeff = fit["coeff"] @ rot
    mu_hat = fit["mu_hat"].clone()
    return {**fit, "basis": basis, "coeff": coeff, "mu_hat": mu_hat}
