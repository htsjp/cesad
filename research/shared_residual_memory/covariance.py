import torch


def _pad_orthonormal_columns(basis: torch.Tensor, rank: int) -> torch.Tensor:
    if basis.shape[1] >= rank:
        return basis[:, :rank]
    dim = basis.shape[0]
    eye = torch.eye(dim, dtype=basis.dtype, device=basis.device)
    candidates = eye if basis.numel() == 0 else torch.cat([basis, eye], dim=1)
    q, _ = torch.linalg.qr(candidates, mode="reduced")
    return q[:, :rank]


def covariance_pca(x: torch.Tensor, max_rank: int):
    """Deterministic FP64 covariance eigensolver for shared residual bases."""
    max_rank = int(max_rank)
    xd = x.double()
    cov = (xd.T @ xd) / max(1, xd.shape[0] - 1)
    try:
        eigvals, eigvecs = torch.linalg.eigh(cov)
    except torch._C._LinAlgError:
        jitter = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device) * 1e-8
        try:
            eigvals, eigvecs = torch.linalg.eigh(cov + jitter)
        except torch._C._LinAlgError:
            scale = max(1, xd.shape[0] - 1) ** 0.5
            try:
                _, singular, vh = torch.linalg.svd(xd / scale, full_matrices=False)
                eigvals = singular.pow(2)
                eigvecs = vh.T
            except torch._C._LinAlgError:
                q, _ = torch.linalg.qr(torch.nan_to_num(xd.T, nan=0.0), mode="reduced")
                eigvecs = q
                eigvals = torch.zeros(q.shape[1], dtype=xd.dtype, device=xd.device)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order].float()
    eigvecs = eigvecs[:, order].float()
    eigvecs = _pad_orthonormal_columns(eigvecs, max_rank)
    if eigvals.numel() < cov.shape[0]:
        eigvals = torch.cat([eigvals, torch.zeros(cov.shape[0] - eigvals.numel(), dtype=eigvals.dtype, device=eigvals.device)])
    return eigvecs[:, :max_rank].contiguous(), eigvals[:max_rank].contiguous(), eigvals.contiguous()


def check_orthonormal(basis: torch.Tensor):
    eye = torch.eye(basis.shape[1], dtype=basis.dtype, device=basis.device)
    return float((basis.T @ basis - eye).abs().max().item())
