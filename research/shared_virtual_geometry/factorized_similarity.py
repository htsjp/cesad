import torch
import torch.nn.functional as F


def coefficients_from_geometry(scales, labels, geoms):
    labels = torch.as_tensor(labels).long().to(scales.device)
    return scales[:, None, None].float() * geoms[labels].float()


def reconstruct_keys(mu_hat, residual_basis, coeff, normalize=True):
    raw = mu_hat[:, None, :].float() + coeff.float() @ residual_basis.float().T
    if normalize:
        raw = F.normalize(raw, dim=-1)
    return raw.reshape(-1, raw.shape[-1])


def equivalence_report(mu_hat, residual_basis, coeff, query):
    raw = mu_hat[:, None, :].float() + coeff.float() @ residual_basis.float().T
    norms = raw.norm(dim=-1).clamp_min(1e-12)
    mat = (raw / norms[:, :, None]).reshape(-1, raw.shape[-1])
    q = F.normalize(query.float(), dim=-1)
    sim_mat = q @ mat.T
    sim_fact = (q @ raw.reshape(-1, raw.shape[-1]).T) / norms.reshape(-1)[None, :]
    diff = (sim_mat - sim_fact).abs().max()
    return {"similarity_max_abs_diff": float(diff), "passed": bool(diff <= 1e-6)}

