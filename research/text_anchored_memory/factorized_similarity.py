import torch


def build_factorized(mu0, basis, coeff, residual_basis, residual_coeff):
    raw = mu0[None, None, :].float() + coeff[:, None, :].float() @ basis.float().T
    raw = raw + residual_coeff.float() @ residual_basis.float().T
    norms = raw.norm(dim=-1).clamp_min(1e-12)
    return {
        "mu0": mu0.float(),
        "basis": basis.float(),
        "coeff": coeff.float(),
        "residual_basis": residual_basis.float(),
        "residual_coeff": residual_coeff.float(),
        "norms": norms.float(),
    }


def similarity(q, fact):
    q = q.float()
    qm = q @ fact["mu0"]
    qb = q @ fact["basis"]
    qr = q @ fact["residual_basis"]
    cent = qb @ fact["coeff"].T
    res = qr @ fact["residual_coeff"].reshape(-1, fact["residual_coeff"].shape[-1]).T
    res = res.reshape(q.shape[0], fact["residual_coeff"].shape[0], fact["residual_coeff"].shape[1])
    sim = (qm[:, None, None] + cent[:, :, None] + res) / fact["norms"][None, :, :]
    return sim.reshape(q.shape[0], -1)


def materialized_from_factorized(fact):
    raw = fact["mu0"][None, None, :] + fact["coeff"][:, None, :] @ fact["basis"].T
    raw = raw + fact["residual_coeff"] @ fact["residual_basis"].T
    return (raw / fact["norms"][:, :, None]).reshape(-1, raw.shape[-1])


def equivalence_report(q, fact):
    mat = materialized_from_factorized(fact)
    sim_fact = similarity(q, fact)
    sim_mat = q.float() @ mat.T
    return {
        "similarity_max_abs_diff": float((sim_fact - sim_mat).abs().max().item()),
        "passed": bool((sim_fact - sim_mat).abs().max().item() <= 1e-6),
    }

