from ultra_compact_memory.shared_rank2 import reconstruct


def virtual_keys(mu_hat, residual):
    return reconstruct(mu_hat, residual["basis"], residual["coeff"], normalize=True)

