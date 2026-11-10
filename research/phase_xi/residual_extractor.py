import numpy as np


def l2_normalize(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


def remove_subspace(x, basis):
    x = np.asarray(x, dtype=np.float32)
    if basis is None or basis.size == 0:
        return x
    return x - (x @ basis.T) @ basis


def residual(x, mu, normal_basis=None):
    centered = np.asarray(x, dtype=np.float32) - np.asarray(mu, dtype=np.float32)
    return l2_normalize(remove_subspace(centered, normal_basis))
