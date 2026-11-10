import numpy as np


def pca_basis(x, rank):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    centered = x - x.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    total = float(np.sum(s ** 2))
    eig = (s ** 2).astype(np.float32)
    evr = eig / max(total, 1e-12)
    return vt[: min(rank, vt.shape[0])].astype(np.float32), eig, evr


def random_orthonormal(dim, rank, rng):
    raw = rng.normal(size=(dim, rank)).astype(np.float32)
    q, _ = np.linalg.qr(raw)
    return q[:, :rank].T.astype(np.float32)
