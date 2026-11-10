import numpy as np
from scipy import sparse


def row_normalize(mat: sparse.spmatrix, identity_fallback: bool = True) -> sparse.csr_matrix:
    mat = mat.tocsr().astype(np.float32)
    row_sum = np.asarray(mat.sum(axis=1)).reshape(-1)
    isolated = row_sum <= 0
    safe = row_sum.copy()
    safe[isolated] = 1.0
    inv = sparse.diags(1.0 / safe, format="csr")
    out = inv @ mat
    if identity_fallback and isolated.any():
        out = out.tolil()
        rows = np.flatnonzero(isolated)
        out[rows, rows] = 1.0
        out = out.tocsr()
    return out.astype(np.float32)


def add_self_loops(mat: sparse.spmatrix, eta: float) -> sparse.csr_matrix:
    n = mat.shape[0]
    return (mat.tocsr() + sparse.eye(n, dtype=np.float32, format="csr") * float(eta)).tocsr()


def row_sums(mat: sparse.spmatrix) -> np.ndarray:
    return np.asarray(mat.tocsr().sum(axis=1)).reshape(-1)

