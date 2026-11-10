import numpy as np
from scipy import sparse

from .graph_normalization import row_normalize


def identity_graph(n: int) -> sparse.csr_matrix:
    return sparse.eye(int(n), dtype=np.float32, format="csr")


def uniform_mixed_metadata(n: int):
    return {"node_count": int(n), "edge_count": int(n) * int(n), "implemented_dense": False}


def random_degree_matched_graph(reference: sparse.csr_matrix, seed: int) -> sparse.csr_matrix:
    rng = np.random.default_rng(int(seed))
    ref = reference.tocsr()
    n = ref.shape[0]
    indptr = [0]
    indices = []
    data = []
    weights = ref.data.copy()
    rng.shuffle(weights)
    pos = 0
    for i in range(n):
        degree = ref.indptr[i + 1] - ref.indptr[i]
        if degree:
            choices = rng.choice(n - 1, size=degree, replace=degree > n - 1)
            choices = np.where(choices >= i, choices + 1, choices)
            indices.extend(choices.tolist())
            data.extend(weights[pos:pos + degree].tolist())
            pos += degree
        indptr.append(len(indices))
    return sparse.csr_matrix((np.asarray(data, np.float32), np.asarray(indices, np.int32), np.asarray(indptr, np.int32)), shape=ref.shape)


def degree_preserving_edge_shuffle(reference: sparse.csr_matrix, seed: int) -> sparse.csr_matrix:
    rng = np.random.default_rng(int(seed))
    coo = reference.tocoo()
    cols = coo.col.copy()
    rng.shuffle(cols)
    data = coo.data.copy()
    return sparse.coo_matrix((data, (coo.row, cols)), shape=reference.shape, dtype=np.float32).tocsr()


def edge_weight_permutation(reference: sparse.csr_matrix, seed: int) -> sparse.csr_matrix:
    rng = np.random.default_rng(int(seed))
    out = reference.copy().tocsr()
    data = out.data.copy()
    rng.shuffle(data)
    out.data = data.astype(np.float32)
    return out


def uniform_neighbor_weights(reference: sparse.csr_matrix) -> sparse.csr_matrix:
    out = reference.copy().tocsr()
    out.data[:] = 1.0
    return out.astype(np.float32)


def filter_edges_by_label(reference: sparse.csr_matrix, labels: np.ndarray, same_label: bool) -> sparse.csr_matrix:
    coo = reference.tocoo()
    src = labels[coo.row]
    dst = labels[coo.col]
    mask = (src == dst) if same_label else (src != dst)
    return sparse.coo_matrix((coo.data[mask], (coo.row[mask], coo.col[mask])), shape=reference.shape, dtype=np.float32).tocsr()


def make_control_transition(reference: sparse.csr_matrix, control: str, labels=None, seed: int = 0):
    n = reference.shape[0]
    if control == "identity":
        return identity_graph(n)
    if control == "random_degree_matched":
        return row_normalize(random_degree_matched_graph(reference, seed), identity_fallback=True)
    if control == "degree_preserving_shuffle":
        return row_normalize(degree_preserving_edge_shuffle(reference, seed), identity_fallback=True)
    if control == "edge_weight_permutation":
        return row_normalize(edge_weight_permutation(reference, seed), identity_fallback=True)
    if control == "uniform_weights":
        return row_normalize(uniform_neighbor_weights(reference), identity_fallback=True)
    if control == "same_label_oracle":
        if labels is None:
            raise ValueError("labels are required for same_label_oracle")
        return row_normalize(filter_edges_by_label(reference, labels, True), identity_fallback=True)
    if control == "cross_label_only":
        if labels is None:
            raise ValueError("labels are required for cross_label_only")
        return row_normalize(filter_edges_by_label(reference, labels, False), identity_fallback=True)
    raise ValueError(f"Unknown control={control}")

