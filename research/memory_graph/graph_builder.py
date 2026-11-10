import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse

from .graph_normalization import row_normalize


def tensor_checksum(x: torch.Tensor) -> str:
    arr = x.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def graph_cache_key(params: dict, checksum: str) -> str:
    text = json.dumps({"params": params, "checksum": checksum}, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@torch.no_grad()
def exact_knn(features: torch.Tensor, k: int, chunk_size: int = 512, device: str = "cuda:0"):
    x = F.normalize(features.detach().float(), dim=-1).to(device)
    n = x.shape[0]
    vals_all, idx_all = [], []
    for start in range(0, n, int(chunk_size)):
        end = min(n, start + int(chunk_size))
        sim = x[start:end] @ x.T
        rows = torch.arange(start, end, device=device) - start
        sim[rows, torch.arange(start, end, device=device)] = -float("inf")
        vals, idx = torch.topk(sim, k=min(int(k), n - 1), dim=-1)
        vals_all.append(vals.cpu())
        idx_all.append(idx.cpu())
        del sim, vals, idx
    return torch.cat(vals_all, dim=0).numpy().astype(np.float32), torch.cat(idx_all, dim=0).numpy().astype(np.int64)


def knn_to_adjacency(knn_idx: np.ndarray, knn_sim: np.ndarray, tau_graph: float, graph_type: str, rho: float = 1.0, epsilon: float = 1e-6) -> sparse.csr_matrix:
    n, k = knn_idx.shape
    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    cols = knn_idx.reshape(-1)
    sims = knn_sim.reshape(-1).astype(np.float32)
    weights = np.exp(sims / float(tau_graph)).astype(np.float32)
    directed = sparse.coo_matrix((weights, (rows, cols)), shape=(n, n), dtype=np.float32).tocsr()

    if graph_type == "directed":
        return directed
    if graph_type == "symmetric":
        return directed.maximum(directed.T).tocsr()
    if graph_type == "mutual":
        return directed.minimum(directed.T).tocsr()
    if graph_type == "shared":
        neighbor_sets = [set(knn_idx[i].tolist()) for i in range(n)]
        factor = np.empty_like(weights)
        flat_cols = cols
        for e, (i, j) in enumerate(zip(rows, flat_cols)):
            a, b = neighbor_sets[int(i)], neighbor_sets[int(j)]
            union = len(a | b)
            jac = (len(a & b) / union) if union else 0.0
            factor[e] = float((jac + float(epsilon)) ** float(rho))
        shared = sparse.coo_matrix((weights * factor, (rows, cols)), shape=(n, n), dtype=np.float32).tocsr()
        return shared
    if graph_type == "local_scale":
        # Self-tuning kernel in cosine-distance space. Larger similarity means smaller distance.
        dist = np.maximum(0.0, 1.0 - knn_sim.astype(np.float32))
        sigma = np.maximum(dist[:, -1], 1e-6)
        row_sigma = np.repeat(sigma, k)
        col_sigma = sigma[cols]
        scaled = np.exp(-(dist.reshape(-1) ** 2) / np.maximum(row_sigma * col_sigma, 1e-12)).astype(np.float32)
        return sparse.coo_matrix((scaled, (rows, cols)), shape=(n, n), dtype=np.float32).tocsr()
    raise ValueError(f"Unknown graph_type={graph_type}")


def save_graph(path: Path, adjacency: sparse.csr_matrix, transition: sparse.csr_matrix, metadata: dict):
    path.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(path / "adjacency.npz", adjacency.astype(np.float32))
    sparse.save_npz(path / "transition.npz", transition.astype(np.float32))
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_graph(path: Path):
    adjacency = sparse.load_npz(path / "adjacency.npz").tocsr()
    transition = sparse.load_npz(path / "transition.npz").tocsr()
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    return adjacency, transition, metadata


def build_or_load_graph(features: torch.Tensor, params: dict, cache_root: Path, device: str = "cuda:0"):
    cache_root.mkdir(parents=True, exist_ok=True)
    checksum = tensor_checksum(features)
    key = graph_cache_key(params, checksum)
    graph_dir = cache_root / key
    if (graph_dir / "transition.npz").exists():
        adjacency, transition, metadata = load_graph(graph_dir)
        metadata["cache_hit"] = True
        return adjacency, transition, metadata

    start = time.perf_counter()
    knn_sim, knn_idx = exact_knn(features, int(params["k_graph"]), int(params.get("chunk_size", 512)), device)
    search_time = time.perf_counter() - start
    graph_start = time.perf_counter()
    adjacency = knn_to_adjacency(
        knn_idx,
        knn_sim,
        float(params["tau_graph"]),
        params["graph_type"],
        rho=float(params.get("rho", 1.0)),
        epsilon=float(params.get("epsilon", 1e-6)),
    )
    transition = row_normalize(adjacency, identity_fallback=True)
    graph_time = time.perf_counter() - graph_start
    metadata = {
        "graph_cache_key": key,
        "graph_type": params["graph_type"],
        "k_graph": int(params["k_graph"]),
        "tau_graph": float(params["tau_graph"]),
        "rho": float(params.get("rho", 1.0)),
        "node_count": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "feature_checksum": checksum,
        "node_checksum": checksum,
        "edge_count": int(adjacency.nnz),
        "exact_search": True,
        "approximate_search": False,
        "neighbor_search_time_sec": search_time,
        "graph_construction_time_sec": graph_time,
        "cache_hit": False,
    }
    save_graph(graph_dir, adjacency, transition, metadata)
    return adjacency, transition, metadata

