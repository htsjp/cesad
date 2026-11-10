import math
import time

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def gini(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or np.allclose(x.sum(), 0):
        return 0.0
    x = np.sort(np.maximum(x, 0))
    n = x.size
    return float((2 * np.arange(1, n + 1).dot(x) / (n * x.sum())) - (n + 1) / n)


def entropy_from_counts(x):
    x = np.asarray(x, dtype=np.float64)
    total = x.sum()
    if total <= 0:
        return 0.0
    p = x[x > 0] / total
    return float(-(p * np.log(p)).sum())


def transition_entropy(transition: sparse.csr_matrix):
    csr = transition.tocsr()
    ent = []
    for i in range(csr.shape[0]):
        row = csr.data[csr.indptr[i]:csr.indptr[i + 1]]
        row = row[row > 0]
        ent.append(float(-(row * np.log(row + 1e-12)).sum()) if row.size else 0.0)
    return float(np.mean(ent)), float(np.median(ent))


def graph_diagnostics(name: str, adjacency: sparse.csr_matrix, transition: sparse.csr_matrix, labels=None, metadata=None):
    start = time.perf_counter()
    adjacency = adjacency.tocsr()
    transition = transition.tocsr()
    n = adjacency.shape[0]
    out_degree = np.diff(adjacency.indptr)
    weak_n, comps = connected_components(adjacency, directed=True, connection="weak")
    largest = np.bincount(comps).max() if comps.size else 0
    reciprocal = adjacency.multiply(adjacency.T)
    reciprocity = float(reciprocal.nnz / max(1, adjacency.nnz))
    raw_data = adjacency.data.astype(np.float64)
    tau = float((metadata or {}).get("tau_graph", 0.0))
    if tau > 0 and raw_data.size and raw_data.max() > 1.0:
        sim_data = np.log(np.maximum(raw_data, 1e-30)) * tau
    else:
        sim_data = raw_data
    row_ent_mean, row_ent_median = transition_entropy(transition)
    row = {
        "graph": name,
        "node_count": int(n),
        "edge_count": int(adjacency.nnz),
        "isolated_nodes": int((out_degree == 0).sum()),
        "connected_components": int(weak_n),
        "largest_component_ratio": float(largest / max(1, n)),
        "mean_degree": float(out_degree.mean()),
        "median_degree": float(np.median(out_degree)),
        "degree_gini": gini(out_degree),
        "degree_entropy": entropy_from_counts(out_degree),
        "reciprocity": reciprocity,
        "edge_similarity_mean": float(sim_data.mean()) if sim_data.size else 0.0,
        "edge_similarity_std": float(sim_data.std()) if sim_data.size else 0.0,
        "transition_entropy_mean": row_ent_mean,
        "transition_entropy_median": row_ent_median,
        "near_duplicate_edge_ratio_095": float((sim_data > 0.95).mean()) if sim_data.size else 0.0,
        "near_duplicate_edge_ratio_099": float((sim_data > 0.99).mean()) if sim_data.size else 0.0,
        "diagnostic_time_sec": time.perf_counter() - start,
    }
    if labels is not None:
        labels = np.asarray(labels)
        coo = adjacency.tocoo()
        src = labels[coo.row]
        dst = labels[coo.col]
        if coo.nnz:
            row.update({
                "edge_label_homophily": float((src == dst).mean()),
                "normal_to_normal_edge_ratio": float(((src == 0) & (dst == 0)).mean()),
                "anomaly_to_anomaly_edge_ratio": float(((src == 1) & (dst == 1)).mean()),
                "normal_to_anomaly_edge_ratio": float(((src == 0) & (dst == 1)).mean()),
                "anomaly_to_normal_edge_ratio": float(((src == 1) & (dst == 0)).mean()),
            })
        else:
            row.update({
                "edge_label_homophily": 0.0,
                "normal_to_normal_edge_ratio": 0.0,
                "anomaly_to_anomaly_edge_ratio": 0.0,
                "normal_to_anomaly_edge_ratio": 0.0,
                "anomaly_to_normal_edge_ratio": 0.0,
            })
    return row


def distribution_diagnostics(w0, w, labels):
    eps = 1e-12
    w0 = w0.float().reshape(-1, w0.shape[-1])
    w = w.float().reshape(-1, w.shape[-1])
    tv = 0.5 * (w0 - w).abs().sum(dim=-1)
    kl01 = (w0 * ((w0 + eps).log() - (w + eps).log())).sum(dim=-1)
    kl10 = (w * ((w + eps).log() - (w0 + eps).log())).sum(dim=-1)
    cos = torch_cosine(w0, w)
    ent = -(w * (w + eps).log()).sum(dim=-1)
    top = torch_top_stats(w)
    labels_t = labels.to(w.device).long()
    anomaly_mass = w[:, labels_t == 1].sum(dim=-1)
    normal_mass = w[:, labels_t == 0].sum(dim=-1)
    return {
        "tv_mean": float(tv.mean().item()),
        "kl_w0_w_mean": float(kl01.mean().item()),
        "kl_w_w0_mean": float(kl10.mean().item()),
        "retrieval_cosine_mean": float(cos.mean().item()),
        "max_weight_change_mean": float((w0 - w).abs().max(dim=-1).values.mean().item()),
        "entropy_mean": float(ent.mean().item()),
        "normalized_entropy_mean": float((ent / math.log(max(2, w.shape[-1]))).mean().item()),
        "effective_support_mean": float(torch_exp(ent).mean().item()),
        "normal_mass_mean": float(normal_mass.mean().item()),
        "anomaly_mass_mean": float(anomaly_mass.mean().item()),
        "anomaly_minus_normal_margin_mean": float((anomaly_mass - normal_mass).mean().item()),
        **top,
    }


def torch_cosine(a, b):
    import torch
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def torch_exp(x):
    import torch
    return torch.exp(x)


def torch_top_stats(w):
    import torch
    vals = torch.topk(w, k=min(20, w.shape[-1]), dim=-1).values
    return {
        "top1_weight_mean": float(vals[:, 0].mean().item()),
        "top5_cumulative_weight_mean": float(vals[:, :min(5, vals.shape[1])].sum(dim=-1).mean().item()),
        "top20_cumulative_weight_mean": float(vals.sum(dim=-1).mean().item()),
    }
