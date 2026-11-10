from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from graph_memory import GraphMemory
from graph_memory.active_router import ActiveGraphRetriever


RELATION_GROUPS = {
    "R0_random": "random",
    "R1_same_cluster": "same_cluster",
    "R2_same_spatial_bin": "same_spatial_bin",
    "R3_cross_scale": "cross_scale",
    "R4_same_source_category": "same_source_category",
    "R5_same_source_image": "same_source_image",
    "R6_normal_contrast": "normal_contrast",
    "R7_anomaly_contrast": "anomaly_contrast",
    "R8_paired_normal_anomaly": "normal_contrast",
}


@dataclass
class PatchAuditOutput:
    scores: torch.Tensor
    selected_ids: List[List[int]]
    fallback_types: List[str]
    relations: List[List[str]]
    normal_counts: List[int]
    anomaly_counts: List[int]


def finite_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, s) * 100.0)


def finite_ap(labels: Sequence[int], scores: Sequence[float]) -> float:
    from sklearn.metrics import average_precision_score

    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    return float(average_precision_score(y, s) * 100.0)


def gini_from_ids(ids: Iterable[int]) -> float:
    counts = np.asarray(list(Counter(int(x) for x in ids).values()), dtype=np.float64)
    if counts.size == 0 or counts.sum() == 0:
        return 0.0
    counts = np.sort(counts)
    n = counts.size
    return float((2 * np.arange(1, n + 1).dot(counts) / counts.sum() - (n + 1)) / n)


def entropy_from_ids(ids: Iterable[int]) -> float:
    counts = np.asarray(list(Counter(int(x) for x in ids).values()), dtype=np.float64)
    if counts.size == 0 or counts.sum() == 0:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log(np.clip(p, 1e-12, None))).sum())


def selected_label_counts(selected: Sequence[int], labels: torch.Tensor) -> tuple[int, int]:
    vals = labels[list(selected)] if selected else torch.empty(0, dtype=torch.long)
    normal = int((vals == 0).sum().item())
    anomaly = int((vals == 1).sum().item())
    return normal, anomaly


def topk_direct(query: torch.Tensor, memory: GraphMemory, k: int, temperature: float = 1.0) -> PatchAuditOutput:
    sim = F.normalize(query.float().cpu(), dim=-1) @ memory.features.T
    labels = memory.values.argmax(dim=-1)
    scores, selected, fallback, relations, n_counts, a_counts = [], [], [], [], [], []
    for row in range(sim.shape[0]):
        idx = torch.topk(sim[row], k=min(k, sim.shape[1])).indices.tolist()
        w = torch.softmax(sim[row, idx] / float(temperature), dim=-1)
        score = float((w @ memory.values[idx].float())[1])
        n, a = selected_label_counts(idx, labels)
        scores.append(score)
        selected.append([int(x) for x in idx])
        fallback.append("none")
        relations.append(["fixed_topk"])
        n_counts.append(n)
        a_counts.append(a)
    return PatchAuditOutput(torch.tensor(scores), selected, fallback, relations, n_counts, a_counts)


def balanced_topk(query: torch.Tensor, memory: GraphMemory, normal_k: int, anomaly_k: int, temperature: float = 1.0) -> PatchAuditOutput:
    sim = F.normalize(query.float().cpu(), dim=-1) @ memory.features.T
    labels = memory.values.argmax(dim=-1)
    normal_ids = torch.where(labels == 0)[0]
    anomaly_ids = torch.where(labels == 1)[0]
    scores, selected, fallback, relations, n_counts, a_counts = [], [], [], [], [], []
    for row in range(sim.shape[0]):
        n_idx = normal_ids[torch.topk(sim[row, normal_ids], k=min(normal_k, normal_ids.numel())).indices].tolist() if normal_k else []
        a_idx = anomaly_ids[torch.topk(sim[row, anomaly_ids], k=min(anomaly_k, anomaly_ids.numel())).indices].tolist() if anomaly_k else []
        idx = [int(x) for x in n_idx + a_idx]
        w = torch.softmax(sim[row, idx] / float(temperature), dim=-1)
        score = float((w @ memory.values[idx].float())[1])
        n, a = selected_label_counts(idx, labels)
        scores.append(score)
        selected.append(idx)
        fallback.append("none")
        relations.append([f"balanced_{normal_k}_{anomaly_k}"])
        n_counts.append(n)
        a_counts.append(a)
    return PatchAuditOutput(torch.tensor(scores), selected, fallback, relations, n_counts, a_counts)


def adaptive_balanced_topk(query: torch.Tensor, memory: GraphMemory, k: int, temperature: float = 1.0) -> PatchAuditOutput:
    sim = F.normalize(query.float().cpu(), dim=-1) @ memory.features.T
    labels = memory.values.argmax(dim=-1)
    normal_ids = torch.where(labels == 0)[0]
    anomaly_ids = torch.where(labels == 1)[0]
    scores, selected, fallback, relations, n_counts, a_counts = [], [], [], [], [], []
    for row in range(sim.shape[0]):
        n_best = torch.topk(sim[row, normal_ids], k=1).values[0]
        a_best = torch.topk(sim[row, anomaly_ids], k=1).values[0]
        margin = float(a_best - n_best)
        anomaly_k = 6 if margin > 0.02 else 2 if margin < -0.02 else 4
        normal_k = int(k) - anomaly_k
        n_idx = normal_ids[torch.topk(sim[row, normal_ids], k=min(normal_k, normal_ids.numel())).indices].tolist()
        a_idx = anomaly_ids[torch.topk(sim[row, anomaly_ids], k=min(anomaly_k, anomaly_ids.numel())).indices].tolist()
        idx = [int(x) for x in n_idx + a_idx]
        w = torch.softmax(sim[row, idx] / float(temperature), dim=-1)
        score = float((w @ memory.values[idx].float())[1])
        n, a = selected_label_counts(idx, labels)
        scores.append(score)
        selected.append(idx)
        fallback.append("none")
        relations.append(["adaptive_balance"])
        n_counts.append(n)
        a_counts.append(a)
    return PatchAuditOutput(torch.tensor(scores), selected, fallback, relations, n_counts, a_counts)


def active_g5(query: torch.Tensor, memory: GraphMemory, cfg: Dict) -> PatchAuditOutput:
    out = ActiveGraphRetriever(cfg).retrieve(query, memory)
    labels = memory.values.argmax(dim=-1)
    fallback_by_row, n_counts, a_counts = [], [], []
    for idx in out.selected_ids:
        n, a = selected_label_counts(idx, labels)
        n_counts.append(n)
        a_counts.append(a)
        if n == 0 and a == 0:
            fallback_by_row.append("missing_both")
        elif n == 0:
            fallback_by_row.append("missing_normal")
        elif a == 0:
            fallback_by_row.append("missing_anomaly")
        else:
            fallback_by_row.append("none")
    return PatchAuditOutput(out.logits[:, 1], out.selected_ids, fallback_by_row, out.relations, n_counts, a_counts)


def relation_only(query: torch.Tensor, memory: GraphMemory, relation_name: str, cfg: Dict) -> PatchAuditOutput:
    if relation_name == "random":
        mode = "random_neighbors"
        return active_g5(query, memory, {**cfg, "mode": mode})
    sim = F.normalize(query.float().cpu(), dim=-1) @ memory.features.T
    labels = memory.values.argmax(dim=-1)
    k1 = int(cfg.get("k1", 4))
    k2 = int(cfg.get("k2", 8))
    max_candidates = int(cfg.get("max_candidates", 32))
    temperature = float(cfg.get("temperature", 1.0))
    scores, selected, fallback, relations, n_counts, a_counts = [], [], [], [], [], []
    for row in range(sim.shape[0]):
        seeds = torch.topk(sim[row], k=min(k1, sim.shape[1])).indices.tolist()
        rels = ["normal_contrast", "anomaly_contrast"] if relation_name == "paired_normal_anomaly" else [relation_name]
        candidates = memory.relation_index.neighbors(seeds, rels, max_candidates)
        idx = []
        seen = set()
        for value in seeds + candidates:
            if int(value) in seen:
                continue
            seen.add(int(value))
            idx.append(int(value))
        idx = sorted(idx, key=lambda x: float(sim[row, x]), reverse=True)[:k2]
        w = torch.softmax(sim[row, idx] / temperature, dim=-1)
        score = float((w @ memory.values[idx].float())[1])
        n, a = selected_label_counts(idx, labels)
        scores.append(score)
        selected.append(idx)
        fallback.append("none" if n and a else "missing_normal" if not n else "missing_anomaly")
        relations.append(rels)
        n_counts.append(n)
        a_counts.append(a)
    return PatchAuditOutput(torch.tensor(scores), selected, fallback, relations, n_counts, a_counts)


def summarize_output(dataset: str, group: str, labels: Sequence[int], output: PatchAuditOutput, seed: int, memory_budget: int, latency_ms: float) -> Dict:
    flat_ids = [x for row in output.selected_ids for x in row]
    return {
        "dataset": dataset,
        "group": group,
        "seed": seed,
        "memory_budget": memory_budget,
        "image_auroc": float("nan"),
        "image_ap": float("nan"),
        "pixel_auroc": finite_auc(labels, output.scores.tolist()),
        "pro": float("nan"),
        "unique_top1": len(set(row[0] for row in output.selected_ids if row)),
        "unique_selected": len(set(flat_ids)),
        "gini": gini_from_ids(flat_ids),
        "entropy": entropy_from_ids(flat_ids),
        "jaccard": mean_adjacent_jaccard(output.selected_ids),
        "fallback_rate": float(np.mean([x != "none" for x in output.fallback_types])) if output.fallback_types else 0.0,
        "latency_ms": latency_ms,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0,
        "patch_count": len(labels),
        "selected_normal_mean": float(np.mean(output.normal_counts)) if output.normal_counts else 0.0,
        "selected_anomaly_mean": float(np.mean(output.anomaly_counts)) if output.anomaly_counts else 0.0,
    }


def mean_adjacent_jaccard(selected_ids: Sequence[Sequence[int]]) -> float:
    vals = []
    previous = None
    for ids in selected_ids:
        cur = set(ids)
        if previous is not None:
            vals.append(len(previous & cur) / max(1, len(previous | cur)))
        previous = cur
    return float(np.mean(vals)) if vals else 0.0


def fallback_breakdown(dataset: str, group: str, labels: Sequence[int], output: PatchAuditOutput, baseline_scores: Sequence[float]) -> List[Dict]:
    rows = []
    by_type: Dict[str, List[int]] = defaultdict(list)
    for i, fb in enumerate(output.fallback_types):
        by_type[fb].append(i)
    for fb in ["none", "missing_normal", "missing_anomaly", "missing_both", "numerical_fallback", "fallback_all"]:
        idx = list(range(len(labels))) if fb == "fallback_all" else by_type.get(fb, [])
        if fb == "fallback_all":
            idx = [i for i, x in enumerate(output.fallback_types) if x != "none"]
        if not idx:
            rows.append({"dataset": dataset, "group": group, "fallback_type": fb, "query_count": 0})
            continue
        y = [labels[i] for i in idx]
        s = [float(output.scores[i]) for i in idx]
        b = [float(baseline_scores[i]) for i in idx]
        rows.append({
            "dataset": dataset,
            "group": group,
            "fallback_type": fb,
            "query_count": len(idx),
            "pixel_auroc_proxy": finite_auc(y, s),
            "patch_ap": finite_ap(y, s),
            "score_difference_vs_g1": float(np.mean(np.asarray(s) - np.asarray(b))),
            "selected_normal_count": float(np.mean([output.normal_counts[i] for i in idx])),
            "selected_anomaly_count": float(np.mean([output.anomaly_counts[i] for i in idx])),
            "relation_branch": dict(Counter(r for i in idx for r in output.relations[i])),
        })
    return rows
