from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence

import numpy as np


def gini(counts: Sequence[int]) -> float:
    arr = np.asarray(counts, dtype=np.float64)
    if arr.size == 0 or arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    return float((2 * np.arange(1, n + 1).dot(arr) / arr.sum() - (n + 1)) / n)


class RetrievalDiagnostics:
    def __init__(self, memory_size: int = 0):
        self.memory_size = int(memory_size)
        self.top1 = Counter()
        self.selected = Counter()
        self.relations = Counter()
        self.fallbacks = Counter()
        self.margins: List[float] = []
        self.candidate_counts: List[int] = []
        self.previous_selected: List[int] | None = None
        self.jaccards: List[float] = []
        self.changed_from_baseline: List[bool] = []
        self.label_balance = defaultdict(int)

    def update(self, query_ids: Iterable[int], selected_nodes: Sequence[Sequence[int]], scores, relations, labels=None, baseline_selected=None, fallback_events=None):
        for i, selected in enumerate(selected_nodes):
            selected = [int(x) for x in selected]
            if selected:
                self.top1[selected[0]] += 1
            self.selected.update(selected)
            self.candidate_counts.append(len(selected))
            if len(scores[i]) > 1:
                ordered = sorted(scores[i], reverse=True)
                self.margins.append(float(ordered[0] - ordered[1]))
            for rel in relations[i]:
                self.relations[str(rel)] += 1
            if labels is not None:
                labs = [int(labels[x]) for x in selected]
                self.label_balance["normal"] += labs.count(0)
                self.label_balance["anomaly"] += labs.count(1)
            if self.previous_selected is not None:
                inter = len(set(self.previous_selected) & set(selected))
                union = len(set(self.previous_selected) | set(selected))
                self.jaccards.append(inter / max(1, union))
            self.previous_selected = selected
            if baseline_selected is not None:
                self.changed_from_baseline.append(set(selected) != set(baseline_selected[i]))
        if fallback_events:
            self.fallbacks.update(fallback_events)

    def summarize(self) -> Dict:
        selected_counts = list(self.selected.values())
        top1_counts = list(self.top1.values())
        probs = np.asarray(selected_counts, dtype=np.float64)
        probs = probs / probs.sum() if probs.sum() else probs
        entropy = float(-(probs * np.log(np.clip(probs, 1e-12, None))).sum()) if probs.size else 0.0
        return {
            "memory_size": self.memory_size,
            "unique_top1_prototypes": len(self.top1),
            "unique_selected_prototypes": len(self.selected),
            "prototype_usage_gini": gini(selected_counts),
            "top1_usage_gini": gini(top1_counts),
            "prototype_usage_entropy": entropy,
            "mean_query_to_query_jaccard": float(np.mean(self.jaccards)) if self.jaccards else 0.0,
            "mean_retrieval_margin": float(np.mean(self.margins)) if self.margins else 0.0,
            "effective_candidate_count": float(np.mean(self.candidate_counts)) if self.candidate_counts else 0.0,
            "relation_branch_utilization": dict(self.relations),
            "fallback_rate": float(sum(self.fallbacks.values()) / max(1, len(self.candidate_counts))),
            "fallback_events": dict(self.fallbacks),
            "selected_node_balance": dict(self.label_balance),
            "predictions_changed_relative_to_g1": float(np.mean(self.changed_from_baseline)) if self.changed_from_baseline else 0.0,
        }
