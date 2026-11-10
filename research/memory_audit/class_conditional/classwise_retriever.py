from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F


@dataclass
class RetrievalResult:
    logits: torch.Tensor
    anomaly_scores: torch.Tensor
    selected_ids: torch.Tensor
    normal_counts: torch.Tensor
    anomaly_counts: torch.Tensor
    diagnostics: Dict[str, torch.Tensor]


class ClasswiseRetriever:
    def __init__(self, keys: torch.Tensor, values: torch.Tensor, temperature: float = 1.0):
        self.keys = F.normalize(keys.detach().float(), dim=-1)
        self.values = values.detach().float()
        self.labels = self.values.argmax(dim=-1).long()
        self.temperature = float(temperature)
        self.normal_ids = torch.where(self.labels == 0)[0]
        self.anomaly_ids = torch.where(self.labels == 1)[0]

    def retrieve(
        self,
        queries: torch.Tensor,
        mode: str,
        normal_k: int = 4,
        anomaly_k: int = 4,
        total_k: int = 8,
        aggregator: str = "candidate_softmax",
        bandwidth: float = 0.05,
        bias: float = 0.0,
    ) -> RetrievalResult:
        queries = F.normalize(queries.detach().float(), dim=-1)
        sim = queries @ self.keys.T
        if mode == "global_topk":
            return self._global_topk(sim, total_k)
        if mode == "balanced_topk":
            return self._balanced_topk(sim, normal_k, anomaly_k, aggregator=aggregator, bandwidth=bandwidth, bias=bias)
        if mode == "classwise_topk":
            return self._balanced_topk(sim, normal_k, anomaly_k, aggregator=aggregator, bandwidth=bandwidth, bias=bias)
        if mode == "adaptive_quota":
            return self._adaptive_quota(sim, aggregator=aggregator, bandwidth=bandwidth, bias=bias)
        raise ValueError(f"Unsupported retrieval mode: {mode}")

    def _global_topk(self, sim: torch.Tensor, k: int) -> RetrievalResult:
        vals, idx = torch.topk(sim, k=min(int(k), sim.shape[-1]), dim=-1)
        weights = torch.softmax(vals / self.temperature, dim=-1)
        selected_values = self.values[idx]
        logits = (weights.unsqueeze(-1) * selected_values).sum(dim=-2)
        selected_labels = self.labels[idx]
        normal_counts = (selected_labels == 0).sum(dim=-1)
        anomaly_counts = (selected_labels == 1).sum(dim=-1)
        diag = self._diagnostics(sim, idx, weights, normal_counts, anomaly_counts)
        return RetrievalResult(logits, logits[:, 1], idx, normal_counts, anomaly_counts, diag)

    def _balanced_topk(self, sim: torch.Tensor, normal_k: int, anomaly_k: int, aggregator: str = "candidate_softmax", bandwidth: float = 0.05, bias: float = 0.0) -> RetrievalResult:
        n_vals, n_local = torch.topk(sim[:, self.normal_ids], k=min(int(normal_k), self.normal_ids.numel()), dim=-1)
        a_vals, a_local = torch.topk(sim[:, self.anomaly_ids], k=min(int(anomaly_k), self.anomaly_ids.numel()), dim=-1)
        n_idx = self.normal_ids[n_local]
        a_idx = self.anomaly_ids[a_local]
        vals = torch.cat([n_vals, a_vals], dim=-1)
        idx = torch.cat([n_idx, a_idx], dim=-1)
        logits, weights = self._aggregate(vals, n_vals, a_vals, aggregator, bandwidth, bias)
        normal_counts = torch.full((sim.shape[0],), n_idx.shape[-1], dtype=torch.long)
        anomaly_counts = torch.full((sim.shape[0],), a_idx.shape[-1], dtype=torch.long)
        diag = self._diagnostics(sim, idx, weights, normal_counts, anomaly_counts)
        return RetrievalResult(logits, logits[:, 1], idx, normal_counts, anomaly_counts, diag)

    def _adaptive_quota(self, sim: torch.Tensor, aggregator: str = "candidate_softmax", bandwidth: float = 0.05, bias: float = 0.0) -> RetrievalResult:
        normal_scores = sim[:, self.normal_ids]
        anomaly_scores = sim[:, self.anomaly_ids]
        margin = anomaly_scores.max(dim=-1).values - normal_scores.max(dim=-1).values
        # Fixed source-calibrated thresholds, independent of target labels.
        a_k = torch.where(margin > 0.02, 6, torch.where(margin < -0.02, 2, 4))
        selected_idx = []
        selected_vals = []
        normal_counts = []
        anomaly_counts = []
        logits_rows = []
        weight_rows = []
        max_k = 8
        for row in range(sim.shape[0]):
            ak = int(a_k[row].item())
            nk = max_k - ak
            n_vals, n_local = torch.topk(normal_scores[row], k=min(nk, self.normal_ids.numel()), dim=-1)
            a_vals, a_local = torch.topk(anomaly_scores[row], k=min(ak, self.anomaly_ids.numel()), dim=-1)
            vals = torch.cat([n_vals, a_vals], dim=-1).unsqueeze(0)
            idx = torch.cat([self.normal_ids[n_local], self.anomaly_ids[a_local]], dim=-1)
            logits, weights = self._aggregate(vals, n_vals.unsqueeze(0), a_vals.unsqueeze(0), aggregator, bandwidth, bias)
            selected_idx.append(idx)
            selected_vals.append(vals.squeeze(0))
            logits_rows.append(logits.squeeze(0))
            weight_rows.append(weights.squeeze(0))
            normal_counts.append(nk)
            anomaly_counts.append(ak)
        idx = torch.stack(selected_idx, dim=0)
        logits = torch.stack(logits_rows, dim=0)
        weights = torch.stack(weight_rows, dim=0)
        normal_counts_t = torch.tensor(normal_counts, dtype=torch.long)
        anomaly_counts_t = torch.tensor(anomaly_counts, dtype=torch.long)
        diag = self._diagnostics(sim, idx, weights, normal_counts_t, anomaly_counts_t)
        return RetrievalResult(logits, logits[:, 1], idx, normal_counts_t, anomaly_counts_t, diag)

    def _aggregate(self, vals: torch.Tensor, n_vals: torch.Tensor, a_vals: torch.Tensor, aggregator: str, bandwidth: float, bias: float) -> tuple[torch.Tensor, torch.Tensor]:
        if aggregator == "candidate_softmax":
            weights = torch.softmax(vals / self.temperature, dim=-1)
            labels = torch.cat([torch.zeros_like(n_vals), torch.ones_like(a_vals)], dim=-1)
            anomaly = (weights * labels).sum(dim=-1)
            return torch.stack([1.0 - anomaly, anomaly], dim=-1), weights
        if aggregator == "class_mean":
            e_n = n_vals.mean(dim=-1)
            e_a = a_vals.mean(dim=-1)
        elif aggregator == "class_weighted_mean":
            w_n = torch.softmax(n_vals / self.temperature, dim=-1)
            w_a = torch.softmax(a_vals / self.temperature, dim=-1)
            e_n = (w_n * n_vals).sum(dim=-1)
            e_a = (w_a * a_vals).sum(dim=-1)
        elif aggregator == "class_logmeanexp":
            e_n = torch.logsumexp(n_vals / self.temperature, dim=-1) - math.log(max(1, n_vals.shape[-1]))
            e_a = torch.logsumexp(a_vals / self.temperature, dim=-1) - math.log(max(1, a_vals.shape[-1]))
        elif aggregator == "class_logsumexp":
            e_n = torch.logsumexp(n_vals / self.temperature, dim=-1)
            e_a = torch.logsumexp(a_vals / self.temperature, dim=-1)
        elif aggregator == "class_max":
            e_n = n_vals.max(dim=-1).values
            e_a = a_vals.max(dim=-1).values
        elif aggregator == "class_kde":
            bw = max(float(bandwidth), 1e-6)
            e_n = torch.logsumexp((n_vals - 1.0) / bw, dim=-1) - math.log(max(1, n_vals.shape[-1]))
            e_a = torch.logsumexp((a_vals - 1.0) / bw, dim=-1) - math.log(max(1, a_vals.shape[-1]))
        else:
            raise ValueError(f"Unsupported aggregator: {aggregator}")
        anomaly = torch.sigmoid(e_a - e_n + float(bias))
        logits = torch.stack([1.0 - anomaly, anomaly], dim=-1)
        weights = torch.softmax(vals / self.temperature, dim=-1)
        return logits, weights

    def _diagnostics(self, sim: torch.Tensor, idx: torch.Tensor, weights: torch.Tensor, normal_counts: torch.Tensor, anomaly_counts: torch.Tensor) -> Dict[str, torch.Tensor]:
        normal_scores = sim[:, self.normal_ids]
        anomaly_scores = sim[:, self.anomaly_ids]
        selected_labels = self.labels[idx]
        anomaly_attention_share = (weights * (selected_labels == 1).float()).sum(dim=-1)
        # Full-class entropy is expensive on CPU for every patch/group in formal runs.
        # Keep lightweight competition diagnostics here; entropy can be sampled in a
        # separate diagnostic-only pass if needed.
        normal_entropy = torch.zeros(sim.shape[0], dtype=sim.dtype, device=sim.device)
        anomaly_entropy = torch.zeros(sim.shape[0], dtype=sim.dtype, device=sim.device)
        return {
            "global_top8_normal_count": normal_counts.float(),
            "global_top8_anomaly_count": anomaly_counts.float(),
            "normal_attention_mass": 1.0 - anomaly_attention_share,
            "anomaly_attention_mass": anomaly_attention_share,
            "normal_max_similarity": normal_scores.max(dim=-1).values,
            "anomaly_max_similarity": anomaly_scores.max(dim=-1).values,
            "normal_mean_similarity": normal_scores.mean(dim=-1),
            "anomaly_mean_similarity": anomaly_scores.mean(dim=-1),
            "normal_entropy": normal_entropy,
            "anomaly_entropy": anomaly_entropy,
            "class_evidence_margin": anomaly_scores.max(dim=-1).values - normal_scores.max(dim=-1).values,
        }
