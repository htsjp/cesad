from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from .graph_builder import GraphMemory
from .prototype_reconstructor import PrototypeReconstructor


ACTIVE_RELATIONS = (
    "same_cluster",
    "same_spatial_bin",
    "cross_scale",
    "same_source_category",
    "normal_contrast",
    "anomaly_contrast",
    "same_source_image",
)


@dataclass
class RetrievalOutput:
    logits: torch.Tensor
    selected_ids: List[List[int]]
    scores: List[List[float]]
    relations: List[List[str]]
    fallback_events: List[str]


class ActiveGraphRetriever:
    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        self.mode = cfg.get("mode", "active_graph")
        self.k1 = int(cfg.get("k1", 4))
        self.k2 = int(cfg.get("k2", 8))
        self.top_relations = int(cfg.get("top_relations", 2))
        self.max_candidates = int(cfg.get("max_candidates", 32))
        self.temperature = float(cfg.get("temperature", 1.0))
        self.lambda_q = float(cfg.get("lambda_q", 1.0))
        self.lambda_s = float(cfg.get("lambda_s", 0.25))
        self.lambda_r = float(cfg.get("lambda_r", 0.15))
        self.lambda_d = float(cfg.get("lambda_d", 0.1))
        self.random_seed = int(cfg.get("random_seed", 0))
        self.reconstructor = PrototypeReconstructor(self.temperature)

    def retrieve(self, query_features: torch.Tensor, graph_memory: GraphMemory, config: Optional[Dict] = None) -> RetrievalOutput:
        if config:
            merged = self.__dict__.copy()
            merged.update(config)
            return ActiveGraphRetriever(merged).retrieve(query_features, graph_memory)

        query = F.normalize(query_features.detach().float().cpu(), dim=-1)
        features = graph_memory.features
        values = graph_memory.values
        sim = query @ features.T
        labels = values.argmax(dim=-1)
        logits_rows, selected_rows, score_rows, relation_rows, fallbacks = [], [], [], [], []

        for row in range(query.shape[0]):
            row_sim = sim[row]
            if self.mode in ("full_softmax", "G0"):
                weights = torch.softmax(row_sim / self.temperature, dim=-1)
                logits_rows.append(weights @ values.float())
                idx = [int(torch.argmax(row_sim).item())]
                selected_rows.append(idx)
                score_rows.append([float(row_sim[idx[0]])])
                relation_rows.append(["full_softmax"])
                continue

            if self.mode in ("fixed_topk", "G1"):
                idx = self._topk(row_sim, self.k2)
                logits_rows.append(self._direct_vote(row_sim, values, idx))
                selected_rows.append(idx)
                score_rows.append([float(row_sim[i]) for i in idx])
                relation_rows.append(["fixed_topk"])
                continue

            seeds = self._topk(row_sim, self.k1)
            rel_scores = self._relation_scores(row_sim, seeds, graph_memory)
            if self.mode in ("random_neighbors", "G2"):
                candidates, rels = self._random_neighbors(seeds, graph_memory)
            elif self.mode in ("knn_expand", "G3"):
                candidates = graph_memory.relation_index.neighbors(seeds, ["knn"], self.max_candidates)
                rels = ["knn"]
            else:
                rels = [r for r, _ in sorted(rel_scores.items(), key=lambda kv: (-kv[1], kv[0]))[: self.top_relations]]
                candidates = graph_memory.relation_index.neighbors(seeds, rels, self.max_candidates)
            candidates = self._unique_budget(seeds + candidates, self.max_candidates)
            idx = self._rerank(row_sim, candidates, seeds, graph_memory, rel_scores, rels)
            has_normal = any(int(labels[i]) == 0 for i in idx)
            has_anomaly = any(int(labels[i]) == 1 for i in idx)
            if self.mode in ("active_direct", "G4"):
                logits_rows.append(self._direct_vote(row_sim, values, idx))
            elif has_normal and has_anomaly:
                recon = self.reconstructor.reconstruct(query[row : row + 1], features[idx], values[idx])
                logits_rows.append(recon["logits"][0])
            else:
                fallback = "missing_normal" if not has_normal else "missing_anomaly"
                fallbacks.append(fallback)
                logits_rows.append(self._direct_vote(row_sim, values, idx))
            selected_rows.append(idx)
            score_rows.append([float(row_sim[i]) for i in idx])
            relation_rows.append(rels)

        return RetrievalOutput(torch.stack(logits_rows, dim=0), selected_rows, score_rows, relation_rows, fallbacks)

    def _topk(self, row_sim: torch.Tensor, k: int) -> List[int]:
        k = min(int(k), row_sim.numel())
        return [int(x) for x in torch.topk(row_sim, k=k).indices.tolist()]

    def _direct_vote(self, row_sim: torch.Tensor, values: torch.Tensor, idx: Sequence[int]) -> torch.Tensor:
        if not idx:
            idx = self._topk(row_sim, 1)
        weights = torch.softmax(row_sim[list(idx)] / self.temperature, dim=-1)
        return weights @ values[list(idx)].float()

    def _relation_scores(self, row_sim: torch.Tensor, seeds: Sequence[int], memory: GraphMemory) -> Dict[str, float]:
        md = memory.metadata
        seed_labels = md["binary_label"][list(seeds)]
        scores: Dict[str, float] = {}
        for rel in ACTIVE_RELATIONS:
            nbrs = memory.relation_index.neighbors(seeds, [rel], self.max_candidates)
            if nbrs:
                support = float(row_sim[nbrs].mean())
                change = len(set(nbrs) - set(seeds)) / max(1, len(nbrs))
            else:
                support = -1.0
                change = 0.0
            balance = 1.0 if seed_labels.unique().numel() > 1 else 0.0
            contrast_bonus = 0.1 if "contrast" in rel and balance == 0.0 else 0.0
            scores[rel] = support + 0.2 * change + contrast_bonus
        return scores

    def _random_neighbors(self, seeds: Sequence[int], memory: GraphMemory):
        all_ids = sorted(set(range(memory.features.shape[0])) - set(int(x) for x in seeds))
        gen = torch.Generator(device="cpu").manual_seed(self.random_seed + sum(int(x) for x in seeds))
        if not all_ids:
            return [], ["random"]
        perm = torch.randperm(len(all_ids), generator=gen).tolist()
        return [all_ids[i] for i in perm[: self.max_candidates]], ["random"]

    def _rerank(self, row_sim, candidates, seeds, memory, rel_scores, rels):
        if not candidates:
            return list(seeds)[: self.k2]
        seed_mean = memory.features[list(seeds)].mean(dim=0, keepdim=True)
        support_sim = (memory.features[candidates] @ F.normalize(seed_mean, dim=-1).T).squeeze(-1)
        base = self.lambda_q * row_sim[candidates] + self.lambda_s * support_sim
        rel_bonus = self.lambda_r * (max((rel_scores.get(r, 0.0) for r in rels), default=0.0))
        scores = base + rel_bonus
        selected: List[int] = []
        selected_feats = []
        for _ in range(min(self.k2, len(candidates))):
            adjusted = scores.clone()
            if selected_feats and self.lambda_d > 0:
                prev = torch.stack(selected_feats)
                redundancy = memory.features[candidates] @ prev.T
                adjusted = adjusted - self.lambda_d * redundancy.max(dim=-1).values
            for pos, cid in enumerate(candidates):
                if cid in selected:
                    adjusted[pos] = -float("inf")
            pick_pos = int(torch.argmax(adjusted).item())
            pick = int(candidates[pick_pos])
            selected.append(pick)
            selected_feats.append(memory.features[pick])
        return selected

    def _unique_budget(self, ids: Sequence[int], budget: int) -> List[int]:
        out = []
        seen = set()
        for value in ids:
            value = int(value)
            if value in seen:
                continue
            seen.add(value)
            out.append(value)
            if len(out) >= budget:
                break
        return out
