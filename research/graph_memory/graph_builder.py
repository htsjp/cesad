from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

import torch
import torch.nn.functional as F

from .relation_index import RELATIONS, RelationIndex, add_undirected, finalize_edges, sorted_limited


@dataclass
class GraphMemory:
    features: torch.Tensor
    values: torch.Tensor
    metadata: Dict[str, torch.Tensor]
    relation_index: RelationIndex


class GraphMemoryBuilder:
    def __init__(self, max_neighbors_per_relation: int = 16, knn_k: int = 16, seed: int = 0):
        self.max_neighbors_per_relation = int(max_neighbors_per_relation)
        self.knn_k = int(knn_k)
        self.seed = int(seed)

    def build(self, memory_features: torch.Tensor, memory_values: torch.Tensor, metadata: Optional[Mapping] = None) -> GraphMemory:
        features = F.normalize(memory_features.detach().float().cpu(), dim=-1)
        values = memory_values.detach().float().cpu()
        n = int(features.shape[0])
        md = self._metadata(n, values, metadata)
        edges = {rel: defaultdict(set) for rel in RELATIONS}

        self._group_relation(edges, "same_cluster", md["cluster_id"])
        self._group_relation(edges, "same_spatial_bin", md["spatial_bin"])
        self._group_relation(edges, "same_source_category", md["source_category_id"])
        self._group_relation(edges, "same_source_image", md["source_image_id"])
        self._cross_scale(edges, md["cluster_id"], md["spatial_bin"], md["scale_id"])
        self._contrast(edges, md["cluster_id"], md["binary_label"])
        self._knn(edges, features)

        relation_index = RelationIndex(finalize_edges(edges, self.max_neighbors_per_relation), n)
        relation_index.validate()
        return GraphMemory(features=features, values=values, metadata=md, relation_index=relation_index)

    def _metadata(self, n: int, values: torch.Tensor, metadata: Optional[Mapping]) -> Dict[str, torch.Tensor]:
        labels = values.argmax(dim=-1).long()
        defaults = {
            "node_id": torch.arange(n, dtype=torch.long),
            "binary_label": labels,
            "source_image_id": torch.arange(n, dtype=torch.long),
            "source_category_id": torch.zeros(n, dtype=torch.long),
            "region_type": labels,
            "spatial_bin": torch.zeros(n, dtype=torch.long),
            "scale_id": torch.zeros(n, dtype=torch.long),
            "cluster_id": labels * max(1, n),
        }
        if metadata:
            for key, value in metadata.items():
                if key not in defaults:
                    continue
                tensor = torch.as_tensor(value).detach().cpu().long()
                if tensor.numel() != n:
                    raise ValueError(f"metadata[{key}] has {tensor.numel()} rows, expected {n}")
                defaults[key] = tensor.reshape(n)
        return defaults

    def _group_relation(self, edges, rel: str, field: torch.Tensor) -> None:
        groups: Dict[int, List[int]] = defaultdict(list)
        for idx, value in enumerate(field.tolist()):
            groups[int(value)].append(idx)
        grouped = sorted_limited(groups, self.max_neighbors_per_relation)
        for src, dsts in grouped.items():
            for dst in dsts:
                add_undirected(edges, rel, src, dst)

    def _cross_scale(self, edges, cluster: torch.Tensor, spatial: torch.Tensor, scale: torch.Tensor) -> None:
        groups: Dict[tuple, List[int]] = defaultdict(list)
        for idx in range(cluster.numel()):
            groups[(int(cluster[idx]), int(spatial[idx]))].append(idx)
        for ids in groups.values():
            ids = sorted(ids)
            for i, src in enumerate(ids):
                for dst in ids[i + 1 :]:
                    if int(scale[src]) != int(scale[dst]):
                        add_undirected(edges, "cross_scale", src, dst)

    def _contrast(self, edges, cluster: torch.Tensor, labels: torch.Tensor) -> None:
        groups: Dict[int, List[int]] = defaultdict(list)
        for idx, value in enumerate(cluster.tolist()):
            groups[int(value)].append(idx)
        for ids in groups.values():
            normals = [i for i in ids if int(labels[i]) == 0]
            anomalies = [i for i in ids if int(labels[i]) == 1]
            for src in normals[: self.max_neighbors_per_relation]:
                for dst in anomalies[: self.max_neighbors_per_relation]:
                    add_undirected(edges, "normal_contrast", src, dst)
                    add_undirected(edges, "anomaly_contrast", dst, src)

    def _knn(self, edges, features: torch.Tensor) -> None:
        if features.numel() == 0:
            return
        k = min(self.knn_k + 1, features.shape[0])
        _, idx = torch.topk(features @ features.T, k=k, dim=-1)
        for src in range(features.shape[0]):
            for dst in idx[src].tolist():
                add_undirected(edges, "knn", src, int(dst))
