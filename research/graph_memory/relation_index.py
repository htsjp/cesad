from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Set


RELATIONS = (
    "same_cluster",
    "same_spatial_bin",
    "cross_scale",
    "same_source_category",
    "normal_contrast",
    "anomaly_contrast",
    "same_source_image",
    "knn",
)


@dataclass
class RelationIndex:
    edges: Dict[str, Dict[int, List[int]]]
    num_nodes: int

    def neighbors(self, node_ids: Iterable[int], relations: Sequence[str], max_candidates: int) -> List[int]:
        seen: Set[int] = set()
        out: List[int] = []
        for rel in relations:
            rel_edges = self.edges.get(rel, {})
            for node_id in node_ids:
                for nbr in rel_edges.get(int(node_id), []):
                    if nbr in seen:
                        continue
                    seen.add(nbr)
                    out.append(nbr)
                    if len(out) >= max_candidates:
                        return out
        return out

    def validate(self) -> None:
        for rel, rel_edges in self.edges.items():
            if rel not in RELATIONS:
                raise ValueError(f"Unknown relation: {rel}")
            for src, dsts in rel_edges.items():
                if src < 0 or src >= self.num_nodes:
                    raise ValueError(f"Relation {rel} has nonexistent source node {src}")
                for dst in dsts:
                    if dst < 0 or dst >= self.num_nodes:
                        raise ValueError(f"Relation {rel} has nonexistent destination node {dst}")


def add_undirected(edges: Dict[str, Dict[int, Set[int]]], rel: str, a: int, b: int) -> None:
    if a == b:
        return
    edges[rel][int(a)].add(int(b))
    edges[rel][int(b)].add(int(a))


def sorted_limited(groups: Mapping[object, List[int]], per_node_budget: int) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = defaultdict(list)
    for ids in groups.values():
        ids = sorted(set(int(x) for x in ids))
        for src in ids:
            nbrs = [x for x in ids if x != src][:per_node_budget]
            out[src].extend(nbrs)
    return {k: sorted(set(v))[:per_node_budget] for k, v in out.items()}


def finalize_edges(edges: Dict[str, Dict[int, Set[int]]], per_node_budget: int) -> Dict[str, Dict[int, List[int]]]:
    final: Dict[str, Dict[int, List[int]]] = {}
    for rel, rel_edges in edges.items():
        final[rel] = {int(src): sorted(int(x) for x in dsts)[:per_node_budget] for src, dsts in rel_edges.items()}
    return final
