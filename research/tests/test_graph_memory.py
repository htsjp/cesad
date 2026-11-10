import copy
import unittest

import torch

from graph_memory import ActiveGraphRetriever, GraphMemoryBuilder, PrototypeReconstructor, RetrievalDiagnostics


def toy():
    torch.manual_seed(0)
    features = torch.nn.functional.normalize(torch.randn(12, 8), dim=-1)
    labels = torch.tensor([0, 1] * 6)
    values = torch.nn.functional.one_hot(labels, num_classes=2).float()
    metadata = {
        "binary_label": labels,
        "source_image_id": torch.arange(12) // 2,
        "source_category_id": torch.arange(12) % 3,
        "region_type": labels,
        "spatial_bin": torch.arange(12) % 4,
        "scale_id": torch.arange(12) % 2,
        "cluster_id": torch.arange(12) % 3,
    }
    return features, values, metadata


class GraphMemoryTests(unittest.TestCase):
    def test_graph_construction_is_deterministic(self):
        f, v, m = toy()
        a = GraphMemoryBuilder(seed=7).build(f, v, m)
        b = GraphMemoryBuilder(seed=7).build(f, v, copy.deepcopy(m))
        self.assertEqual(a.relation_index.edges, b.relation_index.edges)

    def test_no_edge_references_nonexistent_node(self):
        f, v, m = toy()
        graph = GraphMemoryBuilder().build(f, v, m)
        graph.relation_index.validate()

    def test_relation_expansion_respects_candidate_budget(self):
        f, v, m = toy()
        graph = GraphMemoryBuilder().build(f, v, m)
        nbrs = graph.relation_index.neighbors([0, 1], ["same_cluster", "knn"], 3)
        self.assertLessEqual(len(nbrs), 3)

    def test_selected_nodes_are_unique(self):
        f, v, m = toy()
        graph = GraphMemoryBuilder().build(f, v, m)
        out = ActiveGraphRetriever({"mode": "active_graph", "k2": 6}).retrieve(f[:3], graph)
        for ids in out.selected_ids:
            self.assertEqual(len(ids), len(set(ids)))

    def test_reconstruction_returns_finite_scores(self):
        f, v, _ = toy()
        out = PrototypeReconstructor().reconstruct(f[:2], f[:6], v[:6])
        self.assertTrue(torch.isfinite(out["logits"]).all())
        self.assertTrue(torch.allclose(out["logits"].sum(dim=-1), torch.ones(2), atol=1e-5))

    def test_fallback_paths_are_exercised(self):
        f, v, m = toy()
        only_normal = v.clone()
        only_normal[:, 0] = 1
        only_normal[:, 1] = 0
        graph = GraphMemoryBuilder().build(f, only_normal, m)
        out = ActiveGraphRetriever({"mode": "active_graph", "k2": 4}).retrieve(f[:2], graph)
        self.assertTrue(out.fallback_events)

    def test_g1_reproduces_fixed_topk_baseline(self):
        f, v, m = toy()
        graph = GraphMemoryBuilder().build(f, v, m)
        out = ActiveGraphRetriever({"mode": "fixed_topk", "k2": 3}).retrieve(f[:1], graph)
        expected = torch.topk(f[:1] @ graph.features.T, k=3, dim=-1).indices[0].tolist()
        self.assertEqual(out.selected_ids[0], expected)

    def test_active_graph_does_not_alter_clip_weights_placeholder(self):
        f, v, m = toy()
        before = f.clone()
        graph = GraphMemoryBuilder().build(f, v, m)
        ActiveGraphRetriever({"mode": "active_graph"}).retrieve(f[:2], graph)
        self.assertTrue(torch.equal(f, before))

    def test_target_labels_are_not_required_by_graph_construction(self):
        f, v, _ = toy()
        graph = GraphMemoryBuilder().build(f, v, metadata=None)
        self.assertEqual(graph.features.shape[0], f.shape[0])

    def test_diagnostics_match_toy_graph(self):
        diag = RetrievalDiagnostics(memory_size=5)
        diag.update([0, 1], [[1, 2], [2, 3]], [[0.9, 0.8], [0.7, 0.1]], [["a"], ["b"]])
        summary = diag.summarize()
        self.assertEqual(summary["unique_top1_prototypes"], 2)
        self.assertEqual(summary["unique_selected_prototypes"], 3)
        self.assertIn("a", summary["relation_branch_utilization"])


if __name__ == "__main__":
    unittest.main()
