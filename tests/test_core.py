"""Synthetic unit fixtures are math tests, not paper experiment results."""

import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from cesad.heads import (
    CESHead,
    LinearHead,
    NormalizedLinearHead,
    ntr_loss_from_logits,
    balanced_loss,
)
from cesad.geometry import load_head, model_state
from cesad.features import SourceTransform
from cesad.lrp import crop_boxes, fuse_views


class CoreTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        sn = rng.normal(size=1024).astype(np.float32)
        sa = rng.normal(size=1024).astype(np.float32)
        sn /= np.linalg.norm(sn)
        sa /= np.linalg.norm(sa)
        d = (sa - sn) / 2
        self.head = CESHead((sa + sn) / 2, np.linalg.norm(d), d / np.linalg.norm(d))
        self.z = torch.nn.functional.normalize(
            torch.from_numpy(rng.normal(size=(16, 1024)).astype(np.float32)), dim=-1
        )

    def test_matched_logits(self):
        w = self.head.to_spec("test").weight()
        linear = LinearHead(w)
        normalized = NormalizedLinearHead(w, np.linalg.norm(w))
        torch.testing.assert_close(self.head(self.z), linear(self.z))
        torch.testing.assert_close(self.head(self.z), normalized(self.z))
        self.assertEqual(float(linear.b.detach()), 0.0)

    def test_only_direction_trainable(self):
        self.assertEqual(sum(p.numel() for p in self.head.parameters()), 1024)
        self.assertFalse(self.head.m.requires_grad)
        self.assertFalse(self.head.alpha.requires_grad)

    def test_ntr_fixed_top10_percent(self):
        q = torch.arange(20, dtype=torch.float32, requires_grad=True)
        loss = ntr_loss_from_logits(q, torch.zeros(20), 0.1)
        torch.testing.assert_close(loss, torch.nn.functional.softplus(q[-2:]).mean())
        loss.backward()
        self.assertEqual(int((q.grad != 0).sum()), 2)

    def test_balanced_loss(self):
        q = torch.zeros(10)
        y = torch.tensor([1.0, 1.0] + [0.0] * 8)
        total, bce, dice = balanced_loss(q, y)
        torch.testing.assert_close(bce, torch.tensor(np.log(2), dtype=torch.float32))
        torch.testing.assert_close(dice, torch.tensor(1.0 - 3.0 / 8.0))
        torch.testing.assert_close(total, bce + dice)

    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head.pt"
            torch.save(dict(format="cesad_head_v1", model=self.head.state_dict()), path)
            loaded, _ = load_head(path)
            torch.testing.assert_close(self.head(self.z), loaded(self.z))

    def test_reject_missing_trained_direction(self):
        with self.assertRaises(ValueError):
            model_state({"init_v_fallback": np.ones(1024)})

    def test_lrp_tile_order(self):
        views = torch.zeros(1, 5, 37, 37)
        for i in range(1, 5):
            views[:, i] = i
        result = fuse_views(views)[0]
        self.assertEqual(tuple(result.shape), (74, 74))
        for row, col, value in [(0, 0, 0.5), (0, 73, 1.0), (73, 0, 1.5), (73, 73, 2.0)]:
            self.assertEqual(float(result[row, col]), value)

    def test_crop_coverage_odd_dimensions(self):
        boxes = crop_boxes(101, 99)
        area = sum((r - l) * (b - t) for l, t, r, b in boxes)
        self.assertEqual(area, 101 * 99)

    def test_transform_l2(self):
        torch.testing.assert_close(SourceTransform()(self.z), self.z)

    def test_pc1_then_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            pc = np.zeros(1024, np.float32)
            pc[0] = 1
            np.save(directory / "pc.npy", pc)
            a = torch.zeros(4, 1024)
            b = torch.zeros(1024, 4)
            a[0, 1] = 0.1
            b[2, 0] = 0.2
            torch.save(
                {"model": {"a.weight": a, "b.weight": b}}, directory / "adapter.pt"
            )
            transform = SourceTransform(directory / "pc.npy", directory / "adapter.pt")
            direct = self.z.clone()
            direct[:, 0] = 0
            direct = torch.nn.functional.normalize(direct, dim=-1)
            direct = torch.nn.functional.normalize(direct + direct @ a.T @ b.T, dim=-1)
            torch.testing.assert_close(transform(self.z), direct)
            self.assertFalse(any(p.requires_grad for p in transform.parameters()))


if __name__ == "__main__":
    unittest.main()
