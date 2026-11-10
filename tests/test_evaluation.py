"""Tiny artificial fixtures test metric wiring, not model performance."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from cesad.evaluation import run


class EvaluationTests(unittest.TestCase):
    def test_native_mask_and_category_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask = np.zeros((9, 11), np.uint8)
            mask[2:5, 3:6] = 255
            Image.fromarray(mask).save(root / "mask.png")
            rows = []
            for i in range(2):
                score = np.zeros((74, 74), np.float32)
                if i:
                    score[16:42, 18:41] = 1
                np.save(root / f"{i}.npy", score)
                rows.append(
                    dict(
                        sample_id=str(i),
                        category="fixture",
                        label=i,
                        img_path="unused.png",
                        gt_path="mask.png" if i else "",
                        gt_height=9,
                        gt_width=11,
                        map_path=f"{i}.npy",
                    )
                )
            with (root / "manifest.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            run(root / "manifest.csv", root / "output")
            summary = json.loads((root / "output/summary.json").read_text())
            for key in ("I_AUROC", "I_AP", "P_AUROC", "P_AP", "P_F1max", "AUPRO"):
                self.assertIn(key, summary)
                self.assertTrue(np.isfinite(summary[key]))
            self.assertEqual(summary["I_AUROC"], 100.0)
            self.assertEqual(summary["I_AP"], 100.0)
            self.assertTrue((root / "output/per_category.csv").is_file())


if __name__ == "__main__":
    unittest.main()
