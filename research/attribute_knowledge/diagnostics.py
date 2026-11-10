import json
from collections import defaultdict

import numpy as np
import torch


class AttributeDiagnostics:
    def __init__(self, metadata, bins=200):
        self.metadata = metadata
        self.num_attrs = int(metadata["num_attributes"])
        self.bins = int(bins)
        self.selection_normal = np.zeros(self.num_attrs, dtype=np.float64)
        self.selection_anomaly = np.zeros(self.num_attrs, dtype=np.float64)
        self.affinity_normal = np.zeros(self.num_attrs, dtype=np.float64)
        self.affinity_anomaly = np.zeros(self.num_attrs, dtype=np.float64)
        self.count_normal = 0
        self.count_anomaly = 0
        self.hist_normal = np.zeros((self.num_attrs, self.bins), dtype=np.float64)
        self.hist_anomaly = np.zeros((self.num_attrs, self.bins), dtype=np.float64)
        self.category_counts = defaultdict(lambda: np.zeros(self.num_attrs, dtype=np.float64))
        self.examples = []

    def update(self, profiles, selected, gt_patch, category, max_examples=50, image_path=None, base_score=None, new_score=None):
        prof = profiles.detach().cpu().numpy().reshape(-1, profiles.shape[-1])
        sel = selected.detach().cpu().numpy().reshape(-1, selected.shape[-1])
        gt = gt_patch.detach().cpu().numpy().reshape(-1).astype(bool)
        for row, is_anom in zip(sel, gt):
            if is_anom:
                self.selection_anomaly[row] += 1
            else:
                self.selection_normal[row] += 1
            self.category_counts[str(category)][row] += 1
        self.affinity_normal += prof[~gt].sum(axis=0) if (~gt).any() else 0
        self.affinity_anomaly += prof[gt].sum(axis=0) if gt.any() else 0
        self.count_normal += int((~gt).sum())
        self.count_anomaly += int(gt.sum())
        scaled = np.clip((prof * (self.bins - 1)).astype(np.int64), 0, self.bins - 1)
        for a in range(self.num_attrs):
            self.hist_normal[a] += np.bincount(scaled[~gt, a], minlength=self.bins)
            self.hist_anomaly[a] += np.bincount(scaled[gt, a], minlength=self.bins)
        if len(self.examples) < max_examples and image_path is not None and base_score is not None and new_score is not None:
            delta = (new_score.detach().cpu().numpy().reshape(-1) - base_score.detach().cpu().numpy().reshape(-1))
            order = np.argsort(-np.abs(delta))[: max(1, max_examples - len(self.examples))]
            side = int(np.sqrt(delta.shape[0]))
            attrs = self.metadata["attributes"]
            for idx in order:
                y, x = divmod(int(idx), side)
                self.examples.append(
                    {
                        "image_path": str(image_path),
                        "patch_index": int(idx),
                        "patch_coordinate": [int(y), int(x)],
                        "gt_patch_label": int(gt[idx]),
                        "baseline_score": float(base_score.reshape(-1)[idx].detach().cpu().item()),
                        "structured_score": float(new_score.reshape(-1)[idx].detach().cpu().item()),
                        "selected_attributes": [attrs[int(a)] for a in sel[idx].tolist()],
                        "score_change": float(delta[idx]),
                    }
                )
                if len(self.examples) >= max_examples:
                    break

    def _hist_auc(self, normal_hist, anomaly_hist):
        neg = normal_hist.sum()
        pos = anomaly_hist.sum()
        if neg == 0 or pos == 0:
            return float("nan")
        cum_neg_lower = 0.0
        score = 0.0
        for b in range(self.bins):
            pos_b = anomaly_hist[b]
            neg_b = normal_hist[b]
            score += pos_b * cum_neg_lower + 0.5 * pos_b * neg_b
            cum_neg_lower += neg_b
        return float(score / (pos * neg))

    def rows(self):
        rows = []
        total_sel = self.selection_normal + self.selection_anomaly
        cat_total = {cat: counts.sum() for cat, counts in self.category_counts.items()}
        for i, attr in enumerate(self.metadata["attributes"]):
            cat_probs = []
            for cat, counts in self.category_counts.items():
                if cat_total[cat] > 0:
                    cat_probs.append(counts[i] / cat_total[cat])
            cat_probs = np.asarray(cat_probs, dtype=np.float64)
            cat_probs = cat_probs / cat_probs.sum() if cat_probs.sum() > 0 else cat_probs
            entropy = float(-(cat_probs * np.log(cat_probs + 1e-12)).sum()) if cat_probs.size else 0.0
            rows.append(
                {
                    "attribute": attr,
                    "group": self.metadata["groups"][i],
                    "selection_normal": self.selection_normal[i],
                    "selection_anomaly": self.selection_anomaly[i],
                    "anomaly_normal_selection_ratio": float((self.selection_anomaly[i] + 1.0) / (self.selection_normal[i] + 1.0)),
                    "category_entropy": entropy,
                    "spatial_concentration": float(total_sel[i] / max(1.0, total_sel.sum())),
                    "mean_affinity_normal": float(self.affinity_normal[i] / max(1, self.count_normal)),
                    "mean_affinity_anomaly": float(self.affinity_anomaly[i] / max(1, self.count_anomaly)),
                    "label_auroc": self._hist_auc(self.hist_normal[i], self.hist_anomaly[i]),
                }
            )
        return rows

    def save_examples(self, path):
        path.write_text(json.dumps(self.examples[:50], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

