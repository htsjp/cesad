import math
from collections import defaultdict

import numpy as np
import torch


def entropy_stats(weights):
    ent = -(weights * weights.clamp_min(1e-12).log()).sum(dim=-1)
    norm = ent / math.log(max(2, weights.shape[-1]))
    eff = torch.exp(ent)
    return ent, norm, eff, weights.max(dim=-1).values, torch.topk(weights, k=min(5, weights.shape[-1]), dim=-1).values.sum(dim=-1)


def append_basic_diag(rows, seed, variant, cls_name, prob, baseline_prob=None):
    p = prob.detach().float().reshape(-1).cpu().numpy()
    row = {
        "seed": seed,
        "variant": variant,
        "category": cls_name,
        "score_mean": float(np.mean(p)),
        "score_std": float(np.std(p)),
        "score_min": float(np.min(p)),
        "score_max": float(np.max(p)),
    }
    if baseline_prob is not None:
        b = baseline_prob.detach().float().reshape(-1).cpu().numpy()
        row["score_corr_vs_v0"] = float(np.corrcoef(p, b)[0, 1]) if np.std(p) > 0 and np.std(b) > 0 else float("nan")
        row["score_abs_change_vs_v0"] = float(np.mean(np.abs(p - b)))
    rows.append(row)


def false_area(mask, amap, threshold=0.5):
    m = mask.detach().cpu().numpy() > 0.5
    a = amap.detach().cpu().numpy() > threshold
    fp = np.logical_and(a, ~m).sum() / max(1, (~m).sum())
    fn = np.logical_and(~a, m).sum() / max(1, m.sum())
    return float(fp), float(fn)
