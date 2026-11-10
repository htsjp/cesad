from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from .prototype_builder import build_projected_bank
from .prototype_scorer import score_prototype_variant


@dataclass
class AffineCalibrator:
    a: float
    b: float
    fold_variance_a: float
    fold_variance_b: float
    source_scores: List[float]
    source_labels: List[int]
    fold_rows: List[Dict]
    source_split_hash: str


def sigmoid_affine(scores: torch.Tensor, a: float, b: float) -> torch.Tensor:
    z = torch.logit(scores.clamp(1e-6, 1 - 1e-6))
    return torch.sigmoid(float(a) * z + float(b))


def fit_affine(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    if len(np.unique(labels)) < 2 or np.std(scores) <= 0:
        return 1.0, 0.0
    z = np.log(np.clip(scores, 1e-6, 1 - 1e-6) / np.clip(1 - scores, 1e-6, 1))
    clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    clf.fit(z.reshape(-1, 1), labels.astype(int))
    return float(clf.coef_[0, 0]), float(clf.intercept_[0])


def stratified_folds(labels: torch.Tensor, k: int, seed: int) -> List[torch.Tensor]:
    labels = labels.cpu().long()
    n = labels.numel()
    if n < 4:
        return [torch.arange(n)]
    k = max(2, min(int(k), int((labels == 0).sum().item()), int((labels == 1).sum().item())))
    gen = torch.Generator().manual_seed(seed)
    folds = [[] for _ in range(k)]
    for cls in [0, 1]:
        idx = torch.where(labels == cls)[0]
        idx = idx[torch.randperm(idx.numel(), generator=gen)]
        for i, item in enumerate(idx.tolist()):
            folds[i % k].append(item)
    return [torch.tensor(sorted(x), dtype=torch.long) for x in folds]


def score_source_variant(
    variant: str,
    query_raw: torch.Tensor,
    train_keys_raw: torch.Tensor,
    train_values: torch.Tensor,
    image_proj,
    device: torch.device,
) -> torch.Tensor:
    query_raw = query_raw.to(device)
    train_keys_raw = train_keys_raw.to(device)
    train_values = train_values.to(device)
    q_proj, k_proj = image_proj(query_raw, train_keys_raw)
    if variant == "centroid":
        bank = build_projected_bank("D1_ClassCentroid", k_proj.detach().cpu(), train_values.detach().cpu())
        return score_prototype_variant("D1_ClassCentroid", q_proj, bank)[0].detach().cpu()
    if variant == "hqap4":
        bank = build_projected_bank("D3_HQAP4", k_proj.detach().cpu(), train_values.detach().cpu())
        return score_prototype_variant("D3_HQAP4", q_proj, bank)[0].detach().cpu()
    raise ValueError(variant)


def fit_source_calibrator(
    variant: str,
    image_keys: torch.Tensor,
    image_values: torch.Tensor,
    image_proj,
    device: torch.device,
    *,
    folds: int = 5,
    seed: int = 0,
) -> AffineCalibrator:
    labels = image_values.argmax(dim=-1).long().cpu()
    fold_indices = stratified_folds(labels, folds, seed)
    all_scores = torch.full((labels.numel(),), float("nan"))
    fold_rows: List[Dict] = []
    fold_a, fold_b = [], []
    all_idx = torch.arange(labels.numel())
    for fold_id, valid_idx in enumerate(fold_indices):
        train_mask = torch.ones(labels.numel(), dtype=torch.bool)
        train_mask[valid_idx] = False
        train_idx = all_idx[train_mask]
        scores = score_source_variant(variant, image_keys[valid_idx], image_keys[train_idx], image_values[train_idx], image_proj, device)
        all_scores[valid_idx] = scores
        fa, fb = fit_affine(scores.numpy(), labels[valid_idx].numpy())
        fold_a.append(fa)
        fold_b.append(fb)
        fold_rows.append({
            "fold": fold_id,
            "variant": variant,
            "valid_count": int(valid_idx.numel()),
            "normal_count": int((labels[valid_idx] == 0).sum().item()),
            "anomaly_count": int((labels[valid_idx] == 1).sum().item()),
            "a": fa,
            "b": fb,
            "score_mean": float(scores.mean().item()),
            "label_mean": float(labels[valid_idx].float().mean().item()),
        })
    a, b = fit_affine(all_scores.numpy(), labels.numpy())
    split_payload = "|".join(",".join(map(str, x.tolist())) for x in fold_indices)
    import hashlib

    return AffineCalibrator(
        a=a,
        b=b,
        fold_variance_a=float(np.var(fold_a)) if fold_a else 0.0,
        fold_variance_b=float(np.var(fold_b)) if fold_b else 0.0,
        source_scores=all_scores.tolist(),
        source_labels=labels.tolist(),
        fold_rows=fold_rows,
        source_split_hash=hashlib.sha256(split_payload.encode("utf-8")).hexdigest(),
    )
