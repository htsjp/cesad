"""Offline MRAD image-score-policy audit.

Reuses official MRAD raw-map caches and their original image scores.  The only
new score is CESAD's existing top-1% mean map score; there is no model loading,
training, target calibration, or pixel-level metric calculation.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(r"D:\ProjectPython\MRAD-main\MRAD-main")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_dagm_visa_eval import DAGMTargetDataset
from experiments.phase_mrad_dtd_close import load_dtd_metas
from experiments.phase_mva_mrad_same_protocol_visa_factorization import load_metas_direct
from experiments.phase_xlv_cross_domain_transfer import top1_image_score


OUT = ROOT / "results" / "phase_mrad_image_score_unification"
MVA_CACHE = ROOT / "results" / "phase_mva_mrad_same_protocol_visa_factorization" / "formal" / "MAP_CACHE"
DTD_CACHE = ROOT / "results" / "paper_assets" / "mrad_dtd_close" / "MAP_CACHE"
DAGM_CACHE = ROOT / "results" / "paper_assets" / "mrad_dagm_close" / "MAP_CACHE"
VARIANTS = (("M1_MRAD_TF", "MRAD-TF"), ("M2_MRAD_FT", "MRAD-FT"), ("M3_MRAD_CLIP", "MRAD-CLIP"))
CELLS = (
    ("Full MVTec", "VisA", "mva"),
    ("Full VisA", "MVTec", "mva"),
    ("Full VisA", "BTAD", "mva"),
    ("Full VisA", "MPDD", "mva"),
    ("Full VisA", "DTD-Synthetic", "dtd"),
    ("Full VisA", "DAGM 2007", "dagm"),
)


def slug(source: str, target: str) -> str:
    return f"{source.replace('Full ', 'Full_').replace(' ', '_')}__{target.replace(' ', '_')}"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def labels_and_categories(source: str, target: str, kind: str) -> tuple[np.ndarray, list[str]]:
    if kind == "mva":
        metas = load_metas_direct(source, target, device="cpu", smoke=False)
        return np.asarray([int(m["label"]) for m in metas]), [str(m["category"]) for m in metas]
    if kind == "dtd":
        metas = load_dtd_metas(smoke=False)
        return np.asarray([int(m["label"]) for m in metas]), [str(m["category"]) for m in metas]
    if kind == "dagm":
        dataset = DAGMTargetDataset(Path(r"E:\WorkSpace\dataset\DAGM_KaggleUpload"))
        return np.asarray([int(x["anomaly"]) for x in dataset.items]), [str(x["cls_name"]) for x in dataset.items]
    raise ValueError(kind)


def cache_path(source: str, target: str, kind: str, variant: str) -> Path:
    if kind == "mva":
        return MVA_CACHE / f"{slug(source, target)}__{variant}.npz"
    if kind == "dtd":
        return DTD_CACHE / f"Full_VisA__DTD-Synthetic__{variant}.npz"
    if kind == "dagm":
        return DAGM_CACHE / f"Full_VisA__DAGM__{variant}_grid74.npz"
    raise ValueError(kind)


def metric(y: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y)) != 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, scores) * 100.0), float(average_precision_score(y, scores) * 100.0)


def category_macro(y: np.ndarray, scores: np.ndarray, categories: list[str]) -> tuple[float, float, int]:
    vals = []
    cat_arr = np.asarray(categories)
    for category in sorted(set(categories)):
        mask = cat_arr == category
        if np.unique(y[mask]).size == 2:
            vals.append(metric(y[mask], scores[mask]))
    if not vals:
        return float("nan"), float("nan"), 0
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean()), len(vals)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw_rows, per_category, manifest = [], [], []
    for source, target, kind in CELLS:
        y, cats = labels_and_categories(source, target, kind)
        for variant, method in VARIANTS:
            path = cache_path(source, target, kind, variant)
            if not path.exists():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=True) as payload:
                maps = np.asarray(payload["maps"], dtype=np.float32)
                official = np.asarray(payload["image_scores"], dtype=np.float32)
            if len(maps) != len(y) or len(official) != len(y):
                raise RuntimeError(f"Cache/meta length mismatch: {path} maps={len(maps)} scores={len(official)} labels={len(y)}")
            unified = np.asarray([top1_image_score(m) for m in maps], dtype=np.float32)
            off_auc, off_ap = metric(y, official)
            uni_auc, uni_ap = metric(y, unified)
            mac_off_auc, mac_off_ap, ncat = category_macro(y, official, cats)
            mac_uni_auc, mac_uni_ap, ncat2 = category_macro(y, unified, cats)
            if ncat != ncat2:
                raise RuntimeError("Inconsistent valid-category count")
            raw_rows.append({
                "source": source,
                "target": target,
                "method": method,
                "variant": variant,
                "n_images": int(len(y)),
                "n_anomaly_images": int(y.sum()),
                "n_categories_valid": ncat,
                "official_I_AUROC_dataset_global": off_auc,
                "official_I_AP_dataset_global": off_ap,
                "official_I_AUROC_category_macro": mac_off_auc,
                "official_I_AP_category_macro": mac_off_ap,
                "CESAD_top1pct_I_AUROC_dataset_global": uni_auc,
                "CESAD_top1pct_I_AP_dataset_global": uni_ap,
                "CESAD_top1pct_I_AUROC_category_macro": mac_uni_auc,
                "CESAD_top1pct_I_AP_category_macro": mac_uni_ap,
                "delta_macro_I_AUROC": mac_uni_auc - mac_off_auc,
                "delta_macro_I_AP": mac_uni_ap - mac_off_ap,
                "cache": str(path),
                "score_resolution": f"{maps.shape[1]}x{maps.shape[2]}",
            })
            carr = np.asarray(cats)
            for cat in sorted(set(cats)):
                idx = carr == cat
                oa, op = metric(y[idx], official[idx])
                ua, up = metric(y[idx], unified[idx])
                per_category.append({
                    "source": source, "target": target, "method": method, "category": cat,
                    "n_images": int(idx.sum()), "n_anomaly_images": int(y[idx].sum()),
                    "official_I_AUROC": oa, "official_I_AP": op,
                    "CESAD_top1pct_I_AUROC": ua, "CESAD_top1pct_I_AP": up,
                })
            manifest.append({"source": source, "target": target, "method": method, "raw_map_cache": str(path), "status": "REUSED"})

    write_csv(OUT / "MRAD_IMAGE_LEVEL_UNIFIED.csv", raw_rows)
    write_csv(OUT / "PER_CATEGORY_IMAGE_LEVEL.csv", per_category)
    write_csv(OUT / "CACHE_MANIFEST.csv", manifest)
    report = """# MRAD Image-Level Score Unification

This is a cache-only audit over fixed MRAD anomaly maps. It does not load a model, train, infer, or alter pixel-level metrics.

## Policy

The new image score exactly follows the CESAD helper `top1_image_score`:

`score(image) = mean(top max(1, floor(0.01 * H * W)) values of the stored MRAD anomaly map)`.

MRAD's stored official image score is retained as a reference in the same table. We report both dataset-global and category-macro image metrics, with category-macro serving as the paper-compatible aggregation where categories contain both normal and anomalous images.

## Cells

- Full MVTec -> VisA
- Full VisA -> MVTec, BTAD, MPDD, DTD-Synthetic, DAGM 2007

Each cell contains MRAD-TF, MRAD-FT, and MRAD-CLIP. Source assets are the existing explicit source-selected MRAD caches/checkpoints associated with the original raw map cache; no target normal reference or target-derived scoring statistic is introduced.

## Scope

Only `I-AUROC` and `I-AP` are computed. Pixel maps and their P-AUROC/P-AP/P-F1max/AUPRO outputs are deliberately untouched. DAGM's existing raw maps are 74x74 grid maps; its new score is the same top-1% function applied to that fixed cached grid.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    (OUT / "OUTCOME.json").write_text(json.dumps({"outcome": "MRAD_IMAGE_SCORE_POLICY_UNIFIED", "training": False, "inference": False, "pixel_metrics_recomputed": False, "policy": "top1_percent_mean_of_cached_anomaly_map"}, indent=2) + "\n", encoding="utf-8")
    print(pd := __import__("pandas"))
    frame = pd.DataFrame(raw_rows)
    print(frame[["source", "target", "method", "CESAD_top1pct_I_AUROC_category_macro", "CESAD_top1pct_I_AP_category_macro"]].to_string(index=False))


if __name__ == "__main__":
    main()
