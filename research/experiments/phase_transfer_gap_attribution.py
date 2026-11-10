import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_final_cross_dataset_benchmark import OUT as FINAL_OUT, load_mvtec_source, load_visa_source, protocol_cells, sota_reference
from experiments.phase_hard_normal_cross_source_mvtec import load_full_mvtec_source
from experiments.phase_spatial_residual_refinement_standard_metric import aligned_grid, fnum
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xxxviii_visual_state_representation import SIDE, norm_np, patch_labels
from experiments.phase_private_residual_contrast_memory import contrast_logits


OUT = Path("results/phase_transfer_gap_attribution/formal")
TARGETS = {"MVTec", "VisA", "BTAD", "MPDD", "DTD-Synthetic"}
MAIN_VARIANT = "M1_HardNormal"
RNG_SEED = 0
PCA_RANK = 32
SOURCE_NORMAL_CAP = 120_000
CATEGORY_NORMAL_CAP = 40_000
CATEGORY_ANOMALY_CAP = 20_000
PER_IMAGE_NORMAL_CAP = 96
PER_IMAGE_ANOMALY_CAP = 96


def json_default(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=json_default), encoding="utf-8")


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_unit(x):
    x = np.asarray(x, dtype=np.float64)
    n = float(np.linalg.norm(x))
    return x / max(n, 1e-12)


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12))


def cov_matrix(x):
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < 2:
        return np.full((x.shape[1], x.shape[1]), np.nan, dtype=np.float64)
    xc = x - x.mean(axis=0, keepdims=True)
    return (xc.T @ xc) / float(max(x.shape[0] - 1, 1))


def corr_pair(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) <= 0 or np.std(y[ok]) <= 0:
        return float("nan"), float("nan"), int(ok.sum())
    return float(stats.pearsonr(x[ok], y[ok]).statistic), float(stats.spearmanr(x[ok], y[ok]).statistic), int(ok.sum())


class Reservoir:
    def __init__(self, cap, dim, seed):
        self.cap = int(cap)
        self.dim = int(dim)
        self.rng = np.random.default_rng(seed)
        self.data = np.empty((self.cap, self.dim), dtype=np.float32)
        self.n_seen = 0
        self.n_keep = 0

    def add(self, arr):
        arr = np.asarray(arr, dtype=np.float32).reshape(-1, self.dim)
        for row in arr:
            self.n_seen += 1
            if self.n_keep < self.cap:
                self.data[self.n_keep] = row
                self.n_keep += 1
            else:
                j = int(self.rng.integers(0, self.n_seen))
                if j < self.cap:
                    self.data[j] = row

    def array(self):
        return self.data[: self.n_keep].astype(np.float32, copy=False)


def source_dataset(source_name, smoke):
    if source_name == "Full VisA":
        _name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    else:
        _name, dataset, indices, patch = load_full_mvtec_source(False)
    if smoke:
        by_cat = defaultdict(list)
        for pos, idx in enumerate(indices):
            by_cat[str(dataset.data_all[int(idx)]["cls_name"])].append((pos, idx))
        chosen = []
        for cat in sorted(by_cat)[:4]:
            chosen.extend(by_cat[cat][:8])
        patch = np.stack([np.asarray(patch[p], dtype=np.float32) for p, _idx in chosen], axis=0)
        indices = [idx for _p, idx in chosen]
    return dataset, list(indices), patch


def transform_patch_grid(patch_grid, source_info, device):
    if source_info["use_align"]:
        with torch.no_grad():
            return aligned_grid(patch_grid, source_info["q_pc1"], source_info["align_model"], device).detach().cpu().numpy().reshape(SIDE * SIDE, -1).astype(np.float32)
    return norm_np(np.asarray(patch_grid, dtype=np.float32).reshape(SIDE * SIDE, -1)).astype(np.float32)


def sample_ids(ids, cap, rng):
    ids = np.asarray(ids, dtype=np.int64)
    if ids.size <= cap:
        return ids
    return rng.choice(ids, size=int(cap), replace=False)


def category_add(cat_store, category, normal, anomaly):
    st = cat_store.setdefault(
        category,
        {
            "normal": Reservoir(CATEGORY_NORMAL_CAP, normal.shape[1] if normal.size else anomaly.shape[1], RNG_SEED + abs(hash(category)) % 10000),
            "anomaly": Reservoir(CATEGORY_ANOMALY_CAP, normal.shape[1] if normal.size else anomaly.shape[1], RNG_SEED + 13 + abs(hash(category)) % 10000),
            "n_images": 0,
            "n_anomaly_images": 0,
            "normal_seen": 0,
            "anomaly_seen": 0,
        },
    )
    if normal.size:
        st["normal"].add(normal)
        st["normal_seen"] += int(normal.shape[0])
    if anomaly.size:
        st["anomaly"].add(anomaly)
        st["anomaly_seen"] += int(anomaly.shape[0])
        st["n_anomaly_images"] += 1
    st["n_images"] += 1


def collect_source_statistics(source_info, device, smoke):
    cache = OUT / f"{source_info['source'].replace(' ', '_')}_SOURCE_STATS_{'smoke' if smoke else 'formal'}.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        return {
            "muN": data["muN"],
            "muA": data["muA"],
            "ds": data["ds"],
            "U32": data["U32"],
            "cov32": data["cov32"],
            "normal_q99": float(data["normal_q99"]),
            "normal_q95": float(data["normal_q95"]),
            "normal_q995": float(data["normal_q995"]),
            "normal_scores": data["normal_scores"],
            "audit": data["audit"].item(),
        }
    dataset, indices, patch = source_dataset(source_info["source"], smoke)
    dim = np.asarray(patch[0]).reshape(-1, np.asarray(patch[0]).shape[-1]).shape[-1]
    normal_res = Reservoir(SOURCE_NORMAL_CAP, dim, RNG_SEED)
    anomaly_res = Reservoir(SOURCE_NORMAL_CAP // 4, dim, RNG_SEED + 1)
    normal_scores = Reservoir(SOURCE_NORMAL_CAP, 1, RNG_SEED + 2)
    rng = np.random.default_rng(RNG_SEED)
    m = torch.tensor(source_info["m"], dtype=torch.float32, device=device)
    v = torch.tensor(source_info["v1"], dtype=torch.float32, device=device)
    n_images = min(48, len(indices)) if smoke else len(indices)
    for pos in range(n_images):
        item = dataset[int(indices[pos])]
        y = patch_labels(item).astype(np.int64)
        z = transform_patch_grid(patch[pos], source_info, device)
        n_ids = sample_ids(np.flatnonzero(y == 0), PER_IMAGE_NORMAL_CAP, rng)
        a_ids = sample_ids(np.flatnonzero(y == 1), PER_IMAGE_ANOMALY_CAP, rng)
        if n_ids.size:
            zn = z[n_ids]
            normal_res.add(zn)
            with torch.no_grad():
                q = contrast_logits(torch.tensor(zn, dtype=torch.float32, device=device), m, source_info["alpha"], v).detach().cpu().numpy().reshape(-1, 1)
            normal_scores.add(q)
        if a_ids.size:
            anomaly_res.add(z[a_ids])
        if pos % 250 == 0:
            print(f"[TransferGap] source stats {source_info['source']} image {pos}/{n_images}", flush=True)
    zn = normal_res.array()
    za = anomaly_res.array()
    muN = zn.mean(axis=0)
    muA = za.mean(axis=0) if za.size else np.full_like(muN, np.nan)
    centered = zn - muN
    _, _s, vh = np.linalg.svd(centered, full_matrices=False)
    U32 = vh[:PCA_RANK].T.astype(np.float32)
    cov32 = cov_matrix(centered @ U32)
    qs = normal_scores.array().reshape(-1)
    out = {
        "muN": muN.astype(np.float32),
        "muA": muA.astype(np.float32),
        "ds": (muA - muN).astype(np.float32),
        "U32": U32.astype(np.float32),
        "cov32": cov32.astype(np.float32),
        "normal_q95": np.float32(np.percentile(qs, 95)),
        "normal_q99": np.float32(np.percentile(qs, 99)),
        "normal_q995": np.float32(np.percentile(qs, 99.5)),
        "normal_scores": qs.astype(np.float32),
        "audit": {
            "source": source_info["source"],
            "normal_seen": normal_res.n_seen,
            "normal_kept": normal_res.n_keep,
            "anomaly_seen": anomaly_res.n_seen,
            "anomaly_kept": anomaly_res.n_keep,
            "pca_rank": PCA_RANK,
            "source_normal_cap": SOURCE_NORMAL_CAP,
            "deterministic_sampling": True,
        },
    }
    np.savez_compressed(cache, **out)
    return out


def target_item(dataset, idx_or_rows, kind, pos):
    return dataset[int(idx_or_rows[pos])] if kind == "source_dataset" else dataset[pos]


def target_meta(dataset, idx_or_rows, kind, target, pos):
    item = target_item(dataset, idx_or_rows, kind, pos)
    if kind == "source_dataset":
        return str(item.get("cls_name", target)), int(item.get("anomaly", 0))
    row = dict(idx_or_rows[pos])
    return str(row.get("category", item.get("cls_name", target))), int(row.get("label", item.get("anomaly", 0)))


def collect_target_category_statistics(source_info, source_stats, device, smoke):
    cache = OUT / f"{source_info['source'].replace(' ', '_')}_TARGET_CATEGORY_STATS_{'smoke' if smoke else 'formal'}.npz"
    if cache.exists():
        return np.load(cache, allow_pickle=True)["rows"].tolist()
    rows = []
    m = torch.tensor(source_info["m"], dtype=torch.float32, device=device)
    v = torch.tensor(source_info["v1"], dtype=torch.float32, device=device)
    cells, _audits = protocol_cells(source_info["source"], device, smoke)
    for _cell_id, target, dataset, idx_or_rows, patch, kind in cells:
        if target not in TARGETS:
            continue
        n = min(48, len(idx_or_rows)) if smoke else len(idx_or_rows)
        cat_store = {}
        score_store = defaultdict(list)
        rng = np.random.default_rng(RNG_SEED + len(target))
        print(f"[TransferGap] target stats {source_info['source']} -> {target} n={n}", flush=True)
        for pos in range(n):
            item = target_item(dataset, idx_or_rows, kind, pos)
            cat, _label = target_meta(dataset, idx_or_rows, kind, target, pos)
            y = patch_labels(item).astype(np.int64)
            z = transform_patch_grid(patch[pos], source_info, device)
            n_ids = sample_ids(np.flatnonzero(y == 0), PER_IMAGE_NORMAL_CAP, rng)
            a_ids = sample_ids(np.flatnonzero(y == 1), PER_IMAGE_ANOMALY_CAP, rng)
            zn = z[n_ids] if n_ids.size else np.empty((0, z.shape[1]), dtype=np.float32)
            za = z[a_ids] if a_ids.size else np.empty((0, z.shape[1]), dtype=np.float32)
            category_add(cat_store, cat, zn, za)
            if zn.size:
                with torch.no_grad():
                    q = contrast_logits(torch.tensor(zn, dtype=torch.float32, device=device), m, source_info["alpha"], v).detach().cpu().numpy()
                score_store[cat].append(q.astype(np.float32))
            if pos % 250 == 0:
                print(f"[TransferGap] target stats {source_info['source']} -> {target} image {pos}/{n}", flush=True)
        for cat, st in sorted(cat_store.items()):
            zn = st["normal"].array()
            za = st["anomaly"].array()
            if zn.shape[0] < 2:
                continue
            muN = zn.mean(axis=0)
            muA = za.mean(axis=0) if za.shape[0] else np.full_like(muN, np.nan)
            dt = muA - muN
            zproj = (zn - muN) @ source_stats["U32"]
            cov_gap = float(np.linalg.norm(cov_matrix(zproj) - source_stats["cov32"], ord="fro"))
            score = np.concatenate(score_store[cat], axis=0).reshape(-1) if score_store[cat] else np.asarray([], dtype=np.float32)
            trace_cov = float(np.trace(cov_matrix(zn)))
            rows.append({
                "source": source_info["source"],
                "dataset": target,
                "category": cat,
                "n_images": int(st["n_images"]),
                "n_anomaly_images": int(st["n_anomaly_images"]),
                "normal_seen": int(st["normal_seen"]),
                "normal_kept": int(zn.shape[0]),
                "anomaly_seen": int(st["anomaly_seen"]),
                "anomaly_kept": int(za.shape[0]),
                "NORMAL_MEAN_GAP": 1.0 - cosine(source_stats["muN"], muN),
                "NORMAL_COV_GAP": cov_gap,
                "SOURCE_NORMAL_Q95": float(source_stats["normal_q95"]),
                "SOURCE_NORMAL_Q99": float(source_stats["normal_q99"]),
                "SOURCE_NORMAL_Q99_5": float(source_stats["normal_q995"]),
                "TARGET_NORMAL_MEAN_SCORE": float(score.mean()) if score.size else float("nan"),
                "TARGET_NORMAL_STD_SCORE": float(score.std()) if score.size else float("nan"),
                "TARGET_NORMAL_Q95": float(np.percentile(score, 95)) if score.size else float("nan"),
                "TARGET_NORMAL_Q99": float(np.percentile(score, 99)) if score.size else float("nan"),
                "TARGET_NORMAL_Q99_5": float(np.percentile(score, 99.5)) if score.size else float("nan"),
                "NORMAL_TAIL_SHIFT": float(np.percentile(score, 99) - source_stats["normal_q99"]) if score.size else float("nan"),
                "ANOMALY_DIRECTION_ALIGNMENT": cosine(source_stats["ds"], dt) if za.shape[0] else float("nan"),
                "ANOMALY_DIRECTION_GAP": 1.0 - cosine(source_stats["ds"], dt) if za.shape[0] else float("nan"),
                "SOURCE_ANOMALY_DIRECTION_NORM": float(np.linalg.norm(source_stats["ds"])),
                "TARGET_ANOMALY_DIRECTION_NORM": float(np.linalg.norm(dt)) if za.shape[0] else float("nan"),
                "ANOMALY_SALIENCE_L2": float(np.linalg.norm(dt)) if za.shape[0] else float("nan"),
                "ANOMALY_SALIENCE_EFFECT": float(np.linalg.norm(dt) / math.sqrt(max(trace_cov, 1e-12))) if za.shape[0] else float("nan"),
            })
    np.savez_compressed(cache, rows=np.asarray(rows, dtype=object))
    return rows


def metric_rows():
    per = read_csv(FINAL_OUT / "PER_CATEGORY_STANDARD_METRICS.csv")
    return [r for r in per if r.get("variant") == MAIN_VARIANT and r.get("dataset") in TARGETS]


def morphology_rows():
    path = Path("results/phase_final_bottleneck_attribution/formal/CATEGORY_MORPHOLOGY.csv")
    return read_csv(path) if path.exists() else []


def ref_lookup():
    refs = {}
    for r in sota_reference():
        if r.get("comparison_type") != "DIRECT_COMPARABLE":
            continue
        refs[(r.get("training_source"), r.get("target"))] = r
    return refs


def join_rows(feature_rows, metrics, morph):
    metric_by = {(r["source"], r["dataset"], r["category"]): r for r in metrics}
    morph_by = {(r["source"], r["dataset"], r["category"]): r for r in morph}
    refs = ref_lookup()
    out = []
    for r in feature_rows:
        key = (r["source"], r["dataset"], r["category"])
        m = metric_by.get(key, {})
        mo = morph_by.get(key, {})
        ref = refs.get((r["source"], r["dataset"]))
        row = dict(r)
        row.update({
            "OUR_PAP": fnum(m, "P-AP"),
            "OUR_P_AUROC": fnum(m, "P-AUROC"),
            "OUR_P_F1max": fnum(m, "P-F1max"),
            "OUR_PRO": fnum(m, "PRO"),
            "reference_scope": "DATASET_LEVEL_DIRECT_COMPARABLE" if ref else "NO_DIRECT_REFERENCE",
            "REFERENCE_PAP": fnum(ref, "P_AP") if ref else float("nan"),
            "RELATIVE_GAP": fnum(m, "P-AP") - fnum(ref, "P_AP") if ref else float("nan"),
        })
        for name in (
            "anomaly_area_ratio_mean",
            "anomaly_patch_ratio_mean",
            "num_anomaly_patches_mean",
            "mean_component_area_mean",
            "boundary_to_area_ratio_mean",
            "thinness_mean",
        ):
            row[name] = fnum(mo, name)
        out.append(row)
    return out


def correlation_matrix(rows):
    predictors = [
        "NORMAL_MEAN_GAP",
        "NORMAL_COV_GAP",
        "NORMAL_TAIL_SHIFT",
        "ANOMALY_DIRECTION_GAP",
        "ANOMALY_SALIENCE_EFFECT",
        "anomaly_area_ratio_mean",
        "anomaly_patch_ratio_mean",
        "thinness_mean",
    ]
    targets = ["OUR_PAP", "RELATIVE_GAP"]
    out = []
    for y in targets:
        for x in predictors:
            pear, spear, n = corr_pair([r.get(x, float("nan")) for r in rows], [r.get(y, float("nan")) for r in rows])
            out.append({"target": y, "factor": x, "n": n, "pearson": pear, "spearman": spear})
        for source in sorted({r["source"] for r in rows}):
            part = [r for r in rows if r["source"] == source]
            for x in predictors:
                pear, spear, n = corr_pair([r.get(x, float("nan")) for r in part], [r.get(y, float("nan")) for r in part])
                out.append({"target": y, "factor": x, "source_protocol": source, "n": n, "pearson": pear, "spearman": spear})
    return out


def attribution_regression(rows):
    predictors = [
        "NORMAL_MEAN_GAP",
        "NORMAL_COV_GAP",
        "NORMAL_TAIL_SHIFT",
        "ANOMALY_DIRECTION_GAP",
        "ANOMALY_SALIENCE_EFFECT",
        "anomaly_area_ratio_mean",
        "thinness_mean",
    ]
    usable = []
    for r in rows:
        vals = [fnum(r, p) for p in predictors]
        y = fnum(r, "RELATIVE_GAP")
        if np.isfinite(y) and all(np.isfinite(v) for v in vals):
            usable.append((vals, y))
    if len(usable) < len(predictors) + 2:
        return [{"status": "INSUFFICIENT_CATEGORY_LEVEL_DIRECT_REFERENCE", "n": len(usable), "lambda": 1.0}]
    X = np.asarray([u[0] for u in usable], dtype=np.float64)
    y = np.asarray([u[1] for u in usable], dtype=np.float64)
    Xm = X.mean(axis=0)
    Xs = X.std(axis=0)
    Xz = (X - Xm) / np.maximum(Xs, 1e-12)
    yz = y - y.mean()
    model = Ridge(alpha=1.0, fit_intercept=True)
    model.fit(Xz, yz)
    pred = model.predict(Xz)
    ss_res = float(np.sum((yz - pred) ** 2))
    ss_tot = float(np.sum((yz - yz.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    rows_out = [{"status": "OK", "n": len(usable), "lambda": 1.0, "R2": r2}]
    for p, c in zip(predictors, model.coef_):
        rows_out.append({"status": "coefficient", "factor": p, "standardized_coefficient": float(c), "lambda": 1.0, "R2": r2})
    return rows_out


def dataset_summary(rows):
    fields = [
        "OUR_PAP",
        "RELATIVE_GAP",
        "NORMAL_MEAN_GAP",
        "NORMAL_COV_GAP",
        "NORMAL_TAIL_SHIFT",
        "ANOMALY_DIRECTION_ALIGNMENT",
        "ANOMALY_SALIENCE_EFFECT",
        "anomaly_patch_ratio_mean",
        "thinness_mean",
    ]
    out = []
    for source, dataset in sorted({(r["source"], r["dataset"]) for r in rows}):
        part = [r for r in rows if r["source"] == source and r["dataset"] == dataset]
        row = {"source": source, "dataset": dataset, "n_categories": len(part)}
        for f in fields:
            vals = np.asarray([fnum(r, f) for r in part], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            row[f"{f}_mean"] = float(vals.mean()) if vals.size else float("nan")
            row[f"{f}_median"] = float(np.median(vals)) if vals.size else float("nan")
        out.append(row)
    return out


def select_outcome(corr_rows):
    def spear(target, factor, source=None):
        vals = [
            fnum(r, "spearman")
            for r in corr_rows
            if r.get("target") == target and r.get("factor") == factor and (source is None and not r.get("source_protocol") or r.get("source_protocol") == source)
        ]
        return vals[0] if vals else float("nan")

    rel_factors = ("NORMAL_TAIL_SHIFT", "ANOMALY_DIRECTION_GAP")
    rel_ok = False
    for factor in rel_factors:
        all_rho = spear("RELATIVE_GAP", factor)
        src = [spear("RELATIVE_GAP", factor, s) for s in ("Full VisA", "Full MVTec")]
        same_sign = all(np.isfinite(x) for x in src) and np.sign(src[0]) == np.sign(src[1])
        if np.isfinite(all_rho) and abs(all_rho) >= 0.5 and same_sign:
            rel_ok = True
    if rel_ok:
        return "DOMAIN_OR_ANOMALY_GAP_EXPLAINS_TRANSFER_IMBALANCE"

    morph = max(
        abs(spear("OUR_PAP", "anomaly_area_ratio_mean")),
        abs(spear("OUR_PAP", "anomaly_patch_ratio_mean")),
        abs(spear("OUR_PAP", "thinness_mean")),
    )
    non_morph = max(
        abs(spear("OUR_PAP", "NORMAL_MEAN_GAP")),
        abs(spear("OUR_PAP", "NORMAL_COV_GAP")),
        abs(spear("OUR_PAP", "NORMAL_TAIL_SHIFT")),
        abs(spear("OUR_PAP", "ANOMALY_DIRECTION_GAP")),
        abs(spear("OUR_PAP", "ANOMALY_SALIENCE_EFFECT")),
    )
    if np.isfinite(morph) and morph >= 0.4 and (not np.isfinite(non_morph) or morph >= non_morph):
        return "MORPHOLOGY_REMAINS_PRIMARY_TRANSFER_BOTTLENECK"
    return "OBSERVED_DATASET_IMBALANCE_NOT_EXPLAINED_BY_TESTED_FACTORS"


def write_report(outcome, rows, corr_rows, reg_rows, summary_rows, runtime, audits):
    lines = [
        "# Transfer Gap Attribution Audit",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "NO-TRAIN / NO-TUNING. The audit freezes the current Canonical Two-State + NTR checkpoints and uses target GT only for post-hoc category diagnostics.",
        "",
        "## Scope Notes",
        "",
        "- Main analysis unit: source-target-category.",
        "- Category-level direct SOTA references are not available in the current registry; dataset-level direct references are recorded but are not treated as category-specific leaderboard numbers.",
        f"- Source normal PCA rank: {PCA_RANK}. Feature covariance/PCA uses deterministic capped sampling to keep the audit bounded; caps are recorded in `NORMAL_DOMAIN_GAP.csv`.",
        "",
        "## Strongest Correlations",
        "",
        "| Target | Factor | N | Pearson | Spearman |",
        "|---|---|---:|---:|---:|",
    ]
    focus = {
        "NORMAL_MEAN_GAP",
        "NORMAL_COV_GAP",
        "NORMAL_TAIL_SHIFT",
        "ANOMALY_DIRECTION_GAP",
        "ANOMALY_SALIENCE_EFFECT",
        "anomaly_patch_ratio_mean",
        "thinness_mean",
    }
    for r in corr_rows:
        if r.get("target") == "OUR_PAP" and r.get("factor") in focus and not r.get("source_protocol"):
            lines.append(f"| {r['target']} | {r['factor']} | {r['n']} | {fnum(r,'pearson'):.4f} | {fnum(r,'spearman'):.4f} |")
    lines.extend(["", "## Dataset Summary", "", "| Source | Dataset | P-AP | Rel Gap | Normal Tail Shift | Direction Align | Salience | Patch Ratio | Thinness |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for r in summary_rows:
        lines.append(
            f"| {r['source']} | {r['dataset']} | {fnum(r,'OUR_PAP_mean'):.4f} | {fnum(r,'RELATIVE_GAP_mean'):.4f} | "
            f"{fnum(r,'NORMAL_TAIL_SHIFT_mean'):.4f} | {fnum(r,'ANOMALY_DIRECTION_ALIGNMENT_mean'):.4f} | "
            f"{fnum(r,'ANOMALY_SALIENCE_EFFECT_mean'):.4f} | {fnum(r,'anomaly_patch_ratio_mean_mean'):.4f} | {fnum(r,'thinness_mean_mean'):.4f} |"
        )
    lines.extend(["", "## Attribution Regression", ""])
    if reg_rows and reg_rows[0].get("status") == "OK":
        lines.append(f"Ridge regression on relative gap used n={reg_rows[0]['n']} category observations, fixed lambda=1.0, R2={fnum(reg_rows[0], 'R2'):.4f}.")
    else:
        lines.append("Ridge regression on relative gap was not run as a decisive model because reliable category-level direct-reference gaps are insufficient.")
    lines.extend(["", "## Interpretation", ""])
    if outcome == "DOMAIN_OR_ANOMALY_GAP_EXPLAINS_TRANSFER_IMBALANCE":
        lines.append("Normal tail shift or anomaly direction gap passes the relative-gap consistency gate, so source-only stress training is the next permissible direction.")
    elif outcome == "MORPHOLOGY_REMAINS_PRIMARY_TRANSFER_BOTTLENECK":
        lines.append("Morphology, especially sparse patch occupancy / thinness, remains the clearest explanation of uneven transfer. This keeps the earlier small/sparse localization bottleneck alive.")
    else:
        lines.append("The tested factors do not isolate one stable driver of target imbalance. This argues against opening another mechanism branch from this audit alone.")
    lines.extend(["", "## Source Sampling Audit", "", "```json", json.dumps(audits, indent=2, default=json_default), "```", "", f"Runtime: {runtime:.1f}s."])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    source_infos = [load_visa_source(device), load_mvtec_source()]
    feature_rows, source_audits = [], []
    for info in source_infos:
        align_model = info.get("align_model")
        if align_model is not None:
            for p in align_model.parameters():
                p.requires_grad_(False)
        stats_src = collect_source_statistics(info, device, args.smoke)
        source_audits.append(stats_src["audit"])
        feature_rows.extend(collect_target_category_statistics(info, stats_src, device, args.smoke))

    metrics = metric_rows()
    morph = morphology_rows()
    joined = join_rows(feature_rows, metrics, morph)
    corr = correlation_matrix(joined)
    reg = attribution_regression(joined)
    ds_sum = dataset_summary(joined)
    outcome = select_outcome(corr)
    runtime = time.time() - start

    normal_rows = [
        {k: r[k] for k in r if k.startswith("NORMAL_") or k in {"source", "dataset", "category", "normal_seen", "normal_kept", "n_images"}}
        for r in feature_rows
    ]
    direction_rows = [
        {k: r[k] for k in r if k.startswith("ANOMALY_DIRECTION") or k.startswith("SOURCE_ANOMALY") or k.startswith("TARGET_ANOMALY") or k in {"source", "dataset", "category", "anomaly_seen", "anomaly_kept"}}
        for r in feature_rows
    ]
    salience_rows = [
        {k: r[k] for k in r if k.startswith("ANOMALY_SALIENCE") or k in {"source", "dataset", "category", "anomaly_seen", "anomaly_kept"}}
        for r in feature_rows
    ]
    morph_joined = [
        {k: r.get(k, "") for k in ("source", "dataset", "category", "OUR_PAP", "RELATIVE_GAP", "anomaly_area_ratio_mean", "anomaly_patch_ratio_mean", "num_anomaly_patches_mean", "mean_component_area_mean", "boundary_to_area_ratio_mean", "thinness_mean")}
        for r in joined
    ]

    write_csv(OUT / "CATEGORY_GAP_METRICS.csv", joined)
    write_csv(OUT / "DATASET_GAP_SUMMARY.csv", ds_sum)
    write_csv(OUT / "NORMAL_DOMAIN_GAP.csv", normal_rows)
    write_csv(OUT / "ANOMALY_DIRECTION_GAP.csv", direction_rows)
    write_csv(OUT / "ANOMALY_SALIENCE.csv", salience_rows)
    write_csv(OUT / "MORPHOLOGY_JOINED.csv", morph_joined)
    write_csv(OUT / "CORRELATION_MATRIX.csv", corr)
    write_csv(OUT / "ATTRIBUTION_REGRESSION.csv", reg)
    write_json(OUT / "OUTCOME.json", {"outcome": outcome, "runtime_sec": runtime, "device": device, "no_train": True, "no_tuning": True, "main_variant": MAIN_VARIANT, "category_observations": len(joined)})
    write_report(outcome, joined, corr, reg, ds_sum, runtime, source_audits)
    print(f"[TransferGap] OUTCOME={outcome}", flush=True)
    print(f"[TransferGap] wrote {OUT}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
