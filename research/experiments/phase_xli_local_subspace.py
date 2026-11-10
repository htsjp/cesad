import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_xa_smoke import write_csv
from experiments.phase_xxxviii_visual_state_representation import (
    FEATURE_CACHE,
    IMAGE_SIZE,
    SIDE,
    class_category_patch_sample,
    group_gain,
    labels_and_rows,
    load_fold,
    norm_np,
    patch_labels,
    safe_ap,
    safe_auc,
    upsample_map,
)
from experiments.phase_xxxix_local_multistate import evaluate_variant


OUT = Path("results/phase_xli_local_subspace")
XXXIX_OUT = Path("results/phase_xxxix_local_multistate")
PILOT = [("D1", 0), ("D2", 0)]
SEED = 0
PER_CATEGORY_PATCH_CAP = 50_000
LEAK_CAP_PER_CLASS_CATEGORY = 512
CHUNK = 4096


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
    path.write_text(json.dumps(data, indent=2, default=json_default), encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def gp(row):
    return 0.5 * (float(row["P-AUROC"]) + float(row["P-AP"]))


def source_patch_indices_by_category(dataset, indices, patches, wanted_label, cap, seed):
    rng = np.random.default_rng(seed)
    buckets = defaultdict(list)
    for pos, idx in enumerate(indices):
        item = dataset[idx]
        labels = patch_labels(item)
        ids = np.flatnonzero(labels == wanted_label)
        if ids.size == 0:
            continue
        buckets[str(item["cls_name"])].append((pos, ids.astype(np.int64)))
    selected = {}
    for cat, entries in sorted(buckets.items()):
        flat_count = sum(len(ids) for _pos, ids in entries)
        if flat_count <= cap:
            chosen = [(pos, ids) for pos, ids in entries]
        else:
            global_ids = rng.choice(flat_count, size=cap, replace=False)
            global_ids.sort()
            chosen, offset, ptr = [], 0, 0
            for pos, ids in entries:
                end = offset + len(ids)
                local_global = global_ids[(global_ids >= offset) & (global_ids < end)] - offset
                if local_global.size:
                    chosen.append((pos, ids[local_global]))
                offset = end
        selected[cat] = chosen
    return selected


def iter_selected_patches(patches, selected):
    for cat, entries in sorted(selected.items()):
        for pos, ids in entries:
            arr = np.asarray(patches[pos, ids], dtype=np.float32)
            arr = norm_np(arr)
            yield cat, arr


def fit_subspace(patches, selected, ev_threshold=0.99):
    dim = int(patches.shape[-1])
    n = 0
    sum_x = np.zeros((dim,), dtype=np.float64)
    sum_xx = np.zeros((dim, dim), dtype=np.float64)
    per_cat = {}
    for cat, arr in iter_selected_patches(patches, selected):
        x = arr.astype(np.float64, copy=False)
        n += x.shape[0]
        per_cat[cat] = per_cat.get(cat, 0) + int(x.shape[0])
        sum_x += x.sum(axis=0)
        sum_xx += x.T @ x
    if n < 2:
        raise RuntimeError("not enough patches for PCA")
    mu = (sum_x / n).astype(np.float32)
    cov = (sum_xx / n) - np.outer(sum_x / n, sum_x / n)
    cov = (cov + cov.T) * 0.5
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order].astype(np.float32)
    total = float(eigvals.sum())
    csum = np.cumsum(eigvals)
    rank = int(np.searchsorted(csum / max(total, 1e-12), ev_threshold) + 1)
    basis = eigvecs[:, :rank].astype(np.float32)
    evr = float(csum[rank - 1] / max(total, 1e-12))
    return {"mu": mu, "basis": basis, "eigvals": eigvals.astype(np.float32), "rank": rank, "evr": evr, "n": int(n), "per_category": per_cat}


def residual_scores(flat, subspace):
    x = norm_np(np.asarray(flat, dtype=np.float32))
    mu = subspace["mu"].astype(np.float32)
    basis = subspace["basis"].astype(np.float32)
    out = np.zeros((x.shape[0],), dtype=np.float32)
    for start in range(0, x.shape[0], CHUNK):
        c = x[start:start + CHUNK] - mu
        norm2 = np.sum(c * c, axis=1)
        proj = c @ basis
        rec = np.sum(proj * proj, axis=1)
        out[start:start + c.shape[0]] = norm2 - rec
    return out


def score_subspace_maps(held_patch, normal_subspace, anomaly_subspace=None):
    maps = np.zeros((held_patch.shape[0], SIDE, SIDE), dtype=np.float16)
    extra = []
    for i in range(held_patch.shape[0]):
        flat = np.asarray(held_patch[i], dtype=np.float32)
        en = residual_scores(flat, normal_subspace)
        if anomaly_subspace is None:
            score = en
            ea = np.full_like(en, np.nan)
        else:
            ea = residual_scores(flat, anomaly_subspace)
            score = en - ea
        maps[i] = score.reshape(SIDE, SIDE).astype(np.float16)
        extra.append({"EN": en, "EA": ea, "S": score})
    return maps, extra


def residual_distribution_rows(dataset, held_indices, extras, direction, fold, variant):
    rows = []
    vals = {"normal": defaultdict(list), "anomaly": defaultdict(list)}
    for pos, idx in enumerate(held_indices):
        lab = patch_labels(dataset[idx])
        kind = np.where(lab == 1, "anomaly", "normal")
        for key, arr in extras[pos].items():
            for group in ["normal", "anomaly"]:
                m = kind == group
                if np.any(m):
                    vals[group][key].append(arr[m])
    for key in ["EN", "EA", "S"]:
        n = np.concatenate(vals["normal"][key]) if vals["normal"][key] else np.asarray([], dtype=np.float32)
        a = np.concatenate(vals["anomaly"][key]) if vals["anomaly"][key] else np.asarray([], dtype=np.float32)
        for group, arr in [("normal", n), ("anomaly", a)]:
            finite = arr[np.isfinite(arr)]
            rows.append({
                "direction": direction, "fold": fold, "variant": variant, "score": key, "group": group,
                "count": int(finite.size),
                "mean": float(np.mean(finite)) if finite.size else float("nan"),
                "median": float(np.median(finite)) if finite.size else float("nan"),
                "q25": float(np.quantile(finite, 0.25)) if finite.size else float("nan"),
                "q75": float(np.quantile(finite, 0.75)) if finite.size else float("nan"),
                "q90": float(np.quantile(finite, 0.90)) if finite.size else float("nan"),
            })
        nf = n[np.isfinite(n)]
        af = a[np.isfinite(a)]
        if nf.size and af.size:
            pooled = math.sqrt((float(np.var(nf)) + float(np.var(af))) / 2.0 + 1e-12)
            rows.append({"direction": direction, "fold": fold, "variant": variant, "score": key, "group": "anomaly_vs_normal", "count": int(nf.size + af.size), "mean": float((np.mean(af) - np.mean(nf)) / pooled), "median": float(stats.ks_2samp(nf[:200000], af[:200000]).statistic), "q25": float("nan"), "q75": float("nan"), "q90": float("nan")})
    return rows


def category_leakage_rows(train_patch, dataset, train_indices, normal_subspace, anomaly_subspace, direction, fold):
    x, y, _pos, _pid, cats = sample_leakage(train_patch, dataset, train_indices)
    en = residual_scores(x, normal_subspace)
    rows = []
    rows.append(leak_row(cats, en.reshape(-1, 1), direction, fold, "S1_NORMAL_SUBSPACE", 1))
    if anomaly_subspace is not None:
        ea = residual_scores(x, anomaly_subspace)
        rows.append(leak_row(cats, np.stack([en, ea], axis=1), direction, fold, "S2_DUAL_SUBSPACE", 2))
    return rows


def sample_leakage(train_patch, dataset, train_indices):
    x, y, cats = class_category_patch_sample(dataset, train_indices, train_patch, LEAK_CAP_PER_CLASS_CATEGORY, seed=SEED + 11)
    return x, y, None, None, cats


def leak_row(cats, z, direction, fold, variant, dim):
    cats = np.asarray(cats)
    _, counts = np.unique(cats, return_counts=True)
    nfold = min(5, int(counts.min())) if counts.size else 0
    if nfold < 2:
        acc = bacc = float("nan")
    else:
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, class_weight="balanced", random_state=SEED))
        cv = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=SEED)
        acc = float(np.mean(cross_val_score(clf, z, cats, cv=cv, scoring="accuracy")) * 100.0)
        bacc = float(np.mean(cross_val_score(clf, z, cats, cv=cv, scoring="balanced_accuracy")) * 100.0)
    return {"direction": direction, "fold": fold, "variant": variant, "category_accuracy": acc, "balanced_category_accuracy": bacc, "dimensionality": dim}


def load_baseline_rows(direction, fold):
    rows = []
    raw = []
    with open(XXXIX_OUT / "RAW_METRICS.csv", newline="", encoding="utf-8") as fp:
        import csv
        raw = list(csv.DictReader(fp))
    mapping = {
        "P0_INSTANCE_MEMORY": "B0_INSTANCE_MEMORY",
        "P1_SINGLE_STATE": "B1_SINGLE_STATE",
        "P2_STATIC_KMEANS4_STATE": "B2_KMEANS4",
    }
    for r in raw:
        if r["direction"] == direction and int(r["fold"]) == fold and r["variant"] in mapping:
            rows.append({"direction": direction, "fold": fold, "dataset": r.get("dataset", ""), "variant": mapping[r["variant"]], "P-AUROC": float(r["P-AUROC"]), "P-AP": float(r["P-AP"]), "PRO": float(r.get("PRO", "nan")), "source": "phase_xxxix_raw_metrics"})
    return rows


def run_fold(direction, fold):
    start = time.time()
    dataset_name, dataset, rotation, train_indices, held_indices = load_fold(direction, fold)
    cache = np.load(FEATURE_CACHE / f"{direction}_fold{fold}_clip.npz", allow_pickle=True)
    train_patch, held_patch = cache["train_patch"], cache["held_patch"]
    normal_sel = source_patch_indices_by_category(dataset, train_indices, train_patch, 0, PER_CATEGORY_PATCH_CAP, SEED)
    anomaly_sel = source_patch_indices_by_category(dataset, train_indices, train_patch, 1, PER_CATEGORY_PATCH_CAP, SEED)
    normal_sub = fit_subspace(train_patch, normal_sel)
    anomaly_sub = fit_subspace(train_patch, anomaly_sel)
    zeros = np.zeros((len(held_indices),), dtype=np.float32)
    metrics = load_baseline_rows(direction, fold)
    dist_rows = []

    print(f"[XLI-A] {direction} fold{fold} score S1", flush=True)
    s1_maps, s1_extra = score_subspace_maps(held_patch, normal_sub, None)
    s1 = evaluate_variant(dataset, held_indices, zeros, s1_maps)
    metrics.append({"direction": direction, "fold": fold, "dataset": dataset_name, "variant": "S1_NORMAL_SUBSPACE", **{k: s1["mean"].get(k, float("nan")) for k in ["P-AUROC", "P-AP", "PRO"]}})
    dist_rows.extend(residual_distribution_rows(dataset, held_indices, s1_extra, direction, fold, "S1_NORMAL_SUBSPACE"))

    print(f"[XLI-A] {direction} fold{fold} score S2", flush=True)
    s2_maps, s2_extra = score_subspace_maps(held_patch, normal_sub, anomaly_sub)
    s2 = evaluate_variant(dataset, held_indices, zeros, s2_maps)
    metrics.append({"direction": direction, "fold": fold, "dataset": dataset_name, "variant": "S2_DUAL_SUBSPACE", **{k: s2["mean"].get(k, float("nan")) for k in ["P-AUROC", "P-AP", "PRO"]}})
    dist_rows.extend(residual_distribution_rows(dataset, held_indices, s2_extra, direction, fold, "S2_DUAL_SUBSPACE"))

    ranks = [
        {"direction": direction, "fold": fold, "subspace": "normal", "rank": normal_sub["rank"], "dim": int(train_patch.shape[-1]), "rank_over_dim": normal_sub["rank"] / float(train_patch.shape[-1]), "explained_variance": normal_sub["evr"], "patch_count": normal_sub["n"], "effectively_high_dimensional": bool(normal_sub["rank"] > 0.8 * train_patch.shape[-1]), "per_category_counts": json.dumps(normal_sub["per_category"])},
        {"direction": direction, "fold": fold, "subspace": "anomaly", "rank": anomaly_sub["rank"], "dim": int(train_patch.shape[-1]), "rank_over_dim": anomaly_sub["rank"] / float(train_patch.shape[-1]), "explained_variance": anomaly_sub["evr"], "patch_count": anomaly_sub["n"], "effectively_high_dimensional": bool(anomaly_sub["rank"] > 0.8 * train_patch.shape[-1]), "per_category_counts": json.dumps(anomaly_sub["per_category"])},
    ]
    leakage = category_leakage_rows(train_patch, dataset, train_indices, normal_sub, anomaly_sub, direction, fold)
    storage = [
        {"direction": direction, "fold": fold, "variant": "S1_NORMAL_SUBSPACE", "vectors": int(1 + normal_sub["rank"]), "bytes_float32": int((train_patch.shape[-1] * (1 + normal_sub["rank"])) * 4), "projection_flops_per_patch_approx": int(2 * train_patch.shape[-1] * normal_sub["rank"])},
        {"direction": direction, "fold": fold, "variant": "S2_DUAL_SUBSPACE", "vectors": int(2 + normal_sub["rank"] + anomaly_sub["rank"]), "bytes_float32": int((train_patch.shape[-1] * (2 + normal_sub["rank"] + anomaly_sub["rank"])) * 4), "projection_flops_per_patch_approx": int(2 * train_patch.shape[-1] * (normal_sub["rank"] + anomaly_sub["rank"]))},
    ]
    protocol = {"direction": direction, "fold": fold, "dataset": dataset_name, "train_categories": rotation["train_categories"], "heldout_categories": rotation["val_categories"], "normal_patch_cap_per_category": PER_CATEGORY_PATCH_CAP, "anomaly_patch_cap_per_category": PER_CATEGORY_PATCH_CAP, "pca_threshold": 0.99, "runtime_sec": time.time() - start}
    return {"metrics": metrics, "rank": ranks, "distributions": dist_rows, "leakage": leakage, "storage": storage, "protocol": protocol, "parity": {"direction": direction, "fold": fold, "baseline_source": "phase_xxxix_local_multistate/RAW_METRICS.csv", "passed": len(metrics) >= 5}}


def pilot_gate(rows):
    by = {(r["direction"], int(r["fold"]), r["variant"]): r for r in rows}
    out = {"candidates": {}, "continue_full": False, "strong_signal": False}
    for cand in ["S1_NORMAL_SUBSPACE", "S2_DUAL_SUBSPACE"]:
        vals_b2, vals_b1, closures = [], [], []
        for direction, fold in PILOT:
            c = by[(direction, fold, cand)]
            b0 = by[(direction, fold, "B0_INSTANCE_MEMORY")]
            b1 = by[(direction, fold, "B1_SINGLE_STATE")]
            b2 = by[(direction, fold, "B2_KMEANS4")]
            gb2 = gp(c) - gp(b2)
            gb1 = gp(c) - gp(b1)
            gap = gp(b0) - gp(b1)
            closure = gb1 / gap if gap > 0 else float("nan")
            vals_b2.append(gb2); vals_b1.append(gb1); closures.append(closure)
        cond_a = vals_b2[0] >= 0.50 and vals_b2[1] >= 0.50
        cond_b = vals_b1[0] >= 1.50 and vals_b1[1] >= 1.50 and min(closures) >= 0.25
        strong = all((gp(by[(d, f, "B0_INSTANCE_MEMORY")]) - gp(by[(d, f, cand)])) <= 2.0 for d, f in PILOT)
        out["candidates"][cand] = {"D1f0_vs_B2": vals_b2[0], "D2f0_vs_B2": vals_b2[1], "D1f0_vs_B1": vals_b1[0], "D2f0_vs_B1": vals_b1[1], "D1f0_gap_closure": closures[0], "D2f0_gap_closure": closures[1], "condition_A": cond_a, "condition_B": cond_b, "strong_signal": strong}
        out["continue_full"] = out["continue_full"] or cond_a or cond_b
        out["strong_signal"] = out["strong_signal"] or strong
    return out


def aggregate(payloads, start):
    metrics = [r for p in payloads for r in p["metrics"]]
    ranks = [r for p in payloads for r in p["rank"]]
    distributions = [r for p in payloads for r in p["distributions"]]
    leakage = [r for p in payloads for r in p["leakage"]]
    storage = [r for p in payloads for r in p["storage"]]
    protocols = [p["protocol"] for p in payloads]
    parity = [p["parity"] for p in payloads]
    gate = pilot_gate(metrics)

    write_json(OUT / "PARITY_AUDIT.json", {"passed": all(r["passed"] for r in parity), "folds": parity})
    write_json(OUT / "XLI_PROTOCOL.json", {"folds": protocols, "target_accessed": False, "pca": "source-only covariance eig, rank by 0.99 explained variance"})
    write_csv(OUT / "PILOT_METRICS.csv", metrics)
    write_json(OUT / "PILOT_GATE.json", gate)
    write_csv(OUT / "SUBSPACE_RANK.csv", ranks)
    write_csv(OUT / "RAW_METRICS.csv", metrics)
    by = {(r["direction"], int(r["fold"]), r["variant"]): r for r in metrics}
    gains = []
    gap_rows = []
    normal_dual = []
    for direction, fold in PILOT:
        b0 = by[(direction, fold, "B0_INSTANCE_MEMORY")]
        b1 = by[(direction, fold, "B1_SINGLE_STATE")]
        b2 = by[(direction, fold, "B2_KMEANS4")]
        for cand in ["S1_NORMAL_SUBSPACE", "S2_DUAL_SUBSPACE"]:
            c = by[(direction, fold, cand)]
            gains.append({"direction": direction, "fold": fold, "variant": cand, "G_pixel_vs_B0": gp(c) - gp(b0), "G_pixel_vs_B1": gp(c) - gp(b1), "G_pixel_vs_B2": gp(c) - gp(b2)})
            gap = gp(b0) - gp(b1)
            rec = gp(c) - gp(b1)
            gap_rows.append({"direction": direction, "fold": fold, "variant": cand, "gap_B0_minus_B1": gap, "recovery": rec, "gap_closure": rec / gap if gap > 0 else float("nan")})
        normal_dual.append({"direction": direction, "fold": fold, "S2_minus_S1_G_pixel": gp(by[(direction, fold, "S2_DUAL_SUBSPACE")]) - gp(by[(direction, fold, "S1_NORMAL_SUBSPACE")])})
    write_csv(OUT / "GROUPED_GAINS.csv", gains)
    write_csv(OUT / "GAP_CLOSURE.csv", gap_rows)
    write_csv(OUT / "NORMAL_VS_DUAL.csv", normal_dual)
    write_csv(OUT / "RESIDUAL_DISTRIBUTIONS.csv", distributions)
    write_csv(OUT / "CATEGORY_LEAKAGE.csv", leakage)
    write_csv(OUT / "STORAGE_COMPLEXITY.csv", storage)

    if not all(r["passed"] for r in parity):
        outcome = "XLI_A_IMPLEMENTATION_FAILURE"
    elif not gate["continue_full"]:
        outcome = "XLI_A_SUBSPACE_NO_CROSS_CATEGORY_SIGNAL"
    else:
        outcome = "XLI_A_IMPLEMENTATION_FAILURE"

    report = ["# Phase XLI-A Local Subspace Knowledge Representation Audit", "", f"Outcome: **{outcome}**", "", f"Pilot continuation gate: `{gate['continue_full']}`; strong signal: `{gate['strong_signal']}`.", "", "## Pilot Metrics", ""]
    for r in metrics:
        report.append(f"- {r['direction']} fold{r['fold']} {r['variant']}: P-AUROC {float(r['P-AUROC']):.4f}, P-AP {float(r['P-AP']):.4f}, PRO {float(r.get('PRO', float('nan'))):.4f}.")
    report.extend(["", "## Pilot Gate", "", json.dumps(gate, indent=2, default=json_default), "", "## Ranks", ""])
    for r in ranks:
        report.append(f"- {r['direction']} fold{r['fold']} {r['subspace']}: rank {r['rank']}/{r['dim']} ({float(r['rank_over_dim']):.4f}), EV {float(r['explained_variance']):.6f}, high-dimensional={r['effectively_high_dimensional']}.")
    report.extend(["", "Stop: Phase XLI-A ended here.", f"Runtime: {time.time() - start:.1f}s."])
    (OUT / "XLI_A_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(OUT / "OUTCOME.json", {"phase": "XLI-A", "outcome": outcome, "pilot_gate": gate, "runtime_sec": time.time() - start})
    with (ROOT / "EXPERIMENT_LOG.md").open("a", encoding="utf-8") as fp:
        fp.write("\n## Phase XLI-A Local Subspace Knowledge Representation Audit\n")
        fp.write("- Source feature: cached XXXVIII/XXXIX CLIP local patch feature.\n")
        fp.write(f"- PCA sampling: source-only, cap {PER_CATEGORY_PATCH_CAP} patches per category/class where available, rank by 0.99 EV.\n")
        fp.write(f"- Pilot gate: {json.dumps(gate, default=json_default)}.\n")
        fp.write(f"- Outcome: {outcome}.\n")


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = OUT / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    payloads = []
    for direction, fold in PILOT:
        path = ckpt / f"{direction}_fold{fold}_pilot.json"
        if path.exists():
            print(f"[XLI-A] resume {direction} fold{fold}", flush=True)
            payloads.append(read_json(path))
            continue
        print(f"[XLI-A] run {direction} fold{fold}", flush=True)
        payload = run_fold(direction, fold)
        write_json(path, payload)
        payloads.append(payload)
    aggregate(payloads, start)
    print(json.dumps(read_json(OUT / "OUTCOME.json"), indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
