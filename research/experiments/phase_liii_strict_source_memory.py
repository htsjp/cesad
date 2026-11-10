import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import ExternalDataset, evaluate_external, gp, load_full_visa_source, top1_image_score
from experiments.phase_xlv_medical_pixel_transfer import MedicalDataset, eval_scores
from experiments.phase_xxxviii_visual_state_representation import PATCH_TRAIN_SAMPLES_PER_CLASS_PER_CATEGORY, norm_np, patch_labels


OUT = Path("results/strict_source_zsad_memory")
XLV_A = Path("results/phase_xlv_cross_domain_transfer")
XLV_B = Path("results/phase_xlv_medical_pixel_transfer")
DATA_ROOT = Path(r"E:\WorkSpace\dataset")
SEED = 0
PCA_RANK = 16
LOW_RANK = 16
CHUNK_Q = 512


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


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def safe_mean(vals):
    vals = [float(v) for v in vals if np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def source_patch_bank():
    path = OUT / "cache" / "full_visa_source_patch_bank_cap512_v2.npz"
    if path.exists():
        return np.load(path, allow_pickle=True)
    _name, dataset, indices, _cls, patch, _y, _cats, _rows = load_full_visa_source()
    rng = np.random.default_rng(SEED)
    by_key = defaultdict(list)
    counts = defaultdict(int)
    for pos, idx in enumerate(indices):
        item = dataset[idx]
        labels = patch_labels(item)
        cat = str(item["cls_name"])
        for cls in [0, 1]:
            room = PATCH_TRAIN_SAMPLES_PER_CLASS_PER_CATEGORY - counts[(cat, cls)]
            if room <= 0:
                continue
            ids = np.flatnonzero(labels == cls)
            if ids.size == 0:
                continue
            if ids.size > room:
                ids = rng.choice(ids, size=room, replace=False)
            by_key[(cat, cls)].append(norm_np(np.asarray(patch[pos, ids], dtype=np.float32)))
            counts[(cat, cls)] += int(ids.size)
    feats, labels_out, cats_out = [], [], []
    for (cat, cls), parts in sorted(by_key.items()):
        arr = np.concatenate(parts, axis=0).astype(np.float32)
        feats.append(arr)
        labels_out.extend([cls] * arr.shape[0])
        cats_out.extend([cat] * arr.shape[0])
    x = np.concatenate(feats, axis=0)
    y = np.asarray(labels_out, dtype=np.int64)
    cats = np.asarray(cats_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=norm_np(x).astype(np.float32), y=y.astype(np.int64), cats=cats)
    return np.load(path, allow_pickle=True)


def fps_select(x, k, seed=SEED):
    x = norm_np(x.astype(np.float32))
    if x.shape[0] <= k:
        return np.arange(x.shape[0], dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected = np.empty((k,), dtype=np.int64)
    selected[0] = int(rng.integers(0, x.shape[0]))
    best = 1.0 - x @ x[selected[0]]
    for i in range(1, k):
        selected[i] = int(np.argmax(best))
        dist = 1.0 - x @ x[selected[i]]
        best = np.minimum(best, dist)
    return selected


def build_memories():
    bank = source_patch_bank()
    x = np.asarray(bank["x"], dtype=np.float32)
    y = np.asarray(bank["y"], dtype=np.int64)
    cache = OUT / "memory_cache_v2" / "memories.npz"
    stats_path = OUT / "memory_cache_v2" / "memory_stats.json"
    if cache.exists() and stats_path.exists():
        return np.load(cache, allow_pickle=True), json.loads(stats_path.read_text(encoding="utf-8"))
    memories = {}
    stats = []
    full_x, full_y = x, y
    memories["Full Prototype Memory.x"] = full_x
    memories["Full Prototype Memory.y"] = full_y
    stats.append(storage_row("Full Prototype Memory", full_x.size * 4 + full_y.size * 4, full_x.shape[0], x.shape[1], 1.0))
    fps_idx = []
    per_class = min(int(np.sum(y == 0)), int(np.sum(y == 1)), 2048)
    for cls in [0, 1]:
        ids = np.flatnonzero(y == cls)
        sel = fps_select(x[ids], per_class, seed=SEED + cls)
        fps_idx.extend(ids[sel].tolist())
    fps_idx = np.asarray(fps_idx, dtype=np.int64)
    memories["FPS / Coreset Memory.x"] = x[fps_idx]
    memories["FPS / Coreset Memory.y"] = y[fps_idx]
    stats.append(storage_row("FPS / Coreset Memory", x[fps_idx].size * 4 + y[fps_idx].size * 4, len(fps_idx), x.shape[1], (x[fps_idx].size * 4 + y[fps_idx].size * 4) / (full_x.size * 4 + full_y.size * 4)))
    cent = np.stack([norm_np(x[y == cls].mean(axis=0, keepdims=True))[0] for cls in [0, 1]]).astype(np.float32)
    memories["Centroid Memory.centroids"] = cent
    stats.append(storage_row("Centroid Memory", cent.size * 4, 2, x.shape[1], cent.size * 4 / (full_x.size * 4 + full_y.size * 4)))
    for cls in [0, 1]:
        xc = x[y == cls]
        pca = PCA(n_components=PCA_RANK, random_state=SEED).fit(xc)
        memories[f"PCA-r Memory.mu{cls}"] = pca.mean_.astype(np.float32)
        memories[f"PCA-r Memory.components{cls}"] = pca.components_.astype(np.float32)
    stats.append(storage_row("PCA-r Memory", 2 * (x.shape[1] + PCA_RANK * x.shape[1]) * 4, 2, x.shape[1], 2 * (x.shape[1] + PCA_RANK * x.shape[1]) * 4 / (full_x.size * 4 + full_y.size * 4)))
    eps = 1e-4
    for cls in [0, 1]:
        xc = x[y == cls]
        memories[f"Gaussian-Diag Memory.mu{cls}"] = xc.mean(axis=0).astype(np.float32)
        memories[f"Gaussian-Diag Memory.var{cls}"] = (xc.var(axis=0) + eps).astype(np.float32)
        pca = PCA(n_components=LOW_RANK, random_state=SEED).fit(xc)
        memories[f"Gaussian Low-Rank Memory.mu{cls}"] = pca.mean_.astype(np.float32)
        memories[f"Gaussian Low-Rank Memory.components{cls}"] = pca.components_.astype(np.float32)
        memories[f"Gaussian Low-Rank Memory.var{cls}"] = (pca.explained_variance_ + eps).astype(np.float32)
        memories[f"Gaussian Low-Rank Memory.resvar{cls}"] = np.asarray([max(eps, float(np.mean(pca.noise_variance_)))], dtype=np.float32)
    stats.append(storage_row("Gaussian-Diag Memory", 2 * 2 * x.shape[1] * 4, 2, x.shape[1], 2 * 2 * x.shape[1] * 4 / (full_x.size * 4 + full_y.size * 4)))
    stats.append(storage_row("Gaussian Low-Rank Memory", 2 * (x.shape[1] + LOW_RANK * x.shape[1] + LOW_RANK + 1) * 4, 2, x.shape[1], 2 * (x.shape[1] + LOW_RANK * x.shape[1] + LOW_RANK + 1) * 4 / (full_x.size * 4 + full_y.size * 4)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **memories)
    write_json(stats_path, {"memory_stats": stats, "source_patch_bank": int(x.shape[0]), "feature_dim": int(x.shape[1])})
    return np.load(cache, allow_pickle=True), {"memory_stats": stats, "source_patch_bank": int(x.shape[0]), "feature_dim": int(x.shape[1])}


def storage_row(name, bytes_float32, entries, dim, compression_ratio):
    return {"variant": name, "memory_bytes": int(bytes_float32), "entries": int(entries), "feature_dim": int(dim), "compression_ratio_vs_full": float(compression_ratio)}


def score_retrieval(patch, mx, my):
    mx = norm_np(mx.astype(np.float32))
    my = np.asarray(my, dtype=np.int64)
    maps = np.zeros((patch.shape[0], 37, 37), dtype=np.float16)
    image = np.zeros((patch.shape[0],), dtype=np.float32)
    qstats = []
    for i in range(patch.shape[0]):
        q = norm_np(np.asarray(patch[i], dtype=np.float32))
        out = np.zeros((q.shape[0],), dtype=np.float32)
        sim_mean = []
        sim_std = []
        for s in range(0, q.shape[0], CHUNK_Q):
            sims = q[s:s + CHUNK_Q] @ mx.T
            m = sims.max(axis=1, keepdims=True)
            w = np.exp(sims - m)
            out[s:s + CHUNK_Q] = ((w * (my[None, :] == 1)).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-12)).astype(np.float32)
            sim_mean.append(sims.mean(axis=1))
            sim_std.append(sims.std(axis=1))
        mm = out.reshape(37, 37)
        maps[i] = mm.astype(np.float16)
        image[i] = top1_image_score(mm)
        qstats.append({"retrieval_mean": float(np.mean(np.concatenate(sim_mean))), "retrieval_std": float(np.mean(np.concatenate(sim_std)))})
    return image, maps, qstats


def score_centroid(patch, centroids):
    maps = np.zeros((patch.shape[0], 37, 37), dtype=np.float16)
    image = np.zeros((patch.shape[0],), dtype=np.float32)
    stats = []
    c = norm_np(centroids.astype(np.float32))
    for i in range(patch.shape[0]):
        q = norm_np(np.asarray(patch[i], dtype=np.float32))
        sims = q @ c.T
        s = sims[:, 1] - sims[:, 0]
        mm = s.reshape(37, 37).astype(np.float32)
        maps[i] = mm.astype(np.float16)
        image[i] = top1_image_score(mm)
        stats.append({"normal_energy": float(np.mean(1 - sims[:, 0])), "anomaly_energy": float(np.mean(1 - sims[:, 1])), "normal_anomaly_margin": float(np.mean(s))})
    return image, maps, stats


def pca_energy(q, mu, comp):
    d = q - mu[None, :]
    coef = d @ comp.T
    recon = coef @ comp
    return np.sum((d - recon) ** 2, axis=1)


def score_pca(patch, mem):
    maps = np.zeros((patch.shape[0], 37, 37), dtype=np.float16)
    image = np.zeros((patch.shape[0],), dtype=np.float32)
    stats = []
    mu0, c0 = mem["PCA-r Memory.mu0"], mem["PCA-r Memory.components0"]
    mu1, c1 = mem["PCA-r Memory.mu1"], mem["PCA-r Memory.components1"]
    for i in range(patch.shape[0]):
        q = norm_np(np.asarray(patch[i], dtype=np.float32))
        e0, e1 = pca_energy(q, mu0, c0), pca_energy(q, mu1, c1)
        s = e0 - e1
        mm = s.reshape(37, 37).astype(np.float32)
        maps[i] = mm.astype(np.float16)
        image[i] = top1_image_score(mm)
        stats.append({"normal_energy": float(np.mean(e0)), "anomaly_energy": float(np.mean(e1)), "normal_anomaly_margin": float(np.mean(s)), "PCA_residual": float(np.mean(e0))})
    return image, maps, stats


def score_gaussian_diag(patch, mem):
    maps = np.zeros((patch.shape[0], 37, 37), dtype=np.float16)
    image = np.zeros((patch.shape[0],), dtype=np.float32)
    stats = []
    mu0, v0 = mem["Gaussian-Diag Memory.mu0"], mem["Gaussian-Diag Memory.var0"]
    mu1, v1 = mem["Gaussian-Diag Memory.mu1"], mem["Gaussian-Diag Memory.var1"]
    for i in range(patch.shape[0]):
        q = norm_np(np.asarray(patch[i], dtype=np.float32))
        e0 = np.mean(((q - mu0[None, :]) ** 2) / v0[None, :], axis=1)
        e1 = np.mean(((q - mu1[None, :]) ** 2) / v1[None, :], axis=1)
        s = e0 - e1
        mm = s.reshape(37, 37).astype(np.float32)
        maps[i] = mm.astype(np.float16)
        image[i] = top1_image_score(mm)
        stats.append({"normal_energy": float(np.mean(e0)), "anomaly_energy": float(np.mean(e1)), "Mahalanobis_distance": float(np.mean(e0)), "normal_anomaly_margin": float(np.mean(s))})
    return image, maps, stats


def lowrank_energy(q, mu, comp, var, resvar):
    d = q - mu[None, :]
    coef = d @ comp.T
    recon = coef @ comp
    resid = np.sum((d - recon) ** 2, axis=1) / max(float(resvar[0]), 1e-6)
    return np.mean((coef ** 2) / var[None, :], axis=1) + resid / q.shape[1]


def score_gaussian_lowrank(patch, mem):
    maps = np.zeros((patch.shape[0], 37, 37), dtype=np.float16)
    image = np.zeros((patch.shape[0],), dtype=np.float32)
    stats = []
    args0 = [mem["Gaussian Low-Rank Memory.mu0"], mem["Gaussian Low-Rank Memory.components0"], mem["Gaussian Low-Rank Memory.var0"], mem["Gaussian Low-Rank Memory.resvar0"]]
    args1 = [mem["Gaussian Low-Rank Memory.mu1"], mem["Gaussian Low-Rank Memory.components1"], mem["Gaussian Low-Rank Memory.var1"], mem["Gaussian Low-Rank Memory.resvar1"]]
    for i in range(patch.shape[0]):
        q = norm_np(np.asarray(patch[i], dtype=np.float32))
        e0 = lowrank_energy(q, *args0)
        e1 = lowrank_energy(q, *args1)
        s = e0 - e1
        mm = s.reshape(37, 37).astype(np.float32)
        maps[i] = mm.astype(np.float16)
        image[i] = top1_image_score(mm)
        stats.append({"normal_energy": float(np.mean(e0)), "anomaly_energy": float(np.mean(e1)), "Mahalanobis_distance": float(np.mean(e0)), "normal_anomaly_margin": float(np.mean(s))})
    return image, maps, stats


def targets():
    return [
        ("BTAD", ExternalDataset(DATA_ROOT / "btad" / "BTech_Dataset_transformed", "BTAD", mode="test"), XLV_A / "feature_cache" / "BTAD_clip.npz", "external"),
        ("ISBI", MedicalDataset(DATA_ROOT / "ISBI", mode="test"), XLV_B / "feature_cache" / "ISBI_clip.npz", "medical"),
        ("CVC-ClinicDB", MedicalDataset(DATA_ROOT / "CVC-ClinicDB", mode="test"), XLV_B / "feature_cache" / "CVC-ClinicDB_clip.npz", "medical"),
    ]


def evaluate_target(name, dataset, cache_path, kind, image, maps):
    c = np.load(cache_path, allow_pickle=True)
    rows = [dict(r) for r in c["rows"]]
    if kind == "external":
        mean, per = evaluate_external(dataset, rows, image, maps)
    else:
        mean, _per_img, _pix = eval_scores(dataset, rows, image, maps)
        per = {"skin" if name == "ISBI" else "colon": mean}
    return mean, per


def run_phase1():
    start = time.time()
    (OUT / "phase1_distribution_memory").mkdir(parents=True, exist_ok=True)
    mem, stat = build_memories()
    raw, percat, qstats = [], [], []
    variants = [
        ("Full Prototype Memory", lambda p: score_retrieval(p, mem["Full Prototype Memory.x"], mem["Full Prototype Memory.y"])),
        ("FPS / Coreset Memory", lambda p: score_retrieval(p, mem["FPS / Coreset Memory.x"], mem["FPS / Coreset Memory.y"])),
        ("Centroid Memory", lambda p: score_centroid(p, mem["Centroid Memory.centroids"])),
        ("PCA-r Memory", lambda p: score_pca(p, mem)),
        ("Gaussian-Diag Memory", lambda p: score_gaussian_diag(p, mem)),
        ("Gaussian Low-Rank Memory", lambda p: score_gaussian_lowrank(p, mem)),
    ]
    for target, ds, cache_path, kind in targets():
        c = np.load(cache_path, allow_pickle=True)
        patch = c["patch"]
        for variant, scorer in variants:
            print(f"[ZSAD-Memory] Phase1 {target} {variant}", flush=True)
            image, maps, stats = scorer(patch)
            if not np.isfinite(maps.astype(np.float32)).all() or not np.isfinite(image).all():
                write_json(OUT / "OUTCOME.json", {"phase": "phase1", "outcome": "PHASE1_NUMERICAL_FAILURE", "target": target, "variant": variant})
                return False, raw
            mean, per = evaluate_target(target, ds, cache_path, kind, image, maps)
            row = {"phase": "phase1", "target": target, "variant": variant, **mean}
            row["G_pixel"] = gp(row)
            raw.append(row)
            for cat, vals in per.items():
                rr = {"phase": "phase1", "target": target, "category": cat, "variant": variant, **vals}
                rr["G_pixel"] = gp(rr)
                percat.append(rr)
            qstats.append({"phase": "phase1", "target": target, "variant": variant, **aggregate_stats(stats)})
    stats_rows = stat["memory_stats"]
    write_csv(OUT / "phase1_distribution_memory" / "RAW_METRICS.csv", raw)
    write_csv(OUT / "phase1_distribution_memory" / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "phase1_distribution_memory" / "QUERY_DEPENDENT_STATS.csv", qstats)
    write_csv(OUT / "phase1_distribution_memory" / "MEMORY_STATS.csv", stats_rows)
    summary_rows = phase1_summary(raw)
    write_csv(OUT / "phase1_distribution_memory" / "SUMMARY_TABLE.csv", summary_rows)
    outcome, reason = phase1_outcome(raw)
    write_json(OUT / "phase1_distribution_memory" / "OUTCOME.json", {"outcome": outcome, "reason": reason, "runtime_sec": time.time() - start})
    write_phase1_summary(raw, stats_rows, outcome, reason, time.time() - start)
    return outcome == "PHASE1_COMPLETED_PASS_TO_PHASE2", raw


def aggregate_stats(rows):
    keys = sorted({k for r in rows for k in r})
    out = {}
    for k in keys:
        vals = [safe_float(r.get(k, np.nan)) for r in rows]
        out[f"{k}_mean"] = safe_mean(vals)
        out[f"{k}_std"] = float(np.nanstd(vals)) if vals else float("nan")
    return out


def phase1_summary(raw):
    rows = []
    by = defaultdict(list)
    for r in raw:
        by[r["variant"]].append(r)
    for variant, vals in sorted(by.items()):
        rows.append({
            "variant": variant,
            "targets": len(vals),
            "mean_G_pixel": safe_mean([gp(v) for v in vals]),
            "std_G_pixel": float(np.nanstd([gp(v) for v in vals])),
            "mean_P-AUROC": safe_mean([v["P-AUROC"] for v in vals]),
            "mean_P-AP": safe_mean([v["P-AP"] for v in vals]),
        })
    return rows


def phase1_outcome(raw):
    by = {(r["target"], r["variant"]): r for r in raw}
    # sanity: FPS should reproduce prior XLVII CLIP Source Memory approximately on BTAD/ClinicDB.
    refs = {"BTAD": 44.6371, "CVC-ClinicDB": 59.7643}
    for target, ref in refs.items():
        got = gp(by[(target, "FPS / Coreset Memory")])
        if abs(got - ref) > 3.0:
            return "PHASE1_BASELINE_REPRODUCTION_FAILURE", f"{target} FPS G {got:.4f} differs from prior {ref:.4f}"
    centroid = safe_mean([gp(r) for r in raw if r["variant"] == "Centroid Memory"])
    dist_best = max(safe_mean([gp(r) for r in raw if r["variant"] == v]) for v in ["PCA-r Memory", "Gaussian-Diag Memory", "Gaussian Low-Rank Memory"])
    if dist_best <= centroid + 0.25:
        return "PHASE1_DISTRIBUTION_MEMORY_NO_EXTRA_INFO", f"best distribution mean G {dist_best:.4f} <= centroid {centroid:.4f}+0.25"
    return "PHASE1_COMPLETED_PASS_TO_PHASE2", "distribution memory retained extra cross-target signal over centroid"


def write_phase1_summary(raw, stats_rows, outcome, reason, runtime):
    lines = ["# Strict Source-Only ZSAD Memory - Phase 1", "", f"Outcome: **{outcome}**", "", f"Reason: {reason}", "", "## Mean Metrics", ""]
    lines.append("| Variant | Mean G | Mean P-AUROC | Mean P-AP |")
    lines.append("|---|---:|---:|---:|")
    for r in phase1_summary(raw):
        lines.append(f"| {r['variant']} | {r['mean_G_pixel']:.4f} | {r['mean_P-AUROC']:.4f} | {r['mean_P-AP']:.4f} |")
    lines.extend(["", "## Required Answers", "", "1. Phase 1 executed successfully unless outcome is numerical/protocol failure.", "2. Strict source-only protocol was preserved: memories were built only from Full VisA source patch labels; target images/masks were used only for final evaluation.", "3. Distribution-vs-prototype answer follows the outcome reason above.", "4. Compression ratios are in `MEMORY_STATS.csv`.", "5. Memory evidence/model-gate is not tested in Phase 1.", "6. Stable StateAligned improvement is not tested in Phase 1.", f"7. Enter next phase: `{outcome == 'PHASE1_COMPLETED_PASS_TO_PHASE2'}`.", "", f"Runtime: {runtime:.1f}s."])
    (OUT / "phase1_distribution_memory" / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_top_summary(phase1_ok):
    write_json(OUT / "OUTCOME.json", {"phase1_passed": phase1_ok, "phase2_started": False, "phase3_started": False, "strict_source_only": True})
    (OUT / "summary.md").write_text(
        "# Strict Source-Only ZSAD Memory\n\n"
        f"- Phase 1 pass to Phase 2: `{phase1_ok}`.\n"
        "- Phase 2 is not started unless Phase 1 passes.\n"
        "- Phase 3 is not started unless Phase 1 and Phase 2 both pass.\n",
        encoding="utf-8",
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "CONFIG.json", {
        "strict_source_only": True,
        "source": "Full VisA",
        "targets": ["BTAD", "ISBI", "CVC-ClinicDB"],
        "target_training": False,
        "target_normal_memory": False,
        "target_statistics": False,
        "target_mask_model_selection": False,
        "online_test_time_memory": False,
        "phase1_variants": ["Full Prototype Memory", "FPS / Coreset Memory", "Centroid Memory", "PCA-r Memory", "Gaussian-Diag Memory", "Gaussian Low-Rank Memory"],
        "source_patch_cap_per_class_per_category": PATCH_TRAIN_SAMPLES_PER_CLASS_PER_CATEGORY,
        "pca_rank": PCA_RANK,
        "gaussian_low_rank": LOW_RANK,
    })
    phase1_ok, _ = run_phase1()
    write_top_summary(phase1_ok)
    print(json.dumps({"phase1_ok": phase1_ok, "out": str(OUT)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
