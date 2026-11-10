import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_li_asymmetric_state import DATA_ROOT, XLV_A, XLV_B, evaluate_source_maps
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import ExternalDataset, evaluate_external, gp, top1_image_score
from experiments.phase_xlv_medical_pixel_transfer import MedicalDataset, eval_scores
from experiments.phase_xxxviii_visual_state_representation import FEATURE_CACHE, load_fold, norm_np, patch_labels


OUT = Path("results/phase_memory_support_modes")
SMOKE_PATCH_BANK = Path("results/strict_source_zsad_memory/cache/full_visa_source_patch_bank_cap512_v2.npz")
SEED = 0
SIDE = 37
KS = [4, 16, 64, 256]
SMOKE_KS = [4]
MAX_CANDIDATES_PER_CATEGORY = {0: 4096, 1: 20000}
SMOKE_MAX_CANDIDATES_PER_CATEGORY = {0: 384, 1: 384}
SMOKE_SOURCE_IMAGES_PER_CLASS = 24


class ArrayConcat:
    def __init__(self, arrays):
        self.arrays = list(arrays)
        self.lengths = [int(a.shape[0]) for a in self.arrays]
        rest = self.arrays[0].shape[1:]
        self.shape = (sum(self.lengths),) + rest
        self.dtype = self.arrays[0].dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        tail = ()
        if isinstance(idx, tuple):
            idx, tail = idx[0], idx[1:]
        if isinstance(idx, slice):
            arr = np.stack([self[i] for i in range(*idx.indices(len(self)))], axis=0)
            return arr[(slice(None),) + tail] if tail else arr
        if idx < 0:
            idx += len(self)
        offset = 0
        for arr, length in zip(self.arrays, self.lengths):
            if idx < offset + length:
                item = arr[idx - offset]
                return item[tail] if tail else item
            offset += length
        raise IndexError(idx)


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


def read_csv_rows(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def density_from_raw(raw):
    rows = []
    by_cell = defaultdict(dict)
    for r in raw:
        if r.get("experiment") == "B_coverage_vs_density":
            by_cell[r["cell"]][r["variant"]] = r
    for cell, vals in sorted(by_cell.items()):
        if "FPS64-Uniform" in vals and "FPS64-MassWeighted" in vals:
            rows.append({
                "cell": cell,
                "target": vals["FPS64-Uniform"].get("target", vals["FPS64-MassWeighted"].get("target", "")),
                "DeltaG_mass_minus_uniform": gp(vals["FPS64-MassWeighted"]) - gp(vals["FPS64-Uniform"]),
            })
    return rows


def refresh_relative_to_full(raw):
    full_by_cell = {
        r["cell"]: gp(r)
        for r in raw
        if r.get("experiment") == "A_support_saturation" and r.get("variant") == "Full"
    }
    for r in raw:
        if r.get("experiment") == "A_support_saturation" and r.get("cell") in full_by_cell:
            r["relative_to_Full_G"] = safe_float(gp(r)) - safe_float(full_by_cell[r["cell"]])


def load_checkpoint(out):
    raw = read_csv_rows(out / "PARTIAL_RAW_METRICS.csv")
    percat = read_csv_rows(out / "PARTIAL_PER_CATEGORY.csv")
    storage = read_csv_rows(out / "PARTIAL_STORAGE_CURVE.csv")
    resp_all = read_csv_rows(out / "PARTIAL_RESPONSIBILITY_RAW.csv")
    completed = {
        (r.get("cell"), r.get("experiment"), r.get("variant"))
        for r in raw
        if r.get("cell") and r.get("experiment") and r.get("variant")
    }
    if raw:
        print(f"[MemorySupport] resume checkpoint variants={len(completed)}", flush=True)
    return raw, percat, storage, resp_all, completed


def persist_checkpoint(out, raw, percat, storage, resp_all):
    refresh_relative_to_full(raw)
    write_csv(out / "PARTIAL_RAW_METRICS.csv", raw)
    write_csv(out / "PARTIAL_PER_CATEGORY.csv", percat)
    write_csv(out / "PARTIAL_STORAGE_CURVE.csv", storage)
    write_csv(out / "PARTIAL_COVERAGE_VS_DENSITY.csv", density_from_raw(raw))
    write_csv(out / "PARTIAL_RESPONSIBILITY_RAW.csv", resp_all)
    write_json(out / "PARTIAL_STATUS.json", {
        "updated_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "completed_variants": len({
            (r.get("cell"), r.get("experiment"), r.get("variant"))
            for r in raw
            if r.get("cell") and r.get("experiment") and r.get("variant")
        }),
    })


def npz_array_mmap(npz_path, key):
    npz_path = Path(npz_path)
    digest = hashlib.sha1(str(npz_path.resolve()).encode("utf-8")).hexdigest()[:12]
    out = OUT / "array_cache" / f"{npz_path.stem}_{digest}_{key}.npy"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        with zipfile.ZipFile(npz_path) as zf:
            member = f"{key}.npy"
            if member not in zf.namelist():
                raise KeyError(f"{member} not found in {npz_path}")
            with zf.open(member) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
        os.replace(tmp, out)
    return np.load(out, mmap_mode="r")


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def safe_mean(vals):
    vals = [safe_float(v) for v in vals if np.isfinite(safe_float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def smoke_source_subset(dataset, indices):
    normal, anomaly = [], []
    for idx in indices:
        item = dataset[idx]
        if int(item["anomaly"]):
            if len(anomaly) < SMOKE_SOURCE_IMAGES_PER_CLASS:
                anomaly.append(idx)
        else:
            if len(normal) < SMOKE_SOURCE_IMAGES_PER_CLASS:
                normal.append(idx)
        if len(normal) >= SMOKE_SOURCE_IMAGES_PER_CLASS and len(anomaly) >= SMOKE_SOURCE_IMAGES_PER_CLASS:
            break
    return normal + anomaly


def fps_order(x, max_order=None):
    x = norm_np(np.asarray(x, dtype=np.float32))
    n = x.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    length = n if max_order is None else min(int(max_order), n)
    order = np.empty((length,), dtype=np.int64)
    centroid = norm_np(x.mean(axis=0, keepdims=True))[0]
    order[0] = int(np.argmin(x @ centroid))
    best = 1.0 - x @ x[order[0]]
    for i in range(1, length):
        order[i] = int(np.argmax(best))
        best = np.minimum(best, 1.0 - x @ x[order[i]])
    return order


def collect_candidates(dataset, indices, patch, cls, max_per_category):
    rng = np.random.default_rng(SEED + cls)
    by_cat = {}
    seen = defaultdict(int)
    for pos, idx in enumerate(indices):
        item = dataset[idx]
        cat = str(item["cls_name"])
        labels = patch_labels(item)
        ids = np.flatnonzero(labels == cls)
        if ids.size == 0:
            continue
        bucket = by_cat.setdefault(cat, [])
        for patch_id in ids.tolist():
            seen[cat] += 1
            coord = (pos, int(patch_id))
            if len(bucket) < max_per_category:
                bucket.append(coord)
            else:
                j = int(rng.integers(0, seen[cat]))
                if j < max_per_category:
                    bucket[j] = coord
    out = {}
    for cat, coords in sorted(by_cat.items()):
        if coords:
            out[cat] = norm_np(np.stack([np.asarray(patch[p, j], dtype=np.float32) for p, j in coords], axis=0))
        else:
            out[cat] = np.zeros((0, patch.shape[-1]), dtype=np.float32)
    return out


def build_smoke_memory_from_patch_bank(tag):
    path = OUT / "memory_cache" / f"{tag}.npz"
    meta_path = OUT / "memory_cache" / f"{tag}.json"
    if path.exists() and meta_path.exists():
        return np.load(path, allow_pickle=True), json.loads(meta_path.read_text(encoding="utf-8"))
    bank = np.load(SMOKE_PATCH_BANK, allow_pickle=True)
    x_all = norm_np(bank["x"].astype(np.float32))
    y_all = bank["y"].astype(np.int64)
    cats_all = bank["cats"].astype(str)
    arrays = {}
    meta = {"direction": tag, "fold": "smoke", "dataset": "FullVisA_patch_bank_smoke", "groups": [], "smoke": True}
    for cls, name in [(0, "normal"), (1, "anomaly")]:
        parts = []
        for cat in sorted(set(cats_all.tolist())):
            ids = np.flatnonzero((y_all == cls) & (cats_all == cat))[: SMOKE_MAX_CANDIDATES_PER_CATEGORY[cls]]
            if ids.size == 0:
                continue
            group = f"{name}:{cat}"
            x = x_all[ids]
            order = fps_order(x, max_order=max(KS))
            arrays[f"{group}.x"] = x.astype(np.float32)
            arrays[f"{group}.order"] = order.astype(np.int64)
            meta["groups"].append({"group": group, "class": name, "category": cat, "candidate_count": int(x.shape[0])})
            parts.append(x)
        pooled = np.concatenate(parts, axis=0) if parts else np.zeros((0, x_all.shape[-1]), dtype=np.float32)
        arrays[f"{name}.pooled"] = pooled.astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    write_json(meta_path, meta)
    return np.load(path, allow_pickle=True), meta


def build_source_memory(direction, fold, smoke):
    if smoke:
        return build_smoke_memory_from_patch_bank(f"{direction}_fold{fold}_smoke_bank")
    tag = f"{direction}_fold{fold}_{'smoke' if smoke else 'formal'}"
    path = OUT / "memory_cache" / f"{tag}.npz"
    meta_path = OUT / "memory_cache" / f"{tag}.json"
    if path.exists() and meta_path.exists():
        return np.load(path, allow_pickle=True), json.loads(meta_path.read_text(encoding="utf-8"))
    dataset_name, dataset, _rot, train_indices, _held_indices = load_fold(direction, fold)
    cache_path = FEATURE_CACHE / f"{direction}_fold{fold}_clip.npz"
    train_patch = npz_array_mmap(cache_path, "train_patch")
    if smoke:
        keep = smoke_source_subset(dataset, train_indices)
        pos = [list(train_indices).index(i) for i in keep]
        train_indices = keep
        train_patch = train_patch[pos]
    caps = SMOKE_MAX_CANDIDATES_PER_CATEGORY if smoke else MAX_CANDIDATES_PER_CATEGORY
    arrays = {}
    meta = {"direction": direction, "fold": fold, "dataset": dataset_name, "groups": [], "smoke": smoke}
    for cls, name in [(0, "normal"), (1, "anomaly")]:
        candidates = collect_candidates(dataset, train_indices, train_patch, cls, caps[cls])
        parts = []
        for cat, x in candidates.items():
            group = f"{name}:{cat}"
            order = fps_order(x, max_order=max(KS))
            arrays[f"{group}.x"] = x.astype(np.float32)
            arrays[f"{group}.order"] = order.astype(np.int64)
            meta["groups"].append({"group": group, "class": name, "category": cat, "candidate_count": int(x.shape[0])})
            if x.size:
                parts.append(x)
        pooled = np.concatenate(parts, axis=0) if parts else np.zeros((0, train_patch.shape[-1]), dtype=np.float32)
        arrays[f"{name}.pooled"] = pooled.astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    write_json(meta_path, meta)
    return np.load(path, allow_pickle=True), meta


def build_full_visa_memory(smoke):
    if smoke:
        return build_smoke_memory_from_patch_bank("FullVisA_smoke_bank")
    tag = f"FullVisA_{'smoke' if smoke else 'formal'}"
    path = OUT / "memory_cache" / f"{tag}.npz"
    meta_path = OUT / "memory_cache" / f"{tag}.json"
    if path.exists() and meta_path.exists():
        return np.load(path, allow_pickle=True), json.loads(meta_path.read_text(encoding="utf-8"))
    dataset_name, dataset, _rot, train_indices, held_indices = load_fold("D2", 0)
    cache_path = FEATURE_CACHE / "D2_fold0_clip.npz"
    train_patch = npz_array_mmap(cache_path, "train_patch")
    held_patch = npz_array_mmap(cache_path, "held_patch")
    indices = list(train_indices) + list(held_indices)
    patch = ArrayConcat([train_patch, held_patch])
    if smoke:
        keep = smoke_source_subset(dataset, indices)
        pos = [list(indices).index(i) for i in keep]
        indices = keep
        patch = patch[pos]
    caps = SMOKE_MAX_CANDIDATES_PER_CATEGORY if smoke else MAX_CANDIDATES_PER_CATEGORY
    arrays = {}
    meta = {"direction": "FullVisA", "fold": "full", "dataset": dataset_name, "groups": [], "smoke": smoke}
    for cls, name in [(0, "normal"), (1, "anomaly")]:
        candidates = collect_candidates(dataset, indices, patch, cls, caps[cls])
        parts = []
        for cat, x in candidates.items():
            group = f"{name}:{cat}"
            order = fps_order(x, max_order=max(KS))
            arrays[f"{group}.x"] = x.astype(np.float32)
            arrays[f"{group}.order"] = order.astype(np.int64)
            meta["groups"].append({"group": group, "class": name, "category": cat, "candidate_count": int(x.shape[0])})
            if x.size:
                parts.append(x)
        pooled = np.concatenate(parts, axis=0) if parts else np.zeros((0, patch.shape[-1]), dtype=np.float32)
        arrays[f"{name}.pooled"] = pooled.astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    write_json(meta_path, meta)
    return np.load(path, allow_pickle=True), meta


def select_centers(memory, cls_name, k):
    parts = []
    if k == "CENTROID":
        for key in memory.files:
            if key.startswith(f"{cls_name}:") and key.endswith(".x"):
                x = memory[key]
                if x.shape[0]:
                    parts.append(norm_np(x.mean(axis=0, keepdims=True))[0:1])
        return norm_np(np.concatenate(parts, axis=0)) if parts else np.zeros((0, 1024), dtype=np.float32)
    if k == "ALL":
        return norm_np(memory[f"{cls_name}.pooled"])
    for key in memory.files:
        if key.startswith(f"{cls_name}:") and key.endswith(".x"):
            group = key[:-2]
            x = memory[key]
            order = memory[f"{group}.order"]
            take = order[: min(int(k), order.shape[0])]
            if take.size:
                parts.append(x[take])
    return norm_np(np.concatenate(parts, axis=0)) if parts else np.zeros((0, 1024), dtype=np.float32)


def score_hard(patch, mem_n, mem_a, limit_images=0):
    n_img = patch.shape[0] if not limit_images else min(limit_images, patch.shape[0])
    maps = np.zeros((n_img, SIDE, SIDE), dtype=np.float16)
    image = np.zeros((n_img,), dtype=np.float32)
    mn = norm_np(mem_n)
    ma = norm_np(mem_a)
    def max_sim(q, mem, chunk=8192):
        if mem.shape[0] == 0:
            return np.full((q.shape[0],), -np.inf, dtype=np.float32)
        best = np.full((q.shape[0],), -np.inf, dtype=np.float32)
        for start in range(0, mem.shape[0], chunk):
            best = np.maximum(best, (q @ mem[start:start + chunk].T).max(axis=1))
        return best
    for i in range(n_img):
        q = norm_np(np.asarray(patch[i], dtype=np.float32))
        s = max_sim(q, ma) - max_sim(q, mn)
        m = s.reshape(SIDE, SIDE).astype(np.float32)
        maps[i] = m.astype(np.float16)
        image[i] = top1_image_score(m)
    return image, maps


def estimate_kappa(memory):
    vals = []
    for cls_name in ["normal", "anomaly"]:
        centers = select_centers(memory, cls_name, 64)
        full = norm_np(memory[f"{cls_name}.pooled"])
        if centers.shape[0] == 0 or full.shape[0] == 0:
            continue
        sims = full @ centers.T
        vals.append(sims.max(axis=1))
    pooled = np.concatenate(vals) if vals else np.asarray([0.5], dtype=np.float32)
    mean_cos = float(np.clip(np.mean(pooled), 1e-4, 0.999))
    # vMF high-dimensional approximation: E[cos] ~= 1 - (d-1)/(2*kappa).
    dim = 1024
    kappa = float((dim - 1) / max(2.0 * (1.0 - mean_cos), 1e-6))
    return min(kappa, 1000.0), {"pooled_assigned_cos_mean": mean_cos, "formula": "kappa=(dim-1)/(2*(1-mean_assigned_cos)), clipped at 1000"}


def centers_and_priors(memory, cls_name, weighted):
    centers = select_centers(memory, cls_name, 64)
    full = norm_np(memory[f"{cls_name}.pooled"])
    if centers.shape[0] == 0:
        return centers, np.ones((1,), dtype=np.float32)
    if not weighted or full.shape[0] == 0:
        pi = np.ones((centers.shape[0],), dtype=np.float32) / centers.shape[0]
        return centers, pi
    assign = np.argmax(full @ centers.T, axis=1)
    counts = np.bincount(assign, minlength=centers.shape[0]).astype(np.float32)
    pi = counts / max(float(counts.sum()), 1.0)
    pi = np.maximum(pi, 1e-8)
    pi = pi / pi.sum()
    return centers, pi


def score_mixture(patch, memory, weighted, limit_images=0):
    kappa, kinfo = estimate_kappa(memory)
    cn, pin = centers_and_priors(memory, "normal", weighted)
    ca, pia = centers_and_priors(memory, "anomaly", weighted)
    n_img = patch.shape[0] if not limit_images else min(limit_images, patch.shape[0])
    maps = np.zeros((n_img, SIDE, SIDE), dtype=np.float16)
    image = np.zeros((n_img,), dtype=np.float32)
    resp_rows = []
    for i in range(n_img):
        q = norm_np(np.asarray(patch[i], dtype=np.float32))
        ln = np.log(pin[None, :]) + kappa * (q @ cn.T)
        la = np.log(pia[None, :]) + kappa * (q @ ca.T)
        lsn = logsumexp(ln, axis=1)
        lsa = logsumexp(la, axis=1)
        s = lsa - lsn
        m = s.reshape(SIDE, SIDE).astype(np.float32)
        maps[i] = m.astype(np.float16)
        image[i] = top1_image_score(m)
        for cls_name, logits in [("normal", ln), ("anomaly", la)]:
            r = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
            hq = -np.sum(r * np.log(np.maximum(r, 1e-12)), axis=1)
            rbar = r.mean(axis=0)
            hbar = -float(np.sum(rbar * np.log(np.maximum(rbar, 1e-12))))
            resp_rows.append({
                "memory_class": cls_name,
                "image_pos": i,
                "H_mean": float(np.mean(hq)),
                "N_eff_mean": float(np.mean(np.exp(hq))),
                "I_QK": float(hbar - np.mean(hq)),
                "kappa": kappa,
                **kinfo,
            })
    return image, maps, resp_rows


def target_cell(direction, fold, smoke):
    print(f"[MemorySupport] prepare target cell {direction} fold{fold}", flush=True)
    if direction == "D1":
        target_dir = "D2"
    else:
        target_dir = "D1"
    dataset_name, dataset, _rot, train_indices, held_indices = load_fold(target_dir, 0)
    cache_path = FEATURE_CACHE / f"{target_dir}_fold0_clip.npz"
    train_patch = npz_array_mmap(cache_path, "train_patch")
    held_patch = npz_array_mmap(cache_path, "held_patch")
    patch = ArrayConcat([train_patch, held_patch])
    indices = list(train_indices) + list(held_indices)
    if smoke:
        indices = indices[:24]
        patch = patch[:24]
    return f"{direction}_fold{fold}_to_{dataset_name}", dataset_name, dataset, indices, patch, "source_dataset"


def external_cell(name, smoke):
    print(f"[MemorySupport] prepare external cell {name}", flush=True)
    if name == "BTAD":
        dataset = ExternalDataset(DATA_ROOT / "btad" / "BTech_Dataset_transformed", "BTAD", mode="test")
        cache_path = XLV_A / "feature_cache" / "BTAD_clip.npz"
        kind = "external"
    else:
        dataset = MedicalDataset(DATA_ROOT / name, mode="test")
        cache_path = XLV_B / "feature_cache" / f"{name}_clip.npz"
        kind = "medical"
    cache = np.load(cache_path, allow_pickle=True)
    patch = npz_array_mmap(cache_path, "patch")
    rows = [dict(r) for r in cache["rows"]]
    if smoke:
        patch = patch[:24]
        rows = rows[:24]
    return f"FullVisA_to_{name}", name, dataset, rows, patch, kind


def eval_maps(cell_id, target, dataset, idx_or_rows, kind, image, maps):
    if kind == "source_dataset":
        res = evaluate_source_maps(dataset, idx_or_rows, image, maps)
        mean, per = res["mean"], res["per_category"]
    elif kind == "external":
        mean, per = evaluate_external(dataset, idx_or_rows, image, maps)
    else:
        mean, _per_img, _pix = eval_scores(dataset, idx_or_rows, image, maps)
        per = {"skin" if target == "ISBI" else "colon": mean}
    row = {"cell": cell_id, "target": target, **mean}
    row["G_pixel"] = gp(row)
    return row, per


def run(args):
    start = time.time()
    out = OUT / ("smoke" if args.smoke else "formal")
    out.mkdir(parents=True, exist_ok=True)
    raw, percat, storage, resp_all, completed = load_checkpoint(out)
    cells = []
    print(f"[MemorySupport] start smoke={args.smoke}", flush=True)
    dirs = [] if args.smoke else [("D1", 0), ("D1", 1), ("D2", 0), ("D2", 1)]
    for direction, fold in dirs:
        cells.append((direction, fold, *target_cell(direction, fold, args.smoke)))
    for name in (["BTAD"] if args.smoke else ["BTAD", "ISBI", "CVC-ClinicDB"]):
        cells.append(("FullVisA", "full", *external_cell(name, args.smoke)))
    for source, fold, cell_id, target, dataset, idx_or_rows, patch, kind in cells:
        if source == "FullVisA":
            memory, meta = build_full_visa_memory(args.smoke)
        else:
            memory, meta = build_source_memory(source, int(fold), args.smoke)
        ks = SMOKE_KS if args.smoke else KS
        variants = [("K1_CENTROID", "CENTROID")] + [(f"FPS-{k}", k) for k in ks] + [("Full", "ALL")]
        for variant, kval in variants:
            key = (cell_id, "A_support_saturation", variant)
            if key in completed:
                print(f"[MemorySupport] skip completed {cell_id} {variant}", flush=True)
                continue
            print(f"[MemorySupport] {cell_id} {variant}", flush=True)
            mn = select_centers(memory, "normal", kval)
            ma = select_centers(memory, "anomaly", kval)
            image, maps = score_hard(patch, mn, ma)
            metric, per = eval_maps(cell_id, target, dataset, idx_or_rows, kind, image, maps)
            metric.update({"experiment": "A_support_saturation", "variant": variant, "K": kval, "source": source, "fold": fold})
            raw.append(metric)
            for cat, vals in per.items():
                rr = {"cell": cell_id, "target": target, "category": cat, "experiment": "A_support_saturation", "variant": variant, **vals}
                rr["G_pixel"] = gp(rr)
                percat.append(rr)
            storage.append({"cell": cell_id, "variant": variant, "normal_prototypes": int(mn.shape[0]), "anomaly_prototypes": int(ma.shape[0]), "prototype_count": int(mn.shape[0] + ma.shape[0])})
            completed.add(key)
            persist_checkpoint(out, raw, percat, storage, resp_all)
        for weighted, variant in [(False, "FPS64-Uniform"), (True, "FPS64-MassWeighted")]:
            key = (cell_id, "B_coverage_vs_density", variant)
            if key in completed:
                print(f"[MemorySupport] skip completed {cell_id} {variant}", flush=True)
                continue
            print(f"[MemorySupport] {cell_id} {variant}", flush=True)
            image, maps, resp = score_mixture(patch, memory, weighted)
            metric, per = eval_maps(cell_id, target, dataset, idx_or_rows, kind, image, maps)
            metric.update({"experiment": "B_coverage_vs_density", "variant": variant, "K": 64, "source": source, "fold": fold})
            raw.append(metric)
            for cat, vals in per.items():
                rr = {"cell": cell_id, "target": target, "category": cat, "experiment": "B_coverage_vs_density", "variant": variant, **vals}
                rr["G_pixel"] = gp(rr)
                percat.append(rr)
            for rr in resp:
                rr.update({"cell": cell_id, "target": target, "variant": variant})
                resp_all.append(rr)
            completed.add(key)
            persist_checkpoint(out, raw, percat, storage, resp_all)
    density = density_from_raw(raw)
    mean_by_k = summarize_a(raw)
    density_summary = summarize_density(density)
    resp_summary = summarize_resp(resp_all)
    outcome = decide(mean_by_k, density, density_summary)
    full_counts = {
        r["cell"]: max(int(r["prototype_count"]), 1)
        for r in storage
        if r["variant"] == "Full"
    }
    for r in storage:
        full_count = full_counts.get(r["cell"], max(int(r["prototype_count"]), 1))
        r["storage_ratio_vs_full"] = float(int(r["prototype_count"]) / full_count)
        r["compression_vs_full"] = float(full_count / max(int(r["prototype_count"]), 1))
    write_csv(out / "RAW_METRICS.csv", raw)
    write_csv(out / "MEAN_BY_K.csv", mean_by_k)
    write_csv(out / "PER_CATEGORY.csv", percat)
    write_csv(out / "STORAGE_CURVE.csv", storage)
    write_csv(out / "COVERAGE_VS_DENSITY.csv", density + density_summary)
    write_csv(out / "RESPONSIBILITY_STATS.csv", resp_summary)
    write_json(out / "OUTCOME.json", {"outcome": outcome, "smoke": args.smoke, "runtime_sec": time.time() - start})
    write_report(out, outcome, mean_by_k, density_summary, resp_summary, args.smoke, time.time() - start)


def summarize_a(raw):
    rows = []
    vals_by_v = defaultdict(list)
    for r in raw:
        if r["experiment"] == "A_support_saturation":
            vals_by_v[r["variant"]].append(r)
    full_by_cell = {r["cell"]: r for r in raw if r["experiment"] == "A_support_saturation" and r["variant"] == "Full"}
    for variant, vals in sorted(vals_by_v.items()):
        deltas = [gp(r) - gp(full_by_cell[r["cell"]]) for r in vals if r["cell"] in full_by_cell]
        rows.append({"experiment": "A_support_saturation", "variant": variant, "cells": len(vals), "MeanG": safe_mean([gp(r) for r in vals]), "MeanDeltaFull": safe_mean(deltas), "cells_within_Full_minus_1p5": int(sum(d >= -1.5 for d in deltas))})
    return rows


def summarize_density(rows):
    vals = [r["DeltaG_mass_minus_uniform"] for r in rows if "DeltaG_mass_minus_uniform" in r]
    return [{"summary": "B_coverage_vs_density", "cells": len(vals), "mean_DeltaG_mass_minus_uniform": safe_mean(vals), "mass_positive_cells": int(sum(v > 0 for v in vals)), "uniform_positive_cells": int(sum(v < 0 for v in vals))}]


def summarize_resp(rows):
    out = []
    by = defaultdict(list)
    for r in rows:
        by[(r["cell"], r["variant"], r["memory_class"])].append(r)
    for (cell, variant, memcls), vals in sorted(by.items()):
        out.append({"cell": cell, "variant": variant, "memory_class": memcls, "H_mean": safe_mean([v["H_mean"] for v in vals]), "N_eff_mean": safe_mean([v["N_eff_mean"] for v in vals]), "I_QK_mean": safe_mean([v["I_QK"] for v in vals]), "kappa": safe_mean([v["kappa"] for v in vals])})
    return out


def decide(mean_by_k, density, density_summary):
    sat = [r for r in mean_by_k if r["variant"].startswith("FPS-") and r["MeanDeltaFull"] >= -1.0 and r["cells_within_Full_minus_1p5"] >= max(1, min(5, int(r["cells"])))]
    sat_k = min([int(r["variant"].split("-")[1]) for r in sat], default=None)
    ds = density_summary[0] if density_summary else {}
    if ds.get("mass_positive_cells", 0) >= 5 and ds.get("mean_DeltaG_mass_minus_uniform", -999) >= 0.5:
        b = "DENSITY_INFORMATION_SUPPORTED"
    elif ds.get("uniform_positive_cells", 0) >= 5 and ds.get("mean_DeltaG_mass_minus_uniform", 999) <= -0.5:
        b = "SUPPORT_COVERAGE_SUPPORTED"
    else:
        b = "COVERAGE_DENSITY_INCONCLUSIVE"
    return {"practical_saturation_K": sat_k, "coverage_density": b}


def write_report(out, outcome, mean_by_k, density_summary, resp_summary, smoke, runtime):
    lines = ["# Memory Support / Mode Saturation", "", f"Smoke: `{smoke}`", "", f"Outcome: `{json.dumps(outcome)}`", "", "## Mode Saturation", "", "| Variant | MeanG | MeanDeltaFull | Cells within Full-1.5 |", "|---|---:|---:|---:|"]
    for r in mean_by_k:
        lines.append(f"| {r['variant']} | {safe_float(r['MeanG']):.4f} | {safe_float(r['MeanDeltaFull']):+.4f} | {r['cells_within_Full_minus_1p5']} |")
    lines.extend(["", "## Coverage vs Density", "", json.dumps(density_summary, indent=2, default=json_default), "", "## Responsibility", "", "See `RESPONSIBILITY_STATS.csv` for H / N_eff / I_QK.", "", "## Required Answers", "", "Q1/Q2/Q3/Q4 are answered from the formal outputs. Smoke is only a pipeline check." if smoke else "Q1: compare compact FPS and Full rows. Q2: `practical_saturation_K` in OUTCOME. Q3: `coverage_density` in OUTCOME. Q4: `RESPONSIBILITY_STATS.csv`.", "", f"Runtime: {runtime:.1f}s.", "", "STOP: no PCA/Gaussian/OT/gate follow-up was started."])
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
