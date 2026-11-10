import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_liv_memory_support_modes import (
    OUT as SUPPORT_OUT,
    ArrayConcat,
    build_full_visa_memory,
    build_source_memory,
    eval_maps,
    external_cell,
    npz_array_mmap,
    select_centers,
    target_cell,
)
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import gp, top1_image_score
from experiments.phase_xxxii_hard_normal_memory import K_PER_IMAGE, fps_indices
from experiments.phase_xxxviii_visual_state_representation import FEATURE_CACHE, load_fold, norm_np


OUT = Path("results/phase_mrad_remem_residual_evolution")
SIDE = 37
SEED = 0
CHUNK = 8192


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


def max_sim(q, mem):
    q = norm_np(np.asarray(q, dtype=np.float32))
    mem = norm_np(np.asarray(mem, dtype=np.float32))
    if mem.shape[0] == 0:
        return np.full((q.shape[0],), -np.inf, dtype=np.float32)
    best = np.full((q.shape[0],), -np.inf, dtype=np.float32)
    for start in range(0, mem.shape[0], CHUNK):
        best = np.maximum(best, (q @ mem[start:start + CHUNK].T).max(axis=1))
    return best


def score_static_patch(q, mem_n, mem_a):
    ns = max_sim(q, mem_n)
    a = max_sim(q, mem_a)
    return a - ns, ns, a


def score_dynamic_patch(q, mem_n, mem_a, residual):
    s0, ns, a = score_static_patch(q, mem_n, mem_a)
    if residual.shape[0] == 0:
        nr = np.full_like(ns, -np.inf)
        n = ns
        source_used = np.ones_like(ns, dtype=bool)
    else:
        nr = max_sim(q, residual)
        n = np.maximum(ns, nr)
        source_used = ns >= nr
    return a - n, ns, nr, a, source_used


def source_patch_stream(source, fold, smoke):
    if source == "FullVisA":
        dataset_name, dataset, _rot, train_indices, held_indices = load_fold("D2", 0)
        cache = FEATURE_CACHE / "D2_fold0_clip.npz"
        patch = ArrayConcat([npz_array_mmap(cache, "train_patch"), npz_array_mmap(cache, "held_patch")])
        indices = list(train_indices) + list(held_indices)
    else:
        dataset_name, dataset, _rot, train_indices, _held_indices = load_fold(source, int(fold))
        cache = FEATURE_CACHE / f"{source}_fold{int(fold)}_clip.npz"
        patch = npz_array_mmap(cache, "train_patch")
        indices = list(train_indices)
    if smoke:
        indices = indices[:48]
        patch = patch[:48]
    return dataset_name, dataset, indices, patch


def source_reliability_thresholds(source, fold, mem_n, mem_a, smoke):
    if smoke:
        sample = norm_np(mem_n[: min(512, mem_n.shape[0])])
        supports, patch_scores = [], []
        for i in range(sample.shape[0]):
            mask = np.ones(sample.shape[0], dtype=bool)
            mask[i] = False
            if mask.any():
                supports.append(float(max_sim(sample[i:i + 1], sample[mask])[0]))
            ps, _ns, _a = score_static_patch(sample[i:i + 1], mem_n, mem_a)
            patch_scores.append(float(ps[0]))
        return {
            "theta_source": float(np.percentile(supports, 5)) if supports else 1.0,
            "patch_score_normal_threshold": float(np.percentile(patch_scores, 95)) if patch_scores else -np.inf,
            "image_score_normal_threshold": float(np.percentile(patch_scores, 95)) if patch_scores else -np.inf,
            "memory_budget": int(min(256, max(K_PER_IMAGE, mem_n.shape[0] // 16))),
            "source_normal_images": int(max(1, mem_n.shape[0] // K_PER_IMAGE)),
            "smoke_threshold_source": "memory_bank_sample",
        }
    _name, dataset, indices, patch = source_patch_stream(source, fold, smoke)
    owner_tokens, owners = [], []
    image_scores = []
    normal_image_count = 0
    for pos, idx in enumerate(indices):
        item = dataset[idx]
        if int(item["anomaly"]) != 0:
            continue
        normal_image_count += 1
        q = norm_np(np.asarray(patch[pos], dtype=np.float32))
        s, _ns, _a = score_static_patch(q, mem_n, mem_a)
        image_scores.append(top1_image_score(s.reshape(SIDE, SIDE)))
        sel = fps_indices(q, min(K_PER_IMAGE, q.shape[0]))
        owner_tokens.append(q[sel])
        owners.extend([normal_image_count - 1] * len(sel))
    if not owner_tokens:
        return {
            "theta_source": 1.0,
            "patch_score_normal_threshold": -np.inf,
            "image_score_normal_threshold": -np.inf,
            "memory_budget": K_PER_IMAGE,
            "source_normal_images": 0,
        }
    tokens = norm_np(np.concatenate(owner_tokens, axis=0))
    owners = np.asarray(owners, dtype=np.int64)
    supports = []
    patch_scores = []
    for i in range(tokens.shape[0]):
        mask = owners != owners[i]
        if mask.any():
            supports.append(float(max_sim(tokens[i:i + 1], tokens[mask])[0]))
        ps, _ns, _a = score_static_patch(tokens[i:i + 1], mem_n, mem_a)
        patch_scores.append(float(ps[0]))
    return {
        "theta_source": float(np.percentile(supports, 5)) if supports else 1.0,
        "patch_score_normal_threshold": float(np.percentile(patch_scores, 95)) if patch_scores else -np.inf,
        "image_score_normal_threshold": float(np.percentile(image_scores, 95)) if image_scores else -np.inf,
        "memory_budget": int(K_PER_IMAGE * normal_image_count),
        "source_normal_images": int(normal_image_count),
    }


def fps_prune(memory, budget):
    if memory.shape[0] <= budget:
        return norm_np(memory)
    keep = fps_indices(memory, int(budget))
    return norm_np(memory[keep])


def append_residual(memory, candidates, budget):
    if candidates.size == 0 or budget <= 0:
        return memory, 0
    candidates = norm_np(candidates)
    if memory.shape[0] > 0:
        sim = max_sim(candidates, memory)
        candidates = candidates[sim < 0.995]
    if candidates.size == 0:
        return memory, 0
    merged = candidates if memory.shape[0] == 0 else np.concatenate([memory, candidates], axis=0)
    before = memory.shape[0]
    pruned = fps_prune(merged, budget)
    return pruned, max(0, int(pruned.shape[0] - before))


def evaluate_online(cell_id, target, dataset, idx_or_rows, patch, kind, mem_n, mem_a, thresholds, variant, blindspot, smoke):
    n_img = min(24, patch.shape[0]) if smoke else patch.shape[0]
    maps = np.zeros((n_img, SIDE, SIDE), dtype=np.float16)
    image = np.zeros((n_img,), dtype=np.float32)
    residual = np.zeros((0, mem_n.shape[1]), dtype=np.float32)
    evolution, usage = [], []
    for i in range(n_img):
        q = norm_np(np.asarray(patch[i], dtype=np.float32))
        if variant == "E0_STATIC_MRAD":
            s, ns, a = score_static_patch(q, mem_n, mem_a)
            nr = np.full_like(ns, -np.inf)
            source_used = np.ones_like(ns, dtype=bool)
        else:
            s, ns, nr, a, source_used = score_dynamic_patch(q, mem_n, mem_a, residual)
        m = s.reshape(SIDE, SIDE).astype(np.float32)
        maps[i] = m.astype(np.float16)
        img_score = top1_image_score(m)
        image[i] = img_score

        reliable = (img_score <= thresholds["image_score_normal_threshold"]) & (s <= thresholds["patch_score_normal_threshold"])
        blind = ns < thresholds["theta_source"]
        allowed = reliable & (blind if blindspot else True)
        if variant != "E0_STATIC_MRAD":
            residual, accepted = append_residual(residual, q[allowed], int(thresholds["memory_budget"]))
        else:
            accepted = 0
        evolution.append({
            "cell": cell_id,
            "target": target,
            "variant": variant,
            "image_pos": i,
            "memory_size": int(residual.shape[0]),
            "accepted_patches": int(accepted),
            "reliable_patch_count": int(np.sum(reliable)),
            "blindspot_candidate_count": int(np.sum(reliable & blind)),
            "blindspot_candidate_ratio": float(np.mean(reliable & blind)),
            "image_score": float(img_score),
        })
        usage.append({
            "cell": cell_id,
            "target": target,
            "variant": variant,
            "image_pos": i,
            "source_normal_explained_ratio": float(np.mean(source_used)),
            "residual_normal_explained_ratio": float(np.mean(~source_used)),
            "mean_source_support": float(np.mean(ns)),
            "mean_residual_support": float(np.mean(nr[np.isfinite(nr)])) if np.any(np.isfinite(nr)) else float("nan"),
            "mean_anomaly_support": float(np.mean(a)),
        })
    metric, per = eval_maps(cell_id, target, dataset, idx_or_rows[:n_img], kind, image, maps)
    metric.update({"cell": cell_id, "target": target, "variant": variant, "G_pixel": gp(metric)})
    per_rows = []
    for cat, vals in per.items():
        rr = {"cell": cell_id, "target": target, "category": cat, "variant": variant, **vals}
        rr["G_pixel"] = gp(rr)
        per_rows.append(rr)
    return metric, per_rows, evolution, usage


def cells(smoke):
    if smoke:
        return [("FullVisA", "full", *external_cell("BTAD", True))]
    return [
        ("D1", 0, *target_cell("D1", 0, False)),
        ("D2", 0, *target_cell("D2", 0, False)),
        ("FullVisA", "full", *external_cell("BTAD", False)),
        ("FullVisA", "full", *external_cell("CVC-ClinicDB", False)),
    ]


def summarize(raw):
    base = {r["cell"]: r for r in raw if r["variant"] == "E0_STATIC_MRAD"}
    rows = []
    for r in raw:
        b = base.get(r["cell"])
        gain = gp(r) - gp(b) if b else float("nan")
        row = dict(r)
        row["E1_minus_E0_G"] = gain if r["variant"] == "E1_SOURCE_ANCHORED_RESIDUAL" else ""
        rows.append(row)
    return rows


def decide(raw):
    e1 = [r for r in raw if r["variant"] == "E1_SOURCE_ANCHORED_RESIDUAL"]
    base = {r["cell"]: r for r in raw if r["variant"] == "E0_STATIC_MRAD"}
    gains = [gp(r) - gp(base[r["cell"]]) for r in e1 if r["cell"] in base]
    if len(gains) < 4:
        return "SMOKE_ONLY_NOT_DECIDED", {"gains": gains}
    positive = sum(g > 0 for g in gains)
    mean_gain = float(np.mean(gains))
    min_gain = float(np.min(gains))
    if positive >= 3 and mean_gain >= 1.0 and min_gain > -1.5:
        return "E1_PASS_CONTROL_E2_REQUIRED", {"positive": positive, "mean_gain": mean_gain, "min_gain": min_gain}
    return "RESIDUAL_MEMORY_EVOLUTION_NO_SIGNAL", {"positive": positive, "mean_gain": mean_gain, "min_gain": min_gain}


def write_report(out, outcome, info, raw, smoke):
    lines = [
        "# MRAD x ReMem Source-Anchored Residual Memory Evolution",
        "",
        f"Smoke: `{smoke}`",
        f"Outcome: `{outcome}`",
        "",
        "This is online/test-time ZSAD. It is not reported as strict source-only ZSAD.",
        "",
        "| Cell | Variant | P-AUROC | P-AP | G_pixel |",
        "|---|---|---:|---:|---:|",
    ]
    for r in raw:
        lines.append(f"| {r['cell']} | {r['variant']} | {safe_float(r.get('P-AUROC')):.4f} | {safe_float(r.get('P-AP')):.4f} | {gp(r):.4f} |")
    lines.extend([
        "",
        "## Gate",
        "",
        json.dumps(info, indent=2, default=json_default),
        "",
        "E2 is intentionally not started unless E1 satisfies the 4-cell gate.",
    ])
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    out = OUT / ("smoke" if args.smoke else "formal")
    out.mkdir(parents=True, exist_ok=True)
    raw, percat, evo, usage, thresh_rows = [], [], [], [], []
    for source, fold, cell_id, target, dataset, idx_or_rows, patch, kind in cells(args.smoke):
        print(f"[MRAD-ReMem] cell {cell_id}", flush=True)
        memory = build_full_visa_memory(args.smoke)[0] if source == "FullVisA" else build_source_memory(source, int(fold), args.smoke)[0]
        mem_n = select_centers(memory, "normal", "ALL")
        mem_a = select_centers(memory, "anomaly", "ALL")
        thresholds = source_reliability_thresholds(source, fold, mem_n, mem_a, args.smoke)
        thresh_rows.append({"cell": cell_id, "source": source, "fold": fold, **thresholds})
        for variant, blindspot in [("E0_STATIC_MRAD", False), ("E1_SOURCE_ANCHORED_RESIDUAL", True)]:
            print(f"[MRAD-ReMem] {cell_id} {variant}", flush=True)
            metric, pc, ev, us = evaluate_online(cell_id, target, dataset, idx_or_rows, patch, kind, mem_n, mem_a, thresholds, variant, blindspot, args.smoke)
            raw.append(metric); percat.extend(pc); evo.extend(ev); usage.extend(us)
    per_dataset = summarize(raw)
    outcome, info = decide(raw)
    write_csv(out / "RAW_METRICS.csv", raw)
    write_csv(out / "PER_DATASET.csv", per_dataset)
    write_csv(out / "PER_CATEGORY.csv", percat)
    write_csv(out / "MEMORY_EVOLUTION.csv", evo)
    write_csv(out / "SOURCE_VS_RESIDUAL_USAGE.csv", usage)
    write_csv(out / "SOURCE_THRESHOLDS.csv", thresh_rows)
    write_json(out / "OUTCOME.json", {"outcome": outcome, "smoke": args.smoke, "gate": info, "runtime_sec": time.time() - start, "online_test_time_zsad": True})
    write_report(out, outcome, info, raw, args.smoke)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
