import csv
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_xa_smoke import write_csv
from experiments.phase_xxxiv_insplad_cross_asset import (
    CACHE as XXXIV_CACHE,
    PATCHES,
    PREVIOUS_VERIFIED_BUDGET,
    SIDE,
    TOPK,
    asset_records,
    build_memory,
    collect_dataset,
    image_score_from_patch_scores,
    mask_for,
    safe_ap,
    safe_auc,
    topk_distances,
    upsample_patch_scores,
)


OUT = Path("results/phase_xxxvi_homologous_support")
HOMO_CACHE = Path(r"E:\WorkSpace\phase_xxxvi_homology_cache")
NORMAL_PIXEL_SAMPLE_CAP = 2_000_000
EPS = 1e-12


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def norm_rows(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)


def provenance_key(path):
    stem = Path(path).stem
    stem = re.sub(r"_jpg\.rf\..*$", "", stem)
    stem = stem.replace(" ", "-")
    stem = re.sub(r"[_-]\d+$", "", stem)
    return stem.lower()


def file_sha1(path):
    import hashlib
    h = hashlib.sha1()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_feature_index(assets):
    feature_paths, metas = {}, {}
    for asset in assets:
        meta_path = XXXIV_CACHE / f"features_clip_{asset['name']}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        feature_paths[asset["name"]] = Path(meta.get("patch_npy", XXXIV_CACHE / f"features_clip_{asset['name']}.patch.npy"))
        metas[asset["name"]] = meta
    return feature_paths, metas


def split_normal_groups(asset, meta, seed):
    normal_positions = [i for i, r in enumerate(meta["rows"]) if int(r["label"]) == 0]
    groups = defaultdict(list)
    for pos in normal_positions:
        groups[provenance_key(meta["rows"][pos]["path"])].append(pos)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    keys = list(np.asarray(keys)[rng.permutation(len(keys))])
    cut = max(1, len(keys) // 2)
    ref_keys = set(keys[:cut])
    ref = sorted([p for k in ref_keys for p in groups[k]])
    evn = sorted([p for k in keys[cut:] for p in groups[k]])
    anom = sorted([i for i, r in enumerate(meta["rows"]) if int(r["label"]) == 1])
    return ref, evn, anom, groups, ref_keys


def local_patch_pick(patches, count, mode, rng):
    if count <= 0:
        return np.zeros((0,), dtype=np.int64)
    n = patches.shape[0]
    count = min(count, n)
    if mode == "random":
        return np.sort(rng.choice(n, size=count, replace=False)).astype(np.int64)
    x = norm_rows(patches)
    if n <= count:
        return np.arange(n, dtype=np.int64)
    centroid = norm_rows(x.mean(axis=0, keepdims=True))[0]
    first = int(np.argmin(x @ centroid))
    selected = [first]
    max_sim = x @ x[first]
    for _ in range(1, count):
        idx = int(np.argmin(max_sim))
        selected.append(idx)
        max_sim = np.maximum(max_sim, x @ x[idx])
    return np.asarray(selected, dtype=np.int64)


def allocate_counts(total, n):
    if n <= 0:
        return []
    base, rem = divmod(total, n)
    return [base + int(i < rem) for i in range(n)]


def build_memory_positions(asset_positions, feature_paths, metas, total_budget, mode, seed=0):
    rng = np.random.default_rng(seed)
    parts, entries, comp = [], [], []
    assets = sorted(asset_positions)
    per_asset, rem = divmod(total_budget, len(assets))
    for ai, asset in enumerate(assets):
        target = per_asset + int(ai < rem)
        positions = sorted(asset_positions[asset])
        counts = allocate_counts(target, len(positions))
        arr = np.load(feature_paths[asset], mmap_mode="r")
        meta = metas[asset]
        asset_parts, asset_entries = [], []
        for pos, cnt in zip(positions, counts):
            patch = arr[pos].astype(np.float32)
            ids = local_patch_pick(patch, cnt, mode, rng)
            if ids.size:
                asset_parts.append(patch[ids])
                for pid in ids:
                    asset_entries.append({"source_asset": asset, "source_image_id": meta["rows"][pos]["path"], "source_position": int(pos), "memory_patch_id": int(pid)})
        if asset_parts:
            mat = np.concatenate(asset_parts, axis=0)
            if mat.shape[0] > target:
                keep = local_patch_pick(mat, target, mode, rng)
                mat = mat[keep]
                asset_entries = [asset_entries[int(i)] for i in keep]
            parts.append(mat)
            entries.extend(asset_entries)
            comp.append({"source_asset": asset, "mode": mode, "target_entries": target, "actual_entries": int(mat.shape[0]), "source_images": len(positions)})
    bank = np.concatenate(parts, axis=0).astype(np.float32) if parts else np.zeros((0, 1024), dtype=np.float32)
    return norm_rows(bank), entries, comp


def patch_labels_from_mask(mask):
    h, w = mask.shape
    labels = np.zeros((SIDE, SIDE), dtype=np.uint8)
    for py in range(SIDE):
        y0, y1 = int(round(py * h / SIDE)), int(round((py + 1) * h / SIDE))
        for px in range(SIDE):
            x0, x1 = int(round(px * w / SIDE)), int(round((px + 1) * w / SIDE))
            labels[py, px] = int(mask[y0:y1, x0:x1].mean() > 0)
    return labels.reshape(-1)


def evaluate_variant(target, split_id, variant, bank, entries, eval_normal, eval_anom, feature_paths, metas, device):
    arr = np.load(feature_paths[target], mmap_mode="r")
    meta = metas[target]
    y_img, s_img = [], []
    y_pix_parts, s_pix_parts = [], []
    y_patch_parts, s_patch_parts = [], []
    normal_top5, anomaly_top5, normal_top1, anomaly_top1, normal_disp, anomaly_disp = [], [], [], [], [], []
    normal_rng = np.random.default_rng(abs(hash((target, split_id, variant))) % (2**32))
    normal_pixel_seen = 0
    eval_positions = [(p, 0) for p in eval_normal] + [(p, 1) for p in eval_anom]
    for pos, lab in eval_positions:
        patch = arr[pos].astype(np.float32)
        dist, _idx = topk_distances(patch, bank, TOPK, device)
        patch_score = dist.mean(axis=1).reshape(SIDE, SIDE)
        y_img.append(lab)
        s_img.append(image_score_from_patch_scores(patch_score))
        row = meta["rows"][pos]
        if lab == 1:
            with Image.open(row["mask_path"]) as m:
                mask = (np.asarray(m.convert("L")) > 0).astype(np.uint8)
            full = upsample_patch_scores(patch_score, mask.shape)
            y_pix_parts.append(mask.reshape(-1))
            s_pix_parts.append(full.reshape(-1).astype(np.float32))
            py = patch_labels_from_mask(mask)
            y_patch_parts.append(py)
            s_patch_parts.append(patch_score.reshape(-1).astype(np.float32))
            anomaly_top5.extend(dist.mean(axis=1).tolist())
            anomaly_top1.extend(dist[:, 0].tolist())
            anomaly_disp.extend(dist.std(axis=1).tolist())
        else:
            with Image.open(row["path"]) as im:
                shape = (im.size[1], im.size[0])
            full = upsample_patch_scores(patch_score, shape).reshape(-1)
            remaining = max(0, NORMAL_PIXEL_SAMPLE_CAP - normal_pixel_seen)
            if remaining > 0:
                take = min(remaining, full.size)
                ids = normal_rng.choice(full.size, size=take, replace=False)
                y_pix_parts.append(np.zeros(take, dtype=np.uint8))
                s_pix_parts.append(full[ids].astype(np.float32))
                normal_pixel_seen += take
            y_patch_parts.append(np.zeros(PATCHES, dtype=np.uint8))
            s_patch_parts.append(patch_score.reshape(-1).astype(np.float32))
            normal_top5.extend(dist.mean(axis=1).tolist())
            normal_top1.extend(dist[:, 0].tolist())
            normal_disp.extend(dist.std(axis=1).tolist())
    y_img, s_img = np.asarray(y_img), np.asarray(s_img)
    y_pix, s_pix = np.concatenate(y_pix_parts), np.concatenate(s_pix_parts)
    y_patch, s_patch = np.concatenate(y_patch_parts), np.concatenate(s_patch_parts)
    metric = {
        "target_asset": target,
        "split": split_id,
        "variant": variant,
        "I-AUROC": safe_auc(y_img, s_img),
        "I-AP": safe_ap(y_img, s_img),
        "P-AUROC": safe_auc(y_pix, s_pix),
        "P-AP": safe_ap(y_pix, s_pix),
        "pixel_prevalence": float(y_pix.mean() * 100.0),
        "P-AP_lift": safe_ap(y_pix, s_pix) - float(y_pix.mean() * 100.0),
        "G_local": 0.5 * (safe_auc(y_pix, s_pix) + safe_ap(y_pix, s_pix)),
        "Patch_AUROC": safe_auc(y_patch, s_patch),
        "Patch_AP": safe_ap(y_patch, s_patch),
        "normal_pixels_sampled": int((y_pix == 0).sum()),
        "anomaly_pixels_full": int((y_pix == 1).sum()),
    }
    support = {
        "target_asset": target,
        "split": split_id,
        "variant": variant,
        "top5_mean_AUROC": safe_auc(np.r_[np.zeros(len(normal_top5)), np.ones(len(anomaly_top5))], np.r_[normal_top5, anomaly_top5]),
        "top5_mean_AP": safe_ap(np.r_[np.zeros(len(normal_top5)), np.ones(len(anomaly_top5))], np.r_[normal_top5, anomaly_top5]),
        "top1_AUROC": safe_auc(np.r_[np.zeros(len(normal_top1)), np.ones(len(anomaly_top1))], np.r_[normal_top1, anomaly_top1]),
        "top5_dispersion_AUROC": safe_auc(np.r_[np.zeros(len(normal_disp)), np.ones(len(anomaly_disp))], np.r_[normal_disp, anomaly_disp]),
        "normal_q50_top5": float(np.quantile(normal_top5, 0.50)),
        "normal_q75_top5": float(np.quantile(normal_top5, 0.75)),
        "normal_q90_top5": float(np.quantile(normal_top5, 0.90)),
        "normal_q95_top5": float(np.quantile(normal_top5, 0.95)),
        "normal_q99_top5": float(np.quantile(normal_top5, 0.99)),
        "anomaly_q50_top5": float(np.quantile(anomaly_top5, 0.50)),
        "anomaly_q75_top5": float(np.quantile(anomaly_top5, 0.75)),
        "anomaly_q90_top5": float(np.quantile(anomaly_top5, 0.90)),
        "median_gap_anomaly_minus_normal": float(np.median(anomaly_top5) - np.median(normal_top5)),
        "ks_top5": float(ks_2samp(normal_top5, anomaly_top5).statistic),
    }
    return metric, support


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    HOMO_CACHE.mkdir(parents=True, exist_ok=True)
    start = time.time()
    assets, _count_rows, _image_rows, _dups = collect_dataset()
    assets_by_name = {a["name"]: a for a in assets}
    asset_names = [a["name"] for a in assets]
    feature_paths, metas = load_feature_index(assets)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    dup_rows = []
    prov = {"grouping_rule": "remove roboflow suffix, normalize spaces, strip final numeric crop/index suffix", "assets": {}, "limitations": "near-duplicate/adjacent-frame detection is filename-provenance plus exact file hash only; no perceptual hash or visual matching was used."}
    for asset in asset_names:
        rows = metas[asset]["rows"]
        hashes = defaultdict(list)
        groups = defaultdict(list)
        for i, r in enumerate(rows):
            if int(r["label"]) == 0:
                h = file_sha1(r["path"])
                k = provenance_key(r["path"])
                hashes[h].append(r["path"])
                groups[k].append(r["path"])
        prov["assets"][asset] = {"normal_images": sum(1 for r in rows if int(r["label"]) == 0), "provenance_groups": len(groups), "multi_image_groups": sum(len(v) > 1 for v in groups.values()), "exact_duplicate_hash_groups": sum(len(v) > 1 for v in hashes.values())}
        for k, v in groups.items():
            if len(v) > 1:
                dup_rows.append({"asset": asset, "audit_type": "provenance_group", "key": k, "count": len(v), "examples": "|".join(v[:3])})
        for h, v in hashes.items():
            if len(v) > 1:
                dup_rows.append({"asset": asset, "audit_type": "exact_hash_duplicate", "key": h, "count": len(v), "examples": "|".join(v[:3])})
    (OUT / "PROVENANCE_AUDIT.json").write_text(json.dumps(prov, indent=2, default=json_default), encoding="utf-8")
    write_csv(OUT / "DUPLICATE_AUDIT.csv", dup_rows if dup_rows else [{"status": "NO_DUPLICATE_OR_MULTI_GROUP_ROWS"}])

    split_rows, split_payload = [], []
    raw, image_rows, comp_rows, budget_rows, support_rows = [], [], [], [], []
    tail_rows, anomaly_rows = [], []
    manifest = []
    for asset in asset_names:
        meta = metas[asset]
        for split_id, seed in enumerate([0, 1, 2]):
            ref, evn, anom, groups, ref_keys = split_normal_groups(asset, meta, seed)
            split_rows.append({"target_asset": asset, "split": split_id, "seed": seed, "reference_normal_images": len(ref), "eval_normal_images": len(evn), "anomaly_images": len(anom), "reference_groups": len(ref_keys), "eval_groups": len(groups) - len(ref_keys)})
            split_payload.append({"target_asset": asset, "split": split_id, "seed": seed, "reference_normal_positions": ref, "eval_normal_positions": evn, "anomaly_positions": anom})
            wrong_assets = [x for x in asset_names if x != asset]
            same_patches = len(ref) * PATCHES
            wrong_patch_counts = {w: sum(1 for r in metas[w]["rows"] if int(r["label"]) == 0) * PATCHES for w in wrong_assets}
            B = min(PREVIOUS_VERIFIED_BUDGET, same_patches, min(wrong_patch_counts.values()))
            budget_rows.append({"target_asset": asset, "split": split_id, "previous_budget": PREVIOUS_VERIFIED_BUDGET, "same_ref_patches": same_patches, "smallest_wrong_asset_patches": min(wrong_patch_counts.values()), "final_budget": B})

            normal_positions_all = {a: [i for i, r in enumerate(metas[a]["rows"]) if int(r["label"]) == 0] for a in asset_names}
            variants = {
                "P0_CROSS_ASSET_FPS": ({w: normal_positions_all[w] for w in wrong_assets}, "fps"),
                "P1_CROSS_ASSET_RANDOM": ({w: normal_positions_all[w] for w in wrong_assets}, "random"),
                "P2_SAME_ASSET_FPS": ({asset: ref}, "fps"),
                "P3_SAME_ASSET_RANDOM": ({asset: ref}, "random"),
                "P4_WRONG_ASSET_1": ({wrong_assets[0]: normal_positions_all[wrong_assets[0]]}, "fps"),
                "P5_WRONG_ASSET_2": ({wrong_assets[1]: normal_positions_all[wrong_assets[1]]}, "fps"),
            }
            for variant, (source_pos, mode) in variants.items():
                print(f"[XXXVI-A] {asset} split{split_id} {variant}", flush=True)
                bank, entries, comp = build_memory_positions(source_pos, feature_paths, metas, B, mode, seed=0)
                comp_rows.extend([{"target_asset": asset, "split": split_id, "variant": variant, **c} for c in comp])
                metric, support = evaluate_variant(asset, split_id, variant, bank, entries, evn, anom, feature_paths, metas, device)
                raw.append(metric)
                image_rows.append({k: metric[k] for k in ["target_asset", "split", "variant", "I-AUROC", "I-AP"]})
                support_rows.append(support)
                tail_rows.append({k: support[k] for k in ["target_asset", "split", "variant", "normal_q50_top5", "normal_q75_top5", "normal_q90_top5", "normal_q95_top5", "normal_q99_top5"]})
                anomaly_rows.append({k: support[k] for k in ["target_asset", "split", "variant", "anomaly_q50_top5", "anomaly_q75_top5", "anomaly_q90_top5", "median_gap_anomaly_minus_normal"]})
                manifest.append({"target_asset": asset, "split": split_id, "variant": variant, "retrieval_provenance_status": "NOT_SAVED_FULL_RAW_DUE_TO_SIZE_POLICY", "memory_entries": len(entries), "cache_root": str(HOMO_CACHE)})

    (OUT / "SPLIT_PROTOCOL.json").write_text(json.dumps({"protocol": "same-asset normal-reference 50/50 provenance-group split; anomalies evaluation only", "splits": split_payload}, indent=2, default=json_default), encoding="utf-8")
    write_csv(OUT / "SPLIT_COUNTS.csv", split_rows)
    write_csv(OUT / "MEMORY_BUDGET.csv", budget_rows)
    write_csv(OUT / "MEMORY_COMPOSITION.csv", comp_rows)
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "IMAGE_LEVEL_SECONDARY.csv", image_rows)
    write_csv(OUT / "NORMAL_DISTANCE_TAIL.csv", tail_rows)
    write_csv(OUT / "ANOMALY_DISTANCE.csv", anomaly_rows)
    write_csv(OUT / "SUPPORT_SEPARATION.csv", support_rows)
    write_csv(OUT / "RETRIEVAL_PROVENANCE_MANIFEST.csv", manifest)

    by = {(r["target_asset"], int(r["split"]), r["variant"]): r for r in raw}
    homology, wrong_ctrl, same_fps_random = [], [], []
    for asset in asset_names:
        for split_id in [0, 1, 2]:
            p0, p1, p2, p3, p4, p5 = [by[(asset, split_id, v)] for v in ["P0_CROSS_ASSET_FPS", "P1_CROSS_ASSET_RANDOM", "P2_SAME_ASSET_FPS", "P3_SAME_ASSET_RANDOM", "P4_WRONG_ASSET_1", "P5_WRONG_ASSET_2"]]
            homology.append({"target_asset": asset, "split": split_id, "H_FPS_G": p2["G_local"] - p0["G_local"], "H_RANDOM_G": p3["G_local"] - p1["G_local"], "H_FPS_dP-AUROC": p2["P-AUROC"] - p0["P-AUROC"], "H_FPS_dP-AP": p2["P-AP"] - p0["P-AP"], "H_RANDOM_dP-AUROC": p3["P-AUROC"] - p1["P-AUROC"], "H_RANDOM_dP-AP": p3["P-AP"] - p1["P-AP"], "SameFPS_gt_CrossFPS": p2["G_local"] > p0["G_local"], "SameRandom_gt_CrossRandom": p3["G_local"] > p1["G_local"]})
            wm = 0.5 * (p4["G_local"] + p5["G_local"])
            wrong_ctrl.append({"target_asset": asset, "split": split_id, "SameFPS_G": p2["G_local"], "WrongMean_G": wm, "WrongBest_G": max(p4["G_local"], p5["G_local"]), "Same_minus_WrongMean_G": p2["G_local"] - wm, "Same_gt_Wrong1": p2["G_local"] > p4["G_local"], "Same_gt_Wrong2": p2["G_local"] > p5["G_local"]})
            same_fps_random.append({"target_asset": asset, "split": split_id, "P2_minus_P3_dP-AUROC": p2["P-AUROC"] - p3["P-AUROC"], "P2_minus_P3_dP-AP": p2["P-AP"] - p3["P-AP"], "P2_minus_P3_G": p2["G_local"] - p3["G_local"]})
    write_csv(OUT / "HOMOLOGY_GAIN.csv", homology)
    write_csv(OUT / "WRONG_ASSET_CONTROL.csv", wrong_ctrl)
    write_csv(OUT / "SAME_FPS_VS_RANDOM.csv", same_fps_random)

    stability = []
    for asset in asset_names:
        vals = [r["H_FPS_G"] for r in homology if r["target_asset"] == asset]
        same_vals = [by[(asset, s, "P2_SAME_ASSET_FPS")]["G_local"] for s in [0, 1, 2]]
        stability.append({"target_asset": asset, "same_fps_G_mean": float(np.mean(same_vals)), "same_fps_G_std": float(np.std(same_vals)), "same_fps_G_min": float(np.min(same_vals)), "same_fps_G_max": float(np.max(same_vals)), "H_FPS_mean": float(np.mean(vals)), "H_FPS_std": float(np.std(vals)), "H_FPS_min": float(np.min(vals)), "H_FPS_max": float(np.max(vals)), "SPLIT_UNSTABLE": bool(max(vals) > 2.0 and min(vals) < -1.0)})
    write_csv(OUT / "SPLIT_STABILITY.csv", stability)

    per_asset_split = []
    for r in raw:
        per_asset_split.append(dict(r))
    write_csv(OUT / "PER_ASSET_SPLIT.csv", per_asset_split)
    grouped = []
    for variant in sorted({r["variant"] for r in raw}):
        vals = [r for r in raw if r["variant"] == variant]
        grouped.append({"variant": variant, "mean_P-AUROC": float(np.mean([r["P-AUROC"] for r in vals])), "mean_P-AP": float(np.mean([r["P-AP"] for r in vals])), "mean_G_local": float(np.mean([r["G_local"] for r in vals])), "mean_I-AUROC": float(np.mean([r["I-AUROC"] for r in vals])), "mean_I-AP": float(np.mean([r["I-AP"] for r in vals]))})
    write_csv(OUT / "GROUPED_METRICS.csv", grouped)

    mean_h_fps = float(np.mean([r["H_FPS_G"] for r in homology]))
    mean_h_rand = float(np.mean([r["H_RANDOM_G"] for r in homology]))
    cells_fps = int(sum(r["SameFPS_gt_CrossFPS"] for r in homology))
    cells_rand = int(sum(r["SameRandom_gt_CrossRandom"] for r in homology))
    asset_h = {a: float(np.mean([r["H_FPS_G"] for r in homology if r["target_asset"] == a])) for a in asset_names}
    mean_ap_imp = float(np.mean([r["H_FPS_dP-AP"] for r in homology]))
    homology_gate = mean_h_fps >= 1.0 and mean_h_rand >= 1.0 and cells_fps >= 7 and cells_rand >= 7 and all(v > 0 for v in asset_h.values()) and mean_ap_imp >= 1.0 and all(v >= -0.50 for v in asset_h.values())
    wrong_mean = float(np.mean([r["Same_minus_WrongMean_G"] for r in wrong_ctrl]))
    wrong_asset_pass = sum(all(r[f"Same_gt_Wrong{i}"] for i in [1, 2]) for a in asset_names for r in [{"Same_gt_Wrong1": np.mean([x["Same_gt_Wrong1"] for x in wrong_ctrl if x["target_asset"] == a]) > 0.5, "Same_gt_Wrong2": np.mean([x["Same_gt_Wrong2"] for x in wrong_ctrl if x["target_asset"] == a]) > 0.5}])
    wrong_gate = wrong_mean >= 0.75 and wrong_asset_pass >= 2
    proto_delta = float(np.mean([abs(r["P2_minus_P3_G"]) for r in same_fps_random]))
    proto_neutral = proto_delta < 0.20
    split_unstable = any(r["SPLIT_UNSTABLE"] for r in stability)
    if split_unstable:
        outcome = "XXXVI_A_REFERENCE_SPLIT_UNSTABLE"
    elif homology_gate and wrong_gate and proto_neutral:
        outcome = "XXXVI_A_ASSET_IDENTITY_STRONG_PROTOTYPE_IDENTITY_WEAK"
    elif homology_gate and wrong_gate:
        outcome = "XXXVI_A_HOMOLOGOUS_SUPPORT_STRONGLY_SUPPORTED"
    elif any(v > 1.0 for v in asset_h.values()) and not all(v > 0 for v in asset_h.values()):
        outcome = "XXXVI_A_HOMOLOGY_ASSET_SPECIFIC"
    else:
        outcome = "XXXVI_A_SAME_ASSET_SUPPORT_NOT_BETTER"

    report = ["# Phase XXXVI-A Homologous Normal Support Audit", "", f"Outcome: **{outcome}**", "", "No backbone forward was run; XXXIV CLIP patch memmap cache was reused.", f"Full-resolution normal pixels were deterministically sampled with cap {NORMAL_PIXEL_SAMPLE_CAP}; anomaly pixels were evaluated fully.", "", "## Grouped Metrics", "", "| Variant | P-AUROC | P-AP | G_local | I-AUROC | I-AP |", "|---|---:|---:|---:|---:|---:|"]
    for r in grouped:
        report.append(f"| {r['variant']} | {r['mean_P-AUROC']:.4f} | {r['mean_P-AP']:.4f} | {r['mean_G_local']:.4f} | {r['mean_I-AUROC']:.4f} | {r['mean_I-AP']:.4f} |")
    report.extend(["", "## Gates", "", f"- mean H_FPS={mean_h_fps:+.4f}; mean H_RANDOM={mean_h_rand:+.4f}; cells FPS/Random positive={cells_fps}/9, {cells_rand}/9.", f"- asset mean H_FPS={json.dumps(asset_h, default=json_default)}.", f"- mean Same-WrongMean G={wrong_mean:+.4f}; wrong asset pass count={wrong_asset_pass}/3.", f"- mean |Same FPS-Random| G={proto_delta:.4f}; prototype neutral={proto_neutral}.", f"- split unstable={split_unstable}.", f"- Runtime={time.time() - start:.1f}s.", "", "Stop: Phase XXXVI-A ended here."])
    (OUT / "XXXVI_A_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    payload = {"phase": "XXXVI-A", "outcome": outcome, "mean_H_FPS_G": mean_h_fps, "mean_H_RANDOM_G": mean_h_rand, "positive_cells_fps": cells_fps, "positive_cells_random": cells_rand, "asset_mean_H_FPS": asset_h, "mean_dP_AP_same_vs_cross_fps": mean_ap_imp, "wrong_mean_gain": wrong_mean, "wrong_asset_pass_count": wrong_asset_pass, "prototype_selection_neutral": proto_neutral, "split_unstable": split_unstable, "normal_pixel_sample_cap": NORMAL_PIXEL_SAMPLE_CAP, "runtime_sec": time.time() - start}
    (OUT / "OUTCOME.json").write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    with open(ROOT / "EXPERIMENT_LOG.md", "a", encoding="utf-8") as fp:
        fp.write("\n\n## Phase XXXVI-A\n")
        fp.write("- Command: python experiments/phase_xxxvi_homologous_support.py\n")
        fp.write("- Protocol: 3 deterministic same-asset normal-reference provenance-group splits per asset; CLIP only; no backbone forward.\n")
        fp.write(f"- mean H_FPS {mean_h_fps:+.4f}; mean H_RANDOM {mean_h_rand:+.4f}; positive cells FPS/Random {cells_fps}/9 and {cells_rand}/9.\n")
        fp.write(f"- wrong asset mean gain {wrong_mean:+.4f}; prototype neutral {proto_neutral}; split unstable {split_unstable}.\n")
        fp.write(f"- Outcome: {outcome}.\n")
    print(json.dumps({"outcome": outcome, "report": str(OUT / "XXXVI_A_REPORT.md"), "runtime_sec": time.time() - start}, indent=2, default=json_default))


if __name__ == "__main__":
    main()
