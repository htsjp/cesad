import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_xa_checkpoint_source_readout import all_indices_for_categories, direction_cfg
from experiments.phase_xa_smoke import load_dataset, write_csv
from experiments.phase_xxiii_carrier_audit import (
    BANK_CAP,
    BETAS,
    EPS,
    IMAGE_SIZE,
    SIDE,
    balanced_normal_patch_bank,
    map_score,
    normal_distance_map,
    rank_calibrator,
)
from phase_xa.source_folds import deterministic_source_folds


OUT = Path("results/phase_xxx_metric_safe_dino")
FEATURE_CACHE = Path("results/phase_xxiii_carrier_audit/feature_cache")
XXIII_OUT = Path("results/phase_xxiii_carrier_audit")
MAP_CACHE = OUT / "map_cache"
SIGMA = 4

VARIANTS = {
    "C0_CLIP_LOCAL": "exact Phase XXIII-A frozen CLIP normal-distance/local diagnostic map",
    "D0_DINO_LOCAL": "exact Phase XXIII-A frozen DINOv2 ViT-L/14 normal-distance map",
    "F0_CLIP_DINO_FUSION": "exact Phase XXIII-A source-selected pixel fusion",
    "S1_CROSS_IMAGE_SHUFFLED_DINO": "exact Phase XXIII-A cross-image shuffled DINO fusion control",
    "S2_SPATIALLY_SHUFFLED_DINO": "not present in Phase XXIII-A artifacts; reported unavailable, not used for gates",
}


def json_default(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def safe_auc(y, s):
    y = np.asarray(y, dtype=np.uint8)
    s = np.asarray(s, dtype=np.float32)
    return float(roc_auc_score(y, s) * 100.0) if np.unique(y).size == 2 else float("nan")


def safe_ap(y, s):
    y = np.asarray(y, dtype=np.uint8)
    s = np.asarray(s, dtype=np.float32)
    return float(average_precision_score(y, s) * 100.0) if np.unique(y).size == 2 else float("nan")


def load_selected_beta():
    path = XXIII_OUT / "FUSION_METRICS.csv"
    rows = {}
    with open(path, newline="", encoding="utf-8") as fp:
        for r in csv.DictReader(fp):
            if r.get("task") == "pixel":
                rows[(r["direction"], int(r["fold"]))] = float(r["selected_beta_clip"])
    return rows


def load_fold(direction, fold):
    dataset_name, data_path, _ = direction_cfg(direction)
    dataset = load_dataset(dataset_name, data_path)
    rotation = deterministic_source_folds(dataset_name)["rotations"][fold]
    train_indices = all_indices_for_categories(dataset, set(rotation["train_categories"]))
    held_indices = all_indices_for_categories(dataset, set(rotation["val_categories"]))
    return dataset_name, dataset, rotation, train_indices, held_indices


def ensure_distance_maps(direction, fold, dataset, train_indices, held_indices, device):
    MAP_CACHE.mkdir(parents=True, exist_ok=True)
    paths = {
        "clip_train": MAP_CACHE / f"{direction}_fold{fold}_clip_train37.npy",
        "clip_held": MAP_CACHE / f"{direction}_fold{fold}_clip_held37.npy",
        "dino_train": MAP_CACHE / f"{direction}_fold{fold}_dino_train37.npy",
        "dino_held": MAP_CACHE / f"{direction}_fold{fold}_dino_held37.npy",
    }
    if all(p.exists() for p in paths.values()):
        return paths

    clip = np.load(FEATURE_CACHE / f"{direction}_fold{fold}_clip.npz", allow_pickle=True)
    dino = np.load(FEATURE_CACHE / f"{direction}_fold{fold}_dino.npz", allow_pickle=True)
    clip_train_patch, clip_held_patch = clip["train_patch"], clip["held_patch"]
    dino_train_patch, dino_held_patch = dino["train_patch"], dino["held_patch"]

    clip_bank = balanced_normal_patch_bank(
        dataset,
        train_indices,
        lambda idx: clip_train_patch[train_indices.index(idx)],
        clip_train_patch.shape[-1],
        seed=0,
    )
    dino_bank = balanced_normal_patch_bank(
        dataset,
        train_indices,
        lambda idx: dino_train_patch[train_indices.index(idx)],
        dino_train_patch.shape[-1],
        seed=0,
    )

    def write_maps(path, patches, bank):
        if path.exists():
            return
        maps = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(patches.shape[0], SIDE, SIDE))
        for pos in range(patches.shape[0]):
            maps[pos] = normal_distance_map(patches[pos].astype(np.float32), bank, device).astype(np.float16)
            if pos % 128 == 0:
                maps.flush()
        maps.flush()

    write_maps(paths["clip_train"], clip_train_patch, clip_bank)
    write_maps(paths["clip_held"], clip_held_patch, clip_bank)
    write_maps(paths["dino_train"], dino_train_patch, dino_bank)
    write_maps(paths["dino_held"], dino_held_patch, dino_bank)
    return paths


def upsample_smooth(map37):
    x = torch.tensor(map37, dtype=torch.float32).view(1, 1, SIDE, SIDE)
    full = F.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False).squeeze().numpy()
    return gaussian_filter(full.astype(np.float32), sigma=SIGMA).astype(np.float32)


def collect_masks(dataset, indices, positions=None):
    if positions is None:
        positions = range(len(indices))
    masks, cats = [], []
    for pos in positions:
        item = dataset[indices[pos]]
        masks.append((item["img_mask"].detach().cpu().numpy() > 0.5).astype(np.uint8).reshape(-1))
        cats.append(item["cls_name"])
    return np.concatenate(masks), cats


def evaluate_variant(dataset, held_indices, maps37, variant, positions=None):
    if positions is None:
        positions = list(range(len(held_indices)))
    y_parts, s_parts = [], []
    by_cat = defaultdict(lambda: {"y": [], "s": []})
    image_scores = []
    for pos in positions:
        item = dataset[held_indices[pos]]
        mask = (item["img_mask"].detach().cpu().numpy() > 0.5).astype(np.uint8).reshape(-1)
        full = upsample_smooth(np.asarray(maps37[pos], dtype=np.float32)).reshape(-1)
        y_parts.append(mask)
        s_parts.append(full)
        by_cat[item["cls_name"]]["y"].append(mask)
        by_cat[item["cls_name"]]["s"].append(full)
        image_scores.append(map_score(np.asarray(maps37[pos], dtype=np.float32)))
    y = np.concatenate(y_parts)
    s = np.concatenate(s_parts).astype(np.float32)
    row = {
        "variant": variant,
        "Pixel_AUROC": safe_auc(y, s),
        "Pixel_AP": safe_ap(y, s),
        "evaluated_pixels": int(y.size),
        "positive_pixels": int(y.sum()),
        "image_top1_mean": float(np.mean(image_scores)) if image_scores else float("nan"),
    }
    percat = []
    for cat, vals in sorted(by_cat.items()):
        cy = np.concatenate(vals["y"])
        cs = np.concatenate(vals["s"]).astype(np.float32)
        percat.append({
            "category": cat,
            "variant": variant,
            "Pixel_AUROC": safe_auc(cy, cs),
            "Pixel_AP": safe_ap(cy, cs),
            "pixels": int(cy.size),
            "positive_pixels": int(cy.sum()),
        })
    return row, percat


def apply_rank_array(arr, calibrator):
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    return np.asarray([calibrator(v) for v in flat], dtype=np.float32).reshape(arr.shape)


def build_fold_variants(paths, beta):
    clip_train = np.load(paths["clip_train"], mmap_mode="r")
    dino_train = np.load(paths["dino_train"], mmap_mode="r")
    clip_held = np.load(paths["clip_held"], mmap_mode="r")
    dino_held = np.load(paths["dino_held"], mmap_mode="r")

    clip_rank = rank_calibrator(np.asarray(clip_train, dtype=np.float32).reshape(-1))
    dino_rank = rank_calibrator(np.asarray(dino_train, dtype=np.float32).reshape(-1))
    n = clip_held.shape[0]

    fused_path = MAP_CACHE / f"tmp_fused_{time.time_ns()}.npy"
    shuf_path = MAP_CACHE / f"tmp_shuf_{time.time_ns()}.npy"
    fused = np.lib.format.open_memmap(fused_path, mode="w+", dtype=np.float16, shape=(n, SIDE, SIDE))
    shuf = np.lib.format.open_memmap(shuf_path, mode="w+", dtype=np.float16, shape=(n, SIDE, SIDE))

    rng = np.random.default_rng(0)
    _ = rng.permutation(n)
    shuf_idx = rng.permutation(n)

    for i in range(n):
        c = apply_rank_array(clip_held[i], clip_rank)
        d = apply_rank_array(dino_held[i], dino_rank)
        sd = apply_rank_array(dino_held[int(shuf_idx[i])], dino_rank)
        fused[i] = (beta * c + (1.0 - beta) * d).astype(np.float16)
        shuf[i] = (beta * c + (1.0 - beta) * sd).astype(np.float16)
        if i % 128 == 0:
            fused.flush()
            shuf.flush()
    fused.flush()
    shuf.flush()
    return {
        "C0_CLIP_LOCAL": clip_held,
        "D0_DINO_LOCAL": dino_held,
        "F0_CLIP_DINO_FUSION": np.load(fused_path, mmap_mode="r"),
        "S1_CROSS_IMAGE_SHUFFLED_DINO": np.load(shuf_path, mmap_mode="r"),
    }, [fused_path, shuf_path]


def complementarity_and_rank(dataset, held_indices, clip_maps, dino_maps, fused_maps):
    rows_comp, rows_rank = [], []
    for pos, idx in enumerate(held_indices):
        item = dataset[idx]
        mask = (item["img_mask"].detach().cpu().numpy() > 0.5).astype(bool).reshape(-1)
        c = upsample_smooth(np.asarray(clip_maps[pos], dtype=np.float32)).reshape(-1)
        d = upsample_smooth(np.asarray(dino_maps[pos], dtype=np.float32)).reshape(-1)
        f = upsample_smooth(np.asarray(fused_maps[pos], dtype=np.float32)).reshape(-1)
        c_thr = np.percentile(c, 95)
        d_thr = np.percentile(d, 95)
        c_pos = c >= c_thr
        d_pos = d >= d_thr
        fn_c = mask & ~c_pos
        fn_d = mask & ~d_pos
        fp_c = (~mask) & c_pos
        fp_d = (~mask) & d_pos
        oracle = np.where(mask, np.maximum(c, d), np.minimum(c, d))
        y = mask.astype(np.uint8)
        rows_comp.append({
            "idx": idx,
            "category": item["cls_name"],
            "label": int(item["anomaly"]),
            "score_spearman": float(spearmanr(c, d).statistic),
            "error_disagreement": float(np.mean(c_pos != d_pos)),
            "false_negative_overlap": float((fn_c & fn_d).sum() / max(1, (fn_c | fn_d).sum())),
            "false_positive_overlap": float((fp_c & fp_d).sum() / max(1, (fp_c | fp_d).sum())),
            "oracle_Pixel_AUROC": safe_auc(y, oracle),
            "oracle_Pixel_AP": safe_ap(y, oracle),
        })
        n_pairs = min(20000, int(mask.sum()) * int((~mask).sum()))
        if n_pairs > 0 and np.unique(y).size == 2:
            rng = np.random.default_rng(pos)
            pidx = np.flatnonzero(mask)
            nidx = np.flatnonzero(~mask)
            pp = rng.choice(pidx, size=n_pairs, replace=True)
            nn = rng.choice(nidx, size=n_pairs, replace=True)
            c_ord = c[pp] > c[nn]
            f_ord = f[pp] > f[nn]
            pair_change = float(np.mean(c_ord != f_ord))
        else:
            pair_change = float("nan")
        sp = float(spearmanr(c, f).statistic)
        near_identity = bool(sp > 0.999 and np.nanstd(f - c) < 1e-6)
        rows_rank.append({
            "idx": idx,
            "category": item["cls_name"],
            "label": int(item["anomaly"]),
            "clip_fusion_spearman": sp,
            "positive_negative_pair_order_change_fraction": pair_change,
            "RANK_NEAR_IDENTITY": near_identity,
        })
    return rows_comp, rows_rank


def direction_gate(grows, controls):
    by = {(r["direction"], int(r["fold"]), r["variant"]): r for r in grows}
    out = {}
    for d in ["D1", "D2"]:
        dino = [by[(d, f, "D0_DINO_LOCAL")] for f in [0, 1]]
        fus = [by[(d, f, "F0_CLIP_DINO_FUSION")] for f in [0, 1]]
        dino_mean = {k: float(np.mean([r[k] for r in dino])) for k in ["dPixel_AUROC", "dPixel_AP", "G_rank"]}
        fus_mean = {k: float(np.mean([r[k] for r in fus])) for k in ["dPixel_AUROC", "dPixel_AP", "G_rank"]}
        dino_valid = (
            dino_mean["G_rank"] >= 0.30
            and (
                (dino_mean["dPixel_AUROC"] >= 0.30 and dino_mean["dPixel_AP"] >= -0.15)
                or (dino_mean["dPixel_AP"] >= 0.40 and dino_mean["dPixel_AUROC"] >= -0.15)
            )
        )
        fus_valid = (
            fus_mean["G_rank"] >= 0.40
            and fus_mean["dPixel_AUROC"] >= 0.0
            and fus_mean["dPixel_AP"] >= 0.0
            and min(r["G_rank"] for r in fus) >= -0.20
        )
        c = next(r for r in controls if r["direction"] == d)
        specific = True
        if fus_valid:
            specific = c["F0_minus_S1_mean_G_rank"] >= 0.20
        out[d] = {
            "dino_mean": dino_mean,
            "fusion_mean": fus_mean,
            "dino_valid": bool(dino_valid),
            "fusion_valid": bool(fus_valid),
            "specific": bool(specific),
            "valid_contribution": bool((dino_valid or fus_valid) and specific),
        }
    return out


def decide(gates):
    valid = [gates[d]["valid_contribution"] for d in ["D1", "D2"]]
    any_positive = any(
        gates[d]["dino_mean"]["G_rank"] > 0 or gates[d]["fusion_mean"]["G_rank"] > 0
        for d in ["D1", "D2"]
    )
    if all(valid):
        return "PASS_TO_PHASE_XXX_B"
    if valid[0] != valid[1]:
        return "XXX_A_DINO_DIRECTION_SPECIFIC"
    if any((gates[d]["dino_valid"] or gates[d]["fusion_valid"]) and not gates[d]["specific"] for d in ["D1", "D2"]):
        return "XXX_A_DINO_COMPLEMENTARITY_NOT_SPECIFIC"
    if any_positive:
        return "XXX_A_SMALL_METRIC_SAFE_DINO_SIGNAL"
    return "XXX_A_DINO_SIGNAL_WAS_PRO_DRIVEN"


def write_report(outcome, gates, pixel_rows, g_rows, control_rows, percat_rows, comp_rows, rank_rows):
    lines = [
        "# Phase XXX-A Metric-Safe DINO Signal Reconciliation",
        "",
        f"Outcome: **{outcome}**",
        "",
        "Scope: source-held-out D1/D2 fold0/fold1 only. No training, no target data, no new CLIP/DINO extraction, and no PRO-based decision.",
        "",
        "## Primary Pixel AUROC/AP",
        "",
        "| Direction | Fold | Variant | Pixel AUROC | Pixel AP | dAUROC | dAP | G_rank |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in g_rows:
        lines.append(f"| {r['direction']} | {r['fold']} | {r['variant']} | {r['Pixel_AUROC']:.4f} | {r['Pixel_AP']:.4f} | {r['dPixel_AUROC']:+.4f} | {r['dPixel_AP']:+.4f} | {r['G_rank']:+.4f} |")
    lines.extend(["", "## Direction Gates", ""])
    for d, g in gates.items():
        lines.append(
            f"- {d}: DINO mean G={g['dino_mean']['G_rank']:+.4f}, fusion mean G={g['fusion_mean']['G_rank']:+.4f}, "
            f"DINO valid={g['dino_valid']}, fusion valid={g['fusion_valid']}, specific={g['specific']}."
        )
    lines.extend(["", "## Controls", "", "| Direction | F0-S1 mean G_rank | S2 available |", "|---|---:|---|"])
    for r in control_rows:
        lines.append(f"| {r['direction']} | {r['F0_minus_S1_mean_G_rank']:+.4f} | {r['S2_available']} |")
    fus = [r for r in percat_rows if r["variant"] == "F0_CLIP_DINO_FUSION"]
    wins = sum(1 for r in fus if r["WTL_vs_C0"] == "W")
    losses = sum(1 for r in fus if r["WTL_vs_C0"] == "L")
    ties = sum(1 for r in fus if r["WTL_vs_C0"] == "T")
    lines.extend([
        "",
        "## Summary Answers",
        "",
        f"- Q1/Q2: DINO-only metric-safe signal is direction-gated in the table above; it is judged only by Pixel AUROC/AP.",
        f"- Q3/Q4: frozen fusion is compared against CLIP and shuffled DINO controls; W/T/L for fusion categories is {wins}/{ties}/{losses}.",
        f"- Q5: cross-direction continuation is {'satisfied' if outcome == 'PASS_TO_PHASE_XXX_B' else 'not satisfied'}.",
        "",
        "Mandatory stop: Phase XXX-B was not started.",
    ])
    (OUT / "XXX_A_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    MAP_CACHE.mkdir(parents=True, exist_ok=True)
    start = time.time()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    beta_by_fold = load_selected_beta()

    audit_rows, pixel_rows, g_rows, percat_rows = [], [], [], []
    comp_rows, rank_rows = [], []
    temp_paths = []

    for direction in ["D1", "D2"]:
        for fold in [0, 1]:
            dataset_name, dataset, rotation, train_indices, held_indices = load_fold(direction, fold)
            beta = beta_by_fold[(direction, fold)]
            paths = ensure_distance_maps(direction, fold, dataset, train_indices, held_indices, device)
            variants, temps = build_fold_variants(paths, beta)
            temp_paths.extend(temps)
            audit_rows.append({
                "direction": direction,
                "fold": fold,
                "dataset": dataset_name,
                "train_categories": rotation["train_categories"],
                "heldout_categories": rotation["val_categories"],
                "train_images": len(train_indices),
                "heldout_images": len(held_indices),
                "selected_beta_clip_from_phase_xxiii_a": beta,
                "feature_cache_reused": True,
                "distance_map_cache_paths": {k: str(v) for k, v in paths.items()},
                "S2_SPATIALLY_SHUFFLED_DINO_available": False,
            })
            fold_metrics = []
            fold_percat = []
            for name, maps in variants.items():
                row, pc = evaluate_variant(dataset, held_indices, maps, name)
                row.update({"direction": direction, "fold": fold})
                fold_metrics.append(row)
                for p in pc:
                    p.update({"direction": direction, "fold": fold})
                    fold_percat.append(p)
            base = next(r for r in fold_metrics if r["variant"] == "C0_CLIP_LOCAL")
            for row in fold_metrics:
                grow = dict(row)
                grow["dPixel_AUROC"] = float(row["Pixel_AUROC"] - base["Pixel_AUROC"])
                grow["dPixel_AP"] = float(row["Pixel_AP"] - base["Pixel_AP"])
                grow["G_rank"] = 0.5 * grow["dPixel_AUROC"] + 0.5 * grow["dPixel_AP"]
                g_rows.append(grow)
            pc_base = {(r["category"], r["variant"]): r for r in fold_percat}
            gains = []
            for row in fold_percat:
                b = pc_base[(row["category"], "C0_CLIP_LOCAL")]
                row["dPixel_AUROC_vs_C0"] = float(row["Pixel_AUROC"] - b["Pixel_AUROC"])
                row["dPixel_AP_vs_C0"] = float(row["Pixel_AP"] - b["Pixel_AP"])
                row["G_rank_vs_C0"] = 0.5 * row["dPixel_AUROC_vs_C0"] + 0.5 * row["dPixel_AP_vs_C0"]
                row["WTL_vs_C0"] = "T" if abs(row["G_rank_vs_C0"]) <= 0.05 else ("W" if row["G_rank_vs_C0"] > 0 else "L")
                if row["variant"] == "F0_CLIP_DINO_FUSION" and np.isfinite(row["G_rank_vs_C0"]) and row["G_rank_vs_C0"] > 0:
                    gains.append(row["G_rank_vs_C0"])
            concentrated = bool(gains and max(gains) / max(EPS, sum(gains)) > 0.5)
            for row in fold_percat:
                row["CATEGORY_CONCENTRATED"] = concentrated
            pixel_rows.extend(fold_metrics)
            percat_rows.extend(fold_percat)
            crows, rrows = complementarity_and_rank(dataset, held_indices, variants["C0_CLIP_LOCAL"], variants["D0_DINO_LOCAL"], variants["F0_CLIP_DINO_FUSION"])
            for r in crows:
                r.update({"direction": direction, "fold": fold})
            for r in rrows:
                r.update({"direction": direction, "fold": fold})
            comp_rows.extend(crows)
            rank_rows.extend(rrows)
            write_csv(OUT / "PIXEL_AUROC_AP.running.csv", pixel_rows)
            write_csv(OUT / "G_RANK.running.csv", g_rows)

    control_rows = []
    for direction in ["D1", "D2"]:
        vals = []
        for fold in [0, 1]:
            f0 = next(r for r in g_rows if r["direction"] == direction and int(r["fold"]) == fold and r["variant"] == "F0_CLIP_DINO_FUSION")
            s1 = next(r for r in g_rows if r["direction"] == direction and int(r["fold"]) == fold and r["variant"] == "S1_CROSS_IMAGE_SHUFFLED_DINO")
            vals.append(f0["G_rank"] - s1["G_rank"])
        control_rows.append({
            "direction": direction,
            "F0_minus_S1_mean_G_rank": float(np.mean(vals)),
            "F0_minus_S1_fold0_G_rank": float(vals[0]),
            "F0_minus_S1_fold1_G_rank": float(vals[1]),
            "S2_available": False,
            "S2_note": "Phase XXIII-A did not create a spatial-shuffle DINO map; no new S2 was designed in XXX-A.",
        })

    gates = direction_gate(g_rows, control_rows)
    outcome = decide(gates)
    runtime = time.time() - start

    audit = {
        "phase": "XXX-A",
        "source_phase": "XXIII-A",
        "variants": VARIANTS,
        "metric_evaluator": "sklearn roc_auc_score and average_precision_score over pooled full-resolution source-held-out pixels; per-category rows also reported",
        "map_preprocessing": f"37x37 map bilinear upsampled to {IMAGE_SIZE}x{IMAGE_SIZE}; Gaussian sigma={SIGMA}, identical for all available variants",
        "pro_used_for_decision": False,
        "target_accessed": False,
        "trained": False,
        "runtime_sec": runtime,
        "folds": audit_rows,
    }
    (OUT / "MAP_RECONSTRUCTION_AUDIT.json").write_text(json.dumps(audit, indent=2, default=json_default), encoding="utf-8")
    write_csv(OUT / "PIXEL_AUROC_AP.csv", pixel_rows)
    write_csv(OUT / "G_RANK.csv", g_rows)
    write_csv(OUT / "CONTROL_SPECIFICITY.csv", control_rows)
    write_csv(OUT / "PER_CATEGORY.csv", percat_rows)
    write_csv(OUT / "ERROR_COMPLEMENTARITY.csv", comp_rows)
    write_csv(OUT / "RANK_CHANGE.csv", rank_rows)
    write_report(outcome, gates, pixel_rows, g_rows, control_rows, percat_rows, comp_rows, rank_rows)
    payload = {
        "phase": "XXX-A",
        "outcome": outcome,
        "target_accessed": False,
        "trained": False,
        "pro_used_for_decision": False,
        "phase_xxx_b_started": False,
        "runtime_sec": runtime,
        "direction_gates": gates,
    }
    (OUT / "OUTCOME.json").write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    with open(ROOT / "EXPERIMENT_LOG.md", "a", encoding="utf-8") as fp:
        fp.write("\n\n## Phase XXX-A\n")
        fp.write("- Command: python experiments/phase_xxx_metric_safe_dino.py\n")
        fp.write("- Scope: D1 fold0/fold1 and D2 fold0/fold1 source-held-out Phase XXIII-A CLIP/DINO local maps; no target; no training.\n")
        fp.write(f"- Outcome: {outcome}.\n")
        for d, g in gates.items():
            fp.write(f"- {d}: DINO mean G_rank {g['dino_mean']['G_rank']:+.4f}, fusion mean G_rank {g['fusion_mean']['G_rank']:+.4f}, valid {g['valid_contribution']}.\n")
    for p in temp_paths:
        try:
            p.unlink()
        except OSError:
            pass
    print(json.dumps({"outcome": outcome, "report": str(OUT / "XXX_A_REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
