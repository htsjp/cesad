import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, EPOCHS, LR, RANK, SEED, WEIGHT_DECAY, load_train_samples
from experiments.phase_xliii_adapter_head_decomposition import gp
from experiments.phase_xliii_anchor_specificity import (
    geometry_matched_random,
    low_rank_rows,
    score_and_diagnose,
    train_adapter,
    train_margin_drift_rows,
)
from experiments.phase_xxxviii_visual_state_representation import FEATURE_CACHE, labels_and_rows, load_fold, norm_np, patch_labels
from experiments.phase_xxxix_local_multistate import evaluate_variant


OUT = Path("results/phase_xliv_state_alignment_confirmation")
XXXVIII_OUT = Path("results/phase_xxxviii_visual_state_representation")
XXXIX_OUT = Path("results/phase_xxxix_local_multistate")
XLIII_B_OUT = Path("results/phase_xliii_anchor_specificity")
FOLDS = [("D1", 0), ("D1", 1), ("D2", 0), ("D2", 1)]
FOLD1 = [("D1", 1), ("D2", 1)]
ANCHOR_VARIANTS = {
    "A0_LEARNED": "K2_patch",
    "A1_CENTROID": "K1_patch",
    "A2_SHUFFLED": "K3_patch",
    "A3_RANDOM_ANGLE": "RANDOM",
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
    raise TypeError(type(obj).__name__)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=json_default), encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def to_float_row(row):
    out = {}
    for k, v in row.items():
        if v is None:
            out[k] = v
            continue
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = v
    return out


def variant_map(variant):
    mapping = {
        "S0_LEARNED_STATE_ANCHORS": "A0_LEARNED",
        "S1_CENTROID_ANCHORS": "A1_CENTROID",
        "S2_SHUFFLED_LABEL_STATE_ANCHORS": "A2_SHUFFLED",
        "S3_GEOMETRY_MATCHED_RANDOM_ANCHORS": "A3_RANDOM_ANGLE",
    }
    return mapping.get(variant, variant)


def historical_c0_c2():
    rows = read_csv(XXXVIII_OUT / "RAW_METRICS.csv")
    out = []
    for r in rows:
        if r["variant"] == "K2_LEARNED_VISUAL_STATE":
            rr = to_float_row(r)
            rr["variant"] = "C0_SINGLE_STATE"
            rr["G_pixel"] = gp(rr)
            out.append(rr)
        elif r["variant"] == "K0_INSTANCE_MEMORY":
            rr = to_float_row(r)
            rr["variant"] = "C2_INSTANCE_MEMORY"
            rr["G_pixel"] = gp(rr)
            out.append(rr)
    return out


def historical_c0_c2_per_category():
    rows = read_csv(XXXIX_OUT / "PER_CATEGORY.csv")
    out = []
    for r in rows:
        if r["variant"] == "P1_SINGLE_STATE":
            rr = to_float_row(r)
            rr["variant"] = "C0_SINGLE_STATE"
            rr["G_pixel"] = gp(rr)
            out.append(rr)
        elif r["variant"] == "P0_INSTANCE_MEMORY":
            rr = to_float_row(r)
            rr["variant"] = "C2_INSTANCE_MEMORY"
            rr["G_pixel"] = gp(rr)
            out.append(rr)
    return out


def historical_fold0_c1():
    rows = []
    for r in read_csv(XLIII_B_OUT / "RAW_METRICS.csv"):
        if r["variant"] == "S0_LEARNED_STATE_ANCHORS":
            rr = to_float_row(r)
            rr["variant"] = "C1_STATE_ALIGNED"
            rr["G_pixel"] = gp(rr)
            rows.append(rr)
    return rows


def historical_fold0_anchor_metrics():
    rows = []
    for r in read_csv(XLIII_B_OUT / "RAW_METRICS.csv"):
        rr = to_float_row(r)
        rr["variant"] = variant_map(rr["variant"])
        rr["G_pixel"] = gp(rr)
        rows.append(rr)
    return rows


def historical_fold0_anchor_per_category():
    rows = []
    for r in read_csv(XLIII_B_OUT / "PER_CATEGORY.csv"):
        rr = to_float_row(r)
        rr["variant"] = variant_map(rr["variant"])
        rr["G_pixel"] = gp(rr)
        rows.append(rr)
    return rows


def collect_held_patch_labels(dataset, held_indices):
    labels = []
    for idx in held_indices:
        labels.append(patch_labels(dataset[idx]))
    return np.concatenate(labels).astype(np.int64)


def train_score_anchor(direction, fold, anchor_variant, device):
    start = time.time()
    dataset_name, dataset, rotation, train_indices, held_indices = load_fold(direction, fold)
    cache = np.load(FEATURE_CACHE / f"{direction}_fold{fold}_clip.npz", allow_pickle=True)
    train_patch = cache["train_patch"]
    held_patch = cache["held_patch"]
    states38 = np.load(XXXVIII_OUT / "states" / f"{direction}_fold{fold}_states.npz")
    random_states, random_check = geometry_matched_random(states38["K2_patch"])
    anchors = {
        "A0_LEARNED": states38["K2_patch"],
        "A1_CENTROID": states38["K1_patch"],
        "A2_SHUFFLED": states38["K3_patch"],
        "A3_RANDOM_ANGLE": random_states,
    }
    states = anchors[anchor_variant]
    x_train, y_train, _cats = load_train_samples(dataset, train_indices, train_patch)
    held_patch_y = collect_held_patch_labels(dataset, held_indices)
    zeros = np.zeros((len(held_indices),), dtype=np.float32)

    print(f"[XLIV-A] {direction} fold{fold} {anchor_variant} train", flush=True)
    model, curves = train_adapter(x_train, y_train, states, device, anchor_variant)
    for row in curves:
        row.update({"direction": direction, "fold": fold})
    train_m, train_d = train_margin_drift_rows(x_train, y_train, model, states, device, direction, fold, anchor_variant)
    print(f"[XLIV-A] {direction} fold{fold} {anchor_variant} score", flush=True)
    maps, held_m, held_d = score_and_diagnose(held_patch, held_patch_y, model, states, device, direction, fold, anchor_variant)
    res = evaluate_variant(dataset, held_indices, zeros, maps)
    metric = {"direction": direction, "fold": fold, "dataset": dataset_name, "variant": anchor_variant, **res["mean"]}
    metric["G_pixel"] = gp(metric)
    percat = []
    for cat, vals in res["per_category"].items():
        row = {"direction": direction, "fold": fold, "category": cat, "variant": anchor_variant, **vals}
        row["G_pixel"] = gp(row)
        percat.append(row)
    low_rank = low_rank_rows({anchor_variant: model}, direction, fold)
    protocol = {
        "direction": direction,
        "fold": fold,
        "dataset": dataset_name,
        "variant": anchor_variant,
        "train_categories": rotation["train_categories"],
        "heldout_categories": rotation["val_categories"],
        "source": "cached frozen patch features",
        "backbone_forward": False,
        "rank": RANK,
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "train_samples": int(x_train.shape[0]),
        "held_images": int(len(held_indices)),
        "random_angle_check": random_check if anchor_variant == "A3_RANDOM_ANGLE" else None,
        "runtime_sec": time.time() - start,
    }
    return {
        "protocol": protocol,
        "metric": metric,
        "per_category": percat,
        "curves": curves,
        "margins": train_m + held_m,
        "drift": train_d + held_d,
        "low_rank": low_rank,
    }


def ensure_fold1_c1(device):
    ckpt = OUT / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    payloads = {}
    for direction, fold in FOLD1:
        path = ckpt / f"{direction}_fold{fold}_A0_LEARNED.json"
        if path.exists():
            print(f"[XLIV-A] resume {direction} fold{fold} A0_LEARNED", flush=True)
            payloads[(direction, fold, "A0_LEARNED")] = read_json(path)
            continue
        payload = train_score_anchor(direction, fold, "A0_LEARNED", device)
        write_json(path, payload)
        payloads[(direction, fold, "A0_LEARNED")] = payload
    return payloads


def c1_fold1_confirmation_rows(metrics):
    by = {(r["direction"], int(r["fold"]), r["variant"]): r for r in metrics}
    rows = []
    for direction, fold in FOLD1:
        c0 = by[(direction, fold, "C0_SINGLE_STATE")]
        c1 = by[(direction, fold, "C1_STATE_ALIGNED")]
        rows.append({
            "direction": direction,
            "fold": fold,
            "C0_P-AUROC": c0["P-AUROC"],
            "C1_P-AUROC": c1["P-AUROC"],
            "delta_P-AUROC_pp": c1["P-AUROC"] - c0["P-AUROC"],
            "C0_P-AP": c0["P-AP"],
            "C1_P-AP": c1["P-AP"],
            "delta_P-AP_pp": c1["P-AP"] - c0["P-AP"],
            "C0_G_pixel": gp(c0),
            "C1_G_pixel": gp(c1),
            "delta_G_pixel_pp": gp(c1) - gp(c0),
            "strong_signal": bool(gp(c1) - gp(c0) >= 3.0 and c1["P-AUROC"] > c0["P-AUROC"] and c1["P-AP"] > c0["P-AP"]),
            "hard_failure": bool(gp(c1) - gp(c0) < 1.0),
        })
    return rows


def ensure_fold1_controls(device):
    ckpt = OUT / "checkpoints"
    payloads = {}
    for direction, fold in FOLD1:
        for variant in ["A1_CENTROID", "A2_SHUFFLED", "A3_RANDOM_ANGLE"]:
            path = ckpt / f"{direction}_fold{fold}_{variant}.json"
            if path.exists():
                print(f"[XLIV-A] resume {direction} fold{fold} {variant}", flush=True)
                payloads[(direction, fold, variant)] = read_json(path)
                continue
            payload = train_score_anchor(direction, fold, variant, device)
            write_json(path, payload)
            payloads[(direction, fold, variant)] = payload
    return payloads


def direction_means(rows, variants):
    out = []
    for direction in ["D1", "D2"]:
        for variant in variants:
            vals = [r for r in rows if r["direction"] == direction and r["variant"] == variant]
            if not vals:
                continue
            row = {"direction": direction, "variant": variant, "folds": len(vals)}
            for key in ["P-AUROC", "P-AP", "PRO", "G_pixel"]:
                row[key] = float(np.mean([float(v[key]) for v in vals]))
            out.append(row)
    return out


def state_alignment_gain(metrics):
    by = {(r["direction"], int(r["fold"]), r["variant"]): r for r in metrics}
    rows = []
    for direction, fold in FOLDS:
        c0 = by[(direction, fold, "C0_SINGLE_STATE")]
        c1 = by[(direction, fold, "C1_STATE_ALIGNED")]
        rows.append({
            "direction": direction,
            "fold": fold,
            "C0_G_pixel": gp(c0),
            "C1_G_pixel": gp(c1),
            "delta_G_pixel_pp": gp(c1) - gp(c0),
            "delta_P-AUROC_pp": c1["P-AUROC"] - c0["P-AUROC"],
            "delta_P-AP_pp": c1["P-AP"] - c0["P-AP"],
            "positive": bool(gp(c1) - gp(c0) > 0),
        })
    return rows


def anchor_specificity(anchor_metrics):
    by = {(r["direction"], int(r["fold"]), r["variant"]): r for r in anchor_metrics}
    rows = []
    for direction, fold in FOLDS:
        a0 = by[(direction, fold, "A0_LEARNED")]
        a1 = by[(direction, fold, "A1_CENTROID")]
        a2 = by[(direction, fold, "A2_SHUFFLED")]
        a3 = by[(direction, fold, "A3_RANDOM_ANGLE")]
        rows.append({
            "direction": direction,
            "fold": fold,
            "A0_G_pixel": gp(a0),
            "A1_G_pixel": gp(a1),
            "A2_G_pixel": gp(a2),
            "A3_G_pixel": gp(a3),
            "A0_minus_A1_pp": gp(a0) - gp(a1),
            "A0_minus_A2_pp": gp(a0) - gp(a2),
            "A0_minus_A3_pp": gp(a0) - gp(a3),
            "learned_gt_shuffled_1pp": bool(gp(a0) - gp(a2) >= 1.0),
            "learned_gt_random_1pp": bool(gp(a0) - gp(a3) >= 1.0),
        })
    return rows


def instance_memory_comparison(metrics):
    by = {(r["direction"], int(r["fold"]), r["variant"]): r for r in metrics}
    rows = []
    for direction, fold in FOLDS:
        c1 = by[(direction, fold, "C1_STATE_ALIGNED")]
        c2 = by[(direction, fold, "C2_INSTANCE_MEMORY")]
        rows.append({
            "direction": direction,
            "fold": fold,
            "C1_G_pixel": gp(c1),
            "C2_G_pixel": gp(c2),
            "delta_G_pixel_pp": gp(c1) - gp(c2),
            "delta_P-AUROC_pp": c1["P-AUROC"] - c2["P-AUROC"],
            "delta_P-AP_pp": c1["P-AP"] - c2["P-AP"],
        })
    return rows


def gap_closure(metrics):
    by = {(r["direction"], int(r["fold"]), r["variant"]): r for r in metrics}
    rows = []
    for direction in ["D1", "D2"]:
        folds = [f for d, f in FOLDS if d == direction]
        c0 = np.mean([gp(by[(direction, f, "C0_SINGLE_STATE")]) for f in folds])
        c1 = np.mean([gp(by[(direction, f, "C1_STATE_ALIGNED")]) for f in folds])
        c2 = np.mean([gp(by[(direction, f, "C2_INSTANCE_MEMORY")]) for f in folds])
        gap = c2 - c0
        rec = c1 - c0
        rows.append({
            "direction": direction,
            "C0_mean_G_pixel": float(c0),
            "C1_mean_G_pixel": float(c1),
            "C2_mean_G_pixel": float(c2),
            "gap_C2_minus_C0_pp": float(gap),
            "recovery_C1_minus_C0_pp": float(rec),
            "gap_closure": float(rec / gap) if abs(gap) > 1e-12 else float("nan"),
        })
    return rows


def category_stability(per_category):
    by = {(r["direction"], int(r["fold"]), r["category"], r["variant"]): r for r in per_category}
    rows = []
    for direction, fold in FOLDS:
        cats = sorted({r["category"] for r in per_category if r["direction"] == direction and int(r["fold"]) == fold})
        gains = []
        for cat in cats:
            c0 = by.get((direction, fold, cat, "C0_SINGLE_STATE"))
            c1 = by.get((direction, fold, cat, "C1_STATE_ALIGNED"))
            c2 = by.get((direction, fold, cat, "C2_INSTANCE_MEMORY"))
            if not c0 or not c1 or not c2:
                continue
            g10 = gp(c1) - gp(c0)
            g12 = gp(c1) - gp(c2)
            gains.append((cat, g10))
            rows.append({
                "direction": direction,
                "fold": fold,
                "category": cat,
                "C0_G_pixel": gp(c0),
                "C1_G_pixel": gp(c1),
                "C2_G_pixel": gp(c2),
                "C1_minus_C0_pp": g10,
                "C1_minus_C2_pp": g12,
                "C1_vs_C0": "tie" if abs(g10) <= 0.05 else ("win" if g10 > 0 else "loss"),
                "C1_vs_C2": "tie" if abs(g12) <= 0.05 else ("win" if g12 > 0 else "loss"),
            })
        pos_sum = sum(max(g, 0.0) for _cat, g in gains)
        for row in rows:
            if row["direction"] == direction and row["fold"] == fold:
                row["positive_gain_share_of_C1_minus_C0"] = float(max(row["C1_minus_C0_pp"], 0.0) / pos_sum) if pos_sum > 1e-12 else float("nan")
    return rows


def category_concentration_rows(category_rows):
    out = []
    for direction, fold in FOLDS:
        rows = [r for r in category_rows if r["direction"] == direction and int(r["fold"]) == fold]
        shares = [r.get("positive_gain_share_of_C1_minus_C0", float("nan")) for r in rows]
        max_share = max([s for s in shares if np.isfinite(s)], default=float("nan"))
        max_cat = ""
        for r in rows:
            if np.isfinite(r.get("positive_gain_share_of_C1_minus_C0", float("nan"))) and abs(r["positive_gain_share_of_C1_minus_C0"] - max_share) < 1e-12:
                max_cat = r["category"]
                break
        out.append({
            "direction": direction,
            "fold": fold,
            "max_positive_gain_category": max_cat,
            "max_positive_gain_share": max_share,
            "category_concentrated": bool(np.isfinite(max_share) and max_share > 0.5),
            "C1_vs_C0_wins": sum(1 for r in rows if r["C1_vs_C0"] == "win"),
            "C1_vs_C0_ties": sum(1 for r in rows if r["C1_vs_C0"] == "tie"),
            "C1_vs_C0_losses": sum(1 for r in rows if r["C1_vs_C0"] == "loss"),
            "C1_vs_C2_wins": sum(1 for r in rows if r["C1_vs_C2"] == "win"),
            "C1_vs_C2_ties": sum(1 for r in rows if r["C1_vs_C2"] == "tie"),
            "C1_vs_C2_losses": sum(1 for r in rows if r["C1_vs_C2"] == "loss"),
        })
    return out


def storage_rows():
    rows = []
    old = read_csv(XXXVIII_OUT / "STORAGE_COMPLEXITY.csv")
    for r in old:
        if r["variant"] == "K0_INSTANCE_MEMORY":
            rr = to_float_row(r)
            rr["variant"] = "C2_INSTANCE_MEMORY"
            rows.append(rr)
        elif r["variant"] == "K2_LEARNED_VISUAL_STATE":
            rr = to_float_row(r)
            rr["variant"] = "C0_SINGLE_STATE"
            rows.append(rr)
            rows.append({
                "direction": rr["direction"],
                "fold": int(rr["fold"]),
                "variant": "C1_STATE_ALIGNED",
                "image_vectors": rr.get("image_vectors", 2),
                "patch_vectors": 2,
                "bytes_float32": int(2 * 1024 * 4 + 8192 * 4),
                "image_comparisons_per_query": rr.get("image_comparisons_per_query", 2),
                "patch_comparisons_per_query": 2,
                "trainable_adapter_parameters": 8192,
                "frozen_anchor_vectors": 2,
            })
    return rows


def aggregate(start, c1_payloads, control_payloads):
    c0_c2 = historical_c0_c2()
    c1_fold0 = historical_fold0_c1()
    c1_fold1 = []
    per_category = historical_c0_c2_per_category()
    per_category.extend([dict(r, variant="C1_STATE_ALIGNED") for r in historical_fold0_anchor_per_category() if r["variant"] == "A0_LEARNED"])
    protocols = []
    curves, margins, drift, adapter_geom = [], [], [], []
    for payload in c1_payloads.values():
        metric = dict(payload["metric"])
        metric["variant"] = "C1_STATE_ALIGNED"
        c1_fold1.append(metric)
        per_category.extend([dict(r, variant="C1_STATE_ALIGNED") for r in payload["per_category"]])
        protocols.append(payload["protocol"])
        curves.extend(payload["curves"])
        margins.extend(payload["margins"])
        drift.extend(payload["drift"])
        adapter_geom.extend(payload["low_rank"])
    metrics = c0_c2 + c1_fold0 + c1_fold1
    fold1_rows = c1_fold1_confirmation_rows(metrics)
    fold1_pass = all(r["strong_signal"] for r in fold1_rows)
    fold1_hard_fail = any(r["hard_failure"] for r in fold1_rows)
    weak_fold_stability = (not fold1_pass) and (not fold1_hard_fail)
    write_csv(OUT / "FOLD1_CONFIRMATION.csv", fold1_rows)
    write_json(OUT / "FOLD1_GATE.json", {
        "fold1_pass": fold1_pass,
        "fold1_hard_fail": fold1_hard_fail,
        "weak_fold_stability": weak_fold_stability,
        "rows": fold1_rows,
    })

    anchor_metrics = historical_fold0_anchor_metrics()
    anchor_percat = historical_fold0_anchor_per_category()
    if fold1_hard_fail:
        controls_executed = False
    else:
        controls_executed = True
        for payload in c1_payloads.values():
            anchor_metrics.append(dict(payload["metric"]))
            anchor_percat.extend(payload["per_category"])
        for payload in control_payloads.values():
            anchor_metrics.append(payload["metric"])
            anchor_percat.extend(payload["per_category"])
            protocols.append(payload["protocol"])
            curves.extend(payload["curves"])
            margins.extend(payload["margins"])
            drift.extend(payload["drift"])
            adapter_geom.extend(payload["low_rank"])

    checkpointed_runtime_sec = float(sum(float(p.get("runtime_sec", 0.0)) for p in protocols))

    for r in read_csv(XLIII_B_OUT / "LOW_RANK_UPDATE_GEOMETRY.csv"):
        rr = to_float_row(r)
        rr["variant"] = variant_map(str(rr["variant"]))
        adapter_geom.append(rr)
    for r in read_csv(XLIII_B_OUT / "FEATURE_DRIFT.csv"):
        rr = to_float_row(r)
        rr["variant"] = variant_map(str(rr["variant"]))
        drift.append(rr)

    if controls_executed:
        anchor_rows = anchor_specificity(anchor_metrics)
    else:
        anchor_rows = []
    align_rows = state_alignment_gain(metrics)
    im_rows = instance_memory_comparison(metrics)
    gap_rows = gap_closure(metrics)
    category_rows = category_stability(per_category)
    concentration = category_concentration_rows(category_rows)
    grouped = direction_means(metrics, ["C0_SINGLE_STATE", "C1_STATE_ALIGNED", "C2_INSTANCE_MEMORY"])

    performance_gate = (
        all(r["positive"] for r in align_rows)
        and all(r["delta_G_pixel_pp"] >= 1.0 for r in align_rows)
        and all(next(g for g in grouped if g["direction"] == d and g["variant"] == "C1_STATE_ALIGNED")["G_pixel"] - next(g for g in grouped if g["direction"] == d and g["variant"] == "C0_SINGLE_STATE")["G_pixel"] >= 3.0 for d in ["D1", "D2"])
    )
    if controls_executed:
        shuf_ok = sum(1 for r in anchor_rows if r["learned_gt_shuffled_1pp"]) >= 3
        rand_ok = sum(1 for r in anchor_rows if r["learned_gt_random_1pp"]) >= 3
        dir_pos = True
        for d in ["D1", "D2"]:
            vals = [r for r in anchor_rows if r["direction"] == d]
            dir_pos = dir_pos and np.mean([r["A0_minus_A2_pp"] for r in vals]) > 0 and np.mean([r["A0_minus_A3_pp"] for r in vals]) > 0
        anchor_gate = shuf_ok and rand_ok and dir_pos
    else:
        anchor_gate = False
    im_means = {}
    for d in ["D1", "D2"]:
        vals = [r["delta_G_pixel_pp"] for r in im_rows if r["direction"] == d]
        im_means[d] = float(np.mean(vals))
    memory_competitive = im_means["D1"] >= -1.0 and im_means["D2"] >= -1.0
    memory_superior = im_means["D1"] >= 0.5 and im_means["D2"] >= 0.5
    category_concentrated = any(r["category_concentrated"] for r in concentration)

    if fold1_hard_fail or not performance_gate:
        outcome = "XLIV_A_STATE_ALIGNMENT_NOT_FOLD_STABLE"
    elif not anchor_gate:
        outcome = "XLIV_A_ANCHOR_SPECIFICITY_NOT_FOLD_STABLE"
    elif category_concentrated:
        outcome = "XLIV_A_CATEGORY_CONCENTRATED"
    elif memory_superior:
        outcome = "XLIV_A_COMPACT_STATE_ALIGNMENT_SUPERIOR"
    elif memory_competitive:
        outcome = "XLIV_A_COMPACT_STATE_ALIGNMENT_COMPETITIVE"
    else:
        outcome = "XLIV_A_STATE_ALIGNMENT_CONFIRMED_MEMORY_STILL_STRONGER"

    raw = metrics + ([dict(r, variant="C1_STATE_ALIGNED" if r["variant"] == "A0_LEARNED" else r["variant"]) for r in anchor_metrics if r["variant"] != "A0_LEARNED"] if controls_executed else [])
    write_json(OUT / "PARITY_AUDIT.json", {
        "fold0_C1_source": "XLIII-B S0 learned anchors",
        "fold1_C1_source": "new XLIV-A rank4 adapter runs",
        "C0_source": "XXXVIII K2 learned visual state",
        "C2_source": "XXXVIII/XXXIX K0/P0 instance memory",
        "backbone_forward": False,
        "official_mrad_reference": "NOT_AVAILABLE_UNDER_IDENTICAL_FOLD_PROTOCOL",
    })
    write_json(OUT / "XLIV_PROTOCOL.json", {"phase": "XLIV-A", "folds": protocols, "controls_executed": controls_executed, "target_accessed": False})
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "GROUPED_METRICS.csv", grouped)
    write_csv(OUT / "STATE_ALIGNMENT_GAIN.csv", align_rows)
    write_csv(OUT / "ANCHOR_SPECIFICITY.csv", anchor_rows)
    write_csv(OUT / "INSTANCE_MEMORY_COMPARISON.csv", im_rows)
    write_csv(OUT / "GAP_CLOSURE.csv", gap_rows)
    write_csv(OUT / "PER_CATEGORY.csv", per_category + anchor_percat)
    write_csv(OUT / "CATEGORY_CONCENTRATION.csv", concentration)
    write_csv(OUT / "ADAPTER_GEOMETRY.csv", adapter_geom)
    write_csv(OUT / "FEATURE_MOVEMENT.csv", drift)
    write_csv(OUT / "MARGIN_DIAGNOSTIC.csv", margins)
    write_csv(OUT / "TRAINING_CURVES.csv", curves)
    write_csv(OUT / "STORAGE_COMPLEXITY.csv", storage_rows())

    report = ["# Phase XLIV-A Full Cross-Fold State-Alignment Confirmation", "", f"Outcome: **{outcome}**", "", "## Fold1 Confirmation", ""]
    for r in fold1_rows:
        report.append(f"- {r['direction']} fold{r['fold']}: C1-C0 G {r['delta_G_pixel_pp']:+.4f}, P-AUROC {r['delta_P-AUROC_pp']:+.4f}, P-AP {r['delta_P-AP_pp']:+.4f}.")
    report.extend(["", "## Four-Fold Gain", ""])
    for r in align_rows:
        report.append(f"- {r['direction']} fold{r['fold']}: C1-C0 G {r['delta_G_pixel_pp']:+.4f}.")
    if anchor_rows:
        report.extend(["", "## Anchor Specificity", ""])
        for r in anchor_rows:
            report.append(f"- {r['direction']} fold{r['fold']}: Learned-Shuffled {r['A0_minus_A2_pp']:+.4f}, Learned-Random {r['A0_minus_A3_pp']:+.4f}, Learned-Centroid {r['A0_minus_A1_pp']:+.4f}.")
    report.extend(["", "## Instance Memory Comparison", ""])
    for d in ["D1", "D2"]:
        report.append(f"- {d}: mean C1-C2 G {im_means[d]:+.4f} pp.")
    report.extend(["", "## Gates", ""])
    report.append(f"- Performance gate: `{performance_gate}`.")
    report.append(f"- Anchor specificity gate: `{anchor_gate}`.")
    report.append(f"- Instance-memory competitive: `{memory_competitive}`; superior: `{memory_superior}`.")
    report.append(f"- Category concentrated: `{category_concentrated}`.")
    report.append("")
    report.append("Stop: Phase XLIV-A ended here. No new modules or extra variants were run.")
    report.append(f"Runtime from fold1 checkpoints: {checkpointed_runtime_sec:.1f}s.")
    report.append(f"Aggregation runtime: {time.time() - start:.1f}s.")
    (OUT / "XLIV_A_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(OUT / "OUTCOME.json", {
        "phase": "XLIV-A",
        "outcome": outcome,
        "fold1_pass": fold1_pass,
        "fold1_hard_fail": fold1_hard_fail,
        "controls_executed": controls_executed,
        "performance_gate": performance_gate,
        "anchor_specificity_gate": anchor_gate,
        "instance_memory_competitive": memory_competitive,
        "instance_memory_superior": memory_superior,
        "category_concentrated": category_concentrated,
        "checkpointed_fold1_runtime_sec": checkpointed_runtime_sec,
        "runtime_sec": time.time() - start,
    })
    with (ROOT / "EXPERIMENT_LOG.md").open("a", encoding="utf-8") as fp:
        fp.write("\n## Phase XLIV-A Full Cross-Fold State-Alignment Confirmation\n")
        fp.write(f"- Outcome: {outcome}.\n")
        fp.write(f"- Fold1 confirmation: {json.dumps(fold1_rows, default=json_default)}.\n")
        fp.write(f"- Full gain: {json.dumps(align_rows, default=json_default)}.\n")
        fp.write(f"- Anchor specificity: {json.dumps(anchor_rows, default=json_default)}.\n")
        fp.write(f"- InstanceMemory comparison: {json.dumps(im_rows, default=json_default)}.\n")


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    c1_payloads = ensure_fold1_c1(device)
    temp_metrics = historical_c0_c2() + historical_fold0_c1()
    for payload in c1_payloads.values():
        metric = dict(payload["metric"])
        metric["variant"] = "C1_STATE_ALIGNED"
        temp_metrics.append(metric)
    fold1_rows = c1_fold1_confirmation_rows(temp_metrics)
    hard_fail = any(r["hard_failure"] for r in fold1_rows)
    controls = {}
    if not hard_fail:
        controls = ensure_fold1_controls(device)
    else:
        print("[XLIV-A] early stop: fold1 hard failure", flush=True)
    aggregate(start, c1_payloads, controls)
    print(json.dumps(read_json(OUT / "OUTCOME.json"), indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
