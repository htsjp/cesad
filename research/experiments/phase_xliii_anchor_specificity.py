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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import (
    BATCH_SIZE,
    EPOCHS,
    LR,
    RANK,
    SEED,
    WEIGHT_DECAY,
    balanced_loss,
    load_train_samples,
)
from experiments.phase_xliii_adapter_head_decomposition import AdapterToState, gp
from experiments.phase_xxxviii_visual_state_representation import FEATURE_CACHE, SIDE, labels_and_rows, load_fold, norm_np, patch_labels
from experiments.phase_xxxix_local_multistate import evaluate_variant


OUT = Path("results/phase_xliii_anchor_specificity")
XXXVIII_OUT = Path("results/phase_xxxviii_visual_state_representation")
XLIII_A_OUT = Path("results/phase_xliii_adapter_head_decomposition")
PILOT = [("D1", 0), ("D2", 0)]
VARIANTS = [
    ("S0_LEARNED_STATE_ANCHORS", "K2_patch"),
    ("S1_CENTROID_ANCHORS", "K1_patch"),
    ("S2_SHUFFLED_LABEL_STATE_ANCHORS", "K3_patch"),
    ("S3_GEOMETRY_MATCHED_RANDOM_ANCHORS", "RANDOM"),
]


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


def read_xliii_a_t2():
    rows = {}
    with open(XLIII_A_OUT / "RAW_METRICS.csv", newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            if row["variant"] == "T2_ADAPTER_ONLY_FROZEN_STATES":
                rows[(row["direction"], int(row["fold"]))] = row
    return rows


def geometry_matched_random(learned_states):
    learned = norm_np(learned_states)
    c = float(np.clip(learned[0] @ learned[1], -0.999999, 0.999999))
    rng = np.random.default_rng(SEED)
    r_n = rng.normal(size=learned.shape[1]).astype(np.float32)
    r_n = norm_np(r_n[None])[0]
    e = rng.normal(size=learned.shape[1]).astype(np.float32)
    e = e - float(e @ r_n) * r_n
    e = norm_np(e[None])[0]
    r_a = c * r_n + math.sqrt(max(0.0, 1.0 - c * c)) * e
    random_states = norm_np(np.stack([r_n, r_a]).astype(np.float32))
    return random_states, {
        "learned_cos_sN_sA": c,
        "random_cos_rN_rA": float(random_states[0] @ random_states[1]),
        "abs_cos_error": float(abs((random_states[0] @ random_states[1]) - c)),
        "passed": bool(abs((random_states[0] @ random_states[1]) - c) < 1e-6),
    }


def train_adapter(x_np, y_np, states, device, variant):
    model = AdapterToState(x_np.shape[1], states).to(device)
    x_all = norm_np(x_np)
    y_all = y_np.astype(np.float32)
    rng = np.random.default_rng(SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    curves = []
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(y_all))
        vals = []
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            x = torch.tensor(x_all[idx], dtype=torch.float32, device=device)
            y = torch.tensor(y_all[idx], dtype=torch.float32, device=device)
            opt.zero_grad()
            logits, _ = model.logits(x)
            loss, bce, dice = balanced_loss(logits, y)
            loss.backward()
            opt.step()
            vals.append((float(loss.detach().cpu()), float(bce.cpu()), float(dice.cpu())))
        arr = np.asarray(vals)
        curves.append({
            "variant": variant,
            "epoch": epoch,
            "loss": float(arr[:, 0].mean()),
            "bce": float(arr[:, 1].mean()),
            "dice": float(arr[:, 2].mean()),
        })
    return model.eval(), curves


def collect_held_patch_labels(dataset, held_indices):
    labels = []
    image_labels = []
    for idx in held_indices:
        item = dataset[idx]
        labels.append(patch_labels(item))
        image_labels.append(int(item["anomaly"]))
    return np.concatenate(labels).astype(np.int64), np.asarray(image_labels, dtype=np.int64)


def summarize_group(values, labels, split, variant, direction, fold, value_name):
    rows = []
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    for label, name in [(0, "normal"), (1, "anomaly")]:
        vals = values[labels == label]
        if vals.size == 0:
            continue
        rows.append({
            "direction": direction,
            "fold": fold,
            "variant": variant,
            "split": split,
            "group": name,
            "value": value_name,
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "q10": float(np.quantile(vals, 0.10)),
            "q90": float(np.quantile(vals, 0.90)),
            "count": int(vals.size),
        })
    if np.any(labels == 0) and np.any(labels == 1):
        n = values[labels == 0]
        a = values[labels == 1]
        rows.append({
            "direction": direction,
            "fold": fold,
            "variant": variant,
            "split": split,
            "group": "anomaly_minus_normal_gap",
            "value": value_name,
            "mean": float(np.mean(a) - np.mean(n)),
            "median": float(np.median(a) - np.median(n)),
            "q10": float("nan"),
            "q90": float("nan"),
            "count": int(values.size),
        })
    return rows


def score_and_diagnose(held_patch, held_patch_y, model, states, device, direction, fold, variant):
    maps = np.zeros((held_patch.shape[0], SIDE, SIDE), dtype=np.float16)
    margins_before, margins_after, drift = [], [], []
    state_t = torch.tensor(norm_np(states), dtype=torch.float32, device=device)
    with torch.no_grad():
        for i in range(held_patch.shape[0]):
            x = torch.tensor(norm_np(held_patch[i].astype(np.float32)), dtype=torch.float32, device=device)
            logits, z = model.logits(x)
            maps[i] = torch.sigmoid(logits).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float16)
            margins_before.append((x @ state_t[1] - x @ state_t[0]).detach().cpu().numpy())
            margins_after.append((z @ state_t[1] - z @ state_t[0]).detach().cpu().numpy())
            drift.append((x * z).sum(dim=-1).detach().cpu().numpy())
    mb = np.concatenate(margins_before)
    ma = np.concatenate(margins_after)
    cosxz = np.concatenate(drift)
    margin_rows = []
    margin_rows.extend(summarize_group(mb, held_patch_y, "heldout", variant, direction, fold, "m_before"))
    margin_rows.extend(summarize_group(ma, held_patch_y, "heldout", variant, direction, fold, "m_after"))
    drift_rows = summarize_group(cosxz, held_patch_y, "heldout", variant, direction, fold, "cos_x_z")
    return maps, margin_rows, drift_rows


def train_margin_drift_rows(x_train, y_train, model, states, device, direction, fold, variant):
    x = torch.tensor(norm_np(x_train.astype(np.float32)), dtype=torch.float32, device=device)
    state_t = torch.tensor(norm_np(states), dtype=torch.float32, device=device)
    with torch.no_grad():
        _logits, z = model.logits(x)
        mb = (x @ state_t[1] - x @ state_t[0]).detach().cpu().numpy()
        ma = (z @ state_t[1] - z @ state_t[0]).detach().cpu().numpy()
        cosxz = (x * z).sum(dim=-1).detach().cpu().numpy()
    rows = []
    rows.extend(summarize_group(mb, y_train, "source_train_sample", variant, direction, fold, "m_before"))
    rows.extend(summarize_group(ma, y_train, "source_train_sample", variant, direction, fold, "m_after"))
    drift = summarize_group(cosxz, y_train, "source_train_sample", variant, direction, fold, "cos_x_z")
    return rows, drift


def low_rank_rows(models, direction, fold):
    rows = []
    deltas = {}
    flats = {}
    for variant, model in models.items():
        a = model.a.weight.detach().cpu().numpy()
        b = model.b.weight.detach().cpu().numpy()
        delta = (b @ a).astype(np.float32)
        s = np.linalg.svd(delta, compute_uv=False)
        p = s / max(float(np.sum(s)), 1e-12)
        eff_rank = float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0]))))
        deltas[variant] = delta
        flats[variant] = delta.reshape(-1)
        rows.append({
            "direction": direction,
            "fold": fold,
            "variant": variant,
            "frobenius_norm": float(np.linalg.norm(delta)),
            "spectral_norm": float(s[0]) if s.size else 0.0,
            "effective_rank": eff_rank,
        })
    for i, v1 in enumerate(models):
        for v2 in list(models)[i + 1:]:
            f1 = flats[v1]
            f2 = flats[v2]
            rows.append({
                "direction": direction,
                "fold": fold,
                "variant": f"{v1}__vs__{v2}",
                "frobenius_norm": float("nan"),
                "spectral_norm": float("nan"),
                "effective_rank": float("nan"),
                "deltaW_flat_cosine": float(f1 @ f2 / max(np.linalg.norm(f1) * np.linalg.norm(f2), 1e-12)),
            })
    return rows


def anchor_geometry_rows(anchors, random_check, direction, fold):
    rows = []
    for variant, states in anchors.items():
        s = norm_np(states)
        rows.append({
            "direction": direction,
            "fold": fold,
            "variant": variant,
            "cos_anchor_N_A": float(s[0] @ s[1]),
            "norm_anchor_N": float(np.linalg.norm(s[0])),
            "norm_anchor_A": float(np.linalg.norm(s[1])),
            "geometry_matched_random_check": json.dumps(random_check) if variant == "S3_GEOMETRY_MATCHED_RANDOM_ANCHORS" else "",
        })
    return rows


def run_fold(direction, fold, device):
    start = time.time()
    dataset_name, dataset, rotation, train_indices, held_indices = load_fold(direction, fold)
    cache = np.load(FEATURE_CACHE / f"{direction}_fold{fold}_clip.npz", allow_pickle=True)
    train_patch = cache["train_patch"]
    held_patch = cache["held_patch"]
    states38 = np.load(XXXVIII_OUT / "states" / f"{direction}_fold{fold}_states.npz")
    x_train, y_train, _cats = load_train_samples(dataset, train_indices, train_patch)
    held_patch_y, _held_image_y = collect_held_patch_labels(dataset, held_indices)
    zeros = np.zeros((len(held_indices),), dtype=np.float32)

    random_states, random_check = geometry_matched_random(states38["K2_patch"])
    anchors = {
        "S0_LEARNED_STATE_ANCHORS": states38["K2_patch"],
        "S1_CENTROID_ANCHORS": states38["K1_patch"],
        "S2_SHUFFLED_LABEL_STATE_ANCHORS": states38["K3_patch"],
        "S3_GEOMETRY_MATCHED_RANDOM_ANCHORS": random_states,
    }

    metrics, curves, margins, drift, percat = [], [], [], [], []
    models = {}
    for variant, _key in VARIANTS:
        print(f"[XLIII-B] {direction} fold{fold} {variant} train", flush=True)
        model, c = train_adapter(x_train, y_train, anchors[variant], device, variant)
        models[variant] = model
        for row in c:
            row.update({"direction": direction, "fold": fold})
            curves.append(row)
        train_m, train_d = train_margin_drift_rows(x_train, y_train, model, anchors[variant], device, direction, fold, variant)
        margins.extend(train_m)
        drift.extend(train_d)
        print(f"[XLIII-B] {direction} fold{fold} {variant} score", flush=True)
        maps, held_m, held_d = score_and_diagnose(held_patch, held_patch_y, model, anchors[variant], device, direction, fold, variant)
        margins.extend(held_m)
        drift.extend(held_d)
        res = evaluate_variant(dataset, held_indices, zeros, maps)
        row = {"direction": direction, "fold": fold, "dataset": dataset_name, "variant": variant, **res["mean"]}
        row["G_pixel"] = gp(row)
        metrics.append(row)
        for cat, vals in res["per_category"].items():
            percat.append({"direction": direction, "fold": fold, "category": cat, "variant": variant, **vals})

    protocol = {
        "direction": direction,
        "fold": fold,
        "dataset": dataset_name,
        "train_categories": rotation["train_categories"],
        "heldout_categories": rotation["val_categories"],
        "source": "cached frozen patch features",
        "backbone_forward": False,
        "memory": False,
        "text_prompt": False,
        "rank": RANK,
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "train_samples": int(x_train.shape[0]),
        "held_images": int(len(held_indices)),
        "runtime_sec": time.time() - start,
    }
    return {
        "protocol": protocol,
        "metrics": metrics,
        "per_category": percat,
        "curves": curves,
        "anchor_geometry": anchor_geometry_rows(anchors, random_check, direction, fold),
        "margins": margins,
        "drift": drift,
        "low_rank": low_rank_rows(models, direction, fold),
    }


def aggregate(payloads, start):
    metrics = [r for p in payloads for r in p["metrics"]]
    protocols = [p["protocol"] for p in payloads]
    anchor_geometry = [r for p in payloads for r in p["anchor_geometry"]]
    margins = [r for p in payloads for r in p["margins"]]
    drift = [r for p in payloads for r in p["drift"]]
    low_rank = [r for p in payloads for r in p["low_rank"]]
    curves = [r for p in payloads for r in p["curves"]]
    percat = [r for p in payloads for r in p["per_category"]]

    write_json(OUT / "ANCHOR_PROTOCOL.json", {"phase": "XLIII-B", "folds": protocols, "target_accessed": False})
    write_csv(OUT / "ANCHOR_GEOMETRY.csv", anchor_geometry)
    write_csv(OUT / "RAW_METRICS.csv", metrics)
    write_csv(OUT / "MARGIN_DIAGNOSTIC.csv", margins)
    write_csv(OUT / "FEATURE_DRIFT.csv", drift)
    write_csv(OUT / "LOW_RANK_UPDATE_GEOMETRY.csv", low_rank)
    write_csv(OUT / "TRAINING_CURVES.csv", curves)
    write_csv(OUT / "PER_CATEGORY.csv", percat)

    xliii_a = read_xliii_a_t2()
    by = {(r["direction"], int(r["fold"]), r["variant"]): r for r in metrics}
    parity, specificity = [], []
    parity_ok = True
    for direction, fold in PILOT:
        s0 = by[(direction, fold, "S0_LEARNED_STATE_ANCHORS")]
        ref = xliii_a[(direction, fold)]
        delta_pauc = float(s0["P-AUROC"]) - float(ref["P-AUROC"])
        delta_pap = float(s0["P-AP"]) - float(ref["P-AP"])
        delta_g = gp(s0) - gp(ref)
        ok = abs(delta_pauc) <= 0.02 and abs(delta_pap) <= 0.02
        parity_ok = parity_ok and ok
        parity.append({
            "direction": direction,
            "fold": fold,
            "reference": "XLIII-A T2_ADAPTER_ONLY_FROZEN_STATES",
            "S0_P-AUROC": float(s0["P-AUROC"]),
            "ref_P-AUROC": float(ref["P-AUROC"]),
            "delta_P-AUROC_pp": delta_pauc,
            "S0_P-AP": float(s0["P-AP"]),
            "ref_P-AP": float(ref["P-AP"]),
            "delta_P-AP_pp": delta_pap,
            "delta_G_pixel_pp": delta_g,
            "passed": ok,
        })
        s1 = by[(direction, fold, "S1_CENTROID_ANCHORS")]
        s2 = by[(direction, fold, "S2_SHUFFLED_LABEL_STATE_ANCHORS")]
        s3 = by[(direction, fold, "S3_GEOMETRY_MATCHED_RANDOM_ANCHORS")]
        specificity.append({
            "direction": direction,
            "fold": fold,
            "S0_G_pixel": gp(s0),
            "S1_G_pixel": gp(s1),
            "S2_G_pixel": gp(s2),
            "S3_G_pixel": gp(s3),
            "S0_minus_S1_G_pixel_pp": gp(s0) - gp(s1),
            "S0_minus_S2_G_pixel_pp": gp(s0) - gp(s2),
            "S0_minus_S3_G_pixel_pp": gp(s0) - gp(s3),
            "learned_specific_vs_shuffled": bool(gp(s0) - gp(s2) >= 1.0),
            "learned_specific_vs_random": bool(gp(s0) - gp(s3) >= 1.0),
            "task_optimized_beats_centroid": bool(gp(s0) - gp(s1) >= 0.5),
            "learned_approx_centroid": bool(abs(gp(s0) - gp(s1)) <= 0.30),
        })
    write_json(OUT / "PARITY_AUDIT.json", {"S0_recomputed_from_same_seed": True, "passed": parity_ok, "rows": parity})
    write_csv(OUT / "ANCHOR_SPECIFICITY.csv", specificity)

    if not parity_ok:
        outcome = "XLIII_B_R1_SIGNAL_NOT_REPRODUCIBLE"
    else:
        d1 = next(r for r in specificity if r["direction"] == "D1")
        d2 = next(r for r in specificity if r["direction"] == "D2")
        both_specific = all(r["learned_specific_vs_shuffled"] and r["learned_specific_vs_random"] for r in [d1, d2])
        both_task_opt = all(r["task_optimized_beats_centroid"] for r in [d1, d2])
        both_centroid_equiv = all(r["learned_approx_centroid"] for r in [d1, d2])
        controls_close = all(abs(r["S0_minus_S2_G_pixel_pp"]) < 0.5 or abs(r["S0_minus_S3_G_pixel_pp"]) < 0.5 for r in [d1, d2])
        one_dir_specific = any(r["learned_specific_vs_shuffled"] and r["learned_specific_vs_random"] for r in [d1, d2])
        if both_specific and both_task_opt:
            outcome = "XLIII_B_LEARNED_STATE_ANCHOR_SPECIFIC"
        elif both_specific and both_centroid_equiv:
            outcome = "XLIII_B_GENERIC_ANOMALY_AXIS_SUFFICIENT"
        elif controls_close:
            outcome = "XLIII_B_ANCHOR_IDENTITY_NOT_SPECIFIC"
        elif one_dir_specific:
            outcome = "XLIII_B_ANCHOR_SPECIFICITY_DIRECTION_DEPENDENT"
        else:
            outcome = "XLIII_B_ANCHOR_IDENTITY_NOT_SPECIFIC"

    report = ["# Phase XLIII-B Anomaly State Anchor Specificity Audit", "", f"Outcome: **{outcome}**", "", "## Metrics", ""]
    for r in metrics:
        report.append(f"- {r['direction']} fold{r['fold']} {r['variant']}: P-AUROC {float(r['P-AUROC']):.4f}, P-AP {float(r['P-AP']):.4f}, G {float(r['G_pixel']):.4f}, PRO {float(r.get('PRO', float('nan'))):.4f}.")
    report.extend(["", "## Anchor Specificity", ""])
    for r in specificity:
        report.append(f"- {r['direction']} fold{r['fold']}: S0-S1 {r['S0_minus_S1_G_pixel_pp']:+.4f}, S0-S2 {r['S0_minus_S2_G_pixel_pp']:+.4f}, S0-S3 {r['S0_minus_S3_G_pixel_pp']:+.4f} pp G.")
    report.extend(["", "## Parity", ""])
    for r in parity:
        report.append(f"- {r['direction']} fold{r['fold']}: delta P-AUROC {r['delta_P-AUROC_pp']:+.5f}, delta P-AP {r['delta_P-AP_pp']:+.5f}, passed `{r['passed']}`.")
    report.extend(["", "Stop: Phase XLIII-B ended after S0-S3 on D1f0/D2f0. No fold1 or extra variants were run.", f"Runtime: {time.time() - start:.1f}s."])
    (OUT / "XLIII_B_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(OUT / "OUTCOME.json", {
        "phase": "XLIII-B",
        "outcome": outcome,
        "parity_passed": parity_ok,
        "specificity": specificity,
        "runtime_sec": time.time() - start,
    })
    with (ROOT / "EXPERIMENT_LOG.md").open("a", encoding="utf-8") as fp:
        fp.write("\n## Phase XLIII-B Anomaly State Anchor Specificity Audit\n")
        fp.write(f"- Outcome: {outcome}.\n")
        fp.write(f"- Anchor specificity: {json.dumps(specificity, default=json_default)}.\n")


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = OUT / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    payloads = []
    for direction, fold in PILOT:
        path = ckpt / f"{direction}_fold{fold}.json"
        if path.exists():
            print(f"[XLIII-B] resume {direction} fold{fold}", flush=True)
            payloads.append(read_json(path))
            continue
        payload = run_fold(direction, fold, device)
        write_json(path, payload)
        payloads.append(payload)
    aggregate(payloads, start)
    print(json.dumps(read_json(OUT / "OUTCOME.json"), indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
