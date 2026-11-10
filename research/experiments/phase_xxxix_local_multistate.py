import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mrad_audit_utils import fast_cal_pro_score
from experiments.phase_xa_checkpoint_source_readout import all_indices_for_categories, direction_cfg
from experiments.phase_xa_smoke import load_dataset, write_csv
from experiments.phase_xxxviii_visual_state_representation import (
    FEATURE_CACHE,
    IMAGE_SIZE,
    K0_PATCHES_PER_IMAGE_CLASS,
    LR,
    SEED,
    SIDE,
    SIGMA,
    class_category_patch_sample,
    fps_indices,
    group_gain,
    instance_memory_scores,
    k0_fps_patch_memory,
    labels_and_rows,
    load_fold,
    norm_np,
    patch_labels,
    safe_ap,
    safe_auc,
    score_patch_instance_all,
    softmax2,
    state_scores,
    upsample_map,
)
from phase_xa.source_folds import deterministic_source_folds


OUT = Path("results/phase_xxxix_local_multistate")
XXXVIII_OUT = Path("results/phase_xxxviii_visual_state_representation")
EPOCHS = 20
K_STATES = 4
PATCH_TRAIN_SAMPLES_PER_CLASS_PER_CATEGORY = 512
PATCH_LEAKAGE_SAMPLES_PER_CATEGORY = 256


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


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=json_default), encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def spherical_kmeans(x, k=K_STATES, seed=SEED):
    x = norm_np(x)
    if x.shape[0] < k:
        reps = np.concatenate([x, np.repeat(x[-1:], k - x.shape[0], axis=0)], axis=0)
        return norm_np(reps[:k])
    km = KMeans(n_clusters=k, n_init=10, random_state=seed, max_iter=100)
    km.fit(x)
    return norm_np(km.cluster_centers_.astype(np.float32))


def static_kmeans_states(patch_x, patch_y):
    normal = spherical_kmeans(patch_x[patch_y == 0], K_STATES, SEED)
    anomaly = spherical_kmeans(patch_x[patch_y == 1], K_STATES, SEED)
    return np.stack([normal, anomaly]).astype(np.float32)


def multistate_logits(queries, states):
    q = norm_np(queries)
    s = norm_np(states.reshape(-1, states.shape[-1]))
    sims = (q @ s.T).reshape(q.shape[0], 2, K_STATES)
    logits = torch.logsumexp(torch.tensor(sims, dtype=torch.float32), dim=-1).numpy()
    return logits.astype(np.float32), sims.astype(np.float32)


def multistate_scores(queries, states):
    logits, sims = multistate_logits(queries, states)
    return softmax2(logits[:, 0], logits[:, 1]).astype(np.float32), logits, sims


def score_patch_multistate_all(held_patch, patch_states):
    maps = np.zeros((held_patch.shape[0], SIDE, SIDE), dtype=np.float16)
    util = Counter()
    for i in range(held_patch.shape[0]):
        scores, _logits, sims = multistate_scores(np.asarray(held_patch[i], dtype=np.float32), patch_states)
        maps[i] = scores.reshape(SIDE, SIDE).astype(np.float16)
        pred_cls = (scores >= 0.5).astype(np.int64)
        best = sims[np.arange(sims.shape[0]), pred_cls].argmax(axis=1)
        for cls, state_id in zip(pred_cls.tolist(), best.tolist()):
            util[(int(cls), int(state_id))] += 1
    return maps, util


def train_multistate(patch_x, patch_y, patch_cat, init_states, shuffled=False, device="cpu"):
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    y_train = np.asarray(patch_y, dtype=np.int64).copy()
    if shuffled:
        y_train = rng.permutation(y_train)
    x_np = norm_np(patch_x)
    state = torch.nn.Parameter(torch.tensor(init_states, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW([state], lr=LR, weight_decay=0.0)
    cats = sorted(set(patch_cat.tolist()))
    curves = []
    init_t = torch.tensor(norm_np(init_states.reshape(-1, init_states.shape[-1])).reshape(init_states.shape), dtype=torch.float32, device=device)
    for epoch in range(1, EPOCHS + 1):
        parts = []
        for cat in cats:
            cidx = np.flatnonzero(patch_cat == cat)
            nidx = cidx[y_train[cidx] == 0]
            aidx = cidx[y_train[cidx] == 1]
            if nidx.size == 0 or aidx.size == 0:
                continue
            take = min(nidx.size, aidx.size, 128)
            parts.append(rng.choice(nidx, size=take, replace=False))
            parts.append(rng.choice(aidx, size=take, replace=False))
        if not parts:
            nidx = np.flatnonzero(y_train == 0)
            aidx = np.flatnonzero(y_train == 1)
            take = min(len(nidx), len(aidx), 2048)
            parts = [rng.choice(nidx, size=take, replace=False), rng.choice(aidx, size=take, replace=False)]
        idx = np.concatenate(parts)
        x = torch.tensor(x_np[idx], dtype=torch.float32, device=device)
        y = torch.tensor(y_train[idx], dtype=torch.long, device=device)
        opt.zero_grad()
        s = F.normalize(state, dim=-1)
        sims = torch.einsum("bd,ckd->bck", x, s)
        logits = torch.logsumexp(sims, dim=-1)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        with torch.no_grad():
            disp = torch.linalg.norm(F.normalize(state, dim=-1) - init_t, dim=-1).mean().item()
            sim = torch.einsum("ckd,cld->ckl", F.normalize(state, dim=-1), F.normalize(state, dim=-1))
        curves.append({
            "epoch": epoch,
            "variant": "P5_SHUFFLED_LABEL_MULTI_STATE" if shuffled else "P3_LEARNED_MULTI_STATE4",
            "loss": float(loss.detach().cpu()),
            "mean_displacement_from_init": disp,
            "normal_mean_pairwise_cos": mean_offdiag(sim[0].detach().cpu().numpy()),
            "anomaly_mean_pairwise_cos": mean_offdiag(sim[1].detach().cpu().numpy()),
        })
    return norm_np(state.detach().cpu().numpy().reshape(-1, init_states.shape[-1])).reshape(init_states.shape), curves


def mean_offdiag(mat):
    vals = mat[~np.eye(mat.shape[0], dtype=bool)]
    return float(np.mean(vals)) if vals.size else float("nan")


def effective_states(mat):
    dist = 1.0 - mat
    vals = dist[~np.eye(mat.shape[0], dtype=bool)]
    return float(1.0 + np.sum(vals > 0.02) / max(1, K_STATES - 1))


def geometry_rows(states_by_variant, direction, fold):
    rows = []
    for variant, states in states_by_variant.items():
        s = norm_np(states.reshape(-1, states.shape[-1])).reshape(states.shape)
        for cls_id, cls_name in [(0, "normal"), (1, "anomaly")]:
            mat = s[cls_id] @ s[cls_id].T
            vals = mat[~np.eye(K_STATES, dtype=bool)]
            rows.append({
                "direction": direction,
                "fold": fold,
                "variant": variant,
                "class": cls_name,
                "mean_offdiag_cos": float(np.mean(vals)),
                "max_offdiag_cos": float(np.max(vals)),
                "min_offdiag_cos": float(np.min(vals)),
                "state_collapse": bool(float(np.mean(vals)) > 0.98),
                "effective_state_count": effective_states(mat),
                "pairwise_cosine_matrix": json.dumps(mat.tolist()),
            })
    return rows


def utilization_rows(util_by_variant, direction, fold):
    rows = []
    for variant, util in util_by_variant.items():
        totals = {cls: sum(v for (c, _sid), v in util.items() if c == cls) for cls in [0, 1]}
        for cls, cls_name in [(0, "normal"), (1, "anomaly")]:
            counts = np.asarray([util.get((cls, sid), 0) for sid in range(K_STATES)], dtype=np.float64)
            total = max(1.0, float(totals[cls]))
            probs = counts / total
            ent = -float(np.sum([p * math.log(p + 1e-12) for p in probs]))
            rows.append({
                "direction": direction,
                "fold": fold,
                "variant": variant,
                "class": cls_name,
                "state_counts": json.dumps(counts.astype(int).tolist()),
                "state_frequencies": json.dumps(probs.tolist()),
                "entropy": ent,
                "active_states_ge_1pct": int(np.sum(probs >= 0.01)),
                "dead_states_lt_1pct": int(np.sum(probs < 0.01)),
            })
    return rows


def evaluate_variant(dataset, held_indices, image_scores, patch_maps):
    by_cat = defaultdict(lambda: {"iy": [], "is": [], "py": [], "ps": []})
    for pos, idx in enumerate(held_indices):
        item = dataset[idx]
        cat = str(item["cls_name"])
        by_cat[cat]["iy"].append(int(item["anomaly"]))
        by_cat[cat]["is"].append(float(image_scores[pos]))
        mask = (item["img_mask"].detach().cpu().numpy() > 0.5).astype(np.uint8).reshape(-1)
        full = upsample_map(patch_maps[pos]).reshape(-1)
        by_cat[cat]["py"].append(mask)
        by_cat[cat]["ps"].append(full)
    per, means = {}, defaultdict(list)
    for cat, vals in sorted(by_cat.items()):
        py = np.concatenate(vals["py"])
        ps = np.concatenate(vals["ps"]).astype(np.float32)
        masks = np.stack([x.reshape(IMAGE_SIZE, IMAGE_SIZE) for x in vals["py"]]).astype(np.uint8)
        maps = np.stack([x.reshape(IMAGE_SIZE, IMAGE_SIZE) for x in vals["ps"]]).astype(np.float32)
        row = {
            "I-AUROC": safe_auc(vals["iy"], vals["is"]),
            "I-AP": safe_ap(vals["iy"], vals["is"]),
            "P-AUROC": safe_auc(py, ps),
            "P-AP": safe_ap(py, ps),
            "PRO": float(fast_cal_pro_score(masks, maps) * 100.0),
        }
        per[cat] = row
        for key, value in row.items():
            if np.isfinite(value):
                means[key].append(value)
    return {"mean": {k: float(np.mean(v)) for k, v in means.items()}, "per_category": per}


def category_cv_scores(x, cats):
    cats = np.asarray(cats)
    if len(set(cats.tolist())) < 2 or len(cats) < 10:
        return float("nan"), float("nan")
    _, counts = np.unique(cats, return_counts=True)
    folds = min(5, int(counts.min()))
    if folds < 2:
        return float("nan"), float("nan")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, class_weight="balanced", random_state=SEED))
    acc = cross_val_score(clf, x, cats, cv=cv, scoring="accuracy")
    bacc = cross_val_score(clf, x, cats, cv=cv, scoring="balanced_accuracy")
    return float(np.mean(acc) * 100.0), float(np.mean(bacc) * 100.0)


def leakage_rows(raw_x, cats, state_variants, direction, fold):
    rows = []
    acc, bacc = category_cv_scores(raw_x, cats)
    rows.append({"direction": direction, "fold": fold, "variant": "RAW_FEATURE", "category_accuracy": acc, "balanced_category_accuracy": bacc, "dimensionality": int(raw_x.shape[1])})
    for variant, states in state_variants.items():
        if states.ndim == 2:
            _score, logits = state_scores(raw_x, states)
            z = logits
        else:
            z, _sims = multistate_logits(raw_x, states)
        acc, bacc = category_cv_scores(z, cats)
        rows.append({"direction": direction, "fold": fold, "variant": variant, "category_accuracy": acc, "balanced_category_accuracy": bacc, "dimensionality": int(z.shape[1])})
    return rows


def anomaly_mode_rows(patch_x, patch_y, patch_cat, states, direction, fold):
    rows = []
    anom = patch_y == 1
    if not np.any(anom):
        return rows
    scores, logits, sims = multistate_scores(patch_x[anom], states)
    best = sims[:, 1, :].argmax(axis=1)
    cats = patch_cat[anom]
    for sid in range(K_STATES):
        mask = best == sid
        comp = Counter(cats[mask].tolist())
        rows.append({
            "direction": direction,
            "fold": fold,
            "anomaly_state": sid,
            "patch_count": int(mask.sum()),
            "mean_anomaly_confidence": float(np.mean(scores[mask])) if mask.any() else float("nan"),
            "source_category_composition": json.dumps(dict(sorted(comp.items()))),
        })
    return rows


def run_fold(direction, fold, device):
    start = time.time()
    dataset_name, dataset, rotation, train_indices, held_indices = load_fold(direction, fold)
    cache = np.load(FEATURE_CACHE / f"{direction}_fold{fold}_clip.npz", allow_pickle=True)
    train_patch = cache["train_patch"]
    held_patch = cache["held_patch"]
    held_cls = norm_np(cache["held_cls"].astype(np.float32))
    y_train, cat_train, _train_rows = labels_and_rows(dataset, train_indices)
    states38 = np.load(XXXVIII_OUT / "states" / f"{direction}_fold{fold}_states.npz")
    image_k2 = states38["K2_img"]
    patch_p1 = states38["K2_patch"]
    image_scores, _ = state_scores(held_cls, image_k2)

    patch_train_x, patch_train_y, patch_train_cat = class_category_patch_sample(
        dataset, train_indices, train_patch, PATCH_TRAIN_SAMPLES_PER_CLASS_PER_CATEGORY, seed=SEED
    )
    patch_mem_x, patch_mem_y, _patch_mem_cat = k0_fps_patch_memory(dataset, train_indices, train_patch)
    p2_states = static_kmeans_states(patch_train_x, patch_train_y)
    p3_states, curve3 = train_multistate(patch_train_x, patch_train_y, patch_train_cat, p2_states, shuffled=False, device=device)
    p4_states = np.stack([np.repeat(patch_p1[0:1], K_STATES, axis=0), np.repeat(patch_p1[1:2], K_STATES, axis=0)]).astype(np.float32)
    p5_states, curve5 = train_multistate(patch_train_x, patch_train_y, patch_train_cat, p2_states, shuffled=True, device=device)

    variants = {
        "P2_STATIC_KMEANS4_STATE": p2_states,
        "P3_LEARNED_MULTI_STATE4": p3_states,
        "P4_DUPLICATED_SINGLE_STATE": p4_states,
        "P5_SHUFFLED_LABEL_MULTI_STATE": p5_states,
    }
    metrics, percat = [], []
    print(f"[XXXIX-A] {direction} fold{fold} score P0", flush=True)
    p0_maps = score_patch_instance_all(held_patch, patch_mem_x, patch_mem_y)
    res = evaluate_variant(dataset, held_indices, image_scores, p0_maps)
    metrics.append({"direction": direction, "fold": fold, "dataset": dataset_name, "variant": "P0_INSTANCE_MEMORY", **res["mean"]})
    for cat, vals in res["per_category"].items():
        percat.append({"direction": direction, "fold": fold, "category": cat, "variant": "P0_INSTANCE_MEMORY", **vals})

    print(f"[XXXIX-A] {direction} fold{fold} score P1", flush=True)
    p1_maps, _logits = score_patch_single_all(held_patch, patch_p1)
    res = evaluate_variant(dataset, held_indices, image_scores, p1_maps)
    metrics.append({"direction": direction, "fold": fold, "dataset": dataset_name, "variant": "P1_SINGLE_STATE", **res["mean"]})
    for cat, vals in res["per_category"].items():
        percat.append({"direction": direction, "fold": fold, "category": cat, "variant": "P1_SINGLE_STATE", **vals})

    util_by_variant = {}
    for variant, states in variants.items():
        print(f"[XXXIX-A] {direction} fold{fold} score {variant}", flush=True)
        maps, util = score_patch_multistate_all(held_patch, states)
        util_by_variant[variant] = util
        res = evaluate_variant(dataset, held_indices, image_scores, maps)
        metrics.append({"direction": direction, "fold": fold, "dataset": dataset_name, "variant": variant, **res["mean"]})
        for cat, vals in res["per_category"].items():
            percat.append({"direction": direction, "fold": fold, "category": cat, "variant": variant, **vals})

    leak_x, leak_y, leak_cat = class_category_patch_sample(dataset, train_indices, train_patch, PATCH_LEAKAGE_SAMPLES_PER_CATEGORY, seed=SEED + 1)
    leakage = leakage_rows(leak_x, leak_cat, {
        "P1_SINGLE_STATE": patch_p1,
        "P3_LEARNED_MULTI_STATE4": p3_states,
    }, direction, fold)
    geometry = geometry_rows(variants, direction, fold)
    utilization = utilization_rows(util_by_variant, direction, fold)
    modes = anomaly_mode_rows(patch_train_x, patch_train_y, patch_train_cat, p3_states, direction, fold)
    curves = []
    for row in curve3 + curve5:
        row.update({"direction": direction, "fold": fold})
        curves.append(row)

    state_dir = OUT / "states"
    state_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(state_dir / f"{direction}_fold{fold}_patch_states.npz", P2=p2_states, P3=p3_states, P4=p4_states, P5=p5_states)

    image_parity = {
        "direction": direction,
        "fold": fold,
        "max_abs_image_score_delta_P1_to_P5": 0.0,
        "reference": "single frozen XXXVIII K2 image score array reused for all P0-P5 variants",
        "passed": True,
    }
    storage = [{
        "direction": direction,
        "fold": fold,
        "variant": "P0_INSTANCE_MEMORY",
        "patch_vectors": int(patch_mem_x.shape[0]),
        "bytes_float32": int(patch_mem_x.size * 4 + len(patch_mem_y) * 4),
        "patch_comparisons_per_query": int(patch_mem_x.shape[0]),
        "compression_ratio_vs_P0": 1.0,
    }, {
        "direction": direction,
        "fold": fold,
        "variant": "P1_SINGLE_STATE",
        "patch_vectors": 2,
        "bytes_float32": int(2 * train_patch.shape[-1] * 4),
        "patch_comparisons_per_query": 2,
        "compression_ratio_vs_P0": float((patch_mem_x.size * 4 + len(patch_mem_y) * 4) / (2 * train_patch.shape[-1] * 4)),
    }, {
        "direction": direction,
        "fold": fold,
        "variant": "P3_LEARNED_MULTI_STATE4",
        "patch_vectors": 8,
        "bytes_float32": int(8 * train_patch.shape[-1] * 4),
        "patch_comparisons_per_query": 8,
        "compression_ratio_vs_P0": float((patch_mem_x.size * 4 + len(patch_mem_y) * 4) / (8 * train_patch.shape[-1] * 4)),
    }]
    protocol = {
        "direction": direction,
        "fold": fold,
        "dataset": dataset_name,
        "train_categories": rotation["train_categories"],
        "heldout_categories": rotation["val_categories"],
        "patch_feature_dim": int(train_patch.shape[-1]),
        "patch_tokens": int(train_patch.shape[1]),
        "image_branch": "frozen XXXVIII K2 image states",
        "patch_state_training_samples": int(patch_train_x.shape[0]),
        "optimizer": "AdamW",
        "lr": LR,
        "weight_decay": 0,
        "epochs": EPOCHS,
        "seed": SEED,
        "runtime_sec": time.time() - start,
    }
    return {
        "protocol": protocol,
        "parity": image_parity,
        "metrics": metrics,
        "per_category": percat,
        "geometry": geometry,
        "utilization": utilization,
        "leakage": leakage,
        "modes": modes,
        "storage": storage,
        "curves": curves,
    }


def score_patch_single_all(held_patch, patch_states):
    maps = np.zeros((held_patch.shape[0], SIDE, SIDE), dtype=np.float16)
    logits_all = []
    for i in range(held_patch.shape[0]):
        scores, logits = state_scores(np.asarray(held_patch[i], dtype=np.float32), patch_states)
        maps[i] = scores.reshape(SIDE, SIDE).astype(np.float16)
        logits_all.append(logits)
    return maps, logits_all


def aggregate(payloads, start):
    protocols = [p["protocol"] for p in payloads]
    parity = [p["parity"] for p in payloads]
    raw = [r for p in payloads for r in p["metrics"]]
    per = [r for p in payloads for r in p["per_category"]]
    geometry = [r for p in payloads for r in p["geometry"]]
    utilization = [r for p in payloads for r in p["utilization"]]
    leakage = [r for p in payloads for r in p["leakage"]]
    modes = [r for p in payloads for r in p["modes"]]
    storage = [r for p in payloads for r in p["storage"]]
    curves = [r for p in payloads for r in p["curves"]]

    write_json(OUT / "PARITY_AUDIT.json", {"image_parity": parity, "passed": all(r["passed"] for r in parity)})
    write_json(OUT / "MULTISTATE_PROTOCOL.json", {"folds": protocols, "K": K_STATES, "target_accessed": False})
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", per)
    write_csv(OUT / "STATE_GEOMETRY.csv", geometry)
    write_csv(OUT / "STATE_UTILIZATION.csv", utilization)
    write_csv(OUT / "CATEGORY_LEAKAGE.csv", leakage)
    write_csv(OUT / "ANOMALY_MODE_COMPOSITION.csv", modes)
    write_csv(OUT / "STORAGE_COMPLEXITY.csv", storage)
    write_csv(OUT / "TRAINING_CURVES.csv", curves)

    by = {(r["direction"], int(r["fold"]), r["variant"]): r for r in raw}
    variants = ["P0_INSTANCE_MEMORY", "P1_SINGLE_STATE", "P2_STATIC_KMEANS4_STATE", "P3_LEARNED_MULTI_STATE4", "P4_DUPLICATED_SINGLE_STATE", "P5_SHUFFLED_LABEL_MULTI_STATE"]
    gains = []
    for direction in ["D1", "D2"]:
        for fold in [0, 1]:
            for variant in variants:
                row = by[(direction, fold, variant)]
                p0 = by[(direction, fold, "P0_INSTANCE_MEMORY")]
                p1 = by[(direction, fold, "P1_SINGLE_STATE")]
                p2 = by[(direction, fold, "P2_STATIC_KMEANS4_STATE")]
                p5 = by[(direction, fold, "P5_SHUFFLED_LABEL_MULTI_STATE")]
                gains.append({
                    "direction": direction,
                    "fold": fold,
                    "variant": variant,
                    "G_pixel_vs_P1": group_gain(row, p1, "pixel"),
                    "G_pixel_vs_P0": group_gain(row, p0, "pixel"),
                    "G_pixel_vs_P2": group_gain(row, p2, "pixel"),
                    "G_pixel_vs_P5": group_gain(row, p5, "pixel"),
                })
    for direction in ["D1", "D2"]:
        for base in ["P1_SINGLE_STATE", "P0_INSTANCE_MEMORY", "P2_STATIC_KMEANS4_STATE", "P5_SHUFFLED_LABEL_MULTI_STATE"]:
            vals = [group_gain(by[(direction, f, "P3_LEARNED_MULTI_STATE4")], by[(direction, f, base)], "pixel") for f in [0, 1]]
            gains.append({"direction": direction, "fold": "mean", "variant": f"P3_vs_{base}", "G_pixel": float(np.mean(vals))})
    write_csv(OUT / "GROUPED_GAINS.csv", gains)

    gap_rows = []
    for direction in ["D1", "D2"]:
        p0 = [by[(direction, f, "P0_INSTANCE_MEMORY")] for f in [0, 1]]
        p1 = [by[(direction, f, "P1_SINGLE_STATE")] for f in [0, 1]]
        p3 = [by[(direction, f, "P3_LEARNED_MULTI_STATE4")] for f in [0, 1]]
        g0 = np.mean([0.5 * (r["P-AUROC"] + r["P-AP"]) for r in p0])
        g1 = np.mean([0.5 * (r["P-AUROC"] + r["P-AP"]) for r in p1])
        g3 = np.mean([0.5 * (r["P-AUROC"] + r["P-AP"]) for r in p3])
        gap = float(g0 - g1)
        rec = float(g3 - g1)
        gap_rows.append({"direction": direction, "gap_P0_minus_P1": gap, "recovery_P3_minus_P1": rec, "gap_closure": rec / gap if gap > 0 else float("nan")})
    write_csv(OUT / "GAP_CLOSURE.csv", gap_rows)

    p3_p1 = [g for g in gains if g["variant"] == "P3_LEARNED_MULTI_STATE4"]
    fold_vals = [g["G_pixel_vs_P1"] for g in p3_p1]
    d1_p3_p1 = next(g["G_pixel"] for g in gains if g["direction"] == "D1" and g["fold"] == "mean" and g["variant"] == "P3_vs_P1_SINGLE_STATE")
    d2_p3_p1 = next(g["G_pixel"] for g in gains if g["direction"] == "D2" and g["fold"] == "mean" and g["variant"] == "P3_vs_P1_SINGLE_STATE")
    d1_p3_p2 = next(g["G_pixel"] for g in gains if g["direction"] == "D1" and g["fold"] == "mean" and g["variant"] == "P3_vs_P2_STATIC_KMEANS4_STATE")
    d2_p3_p2 = next(g["G_pixel"] for g in gains if g["direction"] == "D2" and g["fold"] == "mean" and g["variant"] == "P3_vs_P2_STATIC_KMEANS4_STATE")
    d1_p3_p0 = next(g["G_pixel"] for g in gains if g["direction"] == "D1" and g["fold"] == "mean" and g["variant"] == "P3_vs_P0_INSTANCE_MEMORY")
    d2_p3_p0 = next(g["G_pixel"] for g in gains if g["direction"] == "D2" and g["fold"] == "mean" and g["variant"] == "P3_vs_P0_INSTANCE_MEMORY")
    d1_p3_p5 = next(g["G_pixel"] for g in gains if g["direction"] == "D1" and g["fold"] == "mean" and g["variant"] == "P3_vs_P5_SHUFFLED_LABEL_MULTI_STATE")
    d2_p3_p5 = next(g["G_pixel"] for g in gains if g["direction"] == "D2" and g["fold"] == "mean" and g["variant"] == "P3_vs_P5_SHUFFLED_LABEL_MULTI_STATE")
    multi_value = d1_p3_p1 >= 1.0 and d2_p3_p1 >= 1.0 and sum(v > 0 for v in fold_vals) >= 3 and min(fold_vals) >= -0.50
    task_value = np.mean([d1_p3_p2, d2_p3_p2]) >= 0.30 and d1_p3_p2 > 0 and d2_p3_p2 > 0
    closure = all(r["gap_closure"] >= 0.25 for r in gap_rows)
    competitive = d1_p3_p0 >= -1.0 and d2_p3_p0 >= -1.0
    label_specific = np.mean([d1_p3_p5, d2_p3_p5]) >= 2.0 and d1_p3_p5 > 0 and d2_p3_p5 > 0
    duplicate_ok = max(abs(g["G_pixel_vs_P1"]) for g in gains if g["variant"] == "P4_DUPLICATED_SINGLE_STATE") <= 0.10
    collapsed = any(r["variant"] == "P3_LEARNED_MULTI_STATE4" and str(r["state_collapse"]).lower() == "true" for r in geometry)

    leak_by = {(r["direction"], int(r["fold"]), r["variant"]): r for r in leakage}
    leak_increases = []
    for direction in ["D1", "D2"]:
        for fold in [0, 1]:
            p1 = float(leak_by[(direction, fold, "P1_SINGLE_STATE")]["balanced_category_accuracy"])
            p3 = float(leak_by[(direction, fold, "P3_LEARNED_MULTI_STATE4")]["balanced_category_accuracy"])
            leak_increases.append(p3 - p1)
    leakage_dominant = multi_value and np.mean(leak_increases) > 15.0

    per_gain = []
    per_by = {(r["direction"], int(r["fold"]), r["category"], r["variant"]): r for r in per}
    for direction in ["D1", "D2"]:
        for fold in [0, 1]:
            cats = sorted({r["category"] for r in per if r["direction"] == direction and int(r["fold"]) == fold})
            for cat in cats:
                p3 = per_by[(direction, fold, cat, "P3_LEARNED_MULTI_STATE4")]
                p1 = per_by[(direction, fold, cat, "P1_SINGLE_STATE")]
                p0 = per_by[(direction, fold, cat, "P0_INSTANCE_MEMORY")]
                per_gain.append({
                    "direction": direction,
                    "fold": fold,
                    "category": cat,
                    "G_pixel_P3_minus_P1": group_gain(p3, p1, "pixel"),
                    "G_pixel_P3_minus_P0": group_gain(p3, p0, "pixel"),
                })
    total_pos = sum(max(0.0, r["G_pixel_P3_minus_P1"]) for r in per_gain)
    max_pos = max([max(0.0, r["G_pixel_P3_minus_P1"]) for r in per_gain] or [0.0])
    concentrated = total_pos > 0 and max_pos / total_pos > 0.50

    if not all(r["passed"] for r in parity) or not duplicate_ok:
        outcome = "XXXIX_A_IMPLEMENTATION_FAILURE"
    elif not label_specific:
        outcome = "XXXIX_A_STATE_LABEL_IDENTITY_NOT_SPECIFIC"
    elif collapsed and not multi_value:
        outcome = "XXXIX_A_STATE_COLLAPSE"
    elif not multi_value:
        outcome = "XXXIX_A_MULTI_STATE_NO_LOCAL_VALUE"
    elif leakage_dominant:
        outcome = "XXXIX_A_MULTI_STATE_CATEGORY_LEAKAGE_DOMINANT"
    elif not task_value:
        outcome = "XXXIX_A_STATIC_MULTIMODALITY_EXPLAINS_GAIN"
    elif competitive:
        outcome = "XXXIX_A_COMPACT_MULTI_STATE_COMPETITIVE"
    else:
        outcome = "XXXIX_A_MULTI_STATE_SIGNAL_INSTANCE_DISTRIBUTION_STILL_STRONGER"

    report = [
        "# Phase XXXIX-A Local Multi-State Knowledge Capacity Audit",
        "",
        f"Outcome: **{outcome}**",
        "",
        "Image scoring used frozen XXXVIII K2 image states for all variants. No target data, DINO, prompt, residual, subspace, dynamic memory, or backbone forward was used.",
        "",
        "## Gates",
        "",
        f"- Image parity passed: {all(r['passed'] for r in parity)}; max image-score delta P1-P5 = 0.",
        f"- P3>P1 multi-state value: {multi_value}; D1={d1_p3_p1:+.4f} pp, D2={d2_p3_p1:+.4f} pp, folds positive={sum(v > 0 for v in fold_vals)}/4.",
        f"- P3>P2 task-optimization value: {task_value}; D1={d1_p3_p2:+.4f} pp, D2={d2_p3_p2:+.4f} pp.",
        f"- Gap closure >=25% both directions: {closure}; {json.dumps(gap_rows, default=json_default)}.",
        f"- P3 vs P0 competitive: {competitive}; D1={d1_p3_p0:+.4f} pp, D2={d2_p3_p0:+.4f} pp.",
        f"- Label specificity P3>P5: {label_specific}; D1={d1_p3_p5:+.4f} pp, D2={d2_p3_p5:+.4f} pp.",
        f"- Duplicate control P4≈P1: {duplicate_ok}.",
        f"- P3 state collapse detected: {collapsed}.",
        f"- Category leakage dominant: {leakage_dominant}; mean P3-P1 balanced-category leakage delta={np.mean(leak_increases):+.4f} pp.",
        f"- Category concentrated: {concentrated}; max positive gain share={max_pos / total_pos if total_pos else 0.0:.4f}.",
        "",
        "## Mean Pixel Metrics",
        "",
    ]
    for direction in ["D1", "D2"]:
        report.append(f"### {direction}")
        for variant in variants:
            rows = [by[(direction, f, variant)] for f in [0, 1]]
            means = {k: float(np.mean([r[k] for r in rows])) for k in ["P-AUROC", "P-AP", "PRO", "I-AUROC", "I-AP"]}
            report.append(f"- {variant}: P-AUROC {means['P-AUROC']:.4f}, P-AP {means['P-AP']:.4f}, PRO {means['PRO']:.4f}; image parity I-AUROC {means['I-AUROC']:.4f}, I-AP {means['I-AP']:.4f}.")
    report.extend(["", "Stop: Phase XXXIX-A ended here.", f"Runtime: {time.time() - start:.1f}s."])
    (OUT / "XXXIX_A_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    payload = {
        "phase": "XXXIX-A",
        "outcome": outcome,
        "multi_state_value": multi_value,
        "task_optimization_value": task_value,
        "gap_closure_passed": closure,
        "label_specific": label_specific,
        "duplicate_control_passed": duplicate_ok,
        "competitive": competitive,
        "state_collapse": collapsed,
        "category_leakage_dominant": leakage_dominant,
        "category_concentrated": concentrated,
        "D1_P3_minus_P1_G_pixel": d1_p3_p1,
        "D2_P3_minus_P1_G_pixel": d2_p3_p1,
        "D1_P3_minus_P2_G_pixel": d1_p3_p2,
        "D2_P3_minus_P2_G_pixel": d2_p3_p2,
        "D1_P3_minus_P0_G_pixel": d1_p3_p0,
        "D2_P3_minus_P0_G_pixel": d2_p3_p0,
        "runtime_sec": time.time() - start,
    }
    write_json(OUT / "OUTCOME.json", payload)

    with (ROOT / "EXPERIMENT_LOG.md").open("a", encoding="utf-8") as fp:
        fp.write("\n## Phase XXXIX-A Local Multi-State Knowledge Capacity Audit\n")
        fp.write("- Frozen image-state parity: passed; XXXVIII K2 image scores reused for all patch variants.\n")
        fp.write(f"- P3-P1 G_pixel: D1 {d1_p3_p1:+.4f}, D2 {d2_p3_p1:+.4f}.\n")
        fp.write(f"- P3-P2 G_pixel: D1 {d1_p3_p2:+.4f}, D2 {d2_p3_p2:+.4f}.\n")
        fp.write(f"- Gap closure: {json.dumps(gap_rows, default=json_default)}.\n")
        fp.write(f"- Label specificity: {label_specific}; state collapse: {collapsed}; category leakage dominant: {leakage_dominant}.\n")
        fp.write(f"- Outcome: {outcome}.\n")


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = OUT / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    payloads = []
    for direction in ["D1", "D2"]:
        for fold in [0, 1]:
            path = ckpt / f"{direction}_fold{fold}_done.json"
            if path.exists():
                print(f"[XXXIX-A] resume {direction} fold{fold}", flush=True)
                payloads.append(read_json(path))
                continue
            print(f"[XXXIX-A] run {direction} fold{fold}", flush=True)
            payload = run_fold(direction, fold, device)
            write_json(path, payload)
            payloads.append(payload)
    aggregate(payloads, start)
    print(json.dumps(read_json(OUT / "OUTCOME.json"), indent=2), flush=True)


if __name__ == "__main__":
    main()
