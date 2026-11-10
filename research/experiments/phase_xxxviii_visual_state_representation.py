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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_xa_checkpoint_source_readout import all_indices_for_categories, direction_cfg
from experiments.mrad_audit_utils import fast_cal_pro_score
from experiments.phase_xa_smoke import load_dataset, write_csv
from phase_xa.source_folds import deterministic_source_folds


OUT = Path("results/phase_xxxviii_visual_state_representation")
FEATURE_CACHE = Path("results/phase_xxiii_carrier_audit/feature_cache")
SIDE = 37
IMAGE_SIZE = 518
SIGMA = 4
EPOCHS = 20
LR = 1e-3
SEED = 0
PATCH_MEMORY_CAP_PER_CLASS_PER_CATEGORY = 1024
PATCH_TRAIN_SAMPLES_PER_CLASS_PER_CATEGORY = 512
PATCH_LEAKAGE_SAMPLES_PER_CATEGORY = 256
K0_CHUNK = 8192
K0_PATCHES_PER_IMAGE_CLASS = 4


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
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    return float(roc_auc_score(y, s) * 100.0) if np.unique(y).size == 2 else float("nan")


def safe_ap(y, s):
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    return float(average_precision_score(y, s) * 100.0) if np.unique(y).size == 2 else float("nan")


def norm_np(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def softmax2(a, b):
    m = np.maximum(a, b)
    ea = np.exp(a - m)
    eb = np.exp(b - m)
    return eb / np.maximum(ea + eb, 1e-12)


def patch_labels(item):
    mask = item["img_mask"].float()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(0)
    small = F.interpolate(mask, size=(SIDE, SIDE), mode="bilinear", align_corners=False).reshape(-1)
    return (small > 0.5).numpy().astype(np.int64)


def upsample_map(map37):
    x = torch.tensor(map37, dtype=torch.float32).view(1, 1, SIDE, SIDE)
    full = F.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False).squeeze().numpy()
    return gaussian_filter(full.astype(np.float32), sigma=SIGMA).astype(np.float32)


def load_fold(direction, fold):
    dataset_name, data_path, _ = direction_cfg(direction)
    dataset = load_dataset(dataset_name, data_path)
    rotation = deterministic_source_folds(dataset_name)["rotations"][fold]
    train_indices = all_indices_for_categories(dataset, set(rotation["train_categories"]))
    held_indices = all_indices_for_categories(dataset, set(rotation["val_categories"]))
    return dataset_name, dataset, rotation, train_indices, held_indices


def labels_and_rows(dataset, indices):
    labels, cats, rows = [], [], []
    for pos, idx in enumerate(indices):
        item = dataset[idx]
        labels.append(int(item["anomaly"]))
        cats.append(str(item["cls_name"]))
        rows.append({"position": pos, "index": int(idx), "category": str(item["cls_name"]), "label": int(item["anomaly"])})
    return np.asarray(labels, dtype=np.int64), np.asarray(cats), rows


def class_category_patch_sample(dataset, indices, patches, cap_per_class_per_category, seed):
    rng = np.random.default_rng(seed)
    by_key = defaultdict(list)
    for pos, idx in enumerate(indices):
        item = dataset[idx]
        labels = patch_labels(item)
        patch = np.asarray(patches[pos], dtype=np.float32)
        for cls in [0, 1]:
            ids = np.flatnonzero(labels == cls)
            if ids.size == 0:
                continue
            key = (str(item["cls_name"]), cls)
            take_room = cap_per_class_per_category - len(by_key[key])
            if take_room <= 0:
                continue
            if ids.size > take_room:
                ids = rng.choice(ids, size=take_room, replace=False)
            by_key[key].append(norm_np(patch[ids]))
    feats, labels, cats = [], [], []
    for (cat, cls), parts in sorted(by_key.items()):
        if not parts:
            continue
        arr = np.concatenate(parts, axis=0).astype(np.float32)
        feats.append(arr)
        labels.extend([cls] * arr.shape[0])
        cats.extend([cat] * arr.shape[0])
    if not feats:
        return np.zeros((0, patches.shape[-1]), dtype=np.float32), np.zeros((0,), dtype=np.int64), np.asarray([])
    return np.concatenate(feats, axis=0), np.asarray(labels, dtype=np.int64), np.asarray(cats)


def fps_indices(tokens, k):
    x = norm_np(tokens)
    n = x.shape[0]
    if n <= k:
        return np.arange(n, dtype=np.int64)
    centroid = norm_np(x.mean(axis=0, keepdims=True))[0]
    first = int(np.argmin(x @ centroid))
    selected = [first]
    max_sim = x @ x[first]
    for _ in range(1, k):
        idx = int(np.argmin(max_sim))
        selected.append(idx)
        max_sim = np.maximum(max_sim, x @ x[idx])
    return np.asarray(selected, dtype=np.int64)


def k0_fps_patch_memory(dataset, indices, patches):
    feats, labels, cats = [], [], []
    for pos, idx in enumerate(indices):
        item = dataset[idx]
        patch = np.asarray(patches[pos], dtype=np.float32)
        plabel = patch_labels(item)
        if int(item["anomaly"]) == 0:
            ids = fps_indices(patch, K0_PATCHES_PER_IMAGE_CLASS)
            feats.append(norm_np(patch[ids]))
            labels.extend([0] * len(ids))
            cats.extend([str(item["cls_name"])] * len(ids))
            continue
        for cls in [0, 1]:
            pool = np.flatnonzero(plabel == cls)
            if pool.size == 0:
                continue
            local = fps_indices(patch[pool], min(K0_PATCHES_PER_IMAGE_CLASS, pool.size))
            ids = pool[local]
            feats.append(norm_np(patch[ids]))
            labels.extend([cls] * len(ids))
            cats.extend([str(item["cls_name"])] * len(ids))
    if not feats:
        return np.zeros((0, patches.shape[-1]), dtype=np.float32), np.zeros((0,), dtype=np.int64), np.asarray([])
    return np.concatenate(feats, axis=0).astype(np.float32), np.asarray(labels, dtype=np.int64), np.asarray(cats)


def centroids(x, y):
    states = []
    for cls in [0, 1]:
        vals = x[y == cls]
        if vals.size == 0:
            vals = x
        states.append(norm_np(vals.mean(axis=0, keepdims=True))[0])
    return np.stack(states).astype(np.float32)


def instance_memory_scores(queries, keys, labels, chunk=K0_CHUNK):
    q = norm_np(queries)
    k = norm_np(keys)
    lab = np.asarray(labels, dtype=np.int64)
    out = np.zeros((q.shape[0],), dtype=np.float32)
    for start in range(0, q.shape[0], chunk):
        sims = q[start : start + chunk] @ k.T
        m = sims.max(axis=1, keepdims=True)
        w = np.exp(sims - m)
        num = (w * (lab[None, :] == 1)).sum(axis=1)
        den = w.sum(axis=1)
        out[start : start + q[start : start + chunk].shape[0]] = (num / np.maximum(den, 1e-12)).astype(np.float32)
    return out


def state_scores(queries, states):
    q = norm_np(queries)
    s = norm_np(states)
    logits = q @ s.T
    return softmax2(logits[:, 0], logits[:, 1]).astype(np.float32), logits.astype(np.float32)


def train_state_vectors(img_x, img_y, img_cat, patch_x, patch_y, patch_cat, init_img, init_patch, shuffled=False, device="cpu"):
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    img_y_train = np.asarray(img_y, dtype=np.int64).copy()
    patch_y_train = np.asarray(patch_y, dtype=np.int64).copy()
    if shuffled:
        img_y_train = rng.permutation(img_y_train)
        patch_y_train = rng.permutation(patch_y_train)

    img_state = torch.nn.Parameter(torch.tensor(init_img, dtype=torch.float32, device=device))
    patch_state = torch.nn.Parameter(torch.tensor(init_patch, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW([img_state, patch_state], lr=LR, weight_decay=0.0)
    img_x_t = torch.tensor(norm_np(img_x), dtype=torch.float32, device=device)
    img_y_t = torch.tensor(img_y_train, dtype=torch.long, device=device)
    patch_x_np = norm_np(patch_x)
    patch_y_np = patch_y_train

    img_counts = np.bincount(img_y_train, minlength=2).astype(np.float32)
    img_weights = torch.tensor((img_counts.sum() / np.maximum(img_counts, 1.0)) / 2.0, dtype=torch.float32, device=device)
    cats = sorted(set(img_cat.tolist()) | set(patch_cat.tolist()))
    curve = []
    init_img_t = torch.tensor(norm_np(init_img), dtype=torch.float32, device=device)
    init_patch_t = torch.tensor(norm_np(init_patch), dtype=torch.float32, device=device)

    for epoch in range(1, EPOCHS + 1):
        opt.zero_grad()
        s_img = F.normalize(img_state, dim=-1)
        s_patch = F.normalize(patch_state, dim=-1)
        img_logits = img_x_t @ s_img.T
        img_loss = F.cross_entropy(img_logits, img_y_t, weight=img_weights)

        p_parts, y_parts = [], []
        for cat in cats:
            cidx = np.flatnonzero(patch_cat == cat)
            if cidx.size == 0:
                continue
            nidx = cidx[patch_y_np[cidx] == 0]
            aidx = cidx[patch_y_np[cidx] == 1]
            if nidx.size == 0 or aidx.size == 0:
                continue
            take = min(nidx.size, aidx.size, 128)
            p_parts.append(rng.choice(nidx, size=take, replace=False))
            p_parts.append(rng.choice(aidx, size=take, replace=False))
        if p_parts:
            pidx = np.concatenate(p_parts)
        else:
            nidx = np.flatnonzero(patch_y_np == 0)
            aidx = np.flatnonzero(patch_y_np == 1)
            take = min(len(nidx), len(aidx), 2048)
            pidx = np.concatenate([rng.choice(nidx, size=take, replace=False), rng.choice(aidx, size=take, replace=False)])
        px = torch.tensor(patch_x_np[pidx], dtype=torch.float32, device=device)
        py = torch.tensor(patch_y_np[pidx], dtype=torch.long, device=device)
        patch_logits = px @ s_patch.T
        patch_loss = F.cross_entropy(patch_logits, py)
        loss = img_loss + patch_loss
        loss.backward()
        opt.step()

        with torch.no_grad():
            img_disp = torch.linalg.norm(F.normalize(img_state, dim=-1) - init_img_t, dim=-1).mean().item()
            patch_disp = torch.linalg.norm(F.normalize(patch_state, dim=-1) - init_patch_t, dim=-1).mean().item()
        curve.append({
            "epoch": epoch,
            "variant": "K3_SHUFFLED_LABEL_STATE" if shuffled else "K2_LEARNED_STATE",
            "loss": float(loss.detach().cpu()),
            "image_loss": float(img_loss.detach().cpu()),
            "patch_loss": float(patch_loss.detach().cpu()),
            "image_state_displacement": img_disp,
            "patch_state_displacement": patch_disp,
        })
    return norm_np(img_state.detach().cpu().numpy()), norm_np(patch_state.detach().cpu().numpy()), curve


def evaluate_variant(dataset, held_indices, held_cls, held_patch, image_scores, patch_maps):
    y_img, cats, _ = labels_and_rows(dataset, held_indices)
    by_cat = defaultdict(lambda: {"iy": [], "is": [], "py": [], "ps": []})
    y_pix_all, s_pix_all = [], []
    for pos, idx in enumerate(held_indices):
        item = dataset[idx]
        cat = str(item["cls_name"])
        by_cat[cat]["iy"].append(int(item["anomaly"]))
        by_cat[cat]["is"].append(float(image_scores[pos]))
        mask = (item["img_mask"].detach().cpu().numpy() > 0.5).astype(np.uint8).reshape(-1)
        full = upsample_map(patch_maps[pos]).reshape(-1)
        by_cat[cat]["py"].append(mask)
        by_cat[cat]["ps"].append(full)
        y_pix_all.append(mask)
        s_pix_all.append(full)

    per, ia, iap, pa, pap, pro = {}, [], [], [], [], []
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
        for key, dst in [("I-AUROC", ia), ("I-AP", iap), ("P-AUROC", pa), ("P-AP", pap), ("PRO", pro)]:
            if np.isfinite(row[key]):
                dst.append(row[key])
    return {
        "mean": {
            "I-AUROC": float(np.mean(ia)) if ia else float("nan"),
            "I-AP": float(np.mean(iap)) if iap else float("nan"),
            "P-AUROC": float(np.mean(pa)) if pa else float("nan"),
            "P-AP": float(np.mean(pap)) if pap else float("nan"),
            "PRO": float(np.mean(pro)) if pro else float("nan"),
        },
        "per_category": per,
    }


def score_patch_states_all(held_patch, patch_states):
    maps = np.zeros((held_patch.shape[0], SIDE, SIDE), dtype=np.float16)
    for i in range(held_patch.shape[0]):
        scores, _ = state_scores(np.asarray(held_patch[i], dtype=np.float32), patch_states)
        maps[i] = scores.reshape(SIDE, SIDE).astype(np.float16)
    return maps


def score_patch_instance_all(held_patch, mem_x, mem_y):
    maps = np.zeros((held_patch.shape[0], SIDE, SIDE), dtype=np.float16)
    for i in range(held_patch.shape[0]):
        scores = instance_memory_scores(np.asarray(held_patch[i], dtype=np.float32), mem_x, mem_y)
        maps[i] = scores.reshape(SIDE, SIDE).astype(np.float16)
    return maps


def category_cv_scores(x, cats):
    cats = np.asarray(cats)
    if len(set(cats.tolist())) < 2 or len(cats) < 10:
        return float("nan"), float("nan")
    counts = min(np.bincount(np.searchsorted(sorted(set(cats.tolist())), cats)).min(), 5)
    if counts < 2:
        return float("nan"), float("nan")
    cv = StratifiedKFold(n_splits=int(counts), shuffle=True, random_state=SEED)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, class_weight="balanced", random_state=SEED, solver="lbfgs"))
    acc = cross_val_score(clf, x, cats, cv=cv, scoring="accuracy")
    bacc = cross_val_score(clf, x, cats, cv=cv, scoring="balanced_accuracy")
    return float(np.mean(acc) * 100.0), float(np.mean(bacc) * 100.0)


def leakage_rows(train_cls, train_cat, patch_x, patch_cat, states_by_variant):
    rows = []
    raw_acc, raw_bacc = category_cv_scores(train_cls, train_cat)
    rows.append({"branch": "image", "variant": "RAW_FEATURE", "category_accuracy": raw_acc, "balanced_category_accuracy": raw_bacc, "dimensionality": int(train_cls.shape[1])})
    rawp_acc, rawp_bacc = category_cv_scores(patch_x, patch_cat)
    rows.append({"branch": "patch", "variant": "RAW_FEATURE", "category_accuracy": rawp_acc, "balanced_category_accuracy": rawp_bacc, "dimensionality": int(patch_x.shape[1])})
    for name, states in states_by_variant.items():
        img_scores, img_logits = state_scores(train_cls, states["image"])
        z_img = img_logits
        acc, bacc = category_cv_scores(z_img, train_cat)
        rows.append({"branch": "image", "variant": name, "category_accuracy": acc, "balanced_category_accuracy": bacc, "dimensionality": 2})
        p_scores, p_logits = state_scores(patch_x, states["patch"])
        acc, bacc = category_cv_scores(p_logits, patch_cat)
        rows.append({"branch": "patch", "variant": name, "category_accuracy": acc, "balanced_category_accuracy": bacc, "dimensionality": 2})
    return rows


def geometry_rows(states_by_variant, init_img, init_patch):
    rows = []
    for name, states in states_by_variant.items():
        for branch, init in [("image", init_img), ("patch", init_patch)]:
            s = norm_np(states[branch])
            cos_na = float(s[0] @ s[1])
            angle = float(np.degrees(np.arccos(np.clip(cos_na, -1, 1))))
            rows.append({
                "variant": name,
                "branch": branch,
                "cos_normal_anomaly": cos_na,
                "angular_separation_deg": angle,
                "normal_state_norm_before_normalization": float(np.linalg.norm(states[branch][0])),
                "anomaly_state_norm_before_normalization": float(np.linalg.norm(states[branch][1])),
                "mean_displacement_from_centroid": float(np.linalg.norm(norm_np(states[branch]) - norm_np(init), axis=1).mean()),
                "cos_to_centroid_normal": float(norm_np(states[branch])[0] @ norm_np(init)[0]),
                "cos_to_centroid_anomaly": float(norm_np(states[branch])[1] @ norm_np(init)[1]),
            })
    return rows


def group_gain(candidate, base, group):
    if group == "image":
        keys = ["I-AUROC", "I-AP"]
    else:
        keys = ["P-AUROC", "P-AP"]
    vals = [candidate[k] - base[k] for k in keys if np.isfinite(candidate[k]) and np.isfinite(base[k])]
    return float(np.mean(vals)) if vals else float("nan")


def run_fold(direction, fold, device):
    fold_start = time.time()
    dataset_name, dataset, rotation, train_indices, held_indices = load_fold(direction, fold)
    cache = np.load(FEATURE_CACHE / f"{direction}_fold{fold}_clip.npz", allow_pickle=True)
    train_cls = norm_np(cache["train_cls"].astype(np.float32))
    held_cls = norm_np(cache["held_cls"].astype(np.float32))
    train_patch = cache["train_patch"]
    held_patch = cache["held_patch"]
    y_train, cat_train, train_rows = labels_and_rows(dataset, train_indices)
    y_held, cat_held, held_rows = labels_and_rows(dataset, held_indices)

    patch_train_x, patch_train_y, patch_train_cat = class_category_patch_sample(
        dataset, train_indices, train_patch, PATCH_TRAIN_SAMPLES_PER_CLASS_PER_CATEGORY, seed=SEED
    )
    patch_mem_x, patch_mem_y, patch_mem_cat = k0_fps_patch_memory(dataset, train_indices, train_patch)
    init_img = centroids(train_cls, y_train)
    init_patch = centroids(patch_train_x, patch_train_y)
    img_k2, patch_k2, curve_k2 = train_state_vectors(
        train_cls, y_train, cat_train, patch_train_x, patch_train_y, patch_train_cat, init_img, init_patch, shuffled=False, device=device
    )
    img_k3, patch_k3, curve_k3 = train_state_vectors(
        train_cls, y_train, cat_train, patch_train_x, patch_train_y, patch_train_cat, init_img, init_patch, shuffled=True, device=device
    )

    variants = {
        "K1_CENTROID_STATE": {"image": init_img, "patch": init_patch},
        "K2_LEARNED_VISUAL_STATE": {"image": img_k2, "patch": patch_k2},
        "K3_SHUFFLED_LABEL_STATE": {"image": img_k3, "patch": patch_k3},
    }
    metrics_rows, per_rows, utilities = [], [], []

    print(f"[XXXVIII-A] {direction} fold{fold} score K0 image", flush=True)
    k0_img = instance_memory_scores(held_cls, train_cls, y_train)
    print(f"[XXXVIII-A] {direction} fold{fold} score K0 patch", flush=True)
    k0_maps = score_patch_instance_all(held_patch, patch_mem_x, patch_mem_y)
    result = evaluate_variant(dataset, held_indices, held_cls, held_patch, k0_img, k0_maps)
    for k, v in result["mean"].items():
        pass
    metrics_rows.append({"direction": direction, "fold": fold, "dataset": dataset_name, "variant": "K0_INSTANCE_MEMORY", **result["mean"]})
    for cat, vals in result["per_category"].items():
        per_rows.append({"direction": direction, "fold": fold, "category": cat, "variant": "K0_INSTANCE_MEMORY", **vals})

    for name, states in variants.items():
        print(f"[XXXVIII-A] {direction} fold{fold} score {name}", flush=True)
        img_scores, _ = state_scores(held_cls, states["image"])
        maps = score_patch_states_all(held_patch, states["patch"])
        result = evaluate_variant(dataset, held_indices, held_cls, held_patch, img_scores, maps)
        metrics_rows.append({"direction": direction, "fold": fold, "dataset": dataset_name, "variant": name, **result["mean"]})
        for cat, vals in result["per_category"].items():
            per_rows.append({"direction": direction, "fold": fold, "category": cat, "variant": name, **vals})

    leak_patch_x, leak_patch_y, leak_patch_cat = class_category_patch_sample(
        dataset, train_indices, train_patch, PATCH_LEAKAGE_SAMPLES_PER_CATEGORY, seed=SEED + 1
    )
    leak_rows = leakage_rows(train_cls, cat_train, leak_patch_x, leak_patch_cat, variants)
    for row in leak_rows:
        row.update({"direction": direction, "fold": fold})
    geom_rows = geometry_rows(variants, init_img, init_patch)
    for row in geom_rows:
        row.update({"direction": direction, "fold": fold})

    train_curve = []
    for row in curve_k2 + curve_k3:
        row.update({"direction": direction, "fold": fold})
        train_curve.append(row)

    state_dir = OUT / "states"
    state_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        state_dir / f"{direction}_fold{fold}_states.npz",
        K1_img=init_img,
        K1_patch=init_patch,
        K2_img=img_k2,
        K2_patch=patch_k2,
        K3_img=img_k3,
        K3_patch=patch_k3,
    )
    audit = {
        "direction": direction,
        "fold": fold,
        "dataset": dataset_name,
        "feature_layer": "official MRAD-CLIP frozen image CLS and normalized patch feature cache",
        "image_feature_dim": int(train_cls.shape[1]),
        "patch_feature_dim": int(train_patch.shape[-1]),
        "patch_tokens": int(train_patch.shape[1]),
        "normalization": "L2-normalized before centroid, state training, and cosine scoring",
        "train_categories": rotation["train_categories"],
        "heldout_categories": rotation["val_categories"],
        "train_images": int(len(train_indices)),
        "heldout_images": int(len(held_indices)),
        "train_image_normal": int((y_train == 0).sum()),
        "train_image_anomaly": int((y_train == 1).sum()),
        "heldout_image_normal": int((y_held == 0).sum()),
        "heldout_image_anomaly": int((y_held == 1).sum()),
        "patch_state_train_samples": int(patch_train_x.shape[0]),
        "patch_state_train_normal": int((patch_train_y == 0).sum()),
        "patch_state_train_anomaly": int((patch_train_y == 1).sum()),
        "patch_memory_vectors": int(patch_mem_x.shape[0]),
        "per_category_image_counts": {
            cat: {
                "train_normal": int(sum(1 for r in train_rows if r["category"] == cat and r["label"] == 0)),
                "train_anomaly": int(sum(1 for r in train_rows if r["category"] == cat and r["label"] == 1)),
                "held_normal": int(sum(1 for r in held_rows if r["category"] == cat and r["label"] == 0)),
                "held_anomaly": int(sum(1 for r in held_rows if r["category"] == cat and r["label"] == 1)),
            }
            for cat in sorted(set(cat_train.tolist()) | set(cat_held.tolist()))
        },
        "runtime_sec": time.time() - fold_start,
    }
    return {
        "audit": audit,
        "metrics": metrics_rows,
        "per_category": per_rows,
        "leakage": leak_rows,
        "geometry": geom_rows,
        "curves": train_curve,
        "storage": [{
            "direction": direction,
            "fold": fold,
            "variant": "K0_INSTANCE_MEMORY",
            "image_vectors": int(train_cls.shape[0]),
            "patch_vectors": int(patch_mem_x.shape[0]),
            "bytes_float32": int((train_cls.size + patch_mem_x.size) * 4 + (len(y_train) + len(patch_mem_y)) * 4),
            "image_comparisons_per_query": int(train_cls.shape[0]),
            "patch_comparisons_per_query": int(patch_mem_x.shape[0]),
        }, {
            "direction": direction,
            "fold": fold,
            "variant": "K2_LEARNED_VISUAL_STATE",
            "image_vectors": 2,
            "patch_vectors": 2,
            "bytes_float32": int((2 * train_cls.shape[1] + 2 * train_patch.shape[-1]) * 4),
            "image_comparisons_per_query": 2,
            "patch_comparisons_per_query": 2,
        }],
    }


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=json_default), encoding="utf-8")


def aggregate_and_write(all_payloads, start_time):
    audits = [p["audit"] for p in all_payloads]
    raw = [r for p in all_payloads for r in p["metrics"]]
    per = [r for p in all_payloads for r in p["per_category"]]
    leakage = [r for p in all_payloads for r in p["leakage"]]
    geometry = [r for p in all_payloads for r in p["geometry"]]
    curves = [r for p in all_payloads for r in p["curves"]]
    storage = [r for p in all_payloads for r in p["storage"]]

    write_json(OUT / "STATE_FEATURE_AUDIT.json", {"folds": audits, "target_accessed": False})
    write_json(OUT / "TRAINING_PROTOCOL.json", {
        "optimizer": "AdamW",
        "lr": LR,
        "weight_decay": 0,
        "epochs": EPOCHS,
        "seed": SEED,
        "checkpoint_selection": "final_epoch_20_only",
        "image_loss": "balanced 2-class cross entropy with inverse-frequency weights",
        "patch_loss": "category-balanced, class-balanced patch sampling cross entropy",
        "temperature": 1.0,
        "target_accessed": False,
    })
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", per)
    write_csv(OUT / "CATEGORY_LEAKAGE.csv", leakage)
    write_csv(OUT / "STATE_GEOMETRY.csv", geometry)
    write_csv(OUT / "TRAINING_CURVES.csv", curves)

    k0_bytes = defaultdict(float)
    k2_bytes = defaultdict(float)
    for r in storage:
        key = (r["direction"], r["fold"])
        if r["variant"] == "K0_INSTANCE_MEMORY":
            k0_bytes[key] = r["bytes_float32"]
        if r["variant"] == "K2_LEARNED_VISUAL_STATE":
            k2_bytes[key] = r["bytes_float32"]
    for r in storage:
        key = (r["direction"], r["fold"])
        r["compression_ratio_vs_K0"] = float(k0_bytes[key] / k2_bytes[key]) if r["variant"] == "K2_LEARNED_VISUAL_STATE" and k2_bytes[key] else 1.0
    write_csv(OUT / "STORAGE_COMPLEXITY.csv", storage)

    by = {(r["direction"], int(r["fold"]), r["variant"]): r for r in raw}
    gains = []
    for direction in ["D1", "D2"]:
        for fold in [0, 1]:
            for variant in ["K1_CENTROID_STATE", "K2_LEARNED_VISUAL_STATE", "K3_SHUFFLED_LABEL_STATE"]:
                row = by[(direction, fold, variant)]
                k0 = by[(direction, fold, "K0_INSTANCE_MEMORY")]
                k1 = by[(direction, fold, "K1_CENTROID_STATE")]
                k3 = by[(direction, fold, "K3_SHUFFLED_LABEL_STATE")]
                gains.append({
                    "direction": direction,
                    "fold": fold,
                    "variant": variant,
                    "G_image_vs_K1": group_gain(row, k1, "image"),
                    "G_pixel_vs_K1": group_gain(row, k1, "pixel"),
                    "G_image_vs_K0": group_gain(row, k0, "image"),
                    "G_pixel_vs_K0": group_gain(row, k0, "pixel"),
                    "G_image_vs_K3": group_gain(row, k3, "image"),
                    "G_pixel_vs_K3": group_gain(row, k3, "pixel"),
                })
    for direction in ["D1", "D2"]:
        rows = [by[(direction, f, "K2_LEARNED_VISUAL_STATE")] for f in [0, 1]]
        for base_name in ["K1_CENTROID_STATE", "K0_INSTANCE_MEMORY", "K3_SHUFFLED_LABEL_STATE"]:
            base_rows = [by[(direction, f, base_name)] for f in [0, 1]]
            gains.append({
                "direction": direction,
                "fold": "mean",
                "variant": f"K2_vs_{base_name}",
                "G_image": float(np.mean([group_gain(r, b, "image") for r, b in zip(rows, base_rows)])),
                "G_pixel": float(np.mean([group_gain(r, b, "pixel") for r, b in zip(rows, base_rows)])),
            })
    write_csv(OUT / "GROUPED_GAINS.csv", gains)

    anomaly_rows = []
    leak_by = {(r["direction"], int(r["fold"]), r["branch"], r["variant"]): r for r in leakage}
    for r in raw:
        if r["variant"] not in {"K1_CENTROID_STATE", "K2_LEARNED_VISUAL_STATE", "K3_SHUFFLED_LABEL_STATE"}:
            continue
        for branch, acc_key in [("image", "I-AUROC"), ("patch", "P-AUROC")]:
            lk = leak_by.get((r["direction"], int(r["fold"]), branch, r["variant"]), {})
            anomaly_rows.append({
                "direction": r["direction"],
                "fold": r["fold"],
                "variant": r["variant"],
                "branch": branch,
                "anomaly_utility": r[acc_key],
                "category_accuracy": lk.get("category_accuracy", float("nan")),
                "balanced_category_accuracy": lk.get("balanced_category_accuracy", float("nan")),
            })
    write_csv(OUT / "ANOMALY_UTILITY_VS_CATEGORY.csv", anomaly_rows)

    k2_k1 = [g for g in gains if g.get("variant") == "K2_LEARNED_VISUAL_STATE"]
    k2_mean_vs_k1 = [g for g in gains if g.get("variant") == "K2_vs_K1_CENTROID_STATE"]
    k2_mean_vs_k0 = [g for g in gains if g.get("variant") == "K2_vs_K0_INSTANCE_MEMORY"]
    k2_mean_vs_k3 = [g for g in gains if g.get("variant") == "K2_vs_K3_SHUFFLED_LABEL_STATE"]

    d1_img = next(g["G_image"] for g in k2_mean_vs_k1 if g["direction"] == "D1")
    d2_img = next(g["G_image"] for g in k2_mean_vs_k1 if g["direction"] == "D2")
    d1_pix = next(g["G_pixel"] for g in k2_mean_vs_k1 if g["direction"] == "D1")
    d2_pix = next(g["G_pixel"] for g in k2_mean_vs_k1 if g["direction"] == "D2")
    fold_img = [g["G_image_vs_K1"] for g in k2_k1]
    fold_pix = [g["G_pixel_vs_K1"] for g in k2_k1]
    img_signal = d1_img >= 0.30 and d2_img >= 0.30 and sum(x > 0 for x in fold_img) >= 3 and min(fold_img) >= -0.30
    pix_signal = d1_pix >= 0.30 and d2_pix >= 0.30 and sum(x > 0 for x in fold_pix) >= 3 and min(fold_pix) >= -0.30
    state_signal = img_signal or pix_signal
    d1_vs_k3 = next(g for g in k2_mean_vs_k3 if g["direction"] == "D1")
    d2_vs_k3 = next(g for g in k2_mean_vs_k3 if g["direction"] == "D2")
    label_specific = (
        (np.mean([d1_vs_k3["G_image"], d2_vs_k3["G_image"]]) >= 1.0 and d1_vs_k3["G_image"] > 0 and d2_vs_k3["G_image"] > 0)
        or (np.mean([d1_vs_k3["G_pixel"], d2_vs_k3["G_pixel"]]) >= 1.0 and d1_vs_k3["G_pixel"] > 0 and d2_vs_k3["G_pixel"] > 0)
    )
    competitive = all(g["G_image"] >= -0.20 and g["G_pixel"] >= -0.20 for g in k2_mean_vs_k0)
    superior = all(
        ((g["G_image"] >= 0.30 and g["G_pixel"] >= -0.20) or (g["G_pixel"] >= 0.30 and g["G_image"] >= -0.20))
        for g in k2_mean_vs_k0
    )

    per_gain = []
    per_by = {(r["direction"], int(r["fold"]), r["category"], r["variant"]): r for r in per}
    for direction in ["D1", "D2"]:
        for fold in [0, 1]:
            cats = sorted({r["category"] for r in per if r["direction"] == direction and int(r["fold"]) == fold})
            for cat in cats:
                k2 = per_by[(direction, fold, cat, "K2_LEARNED_VISUAL_STATE")]
                k1 = per_by[(direction, fold, cat, "K1_CENTROID_STATE")]
                k0 = per_by[(direction, fold, cat, "K0_INSTANCE_MEMORY")]
                per_gain.append({
                    "direction": direction,
                    "fold": fold,
                    "category": cat,
                    "G_image_K2_minus_K1": group_gain(k2, k1, "image"),
                    "G_pixel_K2_minus_K1": group_gain(k2, k1, "pixel"),
                    "G_image_K2_minus_K0": group_gain(k2, k0, "image"),
                    "G_pixel_K2_minus_K0": group_gain(k2, k0, "pixel"),
                })
    total_pos = sum(max(0.0, r["G_image_K2_minus_K1"]) + max(0.0, r["G_pixel_K2_minus_K1"]) for r in per_gain)
    max_pos = max([max(0.0, r["G_image_K2_minus_K1"]) + max(0.0, r["G_pixel_K2_minus_K1"]) for r in per_gain] or [0.0])
    concentrated = total_pos > 0 and (max_pos / total_pos) > 0.50

    if not state_signal:
        outcome = "XXXVIII_A_STATE_OPTIMIZATION_NO_VALUE"
    elif not label_specific:
        outcome = "XXXVIII_A_STATE_LABEL_IDENTITY_NOT_SPECIFIC"
    elif concentrated:
        outcome = "XXXVIII_A_STATE_EFFECT_CATEGORY_CONCENTRATED"
    elif superior:
        outcome = "XXXVIII_A_COMPACT_STATE_SUPERIOR"
    elif competitive:
        outcome = "XXXVIII_A_COMPACT_STATE_COMPETITIVE"
    else:
        outcome = "XXXVIII_A_STATE_SIGNAL_BUT_INSTANCE_MEMORY_STRONGER"

    report = [
        "# Phase XXXVIII-A Transferable Visual State Representation Audit",
        "",
        f"Outcome: **{outcome}**",
        "",
        "No target data, DINO, prompt fusion, residual memory, subspace memory, adapter, or backbone training was used.",
        "",
        "## Gates",
        "",
        f"- K2>K1 image signal: {img_signal}; D1={d1_img:+.4f} pp, D2={d2_img:+.4f} pp.",
        f"- K2>K1 pixel signal: {pix_signal}; D1={d1_pix:+.4f} pp, D2={d2_pix:+.4f} pp.",
        f"- K2 label specificity vs K3: {label_specific}; D1 image/pixel={d1_vs_k3['G_image']:+.4f}/{d1_vs_k3['G_pixel']:+.4f}, D2 image/pixel={d2_vs_k3['G_image']:+.4f}/{d2_vs_k3['G_pixel']:+.4f}.",
        f"- K2 competitive with K0: {competitive}; superior: {superior}.",
        f"- Category concentrated: {concentrated}; max positive-gain share={max_pos / total_pos if total_pos else 0.0:.4f}.",
        "",
        "## Mean Metrics",
        "",
    ]
    for direction in ["D1", "D2"]:
        report.append(f"### {direction}")
        for variant in ["K0_INSTANCE_MEMORY", "K1_CENTROID_STATE", "K2_LEARNED_VISUAL_STATE", "K3_SHUFFLED_LABEL_STATE"]:
            rows = [by[(direction, f, variant)] for f in [0, 1]]
            means = {k: float(np.mean([r[k] for r in rows])) for k in ["I-AUROC", "I-AP", "P-AUROC", "P-AP", "PRO"]}
            report.append(
                f"- {variant}: I-AUROC {means['I-AUROC']:.4f}, I-AP {means['I-AP']:.4f}, P-AUROC {means['P-AUROC']:.4f}, P-AP {means['P-AP']:.4f}, PRO {means['PRO']:.4f}."
            )
    report.extend([
        "",
        "## Interpretation",
        "",
        "K2 is judged only by fixed final epoch 20. PRO is reported as a secondary descriptive metric and is not used for outcome.",
        f"Runtime: {time.time() - start_time:.1f}s.",
        "",
        "Stop: Phase XXXVIII-A ended here.",
    ])
    (OUT / "XXXVIII_A_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    outcome_payload = {
        "phase": "XXXVIII-A",
        "outcome": outcome,
        "state_signal": state_signal,
        "image_signal": img_signal,
        "pixel_signal": pix_signal,
        "label_specific": label_specific,
        "competitive": competitive,
        "superior": superior,
        "category_concentrated": concentrated,
        "D1_K2_minus_K1_G_image": d1_img,
        "D2_K2_minus_K1_G_image": d2_img,
        "D1_K2_minus_K1_G_pixel": d1_pix,
        "D2_K2_minus_K1_G_pixel": d2_pix,
        "runtime_sec": time.time() - start_time,
    }
    write_json(OUT / "OUTCOME.json", outcome_payload)

    log = ROOT / "EXPERIMENT_LOG.md"
    with log.open("a", encoding="utf-8") as fp:
        fp.write("\n## Phase XXXVIII-A Transferable Visual State Representation Audit\n")
        fp.write(f"- Feature spaces: CLIP image CLS dim 768 and patch dim 1024 from `results/phase_xxiii_carrier_audit/feature_cache`.\n")
        fp.write("- State initialization: K1 normal/anomaly centroids; K2/K3 initialized from K1.\n")
        fp.write(f"- Training: AdamW lr={LR}, weight_decay=0, final epoch {EPOCHS}, seed {SEED}, category-balanced source sampling.\n")
        fp.write("- Variants: K0 instance memory, K1 centroid state, K2 learned visual state, K3 shuffled-label state.\n")
        fp.write(f"- Outcome: {outcome}.\n")


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt_dir = OUT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    payloads = []
    for direction in ["D1", "D2"]:
        for fold in [0, 1]:
            done = ckpt_dir / f"{direction}_fold{fold}_done.json"
            if done.exists():
                print(f"[XXXVIII-A] resume {direction} fold{fold}", flush=True)
                payloads.append(read_json(done))
                continue
            print(f"[XXXVIII-A] run {direction} fold{fold}", flush=True)
            payload = run_fold(direction, fold, device)
            write_json(done, payload)
            payloads.append(payload)
    aggregate_and_write(payloads, start)
    print(json.dumps(read_json(OUT / "OUTCOME.json"), indent=2), flush=True)


if __name__ == "__main__":
    main()
