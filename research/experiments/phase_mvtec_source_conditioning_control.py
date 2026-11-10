import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.phase_parameter_free_local_zoom_pyramid_fullres as lzp
from experiments.phase_defect_query_transfer_pilot import ArrayConcat
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import item_at, item_meta, mask_np
from experiments.phase_paper_ablation_close import lrp_map_for_image
from experiments.phase_paper_ready_final_validation import balanced_batch
from experiments.phase_private_residual_contrast_memory import contrast_logits, unit_np
from experiments.phase_statealigned_pc1_debias import remove_pc1
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_checkpoint_source_readout import all_indices_for_categories, direction_cfg
from experiments.phase_xa_smoke import load_dataset, write_csv
from experiments.phase_xlii_patch_specialized_adaptation import (
    BATCH_SIZE,
    EPOCHS as ADAPTER_EPOCHS,
    LR as ADAPTER_LR,
    PATCH_SAMPLE_PER_CLASS_CATEGORY,
    SEED as ADAPTER_SEED,
    WEIGHT_DECAY,
    balanced_loss,
)
from experiments.phase_xliii_adapter_head_decomposition import AdapterToState
from experiments.phase_xlv_cross_domain_transfer import top1_image_score
from experiments.phase_xxxviii_visual_state_representation import (
    EPOCHS as STATE_EPOCHS,
    LR as STATE_LR,
    SIDE,
    centroids,
    norm_np,
    patch_labels,
    train_state_vectors,
)
from phase_xa.official_task_forward import normalized_image_feature, normalized_patch_feature, visual_outputs
from phase_xa.source_folds import deterministic_source_folds


OUT = Path("results/phase_mvtec_source_conditioning_control/formal")
CACHE = OUT / "cache"
E_CACHE = Path("E:/MRAD_final_evidence_cache")
SEEDS = (0, 1, 2)
EPOCHS = 10
LR = 1e-3
TOP_NEG_FRAC = 0.10
TARGET = "VisA"
SETTINGS = (
    ("A_Current", False, False),
    ("B_PC1Only", True, False),
    ("C_AdapterOnly", False, True),
    ("D_PC1PlusAdapter", True, True),
)


def json_default(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")


def read_csv_rows(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def safe_auc(y, s):
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    return float(roc_auc_score(y, s) * 100.0) if np.unique(y).size == 2 else float("nan")


def safe_ap(y, s):
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    return float(average_precision_score(y, s) * 100.0) if np.unique(y).size == 2 else float("nan")


class IdentityAdapter(torch.nn.Module):
    def adapt(self, x):
        return x


def direction_data(direction):
    dataset_name, data_path, checkpoint = direction_cfg(direction)
    if not Path(data_path).exists():
        fallback = Path("E:/WorkSpace/dataset") / ("mvtec" if dataset_name == "mvtec" else "VISA")
        if fallback.exists():
            data_path = str(fallback)
    if not Path(checkpoint).exists():
        fallback_ckpt = ROOT / checkpoint
        if fallback_ckpt.exists():
            checkpoint = str(fallback_ckpt)
    dataset = load_dataset(dataset_name, data_path)
    rotation = deterministic_source_folds(dataset_name)["rotations"][0]
    train_indices = all_indices_for_categories(dataset, set(rotation["train_categories"]))
    held_indices = all_indices_for_categories(dataset, set(rotation["val_categories"]))
    indices = list(train_indices) + list(held_indices)
    return dataset_name, dataset, indices, checkpoint


def smoke_indices_for_dataset(dataset, indices, per_label_per_cat=2, max_total=48):
    chosen = []
    counts = {}
    for idx in indices:
        item = dataset[int(idx)]
        key = (str(item["cls_name"]), int(item["anomaly"]))
        if counts.get(key, 0) >= per_label_per_cat:
            continue
        chosen.append(int(idx))
        counts[key] = counts.get(key, 0) + 1
        if len(chosen) >= max_total:
            break
    return chosen


@torch.no_grad()
def ensure_direction_cache(direction, device, smoke=False):
    dataset_name, dataset, indices, checkpoint = direction_data(direction)
    if smoke:
        indices = smoke_indices_for_dataset(dataset, indices)
    suffix = "_smoke" if smoke else ""
    cls_path = CACHE / f"{direction}_fold0_cls{suffix}.npy"
    patch_path = CACHE / f"{direction}_fold0_patch{suffix}.npy"
    n = min(48, len(indices)) if smoke else len(indices)
    if cls_path.exists() and patch_path.exists():
        cls = np.load(cls_path, mmap_mode="r")
        patch = np.load(patch_path, mmap_mode="r")
        if cls.shape[0] >= n and patch.shape[0] >= n:
            return dataset_name, dataset, indices[:n], cls[:n], patch[:n]

    CACHE.mkdir(parents=True, exist_ok=True)
    print(f"[MVTCond] build frozen CLIP global cache direction={direction} n={n}", flush=True)
    model, *_ = load_components(checkpoint, device)
    model.eval()
    cls_mm = np.lib.format.open_memmap(cls_path, mode="w+", dtype=np.float16, shape=(n, 768))
    patch_mm = np.lib.format.open_memmap(patch_path, mode="w+", dtype=np.float16, shape=(n, SIDE * SIDE, 1024))
    for pos, idx in enumerate(indices[:n]):
        if pos % 50 == 0:
            print(f"[MVTCond] cache {direction} {pos}/{n}", flush=True)
        item = dataset[int(idx)]
        image = item["img"].unsqueeze(0).to(device)
        out = visual_outputs(model, image, adapter=None, grad=False)
        cls_mm[pos] = normalized_image_feature(out).squeeze(0).float().cpu().numpy().astype(np.float16)
        patch_mm[pos] = normalized_patch_feature(out).squeeze(0).float().cpu().numpy().astype(np.float16)
    cls_mm.flush()
    patch_mm.flush()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dataset_name, dataset, indices[:n], np.load(cls_path, mmap_mode="r"), np.load(patch_path, mmap_mode="r")


def compute_pc1(setting, dataset, indices, patch, enabled):
    if not enabled:
        return np.zeros((1024,), dtype=np.float32)
    q_path = OUT / f"{setting}_mvtec_normal_pc1.npy"
    if q_path.exists():
        q = np.load(q_path).astype(np.float32)
        return unit_np(q).astype(np.float32)
    print(f"[MVTCond] compute source-normal PC1 for {setting}", flush=True)
    fmap_sum = np.zeros((SIDE * SIDE, 1024), dtype=np.float64)
    n_normal = 0
    for pos, idx in enumerate(indices):
        item = dataset.data_all[int(idx)]
        if int(item["anomaly"]) != 0:
            continue
        fmap_sum += np.asarray(patch[pos], dtype=np.float32).astype(np.float64)
        n_normal += 1
    if n_normal <= 0:
        raise RuntimeError("No MVTec normal source images found for PC1.")
    fmap = fmap_sum / float(n_normal)
    x = fmap - fmap.mean(axis=0, keepdims=True)
    _u, s, vt = np.linalg.svd(x, full_matrices=False)
    q = unit_np(vt[0].astype(np.float32)).astype(np.float32)
    np.save(q_path, q)
    write_csv(
        OUT / f"{setting}_PC1_SPECTRUM.csv",
        [
            {
                "rank": i + 1,
                "singular_value": float(sv),
                "explained_variance_ratio": float((sv * sv) / max(float(np.sum(s * s)), 1e-12)),
            }
            for i, sv in enumerate(s)
        ],
    )
    return q


def transform_flat(patch_image, q_pc1, adapter, device):
    z = remove_pc1(np.asarray(patch_image, dtype=np.float32), q_pc1)
    z = norm_np(z.reshape(SIDE * SIDE, 1024))
    if adapter is not None:
        outs = []
        with torch.no_grad():
            for start in range(0, z.shape[0], 8192):
                x = torch.tensor(z[start : start + 8192], dtype=torch.float32, device=device)
                outs.append(adapter.adapt(x).detach().cpu().numpy().astype(np.float32))
        z = np.concatenate(outs, axis=0)
    return z.astype(np.float32)


@torch.no_grad()
def score_patch_batch(source_info, v, patch_batch, device, chunk=8192):
    arr = np.asarray(patch_batch, dtype=np.float32).reshape(-1, SIDE * SIDE, 1024)
    b = arr.shape[0]
    flat = arr.reshape(-1, 1024)
    if source_info["use_align"]:
        q = np.asarray(source_info["q_pc1"], dtype=np.float32)
        flat = flat - (flat @ q)[:, None] * q[None, :]
        flat = norm_np(flat)
        outs = []
        adapter = source_info["align_model"]
        for start in range(0, flat.shape[0], chunk):
            x = torch.tensor(flat[start : start + chunk], dtype=torch.float32, device=device)
            outs.append(adapter.adapt(x).detach().cpu().numpy().astype(np.float32))
        flat = np.concatenate(outs, axis=0)
    else:
        flat = norm_np(flat)
    m_t = torch.tensor(source_info["m"], dtype=torch.float32, device=device)
    v_t = torch.tensor(v, dtype=torch.float32, device=device)
    scores = []
    for start in range(0, flat.shape[0], chunk):
        x = torch.tensor(flat[start : start + chunk], dtype=torch.float32, device=device)
        scores.append(contrast_logits(x, m_t, source_info["alpha"], v_t).detach().cpu().numpy().astype(np.float32))
    score = np.concatenate(scores, axis=0).reshape(b, SIDE, SIDE)
    return score


def resize_global_batch(global37, size=(SIDE * 2, SIDE * 2)):
    x = torch.tensor(np.asarray(global37, dtype=np.float32)[:, None, :, :])
    y = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
    return y[:, 0].numpy().astype(np.float32)


def collect_patch_samples(dataset, indices, patch, q_pc1, seed, cap=PATCH_SAMPLE_PER_CLASS_CATEGORY):
    rng = np.random.default_rng(seed)
    by_key = {}
    for pos, idx in enumerate(indices):
        item = dataset[int(idx)]
        cat = str(item["cls_name"])
        y = patch_labels(item).astype(np.int64)
        z = remove_pc1(np.asarray(patch[pos], dtype=np.float32), q_pc1)
        z = norm_np(z.reshape(SIDE * SIDE, 1024))
        for cls in (0, 1):
            ids = np.flatnonzero(y == cls)
            if ids.size == 0:
                continue
            key = (cat, cls)
            parts = by_key.setdefault(key, [])
            room = cap - sum(part.shape[0] for part in parts)
            if room <= 0:
                continue
            if ids.size > room:
                ids = rng.choice(ids, size=room, replace=False)
            parts.append(z[ids])
    feats, ys, cats = [], [], []
    for (cat, cls), parts in sorted(by_key.items()):
        arr = np.concatenate(parts, axis=0).astype(np.float32)
        feats.append(arr)
        ys.extend([cls] * arr.shape[0])
        cats.extend([cat] * arr.shape[0])
    return np.concatenate(feats, axis=0), np.asarray(ys, dtype=np.int64), np.asarray(cats)


def train_adapter_if_needed(setting, use_adapter, dataset, indices, cls, patch, q_pc1, seed, device, smoke=False):
    if not use_adapter:
        return None, {"adapter_checkpoint": "N/A", "adapter_trained": False}
    ckpt = OUT / "checkpoints" / f"{setting}_seed{seed}_adapter.pt"
    if ckpt.exists() and not smoke:
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        model = AdapterToState(1024, payload["patch_states"]).to(device)
        model.load_state_dict(payload["model"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model, {"adapter_checkpoint": str(ckpt), "adapter_trained": False}

    print(f"[MVTCond] train source-only adapter {setting} seed={seed}", flush=True)
    torch.manual_seed(int(seed))
    patch_x, patch_y, patch_cat = collect_patch_samples(dataset, indices, patch, q_pc1, seed)
    y_img = np.asarray([int(dataset[int(idx)]["anomaly"]) for idx in indices], dtype=np.int64)
    cat_img = np.asarray([str(dataset[int(idx)]["cls_name"]) for idx in indices])
    cls_train = norm_np(np.asarray(cls, dtype=np.float32))
    init_img = centroids(cls_train, y_img)
    init_patch = centroids(patch_x, patch_y)
    _img_states, patch_states, _curves = train_state_vectors(
        cls_train,
        y_img,
        cat_img,
        patch_x,
        patch_y,
        patch_cat,
        init_img,
        init_patch,
        shuffled=False,
        device=device,
    )
    model = AdapterToState(1024, patch_states).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=ADAPTER_LR, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(seed)
    x_np = norm_np(patch_x)
    y_np = patch_y.astype(np.float32)
    steps = min(3, max(1, int(np.ceil(len(y_np) / BATCH_SIZE)))) if smoke else max(1, int(np.ceil(len(y_np) / BATCH_SIZE)))
    curve = []
    for epoch in range(1, ADAPTER_EPOCHS + 1):
        vals = []
        for _ in range(steps):
            ids = rng.choice(len(y_np), size=min(BATCH_SIZE, len(y_np)), replace=len(y_np) < BATCH_SIZE)
            xb = torch.tensor(x_np[ids], dtype=torch.float32, device=device)
            yb = torch.tensor(y_np[ids], dtype=torch.float32, device=device)
            logits, _z = model.logits(xb)
            loss, bce, dice = balanced_loss(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            vals.append(float(loss.detach().cpu()))
        curve.append({"epoch": epoch, "adapter_loss": float(np.mean(vals))})
        print(f"[MVTCond] adapter {setting} seed={seed} epoch={epoch}/{ADAPTER_EPOCHS} loss={curve[-1]['adapter_loss']:.4f}", flush=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if not smoke:
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "patch_states": patch_states.astype(np.float32),
                "setting": setting,
                "seed": seed,
                "q_pc1_enabled": not np.allclose(q_pc1, 0),
                "adapter_epochs": ADAPTER_EPOCHS,
                "adapter_lr": ADAPTER_LR,
                "adapter_weight_decay": WEIGHT_DECAY,
                "curve": curve,
            },
            ckpt,
        )
    return model, {"adapter_checkpoint": str(ckpt), "adapter_trained": True}


def collect_conditioned_source(setting, use_pc1, use_adapter, seed, dataset, indices, cls, patch, q_pc1, adapter, device):
    slots = {}
    for pos, idx in enumerate(indices):
        cat = str(dataset.data_all[int(idx)]["cls_name"])
        slots.setdefault(cat, []).append(pos)
    rng = np.random.default_rng(seed)
    groups = {}
    for cat, positions in sorted(slots.items()):
        parts = {"normal_image": [], "anomaly_mask_out": [], "anomaly_mask": []}
        for pos in positions:
            idx = int(indices[pos])
            item = dataset[int(idx)]
            y = patch_labels(item).astype(np.int64)
            z = transform_flat(patch[pos], q_pc1, adapter, device)
            pools = [("normal_image", np.flatnonzero(y == 0))] if int(item["anomaly"]) == 0 else [
                ("anomaly_mask_out", np.flatnonzero(y == 0)),
                ("anomaly_mask", np.flatnonzero(y == 1)),
            ]
            for origin, ids in pools:
                room = PATCH_SAMPLE_PER_CLASS_CATEGORY - sum(x.shape[0] for x in parts[origin])
                if room <= 0 or ids.size == 0:
                    continue
                if ids.size > room:
                    ids = rng.choice(ids, size=room, replace=False)
                parts[origin].append(z[ids])
        xs, ys = [], []
        for origin, cls_lab in (("normal_image", 0), ("anomaly_mask_out", 0), ("anomaly_mask", 1)):
            if parts[origin]:
                arr = np.concatenate(parts[origin], axis=0).astype(np.float32)
                xs.append(arr)
                ys.extend([cls_lab] * arr.shape[0])
        if xs and len(set(ys)) == 2:
            groups[cat] = {"x": np.concatenate(xs, axis=0), "y": np.asarray(ys, dtype=np.float32)}
    pos = np.concatenate([g["x"][g["y"] > 0.5] for g in groups.values()], axis=0)
    neg = np.concatenate([g["x"][g["y"] < 0.5] for g in groups.values()], axis=0)
    pos_mu, neg_mu = unit_np(pos.mean(axis=0)), unit_np(neg.mean(axis=0))
    m = (0.5 * (pos_mu + neg_mu)).astype(np.float32)
    d = (0.5 * (pos_mu - neg_mu)).astype(np.float32)
    alpha = float(np.linalg.norm(d))
    return {
        "source": "Full MVTec",
        "setting": setting,
        "pc1": bool(use_pc1),
        "adapter": bool(use_adapter),
        "use_align": bool(use_pc1 or use_adapter),
        "q_pc1": q_pc1.astype(np.float32),
        "align_model": adapter if adapter is not None else IdentityAdapter().to(device),
        "m": m,
        "alpha": alpha,
        "init_v": unit_np(d).astype(np.float32),
        "groups": groups,
    }


def train_ntr(source_info, seed, device, smoke=False):
    ckpt = OUT / "checkpoints" / f"{source_info['setting']}_seed{seed}_ntr.pt"
    if ckpt.exists() and not smoke:
        return torch.load(ckpt, map_location="cpu", weights_only=False)["v"].astype(np.float32), str(ckpt), False
    rng = np.random.default_rng(seed)
    groups = source_info["groups"]
    total = sum(len(g["y"]) for g in groups.values())
    steps = min(3, max(1, int(np.ceil(total / BATCH_SIZE)))) if smoke else max(1, int(np.ceil(total / BATCH_SIZE)))
    param = torch.nn.Parameter(torch.tensor(source_info["init_v"], dtype=torch.float32, device=device))
    m = torch.tensor(source_info["m"], dtype=torch.float32, device=device)
    opt = torch.optim.AdamW([param], lr=LR, weight_decay=0.0)
    curve = []
    for epoch in range(1, EPOCHS + 1):
        vals = []
        for _ in range(steps):
            xb, yb = balanced_batch(groups, rng)
            z = torch.tensor(xb, dtype=torch.float32, device=device)
            y = torch.tensor(yb, dtype=torch.float32, device=device)
            v = F.normalize(param, dim=-1)
            logits = contrast_logits(z, m, source_info["alpha"], v)
            task, _bce, _dice = balanced_loss(logits, y)
            neg = torch.nonzero(y < 0.5, as_tuple=False).flatten()
            k = max(1, int(np.ceil(float(len(neg)) * TOP_NEG_FRAC)))
            idx = neg[torch.topk(logits.detach()[neg], k=k, largest=True).indices]
            ntr = F.softplus(logits[idx]).mean()
            loss = task + ntr
            opt.zero_grad()
            loss.backward()
            opt.step()
            vals.append(float(loss.detach().cpu()))
        curve.append({"epoch": epoch, "loss": float(np.mean(vals))})
        print(f"[MVTCond] NTR {source_info['setting']} seed={seed} epoch={epoch}/{EPOCHS} loss={curve[-1]['loss']:.4f}", flush=True)
    with torch.no_grad():
        v = F.normalize(param, dim=-1).detach().cpu().numpy().astype(np.float32)
    if not smoke:
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "v": v,
                "setting": source_info["setting"],
                "seed": seed,
                "pc1": source_info["pc1"],
                "adapter": source_info["adapter"],
                "m": source_info["m"],
                "alpha": source_info["alpha"],
                "init_v": source_info["init_v"],
                "curve": curve,
            },
            ckpt,
        )
    return v, str(ckpt), True


def source_metrics(source_info, v, device):
    ys, ss = [], []
    m = torch.tensor(source_info["m"], dtype=torch.float32, device=device)
    vt = torch.tensor(v, dtype=torch.float32, device=device)
    for g in source_info["groups"].values():
        x = torch.tensor(g["x"], dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = contrast_logits(x, m, source_info["alpha"], vt).detach().cpu().numpy()
        ys.append(g["y"])
        ss.append(logits)
    y = np.concatenate(ys)
    s = np.concatenate(ss)
    return safe_ap(y, s), safe_auc(y, s)


def load_target_visa(device, smoke=False):
    _dn, dataset, refs, _cls, patch = ensure_direction_cache("D2", device, smoke=smoke)
    n = min(24, len(refs)) if smoke else len(refs)
    refs = refs[:n]
    return dataset, refs, patch[:n], "source_dataset"


def local_patch_cache_for_mvtec_visa(n, smoke=False):
    path = E_CACHE / "local_patch_Full_MVTec__VisA.npy"
    if path.exists():
        arr = np.load(path, mmap_mode="r")
        if arr.shape[0] >= n:
            return arr[:n]
    return None


@torch.no_grad()
def evaluate_mfr_only(source, target, maps, metas, variant="E2_GlobalLocalZoomPyramid"):
    print(f"[MVTCond] M_FR metrics {source} -> {target} {variant}", flush=True)
    by_cat = {}
    for i, meta in enumerate(metas):
        d = by_cat.setdefault(meta["category"], {"labels": [], "image_scores": [], "masks": [], "maps": []})
        d["labels"].append(meta["label"])
        d["image_scores"].append(top1_image_score(maps[variant][i]))
        d["masks"].append(meta["mask"])
        d["maps"].append(np.asarray(maps[variant][i], dtype=np.float32))
    cat_rows = []
    for cat, vals in sorted(by_cat.items()):
        print(f"[MVTCond] M_FR metrics {source} -> {target} {variant} category={cat}", flush=True)
        scores = [lzp.torch_resize(s, m.shape, align_corners=True) for s, m in zip(vals["maps"], vals["masks"])]
        fr = lzp.summarize_category(vals["labels"], vals["image_scores"], vals["masks"], scores)
        row = {
            "source": source,
            "dataset": target,
            "category": cat,
            "variant": variant,
            "protocol": "M",
            "protocol_name": "MoECLIP-compatible full-res",
            "aggregation": "CATEGORY",
            "score_resolution": f"{vals['maps'][0].shape[0]}x{vals['maps'][0].shape[1]}",
        }
        for met, val in fr.items():
            row[f"M_FR_{met.replace('-', '_')}"] = val
        cat_rows.append(row)
    raw = {
        "source": source,
        "dataset": target,
        "variant": variant,
        "aggregation": "CATEGORY_MACRO",
        "score_resolution": f"{maps[variant][0].shape[0]}x{maps[variant][0].shape[1]}",
    }
    for met in lzp.METRICS:
        vals = np.asarray([fnum(r, f"M_FR_{met.replace('-', '_')}") for r in cat_rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        raw[f"M_FR_{met.replace('-', '_')}"] = float(vals.mean()) if vals.size else float("nan")
    return raw, cat_rows


@torch.no_grad()
def evaluate_mvtec_to_visa(setting, source_info, v, device, smoke=False):
    dataset, refs, patch, kind = load_target_visa(device, smoke=smoke)
    n = min(24, len(refs)) if smoke else len(refs)
    local_cache = local_patch_cache_for_mvtec_visa(n, smoke=smoke)
    maps = {"E2_GlobalLocalZoomPyramid": []}
    metas = []
    if local_cache is None:
        raise RuntimeError(
            "Missing local patch cache for Full_MVTec__VisA. "
            "Expected E:/MRAD_final_evidence_cache/local_patch_Full_MVTec__VisA.npy"
        )
    batch = 8
    for start in range(0, n, batch):
        end = min(n, start + batch)
        if start % 50 == 0:
            print(f"[MVTCond] eval {setting} MVTec->VisA image {start}/{n}", flush=True)
        gb = score_patch_batch(source_info, v, np.asarray(patch[start:end], dtype=np.float32), device)
        lp = np.asarray(local_cache[start:end], dtype=np.float32)
        lb_flat = lp.reshape(-1, SIDE * SIDE, 1024)
        lb_scores = score_patch_batch(source_info, v, lb_flat, device).reshape(end - start, 4, SIDE, SIDE)
        local74 = np.zeros((end - start, SIDE * 2, SIDE * 2), dtype=np.float32)
        local74[:, :SIDE, :SIDE] = lb_scores[:, 0]
        local74[:, :SIDE, SIDE:] = lb_scores[:, 1]
        local74[:, SIDE:, :SIDE] = lb_scores[:, 2]
        local74[:, SIDE:, SIDE:] = lb_scores[:, 3]
        global74 = resize_global_batch(gb)
        final = (0.5 * global74 + 0.5 * local74).astype(np.float32)
        for j, pos in enumerate(range(start, end)):
            ref = refs[pos]
            item = item_at(dataset, kind, ref, pos)
            cat, label, path = item_meta(item, kind, ref, TARGET)
            maps["E2_GlobalLocalZoomPyramid"].append(final[j])
            metas.append({"category": cat, "label": int(label), "mask": mask_np(item), "img_path": path})
    raw, cat_rows = evaluate_mfr_only("Full MVTec", TARGET, maps, metas)
    return raw, cat_rows


def fnum(row, key, default=float("nan")):
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def summarize(rows):
    out = []
    for setting in [s[0] for s in SETTINGS]:
        part = [r for r in rows if r["setting"] == setting]
        if not part:
            continue
        row = {"setting": setting, "n": len(part)}
        for key in ("source_patch_ap", "source_patch_auroc", "i_auroc", "i_ap", "p_auroc", "p_ap", "p_f1max", "aupro"):
            vals = np.asarray([float(r[key]) for r in part], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            row[f"{key}_mean"] = float(vals.mean()) if vals.size else float("nan")
            row[f"{key}_std"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        out.append(row)
    base = next((r for r in out if r["setting"] == "A_Current"), None)
    if base:
        for r in out:
            r["delta_p_ap_vs_current"] = fnum(r, "p_ap_mean") - fnum(base, "p_ap_mean")
            r["delta_aupro_vs_current"] = fnum(r, "aupro_mean") - fnum(base, "aupro_mean")
    return out


def report(summary, rows, audit_rows, runtime):
    lines = [
        "# MVTec Source Conditioning Control",
        "",
        "Diagnostic source-preprocessing control for `Full MVTec -> VisA`.",
        "",
        "## Settings",
        "",
        "| Setting | PC1 removal | Adapter |",
        "|---|---:|---:|",
    ]
    for setting, pc1, adapter in SETTINGS:
        lines.append(f"| {setting} | {pc1} | {adapter} |")
    lines += [
        "",
        "All variants keep the frozen CLIP backbone, CES/NTR training budget, NTR top-10% hard-normal policy, Fixed 2x2 LRP inference, and unified full-resolution M_FR evaluator unchanged. VisA is used only for final evaluation.",
        "",
        "## Mean ± Std",
        "",
        "| Setting | Source patch AP | Source patch AUROC | I-AUROC | I-AP | P-AUROC | P-AP | P-F1max | AUPRO | Delta P-AP vs current |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"| {r['setting']} | {r['source_patch_ap_mean']:.4f}±{r['source_patch_ap_std']:.4f} | "
            f"{r['source_patch_auroc_mean']:.4f}±{r['source_patch_auroc_std']:.4f} | "
            f"{r['i_auroc_mean']:.4f}±{r['i_auroc_std']:.4f} | {r['i_ap_mean']:.4f}±{r['i_ap_std']:.4f} | "
            f"{r['p_auroc_mean']:.4f}±{r['p_auroc_std']:.4f} | {r['p_ap_mean']:.4f}±{r['p_ap_std']:.4f} | "
            f"{r['p_f1max_mean']:.4f}±{r['p_f1max_std']:.4f} | {r['aupro_mean']:.4f}±{r['aupro_std']:.4f} | "
            f"{r.get('delta_p_ap_vs_current', float('nan')):+.4f} |"
        )
    lines += [
        "",
        "## Protocol Audit",
        "",
        f"- Seeds: {', '.join(str(s) for s in SEEDS)}.",
        f"- Source NTR epochs: {EPOCHS}; lr {LR}; dynamic top-normal tail fraction {TOP_NEG_FRAC}.",
        f"- Adapter config, when enabled: rank 4 residual `AdapterToState`, epochs {ADAPTER_EPOCHS}, lr {ADAPTER_LR}, weight decay {WEIGHT_DECAY}.",
        "- MVTec source PC1 is estimated only from source normal-image patch grids; target data is never used.",
        "- Fixed LRP local branch reuses `E:/MRAD_final_evidence_cache/local_patch_Full_MVTec__VisA.npy` when available.",
        f"- Runtime: {runtime:.1f}s.",
        "",
        "## Artifact Trace",
        "",
    ]
    for r in audit_rows:
        lines.append(f"- {r['setting']} seed {r['seed']}: adapter={r['adapter_checkpoint']}; ntr={r['ntr_checkpoint']}")
    (OUT / "mvt_source_conditioning_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    wanted_settings = {s.strip() for s in str(args.settings).split(",") if s.strip()}
    wanted_seeds = tuple(int(s.strip()) for s in str(args.seeds).split(",") if s.strip())
    _name, source_dataset, source_indices, source_cls, source_patch = ensure_direction_cache("D1", device, smoke=args.smoke)
    # Target global cache is needed by the M_FR evaluator; local cache is reused from E:.
    ensure_direction_cache("D2", device, smoke=args.smoke)

    rows = read_csv_rows(OUT / "mvt_source_conditioning_control.partial.csv")
    cat_rows_all = read_csv_rows(OUT / "mvt_to_visa_category_control.partial.csv")
    completed = {(r.get("setting"), int(r.get("seed", -999))) for r in rows if r.get("setting") and str(r.get("seed", "")).strip()}
    audit_rows = []
    for setting, use_pc1, use_adapter in SETTINGS:
        if wanted_settings and setting not in wanted_settings:
            continue
        q_pc1 = compute_pc1(setting, source_dataset, source_indices, source_patch, use_pc1)
        for seed in wanted_seeds:
            if (setting, seed) in completed:
                print(f"[MVTCond] skip completed {setting} seed={seed}", flush=True)
                continue
            adapter_model, adapter_audit = train_adapter_if_needed(
                setting, use_adapter, source_dataset, source_indices, source_cls, source_patch, q_pc1, seed, device, smoke=args.smoke
            )
            source_info = collect_conditioned_source(
                setting, use_pc1, use_adapter, seed, source_dataset, source_indices, source_cls, source_patch, q_pc1, adapter_model, device
            )
            v, ntr_ckpt, _trained = train_ntr(source_info, seed, device, smoke=args.smoke)
            sap, sauc = source_metrics(source_info, v, device)
            raw, cat_rows = evaluate_mvtec_to_visa(setting, source_info, v, device, smoke=args.smoke)
            rows.append(
                {
                    "setting": setting,
                    "seed": seed,
                    "pc1": bool(use_pc1),
                    "adapter": bool(use_adapter),
                    "source_patch_ap": sap,
                    "source_patch_auroc": sauc,
                    "i_auroc": fnum(raw, "M_FR_I_AUROC"),
                    "i_ap": fnum(raw, "M_FR_I_AP"),
                    "p_auroc": fnum(raw, "M_FR_P_AUROC"),
                    "p_ap": fnum(raw, "M_FR_P_AP"),
                    "p_f1max": fnum(raw, "M_FR_P_F1max"),
                    "aupro": fnum(raw, "M_FR_PRO"),
                    "ntr_checkpoint": ntr_ckpt,
                    "adapter_checkpoint": adapter_audit["adapter_checkpoint"],
                }
            )
            for cr in cat_rows:
                if cr.get("variant") != "E2_GlobalLocalZoomPyramid":
                    continue
                cat_rows_all.append(
                    {
                        "setting": setting,
                        "seed": seed,
                        "category": cr.get("category"),
                        "p_ap": fnum(cr, "M_FR_P_AP"),
                        "p_auroc": fnum(cr, "M_FR_P_AUROC"),
                        "p_f1max": fnum(cr, "M_FR_P_F1max"),
                        "aupro": fnum(cr, "M_FR_PRO"),
                    }
                )
            audit_rows.append(
                {
                    "setting": setting,
                    "seed": seed,
                    "adapter_checkpoint": adapter_audit["adapter_checkpoint"],
                    "ntr_checkpoint": ntr_ckpt,
                }
            )
            write_csv(OUT / "mvt_source_conditioning_control.partial.csv", rows)
            write_csv(OUT / "mvt_to_visa_category_control.partial.csv", cat_rows_all)

    summary = summarize(rows)
    write_csv(OUT / "mvt_source_conditioning_control.csv", rows)
    write_csv(OUT / "mvt_source_conditioning_summary.csv", summary)
    write_csv(OUT / "mvt_to_visa_category_control.csv", cat_rows_all)
    write_json(
        OUT / "OUTCOME.json",
        {
            "outcome": "MVT_SOURCE_CONDITIONING_CONTROL_COMPLETE",
            "settings": [{"setting": s, "pc1": p, "adapter": a} for s, p, a in SETTINGS],
            "seeds": list(SEEDS),
            "runtime_sec": time.time() - start,
            "target": "VisA",
            "source": "Full MVTec",
            "local_cache": str(E_CACHE / "local_patch_Full_MVTec__VisA.npy"),
        },
    )
    report(summary, rows, audit_rows, time.time() - start)
    print("=== MVTec Source Conditioning Control ===")
    for r in summary:
        print(
            f"{r['setting']}: P-AP {r['p_ap_mean']:.4f}±{r['p_ap_std']:.4f}, "
            f"AUPRO {r['aupro_mean']:.4f}±{r['aupro_std']:.4f}, "
            f"Delta P-AP {r.get('delta_p_ap_vs_current', float('nan')):+.4f}",
            flush=True,
        )
    print(f"Saved: {OUT / 'mvt_source_conditioning_control.csv'}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser("MVTec source preprocessing control")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--settings", default=",".join(s[0] for s in SETTINGS))
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
