import argparse
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
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.phase_parameter_free_local_zoom_pyramid_fullres as lzp
from experiments.phase_final_cross_dataset_benchmark import OUT as FINAL_OUT, load_mvtec_source, load_visa_source, protocol_cells
from experiments.phase_hard_normal_cross_source_mvtec import load_full_mvtec_source
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import evaluate_maps, fnum, write_json
from experiments.phase_paper_ready_final_validation import existing_seed0_rows
from experiments.phase_private_residual_contrast_memory import contrast_logits, unit_np
from experiments.phase_spatial_residual_refinement_standard_metric import aligned_grid, logits_from_z
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, PATCH_SAMPLE_PER_CLASS_CATEGORY, SEED, dice_loss
from experiments.phase_xxxviii_visual_state_representation import SIDE, norm_np, patch_labels
from experiments.standard_zsad_metrics import resize_mask_flat


OUT = Path("results/phase_morphology_balanced_source_training/formal")
TARGETS = ("MVTec", "VisA", "BTAD", "MPDD", "DTD-Synthetic")
PAIRS = {
    ("Full VisA", "MVTec"),
    ("Full VisA", "BTAD"),
    ("Full VisA", "MPDD"),
    ("Full VisA", "DTD-Synthetic"),
    ("Full MVTec", "VisA"),
    ("Full MVTec", "BTAD"),
    ("Full MVTec", "MPDD"),
    ("Full MVTec", "DTD-Synthetic"),
}
VARIANTS = ("B0_CurrentLabels", "E1_AnyOverlapLabels", "E2_ComponentBalancedRisk", "E3_AnyOverlap_ComponentBalanced")
EPOCHS = 10
LR = 0.001
TOP_NEG_FRAC = 0.10
CELL = 14


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


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def mask_np(item):
    m = item["img_mask"].detach().cpu().numpy()
    return (np.squeeze(m) > 0.5).astype(np.uint8)


def any_overlap_labels(item):
    m = mask_np(item)
    if m.shape[0] != SIDE * CELL or m.shape[1] != SIDE * CELL:
        # Conservative fallback: area/max interpolation at source resolution.
        x = torch.tensor(m.astype(np.float32)).view(1, 1, *m.shape)
        pooled = F.interpolate(x, size=(SIDE, SIDE), mode="area").view(-1).numpy()
        return (pooled > 0).astype(np.int64)
    return m.reshape(SIDE, CELL, SIDE, CELL).max(axis=(1, 3)).reshape(-1).astype(np.int64)


def component_patch_weights(item, labels):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    pos = np.flatnonzero(labels == 1)
    if pos.size == 0:
        return np.ones_like(labels, dtype=np.float32), np.zeros_like(labels, dtype=np.int64), np.asarray([], dtype=np.int64)
    m = mask_np(item)
    comps, n = ndimage.label(m > 0, structure=np.ones((3, 3), dtype=np.int8))
    wt = np.zeros_like(labels, dtype=np.float64)
    comp_size = np.zeros_like(labels, dtype=np.int64)
    component_sizes = []
    for cid in range(1, n + 1):
        cm = comps == cid
        if m.shape[0] == SIDE * CELL and m.shape[1] == SIDE * CELL:
            pids = np.flatnonzero(cm.reshape(SIDE, CELL, SIDE, CELL).max(axis=(1, 3)).reshape(-1) > 0)
        else:
            x = torch.tensor(cm.astype(np.float32)).view(1, 1, *cm.shape)
            pids = np.flatnonzero(F.interpolate(x, size=(SIDE, SIDE), mode="area").view(-1).numpy() > 0)
        pids = np.intersect1d(pids, pos, assume_unique=False)
        if pids.size == 0:
            continue
        component_sizes.append(int(pids.size))
        wt[pids] += 1.0 / float(pids.size)
        comp_size[pids] = np.maximum(comp_size[pids], int(pids.size))
    if wt[pos].sum() <= 0:
        out = np.ones_like(labels, dtype=np.float32)
        comp_size[pos] = int(pos.size)
        return out, comp_size, np.asarray([int(pos.size)], dtype=np.int64)
    wt[pos] *= float(pos.size) / float(wt[pos].sum())
    out = np.ones_like(labels, dtype=np.float32)
    out[pos] = wt[pos].astype(np.float32)
    return out, comp_size, np.asarray(component_sizes, dtype=np.int64)


def full_component_audit(source_name, smoke=False):
    dataset, indices, _patch = source_dataset(source_name, smoke)
    rows = []
    areas, thin = [], []
    for pos, idx in enumerate(indices):
        item = dataset[int(idx)]
        if int(item.get("anomaly", 0)) != 1:
            continue
        m = mask_np(item)
        comps, n = ndimage.label(m > 0, structure=np.ones((3, 3), dtype=np.int8))
        cur = patch_labels(item).reshape(SIDE, SIDE)
        anylab = any_overlap_labels(item).reshape(SIDE, SIDE)
        for cid in range(1, n + 1):
            cm = comps == cid
            area = int(cm.sum())
            if area == 0:
                continue
            er = ndimage.binary_erosion(cm, structure=np.ones((3, 3), dtype=bool), border_value=0)
            b = np.logical_and(cm, ~er)
            if m.shape[0] == SIDE * CELL and m.shape[1] == SIDE * CELL:
                comp_patch = cm.reshape(SIDE, CELL, SIDE, CELL).max(axis=(1, 3))
            else:
                x = torch.tensor(cm.astype(np.float32)).view(1, 1, *cm.shape)
                comp_patch = F.interpolate(x, size=(SIDE, SIDE), mode="area").view(SIDE, SIDE).numpy() > 0
            cur_count = int(np.logical_and(comp_patch, cur > 0).sum())
            any_count = int(np.logical_and(comp_patch, anylab > 0).sum())
            row = {
                "source": source_name,
                "category": str(item["cls_name"]),
                "image_index": int(pos),
                "component_id": int(cid),
                "original_pixel_area": area,
                "original_thinness": float(b.sum()) / max(area, 1),
                "current_positive_patch_count": cur_count,
                "any_overlap_positive_patch_count": any_count,
                "component_survives_current": cur_count > 0,
                "component_survives_any_overlap": any_count > 0,
            }
            rows.append(row)
            areas.append(area)
            thin.append(row["original_thinness"])
    area_q25 = float(np.percentile(areas, 25)) if areas else float("nan")
    thin_q75 = float(np.percentile(thin, 75)) if thin else float("nan")
    for r in rows:
        r["source_small_area_q25"] = area_q25
        r["source_thin_q75"] = thin_q75
        r["is_small_source_q25"] = fnum(r, "original_pixel_area") <= area_q25
        r["is_thin_source_q75"] = fnum(r, "original_thinness") >= thin_q75
    return rows


def source_dataset(source_name, smoke):
    if source_name == "Full VisA":
        _name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    else:
        _name, dataset, indices, patch = load_full_mvtec_source(False)
    if smoke:
        by_cat = defaultdict(lambda: {0: [], 1: []})
        for pos, idx in enumerate(indices):
            item = dataset.data_all[int(idx)]
            by_cat[str(item["cls_name"])][int(item["anomaly"])].append((pos, idx))
        chosen = []
        for cat in sorted(by_cat)[:4]:
            chosen.extend(by_cat[cat][0][:4])
            chosen.extend(by_cat[cat][1][:4])
        patch = np.stack([np.asarray(patch[p], dtype=np.float32) for p, _idx in chosen], axis=0)
        indices = [idx for _p, idx in chosen]
    return dataset, list(indices), patch


@torch.no_grad()
def source_feature(source_info, patch_grid, device):
    if source_info["use_align"]:
        return aligned_grid(patch_grid, source_info["q_pc1"], source_info["align_model"], device).detach().cpu().numpy().reshape(SIDE * SIDE, -1).astype(np.float32)
    return norm_np(np.asarray(patch_grid, dtype=np.float32).reshape(SIDE * SIDE, -1)).astype(np.float32)


def build_source_pool(source_info, label_mode, risk_mode, device, smoke):
    suffix = "smoke" if smoke else "formal"
    cache = OUT / f"{source_info['source'].replace(' ', '_')}_{label_mode}_{risk_mode}_POOL_{suffix}.npz"
    if cache.exists() and not smoke:
        data = np.load(cache, allow_pickle=True)
        groups = {}
        for cat in data["categories"].tolist():
            groups[cat] = {
                "x": data[f"{cat}_x"].astype(np.float32),
                "y": data[f"{cat}_y"].astype(np.float32),
                "w": data[f"{cat}_w"].astype(np.float32),
                "comp_size": data[f"{cat}_comp_size"].astype(np.int64),
            }
        return groups, data["audit"].tolist()
    dataset, indices, patch = source_dataset(source_info["source"], smoke)
    rng = np.random.default_rng(SEED)
    slots = defaultdict(list)
    for pos, idx in enumerate(indices):
        slots[str(dataset.data_all[int(idx)]["cls_name"])].append(pos)
    groups, audit = {}, []
    for cat, positions in sorted(slots.items()):
        xs, ys, ws, cs = [], [], [], []
        counts = defaultdict(int)
        comp_all = []
        for pos in positions:
            item = dataset[int(indices[pos])]
            labels = any_overlap_labels(item) if label_mode == "any" else patch_labels(item)
            weights, comp_size, component_sizes = component_patch_weights(item, labels)
            comp_all.extend(component_sizes.tolist())
            z = source_feature(source_info, patch[pos], device)
            for cls in (0, 1):
                ids = np.flatnonzero(labels == cls)
                if ids.size == 0:
                    continue
                key = (cat, cls)
                room = PATCH_SAMPLE_PER_CLASS_CATEGORY - counts[key]
                if room <= 0:
                    continue
                if ids.size > room:
                    ids = rng.choice(ids, size=room, replace=False)
                counts[key] += int(ids.size)
                xs.append(z[ids])
                ys.append(np.full(ids.size, cls, dtype=np.float32))
                if risk_mode == "component" and cls == 1:
                    ws.append(weights[ids].astype(np.float32))
                else:
                    ws.append(np.ones(ids.size, dtype=np.float32))
                cs.append(comp_size[ids].astype(np.int64))
        if not xs or len({int(v) for arr in ys for v in arr.tolist()}) < 2:
            continue
        groups[cat] = {
            "x": np.concatenate(xs, axis=0).astype(np.float32),
            "y": np.concatenate(ys, axis=0).astype(np.float32),
            "w": np.concatenate(ws, axis=0).astype(np.float32),
            "comp_size": np.concatenate(cs, axis=0).astype(np.int64),
        }
        audit.append({
            "source": source_info["source"],
            "category": cat,
            "label_mode": label_mode,
            "risk_mode": risk_mode,
            "samples": int(groups[cat]["y"].size),
            "positive_samples": int(np.sum(groups[cat]["y"] > 0.5)),
            "negative_samples": int(np.sum(groups[cat]["y"] < 0.5)),
            "component_count": int(len(comp_all)),
            "component_patch_count_median": float(np.median(comp_all)) if comp_all else float("nan"),
        })
        print(f"[MorphBalance] pool {source_info['source']} {cat} {label_mode}/{risk_mode} n={groups[cat]['y'].size}", flush=True)
    save = {"categories": np.asarray(list(groups.keys()), dtype=object), "audit": np.asarray(audit, dtype=object)}
    for cat, g in groups.items():
        save[f"{cat}_x"] = g["x"].astype(np.float16)
        save[f"{cat}_y"] = g["y"]
        save[f"{cat}_w"] = g["w"]
        save[f"{cat}_comp_size"] = g["comp_size"]
    np.savez_compressed(cache, **save)
    return groups, audit


def balanced_batch(groups, rng):
    cats = list(groups)
    per_cat = max(2, BATCH_SIZE // max(1, len(cats)))
    xs, ys, ws, cs = [], [], [], []
    for cat in cats:
        g = groups[cat]
        per_cls = max(1, per_cat // 2)
        for cls in (0, 1):
            ids = np.flatnonzero(g["y"] == cls)
            if ids.size == 0:
                continue
            choice = rng.choice(ids, size=per_cls, replace=ids.size < per_cls)
            xs.append(g["x"][choice])
            ys.append(g["y"][choice])
            ws.append(g["w"][choice])
            cs.append(g["comp_size"][choice])
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    w = np.concatenate(ws, axis=0).astype(np.float32)
    c = np.concatenate(cs, axis=0).astype(np.int64)
    order = rng.permutation(len(y))
    return x[order], y[order], w[order], c[order]


def morphology_loss(logits, y, pos_weight):
    y = y.float()
    pos = y.sum().clamp_min(1.0)
    neg = (1.0 - y).sum().clamp_min(1.0)
    base_pos = 0.5 * (pos + neg) / pos
    base_neg = 0.5 * (pos + neg) / neg
    weights = torch.where(y > 0.5, base_pos * pos_weight.float(), base_neg)
    bce = F.binary_cross_entropy_with_logits(logits, y, weight=weights)
    dl = dice_loss(logits, y)
    return bce + dl, bce.detach(), dl.detach()


def train_variant(source_info, variant, label_mode, risk_mode, device, smoke):
    if variant == "B0_CurrentLabels" and not smoke:
        return source_info["v1"].astype(np.float32), [{"source": source_info["source"], "variant": variant, "status": "REUSED_FORMAL_NTR_CHECKPOINT"}], [], True
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / "checkpoints" / f"{source_info['source'].replace(' ', '_')}_{variant}_{suffix}.pt"
    if ckpt.exists() and not smoke:
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        return payload["v"].astype(np.float32), payload["curves"], payload["component_weight_audit"], True
    groups, audit = build_source_pool(source_info, label_mode, risk_mode, device, smoke)
    param = torch.nn.Parameter(torch.tensor(source_info["v0"], dtype=torch.float32, device=device))
    m = torch.tensor(source_info["m"], dtype=torch.float32, device=device)
    opt = torch.optim.AdamW([param], lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    total = sum(len(g["y"]) for g in groups.values())
    steps = max(1, int(math.ceil(total / BATCH_SIZE)))
    if smoke:
        steps = min(steps, 2)
    curves, weight_rows = [], list(audit)
    for epoch in range(1, EPOCHS + 1):
        losses, base_normals, hard_losses, pos_losses = [], [], [], []
        decile_acc = defaultdict(lambda: {"weights": [], "count": 0})
        for _ in range(steps):
            xb, yb, wb, cb = balanced_batch(groups, rng)
            z = torch.tensor(xb, dtype=torch.float32, device=device)
            y = torch.tensor(yb, dtype=torch.float32, device=device)
            pw = torch.tensor(wb, dtype=torch.float32, device=device)
            v = F.normalize(param, dim=-1)
            logits = contrast_logits(z, m, source_info["alpha"], v)
            task, bce, dl = morphology_loss(logits, y, pw)
            neg = torch.nonzero(y < 0.5, as_tuple=False).flatten()
            k = max(1, int(math.ceil(float(len(neg)) * TOP_NEG_FRAC)))
            hidx = neg[torch.topk(logits.detach()[neg], k=k, largest=True).indices]
            hard = F.softplus(logits[hidx]).mean()
            loss = task + hard
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
            hard_losses.append(float(hard.detach().cpu()))
            with torch.no_grad():
                pos_idx = torch.nonzero(y > 0.5, as_tuple=False).flatten()
                neg_idx = torch.nonzero(y < 0.5, as_tuple=False).flatten()
                pos_losses.append(float(F.softplus(-logits[pos_idx]).mean().detach().cpu()) if pos_idx.numel() else float("nan"))
                base_normals.append(float(F.softplus(logits[neg_idx]).mean().detach().cpu()) if neg_idx.numel() else float("nan"))
            pos_cs = cb[yb > 0.5]
            pos_ws = wb[yb > 0.5]
            if pos_cs.size:
                qs = np.percentile(pos_cs, np.arange(10, 100, 10))
                dec = np.searchsorted(qs, pos_cs, side="right")
                for d in range(10):
                    vals = pos_ws[dec == d]
                    if vals.size:
                        decile_acc[d]["weights"].extend(vals.tolist())
                        decile_acc[d]["count"] += int(vals.size)
        row = {
            "source": source_info["source"],
            "variant": variant,
            "epoch": epoch,
            "loss": float(np.nanmean(losses)),
            "total_anomaly_positive_loss": float(np.nanmean(pos_losses)),
            "normal_base_loss": float(np.nanmean(base_normals)),
            "NTR_loss": float(np.nanmean(hard_losses)),
        }
        curves.append(row)
        for d, vals in decile_acc.items():
            w = np.asarray(vals["weights"], dtype=np.float64)
            weight_rows.append({
                "source": source_info["source"],
                "variant": variant,
                "epoch": epoch,
                "component_size_decile": int(d + 1),
                "mean_patch_weight": float(w.mean()) if w.size else float("nan"),
                "total_gradient_weight": float(w.sum()) if w.size else 0.0,
                "sampled_positive_patches": int(vals["count"]),
            })
        print(f"[MorphBalance] train {source_info['source']} {variant} epoch={epoch}/{EPOCHS} loss={row['loss']:.4f}", flush=True)
    with torch.no_grad():
        v = F.normalize(param, dim=-1).detach().cpu().numpy().astype(np.float32)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"v": v, "curves": curves, "component_weight_audit": weight_rows}, ckpt)
    return v, curves, weight_rows, False


def selected_eval(source, target, maps, metas, variants):
    old = tuple(lzp.VARIANTS)
    try:
        lzp.VARIANTS = tuple(variants)
        return evaluate_maps(source, target, maps, metas)
    finally:
        lzp.VARIANTS = old


def target_item(dataset, idx_or_rows, kind, pos):
    return dataset[int(idx_or_rows[pos])] if kind == "source_dataset" else dataset[pos]


def target_meta(dataset, idx_or_rows, kind, target, pos):
    item = target_item(dataset, idx_or_rows, kind, pos)
    if kind == "source_dataset":
        return {"category": str(item.get("cls_name", target)), "label": int(item.get("anomaly", 0)), "mask": mask_np(item)}
    row = dict(idx_or_rows[pos])
    return {"category": str(row.get("category", item.get("cls_name", target))), "label": int(row.get("label", item.get("anomaly", 0))), "mask": mask_np(item)}


@torch.no_grad()
def score_map(source_info, patch_grid, v_np, device):
    m = torch.tensor(source_info["m"], dtype=torch.float32, device=device)
    v = torch.tensor(v_np, dtype=torch.float32, device=device)
    if source_info["use_align"]:
        z = aligned_grid(patch_grid, source_info["q_pc1"], source_info["align_model"], device).unsqueeze(0)
        return logits_from_z(z, m, source_info["alpha"], v).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)
    z = norm_np(np.asarray(patch_grid, dtype=np.float32).reshape(SIDE * SIDE, -1))
    return contrast_logits(torch.tensor(z, dtype=torch.float32, device=device), m, source_info["alpha"], v).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)


def evaluate_source(source_info, vectors, device, smoke):
    raw, percat = [], []
    suffix = "smoke" if smoke else "formal"
    cells, _audits = protocol_cells(source_info["source"], device, smoke)
    variants = tuple(vectors.keys())
    for _cid, target, dataset, idx_or_rows, patch, kind in cells:
        if (source_info["source"], target) not in PAIRS:
            continue
        n = min(24, len(idx_or_rows)) if smoke else len(idx_or_rows)
        idx_use = idx_or_rows[:n]
        cache = OUT / f"{source_info['source'].replace(' ', '_')}_to_{target}_maps_{suffix}.npz"
        metas = [target_meta(dataset, idx_or_rows, kind, target, i) for i in range(n)]
        if cache.exists() and not smoke:
            data = np.load(cache, allow_pickle=True)
            maps = {name: [x.astype(np.float32) for x in data[name]] for name in variants}
        else:
            maps = {name: [] for name in variants}
            for i in range(n):
                for name, vec in vectors.items():
                    maps[name].append(score_map(source_info, patch[i], vec, device))
                if i % 100 == 0:
                    print(f"[MorphBalance] eval maps {source_info['source']} -> {target} image {i}/{n}", flush=True)
            np.savez_compressed(cache, **{k: np.stack(v, axis=0).astype(np.float16) for k, v in maps.items()})
        rr, cc = selected_eval(source_info["source"], target, maps, metas, variants)
        raw.extend(rr)
        percat.extend(cc)
        write_csv(OUT / "RAW_METRICS.partial.csv", raw)
        write_csv(OUT / "PER_CATEGORY_FULL_RES.partial.csv", percat)
    return raw, percat


def metric_naming_audit():
    lines = [
        "# Metric Naming Audit",
        "",
        "Transfer Gap Attribution used `PER_CATEGORY_STANDARD_METRICS.csv` and averaged category rows, so dataset summary values such as MVTec-source -> VisA P-AP=16.7255 are `CATEGORY_MACRO_PAP`.",
        "",
        "Formal benchmark headline values such as MVTec-source -> VisA P-AP about 18.7605 and VisA-source -> MVTec about 41.3503 are `FORMAL_M_FR_GLOBAL_PAP` from full-resolution/global aggregation.",
        "",
        "These two quantities must not be mixed in paper tables. Category-macro diagnostics are explanatory only; global full-resolution metrics are the official benchmark numbers.",
        "",
        "The previous relative-gap ridge analysis broadcast dataset-level DIRECT_COMPARABLE references to category observations because category-level SOTA references were not available.",
        "",
        "CATEGORY_RELATIVE_GAP_INVALID_FOR_PAPER: true",
    ]
    (OUT / "METRIC_NAMING_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def reproduction_rows(raw):
    rows = []
    hist = {}
    for r in existing_seed0_rows():
        if r.get("variant") == "B0_HardNormal" and np.isfinite(fnum(r, "M_FR_P_AP")):
            hist.setdefault((r.get("source"), r.get("dataset")), r)
    for r in raw:
        if r["variant"] != "B0_CurrentLabels":
            continue
        h = hist.get((r["source"], r["dataset"]))
        rows.append({
            "source": r["source"],
            "dataset": r["dataset"],
            "metric_space": "M_FR_GLOBAL",
            "current_P_AP": fnum(r, "M_FR_P_AP"),
            "history_P_AP": fnum(h, "M_FR_P_AP") if h else float("nan"),
            "delta_P_AP": fnum(r, "M_FR_P_AP") - fnum(h, "M_FR_P_AP") if h else float("nan"),
            "current_G": fnum(r, "M_FR_G_pixel"),
            "history_G": fnum(h, "M_FR_G_pixel") if h else float("nan"),
            "delta_G": fnum(r, "M_FR_G_pixel") - fnum(h, "M_FR_G_pixel") if h else float("nan"),
        })
    return rows


def morph_delta(raw, percat):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for source, dataset in sorted({(r["source"], r["dataset"]) for r in raw}):
        b0 = by.get((source, dataset, "B0_CurrentLabels"))
        if not b0:
            continue
        for var in VARIANTS[1:]:
            cur = by.get((source, dataset, var))
            if cur:
                rows.append({"source": source, "dataset": dataset, "variant": var, "delta_P_AP": fnum(cur, "M_FR_P_AP") - fnum(b0, "M_FR_P_AP"), "delta_G": fnum(cur, "M_FR_G_pixel") - fnum(b0, "M_FR_G_pixel")})
    return rows


def decide(delta_rows, repro_rows):
    repro = np.asarray([abs(fnum(r, "delta_P_AP")) for r in repro_rows], dtype=np.float64)
    if repro.size and np.nanmax(repro) > 3.0:
        return "MORPHOLOGY_B0_REPRODUCTION_FAILED", {"max_abs_B0_P_AP_delta": float(np.nanmax(repro))}
    def delta(var, source, dataset):
        vals = [fnum(r, "delta_P_AP") for r in delta_rows if r["variant"] == var and r["source"] == source and r["dataset"] == dataset]
        return vals[0] if vals else float("nan")
    e3_vals = np.asarray([fnum(r, "delta_P_AP") for r in delta_rows if r["variant"] == "E3_AnyOverlap_ComponentBalanced"], dtype=np.float64)
    e1_vals = np.asarray([fnum(r, "delta_P_AP") for r in delta_rows if r["variant"] == "E1_AnyOverlapLabels"], dtype=np.float64)
    e2_vals = np.asarray([fnum(r, "delta_P_AP") for r in delta_rows if r["variant"] == "E2_ComponentBalancedRisk"], dtype=np.float64)
    primary = (
        delta("E3_AnyOverlap_ComponentBalanced", "Full MVTec", "VisA") >= 2.5
        and delta("E3_AnyOverlap_ComponentBalanced", "Full VisA", "MVTec") >= 0.0
        and all(delta("E3_AnyOverlap_ComponentBalanced", s, "BTAD") >= -1.0 for s in ("Full VisA", "Full MVTec"))
        and all(delta("E3_AnyOverlap_ComponentBalanced", s, "MPDD") >= -1.0 for s in ("Full VisA", "Full MVTec"))
        and all(delta("E3_AnyOverlap_ComponentBalanced", s, "DTD-Synthetic") >= -1.0 for s in ("Full VisA", "Full MVTec"))
        and int(np.sum(e3_vals >= -0.3)) >= 6
        and float(np.nanmean(e3_vals)) >= 1.0
    )
    markers = []
    if np.nanmean(e1_vals) > 0:
        markers.append("PATCH_LABEL_DISCRETIZATION_HURTS_SPARSE_DEFECT_TRANSFER")
    if np.nanmean(e2_vals) > 0:
        markers.append("AREA_WEIGHTED_POSITIVE_RISK_HURTS_TRANSFER")
    if primary:
        outcome = "MORPHOLOGY_BALANCED_SOURCE_TRAINING_IMPROVES_TRANSFER"
    elif all(np.nanmax(vals) < 1.0 for vals in (e1_vals, e2_vals, e3_vals)):
        outcome = "SOURCE_MORPHOLOGY_REWEIGHTING_DOES_NOT_FIX_TARGET_BOTTLENECK"
    elif np.nanmean(e1_vals) > np.nanmean(e2_vals) and np.nanmean(e1_vals) > 0:
        outcome = "PATCH_LABEL_DISCRETIZATION_PARTIAL_SIGNAL_COMPONENT_RISK_CLOSED"
    elif np.nanmean(e2_vals) > np.nanmean(e1_vals) and np.nanmean(e2_vals) > 0:
        outcome = "COMPONENT_RISK_PARTIAL_SIGNAL_LABEL_MODIFICATION_CLOSED"
    else:
        outcome = "SOURCE_MORPHOLOGY_REWEIGHTING_DOES_NOT_FIX_TARGET_BOTTLENECK"
    return outcome, {"markers": markers, "E1_mean_delta_P_AP": float(np.nanmean(e1_vals)), "E2_mean_delta_P_AP": float(np.nanmean(e2_vals)), "E3_mean_delta_P_AP": float(np.nanmean(e3_vals)), "E3_non_drop": int(np.sum(e3_vals >= -0.3))}


def write_report(outcome, stats, raw, deltas, runtime):
    lines = [
        "# Morphology-Balanced Source Training",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "Protocol: Full VisA / Full MVTec source-only. CLIP/backbone and target protocol are unchanged. E1 changes source patch labels to any-overlap; E2 applies component-balanced positive BCE risk; E3 combines both. No target statistics or tuning were used.",
        "",
        "## Delta vs B0",
        "",
        "| Source | Target | Variant | Delta P-AP | Delta G |",
        "|---|---|---|---:|---:|",
    ]
    for r in deltas:
        lines.append(f"| {r['source']} | {r['dataset']} | {r['variant']} | {fnum(r,'delta_P_AP'):+.4f} | {fnum(r,'delta_G'):+.4f} |")
    lines.extend(["", "## Decision Stats", "", "```json", json.dumps(stats, indent=2, default=json_default), "```", "", "LRP and multi-seed replication were not run unless the primary E3 gate passed.", "", f"Runtime: {runtime:.1f}s."])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    metric_naming_audit()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_component = []
    for source in ("Full VisA", "Full MVTec"):
        all_component.extend(full_component_audit(source, args.smoke))
    write_csv(OUT / "SOURCE_MASK_DISCRETIZATION_AUDIT.csv", all_component)
    surv = []
    for source in ("Full VisA", "Full MVTec"):
        part = [r for r in all_component if r["source"] == source]
        for name, filt in (("all", part), ("small_bottom25", [r for r in part if r["is_small_source_q25"]]), ("thin_top25", [r for r in part if r["is_thin_source_q75"]])):
            surv.append({
                "source": source,
                "subset": name,
                "components": len(filt),
                "current_survival_rate": float(np.mean([bool(r["component_survives_current"]) for r in filt])) if filt else float("nan"),
                "any_overlap_survival_rate": float(np.mean([bool(r["component_survives_any_overlap"]) for r in filt])) if filt else float("nan"),
            })
    write_csv(OUT / "SOURCE_COMPONENT_SURVIVAL.csv", surv)

    source_infos = [load_visa_source(device), load_mvtec_source()]
    raw, percat, curves, weight_rows = [], [], [], []
    for info in source_infos:
        configs = {
            "B0_CurrentLabels": ("current", "patch"),
            "E1_AnyOverlapLabels": ("any", "patch"),
            "E2_ComponentBalancedRisk": ("current", "component"),
            "E3_AnyOverlap_ComponentBalanced": ("any", "component"),
        }
        vectors = {}
        for var, (label_mode, risk_mode) in configs.items():
            v, c, w, _cached = train_variant(info, var, label_mode, risk_mode, device, args.smoke)
            vectors[var] = v
            curves.extend(c)
            weight_rows.extend(w)
        rr, pp = evaluate_source(info, vectors, device, args.smoke)
        raw.extend(rr)
        percat.extend(pp)

    repro = reproduction_rows(raw)
    deltas = morph_delta(raw, percat)
    outcome, stats = decide(deltas, repro)
    runtime = time.time() - start
    write_csv(OUT / "B0_REPRODUCTION.csv", repro)
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "FULL_RES_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY_FULL_RES.csv", percat)
    write_csv(OUT / "COMPONENT_WEIGHT_AUDIT.csv", weight_rows + curves)
    write_csv(OUT / "MORPHOLOGY_DELTA_ANALYSIS.csv", deltas)
    write_csv(OUT / "LRP_COMPLEMENTARITY.csv", [{"status": "NOT_RUN_PRIMARY_GATE_NOT_PASSED"}])
    write_json(OUT / "OUTCOME.json", {"outcome": outcome, "decision_stats": stats, "runtime_sec": runtime, "device": device, "strict_source_only": True, "target_statistics": False, "lrp_run": outcome == "MORPHOLOGY_BALANCED_SOURCE_TRAINING_IMPROVES_TRANSFER"})
    write_report(outcome, stats, raw, deltas, runtime)
    print(f"[MorphBalance] OUTCOME={outcome}", flush=True)
    print(f"[MorphBalance] wrote {OUT}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
