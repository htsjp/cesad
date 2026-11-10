import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_final_cross_dataset_benchmark import load_mvtec_source, load_visa_source, protocol_cells
from experiments.phase_hard_normal_cross_source_mvtec import load_full_mvtec_source
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import (
    item_at,
    item_meta,
    mask_np,
    quadrant_crops,
    torch_resize,
)
from experiments.phase_private_residual_contrast_memory import contrast_logits
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import evaluate_maps
from experiments.phase_spatial_residual_refinement_standard_metric import aligned_grid, fnum
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import SEED, balanced_loss
from experiments.phase_xlv_cross_domain_transfer import CHECKPOINT
from experiments.phase_xxxviii_visual_state_representation import IMAGE_SIZE, SIDE, norm_np, patch_labels
from phase_xa.official_task_forward import normalized_patch_feature, visual_outputs
from utils.transforms import get_transform


OUT = Path("results/phase_orsl_observation_robust_state_learning/formal")
VARIANTS = ("E0_Baseline", "E1_SharedMultiObservationState", "E2_WorstViewORSL")
PILOT_PAIRS = (
    ("Full MVTec", "VisA"),
    ("Full VisA", "MPDD"),
    ("Full MVTec", "BTAD"),
    ("Full VisA", "DTD-Synthetic"),
)
FULL_PAIRS = (
    ("Full VisA", "MVTec"),
    ("Full VisA", "BTAD"),
    ("Full VisA", "MPDD"),
    ("Full VisA", "DTD-Synthetic"),
    ("Full MVTec", "VisA"),
    ("Full MVTec", "BTAD"),
    ("Full MVTec", "MPDD"),
    ("Full MVTec", "DTD-Synthetic"),
)
EPOCHS = 10
TOP_NEG_FRAC = 0.10
MAX_SOURCE_IMAGES = 480
SMOKE_SOURCE_IMAGES = 24
ORSL_LR = 0.001


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


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def finite_mean(vals):
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def finite_median(vals):
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def source_refs(source, smoke):
    if source == "Full VisA":
        _name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    else:
        _name, dataset, indices, patch = load_full_mvtec_source(False)
    by = {}
    for pos, idx in enumerate(indices):
        item = dataset[int(idx)]
        key = (str(item.get("cls_name", source)), int(item.get("anomaly", 0)))
        by.setdefault(key, []).append((pos, int(idx)))
    limit = SMOKE_SOURCE_IMAGES if smoke else MAX_SOURCE_IMAGES
    rng = np.random.default_rng(SEED)
    chosen = []
    keys = sorted(by)
    while len(chosen) < limit and keys:
        progressed = False
        for key in keys:
            bucket = by[key]
            if not bucket:
                continue
            j = int(rng.integers(0, len(bucket)))
            chosen.append(bucket.pop(j))
            progressed = True
            if len(chosen) >= limit:
                break
        if not progressed:
            break
    return dataset, indices, patch, chosen


def canonicalize(model_info, z):
    z = F.normalize(z.float(), dim=-1)
    if not model_info["use_align"]:
        return z
    q = torch.tensor(model_info["q_pc1"], dtype=torch.float32, device=z.device)
    flat = z.reshape(-1, z.shape[-1])
    flat = flat - (flat @ q).unsqueeze(-1) * q.unsqueeze(0)
    flat = F.normalize(flat, dim=-1)
    flat = model_info["align_model"].adapt(flat)
    return F.normalize(flat, dim=-1).reshape_as(z)


@torch.no_grad()
def local_features_from_path(model_info, clip_model, preprocess, path, device):
    crops = quadrant_crops(path)
    batch = torch.stack([preprocess(c) for c in crops], dim=0).to(device)
    out = visual_outputs(clip_model, batch, adapter=None, grad=False)
    patches = normalized_patch_feature(out).float()
    return canonicalize(model_info, patches).detach().cpu().numpy().astype(np.float16)


def local_labels_from_mask(mask):
    m = torch.tensor(np.asarray(mask, dtype=np.float32)).view(1, 1, *mask.shape)
    h, w = mask.shape
    boxes = [
        (0, 0, w // 2, h // 2),
        (w // 2, 0, w, h // 2),
        (0, h // 2, w // 2, h),
        (w // 2, h // 2, w, h),
    ]
    labels = []
    for x1, y1, x2, y2 in boxes:
        crop = m[:, :, y1:y2, x1:x2]
        small = F.interpolate(crop, size=(SIDE, SIDE), mode="bilinear", align_corners=False).view(-1)
        labels.append((small > 0.5).numpy().astype(np.float32))
    return np.stack(labels, axis=0).astype(np.float32)


def build_source_observation_pool(source_info, clip_model, preprocess, device, smoke):
    suffix = "smoke" if smoke else "formal"
    cache = OUT / f"{source_info['source'].replace(' ', '_')}_OBS_POOL_{suffix}.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        return {
            "global_x": data["global_x"],
            "global_y": data["global_y"].astype(np.float32),
            "local_x": data["local_x"],
            "local_y": data["local_y"].astype(np.float32),
            "categories": data["categories"].astype(str),
            "labels": data["labels"].astype(np.int64),
        }, data["audit"].tolist()

    dataset, indices, patch, chosen = source_refs(source_info["source"], smoke)
    gx, gy, lx, ly, cats, labs, audit = [], [], [], [], [], [], []
    for k, (pos, idx) in enumerate(chosen):
        if k % 25 == 0:
            print(f"[ORSL] source pool {source_info['source']} image {k}/{len(chosen)}", flush=True)
        item = dataset[int(idx)]
        cat = str(item.get("cls_name", source_info["source"]))
        label = int(item.get("anomaly", 0))
        path = str(item.get("img_path", ""))
        if not path:
            continue
        if source_info["use_align"]:
            g = aligned_grid(patch[pos], source_info["q_pc1"], source_info["align_model"], device).detach().cpu().numpy().astype(np.float32)
        else:
            g = norm_np(np.asarray(patch[pos], dtype=np.float32).reshape(SIDE * SIDE, 1024)).astype(np.float32)
        y = patch_labels(item).astype(np.float32)
        loc_x = local_features_from_path(source_info, clip_model, preprocess, path, device)
        loc_y = local_labels_from_mask(mask_np(item))
        gx.append(g.astype(np.float16))
        gy.append(y.astype(np.float32))
        lx.append(loc_x.astype(np.float16))
        ly.append(loc_y.astype(np.float32))
        cats.append(cat)
        labs.append(label)
        audit.append({
            "source": source_info["source"],
            "category": cat,
            "image_index": int(pos),
            "dataset_index": int(idx),
            "image_label": label,
            "global_positive_patches": int(y.sum()),
            "local_positive_patches": int(loc_y.sum()),
        })
    pool = {
        "global_x": np.stack(gx, axis=0).astype(np.float16),
        "global_y": np.stack(gy, axis=0).astype(np.float32),
        "local_x": np.stack(lx, axis=0).astype(np.float16),
        "local_y": np.stack(ly, axis=0).astype(np.float32),
        "categories": np.asarray(cats, dtype=object),
        "labels": np.asarray(labs, dtype=np.int64),
    }
    np.savez_compressed(cache, **pool, audit=np.asarray(audit, dtype=object))
    return pool, audit


def sample_image_batch(pool, rng, batch_images):
    cats = np.asarray(pool["categories"]).astype(str)
    labels = np.asarray(pool["labels"], dtype=np.int64)
    chosen = []
    keys = sorted({(c, int(l)) for c, l in zip(cats, labels)})
    per_key = max(1, math.ceil(batch_images / max(len(keys), 1)))
    for key in keys:
        ids = np.flatnonzero((cats == key[0]) & (labels == key[1]))
        if ids.size:
            take = min(per_key, max(1, ids.size))
            chosen.extend(rng.choice(ids, size=take, replace=ids.size < take).tolist())
    if len(chosen) > batch_images:
        chosen = rng.choice(np.asarray(chosen), size=batch_images, replace=False).tolist()
    return np.asarray(chosen, dtype=np.int64)


def logits_for(z, source_info, v):
    m = torch.tensor(source_info["m"], dtype=torch.float32, device=z.device)
    return contrast_logits(z.reshape(-1, z.shape[-1]), m, float(source_info["alpha"]), v).reshape(z.shape[:-1])


def ntr_loss_from_logits(logits, y):
    neg = torch.nonzero(y.reshape(-1) < 0.5, as_tuple=False).flatten()
    if neg.numel() == 0:
        return torch.zeros((), dtype=logits.dtype, device=logits.device), 0
    flat = logits.reshape(-1)
    k = max(1, int(math.ceil(float(neg.numel()) * TOP_NEG_FRAC)))
    idx = neg[torch.topk(flat.detach()[neg], k=k, largest=True).indices]
    return F.softplus(flat[idx]).mean(), int(k)


def state_loss(logits, y):
    return balanced_loss(logits.reshape(-1), y.reshape(-1))[0]


def train_variant(source_info, pool, variant, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / "checkpoints" / f"{source_info['source'].replace(' ', '_')}_{variant}_{suffix}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        return payload["v"].astype(np.float32), payload["diagnostics"], True
    if variant == "E0_Baseline":
        return source_info["v1"].astype(np.float32), [], True

    param = torch.nn.Parameter(torch.tensor(source_info["v0"], dtype=torch.float32, device=device))
    opt = torch.optim.AdamW([param], lr=ORSL_LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    n_img = int(pool["global_x"].shape[0])
    batch_images = 4 if smoke else 8
    steps = max(1, math.ceil(n_img / batch_images))
    if smoke:
        steps = min(steps, 2)
    diagnostics = []
    for epoch in range(1, EPOCHS + 1):
        rows = []
        for _step in range(steps):
            ids = sample_image_batch(pool, rng, batch_images)
            gx = torch.tensor(pool["global_x"][ids], dtype=torch.float32, device=device)
            gy = torch.tensor(pool["global_y"][ids], dtype=torch.float32, device=device)
            lx = torch.tensor(pool["local_x"][ids], dtype=torch.float32, device=device)
            ly = torch.tensor(pool["local_y"][ids], dtype=torch.float32, device=device)
            v = F.normalize(param, dim=-1)
            g_logits = logits_for(gx, source_info, v)
            l_logits = logits_for(lx, source_info, v)
            gy_view = gy.reshape_as(g_logits)
            ly_view = ly.reshape_as(l_logits)
            loss_g = state_loss(g_logits, gy_view)
            loss_l = state_loss(l_logits, ly_view)
            if variant == "E1_SharedMultiObservationState":
                ntr, hard_count = ntr_loss_from_logits(g_logits, gy_view)
                loss = loss_g + loss_l + ntr
                worst_anom = torch.zeros((), dtype=loss.dtype, device=device)
                cross_ntr = ntr
            else:
                local74 = l_logits.reshape(-1, 4, SIDE, SIDE)
                stitched = torch.zeros((local74.shape[0], SIDE * 2, SIDE * 2), dtype=local74.dtype, device=device)
                stitched[:, :SIDE, :SIDE] = local74[:, 0]
                stitched[:, :SIDE, SIDE:] = local74[:, 1]
                stitched[:, SIDE:, :SIDE] = local74[:, 2]
                stitched[:, SIDE:, SIDE:] = local74[:, 3]
                local_down = F.interpolate(stitched.unsqueeze(1), size=(SIDE, SIDE), mode="bilinear", align_corners=True).squeeze(1).reshape_as(g_logits)
                p_g = torch.sigmoid(g_logits)
                p_l = torch.sigmoid(local_down)
                pos = gy_view > 0.5
                if pos.any():
                    p_min = torch.minimum(p_g[pos], p_l[pos])
                    worst_anom = -torch.log(p_min.clamp_min(1e-6)).mean()
                else:
                    worst_anom = torch.zeros((), dtype=g_logits.dtype, device=device)
                risk_logits = torch.logit(torch.maximum(p_g, p_l).clamp(1e-6, 1 - 1e-6))
                cross_ntr, hard_count = ntr_loss_from_logits(risk_logits, gy_view)
                loss = loss_g + loss_l + worst_anom + cross_ntr
            with torch.no_grad():
                local74 = l_logits.reshape(-1, 4, SIDE, SIDE)
                stitched = torch.zeros((local74.shape[0], SIDE * 2, SIDE * 2), dtype=local74.dtype, device=device)
                stitched[:, :SIDE, :SIDE] = local74[:, 0]
                stitched[:, :SIDE, SIDE:] = local74[:, 1]
                stitched[:, SIDE:, :SIDE] = local74[:, 2]
                stitched[:, SIDE:, SIDE:] = local74[:, 3]
                local_down_diag = F.interpolate(stitched.unsqueeze(1), size=(SIDE, SIDE), mode="bilinear", align_corners=True).squeeze(1).reshape_as(g_logits)
                disagreement = torch.abs(torch.sigmoid(g_logits) - torch.sigmoid(local_down_diag)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            rows.append({
                "loss": float(loss.detach().cpu()),
                "global_state_loss": float(loss_g.detach().cpu()),
                "local_state_loss": float(loss_l.detach().cpu()),
                "worst_view_anomaly_loss": float(worst_anom.detach().cpu()),
                "cross_view_normal_ntr_loss": float(cross_ntr.detach().cpu()),
                "global_local_anomaly_score_disagreement": float(disagreement.detach().cpu()),
                "hard_patch_count": hard_count,
            })
        row = {"source": source_info["source"], "variant": variant, "epoch": epoch}
        for key in rows[0]:
            row[key] = finite_mean([r[key] for r in rows])
        diagnostics.append(row)
        print(f"[ORSL] train {source_info['source']} {variant} epoch={epoch}/{EPOCHS} loss={row['loss']:.4f} disag={row['global_local_anomaly_score_disagreement']:.4f}", flush=True)
    with torch.no_grad():
        v = F.normalize(param, dim=-1).detach().cpu().numpy().astype(np.float32)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"v": v, "diagnostics": diagnostics}, ckpt)
    return v, diagnostics, False


@torch.no_grad()
def score_feature_grid(model_info, patch_np, v_np, device):
    v = torch.tensor(v_np, dtype=torch.float32, device=device)
    if model_info["use_align"]:
        z = aligned_grid(patch_np, model_info["q_pc1"], model_info["align_model"], device).unsqueeze(0)
        return logits_for(z, model_info, v).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)
    z = norm_np(np.asarray(patch_np, dtype=np.float32).reshape(SIDE * SIDE, 1024))
    zt = torch.tensor(z, dtype=torch.float32, device=device)
    return logits_for(zt, model_info, v).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)


@torch.no_grad()
def local_zoom_features_np(model_info, clip_model, preprocess, img_path, device):
    crops = quadrant_crops(img_path)
    batch = torch.stack([preprocess(c) for c in crops], dim=0).to(device)
    out = visual_outputs(clip_model, batch, adapter=None, grad=False)
    patches = normalized_patch_feature(out).float()
    return canonicalize(model_info, patches).detach().cpu().numpy().astype(np.float16)


@torch.no_grad()
def local_zoom_map_from_features(model_info, z_np, v_np, device):
    z = torch.tensor(np.asarray(z_np, dtype=np.float32), dtype=torch.float32, device=device)
    v = torch.tensor(v_np, dtype=torch.float32, device=device)
    scores = [logits_for(z[i], model_info, v).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32) for i in range(4)]
    s = np.zeros((SIDE * 2, SIDE * 2), dtype=np.float32)
    s[:SIDE, :SIDE] = scores[0]
    s[:SIDE, SIDE:] = scores[1]
    s[SIDE:, :SIDE] = scores[2]
    s[SIDE:, SIDE:] = scores[3]
    return s


def build_maps(model_info, vectors, clip_model, preprocess, patch, refs, dataset, kind, target, device, smoke):
    n = min(24, len(refs)) if smoke else len(refs)
    maps = {v: [] for v in VARIANTS}
    metas = []
    for pos, ref in enumerate(refs[:n]):
        if pos % 25 == 0:
            print(f"[ORSL] {model_info['source']} -> {target} image {pos}/{n}", flush=True)
        item = item_at(dataset, kind, ref, pos)
        cat, label, path = item_meta(item, kind, ref, target)
        if not path:
            raise ValueError(f"missing img_path for {model_info['source']} -> {target} pos={pos}")
        metas.append({"category": cat, "label": label, "mask": mask_np(item), "img_path": path})
        local_z = local_zoom_features_np(model_info, clip_model, preprocess, path, device)
        for variant, v_np in vectors.items():
            global37 = score_feature_grid(model_info, patch[pos], v_np, device)
            local74 = local_zoom_map_from_features(model_info, local_z, v_np, device)
            global74 = torch_resize(global37, (SIDE * 2, SIDE * 2), align_corners=True)
            final74 = 0.5 * global74 + 0.5 * local74
            maps[variant].append(final74.astype(np.float32))
    return maps, metas


def selected_eval(source, target, maps, metas):
    import experiments.phase_parameter_free_local_zoom_pyramid_fullres as lzp
    old = tuple(lzp.VARIANTS)
    try:
        lzp.VARIANTS = tuple(VARIANTS)
        return evaluate_maps(source, target, maps, metas)
    finally:
        lzp.VARIANTS = old


def morphology_rows(raw_percat):
    by = {(r["source"], r["dataset"], r["category"], r["variant"], r["protocol"]): r for r in raw_percat}
    rows = []
    groups = {}
    for key, r in by.items():
        if r["variant"] != "E0_Baseline" or r["protocol"] != "M":
            continue
        area = fnum(r, "M_FR_P_AP")
        # Category morphology labels are unavailable in this phase output; use
        # prior bottleneck categories if present, otherwise leave conservative.
        _ = area
    for source, dataset, category, _variant, protocol in sorted(by):
        if protocol != "M":
            continue
        b = by.get((source, dataset, category, "E0_Baseline", "M"))
        if not b:
            continue
        for variant in ("E1_SharedMultiObservationState", "E2_WorstViewORSL"):
            e = by.get((source, dataset, category, variant, "M"))
            if not e:
                continue
            rows.append({
                "source": source,
                "dataset": dataset,
                "category": category,
                "variant": variant,
                "Delta_M_FR_P_AP": fnum(e, "M_FR_P_AP") - fnum(b, "M_FR_P_AP"),
                "group_small": "",
                "group_thin": "",
                "group_large": "",
                "group_thick": "",
            })
    return rows


def delta_summary(raw, pairs):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for variant in ("E1_SharedMultiObservationState", "E2_WorstViewORSL"):
        vals = []
        for source, target in pairs:
            b = by.get((source, target, "E0_Baseline"))
            e = by.get((source, target, variant))
            if not (b and e):
                continue
            d = fnum(e, "M_FR_P_AP") - fnum(b, "M_FR_P_AP")
            vals.append(d)
            rows.append({
                "source": source,
                "dataset": target,
                "variant": variant,
                "Delta_M_FR_P_AUROC": fnum(e, "M_FR_P_AUROC") - fnum(b, "M_FR_P_AUROC"),
                "Delta_M_FR_P_AP": d,
                "Delta_M_FR_P_F1max": fnum(e, "M_FR_P_F1max") - fnum(b, "M_FR_P_F1max"),
                "Delta_M_FR_PRO": fnum(e, "M_FR_PRO") - fnum(b, "M_FR_PRO"),
            })
        rows.append({
            "source": "SUMMARY",
            "dataset": "PILOT",
            "variant": variant,
            "mean_delta_M_FR_P_AP": finite_mean(vals),
            "median_delta_M_FR_P_AP": finite_median(vals),
            "improved": int(np.sum(np.asarray(vals) > 0)) if vals else 0,
            "non_drop_ge_minus_0_2": int(np.sum(np.asarray(vals) >= -0.2)) if vals else 0,
            "total": int(len(vals)),
        })
    return rows


def geometry_rows(source_infos, vectors_by_source):
    rows = []
    for source, info in source_infos.items():
        b = info["v1"] / max(float(np.linalg.norm(info["v1"])), 1e-12)
        for variant, v in vectors_by_source[source].items():
            vv = v / max(float(np.linalg.norm(v)), 1e-12)
            rows.append({
                "source": source,
                "variant": variant,
                "cos_d_E0": float(np.dot(b, vv)),
                "state_direction_norm": float(np.linalg.norm(v)),
            })
    return rows


def decide(summary):
    rows = [r for r in summary if r.get("source") == "SUMMARY"]
    best = max(rows, key=lambda r: fnum(r, "mean_delta_M_FR_P_AP")) if rows else None
    if not best:
        return "OBSERVATION_ROBUST_STATE_LEARNING_DOES_NOT_IMPROVE_TRANSFER", {}
    mean_delta = fnum(best, "mean_delta_M_FR_P_AP")
    improved = int(float(best.get("improved", 0)))
    vals = [r for r in summary if r.get("variant") == best["variant"] and r.get("source") != "SUMMARY"]
    focus = [r for r in vals if (r["source"], r["dataset"]) in {("Full MVTec", "VisA"), ("Full VisA", "MPDD")}]
    focus_gain = max([fnum(r, "Delta_M_FR_P_AP") for r in focus], default=float("nan"))
    catastrophic = any(fnum(r, "Delta_M_FR_P_AP") < -5.0 for r in vals)
    gate = mean_delta >= 1.0 and improved >= 3 and not catastrophic and focus_gain > 0
    stats = {"best_variant": best["variant"], "mean_delta_P_AP": mean_delta, "improved": improved, "focus_best_delta_P_AP": focus_gain, "catastrophic_drop": catastrophic, "pilot_gate_pass": gate}
    if gate:
        if mean_delta >= 1.5:
            return "OBSERVATION_ROBUST_STATE_LEARNING_SIGNIFICANTLY_IMPROVES_TRANSFER", stats
        return "MULTI_OBSERVATION_STATE_LEARNING_HAS_WEAK_VALUE", stats
    e1 = next((r for r in rows if r["variant"] == "E1_SharedMultiObservationState"), None)
    e2 = next((r for r in rows if r["variant"] == "E2_WorstViewORSL"), None)
    if e1 and e2 and fnum(e2, "mean_delta_M_FR_P_AP") <= fnum(e1, "mean_delta_M_FR_P_AP") + 0.2 and fnum(e1, "mean_delta_M_FR_P_AP") > 0:
        return "WORST_VIEW_OBJECTIVE_HAS_NO_ADDED_VALUE", stats
    return "OBSERVATION_ROBUST_STATE_LEARNING_DOES_NOT_IMPROVE_TRANSFER", stats


def write_report(raw, summary, geom, outcome, stats, runtime):
    lines = [
        "# Phase ORSL: Observation-Robust State Learning",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "Baseline is Canonical Two-State + NTR with the locked fixed 2x2 Local Re-encoding Pyramid at inference. CLIP, alignment/interface, and LRP are frozen; ORSL only changes source-side state-direction training.",
        "",
        "## Pilot Metrics",
        "",
        "| Source | Target | Variant | P-AUROC | P-AP | P-F1max | PRO |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in raw:
        if (r["source"], r["dataset"]) in PILOT_PAIRS:
            lines.append(f"| {r['source']} | {r['dataset']} | {r['variant']} | {fnum(r,'M_FR_P_AUROC'):.4f} | {fnum(r,'M_FR_P_AP'):.4f} | {fnum(r,'M_FR_P_F1max'):.4f} | {fnum(r,'M_FR_PRO'):.4f} |")
    lines.extend(["", "## Delta vs E0", "", "| Source | Target | Variant | Delta P-AP |", "|---|---|---|---:|"])
    for r in summary:
        if r.get("source") != "SUMMARY":
            lines.append(f"| {r['source']} | {r['dataset']} | {r['variant']} | {fnum(r,'Delta_M_FR_P_AP'):+.4f} |")
    lines.extend(["", "## Summary", "", "| Variant | mean Delta P-AP | median | improved | non-drop |", "|---|---:|---:|---:|---:|"])
    for r in summary:
        if r.get("source") == "SUMMARY":
            lines.append(f"| {r['variant']} | {fnum(r,'mean_delta_M_FR_P_AP'):+.4f} | {fnum(r,'median_delta_M_FR_P_AP'):+.4f} | {int(float(r.get('improved',0)))}/{int(float(r.get('total',0)))} | {int(float(r.get('non_drop_ge_minus_0_2',0)))}/{int(float(r.get('total',0)))} |")
    lines.extend(["", "## State Geometry", "", "| Source | Variant | cos(d,E0) | norm |", "|---|---|---:|---:|"])
    for r in geom:
        lines.append(f"| {r['source']} | {r['variant']} | {fnum(r,'cos_d_E0'):.6f} | {fnum(r,'state_direction_norm'):.6f} |")
    lines.extend(["", f"Decision stats: `{json.dumps(stats, default=json_default)}`", "", f"Runtime: {runtime:.1f}s."])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ORSL] device={device}", flush=True)
    preprocess, _target_transform = get_transform(argparse.Namespace(image_size=IMAGE_SIZE))
    clip_model, *_ = load_components(str(CHECKPOINT), device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    source_infos = {"Full VisA": load_visa_source(device), "Full MVTec": load_mvtec_source()}
    pool_audit, vectors_by_source, train_rows = [], {}, []
    for source, info in source_infos.items():
        pool, audit = build_source_observation_pool(info, clip_model, preprocess, device, args.smoke)
        pool_audit.extend(audit)
        vectors_by_source[source] = {}
        for variant in VARIANTS:
            v, diag, reused = train_variant(info, pool, variant, device, args.smoke)
            vectors_by_source[source][variant] = v
            for row in diag:
                row["checkpoint_reused"] = bool(reused)
                train_rows.append(row)
        del pool
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    raw, percat = [], []
    pairs = FULL_PAIRS if args.full else PILOT_PAIRS
    for source, target in pairs:
        info = source_infos[source]
        cells, _audits = protocol_cells(source, device, args.smoke)
        matches = [c for c in cells if c[1] == target]
        if not matches:
            raise RuntimeError(f"No target cell for {source}->{target}")
        _cell_id, _target, dataset, idx_or_rows, patch, kind = matches[0]
        n = min(24, len(idx_or_rows)) if args.smoke else len(idx_or_rows)
        print(f"[ORSL] evaluate {source}->{target} n={n}", flush=True)
        maps, metas = build_maps(info, vectors_by_source[source], clip_model, preprocess, patch[:n], idx_or_rows[:n], dataset, kind, target, device, args.smoke)
        rr, cc = selected_eval(source, target, maps, metas)
        raw.extend(rr)
        percat.extend(cc)
        write_csv(OUT / "RAW_METRICS.partial.csv", raw)
        write_csv(OUT / "PER_CATEGORY_FULL_RES.partial.csv", percat)

    summary = delta_summary(raw, pairs)
    geom = geometry_rows(source_infos, vectors_by_source)
    morph = morphology_rows(percat)
    outcome, stats = decide(summary)
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY_FULL_RES.csv", percat)
    write_csv(OUT / "PILOT_DELTA_SUMMARY.csv", summary)
    write_csv(OUT / "SOURCE_TRAINING_DIAGNOSTICS.csv", train_rows)
    write_csv(OUT / "SOURCE_OBSERVATION_POOL_AUDIT.csv", pool_audit)
    write_csv(OUT / "STATE_GEOMETRY.csv", geom)
    write_csv(OUT / "MORPHOLOGY_DIAGNOSTICS.csv", morph)
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": stats,
        "runtime_sec": runtime,
        "device": device,
        "strict_source_only": True,
        "target_training": False,
        "target_statistics": False,
        "target_tuning": False,
        "clip_frozen": True,
        "fixed_lrp_inference": True,
        "max_source_images": SMOKE_SOURCE_IMAGES if args.smoke else MAX_SOURCE_IMAGES,
        "epochs": EPOCHS,
        "lr": ORSL_LR,
        "top_neg_frac": TOP_NEG_FRAC,
        "full_stage": bool(args.full),
    })
    write_report(raw, summary, geom, outcome, stats, runtime)
    print(f"[ORSL] OUTCOME={outcome} runtime={runtime:.1f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--full", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
