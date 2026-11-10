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
from PIL import Image
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_cmab_compact_multi_anomaly_basis import load_mvtec_groups, load_visa_groups
from experiments.phase_final_cross_dataset_benchmark import load_mvtec_source, load_visa_source, protocol_cells
from experiments.phase_hard_normal_contrast import PATCH_SAMPLE_PER_CLASS_CATEGORY, TOP_NEG_FRAC
from experiments.phase_hard_normal_cross_source_mvtec import load_full_mvtec_source
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import item_at, item_meta, mask_np, torch_resize
from experiments.phase_private_residual_contrast_memory import unit_np
from experiments.phase_scee_lite_source_composition_experience_expansion import fnum, load_runtime, scorer_z_grid
from experiments.phase_sgpr import evaluate_maps, fixed_quadrant_boxes, stitch_quadrant74
from experiments.phase_ssec_state_space_experience_consolidation import balanced_batch, two_state_logits, two_state_loss
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, SEED
from experiments.phase_xxxviii_visual_state_representation import LR, SIDE, patch_labels
from phase_xa.official_task_forward import normalized_patch_feature, visual_outputs


OUT = Path("results/phase_ldsb_local_differential_state_branch/formal")
VARIANTS = ("E0_Baseline", "E1_DifferentialOnly", "E2_AbsPlusDifferential")
GATE_PAIRS = (("Full MVTec", "VisA"), ("Full VisA", "MPDD"))
PILOT_PAIRS = (
    ("Full MVTec", "VisA"),
    ("Full VisA", "MPDD"),
    ("Full MVTec", "BTAD"),
    ("Full VisA", "DTD-Synthetic"),
)
EPOCHS = 10
FUSION_WEIGHT = 0.25
GATE0_AUC_THRESHOLD = 55.0


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


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def safe_auc_np(y, s):
    y = np.asarray(y, dtype=np.uint8).reshape(-1)
    s = np.asarray(s, dtype=np.float32).reshape(-1)
    keep = np.isfinite(s)
    y, s = y[keep], s[keep]
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_s = s[order]
    n = len(sorted_s)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_s[j] == sorted_s[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    pos = y.astype(bool)
    n_pos = float(pos.sum())
    n_neg = float(y.size - pos.sum())
    return float((ranks[pos].sum() - n_pos * (n_pos + 1.0) / 2.0) / max(n_pos * n_neg, 1.0) * 100.0)


def safe_ap_np(y, s):
    y = np.asarray(y, dtype=np.uint8).reshape(-1)
    s = np.asarray(s, dtype=np.float32).reshape(-1)
    keep = np.isfinite(s)
    y, s = y[keep], s[keep]
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    return float(average_precision_score(y, s) * 100.0)


def resize_mask74(mask):
    arr = np.asarray(mask, dtype=np.float32)
    return (F.interpolate(torch.tensor(arr).view(1, 1, *arr.shape), size=(SIDE * 2, SIDE * 2), mode="nearest").view(-1).numpy() > 0.5).astype(np.uint8)


def unpack_groups(result):
    return result[0] if isinstance(result, tuple) else result


def model_states(model_info):
    m = np.asarray(model_info["m"], dtype=np.float32)
    alpha = float(model_info["alpha"])
    v = unit_np(np.asarray(model_info["v1"], dtype=np.float32))
    return (m - alpha * v).astype(np.float32), (m + alpha * v).astype(np.float32)


def contrast_direction_np(s_n, s_a):
    sn = unit_np(s_n)
    sa = unit_np(s_a)
    return unit_np(sa - sn)


def differential_grid_np(z_grid):
    z = np.asarray(z_grid, dtype=np.float32)
    h, w, d = z.shape
    out = np.zeros_like(z, dtype=np.float32)
    for y in range(h):
        y1, y2 = max(0, y - 1), min(h, y + 2)
        for x in range(w):
            x1, x2 = max(0, x - 1), min(w, x + 2)
            neigh = z[y1:y2, x1:x2].reshape(-1, d)
            center_linear = (y - y1) * (x2 - x1) + (x - x1)
            neigh = np.delete(neigh, center_linear, axis=0)
            if neigh.size == 0:
                delta = z[y, x]
            else:
                delta = z[y, x] - neigh.mean(axis=0)
            out[y, x] = delta
    n = np.linalg.norm(out.reshape(-1, d), axis=1, keepdims=True)
    return (out.reshape(-1, d) / np.maximum(n, 1e-12)).reshape(h, w, d).astype(np.float32)


def differential_magnitude_np(z_grid):
    z = np.asarray(z_grid, dtype=np.float32)
    h, w, d = z.shape
    out = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        y1, y2 = max(0, y - 1), min(h, y + 2)
        for x in range(w):
            x1, x2 = max(0, x - 1), min(w, x + 2)
            neigh = z[y1:y2, x1:x2].reshape(-1, d)
            center_linear = (y - y1) * (x2 - x1) + (x - x1)
            neigh = np.delete(neigh, center_linear, axis=0)
            if neigh.size:
                out[y, x] = float(np.linalg.norm(z[y, x] - neigh.mean(axis=0)))
    return out


def image_group(mask37):
    y = np.asarray(mask37, dtype=bool)
    pos = int(y.sum())
    if pos == 0:
        return {"small": False, "large": False, "thin": False, "thick": False}
    boundary = y.copy()
    try:
        from scipy import ndimage
        boundary = y & (~ndimage.binary_erosion(y, structure=np.ones((3, 3), dtype=bool), border_value=0))
    except Exception:
        pass
    thinness = float(boundary.sum() / max(pos, 1))
    return {"small": pos <= 8, "large": pos > 8, "thin": thinness >= 0.5, "thick": thinness < 0.5}


def source_dataset(source, smoke):
    if source == "Full VisA":
        _name, dataset, indices, _cls, patch, _y, _cat = load_full_visa_source_lazy(smoke)
        return dataset, indices, patch
    _name, dataset, indices, patch = load_full_mvtec_source(smoke)
    return dataset, indices, patch


def source_delta_cache(source, model_info, device, smoke):
    suffix = "smoke" if smoke else "formal"
    path = OUT / f"{source.replace(' ', '_')}_source_delta_features_{suffix}.npz"
    if path.exists():
        data = np.load(path, allow_pickle=True)
        return data["x"], data["y"], data["cat"], data["origin"], data["abs_logit"], data["delta_mag"]
    dataset, indices, patch = source_dataset(source, smoke)
    s_n, s_a = model_states(model_info)
    s_n_t = torch.tensor(s_n, dtype=torch.float32, device=device)
    s_a_t = torch.tensor(s_a, dtype=torch.float32, device=device)
    xs, ys, cats, origins, abs_logits, mags = [], [], [], [], [], []
    max_images = min(len(indices), 96) if smoke else len(indices)
    for pos, idx in enumerate(indices[:max_images]):
        if pos % 200 == 0:
            print(f"[LDSB] source delta {source} {pos}/{max_images}", flush=True)
        item = dataset[int(idx)]
        y = patch_labels(item).reshape(SIDE, SIDE).astype(np.float32)
        z = scorer_z_grid(model_info, patch[pos], device).detach().cpu().numpy().reshape(SIDE, SIDE, -1)
        dz = differential_grid_np(z)
        mag = differential_magnitude_np(z).reshape(-1)
        zt = torch.tensor(z.reshape(SIDE * SIDE, -1), dtype=torch.float32, device=device)
        with torch.no_grad():
            l_abs = two_state_logits(zt, s_n_t, s_a_t).detach().cpu().numpy().astype(np.float32)
        image_is_anomaly = int(item["anomaly"]) == 1
        flat_y = y.reshape(-1)
        for origin, cls, ids in (
            ("anomaly_mask", 1, np.flatnonzero(flat_y > 0.5)),
            ("anomaly_mask_out", 0, np.flatnonzero((flat_y < 0.5) & image_is_anomaly)),
            ("normal_image", 0, np.flatnonzero((flat_y < 0.5) & (not image_is_anomaly))),
        ):
            if ids.size == 0:
                continue
            take = min(PATCH_SAMPLE_PER_CLASS_CATEGORY, ids.size)
            rng = np.random.default_rng(SEED + pos + cls)
            if ids.size > take:
                ids = rng.choice(ids, take, replace=False)
            xs.append(dz.reshape(SIDE * SIDE, -1)[ids])
            ys.append(flat_y[ids])
            cats.extend([str(item["cls_name"])] * len(ids))
            origins.extend([origin] * len(ids))
            abs_logits.append(l_abs[ids])
            mags.append(mag[ids])
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    cat = np.asarray(cats, dtype=object)
    origin = np.asarray(origins, dtype=object)
    abs_logit = np.concatenate(abs_logits, axis=0).astype(np.float32)
    delta_mag = np.concatenate(mags, axis=0).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=x, y=y, cat=cat, origin=origin, abs_logit=abs_logit, delta_mag=delta_mag)
    return x, y, cat, origin, abs_logit, delta_mag


def make_groups(x, y, cat, origin):
    groups = {}
    for c in sorted(set(cat.tolist())):
        ids = np.flatnonzero(cat == c)
        if ids.size and len(set(y[ids].astype(int).tolist())) >= 2:
            groups[str(c)] = {"x": x[ids].astype(np.float32), "y": y[ids].astype(np.float32), "origin": origin[ids]}
    return groups


def init_states_from_delta(x, y):
    neg = x[y < 0.5]
    pos = x[y > 0.5]
    if pos.size == 0:
        pos = x
    if neg.size == 0:
        neg = x
    return unit_np(neg.mean(axis=0)).astype(np.float32), unit_np(pos.mean(axis=0)).astype(np.float32)


def train_delta_state(source, groups, s_n0, s_a0, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{source.replace(' ', '_')}_E1_DifferentialTwoState_{suffix}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        return payload["s_n"].astype(np.float32), payload["s_a"].astype(np.float32), payload["curves"], True
    print(f"[LDSB] train differential Two-State {source}", flush=True)
    s_n = torch.nn.Parameter(torch.tensor(s_n0, dtype=torch.float32, device=device))
    s_a = torch.nn.Parameter(torch.tensor(s_a0, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW([s_n, s_a], lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    total = sum(len(g["y"]) for g in groups.values())
    steps = max(1, int(math.ceil(total / BATCH_SIZE)))
    if smoke:
        steps = min(steps, 2)
    curves = []
    for epoch in range(1, EPOCHS + 1):
        vals = []
        for _ in range(steps):
            xb, yb, _orig = balanced_batch(groups, rng, BATCH_SIZE)
            z = torch.tensor(xb, dtype=torch.float32, device=device)
            y = torch.tensor(yb, dtype=torch.float32, device=device)
            loss, task, hard, bce, dice = two_state_loss(z, y, s_n, s_a)
            opt.zero_grad()
            loss.backward()
            opt.step()
            vals.append((float(loss.detach().cpu()), float(task.cpu()), float(hard.cpu()), float(bce.cpu()), float(dice.cpu())))
        arr = np.asarray(vals, dtype=np.float64)
        row = {"source": source, "variant": "E1_DifferentialTwoState", "epoch": epoch, "loss": float(arr[:, 0].mean()), "state_loss": float(arr[:, 1].mean()), "ntr_loss": float(arr[:, 2].mean()), "bce": float(arr[:, 3].mean()), "dice": float(arr[:, 4].mean())}
        curves.append(row)
        print(f"[LDSB] {source} epoch {epoch}/{EPOCHS} loss={row['loss']:.4f} ntr={row['ntr_loss']:.4f}", flush=True)
    payload = {"s_n": s_n.detach().cpu().numpy().astype(np.float32), "s_a": s_a.detach().cpu().numpy().astype(np.float32), "curves": curves}
    torch.save(payload, ckpt)
    return payload["s_n"], payload["s_a"], curves, False


@torch.no_grad()
def score_abs_delta_maps(model_info, z_grid, delta_states, scale, device):
    s_n_abs, s_a_abs = model_states(model_info)
    s_n_abs = torch.tensor(s_n_abs, dtype=torch.float32, device=device)
    s_a_abs = torch.tensor(s_a_abs, dtype=torch.float32, device=device)
    s_n_d = torch.tensor(delta_states[0], dtype=torch.float32, device=device)
    s_a_d = torch.tensor(delta_states[1], dtype=torch.float32, device=device)
    z = z_grid.reshape(SIDE, SIDE, -1).detach().cpu().numpy()
    dz = differential_grid_np(z)
    zt = torch.tensor(z.reshape(SIDE * SIDE, -1), dtype=torch.float32, device=device)
    dzt = torch.tensor(dz.reshape(SIDE * SIDE, -1), dtype=torch.float32, device=device)
    l_abs = two_state_logits(zt, s_n_abs, s_a_abs).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)
    l_delta = two_state_logits(dzt, s_n_d, s_a_d).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)
    return l_abs, l_delta, (l_abs + FUSION_WEIGHT * scale * l_delta).astype(np.float32)


def target_gate0(model_infos, delta_scale, states_by_source, clip_model, preprocess, device, smoke):
    rows = []
    for source, target in GATE_PAIRS:
        info = model_infos[source]
        cells, _ = protocol_cells(source, device, smoke)
        match = [c for c in cells if c[1] == target][0]
        _cid, _target, dataset, refs, patch, kind = match
        if smoke:
            refs = refs[: min(24, len(refs))]
            patch = patch[: min(24, len(patch))]
        all_y, all_mag, small_y, small_mag, thin_y, thin_mag, large_y, large_mag, thick_y, thick_mag = [], [], [], [], [], [], [], [], [], []
        fn_vals, hit_vals = [], []
        for pos, ref in enumerate(refs):
            item = item_at(dataset, kind, ref, pos)
            mask = mask_np(item)
            y37 = (F.interpolate(torch.tensor(mask.astype(np.float32)).view(1, 1, *mask.shape), size=(SIDE, SIDE), mode="nearest").view(SIDE, SIDE).numpy() > 0.5).astype(np.uint8)
            z = scorer_z_grid(info, patch[pos], device).detach().cpu().numpy().reshape(SIDE, SIDE, -1)
            mag = differential_magnitude_np(z)
            l_abs, _ld, _lf = score_abs_delta_maps(info, torch.tensor(z, dtype=torch.float32, device=device), states_by_source[source], delta_scale[source], device)
            flat_y = y37.reshape(-1)
            flat_m = mag.reshape(-1)
            all_y.append(flat_y)
            all_mag.append(flat_m)
            groups = image_group(y37)
            for key, yy, mm in (("small", small_y, small_mag), ("thin", thin_y, thin_mag), ("large", large_y, large_mag), ("thick", thick_y, thick_mag)):
                if groups[key]:
                    yy.append(flat_y)
                    mm.append(flat_m)
            if flat_y.sum() > 0:
                a_scores = l_abs.reshape(-1)[flat_y > 0]
                m_scores = flat_m[flat_y > 0]
                p20 = np.percentile(a_scores, 20)
                fn_vals.extend(m_scores[a_scores <= p20].tolist())
                hit_vals.extend(m_scores[a_scores > p20].tolist())
        def auc_of(ys, ss):
            if not ys:
                return float("nan")
            return safe_auc_np(np.concatenate(ys), np.concatenate(ss))
        rows.append({
            "source": source,
            "target": target,
            "overall_delta_magnitude_auc": auc_of(all_y, all_mag),
            "small_delta_magnitude_auc": auc_of(small_y, small_mag),
            "thin_delta_magnitude_auc": auc_of(thin_y, thin_mag),
            "large_delta_magnitude_auc": auc_of(large_y, large_mag),
            "thick_delta_magnitude_auc": auc_of(thick_y, thick_mag),
            "fn_delta_magnitude_mean": finite_mean(fn_vals),
            "correct_anomaly_delta_magnitude_mean": finite_mean(hit_vals),
            "fn_minus_correct_anomaly_magnitude": finite_mean(fn_vals) - finite_mean(hit_vals),
        })
    return rows


def source_scale_stats(model_info, delta_states, x_delta, abs_logit, device):
    zt = torch.tensor(x_delta, dtype=torch.float32, device=device)
    s_n = torch.tensor(delta_states[0], dtype=torch.float32, device=device)
    s_a = torch.tensor(delta_states[1], dtype=torch.float32, device=device)
    with torch.no_grad():
        dlog = two_state_logits(zt, s_n, s_a).detach().cpu().numpy().astype(np.float32)
    sd_abs = float(np.std(abs_logit))
    sd_delta = float(np.std(dlog))
    scale = sd_abs / max(sd_delta, 1e-12)
    return {"source_abs_logit_std": sd_abs, "source_delta_logit_std": sd_delta, "delta_scale_to_abs": scale, "source_abs_delta_pearson": float(np.corrcoef(abs_logit, dlog)[0, 1]) if len(abs_logit) > 1 else float("nan")}


def evaluate_pilot(model_infos, states_by_source, scales, clip_model, preprocess, pairs, device, smoke):
    raw, cats, subgroup_rows = [], [], []
    for source, target in pairs:
        info = model_infos[source]
        cells, _ = protocol_cells(source, device, smoke)
        match = [c for c in cells if c[1] == target][0]
        _cid, _target, dataset, refs, patch, kind = match
        if smoke:
            refs = refs[: min(24, len(refs))]
            patch = patch[: min(24, len(patch))]
        maps = {v: [] for v in VARIANTS}
        metas = []
        subgroup_store = {(g, v): {"y": [], "s": []} for g in ("small", "thin", "large", "thick") for v in VARIANTS}
        for pos, ref in enumerate(refs):
            if pos % 25 == 0:
                print(f"[LDSB] eval {source}->{target} image {pos}/{len(refs)}", flush=True)
            item = item_at(dataset, kind, ref, pos)
            cat, label, path = item_meta(item, kind, ref, target)
            mask = mask_np(item)
            y37 = (F.interpolate(torch.tensor(mask.astype(np.float32)).view(1, 1, *mask.shape), size=(SIDE, SIDE), mode="nearest").view(SIDE, SIDE).numpy() > 0.5).astype(np.uint8)
            y74 = resize_mask74(mask)
            img_groups = image_group(y37)
            img = Image.open(path).convert("RGB")
            global_z = scorer_z_grid(info, patch[pos], device)
            local_z = []
            batch = torch.stack([preprocess(img.crop(tuple(b))) for b in fixed_quadrant_boxes(*img.size)], dim=0).to(device)
            out = visual_outputs(clip_model, batch, adapter=None, grad=False)
            patches = normalized_patch_feature(out).float().cpu().numpy().astype(np.float32)
            for i in range(4):
                local_z.append(scorer_z_grid(info, patches[i], device))
            g_abs, g_del, g_fuse = score_abs_delta_maps(info, global_z, states_by_source[source], scales[source], device)
            l_abs_parts, l_del_parts, l_fuse_parts = [], [], []
            for z in local_z:
                a, d, fu = score_abs_delta_maps(info, z, states_by_source[source], scales[source], device)
                l_abs_parts.append(a)
                l_del_parts.append(d)
                l_fuse_parts.append(fu)
            for variant, g37, locals37 in (
                ("E0_Baseline", g_abs, l_abs_parts),
                ("E1_DifferentialOnly", g_del, l_del_parts),
                ("E2_AbsPlusDifferential", g_fuse, l_fuse_parts),
            ):
                l74 = stitch_quadrant74(locals37)
                g74 = torch_resize(g37, (SIDE * 2, SIDE * 2), align_corners=True)
                final_map = (0.5 * g74 + 0.5 * l74).astype(np.float32)
                maps[variant].append(final_map)
                for group_name, active in img_groups.items():
                    if active:
                        subgroup_store[(group_name, variant)]["y"].append(y74)
                        subgroup_store[(group_name, variant)]["s"].append(final_map.reshape(-1).astype(np.float32))
            metas.append({"category": cat, "label": int(label), "mask": mask, "img_path": path})
            img.close()
        rr, cc = evaluate_maps(source, target, refs, dataset, kind, maps, metas, variants=VARIANTS)
        raw.extend(rr)
        cats.extend(cc)
        for group_name in ("small", "thin", "large", "thick"):
            base = subgroup_store[(group_name, "E0_Baseline")]
            base_ap = safe_ap_np(np.concatenate(base["y"]), np.concatenate(base["s"])) if base["y"] else float("nan")
            for variant in VARIANTS:
                cur = subgroup_store[(group_name, variant)]
                ap = safe_ap_np(np.concatenate(cur["y"]), np.concatenate(cur["s"])) if cur["y"] else float("nan")
                subgroup_rows.append({
                    "source": source,
                    "dataset": target,
                    "morphology_group": group_name,
                    "variant": variant,
                    "M_FR_P_AP": ap,
                    "Delta_M_FR_P_AP_vs_E0": ap - base_ap if np.isfinite(ap) and np.isfinite(base_ap) else float("nan"),
                    "images": len(cur["y"]),
                })
        write_csv(OUT / "RAW_METRICS.partial.csv", raw)
        write_csv(OUT / "PER_CATEGORY_FULL_RES.partial.csv", cats)
        write_csv(OUT / "MORPHOLOGY_SUBGROUP_DELTA.partial.csv", subgroup_rows)
    return raw, cats, subgroup_rows


def delta_rows(raw):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for source, target in sorted({(r["source"], r["dataset"]) for r in raw}):
        base = by.get((source, target, "E0_Baseline"))
        if not base:
            continue
        for variant in VARIANTS[1:]:
            cur = by.get((source, target, variant))
            if not cur:
                continue
            rows.append({
                "source": source,
                "dataset": target,
                "variant": variant,
                "Delta_M_FR_P_AUROC": fnum(cur, "M_FR_P_AUROC") - fnum(base, "M_FR_P_AUROC"),
                "Delta_M_FR_P_AP": fnum(cur, "M_FR_P_AP") - fnum(base, "M_FR_P_AP"),
                "Delta_M_FR_P_F1max": fnum(cur, "M_FR_P_F1max") - fnum(base, "M_FR_P_F1max"),
                "Delta_M_FR_PRO": fnum(cur, "M_FR_PRO") - fnum(base, "M_FR_PRO"),
            })
    for variant in VARIANTS[1:]:
        vals = [fnum(r, "Delta_M_FR_P_AP") for r in rows if r["variant"] == variant]
        rows.append({"source": "SUMMARY", "dataset": "PILOT", "variant": variant, "mean_delta_M_FR_P_AP": finite_mean(vals), "median_delta_M_FR_P_AP": finite_median(vals), "improved": int(sum(v > 0 for v in vals if np.isfinite(v))), "non_drop_ge_neg_0_2": int(sum(v >= -0.2 for v in vals if np.isfinite(v)))})
    return rows


def error_response_diagnostic(model_infos, states_by_source, scales, clip_model, preprocess, pairs, device, smoke):
    rows = []
    for source, target in pairs:
        info = model_infos[source]
        cells, _ = protocol_cells(source, device, smoke)
        match = [c for c in cells if c[1] == target][0]
        _cid, _target, dataset, refs, patch, kind = match
        if smoke:
            refs = refs[: min(24, len(refs))]
            patch = patch[: min(24, len(patch))]
        fn_delta, corr_delta, fp_delta, tn_delta = [], [], [], []
        for pos, ref in enumerate(refs):
            item = item_at(dataset, kind, ref, pos)
            mask = mask_np(item)
            y37 = (F.interpolate(torch.tensor(mask.astype(np.float32)).view(1, 1, *mask.shape), size=(SIDE, SIDE), mode="nearest").view(-1).numpy() > 0.5).astype(np.uint8)
            z = scorer_z_grid(info, patch[pos], device)
            l_abs, l_delta, _ = score_abs_delta_maps(info, z, states_by_source[source], scales[source], device)
            a = l_abs.reshape(-1)
            d = l_delta.reshape(-1)
            if np.any(y37 == 1):
                p20 = np.percentile(a[y37 == 1], 20)
                fn_delta.extend(d[(y37 == 1) & (a <= p20)].tolist())
                corr_delta.extend(d[(y37 == 1) & (a > p20)].tolist())
            if np.any(y37 == 0):
                p95 = np.percentile(a[y37 == 0], 95)
                fp_delta.extend(d[(y37 == 0) & (a >= p95)].tolist())
                tn_delta.extend(d[(y37 == 0) & (a < p95)].tolist())
        rows.append({"source": source, "target": target, "fn_delta_score_mean": finite_mean(fn_delta), "correct_anomaly_delta_score_mean": finite_mean(corr_delta), "fn_minus_correct_anomaly_delta": finite_mean(fn_delta) - finite_mean(corr_delta), "fp_delta_score_mean": finite_mean(fp_delta), "tn_delta_score_mean": finite_mean(tn_delta), "fp_minus_tn_delta": finite_mean(fp_delta) - finite_mean(tn_delta)})
    return rows


def decide(gate_rows, delta):
    signal = any((fnum(r, "small_delta_magnitude_auc") >= GATE0_AUC_THRESHOLD or fnum(r, "thin_delta_magnitude_auc") >= GATE0_AUC_THRESHOLD) for r in gate_rows)
    if not signal:
        return "LOCAL_DIFFERENTIAL_FEATURE_HAS_NO_USEFUL_SIGNAL", {"gate0_signal": False}
    e2 = [r for r in delta if r.get("variant") == "E2_AbsPlusDifferential" and r.get("source") != "SUMMARY"]
    vals = [fnum(r, "Delta_M_FR_P_AP") for r in e2]
    mean_delta = finite_mean(vals)
    improved = sum(v > 0 for v in vals if np.isfinite(v))
    non_drop = sum(v >= -0.2 for v in vals if np.isfinite(v))
    hard_big = any((r["source"], r["dataset"]) in GATE_PAIRS and fnum(r, "Delta_M_FR_P_AP") >= 1.0 for r in e2)
    stats = {"gate0_signal": True, "mean_delta": mean_delta, "improved": improved, "non_drop_ge_neg_0_2": non_drop, "hard_pair_ge_1": hard_big}
    if mean_delta >= 1.0 and improved >= 3 and hard_big and non_drop >= 3:
        return "LOCAL_DIFFERENTIAL_STATE_SIGNIFICANTLY_IMPROVES_FINE_GRAINED_TRANSFER", stats
    if mean_delta > 0 and improved >= 2:
        return "LOCAL_DIFFERENTIAL_STATE_HAS_WEAK_COMPLEMENTARY_VALUE", stats
    if signal and mean_delta <= 0:
        return "LOCAL_DIFFERENTIAL_FEATURE_IS_DIAGNOSTIC_ONLY", stats
    return "LOCAL_DIFFERENTIAL_STATE_HAS_NO_COMPETITIVE_VALUE", stats


def write_report(outcome, decision, gate_rows, raw, delta, scale_rows, state_rows, error_rows, runtime):
    lines = [
        "# Phase LDSB: Local Differential State Branch",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "Baseline is Canonical Two-State + NTR + Fixed LRP. The differential branch uses frozen CLIP patch center-surround features and trains an independent Canonical Two-State + NTR scorer; the absolute branch is unchanged.",
        "",
        "## Gate 0 Differential Magnitude Audit",
        "",
        "| Source | Target | Overall AUROC | Small AUROC | Thin AUROC | FN-correct mag |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in gate_rows:
        lines.append(f"| {r['source']} | {r['target']} | {fnum(r,'overall_delta_magnitude_auc'):.4f} | {fnum(r,'small_delta_magnitude_auc'):.4f} | {fnum(r,'thin_delta_magnitude_auc'):.4f} | {fnum(r,'fn_minus_correct_anomaly_magnitude'):+.4f} |")
    if raw:
        lines += ["", "## Pilot Metrics", "", "| Source | Target | Variant | P-AUROC | P-AP | P-F1max | PRO |", "|---|---|---|---:|---:|---:|---:|"]
        for r in raw:
            lines.append(f"| {r['source']} | {r['dataset']} | {r['variant']} | {fnum(r,'M_FR_P_AUROC'):.4f} | {fnum(r,'M_FR_P_AP'):.4f} | {fnum(r,'M_FR_P_F1max'):.4f} | {fnum(r,'M_FR_PRO'):.4f} |")
        lines += ["", "## Delta P-AP vs E0", "", "| Source | Target | Variant | Delta P-AP | Delta PRO |", "|---|---|---|---:|---:|"]
        for r in delta:
            if r.get("source") != "SUMMARY":
                lines.append(f"| {r['source']} | {r['dataset']} | {r['variant']} | {fnum(r,'Delta_M_FR_P_AP'):+.4f} | {fnum(r,'Delta_M_FR_PRO'):+.4f} |")
    lines += ["", "## Scale And Geometry", "", "| Source | scale(delta->abs) | corr(abs,delta) | cos(d_abs,d_delta) |", "|---|---:|---:|---:|"]
    geo = {r["source"]: r for r in state_rows}
    for r in scale_rows:
        g = geo.get(r["source"], {})
        lines.append(f"| {r['source']} | {fnum(r,'delta_scale_to_abs'):.4f} | {fnum(r,'source_abs_delta_pearson'):.4f} | {fnum(g,'cos_d_abs_d_delta'):.4f} |")
    if error_rows:
        lines += ["", "## Error Response", "", "| Source | Target | FN-correct delta | FP-TN delta |", "|---|---|---:|---:|"]
        for r in error_rows:
            lines.append(f"| {r['source']} | {r['target']} | {fnum(r,'fn_minus_correct_anomaly_delta'):+.4f} | {fnum(r,'fp_minus_tn_delta'):+.4f} |")
    lines += [
        "",
        "## Decision",
        "",
        f"- Gate0 signal: `{decision.get('gate0_signal')}`",
        f"- E2 mean Delta P-AP: `{decision.get('mean_delta', float('nan')):+.4f}`",
        f"- E2 improved transfers: `{decision.get('improved', 0)}/4`",
        f"- E2 non-drop >= -0.2: `{decision.get('non_drop_ge_neg_0_2', 0)}/4`",
        f"- Hard pair >= +1.0: `{decision.get('hard_pair_ge_1')}`",
        "",
        f"Runtime: {runtime:.1f}s.",
    ]
    write_text(OUT / "REPORT.md", "\n".join(lines) + "\n")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[LDSB] device={device}", flush=True)
    preprocess, clip_model = load_runtime(device)
    model_infos = {"Full VisA": load_visa_source(device), "Full MVTec": load_mvtec_source()}
    state_rows, train_rows, scale_rows = [], [], []
    states_by_source, scales = {}, {}
    for source, info in model_infos.items():
        x, y, cat, origin, abs_logit, delta_mag = source_delta_cache(source, info, device, args.smoke)
        groups = make_groups(x, y, cat, origin)
        s_n0, s_a0 = init_states_from_delta(x, y)
        s_n, s_a, curves, _cached = train_delta_state(source, groups, s_n0, s_a0, device, args.smoke)
        states_by_source[source] = (s_n, s_a)
        train_rows.extend(curves)
        scale = source_scale_stats(info, (s_n, s_a), x, abs_logit, device)
        scale["source"] = source
        scale_rows.append(scale)
        scales[source] = fnum(scale, "delta_scale_to_abs")
        abs_n, abs_a = model_states(info)
        state_rows.append({"source": source, "cos_d_abs_d_delta": float(np.dot(contrast_direction_np(abs_n, abs_a), contrast_direction_np(s_n, s_a))), "delta_sN_norm": float(np.linalg.norm(s_n)), "delta_sA_norm": float(np.linalg.norm(s_a))})
    write_csv(OUT / "DIFFERENTIAL_SOURCE_TRAINING.csv", train_rows)
    write_csv(OUT / "SCALE_MATCHING.csv", scale_rows)
    write_csv(OUT / "STATE_GEOMETRY.csv", state_rows)
    gate_rows = target_gate0(model_infos, scales, states_by_source, clip_model, preprocess, device, args.smoke)
    write_csv(OUT / "GATE0_DIFFERENTIAL_AUDIT.csv", gate_rows)
    gate_signal = any((fnum(r, "small_delta_magnitude_auc") >= GATE0_AUC_THRESHOLD or fnum(r, "thin_delta_magnitude_auc") >= GATE0_AUC_THRESHOLD) for r in gate_rows)
    if not gate_signal:
        outcome, decision = decide(gate_rows, [])
        runtime = time.time() - start
        write_json(OUT / "OUTCOME.json", {"outcome": outcome, "decision": decision, "runtime_sec": runtime})
        write_report(outcome, decision, gate_rows, [], [], scale_rows, state_rows, [], runtime)
        print(f"[LDSB] OUTCOME={outcome} runtime={runtime:.1f}s", flush=True)
        return
    raw, cats, subgroup = evaluate_pilot(model_infos, states_by_source, scales, clip_model, preprocess, PILOT_PAIRS, device, args.smoke)
    deltas = delta_rows(raw)
    error_rows = error_response_diagnostic(model_infos, states_by_source, scales, clip_model, preprocess, PILOT_PAIRS, device, args.smoke)
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY_FULL_RES.csv", cats)
    write_csv(OUT / "DELTA_SUMMARY.csv", deltas)
    write_csv(OUT / "MORPHOLOGY_SUBGROUP_DELTA.csv", subgroup)
    write_csv(OUT / "ERROR_RESPONSE_DIAGNOSTIC.csv", error_rows)
    outcome, decision = decide(gate_rows, deltas)
    runtime = time.time() - start
    write_json(OUT / "OUTCOME.json", {"outcome": outcome, "decision": decision, "runtime_sec": runtime, "fusion_weight": FUSION_WEIGHT})
    write_report(outcome, decision, gate_rows, raw, deltas, scale_rows, state_rows, error_rows, runtime)
    print(f"[LDSB] OUTCOME={outcome} runtime={runtime:.1f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
