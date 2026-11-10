import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_cmab_compact_multi_anomaly_basis import load_mvtec_groups, load_visa_groups
from experiments.phase_final_cross_dataset_benchmark import load_mvtec_source, load_visa_source, protocol_cells
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import (
    item_at,
    item_meta,
    mask_np,
    quadrant_crops,
    torch_resize,
)
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import evaluate_maps
from experiments.phase_private_residual_contrast_memory import contrast_logits
from experiments.phase_spatial_residual_refinement_standard_metric import aligned_grid, fnum
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import CHECKPOINT
from experiments.phase_xxxviii_visual_state_representation import IMAGE_SIZE, SIDE, norm_np
from phase_xa.official_task_forward import normalized_patch_feature, visual_outputs
from utils.transforms import get_transform


OUT = Path("results/phase_gcra_geometry_constrained_residual_adaptation/formal")
VARIANTS = (
    "E0_Baseline",
    "E1_UnconstrainedResidualAdapter",
    "E2_HardBoundedResidualAdapter",
    "E3_GeometryConstrainedAdapter",
)
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
GCRA_LR = 0.001
BATCH_SIZE = 8192
RANK = 32
RHO = 0.05
LAMBDA_GEO = 1.0
TOP_NEG_FRAC = 0.10
PAIR_SAMPLE = 512
SEED = 0


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


def checksum_module(module):
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def dice_loss(logits, y):
    p = torch.sigmoid(logits)
    y = y.float()
    return 1.0 - (2.0 * (p * y).sum() + 1.0) / (p.sum() + y.sum() + 1.0)


def balanced_loss(logits, y):
    y = y.float()
    pos = y.sum().clamp_min(1.0)
    neg = (1.0 - y).sum().clamp_min(1.0)
    w = torch.where(y > 0.5, 0.5 * (pos + neg) / pos, 0.5 * (pos + neg) / neg)
    bce = F.binary_cross_entropy_with_logits(logits, y, weight=w)
    dl = dice_loss(logits, y)
    return bce + dl, bce.detach(), dl.detach()


class LowRankResidualAdapter(nn.Module):
    def __init__(self, dim, rank=RANK, mode="unbounded"):
        super().__init__()
        self.mode = mode
        torch.manual_seed(SEED)
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, z):
        delta = self.up(F.relu(self.down(z)))
        if self.mode == "unbounded":
            applied = delta
        else:
            delta_norm = torch.linalg.norm(delta, dim=-1, keepdim=True)
            z_norm = torch.linalg.norm(z, dim=-1, keepdim=True).clamp_min(1e-12)
            mag = torch.minimum(delta_norm, RHO * z_norm)
            applied = mag * delta / delta_norm.clamp_min(1e-12)
        return F.normalize(z + applied, dim=-1), delta, applied


class IdentityAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.mode = "identity"

    def forward(self, z):
        zero = torch.zeros_like(z)
        return F.normalize(z, dim=-1), zero, zero


def state_logits(z, source_info, v):
    m = torch.tensor(source_info["m"], dtype=torch.float32, device=z.device)
    return contrast_logits(z.reshape(-1, z.shape[-1]), m, float(source_info["alpha"]), v).reshape(z.shape[:-1])


def source_state_pair(source_info):
    m = np.asarray(source_info["m"], dtype=np.float32)
    a = float(source_info["alpha"])
    v = np.asarray(source_info["v1"], dtype=np.float32)
    return (m - a * v).astype(np.float32), (m + a * v).astype(np.float32)


def balanced_category_batch(groups, rng, total_batch):
    cats = list(groups)
    per_cat = max(2, total_batch // max(1, len(cats)))
    xs, ys, origins = [], [], []
    for cat in cats:
        data = groups[cat]
        per_cls = max(1, per_cat // 2)
        for cls in (0, 1):
            ids = np.flatnonzero(data["y"] == cls)
            if ids.size == 0:
                continue
            choice = rng.choice(ids, size=per_cls, replace=ids.size < per_cls)
            xs.append(data["x"][choice])
            ys.append(data["y"][choice])
            origins.extend(data["origin"][choice].tolist())
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    order = rng.permutation(len(y))
    return x[order], y[order], np.asarray(origins, dtype=object)[order]


def geometry_loss(z, zp, rng):
    n = int(z.shape[0])
    k = min(PAIR_SAMPLE, n)
    if k < 2:
        return torch.zeros((), dtype=z.dtype, device=z.device)
    ids = torch.tensor(rng.choice(n, size=k, replace=False), dtype=torch.long, device=z.device)
    a = F.normalize(z[ids], dim=-1)
    b = F.normalize(zp[ids], dim=-1)
    return torch.mean((a @ a.T - b @ b.T) ** 2)


def hardnormal_loss(logits, y):
    neg = torch.nonzero(y.reshape(-1) < 0.5, as_tuple=False).flatten()
    if neg.numel() == 0:
        return torch.zeros((), dtype=logits.dtype, device=logits.device), 0
    flat = logits.reshape(-1)
    k = max(1, int(math.ceil(float(neg.numel()) * TOP_NEG_FRAC)))
    idx = neg[torch.topk(flat.detach()[neg], k=k, largest=True).indices]
    return F.softplus(flat[idx]).mean(), int(k)


def train_variant(source, source_info, groups, variant, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / "checkpoints" / f"{source.replace(' ', '_')}_{variant}_{suffix}.pt"
    if variant == "E0_Baseline":
        return IdentityAdapter().to(device).eval(), source_info["v1"].astype(np.float32), [], True
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        adapter = LowRankResidualAdapter(int(payload["dim"]), mode=payload["mode"]).to(device)
        adapter.load_state_dict(payload["adapter"], strict=True)
        return adapter.eval(), payload["v"].astype(np.float32), payload["diagnostics"], True

    mode = "unbounded" if variant == "E1_UnconstrainedResidualAdapter" else "bounded"
    dim = int(next(iter(groups.values()))["x"].shape[1])
    adapter = LowRankResidualAdapter(dim, mode=mode).to(device)
    v_param = torch.nn.Parameter(torch.tensor(source_info["v1"], dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(list(adapter.parameters()) + [v_param], lr=GCRA_LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    total = sum(len(g["y"]) for g in groups.values())
    steps = max(1, math.ceil(total / BATCH_SIZE))
    if smoke:
        steps = min(steps, 2)
    diagnostics = []
    print(f"[GCRA] train {source} {variant} mode={mode}", flush=True)
    for epoch in range(1, EPOCHS + 1):
        rows = []
        for step in range(1, steps + 1):
            xb, yb, _origin = balanced_category_batch(groups, rng, BATCH_SIZE)
            z = torch.tensor(xb, dtype=torch.float32, device=device)
            y = torch.tensor(yb, dtype=torch.float32, device=device)
            v = F.normalize(v_param, dim=-1)
            zp, delta, applied = adapter(z)
            logits = state_logits(zp, source_info, v)
            task, bce, dice = balanced_loss(logits.reshape(-1), y.reshape(-1))
            ntr, hard_count = hardnormal_loss(logits, y)
            geo = geometry_loss(z, zp, rng) if variant == "E3_GeometryConstrainedAdapter" else torch.zeros((), dtype=task.dtype, device=device)
            loss = task + ntr + LAMBDA_GEO * geo
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                ratio = torch.linalg.norm(applied, dim=-1) / torch.linalg.norm(z, dim=-1).clamp_min(1e-12)
                cos = (F.normalize(z, dim=-1) * F.normalize(zp, dim=-1)).sum(dim=-1).clamp(-1.0, 1.0)
                angle = torch.rad2deg(torch.acos(cos))
                rows.append({
                    "loss": float(loss.detach().cpu()),
                    "state_loss": float(task.detach().cpu()),
                    "ntr_loss": float(ntr.detach().cpu()),
                    "geo_loss": float(geo.detach().cpu()),
                    "delta_ratio_mean": float(ratio.mean().detach().cpu()),
                    "delta_ratio_p95": float(torch.quantile(ratio, 0.95).detach().cpu()),
                    "delta_ratio_max": float(ratio.max().detach().cpu()),
                    "cos_z_zp_mean": float(cos.mean().detach().cpu()),
                    "angle_mean": float(angle.mean().detach().cpu()),
                    "angle_p95": float(torch.quantile(angle, 0.95).detach().cpu()),
                    "angle_max": float(angle.max().detach().cpu()),
                    "hard_patch_count": hard_count,
                    "bce": float(bce.detach().cpu()),
                    "dice": float(dice.detach().cpu()),
                })
        row = {"source": source, "variant": variant, "epoch": epoch}
        for key in rows[0]:
            row[key] = finite_mean([r[key] for r in rows])
        diagnostics.append(row)
        print(f"[GCRA] {source} {variant} epoch={epoch}/{EPOCHS} loss={row['loss']:.4f} ratio={row['delta_ratio_mean']:.4f} geo={row['geo_loss']:.6f}", flush=True)
    with torch.no_grad():
        v_np = F.normalize(v_param, dim=-1).detach().cpu().numpy().astype(np.float32)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"adapter": adapter.eval().state_dict(), "v": v_np, "dim": dim, "mode": mode, "diagnostics": diagnostics}, ckpt)
    return adapter.eval(), v_np, diagnostics, False


@torch.no_grad()
def canonical_grid(model_info, patch_np, device):
    if model_info["use_align"]:
        return aligned_grid(patch_np, model_info["q_pc1"], model_info["align_model"], device).reshape(SIDE * SIDE, 1024)
    z = norm_np(np.asarray(patch_np, dtype=np.float32).reshape(SIDE * SIDE, 1024))
    return torch.tensor(z, dtype=torch.float32, device=device)


@torch.no_grad()
def score_grid(model_info, z, adapter, v_np, device):
    v = torch.tensor(v_np, dtype=torch.float32, device=device)
    zp, _delta, _applied = adapter(z)
    return state_logits(zp, model_info, v).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)


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
def local_zoom_features(model_info, clip_model, preprocess, img_path, device):
    crops = quadrant_crops(img_path)
    batch = torch.stack([preprocess(c) for c in crops], dim=0).to(device)
    out = visual_outputs(clip_model, batch, adapter=None, grad=False)
    patches = normalized_patch_feature(out).float()
    return canonicalize(model_info, patches)


@torch.no_grad()
def local_zoom_score(model_info, local_z, adapter, v_np, device):
    qs = [score_grid(model_info, local_z[i], adapter, v_np, device) for i in range(4)]
    s = np.zeros((SIDE * 2, SIDE * 2), dtype=np.float32)
    s[:SIDE, :SIDE] = qs[0]
    s[:SIDE, SIDE:] = qs[1]
    s[SIDE:, :SIDE] = qs[2]
    s[SIDE:, SIDE:] = qs[3]
    return s


def build_maps(source_info, models, clip_model, preprocess, patch, refs, dataset, kind, target, device, smoke):
    n = min(24, len(refs)) if smoke else len(refs)
    maps = {v: [] for v in VARIANTS}
    metas = []
    for pos, ref in enumerate(refs[:n]):
        if pos % 25 == 0:
            print(f"[GCRA] {source_info['source']} -> {target} image {pos}/{n}", flush=True)
        item = item_at(dataset, kind, ref, pos)
        cat, label, path = item_meta(item, kind, ref, target)
        if not path:
            raise ValueError(f"missing img_path for {source_info['source']} -> {target} pos={pos}")
        metas.append({"category": cat, "label": label, "mask": mask_np(item), "img_path": path})
        global_z = canonical_grid(source_info, patch[pos], device)
        local_z = local_zoom_features(source_info, clip_model, preprocess, path, device)
        for variant, (adapter, v_np) in models.items():
            global37 = score_grid(source_info, global_z, adapter, v_np, device)
            local74 = local_zoom_score(source_info, local_z, adapter, v_np, device)
            global74 = torch_resize(global37, (SIDE * 2, SIDE * 2), align_corners=True)
            maps[variant].append((0.5 * global74 + 0.5 * local74).astype(np.float32))
    return maps, metas


def selected_eval(source, target, maps, metas):
    import experiments.phase_parameter_free_local_zoom_pyramid_fullres as lzp
    old = tuple(lzp.VARIANTS)
    try:
        lzp.VARIANTS = tuple(VARIANTS)
        return evaluate_maps(source, target, maps, metas)
    finally:
        lzp.VARIANTS = old


def delta_summary(raw, pairs):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for variant in VARIANTS:
        if variant == "E0_Baseline":
            continue
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
        arr = np.asarray(vals, dtype=np.float64)
        rows.append({
            "source": "SUMMARY",
            "dataset": "PILOT",
            "variant": variant,
            "mean_delta_M_FR_P_AP": finite_mean(arr),
            "median_delta_M_FR_P_AP": finite_median(arr),
            "improved": int(np.sum(arr > 0)) if arr.size else 0,
            "non_drop_ge_minus_0_2": int(np.sum(arr >= -0.2)) if arr.size else 0,
            "total": int(arr.size),
        })
    return rows


def adapter_geometry(source, groups, model, source_info, device):
    adapter, _v = model
    rng = np.random.default_rng(SEED)
    xb, yb, _origin = balanced_category_batch(groups, rng, min(BATCH_SIZE, 8192))
    z = torch.tensor(xb, dtype=torch.float32, device=device)
    with torch.no_grad():
        zp, _delta, applied = adapter(z)
        ratio = (torch.linalg.norm(applied, dim=-1) / torch.linalg.norm(z, dim=-1).clamp_min(1e-12)).detach().cpu().numpy()
        cos = (F.normalize(z, dim=-1) * F.normalize(zp, dim=-1)).sum(dim=-1).clamp(-1, 1).detach().cpu().numpy()
        angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        ids = rng.choice(len(xb), size=min(PAIR_SAMPLE, len(xb)), replace=False)
        pair_before = np.asarray(xb[ids] @ xb[ids].T, dtype=np.float32)
        pair_after = (F.normalize(zp[ids], dim=-1) @ F.normalize(zp[ids], dim=-1).T).detach().cpu().numpy()
        distort = np.square(pair_before - pair_after).reshape(-1)
    return {
        "source": source,
        "variant": next(k for k, v in VARIANT_MODES.items() if k == source_info.get("_variant_probe", k)) if False else "",
        "patches": int(len(xb)),
        "delta_ratio_mean": float(np.mean(ratio)),
        "delta_ratio_p95": float(np.percentile(ratio, 95)),
        "delta_ratio_max": float(np.max(ratio)),
        "cos_z_zp_mean": float(np.mean(cos)),
        "cos_z_zp_p05": float(np.percentile(cos, 5)),
        "angular_displacement_mean": float(np.mean(angle)),
        "angular_displacement_p95": float(np.percentile(angle, 95)),
        "angular_displacement_max": float(np.max(angle)),
        "pairwise_cosine_distortion": float(np.mean(distort)),
    }


def geometry_rows(source_infos, groups_by_source, models_by_source, device):
    rows = []
    for source, info in source_infos.items():
        b = info["v1"] / max(float(np.linalg.norm(info["v1"])), 1e-12)
        for variant, model in models_by_source[source].items():
            adapter, v = model
            row = adapter_geometry(source, groups_by_source[source], model, info, device)
            vv = v / max(float(np.linalg.norm(v)), 1e-12)
            row.update({
                "variant": variant,
                "cos_d_E0": float(np.dot(b, vv)),
                "state_direction_norm": float(np.linalg.norm(v)),
                "adapter_parameter_count": int(sum(p.numel() for p in adapter.parameters())),
            })
            rows.append(row)
    return rows


def decide(summary, geom):
    summaries = [r for r in summary if r.get("source") == "SUMMARY"]
    best = max(summaries, key=lambda r: fnum(r, "mean_delta_M_FR_P_AP")) if summaries else None
    if not best:
        return "CONSTRAINED_VISUAL_ADAPTATION_HAS_NO_COMPETITIVE_VALUE", {}
    vals = [r for r in summary if r.get("variant") == best["variant"] and r.get("source") != "SUMMARY"]
    mean_delta = fnum(best, "mean_delta_M_FR_P_AP")
    improved = int(float(best.get("improved", 0)))
    focus = max([fnum(r, "Delta_M_FR_P_AP") for r in vals if (r["source"], r["dataset"]) in {("Full MVTec", "VisA"), ("Full VisA", "MPDD")}], default=float("nan"))
    catastrophic = any(fnum(r, "Delta_M_FR_P_AP") < -5.0 for r in vals)
    geom_by = {(r["source"], r["variant"]): r for r in geom}
    e1_deform = finite_mean([fnum(r, "angular_displacement_p95") for r in geom if r["variant"] == "E1_UnconstrainedResidualAdapter"])
    best_deform = finite_mean([fnum(r, "angular_displacement_p95") for r in geom if r["variant"] == best["variant"]])
    gate = mean_delta >= 1.5 and improved >= 3 and focus > 0 and not catastrophic and best_deform < e1_deform
    stats = {"best_variant": best["variant"], "mean_delta_P_AP": mean_delta, "improved": improved, "focus_best_delta_P_AP": focus, "catastrophic_drop": catastrophic, "E1_p95_angle": e1_deform, "best_p95_angle": best_deform, "pilot_gate_pass": gate}
    if gate:
        return "GEOMETRY_CONSTRAINED_ADAPTATION_SIGNIFICANTLY_IMPROVES_TRANSFER", stats
    e2 = next((r for r in summaries if r["variant"] == "E2_HardBoundedResidualAdapter"), None)
    e3 = next((r for r in summaries if r["variant"] == "E3_GeometryConstrainedAdapter"), None)
    if e2 and e3 and fnum(e3, "mean_delta_M_FR_P_AP") <= fnum(e2, "mean_delta_M_FR_P_AP") + 0.2:
        if max(fnum(e2, "mean_delta_M_FR_P_AP"), fnum(e3, "mean_delta_M_FR_P_AP")) > 0:
            return "GEOMETRY_PRESERVATION_HAS_NO_ADDED_VALUE", stats
    if mean_delta > 0:
        return "BOUNDED_ADAPTATION_RECOVERS_TRANSFER_BUT_GAIN_IS_WEAK", stats
    return "CONSTRAINED_VISUAL_ADAPTATION_HAS_NO_COMPETITIVE_VALUE", stats


def write_report(raw, summary, geom, outcome, stats, checksum, runtime):
    lines = [
        "# Phase GCRA: Geometry-Constrained Residual Adaptation",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "Baseline is Canonical Two-State + NTR + fixed 2x2 Local Re-encoding Pyramid. CLIP is frozen and unchanged; the adapter acts only on final scorer-space patch features, with Fixed LRP unchanged at inference.",
        "",
        f"Frozen CLIP checksum unchanged: `{checksum['unchanged']}`.",
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
    lines.extend(["", "## Adaptation Geometry", "", "| Source | Variant | delta ratio mean | delta ratio P95 | angle mean | angle P95 | pairwise distortion | params |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for r in geom:
        lines.append(f"| {r['source']} | {r['variant']} | {fnum(r,'delta_ratio_mean'):.6f} | {fnum(r,'delta_ratio_p95'):.6f} | {fnum(r,'angular_displacement_mean'):.4f} | {fnum(r,'angular_displacement_p95'):.4f} | {fnum(r,'pairwise_cosine_distortion'):.6e} | {int(float(r.get('adapter_parameter_count',0)))} |")
    lines.extend(["", f"Decision stats: `{json.dumps(stats, default=json_default)}`", "", f"Runtime: {runtime:.1f}s."])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[GCRA] device={device}", flush=True)
    preprocess, _ = get_transform(argparse.Namespace(image_size=IMAGE_SIZE))
    clip_model, *_ = load_components(str(CHECKPOINT), device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)
    checksum_before = checksum_module(clip_model.visual)

    source_infos = {"Full VisA": load_visa_source(device), "Full MVTec": load_mvtec_source()}
    groups_by_source = {}
    pool_rows = []
    for source, info in source_infos.items():
        print(f"[GCRA] load source pool {source}", flush=True)
        s_n, s_a = source_state_pair(info)
        if source == "Full VisA":
            groups, audit = load_visa_groups(info, device, args.smoke)
        else:
            groups, audit = load_mvtec_groups(args.smoke)
        groups_by_source[source] = groups
        pool_rows.extend({"source": source, **r} for r in audit)

    models_by_source = {}
    train_rows = []
    for source, info in source_infos.items():
        models_by_source[source] = {}
        for variant in VARIANTS:
            adapter, v, diag, reused = train_variant(source, info, groups_by_source[source], variant, device, args.smoke)
            models_by_source[source][variant] = (adapter, v)
            for row in diag:
                row["checkpoint_reused"] = bool(reused)
                train_rows.append(row)

    geom = geometry_rows(source_infos, groups_by_source, models_by_source, device)
    raw, percat = [], []
    pairs = FULL_PAIRS if args.full else PILOT_PAIRS
    for source, target in pairs:
        info = source_infos[source]
        cells, _audits = protocol_cells(source, device, args.smoke)
        matches = [c for c in cells if c[1] == target]
        if not matches:
            raise RuntimeError(f"No target cell for {source}->{target}")
        _cid, _target, dataset, idx_or_rows, patch, kind = matches[0]
        print(f"[GCRA] evaluate {source}->{target}", flush=True)
        maps, metas = build_maps(info, models_by_source[source], clip_model, preprocess, patch, idx_or_rows, dataset, kind, target, device, args.smoke)
        rr, cc = selected_eval(source, target, maps, metas)
        raw.extend(rr)
        percat.extend(cc)
        write_csv(OUT / "RAW_METRICS.partial.csv", raw)
        write_csv(OUT / "PER_CATEGORY_FULL_RES.partial.csv", percat)

    summary = delta_summary(raw, pairs)
    checksum_after = checksum_module(clip_model.visual)
    checksum = {"clip_visual_before": checksum_before, "clip_visual_after": checksum_after, "unchanged": checksum_before == checksum_after}
    outcome, stats = decide(summary, geom)
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY_FULL_RES.csv", percat)
    write_csv(OUT / "PILOT_DELTA_SUMMARY.csv", summary)
    write_csv(OUT / "ADAPTATION_GEOMETRY.csv", geom)
    write_csv(OUT / "SOURCE_TRAINING_DIAGNOSTICS.csv", train_rows)
    write_csv(OUT / "SOURCE_POOL_AUDIT.csv", pool_rows)
    write_json(OUT / "CHECKSUM.json", checksum)
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
        "fixed_lrp": True,
        "rank": RANK,
        "rho": RHO,
        "lambda_geo": LAMBDA_GEO,
        "epochs": EPOCHS,
        "lr": GCRA_LR,
        "full_stage": bool(args.full),
    })
    write_report(raw, summary, geom, outcome, stats, checksum, runtime)
    print(f"[GCRA] OUTCOME={outcome} runtime={runtime:.1f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--full", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
