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

from experiments.phase_cmab_compact_multi_anomaly_basis import load_mvtec_groups, load_visa_groups
from experiments.phase_final_cross_dataset_benchmark import load_mvtec_source, load_visa_source, protocol_cells
from experiments.phase_gcra_geometry_constrained_residual_adaptation import canonical_grid, local_zoom_features
from experiments.phase_hard_normal_contrast import TOP_NEG_FRAC
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import (
    evaluate_maps,
    item_at,
    item_meta,
    mask_np,
    torch_resize,
)
from experiments.phase_private_residual_contrast_memory import unit_np
from experiments.phase_sgpr import stitch_quadrant74
from experiments.phase_ssec_state_space_experience_consolidation import balanced_batch, two_state_logits
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, SEED, balanced_loss
from experiments.phase_xxxviii_visual_state_representation import LR, SIDE
from phase_xa.official_task_forward import normalized_patch_feature, visual_outputs
from utils.transforms import get_transform
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xlv_cross_domain_transfer import CHECKPOINT
from experiments.phase_xxxviii_visual_state_representation import IMAGE_SIZE


OUT = Path("results/phase_source_nuisance_invariant_state_regularization/formal")
VARIANTS = ("E0_Baseline", "E1_PosthocNuisanceRemoval", "E2_NuisanceRegularized")
E1_PAIRS = (
    ("Full MVTec", "VisA"),
    ("Full VisA", "MPDD"),
    ("Full MVTec", "BTAD"),
)
PILOT_PAIRS = (
    ("Full MVTec", "VisA"),
    ("Full VisA", "MPDD"),
    ("Full MVTec", "BTAD"),
    ("Full VisA", "DTD-Synthetic"),
)
EPOCHS = 10
LAMBDA_NUIS = 0.1
NUISANCE_RATIO_STOP = 0.05


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


def fnum(row, key, default=float("nan")):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def finite_mean(vals):
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def finite_median(vals):
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def source_state_pair(model_info):
    m = np.asarray(model_info["m"], dtype=np.float32)
    alpha = float(model_info["alpha"])
    v = unit_np(np.asarray(model_info["v1"], dtype=np.float32))
    return (m - alpha * v).astype(np.float32), (m + alpha * v).astype(np.float32)


def contrast_direction_np(s_n, s_a):
    return unit_np(unit_np(np.asarray(s_a, dtype=np.float32)) - unit_np(np.asarray(s_n, dtype=np.float32))).astype(np.float32)


def states_from_direction(model_info, direction):
    m = np.asarray(model_info["m"], dtype=np.float32)
    alpha = float(model_info["alpha"])
    v = unit_np(np.asarray(direction, dtype=np.float32))
    return (m - alpha * v).astype(np.float32), (m + alpha * v).astype(np.float32)


def load_source_groups(source, model_info, device, smoke):
    if source == "Full VisA":
        groups, audit = load_visa_groups(model_info, device, smoke)
    else:
        groups, audit = load_mvtec_groups(smoke)
    return groups, audit


def build_nuisance_basis(source, groups, baseline_states):
    cats = sorted(groups)
    centroids, counts = [], []
    for cat in cats:
        data = groups[cat]
        x = np.asarray(data["x"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.float32)
        normal = x[y < 0.5]
        if normal.size == 0:
            normal = x
        centroids.append(normal.mean(axis=0))
        counts.append(int(normal.shape[0]))
    centroids = np.stack(centroids, axis=0).astype(np.float32)
    weights = np.asarray(counts, dtype=np.float64)
    weights = weights / max(float(weights.sum()), 1e-12)
    mu_global = np.sum(centroids * weights[:, None].astype(np.float32), axis=0)
    deviations = centroids - mu_global[None, :]
    rank = min(4, max(1, len(cats) - 1))
    _u, s, vt = np.linalg.svd(deviations.astype(np.float64), full_matrices=False)
    basis = vt[:rank].astype(np.float32)
    basis = basis / np.maximum(np.linalg.norm(basis, axis=1, keepdims=True), 1e-12)
    s_n, s_a = baseline_states
    d0 = contrast_direction_np(s_n, s_a)
    proj = basis @ d0
    nuisance_ratio = float(np.sum(proj ** 2) / max(float(np.sum(d0 ** 2)), 1e-12))
    total_var = float(np.sum(s ** 2))
    explained = (s[:rank] ** 2) / max(total_var, 1e-12)
    audit = {
        "source": source,
        "num_categories": len(cats),
        "rank": rank,
        "normal_patch_count": int(sum(counts)),
        "centroid_variance_total": total_var,
        "explained_variance_top_rank": float(np.sum(explained)),
        "nuisance_ratio_E0": nuisance_ratio,
    }
    cat_rows = []
    for cat, cnt, dev in zip(cats, counts, deviations):
        p = basis @ dev
        cat_rows.append({
            "source": source,
            "category": cat,
            "normal_patch_count": cnt,
            "centroid_deviation_norm": float(np.linalg.norm(dev)),
            "basis_projection_norm": float(np.linalg.norm(p)),
            "basis_projection_energy": float(np.sum(p ** 2)),
            "weighted_contribution": float(cnt * np.sum(p ** 2)),
        })
    return basis.astype(np.float32), audit, cat_rows


def project_direction_out(d, basis):
    clean = np.asarray(d, dtype=np.float32) - basis.T @ (basis @ np.asarray(d, dtype=np.float32))
    return unit_np(clean).astype(np.float32)


def nuisance_ratio(direction, basis):
    d = np.asarray(direction, dtype=np.float32)
    return float(np.sum((basis @ d) ** 2) / max(float(np.sum(d ** 2)), 1e-12))


def hardnormal_loss(logits, y):
    neg = torch.nonzero(y.reshape(-1) < 0.5, as_tuple=False).flatten()
    if neg.numel() == 0:
        return torch.zeros((), dtype=logits.dtype, device=logits.device), 0
    flat = logits.reshape(-1)
    k = max(1, int(math.ceil(float(neg.numel()) * TOP_NEG_FRAC)))
    selected = neg[torch.topk(flat.detach()[neg], k=k, largest=True).indices]
    return F.softplus(flat[selected]).mean(), int(k)


def snir_loss(z, y, s_n, s_a, basis_t, lambda_nuis):
    logits = two_state_logits(z, s_n, s_a)
    task, bce, dice = balanced_loss(logits, y)
    ntr, hard_count = hardnormal_loss(logits, y)
    d = F.normalize(s_a, dim=-1) - F.normalize(s_n, dim=-1)
    nuis = torch.sum((basis_t @ d) ** 2) / torch.sum(d ** 2).clamp_min(1e-12)
    return task + ntr + lambda_nuis * nuis, task.detach(), ntr.detach(), nuis, bce.detach(), dice.detach(), hard_count


def train_snir(source, groups, init_states, basis, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / "checkpoints" / f"{source.replace(' ', '_')}_E2_NuisanceRegularized_{suffix}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        return payload["s_n"].astype(np.float32), payload["s_a"].astype(np.float32), payload["curves"], True
    print(f"[SNIR] train E2 {source}", flush=True)
    s_n0, s_a0 = init_states
    s_n = torch.nn.Parameter(torch.tensor(s_n0, dtype=torch.float32, device=device))
    s_a = torch.nn.Parameter(torch.tensor(s_a0, dtype=torch.float32, device=device))
    basis_t = torch.tensor(basis, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW([s_n, s_a], lr=LR, weight_decay=0.0)
    total = sum(len(g["y"]) for g in groups.values())
    steps = max(1, int(math.ceil(total / BATCH_SIZE)))
    if smoke:
        steps = min(steps, 2)
    rng = np.random.default_rng(SEED)
    curves = []
    for epoch in range(1, EPOCHS + 1):
        vals = []
        for _ in range(steps):
            xb, yb, _origin = balanced_batch(groups, rng, BATCH_SIZE)
            z = torch.tensor(xb, dtype=torch.float32, device=device)
            y = torch.tensor(yb, dtype=torch.float32, device=device)
            loss, task, ntr, nuis, bce, dice, hard_count = snir_loss(z, y, s_n, s_a, basis_t, LAMBDA_NUIS)
            opt.zero_grad()
            loss.backward()
            opt.step()
            vals.append((float(loss.detach().cpu()), float(task.cpu()), float(ntr.cpu()), float(nuis.detach().cpu()), float(bce.cpu()), float(dice.cpu()), hard_count))
        arr = np.asarray(vals, dtype=np.float64)
        d_np = contrast_direction_np(s_n.detach().cpu().numpy(), s_a.detach().cpu().numpy())
        row = {
            "source": source,
            "variant": "E2_NuisanceRegularized",
            "epoch": epoch,
            "loss": float(arr[:, 0].mean()),
            "state_loss": float(arr[:, 1].mean()),
            "ntr_loss": float(arr[:, 2].mean()),
            "nuisance_loss": float(arr[:, 3].mean()),
            "bce": float(arr[:, 4].mean()),
            "dice": float(arr[:, 5].mean()),
            "hard_patch_count": float(arr[:, 6].mean()),
            "nuisance_ratio": nuisance_ratio(d_np, basis),
        }
        curves.append(row)
        print(f"[SNIR] {source} epoch {epoch}/{EPOCHS} loss={row['loss']:.4f} nuis={row['nuisance_ratio']:.4f}", flush=True)
    payload = {"s_n": s_n.detach().cpu().numpy().astype(np.float32), "s_a": s_a.detach().cpu().numpy().astype(np.float32), "curves": curves}
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, ckpt)
    return payload["s_n"], payload["s_a"], curves, False


@torch.no_grad()
def score_grid(states, z, device):
    s_n, s_a = states
    s_n_t = torch.tensor(s_n, dtype=torch.float32, device=device)
    s_a_t = torch.tensor(s_a, dtype=torch.float32, device=device)
    return two_state_logits(z.reshape(-1, z.shape[-1]), s_n_t, s_a_t).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)


@torch.no_grad()
def local_zoom_map(local_z, states, device):
    qs = [score_grid(states, local_z[i], device) for i in range(4)]
    return stitch_quadrant74(qs).astype(np.float32)


def lrp_map(global37, local74):
    global74 = torch_resize(global37, (SIDE * 2, SIDE * 2), align_corners=True)
    return (0.5 * global74 + 0.5 * local74).astype(np.float32)


def selected_eval(source, target, maps, metas, variants):
    import experiments.phase_parameter_free_local_zoom_pyramid_fullres as lzp
    old = tuple(lzp.VARIANTS)
    try:
        lzp.VARIANTS = tuple(variants)
        return evaluate_maps(source, target, maps, metas)
    finally:
        lzp.VARIANTS = old


def build_eval_maps(source_info, states_by_variant, clip_model, preprocess, patch, refs, dataset, kind, target, device, smoke, variants):
    n = min(24, len(refs)) if smoke else len(refs)
    maps = {v: [] for v in variants}
    metas = []
    for pos, ref in enumerate(refs[:n]):
        if pos % 25 == 0:
            print(f"[SNIR] eval {source_info['source']}->{target} image {pos}/{n}", flush=True)
        item = item_at(dataset, kind, ref, pos)
        cat, label, path = item_meta(item, kind, ref, target)
        metas.append({"category": cat, "label": int(label), "mask": mask_np(item), "img_path": path})
        global_z = canonical_grid(source_info, patch[pos], device)
        local_z = local_zoom_features(source_info, clip_model, preprocess, path, device)
        for variant in variants:
            states = states_by_variant[variant]
            g37 = score_grid(states, global_z, device)
            l74 = local_zoom_map(local_z, states, device)
            maps[variant].append(lrp_map(g37, l74))
    return maps, metas


def evaluate_pairs(model_infos, states_by_source_variant, clip_model, preprocess, pairs, device, smoke, variants):
    raw, cats = [], []
    for source, target in pairs:
        cells, _audits = protocol_cells(source, device, smoke)
        matches = [c for c in cells if c[1] == target]
        if not matches:
            raise RuntimeError(f"No target cell for {source}->{target}")
        _cid, _target, dataset, refs, patch, kind = matches[0]
        maps, metas = build_eval_maps(model_infos[source], states_by_source_variant[source], clip_model, preprocess, patch, refs, dataset, kind, target, device, smoke, variants)
        rr, cc = selected_eval(source, target, maps, metas, variants)
        raw.extend(rr)
        cats.extend(cc)
        write_csv(OUT / "RAW_METRICS.partial.csv", raw)
        write_csv(OUT / "PER_CATEGORY_FULL_RES.partial.csv", cats)
    return raw, cats


def delta_summary(raw, pairs, variants):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for variant in variants:
        if variant == "E0_Baseline":
            continue
        vals = []
        for source, target in pairs:
            base = by.get((source, target, "E0_Baseline"))
            cur = by.get((source, target, variant))
            if not base or not cur:
                continue
            dap = fnum(cur, "M_FR_P_AP") - fnum(base, "M_FR_P_AP")
            vals.append(dap)
            rows.append({
                "source": source,
                "dataset": target,
                "variant": variant,
                "Delta_M_FR_P_AUROC": fnum(cur, "M_FR_P_AUROC") - fnum(base, "M_FR_P_AUROC"),
                "Delta_M_FR_P_AP": dap,
                "Delta_M_FR_P_F1max": fnum(cur, "M_FR_P_F1max") - fnum(base, "M_FR_P_F1max"),
                "Delta_M_FR_PRO": fnum(cur, "M_FR_PRO") - fnum(base, "M_FR_PRO"),
            })
        rows.append({
            "source": "SUMMARY",
            "dataset": "PILOT",
            "variant": variant,
            "mean_delta_M_FR_P_AP": finite_mean(vals),
            "median_delta_M_FR_P_AP": finite_median(vals),
            "improved": int(sum(v > 0 for v in vals if np.isfinite(v))),
            "non_drop_ge_minus_0_2": int(sum(v >= -0.2 for v in vals if np.isfinite(v))),
            "total": int(sum(np.isfinite(v) for v in vals)),
        })
    return rows


def state_geometry_rows(model_infos, bases, states_by_source_variant):
    rows = []
    for source, variants in states_by_source_variant.items():
        s0n, s0a = variants["E0_Baseline"]
        d0 = contrast_direction_np(s0n, s0a)
        basis = bases[source]
        for variant, states in variants.items():
            sn, sa = states
            d = contrast_direction_np(sn, sa)
            rows.append({
                "source": source,
                "variant": variant,
                "nuisance_ratio": nuisance_ratio(d, basis),
                "cos_d_E0": float(np.dot(d0, d)),
                "state_angle_deg": float(np.degrees(np.arccos(np.clip(np.dot(d0, d), -1.0, 1.0)))),
                "sN_norm": float(np.linalg.norm(sn)),
                "sA_norm": float(np.linalg.norm(sa)),
            })
    return rows


def decide_early_nuisance(nuisance_rows):
    ratios = [fnum(r, "nuisance_ratio_E0") for r in nuisance_rows]
    return bool(ratios) and max(ratios) < NUISANCE_RATIO_STOP


def e1_has_signal(delta):
    rows = [r for r in delta if r.get("variant") == "E1_PosthocNuisanceRemoval" and r.get("source") != "SUMMARY"]
    vals = [fnum(r, "Delta_M_FR_P_AP") for r in rows]
    return any(v > 0 for v in vals if np.isfinite(v)) and finite_mean(vals) > -1.0


def decide_final(delta, geometry):
    e2 = [r for r in delta if r.get("variant") == "E2_NuisanceRegularized" and r.get("source") != "SUMMARY"]
    vals = [fnum(r, "Delta_M_FR_P_AP") for r in e2]
    mean_delta = finite_mean(vals)
    improved = sum(v > 0 for v in vals if np.isfinite(v))
    non_drop = sum(v >= -0.2 for v in vals if np.isfinite(v))
    focus = max([fnum(r, "Delta_M_FR_P_AP") for r in e2 if (r["source"], r["dataset"]) in {("Full MVTec", "VisA"), ("Full VisA", "MPDD")}], default=float("nan"))
    catastrophic = any(v < -5.0 for v in vals if np.isfinite(v))
    by = {(r["source"], r["variant"]): r for r in geometry}
    drops = []
    for source in sorted({r["source"] for r in geometry}):
        b = by.get((source, "E0_Baseline"))
        e = by.get((source, "E2_NuisanceRegularized"))
        if b and e:
            drops.append(fnum(b, "nuisance_ratio") - fnum(e, "nuisance_ratio"))
    projection_drop = finite_mean(drops)
    stats = {
        "E2_mean_delta_P_AP": mean_delta,
        "E2_improved": improved,
        "E2_non_drop_ge_minus_0_2": non_drop,
        "focus_best_delta_P_AP": focus,
        "catastrophic_drop": catastrophic,
        "mean_nuisance_ratio_drop": projection_drop,
    }
    if mean_delta >= 1.0 and improved >= 3 and focus >= 1.0 and projection_drop > 0.01 and not catastrophic:
        return "SOURCE_NUISANCE_SUPPRESSION_SIGNIFICANTLY_IMPROVES_TRANSFER", stats
    if mean_delta > 0 and improved >= 2 and projection_drop > 0:
        return "SOURCE_NUISANCE_SUPPRESSION_HAS_WEAK_VALUE", stats
    return "SOURCE_NUISANCE_REMOVAL_HAS_NO_COMPETITIVE_VALUE", stats


def write_report(outcome, nuisance_rows, cat_rows, raw, delta, geometry, train_rows, decision, runtime):
    lines = [
        "# Phase SNIR: Source-Nuisance Invariant State Regularization",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "Baseline is Canonical Two-State + NTR + fixed 2x2 Local Re-encoding Pyramid. CLIP features, visual path, absolute Two-State structure, and Fixed LRP inference are unchanged.",
        "",
        "## Gate 0 Source Nuisance Audit",
        "",
        "| Source | categories | rank | normal patches | centroid variance | top-rank explained | nuisance ratio E0 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in nuisance_rows:
        lines.append(f"| {r['source']} | {int(fnum(r,'num_categories'))} | {int(fnum(r,'rank'))} | {int(fnum(r,'normal_patch_count'))} | {fnum(r,'centroid_variance_total'):.6f} | {fnum(r,'explained_variance_top_rank'):.4f} | {fnum(r,'nuisance_ratio_E0'):.4f} |")
    if raw:
        lines += ["", "## Metrics", "", "| Source | Target | Variant | P-AUROC | P-AP | P-F1max | PRO |", "|---|---|---|---:|---:|---:|---:|"]
        for r in raw:
            lines.append(f"| {r['source']} | {r['dataset']} | {r['variant']} | {fnum(r,'M_FR_P_AUROC'):.4f} | {fnum(r,'M_FR_P_AP'):.4f} | {fnum(r,'M_FR_P_F1max'):.4f} | {fnum(r,'M_FR_PRO'):.4f} |")
        lines += ["", "## Delta vs E0", "", "| Source | Target | Variant | Delta P-AP | Delta PRO |", "|---|---|---|---:|---:|"]
        for r in delta:
            if r.get("source") != "SUMMARY":
                lines.append(f"| {r['source']} | {r['dataset']} | {r['variant']} | {fnum(r,'Delta_M_FR_P_AP'):+.4f} | {fnum(r,'Delta_M_FR_PRO'):+.4f} |")
        lines += ["", "## Summary", "", "| Variant | mean Delta P-AP | median | improved | non-drop |", "|---|---:|---:|---:|---:|"]
        for r in delta:
            if r.get("source") == "SUMMARY":
                lines.append(f"| {r['variant']} | {fnum(r,'mean_delta_M_FR_P_AP'):+.4f} | {fnum(r,'median_delta_M_FR_P_AP'):+.4f} | {int(fnum(r,'improved',0))}/{int(fnum(r,'total',0))} | {int(fnum(r,'non_drop_ge_minus_0_2',0))}/{int(fnum(r,'total',0))} |")
    if geometry:
        lines += ["", "## State Geometry", "", "| Source | Variant | nuisance ratio | cos(d,E0) | angle |", "|---|---|---:|---:|---:|"]
        for r in geometry:
            lines.append(f"| {r['source']} | {r['variant']} | {fnum(r,'nuisance_ratio'):.4f} | {fnum(r,'cos_d_E0'):.4f} | {fnum(r,'state_angle_deg'):.3f} |")
    if train_rows:
        last = {}
        for r in train_rows:
            last[r["source"]] = r
        lines += ["", "## E2 Training Endpoints", "", "| Source | source loss | NTR | nuisance ratio |", "|---|---:|---:|---:|"]
        for source, r in sorted(last.items()):
            lines.append(f"| {source} | {fnum(r,'loss'):.4f} | {fnum(r,'ntr_loss'):.4f} | {fnum(r,'nuisance_ratio'):.4f} |")
    lines += ["", f"Decision stats: `{json.dumps(decision, default=json_default)}`", "", f"Runtime: {runtime:.1f}s."]
    write_text(OUT / "REPORT.md", "\n".join(lines) + "\n")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SNIR] device={device}", flush=True)
    preprocess, _ = get_transform(argparse.Namespace(image_size=IMAGE_SIZE))
    clip_model, *_ = load_components(str(CHECKPOINT), device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    model_infos = {"Full VisA": load_visa_source(device), "Full MVTec": load_mvtec_source()}
    groups_by_source, bases, nuisance_rows, category_rows = {}, {}, [], []
    states_by_source_variant = {}
    for source, info in model_infos.items():
        print(f"[SNIR] load source groups {source}", flush=True)
        groups, _audit = load_source_groups(source, info, device, args.smoke)
        groups_by_source[source] = groups
        s0 = source_state_pair(info)
        basis, audit, cat_rows = build_nuisance_basis(source, groups, s0)
        bases[source] = basis
        nuisance_rows.append(audit)
        category_rows.extend(cat_rows)
        d0 = contrast_direction_np(*s0)
        d_clean = project_direction_out(d0, basis)
        states_by_source_variant[source] = {
            "E0_Baseline": s0,
            "E1_PosthocNuisanceRemoval": states_from_direction(info, d_clean),
        }
    write_csv(OUT / "SOURCE_NUISANCE_AUDIT.csv", nuisance_rows)
    write_csv(OUT / "CATEGORY_NUISANCE_CONTRIBUTION.csv", category_rows)

    if decide_early_nuisance(nuisance_rows):
        outcome = "TRANSFERABLE_STATE_HAS_LITTLE_SOURCE_NUISANCE"
        geometry = state_geometry_rows(model_infos, bases, states_by_source_variant)
        runtime = time.time() - start
        write_csv(OUT / "STATE_GEOMETRY.csv", geometry)
        decision = {"max_nuisance_ratio_E0": max(fnum(r, "nuisance_ratio_E0") for r in nuisance_rows)}
        write_json(OUT / "OUTCOME.json", {"outcome": outcome, "decision_stats": decision, "runtime_sec": runtime})
        write_report(outcome, nuisance_rows, category_rows, [], [], geometry, [], decision, runtime)
        print(f"[SNIR] OUTCOME={outcome} runtime={runtime:.1f}s", flush=True)
        return

    raw_e1, cats_e1 = evaluate_pairs(model_infos, states_by_source_variant, clip_model, preprocess, E1_PAIRS, device, args.smoke, ("E0_Baseline", "E1_PosthocNuisanceRemoval"))
    delta_e1 = delta_summary(raw_e1, E1_PAIRS, ("E0_Baseline", "E1_PosthocNuisanceRemoval"))
    write_csv(OUT / "E1_POSTHOC_METRICS.csv", raw_e1)
    write_csv(OUT / "E1_POSTHOC_DELTA.csv", delta_e1)
    write_csv(OUT / "PER_CATEGORY_E1_POSTHOC.csv", cats_e1)

    if not e1_has_signal(delta_e1):
        outcome = "SOURCE_NUISANCE_REMOVAL_HAS_NO_USEFUL_SIGNAL"
        geometry = state_geometry_rows(model_infos, bases, states_by_source_variant)
        runtime = time.time() - start
        decision = {"E1_signal": False}
        write_csv(OUT / "STATE_GEOMETRY.csv", geometry)
        write_json(OUT / "OUTCOME.json", {"outcome": outcome, "decision_stats": decision, "runtime_sec": runtime})
        write_report(outcome, nuisance_rows, category_rows, raw_e1, delta_e1, geometry, [], decision, runtime)
        print(f"[SNIR] OUTCOME={outcome} runtime={runtime:.1f}s", flush=True)
        return

    train_rows = []
    for source, info in model_infos.items():
        s_n, s_a, curves, reused = train_snir(source, groups_by_source[source], states_by_source_variant[source]["E0_Baseline"], bases[source], device, args.smoke)
        states_by_source_variant[source]["E2_NuisanceRegularized"] = (s_n, s_a)
        for r in curves:
            r["checkpoint_reused"] = bool(reused)
            train_rows.append(r)
    raw, cats = evaluate_pairs(model_infos, states_by_source_variant, clip_model, preprocess, PILOT_PAIRS, device, args.smoke, VARIANTS)
    delta = delta_summary(raw, PILOT_PAIRS, VARIANTS)
    geometry = state_geometry_rows(model_infos, bases, states_by_source_variant)
    outcome, decision = decide_final(delta, geometry)
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY_FULL_RES.csv", cats)
    write_csv(OUT / "PILOT_DELTA_SUMMARY.csv", delta)
    write_csv(OUT / "STATE_GEOMETRY.csv", geometry)
    write_csv(OUT / "SOURCE_TRAINING_DIAGNOSTICS.csv", train_rows)
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": decision,
        "runtime_sec": runtime,
        "lambda_nuis": LAMBDA_NUIS,
        "strict_source_only": True,
        "target_training": False,
        "target_statistics": False,
        "target_tuning": False,
        "fixed_lrp": True,
        "epochs": EPOCHS,
        "lr": LR,
    })
    write_report(outcome, nuisance_rows, category_rows, raw, delta, geometry, train_rows, decision, runtime)
    print(f"[SNIR] OUTCOME={outcome} runtime={runtime:.1f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
