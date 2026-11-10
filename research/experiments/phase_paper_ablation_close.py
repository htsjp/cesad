import argparse
import csv
import json
import math
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_final_cross_dataset_benchmark import protocol_cells
from experiments.phase_paper_ready_final_validation import (
    EPOCHS,
    METRIC_KEYS,
    OUT as PAPER_OUT,
    SEEDS,
    balanced_batch,
    collect_mvtec_pool,
    collect_visa_pool,
    eval_core_variant,
    evaluate_selected_maps,
    fnum,
    score_feature_grid,
)
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import (
    METRICS,
    PROTOCOLS,
    evaluate_maps,
    item_at,
    item_meta,
    mask_np,
    quadrant_crops,
    summarize_category,
    torch_resize,
)
from experiments.phase_private_residual_contrast_memory import contrast_logits
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, balanced_loss
from experiments.phase_xlv_cross_domain_transfer import CHECKPOINT, top1_image_score
from experiments.phase_xxxviii_visual_state_representation import IMAGE_SIZE, SIDE
from phase_xa.official_task_forward import normalized_patch_feature, visual_outputs
from utils.transforms import get_transform


OUT = Path("results/paper_assets/final_ablation_close")
FIG = OUT / "figures"
LATEX = OUT / "latex"
CACHE = OUT / "cache"
RHO_VALUES = (0.05, 0.10, 0.15)
PAIRS = (
    ("Full MVTec", "MPDD"),
    ("Full MVTec", "VisA"),
    ("Full VisA", "BTAD"),
    ("Full VisA", "MVTec"),
)
DISPLAY = {
    "Base": "Base",
    "Base+NTR": "+NTR",
    "Base+LRP": "+LRP",
    "Base+NTR+LRP": "+NTR+LRP",
}
SUMMARY_METRICS = ("I_AUROC", "I_AP", "P_AUROC", "P_AP", "P_F1MAX", "AUPRO")
MFR_TO_SIMPLE = {
    "M_FR_I_AUROC": "I_AUROC",
    "M_FR_I_AP": "I_AP",
    "M_FR_I_F1max": "PAPER_I_F1MAX",
    "M_FR_P_AUROC": "P_AUROC",
    "M_FR_P_AP": "P_AP",
    "M_FR_P_F1max": "P_F1MAX",
    "M_FR_PRO": "AUPRO",
}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=json_default), encoding="utf-8")


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


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def metric_from_row(row, key):
    if key == "I_AUROC":
        return fnum(row, "M_FR_I_AUROC")
    if key == "I_AP":
        return fnum(row, "M_FR_I_AP")
    if key == "P_AUROC":
        return fnum(row, "M_FR_P_AUROC")
    if key == "P_AP":
        return fnum(row, "M_FR_P_AP")
    if key == "P_F1MAX":
        return fnum(row, "M_FR_P_F1max")
    if key == "AUPRO":
        return fnum(row, "M_FR_PRO")
    return float("nan")


def simplify_row(row):
    out = {}
    for k in SUMMARY_METRICS:
        out[k] = metric_from_row(row, k)
    return out


def source_slug(source):
    return source.replace("Full ", "").lower()


def pair_slug(source, target):
    return f"{source.replace(' ', '_')}__{target}".replace("/", "_")


def load_existing_multiseed():
    path = PAPER_OUT / "MULTISEED_METRICS.csv"
    return read_csv(path) if path.exists() else []


def existing_lookup(rows):
    out = {}
    for r in rows:
        if (r.get("source"), r.get("dataset")) not in PAIRS:
            continue
        method = r.get("method")
        if method == "StateAligned":
            key = ("Base", 0.0, int(r["seed"]), r["source"], r["dataset"])
        elif method == "StateAligned+NTR":
            key = ("Base+NTR", 0.10, int(r["seed"]), r["source"], r["dataset"])
        elif method == "StateAligned+NTR+LRP":
            key = ("Base+NTR+LRP", 0.10, int(r["seed"]), r["source"], r["dataset"])
        else:
            continue
        out[key] = r
    return out


def train_contrast_rho(source_info, seed, rho, device, smoke=False):
    ckpt = OUT / "checkpoints" / f"{source_info['source'].replace(' ', '_')}_seed{seed}_rho{rho:.2f}.pt"
    if ckpt.exists() and not smoke:
        return torch.load(ckpt, map_location="cpu", weights_only=False)["v"].astype(np.float32), str(ckpt), True
    rng = np.random.default_rng(seed)
    groups = source_info["groups"]
    total_samples = sum(len(g["y"]) for g in groups.values())
    steps_per_epoch = min(3, max(1, int(np.ceil(total_samples / BATCH_SIZE)))) if smoke else max(1, int(np.ceil(total_samples / BATCH_SIZE)))
    param = torch.nn.Parameter(torch.tensor(source_info["init_v"], dtype=torch.float32, device=device))
    m = torch.tensor(source_info["m"], dtype=torch.float32, device=device)
    opt = torch.optim.AdamW([param], lr=1e-3, weight_decay=0.0)
    for epoch in range(1, EPOCHS + 1):
        vals = []
        for _step in range(steps_per_epoch):
            xb, yb = balanced_batch(groups, rng)
            z = torch.tensor(xb, dtype=torch.float32, device=device)
            y = torch.tensor(yb, dtype=torch.float32, device=device)
            v = F.normalize(param, dim=-1)
            logits = contrast_logits(z, m, source_info["alpha"], v)
            task, _bce, _dice = balanced_loss(logits, y)
            neg = torch.nonzero(y < 0.5, as_tuple=False).flatten()
            k = max(1, int(np.ceil(float(len(neg)) * float(rho))))
            idx = neg[torch.topk(logits.detach()[neg], k=k, largest=True).indices]
            ntr = F.softplus(logits[idx]).mean()
            loss = task + ntr
            opt.zero_grad()
            loss.backward()
            opt.step()
            vals.append(float(loss.detach().cpu()))
        print(f"[AblClose] train {source_info['source']} seed={seed} rho={rho:.2f} epoch={epoch}/{EPOCHS} loss={np.mean(vals):.4f}", flush=True)
    with torch.no_grad():
        v = F.normalize(param, dim=-1).detach().cpu().numpy().astype(np.float32)
    if not smoke:
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"v": v, "seed": seed, "rho": rho, "source": source_info["source"]}, ckpt)
    return v, str(ckpt), False


def load_base_v(source_info, seed):
    ckpt = PAPER_OUT / "checkpoints" / f"{source_info['source'].replace(' ', '_')}_seed{seed}_base.pt"
    if ckpt.exists():
        return torch.load(ckpt, map_location="cpu", weights_only=False)["v"].astype(np.float32), str(ckpt)
    return np.asarray(source_info["init_v"], dtype=np.float32), "init_v_fallback"


@torch.no_grad()
def local_feature_cache_for_cell(source, target, dataset, refs, kind, clip_model, preprocess, device, smoke=False):
    suffix = "_smoke" if smoke else ""
    cpath = CACHE / f"local_patch_{pair_slug(source, target)}{suffix}.npz"
    mmap_path = CACHE / f"local_patch_{pair_slug(source, target)}{suffix}.npy"
    meta_path = CACHE / f"local_patch_{pair_slug(source, target)}{suffix}.pkl"
    n = min(24, len(refs)) if smoke else len(refs)
    if mmap_path.exists() and meta_path.exists():
        patches = np.load(mmap_path, mmap_mode="r")
        with meta_path.open("rb") as f:
            meta_pack = pickle.load(f)
        if int(meta_pack.get("n", 0)) >= n and patches.shape[0] >= n:
            return patches[:n], list(meta_pack["metas"][:n])
    if cpath.exists():
        data = np.load(cpath, allow_pickle=True)
        if int(data["n"]) >= n:
            patches = data["patches"][:n]
            metas = list(data["metas"][:n])
            return patches, metas
    cpath.parent.mkdir(parents=True, exist_ok=True)
    patches = np.lib.format.open_memmap(mmap_path, mode="w+", dtype=np.float16, shape=(n, 4, SIDE * SIDE, 1024))
    metas = []
    for pos in range(n):
        if pos % 25 == 0:
            print(f"[AblClose] cache local {source}->{target} {pos}/{n}", flush=True)
        ref = refs[pos]
        item = item_at(dataset, kind, ref, pos)
        cat, label, path = item_meta(item, kind, ref, target)
        crops = quadrant_crops(path)
        batch = torch.stack([preprocess(c) for c in crops], dim=0).to(device)
        out = visual_outputs(clip_model, batch, adapter=None, grad=False)
        patch = normalized_patch_feature(out).float().detach().cpu().numpy().astype(np.float16)
        patches[pos] = patch
        metas.append({"category": cat, "label": int(label), "mask": mask_np(item), "img_path": path})
    patches.flush()
    with meta_path.open("wb") as f:
        pickle.dump({"metas": metas, "n": n}, f)
    return patches, metas


def lrp_map_for_image(model_info, v, global_patch, local_patches, device):
    shim = dict(model_info)
    shim["v1"] = v
    global37 = score_feature_grid(shim, v, global_patch, device)
    qs = [score_feature_grid(shim, v, local_patches[i], device) for i in range(4)]
    local74 = np.zeros((SIDE * 2, SIDE * 2), dtype=np.float32)
    local74[:SIDE, :SIDE] = qs[0]
    local74[:SIDE, SIDE:] = qs[1]
    local74[SIDE:, :SIDE] = qs[2]
    local74[SIDE:, SIDE:] = qs[3]
    global74 = torch_resize(global37, (SIDE * 2, SIDE * 2), align_corners=True)
    return (0.5 * global74 + 0.5 * local74).astype(np.float32)


def evaluate_maps_fast(source, target, maps, metas):
    old_variants = tuple(__import__("experiments.phase_parameter_free_local_zoom_pyramid_fullres", fromlist=["VARIANTS"]).VARIANTS)
    import experiments.phase_parameter_free_local_zoom_pyramid_fullres as lzp
    try:
        lzp.VARIANTS = tuple(maps.keys())
        raw_rows, _cat_rows = evaluate_maps(source, target, maps, metas)
        return raw_rows
    finally:
        lzp.VARIANTS = old_variants


def pg_metrics_like(labels, image_scores, masks, maps):
    side = int(maps[0].shape[0]) if maps else SIDE
    gmasks = []
    for m in masks:
        t = torch.tensor(m.astype(np.float32)).view(1, 1, *m.shape)
        gm = F.interpolate(t, size=(side, side), mode="nearest").view(side, side).numpy()
        gmasks.append((gm > 0.5).astype(np.uint8))
    return summarize_category(labels, image_scores, gmasks, maps)


def eval_lrp_variant(model_info, v, cells, clip_model, preprocess, device, smoke=False):
    rows = []
    for _cell_id, target, dataset, refs, patch, kind in cells:
        if (model_info["source"], target) not in PAIRS:
            continue
        n = min(24, len(refs)) if smoke else len(refs)
        if smoke:
            local_patches, metas = local_feature_cache_for_cell(model_info["source"], target, dataset, refs[:n], kind, clip_model, preprocess, device, smoke=smoke)
        else:
            local_patches, metas = None, []
        maps = {"E2_GlobalLocalZoomPyramid": []}
        for pos in range(n):
            if pos % 50 == 0:
                print(f"[AblClose] score LRP {model_info['source']}->{target} {pos}/{n}", flush=True)
            if smoke:
                lp = local_patches[pos].astype(np.float32)
            else:
                ref = refs[pos]
                item = item_at(dataset, kind, ref, pos)
                cat, label, path = item_meta(item, kind, ref, target)
                crops = quadrant_crops(path)
                batch = torch.stack([preprocess(c) for c in crops], dim=0).to(device)
                with torch.no_grad():
                    out = visual_outputs(clip_model, batch, adapter=None, grad=False)
                    lp = normalized_patch_feature(out).float().detach().cpu().numpy().astype(np.float32)
                metas.append({"category": cat, "label": int(label), "mask": mask_np(item), "img_path": path})
            maps["E2_GlobalLocalZoomPyramid"].append(lrp_map_for_image(model_info, v, patch[pos], lp, device))
        raw = evaluate_maps_fast(model_info["source"], target, maps, metas)
        rows.append(next(r for r in raw if r["variant"] == "E2_GlobalLocalZoomPyramid"))
    return rows


def eval_lrp_variants(model_info, variant_specs, cells, clip_model, preprocess, device, smoke=False):
    out = {name: [] for name, _v in variant_specs}
    for _cell_id, target, dataset, refs, patch, kind in cells:
        if (model_info["source"], target) not in PAIRS:
            continue
        n = min(24, len(refs)) if smoke else len(refs)
        variant_names = [name for name, _v in variant_specs]
        maps = {name: [] for name in variant_names}
        metas = []
        if smoke:
            local_patches, metas = local_feature_cache_for_cell(model_info["source"], target, dataset, refs[:n], kind, clip_model, preprocess, device, smoke=smoke)
        else:
            local_patches = None
        for pos in range(n):
            if pos % 50 == 0:
                print(f"[AblClose] score multi-LRP {model_info['source']}->{target} {pos}/{n}", flush=True)
            if smoke:
                lp = local_patches[pos].astype(np.float32)
            else:
                ref = refs[pos]
                item = item_at(dataset, kind, ref, pos)
                cat, label, path = item_meta(item, kind, ref, target)
                crops = quadrant_crops(path)
                batch = torch.stack([preprocess(c) for c in crops], dim=0).to(device)
                with torch.no_grad():
                    vis_out = visual_outputs(clip_model, batch, adapter=None, grad=False)
                    lp = normalized_patch_feature(vis_out).float().detach().cpu().numpy().astype(np.float32)
                metas.append({"category": cat, "label": int(label), "mask": mask_np(item), "img_path": path})
            for name, vv in variant_specs:
                maps[name].append(lrp_map_for_image(model_info, vv, patch[pos], lp, device))

        raw = evaluate_maps_fast(model_info["source"], target, maps, metas)
        for row in raw:
            out[row["variant"]].append(row)
    return out


def eval_core_variant_fast(model_info, v, cells, device, smoke=False):
    rows = []
    shim = dict(model_info)
    shim["v1"] = v
    for _cell_id, target, dataset, refs, patch, kind in cells:
        if (model_info["source"], target) not in PAIRS:
            continue
        n = min(24, len(refs)) if smoke else len(refs)
        print(f"[AblClose] eval fast core {model_info['source']} -> {target} n={n}", flush=True)
        maps = {"B0_HardNormal": []}
        metas = []
        for pos in range(n):
            ref = refs[pos]
            item = item_at(dataset, kind, ref, pos)
            cat, label, _path = item_meta(item, kind, ref, target)
            maps["B0_HardNormal"].append(score_feature_grid(shim, v, patch[pos], device))
            metas.append({"category": cat, "label": int(label), "mask": mask_np(item)})
        raw = evaluate_maps_fast(model_info["source"], target, maps, metas)
        rows.append(next(r for r in raw if r["variant"] == "B0_HardNormal"))
    return rows


def add_raw_rows(store, source, seed, rho, variant, rows, checkpoint, cache, formal_status):
    for r in rows:
        if (source, r["dataset"]) not in PAIRS:
            continue
        simple = simplify_row(r)
        row = {
            "rho": rho,
            "variant": variant,
            "transfer": f"{source} -> {r['dataset']}",
            "source": source,
            "target": r["dataset"],
            "seed": seed,
            **simple,
            "checkpoint": checkpoint,
            "cache": cache,
            "formal_status": formal_status,
        }
        store.append(row)


def mean_std_by_seed(rows, metrics=SUMMARY_METRICS):
    out = {}
    for m in metrics:
        vals = []
        per_transfer_stds = []
        for source, target in PAIRS:
            part = [r for r in rows if r["source"] == source and r["target"] == target]
            arr = np.asarray([float(r[m]) for r in part if np.isfinite(float(r[m]))], dtype=np.float64)
            if arr.size:
                vals.append(float(arr.mean()))
                per_transfer_stds.append(float(arr.std(ddof=0)))
        out[f"{m}_mean"] = float(np.mean(vals)) if vals else float("nan")
        out[f"{m}_std"] = float(np.mean(per_transfer_stds)) if per_transfer_stds else float("nan")
        out[f"{m}_cell_std"] = float(np.std([float(r[m]) for r in rows if np.isfinite(float(r[m]))], ddof=0)) if rows else float("nan")
    return out


def fmt(mean, std):
    if not np.isfinite(mean):
        return "NA"
    return f"{mean:.2f} ± {std:.2f}"


def build_summary(rows, group_keys):
    out = []
    keys = sorted({tuple(r[k] for k in group_keys) for r in rows})
    for key in keys:
        part = [r for r in rows if tuple(r[k] for k in group_keys) == key]
        stats = mean_std_by_seed(part)
        row = {k: v for k, v in zip(group_keys, key)}
        row.update(stats)
        for m in SUMMARY_METRICS:
            row[m] = fmt(row[f"{m}_mean"], row[f"{m}_std"])
        out.append(row)
    return out


def build_interaction(factor_rows):
    lookup = {}
    for r in factor_rows:
        lookup[(r["variant"], r["source"], r["target"], int(r["seed"]))] = r
    out = []
    for source, target in PAIRS:
        for seed in SEEDS:
            base = lookup.get(("Base", source, target, seed))
            ntr = lookup.get(("Base+NTR", source, target, seed))
            lrp = lookup.get(("Base+LRP", source, target, seed))
            final = lookup.get(("Base+NTR+LRP", source, target, seed))
            if not all([base, ntr, lrp, final]):
                continue
            row = {"source": source, "target": target, "transfer": f"{source} -> {target}", "seed": seed}
            for m in SUMMARY_METRICS:
                row[f"Delta_NTR_{m}"] = float(ntr[m]) - float(base[m])
                row[f"Delta_LRP_base_{m}"] = float(lrp[m]) - float(base[m])
                row[f"Delta_LRP_after_NTR_{m}"] = float(final[m]) - float(ntr[m])
                row[f"Delta_NTR_with_LRP_{m}"] = float(final[m]) - float(lrp[m])
                row[f"Interaction_{m}"] = float(final[m]) - float(ntr[m]) - float(lrp[m]) + float(base[m])
            out.append(row)
    return out


def per_transfer_effect(factor_rows):
    summary = []
    lookup = {}
    for r in factor_rows:
        lookup.setdefault((r["variant"], r["source"], r["target"]), []).append(r)
    for source, target in PAIRS:
        row = {"source": source, "target": target, "transfer": f"{source} -> {target}"}
        means = {}
        for variant in ("Base", "Base+NTR", "Base+LRP", "Base+NTR+LRP"):
            part = lookup.get((variant, source, target), [])
            means[variant] = {}
            for m in ("P_AP", "P_F1MAX", "AUPRO"):
                vals = [float(r[m]) for r in part if np.isfinite(float(r[m]))]
                means[variant][m] = float(np.mean(vals)) if vals else float("nan")
                row[f"{variant}_{m}"] = means[variant][m]
        for m in ("P_AP", "P_F1MAX", "AUPRO"):
            row[f"Delta_NTR_{m}"] = means["Base+NTR"][m] - means["Base"][m]
            row[f"Delta_LRP_base_{m}"] = means["Base+LRP"][m] - means["Base"][m]
            row[f"Delta_LRP_after_NTR_{m}"] = means["Base+NTR+LRP"][m] - means["Base+NTR"][m]
            row[f"Delta_Final_{m}"] = means["Base+NTR+LRP"][m] - means["Base"][m]
        summary.append(row)
    return summary


def source_trace(rho_rows, factor_rows):
    rows = []
    for experiment, src_rows in (("rho_sensitivity", rho_rows), ("factorial_ablation", factor_rows)):
        for r in src_rows:
            for m in SUMMARY_METRICS:
                rows.append({
                    "experiment": experiment,
                    "rho": r.get("rho", ""),
                    "variant": r["variant"],
                    "source": r["source"],
                    "target": r["target"],
                    "seed": r["seed"],
                    "checkpoint": r.get("checkpoint", ""),
                    "metric": m,
                    "value": r[m],
                    "evaluator": "M_FR_full_resolution",
                    "cache": r.get("cache", ""),
                    "formal_status": r.get("formal_status", ""),
                })
    return rows


def latex_table(path, rows, first_cols):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = list(first_cols) + list(SUMMARY_METRICS)
    lines = ["\\begin{tabular}{" + "l" * len(cols) + "}", "\\toprule", " & ".join(cols).replace("_", "\\_") + " \\\\", "\\midrule"]
    for r in rows:
        vals = [str(r.get(c, "")) for c in first_cols] + [str(r.get(m, "")) for m in SUMMARY_METRICS]
        lines.append(" & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def make_figures(rho_summary, factorial_summary):
    FIG.mkdir(parents=True, exist_ok=True)
    # Figure S1
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for variant, label in (("Base+NTR", "NTR"), ("Base+NTR+LRP", "NTR+LRP")):
        part = sorted([r for r in rho_summary if r["variant"] == variant], key=lambda x: float(x["rho"]))
        xs = [float(r["rho"]) for r in part]
        ys = [float(r["P_AP_mean"]) for r in part]
        es = [float(r["P_AP_std"]) for r in part]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=label)
    ax.set_xlabel("NTR tail ratio rho")
    ax.set_ylabel("M_FR P-AP")
    ax.set_xticks(list(RHO_VALUES))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "rho_sensitivity.png", dpi=200)
    fig.savefig(FIG / "rho_sensitivity.pdf")
    plt.close(fig)

    # Figure S2
    order = ("Base", "Base+NTR", "Base+LRP", "Base+NTR+LRP")
    look = {r["variant"]: r for r in factorial_summary}
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ys = [float(look[v]["P_AP_mean"]) for v in order]
    es = [float(look[v]["P_AP_std"]) for v in order]
    ax.bar([DISPLAY[v] for v in order], ys, yerr=es, capsize=3, color=["#777777", "#4c78a8", "#72b7b2", "#54a24b"])
    ax.set_ylabel("M_FR P-AP")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "component_factorial_pap.png", dpi=200)
    fig.savefig(FIG / "component_factorial_pap.pdf")
    plt.close(fig)


def write_report(rho_summary, factorial_summary, interaction_rows, per_transfer_rows, outcome, runtime):
    def find(rows, **kw):
        for r in rows:
            if all(str(r.get(k)) == str(v) for k, v in kw.items()):
                return r
        return None

    rho_paps = {float(r["rho"]): float(r["P_AP_mean"]) for r in rho_summary if r["variant"] == "Base+NTR"}
    rho_lrp_paps = {float(r["rho"]): float(r["P_AP_mean"]) for r in rho_summary if r["variant"] == "Base+NTR+LRP"}
    gap = max(rho_paps.values()) - min(rho_paps.values()) if rho_paps else float("nan")
    gap_lrp = max(rho_lrp_paps.values()) - min(rho_lrp_paps.values()) if rho_lrp_paps else float("nan")
    look = {r["variant"]: r for r in factorial_summary}
    base = look.get("Base", {})
    ntr = look.get("Base+NTR", {})
    lrp = look.get("Base+LRP", {})
    final = look.get("Base+NTR+LRP", {})
    d_ntr = float(ntr.get("P_AP_mean", np.nan)) - float(base.get("P_AP_mean", np.nan))
    d_lrp = float(lrp.get("P_AP_mean", np.nan)) - float(base.get("P_AP_mean", np.nan))
    d_lrp_after = float(final.get("P_AP_mean", np.nan)) - float(ntr.get("P_AP_mean", np.nan))
    d_ntr_with_lrp = float(final.get("P_AP_mean", np.nan)) - float(lrp.get("P_AP_mean", np.nan))
    inter_vals = [float(r["Interaction_P_AP"]) for r in interaction_rows if np.isfinite(float(r["Interaction_P_AP"]))]
    inter = float(np.mean(inter_vals)) if inter_vals else float("nan")
    if abs(inter) <= 0.5:
        inter_text = "approximately additive"
    elif inter > 0:
        inter_text = "positively synergistic"
    else:
        inter_text = "partially overlapping"
    ntr_pos = sum(1 for r in per_transfer_rows if float(r["Delta_NTR_P_AP"]) > 0)
    lrp_pos = sum(1 for r in per_transfer_rows if float(r["Delta_LRP_base_P_AP"]) > 0)
    final_pos = sum(1 for r in per_transfer_rows if float(r["Delta_Final_P_AP"]) > 0)

    lines = [
        "# Phase PAPER-ABLATION-CLOSE",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "No backbone, evaluator, optimizer, LRP fusion, source initialization, or target protocol was changed.",
        "Evaluation uses the same full-resolution/category-macro path as the locked paper benchmark; no patch-grid fast substitute is used for the final report.",
        "",
        "## A. NTR Rho Sensitivity",
        "",
        "| rho | NTR P-AP | NTR+LRP P-AP |",
        "|---:|---:|---:|",
    ]
    for rho in RHO_VALUES:
        a = find(rho_summary, rho=rho, variant="Base+NTR")
        b = find(rho_summary, rho=rho, variant="Base+NTR+LRP")
        lines.append(f"| {rho:.2f} | {a['P_AP'] if a else 'NA'} | {b['P_AP'] if b else 'NA'} |")
    lines += [
        "",
        f"- NTR rho sensitivity gap over P-AP: `{gap:.4f}`.",
        f"- NTR+LRP rho sensitivity gap over P-AP: `{gap_lrp:.4f}`.",
        "- Formal paper setting remains the pre-frozen `rho=0.10`; no target-based reselection was performed.",
        "",
        "## B. 2x2 Factorial Ablation",
        "",
        "| Variant | I-AUROC | I-AP | P-AUROC | P-AP | P-F1max | AUPRO |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in ("Base", "Base+NTR", "Base+LRP", "Base+NTR+LRP"):
        r = look.get(variant, {})
        lines.append(f"| {DISPLAY[variant]} | {r.get('I_AUROC','NA')} | {r.get('I_AP','NA')} | {r.get('P_AUROC','NA')} | {r.get('P_AP','NA')} | {r.get('P_F1MAX','NA')} | {r.get('AUPRO','NA')} |")
    lines += [
        "",
        "## Main Effects (P-AP)",
        "",
        f"- NTR on Base: `{d_ntr:+.4f}`.",
        f"- LRP on Base: `{d_lrp:+.4f}`.",
        f"- LRP after NTR: `{d_lrp_after:+.4f}`.",
        f"- NTR with LRP: `{d_ntr_with_lrp:+.4f}`.",
        f"- Interaction: `{inter:+.4f}` => `{inter_text}`.",
        "",
        "## Per-Transfer P-AP Effects",
        "",
        "| Transfer | Base | +NTR | +LRP | +NTR+LRP | ΔNTR | ΔLRP(base) | Final-Base |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in per_transfer_rows:
        lines.append(
            f"| {r['transfer']} | {r['Base_P_AP']:.4f} | {r['Base+NTR_P_AP']:.4f} | {r['Base+LRP_P_AP']:.4f} | {r['Base+NTR+LRP_P_AP']:.4f} | {r['Delta_NTR_P_AP']:+.4f} | {r['Delta_LRP_base_P_AP']:+.4f} | {r['Delta_Final_P_AP']:+.4f} |"
        )
    lines += [
        "",
        "## Required Answers",
        "",
        f"1. rho=0.10 stable? {'Yes' if max(gap, gap_lrp) <= 1.0 else 'No / disclose sensitivity'}.",
        "2. rho P-AP values are listed in the rho table above.",
        f"3. Maximum P-AP sensitivity gap: `{max(gap, gap_lrp):.4f}`.",
        f"4. Base+LRP independent P-AP gain: `{d_lrp:+.4f}`.",
        f"5. NTR standalone P-AP gain: `{d_ntr:+.4f}`.",
        f"6. LRP after NTR P-AP gain: `{d_lrp_after:+.4f}`.",
        f"7. NTR/LRP relation: `{inter_text}`.",
        f"8. NTR positive transfers: `{ntr_pos}/4`.",
        f"9. LRP standalone positive transfers: `{lrp_pos}/4`.",
        f"10. Final positive transfers over Base: `{final_pos}/4`.",
        "11. Core claim changes: none unless the rho sensitivity outcome above is non-robust; report factual sensitivity if present.",
        "12. Suggested placement: factorial 2x2 ablation in main text or main supplement; rho sensitivity curve/table in supplement.",
        "",
        "## Files",
        "",
        "- `RHO_SENSITIVITY.csv`, `RHO_SENSITIVITY_SUMMARY.csv`",
        "- `FACTORIAL_ABLATION.csv`, `FACTORIAL_ABLATION_SUMMARY.csv`",
        "- `NTR_LRP_INTERACTION.csv`",
        "- `PER_TRANSFER_COMPONENT_EFFECT.csv`",
        "- `SOURCE_TRACE.csv`",
        "- `figures/rho_sensitivity.*`, `figures/component_factorial_pap.*`",
        f"",
        f"Runtime: `{runtime:.1f}s`.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    LATEX.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"[AblClose] device={device}", flush=True)

    existing = {}
    preprocess, _target_transform = get_transform(argparse.Namespace(image_size=IMAGE_SIZE))
    clip_model, *_ = load_components(str(CHECKPOINT), device)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad_(False)

    source_infos = {
        "Full VisA": {seed: collect_visa_pool(seed, device, args.smoke) for seed in SEEDS},
        "Full MVTec": {seed: collect_mvtec_pool(seed, args.smoke) for seed in SEEDS},
    }
    cells_by_source = {source: protocol_cells(source, device, args.smoke)[0] for source in source_infos}

    rho_rows = []
    factorial_rows = []

    for source in ("Full MVTec", "Full VisA"):
        cells = [c for c in cells_by_source[source] if (source, c[1]) in PAIRS]
        for seed in SEEDS:
            sinfo = source_infos[source][seed]
            base_v, base_ckpt = load_base_v(sinfo, seed)
            lrp_specs = []
            base_core_rows = eval_core_variant_fast(sinfo, base_v, cells, device, args.smoke)
            add_raw_rows(factorial_rows, source, seed, 0.0, "Base", base_core_rows, base_ckpt, "target_patch_cache", "EVALUATED_FAST_GLOBAL")
            # Base+LRP is evaluated together with other missing LRP variants.
            lrp_specs.append(("LRP_Base", base_v, 0.0, "Base+LRP", base_ckpt))

            for rho in RHO_VALUES:
                if abs(rho - 0.10) < 1e-9:
                    ntr_ckpt = str(PAPER_OUT / "checkpoints" / f"{source.replace(' ', '_')}_seed{seed}_ntr.pt")
                    ntr_v = torch.load(ntr_ckpt, map_location="cpu", weights_only=False)["v"].astype(np.float32) if Path(ntr_ckpt).exists() else None
                else:
                    ntr_v, ntr_ckpt, _loaded = train_contrast_rho(sinfo, seed, rho, device, args.smoke)
                if ntr_v is None:
                    ntr_v, ntr_ckpt, _loaded = train_contrast_rho(sinfo, seed, rho, device, args.smoke)

                core_rows = []
                reused_all = True
                for target in [t for s, t in PAIRS if s == source]:
                    old = existing.get(("Base+NTR", rho, seed, source, target)) if abs(rho - 0.10) < 1e-9 else None
                    if old:
                        core_rows.append(old)
                    else:
                        reused_all = False
                if not reused_all:
                    core_rows = eval_core_variant_fast(sinfo, ntr_v, cells, device, args.smoke)
                add_raw_rows(rho_rows, source, seed, rho, "Base+NTR", core_rows, ntr_ckpt, "target_patch_cache", "REUSED_EXISTING" if reused_all else "EVALUATED")
                if abs(rho - 0.10) < 1e-9:
                    add_raw_rows(factorial_rows, source, seed, rho, "Base+NTR", core_rows, ntr_ckpt, "target_patch_cache", "REUSED_EXISTING" if reused_all else "EVALUATED")

                lrp_reused = True
                lrp_ntr_rows = []
                for target in [t for s, t in PAIRS if s == source]:
                    old = existing.get(("Base+NTR+LRP", rho, seed, source, target)) if abs(rho - 0.10) < 1e-9 else None
                    if old:
                        lrp_ntr_rows.append(old)
                    else:
                        lrp_reused = False
                if not lrp_reused:
                    lrp_specs.append((f"LRP_NTR_rho{rho:.2f}", ntr_v, rho, "Base+NTR+LRP", ntr_ckpt))
                    continue
                add_raw_rows(rho_rows, source, seed, rho, "Base+NTR+LRP", lrp_ntr_rows, ntr_ckpt, "local_patch_cache", "REUSED_EXISTING" if lrp_reused else "EVALUATED")
                if abs(rho - 0.10) < 1e-9:
                    add_raw_rows(factorial_rows, source, seed, rho, "Base+NTR+LRP", lrp_ntr_rows, ntr_ckpt, "local_patch_cache", "REUSED_EXISTING" if lrp_reused else "EVALUATED")

            if lrp_specs:
                multi = eval_lrp_variants(sinfo, [(name, vv) for name, vv, _rho, _variant, _ckpt in lrp_specs], cells, clip_model, preprocess, device, args.smoke)
                for name, _vv, rho, variant, ckpt in lrp_specs:
                    rows = multi[name]
                    add_raw_rows(rho_rows if variant == "Base+NTR+LRP" else factorial_rows, source, seed, rho, variant, rows, ckpt, "streamed_local_crops", "EVALUATED")
                    if variant == "Base+NTR+LRP" and abs(rho - 0.10) < 1e-9:
                        add_raw_rows(factorial_rows, source, seed, rho, variant, rows, ckpt, "streamed_local_crops", "EVALUATED")

    write_csv(OUT / "RHO_SENSITIVITY.csv", rho_rows)
    write_csv(OUT / "FACTORIAL_ABLATION.csv", factorial_rows)
    rho_summary = build_summary(rho_rows, ["rho", "variant"])
    factorial_summary = build_summary(factorial_rows, ["variant"])
    write_csv(OUT / "RHO_SENSITIVITY_SUMMARY.csv", rho_summary)
    write_csv(OUT / "FACTORIAL_ABLATION_SUMMARY.csv", factorial_summary)
    interaction = build_interaction(factorial_rows)
    per_transfer = per_transfer_effect(factorial_rows)
    trace = source_trace(rho_rows, factorial_rows)
    write_csv(OUT / "NTR_LRP_INTERACTION.csv", interaction)
    write_csv(OUT / "PER_TRANSFER_COMPONENT_EFFECT.csv", per_transfer)
    write_csv(OUT / "SOURCE_TRACE.csv", trace)
    latex_table(LATEX / "table_rho_sensitivity.tex", rho_summary, ["rho", "variant"])
    latex_table(LATEX / "table_factorial_ablation.tex", factorial_summary, ["variant"])
    make_figures(rho_summary, factorial_summary)

    ntr_pap = [float(r["P_AP_mean"]) for r in rho_summary if r["variant"] == "Base+NTR"]
    ntr_lrp_pap = [float(r["P_AP_mean"]) for r in rho_summary if r["variant"] == "Base+NTR+LRP"]
    gap_ntr = max(ntr_pap) - min(ntr_pap) if ntr_pap else float("nan")
    gap_ntr_lrp = max(ntr_lrp_pap) - min(ntr_lrp_pap) if ntr_lrp_pap else float("nan")
    rho_gap = float(np.nanmax([gap_ntr, gap_ntr_lrp]))
    look = {r["variant"]: r for r in factorial_summary}
    d_lrp = float(look.get("Base+LRP", {}).get("P_AP_mean", np.nan)) - float(look.get("Base", {}).get("P_AP_mean", np.nan))
    if np.isfinite(rho_gap) and rho_gap > 1.0:
        outcome = "RHO_SENSITIVITY_MUST_BE_DISCLOSED"
    elif np.isfinite(d_lrp) and d_lrp <= 0:
        outcome = "LRP_REQUIRES_NTR_CONDITIONING_FOR_GAIN"
    elif np.isfinite(d_lrp) and d_lrp < 1.0:
        outcome = "LRP_MAINLY_COMPLEMENTS_NTR"
    else:
        outcome = "ABLATION_EVIDENCE_COMPLETE"
    write_json(OUT / "OUTCOME.json", {"outcome": outcome, "runtime_sec": time.time() - start})
    write_report(rho_summary, factorial_summary, interaction, per_transfer, outcome, time.time() - start)
    print(f"[AblClose] wrote {OUT} outcome={outcome} runtime={time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
