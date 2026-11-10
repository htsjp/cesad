import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.phase_paper_ablation_close as abl_close  # noqa: E402
from experiments.phase_cmab_compact_multi_anomaly_basis import (  # noqa: E402
    K_VARIANTS as CMAB_K_VARIANTS,
    basis_rows,
    build_pair_maps as cmab_build_pair_maps,
    evaluate_maps as cmab_evaluate_maps,
    state_pair_from_model_info,
)
from experiments.phase_final_cross_dataset_benchmark import load_mvtec_source, load_visa_source, protocol_cells  # noqa: E402
from experiments.phase_paper_ablation_close import (  # noqa: E402
    CACHE as ABL_CACHE,
    PAIRS as ABL_PAIRS,
    SUMMARY_METRICS,
    add_raw_rows,
    collect_mvtec_pool,
    collect_visa_pool,
    eval_core_variant_fast,
    evaluate_maps_fast,
    lrp_map_for_image,
    local_feature_cache_for_cell,
    load_base_v,
    train_contrast_rho,
)
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import item_at, item_meta, mask_np, quadrant_crops, torch_resize  # noqa: E402
from experiments.phase_x0a_mandatory_gates import load_components  # noqa: E402
from experiments.phase_xa_smoke import write_csv  # noqa: E402
from experiments.phase_xlv_cross_domain_transfer import CHECKPOINT, top1_image_score  # noqa: E402
from experiments.phase_xxxviii_visual_state_representation import IMAGE_SIZE, SIDE  # noqa: E402
from phase_xa.official_task_forward import normalized_patch_feature, visual_outputs  # noqa: E402
from utils.transforms import get_transform  # noqa: E402


OUT = Path("results/paper_assets/final_evidence")
FIG = OUT / "figures"
E_CACHE = Path("E:/MRAD_final_evidence_cache")
abl_close.CACHE = E_CACHE
CAPACITY_PAIRS = (
    ("Full MVTec", "MPDD"),
    ("Full MVTec", "VisA"),
    ("Full VisA", "BTAD"),
    ("Full VisA", "MVTec"),
)
LAMBDA_VALUES = (0.3, 0.4, 0.5, 0.6, 0.7)
RHO_EXISTING = (0.05, 0.10, 0.15)
RHO_NEW = (0.01, 0.03, 0.20, 0.30)
RHO_ALL = (0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30)
CAPACITY_VARIANTS = {
    "K2_CES": "E0_K1_Baseline",
    "K3_states": "E1_K2_CMAB",
    "K5_states": "E2_K4_CMAB",
}


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fnum(row, key, default=float("nan")):
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def model_infos(device):
    return {"Full MVTec": load_mvtec_source(), "Full VisA": load_visa_source(device)}


def load_cmab_states(source, model_info):
    s_n0, s_a0 = state_pair_from_model_info(model_info)
    states = {"E0_K1_Baseline": (s_n0.astype(np.float32), s_a0[None].astype(np.float32))}
    for variant in ("E1_K2_CMAB", "E2_K4_CMAB"):
        ckpt = Path("results/phase_cmab_compact_multi_anomaly_basis/formal") / f"{source.replace(' ', '_')}_{variant}_formal.pt"
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        states[variant] = (payload["s_n"].astype(np.float32), payload["s_a"].astype(np.float32))
    return states


def existing_capacity_rows():
    rows = read_csv("results/phase_cmab_compact_multi_anomaly_basis/formal/RAW_METRICS.csv")
    out = []
    for r in rows:
        if (r.get("source"), r.get("dataset")) in CAPACITY_PAIRS and r.get("variant") in CAPACITY_VARIANTS.values():
            rr = dict(r)
            rr["record_source"] = "existing_cmab"
            out.append(rr)
    for r in read_csv(OUT / "STATE_CAPACITY_METRICS.csv"):
        if (r.get("source"), r.get("target")) in CAPACITY_PAIRS and r.get("variant") in CAPACITY_VARIANTS.values():
            rr = {
                "source": r["source"],
                "dataset": r["target"],
                "variant": r["variant"],
                "M_FR_P_AUROC": r.get("M_FR_P_AUROC", ""),
                "M_FR_P_AP": r.get("M_FR_P_AP", ""),
                "M_FR_P_F1max": r.get("M_FR_P_F1max", ""),
                "M_FR_PRO": r.get("M_FR_AUPRO", ""),
                "record_source": "existing_final_evidence_matched",
            }
            out.append(rr)
    return out


def run_part_a(model_info_by_source, clip_model, preprocess, device, smoke=False):
    raw = existing_capacity_rows()
    have = {(r["source"], r["dataset"], r["variant"]) for r in raw}
    percat = []
    for source, target in CAPACITY_PAIRS:
        missing = [v for v in CAPACITY_VARIANTS.values() if (source, target, v) not in have]
        if not missing:
            continue
        cells, _ = protocol_cells(source, device, smoke)
        cell = next(c for c in cells if c[1] == target)
        _cid, _target, dataset, refs, patch, kind = cell
        states = load_cmab_states(source, model_info_by_source[source])
        states = {v: states[v] for v in missing}
        n = min(24, len(refs)) if smoke else len(refs)
        print(f"[FinalEvidence:A] evaluate capacity {source}->{target} missing={missing} n={n}", flush=True)
        maps, metas = cmab_build_pair_maps(model_info_by_source[source], states, clip_model, preprocess, patch[:n], refs[:n], dataset, kind, target, device, smoke)
        rr, cc = cmab_evaluate_maps(source, target, maps, metas, missing)
        for r in rr:
            r["record_source"] = "new_matched_subset_eval"
        raw.extend(rr)
        percat.extend(cc)
        write_csv(OUT / "STATE_CAPACITY_METRICS.partial.csv", raw)
    rows = []
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    for source, target in CAPACITY_PAIRS:
        base = by.get((source, target, "E0_K1_Baseline"))
        for display, variant in CAPACITY_VARIANTS.items():
            r = by.get((source, target, variant))
            if not r:
                continue
            row = {
                "source": source,
                "target": target,
                "capacity": display,
                "variant": variant,
                "total_states": {"K2_CES": 2, "K3_states": 3, "K5_states": 5}[display],
                "M_FR_P_AUROC": fnum(r, "M_FR_P_AUROC"),
                "M_FR_P_AP": fnum(r, "M_FR_P_AP"),
                "M_FR_P_F1max": fnum(r, "M_FR_P_F1max"),
                "M_FR_AUPRO": fnum(r, "M_FR_PRO"),
                "Delta_P_AP_vs_K2": fnum(r, "M_FR_P_AP") - fnum(base, "M_FR_P_AP") if base else float("nan"),
                "record_source": r.get("record_source", ""),
            }
            rows.append(row)
    summary = []
    for display in CAPACITY_VARIANTS:
        part = [r for r in rows if r["capacity"] == display]
        vals = np.asarray([float(r["M_FR_P_AP"]) for r in part], dtype=np.float64)
        deltas = np.asarray([float(r["Delta_P_AP_vs_K2"]) for r in part], dtype=np.float64)
        summary.append({
            "capacity": display,
            "total_states": {"K2_CES": 2, "K3_states": 3, "K5_states": 5}[display],
            "mean_P_AP": float(np.nanmean(vals)) if vals.size else float("nan"),
            "mean_delta_P_AP_vs_K2": float(np.nanmean(deltas)) if deltas.size else float("nan"),
            "improved_over_K2": int(np.nansum(deltas > 0)) if deltas.size else 0,
            "total": int(np.sum(np.isfinite(deltas))) if deltas.size else 0,
        })
    write_csv(OUT / "STATE_CAPACITY_METRICS.csv", rows)
    write_csv(OUT / "STATE_CAPACITY_SUMMARY.csv", summary)
    if percat:
        write_csv(OUT / "STATE_CAPACITY_PER_CATEGORY_NEW.csv", percat)
    return rows, summary


def score_global_local(model_info, v, global_patch, local_patches, device):
    shim = dict(model_info)
    shim["v1"] = v
    from experiments.phase_paper_ablation_close import score_feature_grid
    global37 = score_feature_grid(shim, v, global_patch, device)
    qs = [score_feature_grid(shim, v, local_patches[i], device) for i in range(4)]
    local74 = np.zeros((SIDE * 2, SIDE * 2), dtype=np.float32)
    local74[:SIDE, :SIDE] = qs[0]
    local74[:SIDE, SIDE:] = qs[1]
    local74[SIDE:, :SIDE] = qs[2]
    local74[SIDE:, SIDE:] = qs[3]
    global74 = torch_resize(global37, (SIDE * 2, SIDE * 2), align_corners=True)
    return global74.astype(np.float32), local74.astype(np.float32)


def load_seed0_ntr_v(source):
    p = Path("results/phase_paper_ready_final_validation/formal/checkpoints") / f"{source.replace(' ', '_')}_seed0_ntr.pt"
    if p.exists():
        return torch.load(p, map_location="cpu", weights_only=False)["v"].astype(np.float32), str(p)
    p2 = Path("results/paper_assets/final_ablation_close/checkpoints") / f"{source.replace(' ', '_')}_seed0_rho0.10.pt"
    return torch.load(p2, map_location="cpu", weights_only=False)["v"].astype(np.float32), str(p2)


def run_part_b(model_info_by_source, clip_model, preprocess, device, smoke=False):
    if not smoke and (OUT / "LRP_FUSION_SENSITIVITY.csv").exists() and (OUT / "LRP_FUSION_SENSITIVITY_SUMMARY.csv").exists():
        return read_csv(OUT / "LRP_FUSION_SENSITIVITY.csv"), read_csv(OUT / "LRP_FUSION_SENSITIVITY_SUMMARY.csv")
    rows = []
    percat = []
    for source, target in CAPACITY_PAIRS:
        cells, _ = protocol_cells(source, device, smoke)
        _cid, _target, dataset, refs, patch, kind = next(c for c in cells if c[1] == target)
        n = min(24, len(refs)) if smoke else len(refs)
        v, ckpt = load_seed0_ntr_v(source)
        local_patches, metas = local_feature_cache_for_cell(source, target, dataset, refs[:n], kind, clip_model, preprocess, device, smoke=smoke)
        maps = {f"lambda_{lam:.1f}": [] for lam in LAMBDA_VALUES}
        for pos in range(n):
            if pos % 50 == 0:
                print(f"[FinalEvidence:B] fusion {source}->{target} {pos}/{n}", flush=True)
            g, l = score_global_local(model_info_by_source[source], v, patch[pos], local_patches[pos].astype(np.float32), device)
            for lam in LAMBDA_VALUES:
                maps[f"lambda_{lam:.1f}"].append((lam * g + (1.0 - lam) * l).astype(np.float32))
        raw = evaluate_maps_fast(source, target, maps, metas)
        for r in raw:
            lam = float(r["variant"].replace("lambda_", ""))
            rows.append({
                "source": source,
                "target": target,
                "lambda": lam,
                "checkpoint": ckpt,
                "M_FR_P_AUROC": fnum(r, "M_FR_P_AUROC"),
                "M_FR_P_AP": fnum(r, "M_FR_P_AP"),
                "M_FR_P_F1max": fnum(r, "M_FR_P_F1max"),
                "M_FR_AUPRO": fnum(r, "M_FR_PRO"),
                "M_FR_I_AUROC": fnum(r, "M_FR_I_AUROC"),
                "M_FR_I_AP": fnum(r, "M_FR_I_AP"),
            })
    summary = []
    for lam in LAMBDA_VALUES:
        part = [r for r in rows if abs(float(r["lambda"]) - lam) < 1e-9]
        for metric in ("M_FR_P_AP", "M_FR_AUPRO"):
            vals = np.asarray([float(r[metric]) for r in part], dtype=np.float64)
            summary.append({"lambda": lam, "metric": metric, "mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals)), "n": int(vals.size)})
    write_csv(OUT / "LRP_FUSION_SENSITIVITY.csv", rows)
    write_csv(OUT / "LRP_FUSION_SENSITIVITY_SUMMARY.csv", summary)
    return rows, summary


def eval_cached_lrp_for_v(source, target, model_info, v, cell, clip_model, preprocess, device, smoke=False):
    _cid, _target, dataset, refs, patch, kind = cell
    n = min(24, len(refs)) if smoke else len(refs)
    local_patches, metas = local_feature_cache_for_cell(
        source, target, dataset, refs[:n], kind, clip_model, preprocess, device, smoke=smoke
    )
    maps = {"Base+NTR+LRP": []}
    for pos in range(n):
        if pos % 100 == 0:
            print(f"[FinalEvidence:C] cached LRP {source}->{target} {pos}/{n}", flush=True)
        g, l = score_global_local(model_info, v, patch[pos], local_patches[pos].astype(np.float32), device)
        maps["Base+NTR+LRP"].append((0.5 * g + 0.5 * l).astype(np.float32))
    return evaluate_maps_fast(source, target, maps, metas)


def run_part_c(model_info_by_source, clip_model, preprocess, device, smoke=False):
    rows = []
    existing = read_csv("results/paper_assets/final_ablation_close/RHO_SENSITIVITY.csv")
    for r in existing:
        if int(float(r.get("seed", -1))) == 0 and (r.get("source"), r.get("target")) in CAPACITY_PAIRS and r.get("variant") == "Base+NTR+LRP":
            rr = dict(r)
            rr["rho_source"] = "existing_3seed_grid_seed0"
            rows.append(rr)
    for source in ("Full MVTec", "Full VisA"):
        sinfo = collect_mvtec_pool(0, smoke) if source == "Full MVTec" else collect_visa_pool(0, device, smoke)
        cells = [c for c in protocol_cells(source, device, smoke)[0] if (source, c[1]) in CAPACITY_PAIRS]
        for rho in RHO_NEW:
            v, ckpt, loaded = train_contrast_rho(sinfo, 0, rho, device, smoke)
            eval_rows = []
            # Reuse the locked LRP evaluator but avoid redundant CLIP forward passes:
            # only the state direction changes across rho, while frozen global/local
            # features are identical to Part B.
            for cell in cells:
                eval_rows.extend(
                    eval_cached_lrp_for_v(
                        source,
                        cell[1],
                        model_info_by_source[source],
                        v,
                        cell,
                        clip_model,
                        preprocess,
                        device,
                        smoke,
                    )
                )
            for er in eval_rows:
                rows.append({
                    "rho": rho,
                    "variant": "Base+NTR+LRP",
                    "transfer": f"{source} -> {er['dataset']}",
                    "source": source,
                    "target": er["dataset"],
                    "seed": 0,
                    "I_AUROC": fnum(er, "M_FR_I_AUROC"),
                    "I_AP": fnum(er, "M_FR_I_AP"),
                    "P_AUROC": fnum(er, "M_FR_P_AUROC"),
                    "P_AP": fnum(er, "M_FR_P_AP"),
                    "P_F1MAX": fnum(er, "M_FR_P_F1max"),
                    "AUPRO": fnum(er, "M_FR_PRO"),
                    "checkpoint": ckpt,
                    "rho_source": "new_seed0_exploratory",
                    "formal_status": "seed0_only",
                })
    summary = []
    for rho in RHO_ALL:
        part = [r for r in rows if abs(float(r["rho"]) - rho) < 1e-9 and r.get("variant") == "Base+NTR+LRP"]
        for metric in ("P_AP", "AUPRO"):
            vals = np.asarray([float(r[metric]) for r in part if np.isfinite(float(r[metric]))], dtype=np.float64)
            summary.append({"rho": rho, "metric": metric, "mean_seed0": float(vals.mean()) if vals.size else float("nan"), "n": int(vals.size), "point_type": "existing_3seed_grid_seed0" if rho in RHO_EXISTING else "new_seed0_exploratory"})
    write_csv(OUT / "WIDE_RHO_TREND.csv", rows)
    write_csv(OUT / "WIDE_RHO_TREND_SUMMARY.csv", summary)
    return rows, summary


def normalize_map(x):
    x = np.asarray(x, dtype=np.float32)
    mn, mx = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def run_part_d(model_info_by_source, clip_model, preprocess, device, smoke=False):
    FIG.mkdir(parents=True, exist_ok=True)
    source, target = "Full MVTec", "VisA"
    cells, _ = protocol_cells(source, device, smoke)
    _cid, _target, dataset, refs, patch, kind = next(c for c in cells if c[1] == target)
    fb = np.load("results/phase_core_specialist_complementarity_oracle/formal/MAP_CACHE/Full_MVTec__VisA__FBCLIP_Official_Pretrained.npz", mmap_mode="r")["maps"]
    vis = np.load("results/phase_core_specialist_complementarity_oracle/formal/MAP_CACHE/Full_MVTec__VisA__VisualAD_E1c_Full.npz", mmap_mode="r")["maps"]
    wanted = ["macaroni2", "pcb3", "chewinggum"]
    candidates = {w: [] for w in wanted}
    for pos, ref in enumerate(refs):
        item = item_at(dataset, kind, ref, pos)
        cat, label, path = item_meta(item, kind, ref, target)
        if cat in candidates and int(label) == 1:
            m = mask_np(item)
            candidates[cat].append((int(m.sum()), str(path), pos, m))
    selected = []
    for cat, vals in candidates.items():
        vals = sorted(vals, key=lambda x: (x[0], x[1]))
        if vals:
            selected.append((cat, vals[len(vals) // 2]))
    if not selected:
        write_csv(OUT / "VISUAL_FAILURE_CASES_SELECTION.csv", [])
        return []
    v, _ckpt = load_seed0_ntr_v(source)
    rows = []
    fig, axes = plt.subplots(len(selected), 5, figsize=(14, 3.1 * max(1, len(selected))))
    if len(selected) == 1:
        axes = np.asarray([axes])
    for row_idx, (cat, (_area, path, pos, mask)) in enumerate(selected):
        from PIL import Image
        img = Image.open(path).convert("RGB")
        crops = quadrant_crops(path)
        batch = torch.stack([preprocess(c) for c in crops], dim=0).to(device)
        with torch.no_grad():
            out = visual_outputs(clip_model, batch, adapter=None, grad=False)
            lp = normalized_patch_feature(out).float().detach().cpu().numpy().astype(np.float32)
        ours = lrp_map_for_image(model_info_by_source[source], v, patch[pos], lp, device)
        panels = [
            ("input", np.asarray(img)),
            ("GT", mask),
            ("Ours", normalize_map(torch_resize(ours, mask.shape, align_corners=True))),
            ("VisualAD", normalize_map(torch_resize(np.asarray(vis[pos], dtype=np.float32), mask.shape, align_corners=True))),
            ("FB-CLIP", normalize_map(torch_resize(np.asarray(fb[pos], dtype=np.float32), mask.shape, align_corners=True))),
        ]
        for col, (title, arr) in enumerate(panels):
            ax = axes[row_idx, col]
            if title == "input":
                ax.imshow(arr)
            elif title == "GT":
                ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(arr, cmap="magma", vmin=0, vmax=1)
            ax.set_title(f"{cat}\n{title}" if col == 0 else title)
            ax.axis("off")
        rows.append({"category": cat, "image_index": pos, "img_path": path, "mask_area_pixels": _area, "selection_rule": "median anomaly area after sorting by area then path"})
    fig.tight_layout()
    png = FIG / "VISUAL_FAILURE_CASES.png"
    pdf = FIG / "VISUAL_FAILURE_CASES.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    write_csv(OUT / "VISUAL_FAILURE_CASES_SELECTION.csv", rows)
    return rows


def make_figures(cap_summary, fusion_summary, rho_summary):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    xs = [int(r["total_states"]) for r in cap_summary]
    ys = [float(r["mean_P_AP"]) for r in cap_summary]
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("Total state capacity")
    ax.set_ylabel("Mean P-AP")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "state_capacity_vs_transfer.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    pap = [r for r in fusion_summary if r["metric"] == "M_FR_P_AP"]
    ax.plot([float(r["lambda"]) for r in pap], [float(r["mean"]) for r in pap], marker="o")
    ax.axvline(0.5, color="k", ls="--", lw=1)
    ax.set_xlabel("lambda for global score")
    ax.set_ylabel("Mean P-AP")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "lrp_fusion_sensitivity.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    pap = [r for r in rho_summary if r.get("metric") == "P_AP"]
    if not pap and rho_summary and "P_AP_mean" in rho_summary[0]:
        pap = [{"rho": r["rho"], "mean_seed0": r["P_AP_mean"], "point_type": "existing_summary"} for r in rho_summary if r.get("variant") == "Base+NTR+LRP"]
    ax.plot([float(r["rho"]) for r in pap], [float(r["mean_seed0"]) for r in pap], marker="o")
    for r in pap:
        if r.get("point_type") == "new_seed0_exploratory":
            ax.scatter([float(r["rho"])], [float(r["mean_seed0"])], s=80, facecolors="none", edgecolors="r")
    ax.axvline(0.10, color="k", ls="--", lw=1)
    ax.set_xlabel("NTR rho")
    ax.set_ylabel("Mean P-AP, seed0")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "wide_rho_trend.png", dpi=180)
    plt.close(fig)


def write_report(cap_summary, fusion_summary, rho_summary, visual_rows, runtime):
    def find_fusion(lam, metric):
        for r in fusion_summary:
            if abs(float(r["lambda"]) - lam) < 1e-9 and r["metric"] == metric:
                return float(r["mean"])
        return float("nan")
    pap_vals = [find_fusion(l, "M_FR_P_AP") for l in LAMBDA_VALUES]
    pro_vals = [find_fusion(l, "M_FR_AUPRO") for l in LAMBDA_VALUES]
    lines = [
        "# Phase PAPER-FINAL-EVIDENCE",
        "",
        "No target tuning, evaluator change, or new method exploration was performed.",
        "",
        "## A. CES State-Capacity Study",
        "",
        "| Capacity | Total states | Mean P-AP | Mean Delta vs K=2 | Improved/Total |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in cap_summary:
        lines.append(f"| {r['capacity']} | {r['total_states']} | {float(r['mean_P_AP']):.4f} | {float(r['mean_delta_P_AP_vs_K2']):+.4f} | {r['improved_over_K2']}/{r['total']} |")
    lines += [
        "",
        "## B. LRP Fusion Sensitivity",
        "",
        "| lambda(global) | Mean P-AP | Mean AUPRO |",
        "|---:|---:|---:|",
    ]
    for lam in LAMBDA_VALUES:
        lines.append(f"| {lam:.1f} | {find_fusion(lam,'M_FR_P_AP'):.4f} | {find_fusion(lam,'M_FR_AUPRO'):.4f} |")
    lines += [
        "",
        f"- P-AP fluctuation across lambdas: `{np.nanmax(pap_vals)-np.nanmin(pap_vals):.4f}`.",
        f"- AUPRO fluctuation across lambdas: `{np.nanmax(pro_vals)-np.nanmin(pro_vals):.4f}`.",
        "- Formal setting remains lambda=0.5.",
        "",
        "## C. Wide Rho Trend",
        "",
        "| rho | Mean P-AP seed0 | Mean AUPRO seed0 | point type |",
        "|---:|---:|---:|---|",
    ]
    for rho in RHO_ALL:
        p = next((r for r in rho_summary if abs(float(r["rho"]) - rho) < 1e-9 and r.get("metric") == "P_AP"), None)
        a = next((r for r in rho_summary if abs(float(r["rho"]) - rho) < 1e-9 and r.get("metric") == "AUPRO"), None)
        if p is None:
            old = next((r for r in rho_summary if abs(float(r.get("rho", -1)) - rho) < 1e-9 and r.get("variant") == "Base+NTR+LRP"), None)
            if old:
                p = {"mean_seed0": old.get("P_AP_mean", float("nan")), "point_type": "existing_summary"}
                a = {"mean_seed0": old.get("AUPRO_mean", float("nan"))}
        lines.append(f"| {rho:.2f} | {float(p['mean_seed0']) if p else float('nan'):.4f} | {float(a['mean_seed0']) if a else float('nan'):.4f} | {(p or {}).get('point_type','')} |")
    lines += [
        "",
        "## D. VisA Qualitative Cases",
        "",
        "Selected by fixed rule: per category, anomaly images sorted by mask area then path; choose median anomaly area.",
        "",
        "| Category | image index | mask area |",
        "|---|---:|---:|",
    ]
    for r in visual_rows:
        lines.append(f"| {r['category']} | {r['image_index']} | {r['mask_area_pixels']} |")
    lines += [
        "",
        "Artifacts: `figures/VISUAL_FAILURE_CASES.png` and `.pdf`.",
        "",
        f"Runtime: {runtime:.1f}s.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--skip-c", action="store_true", help="skip new rho training/eval")
    args = p.parse_args()
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess, _ = get_transform(argparse.Namespace(image_size=IMAGE_SIZE))
    clip_model, *_ = load_components(str(CHECKPOINT), device)
    clip_model.eval()
    for prm in clip_model.parameters():
        prm.requires_grad_(False)
    infos = model_infos(device)

    cap_rows, cap_summary = run_part_a(infos, clip_model, preprocess, device, args.smoke)
    fusion_rows, fusion_summary = run_part_b(infos, clip_model, preprocess, device, args.smoke)
    if args.skip_c:
        rho_rows = read_csv("results/paper_assets/final_ablation_close/RHO_SENSITIVITY.csv")
        rho_summary = read_csv("results/paper_assets/final_ablation_close/RHO_SENSITIVITY_SUMMARY.csv")
    else:
        rho_rows, rho_summary = run_part_c(infos, clip_model, preprocess, device, args.smoke)
    visual_rows = run_part_d(infos, clip_model, preprocess, device, args.smoke)
    make_figures(cap_summary, fusion_summary, rho_summary)
    write_json(OUT / "OUTCOME.json", {"outcome": "PAPER_FINAL_EVIDENCE_COMPLETE", "runtime_sec": time.time() - start, "smoke": args.smoke})
    write_report(cap_summary, fusion_summary, rho_summary, visual_rows, time.time() - start)
    print(f"[FinalEvidence] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
