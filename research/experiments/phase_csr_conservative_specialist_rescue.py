import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_core_specialist_complementarity_oracle import (
    MAP_CACHE as CSCA_MAP_CACHE,
    VAR_E0 as CSCA_E0,
    VAR_FB,
    VAR_VIS,
    f1_threshold,
    load_e0_maps,
    load_metas,
    load_specialist_maps,
    pair_slug,
    resize_to_mask,
)
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import fnum, summarize_category
import experiments.phase_parameter_free_local_zoom_pyramid_fullres as lzp
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import top1_image_score


OUT = Path("results/phase_conservative_specialist_rescue/formal")

PAIRS = (
    ("Full MVTec", "VisA"),
    ("Full MVTec", "BTAD"),
    ("Full VisA", "MPDD"),
    ("Full VisA", "MVTec"),
)

LAMBDA_RESCUE = 0.10
AMBIGUOUS_LOW_RANK = 0.50
AMBIGUOUS_HIGH_RANK = 0.95

A0 = "A0_Core"
A1 = "A1_VisualAD_PositiveOnlyRescue"
A2 = "A2_FBCLIP_PositiveOnlyRescue"
A3 = "A3_Consensus_PositiveOnlyRescue"
VARIANTS = (A0, A1, A2, A3)


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


def rank_map(score):
    flat = np.asarray(score, dtype=np.float32).reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty(flat.shape[0], dtype=np.float32)
    if flat.size <= 1:
        ranks[:] = 0.0
    else:
        ranks[order] = np.linspace(0.0, 1.0, flat.size, dtype=np.float32)
    return ranks.reshape(score.shape)


def load_all_maps(source, target, metas, device, smoke):
    q_core = load_e0_maps(source, target, metas)
    q_vis = load_specialist_maps(source, target, VAR_VIS, device, smoke=smoke, batch_size=1)
    q_fb = load_specialist_maps(source, target, VAR_FB, device, smoke=smoke, batch_size=1)
    n = min(len(metas), q_core.shape[0], q_vis.shape[0], q_fb.shape[0])
    return {
        CSCA_E0: q_core[:n],
        VAR_VIS: q_vis[:n],
        VAR_FB: q_fb[:n],
    }, n


def conservative_rescue_maps(source, target, metas, maps_by_model):
    out = {v: [] for v in VARIANTS}
    diag = []
    for i, meta in enumerate(metas):
        mask = np.asarray(meta["mask"], dtype=np.uint8)
        qc_raw = resize_to_mask(maps_by_model[CSCA_E0][i], mask, align_corners=True)
        qv_raw = resize_to_mask(maps_by_model[VAR_VIS][i], mask, align_corners=True)
        qf_raw = resize_to_mask(maps_by_model[VAR_FB][i], mask, align_corners=True)

        qc_rank = rank_map(qc_raw)
        qv_rank = rank_map(qv_raw)
        qf_rank = rank_map(qf_raw)
        ambiguity = (qc_rank >= AMBIGUOUS_LOW_RANK) & (qc_rank <= AMBIGUOUS_HIGH_RANK)
        rv = np.maximum(qv_rank - qc_rank, 0.0)
        rf = np.maximum(qf_rank - qc_rank, 0.0)
        rcons = np.minimum(rv, rf)
        core_scale = float(np.percentile(qc_raw, 95) - np.percentile(qc_raw, 50))
        if not np.isfinite(core_scale) or core_scale <= 1e-8:
            core_scale = float(np.std(qc_raw))
        if not np.isfinite(core_scale) or core_scale <= 1e-8:
            core_scale = 1.0

        variants = {
            A0: qc_raw,
            A1: qc_raw + LAMBDA_RESCUE * core_scale * ambiguity.astype(np.float32) * rv,
            A2: qc_raw + LAMBDA_RESCUE * core_scale * ambiguity.astype(np.float32) * rf,
            A3: qc_raw + LAMBDA_RESCUE * core_scale * ambiguity.astype(np.float32) * rcons,
        }
        for name, arr in variants.items():
            out[name].append(arr.astype(np.float32))

        diag.append({
            "source": source,
            "dataset": target,
            "image_index": i,
            "category": meta["category"],
            "label": int(meta["label"]),
            "core_raw_P50": float(np.percentile(qc_raw, 50)),
            "core_raw_P95": float(np.percentile(qc_raw, 95)),
            "core_rescue_scale": core_scale,
            "ambiguous_pixel_fraction": float(np.mean(ambiguity)),
            "visualad_positive_residual_mean": float(rv.mean()),
            "fbclip_positive_residual_mean": float(rf.mean()),
            "consensus_positive_residual_mean": float(rcons.mean()),
            "visualad_rescue_active_fraction": float(np.mean((ambiguity) & (rv > 0))),
            "fbclip_rescue_active_fraction": float(np.mean((ambiguity) & (rf > 0))),
            "consensus_rescue_active_fraction": float(np.mean((ambiguity) & (rcons > 0))),
            "min_A1_minus_A0": float(np.min(variants[A1] - variants[A0])),
            "min_A2_minus_A0": float(np.min(variants[A2] - variants[A0])),
            "min_A3_minus_A0": float(np.min(variants[A3] - variants[A0])),
        })
    return out, diag


def evaluate_variant_maps(source, target, maps, metas):
    raw_rows, cat_rows = [], []
    for variant in maps:
        print(f"[CSR] M_FR metrics {source} -> {target} {variant}", flush=True)
        by_cat = {}
        for i, meta in enumerate(metas):
            d = by_cat.setdefault(meta["category"], {"labels": [], "image_scores": [], "masks": [], "scores": []})
            score = np.asarray(maps[variant][i], dtype=np.float32)
            d["labels"].append(int(meta["label"]))
            d["image_scores"].append(top1_image_score(score))
            d["masks"].append(np.asarray(meta["mask"], dtype=np.uint8))
            d["scores"].append(score)
        fr_cat = []
        for cat, vals in sorted(by_cat.items()):
            print(f"[CSR] M_FR metrics {source} -> {target} {variant} category={cat}", flush=True)
            resized = [resize_to_mask(s, m, align_corners=True) for s, m in zip(vals["scores"], vals["masks"])]
            met = summarize_category(vals["labels"], vals["image_scores"], vals["masks"], resized)
            row = {
                "source": source,
                "dataset": target,
                "category": cat,
                "variant": variant,
                "protocol": "M_FR",
                "protocol_name": "MoECLIP-compatible full-res",
                "aggregation": "CATEGORY",
                "score_resolution": f"{vals['scores'][0].shape[0]}x{vals['scores'][0].shape[1]}",
            }
            for k, v in met.items():
                row[f"M_FR_{k.replace('-', '_')}"] = v
            cat_rows.append(row)
            fr_cat.append(row)
        raw = {
            "source": source,
            "dataset": target,
            "variant": variant,
            "aggregation": "CATEGORY_MACRO",
            "score_resolution": f"{maps[variant][0].shape[0]}x{maps[variant][0].shape[1]}",
        }
        for met in ("I_AUROC", "I_AP", "I_F1max", "P_AUROC", "P_AP", "P_F1max", "PRO", "G_pixel"):
            vals = np.asarray([fnum(r, f"M_FR_{met}") for r in fr_cat], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            raw[f"M_FR_{met}"] = float(vals.mean()) if vals.size else float("nan")
        raw_rows.append(raw)
    return raw_rows, cat_rows


def delta_summary(raw):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for source, target in PAIRS:
        base = by.get((source, target, A0))
        if not base:
            continue
        for variant in VARIANTS[1:]:
            cur = by.get((source, target, variant))
            if not cur:
                continue
            row = {
                "source": source,
                "dataset": target,
                "variant": variant,
                "A0_M_FR_P_AP": fnum(base, "M_FR_P_AP"),
                "variant_M_FR_P_AP": fnum(cur, "M_FR_P_AP"),
                "Delta_M_FR_P_AP": fnum(cur, "M_FR_P_AP") - fnum(base, "M_FR_P_AP"),
                "Delta_M_FR_P_AUROC": fnum(cur, "M_FR_P_AUROC") - fnum(base, "M_FR_P_AUROC"),
                "Delta_M_FR_P_F1max": fnum(cur, "M_FR_P_F1max") - fnum(base, "M_FR_P_F1max"),
                "Delta_M_FR_PRO": fnum(cur, "M_FR_PRO") - fnum(base, "M_FR_PRO"),
            }
            rows.append(row)
    for variant in VARIANTS[1:]:
        vals = np.asarray([fnum(r, "Delta_M_FR_P_AP") for r in rows if r["variant"] == variant], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        rows.append({
            "source": "SUMMARY",
            "dataset": "PART_A_PILOT",
            "variant": variant,
            "mean_Delta_M_FR_P_AP": float(vals.mean()) if vals.size else float("nan"),
            "median_Delta_M_FR_P_AP": float(np.median(vals)) if vals.size else float("nan"),
            "improved": int(np.sum(vals > 0)),
            "non_drop_ge_minus_0_2": int(np.sum(vals >= -0.2)),
            "non_drop_ge_minus_1_0": int(np.sum(vals >= -1.0)),
            "total": int(vals.size),
        })
    return rows


def posthoc_error_rows(source, target, metas, maps):
    rows = []
    totals = {v: {"fn": 0, "fn_rescue": 0, "tp": 0, "tp_lost": 0, "tn": 0, "new_fp": 0, "fp": 0, "fp_kept": 0} for v in VARIANTS[1:]}
    for i, meta in enumerate(metas):
        y = np.asarray(meta["mask"], dtype=bool).reshape(-1)
        q0 = maps[A0][i].reshape(-1)
        t0 = np.percentile(q0, 95)
        pred0 = q0 >= t0
        for variant in VARIANTS[1:]:
            q = maps[variant][i].reshape(-1)
            t = np.percentile(q, 95)
            pred = q >= t
            fn = y & (~pred0)
            tp = y & pred0
            tn = (~y) & (~pred0)
            fp = (~y) & pred0
            totals[variant]["fn"] += int(fn.sum())
            totals[variant]["fn_rescue"] += int((fn & pred).sum())
            totals[variant]["tp"] += int(tp.sum())
            totals[variant]["tp_lost"] += int((tp & (~pred)).sum())
            totals[variant]["tn"] += int(tn.sum())
            totals[variant]["new_fp"] += int((tn & pred).sum())
            totals[variant]["fp"] += int(fp.sum())
            totals[variant]["fp_kept"] += int((fp & pred).sum())
    for variant, t in totals.items():
        rows.append({
            "source": source,
            "dataset": target,
            "variant": variant,
            "scope": "per_image_top5pct_posthoc",
            "e0_fn_pixels": t["fn"],
            "fn_rescue_pixels": t["fn_rescue"],
            "fn_rescue_ratio": t["fn_rescue"] / max(t["fn"], 1),
            "e0_tp_pixels": t["tp"],
            "tp_destruction_pixels": t["tp_lost"],
            "tp_destruction_ratio": t["tp_lost"] / max(t["tp"], 1),
            "e0_tn_pixels": t["tn"],
            "new_fp_pixels": t["new_fp"],
            "new_fp_ratio": t["new_fp"] / max(t["tn"], 1),
            "e0_fp_pixels": t["fp"],
            "e0_fp_kept_pixels": t["fp_kept"],
            "rescue_efficiency_fn_per_new_fp": t["fn_rescue"] / max(t["new_fp"], 1),
        })
    return rows


def decide_part_a(delta_rows):
    summaries = [r for r in delta_rows if r["source"] == "SUMMARY"]
    best = None
    for row in summaries:
        if best is None or fnum(row, "mean_Delta_M_FR_P_AP") > fnum(best, "mean_Delta_M_FR_P_AP"):
            best = row
    if best is None:
        return "CONSERVATIVE_SPECIALIST_RESCUE_HAS_NO_USEFUL_VALUE", {"part_a_gate": "NO_VALID_ROWS"}
    variant = best["variant"]
    vals = [r for r in delta_rows if r.get("variant") == variant and r.get("source") != "SUMMARY"]
    deltas = np.asarray([fnum(r, "Delta_M_FR_P_AP") for r in vals], dtype=np.float64)
    mean_delta = fnum(best, "mean_Delta_M_FR_P_AP")
    improved = int(float(best.get("improved", 0)))
    nondrop02 = int(float(best.get("non_drop_ge_minus_0_2", 0)))
    nondrop10 = int(float(best.get("non_drop_ge_minus_1_0", 0)))
    d_lookup = {(r["source"], r["dataset"]): fnum(r, "Delta_M_FR_P_AP") for r in vals}
    mvtec_visa = d_lookup.get(("Full MVTec", "VisA"), float("nan"))
    btad = d_lookup.get(("Full MVTec", "BTAD"), 0.0)
    mpdd = d_lookup.get(("Full VisA", "MPDD"), 0.0)
    if (
        mean_delta >= 0.5
        and improved >= 3
        and nondrop02 >= 4
        and mvtec_visa >= 1.0
        and btad >= -0.5
        and mpdd >= -0.5
    ):
        gate = "CONSERVATIVE_RESCUE_PASSES_STRONG_GATE"
    elif (
        mean_delta > 0
        and improved >= 2
        and nondrop10 >= 3
        and (np.isfinite(deltas).any() and float(np.nanmax(deltas)) >= 1.0)
        and (np.isfinite(deltas).all() and float(np.nanmin(deltas)) > -1.0)
    ):
        gate = "CONSERVATIVE_RESCUE_HAS_PARTIAL_SIGNAL"
    else:
        gate = "CONSERVATIVE_SPECIALIST_RESCUE_HAS_NO_USEFUL_VALUE"
    return gate, {
        "part_a_gate": gate,
        "best_part_a_variant": variant,
        "mean_delta_M_FR_P_AP": mean_delta,
        "median_delta_M_FR_P_AP": fnum(best, "median_Delta_M_FR_P_AP"),
        "improved": improved,
        "non_drop_ge_minus_0_2": nondrop02,
        "non_drop_ge_minus_1_0": nondrop10,
        "mvtec_to_visa_delta": mvtec_visa,
        "btad_delta": btad,
        "mpdd_delta": mpdd,
    }


def write_report(raw, deltas, errors, gate, decision, runtime, part_b_status):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    lines = [
        "# Phase CSR: Conservative Specialist Rescue",
        "",
        f"OUTCOME: `{gate if part_b_status == 'SKIPPED' else part_b_status}`",
        "",
        "No VisualAD / FB-CLIP retraining was performed. Part A uses cached Core, VisualAD, and FB-CLIP anomaly maps from CSCA. A0 keeps the raw Core score map; specialist residuals are computed in per-image rank space and added back on the raw Core score scale with a positive-only ambiguity gate.",
        "",
        f"- lambda: {LAMBDA_RESCUE}",
        f"- ambiguity band: rank [{AMBIGUOUS_LOW_RANK}, {AMBIGUOUS_HIGH_RANK}]",
        f"- Part B status: {part_b_status}",
        "",
        "## Part A Metrics",
        "",
        "| Source | Target | Variant | M-FR P-AUROC | M-FR P-AP | M-FR P-F1max | M-FR PRO | Delta P-AP |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for source, target in PAIRS:
        base = by.get((source, target, A0))
        for variant in VARIANTS:
            r = by.get((source, target, variant))
            if not r:
                continue
            delta = 0.0 if variant == A0 or not base else fnum(r, "M_FR_P_AP") - fnum(base, "M_FR_P_AP")
            lines.append(f"| {source} | {target} | {variant} | {fnum(r,'M_FR_P_AUROC'):.4f} | {fnum(r,'M_FR_P_AP'):.4f} | {fnum(r,'M_FR_P_F1max'):.4f} | {fnum(r,'M_FR_PRO'):.4f} | {delta:+.4f} |")
    lines += ["", "## Part A Summary", "", "| Variant | Mean Delta P-AP | Median | Improved / 4 | Non-drop >= -0.2 |", "|---|---:|---:|---:|---:|"]
    for r in deltas:
        if r.get("source") == "SUMMARY":
            lines.append(f"| {r['variant']} | {fnum(r,'mean_Delta_M_FR_P_AP'):+.4f} | {fnum(r,'median_Delta_M_FR_P_AP'):+.4f} | {int(float(r.get('improved',0)))}/{int(float(r.get('total',0)))} | {int(float(r.get('non_drop_ge_minus_0_2',0)))}/{int(float(r.get('total',0)))} |")
    lines += ["", "## Error Diagnostics", "", "| Transfer | Variant | FN rescue | TP destruction | New FP | Rescue efficiency |", "|---|---|---:|---:|---:|---:|"]
    for r in errors:
        lines.append(f"| {r['source']}->{r['dataset']} | {r['variant']} | {fnum(r,'fn_rescue_ratio'):.4f} | {fnum(r,'tp_destruction_ratio'):.4f} | {fnum(r,'new_fp_ratio'):.4f} | {fnum(r,'rescue_efficiency_fn_per_new_fp'):.4f} |")
    lines += [
        "",
        "## Decision",
        "",
        f"- Part A gate: `{decision.get('part_a_gate')}`",
        f"- Best fixed rescue: `{decision.get('best_part_a_variant')}`",
        f"- Mean Delta M-FR P-AP: {decision.get('mean_delta_M_FR_P_AP', float('nan')):+.4f}",
        f"- Improved: {decision.get('improved', 0)}/4",
        "",
        "Part B learned gate is intentionally skipped unless Part A reaches the strong or partial gate. This keeps the phase faithful to the stop rule and avoids training a reliability gate when the fixed conservative residual has no deployable signal.",
        "",
        f"Runtime: {runtime:.1f}s.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    raw, percat, rescue_diag, error_diag = [], [], [], []
    for source, target in PAIRS:
        print(f"[CSR] Part A {source} -> {target}", flush=True)
        metas = load_metas(source, target, device, smoke=args.smoke)
        maps_by_model, n_maps = load_all_maps(source, target, metas, device, args.smoke)
        metas = metas[:n_maps]
        maps, diag = conservative_rescue_maps(source, target, metas, maps_by_model)
        rescue_diag.extend(diag)
        err = posthoc_error_rows(source, target, metas, maps)
        error_diag.extend(err)
        r, c = evaluate_variant_maps(source, target, maps, metas)
        raw.extend(r)
        percat.extend(c)
        write_csv(OUT / "RAW_METRICS.partial.csv", raw)
        write_csv(OUT / "GATE_DIAGNOSTICS.partial.csv", rescue_diag)
    deltas = delta_summary(raw)
    gate, decision = decide_part_a(deltas)
    part_b_status = "SKIPPED"
    if gate in {"CONSERVATIVE_RESCUE_PASSES_STRONG_GATE", "CONSERVATIVE_RESCUE_HAS_PARTIAL_SIGNAL"}:
        part_b_status = "PART_B_NOT_IMPLEMENTED_IN_THIS_RUN_SOURCE_MAP_GATE_TRAINING_REQUIRED"

    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY_FULL_RES.csv", percat)
    write_csv(OUT / "PART_A_DELTA_SUMMARY.csv", deltas)
    write_csv(OUT / "GATE_DIAGNOSTICS.csv", rescue_diag)
    write_csv(OUT / "ERROR_DIAGNOSTICS.csv", error_diag)
    runtime = time.time() - t0
    final_outcome = gate if part_b_status == "SKIPPED" else part_b_status
    write_json(OUT / "OUTCOME.json", {
        "outcome": final_outcome,
        "part_a_gate": gate,
        "part_b_status": part_b_status,
        "decision": decision,
        "lambda": LAMBDA_RESCUE,
        "ambiguity_low_rank": AMBIGUOUS_LOW_RANK,
        "ambiguity_high_rank": AMBIGUOUS_HIGH_RANK,
        "no_visualad_retraining": True,
        "no_fbclip_retraining": True,
        "no_target_tuning": True,
        "runtime_sec": runtime,
    })
    write_report(raw, deltas, error_diag, final_outcome, decision, runtime, part_b_status)
    print(f"[CSR] OUTCOME={final_outcome} runtime={runtime:.1f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
