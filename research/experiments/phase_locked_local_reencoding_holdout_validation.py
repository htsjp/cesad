import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_final_cross_dataset_benchmark import load_mvtec_source, load_visa_source, protocol_cells
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import (
    CORE_PAIRS,
    OUT as LZP_OUT,
    VARIANTS,
    build_maps,
    evaluate_maps,
    fnum,
    json_default,
    resolution_control,
    sota_gap,
    write_json,
)
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import CHECKPOINT
from experiments.phase_xxxviii_visual_state_representation import IMAGE_SIZE
from phase_xa.official_task_forward import visual_outputs
from utils.transforms import get_transform


OUT = Path("results/phase_locked_local_reencoding_holdout_validation/formal")
HOLDOUT_DATASETS = {"DTD-Synthetic", "CVC-ColonDB", "Kvasir"}
HOLDOUT_PAIRS = {
    ("Full VisA", "DTD-Synthetic"),
    ("Full VisA", "CVC-ColonDB"),
    ("Full VisA", "Kvasir"),
    ("Full MVTec", "DTD-Synthetic"),
    ("Full MVTec", "CVC-ColonDB"),
    ("Full MVTec", "Kvasir"),
}
INDUSTRIAL = {"MVTec", "VisA", "BTAD", "MPDD", "DTD-Synthetic"}
MEDICAL = {"CVC-ClinicDB", "CVC-ColonDB", "Kvasir"}


def read_csv(path):
    import csv

    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def run_holdouts(smoke=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocess, _ = get_transform(argparse.Namespace(image_size=IMAGE_SIZE))
    clip_model, *_ = load_components(str(CHECKPOINT), device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)
    # Keep visual_outputs imported in this module so the same frozen visual path is loaded.
    _ = visual_outputs
    models = [load_visa_source(device), load_mvtec_source()]
    raw, cats = [], []
    partial = OUT / "HOLDOUT_FULLRES_METRICS.partial.csv"
    if partial.exists() and not smoke:
        raw = read_csv(partial)
    done = {(r["source"], r["dataset"]) for r in raw if r.get("variant") == "E3_ReencodingResolutionControl"}
    for model_info in models:
        cells, _audits = protocol_cells(model_info["source"], device, smoke)
        for cell in cells:
            _cell_id, target, dataset, idx_or_rows, patch, kind = cell
            pair = (model_info["source"], target)
            if pair not in HOLDOUT_PAIRS or pair in done:
                if pair in done:
                    print(f"[LockedLZP] skip completed {pair}", flush=True)
                continue
            n = min(24, patch.shape[0]) if smoke else patch.shape[0]
            refs = idx_or_rows[:n]
            print(f"[LockedLZP] evaluate {model_info['source']} -> {target} n={n}", flush=True)
            maps, metas, _scale = build_maps(model_info, clip_model, preprocess, patch[:n], refs, dataset, kind, target, device)
            rr, cc = evaluate_maps(model_info["source"], target, maps, metas)
            raw.extend(rr)
            cats.extend(cc)
            write_csv(OUT / "HOLDOUT_FULLRES_METRICS.partial.csv", raw)
            write_csv(OUT / "HOLDOUT_PER_CATEGORY_FULL_RES.partial.csv", cats)
    return raw, cats


def domain_summary(all_rows):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in all_rows}
    rows = []
    pairs = sorted({(r["source"], r["dataset"]) for r in all_rows})
    delta_rows = []
    for source, dataset in pairs:
        b = by.get((source, dataset, "B0_HardNormal"))
        e = by.get((source, dataset, "E2_GlobalLocalZoomPyramid"))
        if not b or not e:
            continue
        row = {"source": source, "dataset": dataset, "domain": "industrial" if dataset in INDUSTRIAL else "medical"}
        for met in ("P_AUROC", "P_AP", "P_F1max", "PRO", "G_pixel"):
            row[f"Delta_M_FR_{met}"] = fnum(e, f"M_FR_{met}") - fnum(b, f"M_FR_{met}")
            row[f"Delta_V_FR_{met}"] = fnum(e, f"V_FR_{met}") - fnum(b, f"V_FR_{met}")
        delta_rows.append(row)
    rows.extend(delta_rows)
    for domain in ("industrial", "medical", "all"):
        part = delta_rows if domain == "all" else [r for r in delta_rows if r["domain"] == domain]
        for met in ("P_AUROC", "P_AP", "P_F1max", "PRO", "G_pixel"):
            vals = np.asarray([fnum(r, f"Delta_M_FR_{met}") for r in part], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            rows.append({
                "source": "SUMMARY",
                "dataset": "SUMMARY",
                "domain": domain,
                "metric": f"M_FR_{met}",
                "total": int(vals.size),
                "improved": int(np.sum(vals > 0)),
                "mean_delta": float(vals.mean()) if vals.size else float("nan"),
                "median_delta": float(np.median(vals)) if vals.size else float("nan"),
                "worst_delta": float(vals.min()) if vals.size else float("nan"),
            })
    return rows, delta_rows


def resolution_summary(all_rows):
    res = resolution_control(all_rows)
    vals = np.asarray([fnum(r, "E2_minus_E3_M_FR_P_AP") for r in res], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    rows = list(res)
    rows.append({
        "source": "SUMMARY",
        "dataset": "ALL_14",
        "metric": "M_FR_P_AP",
        "mean_abs_E2_minus_E3": float(np.mean(np.abs(vals))) if vals.size else float("nan"),
        "mean_E2_minus_E3": float(np.mean(vals)) if vals.size else float("nan"),
        "median_E2_minus_E3": float(np.median(vals)) if vals.size else float("nan"),
        "n": int(vals.size),
        "interpretation": "LOCAL_REENCODING_NOT_OUTPUT_RESOLUTION_IS_PRIMARY_SIGNAL" if vals.size and float(np.mean(np.abs(vals))) < 1.0 else "OUTPUT_RESOLUTION_OR_DOWNSAMPLING_EFFECT_NONTRIVIAL",
    })
    return rows


def decide(delta_rows):
    industrial = [r for r in delta_rows if r["domain"] == "industrial"]
    medical = [r for r in delta_rows if r["domain"] == "medical"]
    ind_ap = np.asarray([fnum(r, "Delta_M_FR_P_AP") for r in industrial], dtype=np.float64)
    med_ap = np.asarray([fnum(r, "Delta_M_FR_P_AP") for r in medical], dtype=np.float64)
    ind_ap = ind_ap[np.isfinite(ind_ap)]
    med_ap = med_ap[np.isfinite(med_ap)]
    tags = []
    if ind_ap.size and np.mean(ind_ap > 0) >= 0.8 and np.mean(ind_ap) >= 2.0:
        tags.append("LOCAL_REENCODING_IS_STABLE_FOR_INDUSTRIAL_ZSAD")
    if med_ap.size and np.mean(med_ap) < 0:
        tags.append("LOCAL_REENCODING_DEPENDS_ON_GLOBAL_MEDICAL_CONTEXT")
    lookup = {(r["source"], r["dataset"]): fnum(r, "Delta_M_FR_P_AP") for r in delta_rows}
    col_kv = [lookup.get(("Full VisA", "CVC-ColonDB")), lookup.get(("Full VisA", "Kvasir")), lookup.get(("Full MVTec", "CVC-ColonDB")), lookup.get(("Full MVTec", "Kvasir"))]
    col_kv = [x for x in col_kv if x is not None and np.isfinite(x)]
    clinic = [lookup.get(("Full VisA", "CVC-ClinicDB")), lookup.get(("Full MVTec", "CVC-ClinicDB"))]
    clinic = [x for x in clinic if x is not None and np.isfinite(x)]
    if len(col_kv) >= 4 and np.mean(np.asarray(col_kv) > 1.0) >= 0.75 and clinic and np.mean(clinic) < 0:
        tags.append("CLINICDB_CONTEXT_FAILURE_IS_DATASET_SPECIFIC")
    dtd = [lookup.get(("Full VisA", "DTD-Synthetic")), lookup.get(("Full MVTec", "DTD-Synthetic"))]
    dtd = [x for x in dtd if x is not None and np.isfinite(x)]
    if dtd and np.mean(dtd) <= 0:
        tags.append("DTD_HOLDOUT_DOES_NOT_SUPPORT_INDUSTRIAL_GENERALITY")
    if not tags:
        tags.append("LOCAL_REENCODING_HOLDOUT_MIXED")
    return tags


def write_report(holdout_rows, all_rows, domain_rows, resolution_rows, tags, runtime):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in all_rows}
    lines = [
        "# Locked Local Re-Encoding -- Holdout Domain Validation",
        "",
        "OUTCOME: `" + " + ".join(tags) + "`",
        "",
        "No training, tuning, Median, normalization, or Local Zoom definition change was introduced. This run only filled the six locked holdout full-resolution evaluations and then merged them with the existing 8-core LZP table.",
        "",
        "## Holdout M-FR Results",
        "",
        "| Source | Target | Variant | P-AUROC | P-AP | P-F1max | PRO | G |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for source, dataset in sorted(HOLDOUT_PAIRS):
        for variant in ("B0_HardNormal", "E2_GlobalLocalZoomPyramid", "E3_ReencodingResolutionControl"):
            r = by.get((source, dataset, variant))
            if r:
                lines.append(f"| {source} | {dataset} | {variant} | {fnum(r,'M_FR_P_AUROC'):.4f} | {fnum(r,'M_FR_P_AP'):.4f} | {fnum(r,'M_FR_P_F1max'):.4f} | {fnum(r,'M_FR_PRO'):.4f} | {fnum(r,'M_FR_G_pixel'):.4f} |")
    lines += ["", "## Domain Summary", ""]
    for r in domain_rows:
        if r.get("source") == "SUMMARY" and r.get("metric") == "M_FR_P_AP":
            lines.append(f"- {r['domain']}: P-AP improved {r['improved']}/{r['total']}, mean {fnum(r,'mean_delta'):+.4f}, median {fnum(r,'median_delta'):+.4f}, worst {fnum(r,'worst_delta'):+.4f}.")
    rs = next((r for r in resolution_rows if r.get("source") == "SUMMARY"), {})
    lines += [
        "",
        "## Resolution Mechanism",
        "",
        f"- Mean abs E2-E3 M_FR P-AP across merged evaluations: {fnum(rs,'mean_abs_E2_minus_E3'):.4f}.",
        f"- Interpretation: `{rs.get('interpretation','NA')}`.",
        "",
        f"Runtime: {runtime:.1f}s.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    holdout_rows, holdout_cats = run_holdouts(smoke=args.smoke)
    core_rows = read_csv(LZP_OUT / "RAW_METRICS.csv") if (LZP_OUT / "RAW_METRICS.csv").exists() and not args.smoke else []
    all_rows = [r for r in core_rows if (r.get("source"), r.get("dataset")) in CORE_PAIRS] + holdout_rows
    domain_rows, delta_rows = domain_summary(all_rows)
    resolution_rows = resolution_summary(all_rows)
    tags = decide(delta_rows)
    gaps = sota_gap(all_rows)
    runtime = time.time() - start
    write_csv(OUT / "HOLDOUT_FULLRES_METRICS.csv", holdout_rows)
    write_csv(OUT / "HOLDOUT_PER_CATEGORY_FULL_RES.csv", holdout_cats)
    write_csv(OUT / "DOMAIN_GENERALIZATION_SUMMARY.csv", domain_rows)
    write_csv(OUT / "RESOLUTION_MECHANISM_SUMMARY.csv", resolution_rows)
    write_csv(OUT / "SOTA_GAP_FULL_RES.csv", gaps)
    write_json(OUT / "OUTCOME.json", {"outcome": tags, "runtime_sec": runtime, "holdout_pairs": sorted(HOLDOUT_PAIRS)})
    write_report(holdout_rows, all_rows, domain_rows, resolution_rows, tags, runtime)
    print(f"[LockedLZP] OUTCOME={'+'.join(tags)} runtime={runtime:.1f}s", flush=True)


if __name__ == "__main__":
    main()
