import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_final_cross_dataset_benchmark import (
    OUT as FINAL_OUT,
    VARIANTS,
    load_mvtec_source,
    load_visa_source,
    protocol_cells,
    score_cell,
    sota_reference,
)
from experiments.phase_spatial_residual_refinement_standard_metric import fnum
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xxxviii_visual_state_representation import SIDE
from experiments.standard_zsad_metrics import f1max, fast_aupro, safe_ap, safe_auc


OUT = Path("results/phase_official_pixel_evaluation_resolution_compatibility/formal")
DIRECT_PAIRS = {("Full VisA", "MVTec"), ("Full VisA", "BTAD"), ("Full VisA", "CVC-ClinicDB"), ("Full MVTec", "VisA")}


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
    raise TypeError(type(obj).__name__)


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def item_at(dataset, kind, ref, pos):
    return dataset[int(ref)] if kind == "source_dataset" else dataset[pos]


def item_meta(item, kind, ref, target):
    if kind == "source_dataset":
        return str(item.get("cls_name", target)), int(item.get("anomaly", 0))
    row = dict(ref)
    return str(row.get("category", item.get("cls_name", target))), int(row.get("label", item.get("anomaly", 0)))


def mask_np(item):
    m = item["img_mask"].detach().cpu().numpy()
    return (np.squeeze(m) > 0.5).astype(np.uint8)


def resize_score_to_mask(score, mask_shape, align_corners=True):
    x = torch.tensor(np.asarray(score, dtype=np.float32), dtype=torch.float32).view(1, 1, SIDE, SIDE)
    y = F.interpolate(x, size=tuple(mask_shape), mode="bilinear", align_corners=align_corners)
    return y.view(*mask_shape).numpy().astype(np.float32)


def summarize_full(iy, iscore, masks, scores):
    py = np.concatenate([m.reshape(-1) for m in masks]) if masks else np.asarray([], dtype=np.uint8)
    ps = np.concatenate([s.reshape(-1) for s in scores]) if scores else np.asarray([], dtype=np.float32)
    out = {
        "I-AUROC": safe_auc(iy, iscore),
        "I-AP": safe_ap(iy, iscore),
        "I-F1max": f1max(iy, iscore),
        "P-AUROC": safe_auc(py, ps),
        "P-AP": safe_ap(py, ps),
        "P-F1max": f1max(py, ps),
        "PRO": fast_aupro(masks, scores),
    }
    out["AUPRO"] = out["PRO"]
    out["G_pixel"] = 0.5 * (out["P-AUROC"] + out["P-AP"]) if np.isfinite(out["P-AUROC"]) and np.isfinite(out["P-AP"]) else float("nan")
    return out


def evaluate_full_res(model_info, device, smoke, direct_only=True, variants=("M2_Ours_HardNormal_Median",)):
    raw, percat = [], []
    cells, _audits = protocol_cells(model_info["source"], device, smoke)
    for cell_id, target, dataset, idx_or_rows, patch, kind in cells:
        if direct_only and (model_info["source"], target) not in DIRECT_PAIRS:
            continue
        n = min(24, patch.shape[0]) if smoke else patch.shape[0]
        refs = idx_or_rows[:n]
        print(f"[PixelProtocol] {model_info['source']} -> {target} n={n}", flush=True)
        image_scores, maps = score_cell(model_info, patch, device, smoke)
        per_image = []
        for pos, ref in enumerate(refs):
            item = item_at(dataset, kind, ref, pos)
            cat, label = item_meta(item, kind, ref, target)
            gt = mask_np(item) if "img_mask" in item else None
            per_image.append((cat, label, gt))
        for variant in variants:
            cat_acc = {}
            for pos, (cat, label, gt) in enumerate(per_image):
                d = cat_acc.setdefault(cat, {"iy": [], "is": [], "masks": [], "scores": []})
                d["iy"].append(label)
                d["is"].append(float(image_scores[variant][pos]))
                if gt is None:
                    continue
                full_score = resize_score_to_mask(maps[variant][pos], gt.shape, align_corners=True)
                d["masks"].append(gt)
                d["scores"].append(full_score)
            cat_rows = []
            for cat, vals in sorted(cat_acc.items()):
                row = summarize_full(vals["iy"], vals["is"], vals["masks"], vals["scores"])
                row.update({"source": model_info["source"], "dataset": target, "category": cat, "variant": variant, "pixel_protocol": "FULL_RES_518_BILINEAR_ALIGN_CORNERS_TRUE", "mask_images": len(vals["masks"])})
                percat.append(row)
                cat_rows.append(row)
            macro = {"source": model_info["source"], "dataset": target, "variant": variant, "cell": cell_id, "pixel_protocol": "FULL_RES_518_BILINEAR_ALIGN_CORNERS_TRUE", "aggregation": "category_macro_mean", "score_resize": "37x37_to_GT_mask_shape_bilinear_align_corners_true", "gt_resolution": "dataset_transformed_mask_shape", "num_categories": len(cat_rows), "mask_images": int(sum(fnum(r, "mask_images") for r in cat_rows))}
            for met in ("I-AUROC", "I-AP", "I-F1max", "P-AUROC", "P-AP", "P-F1max", "PRO", "AUPRO", "G_pixel"):
                vals = np.asarray([fnum(r, met) for r in cat_rows], dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                macro[met] = float(vals.mean()) if vals.size else float("nan")
            raw.append(macro)
    return raw, percat


def local_protocol_rows():
    return [
        {
            "component": "MRAD_CURRENT_STANDARD",
            "score_resolution": f"{SIDE}x{SIDE}",
            "gt_resolution_for_pixel_metrics": f"{SIDE}x{SIDE}",
            "gt_resize": "torch.nn.functional.interpolate(..., mode='nearest') after binary threshold >0.5",
            "score_resize": "none",
            "metrics": "P-AUROC/P-AP/P-F1max/PRO computed on patch grid",
            "source_file": "experiments/standard_zsad_metrics.py:40-46,121-146",
        },
        {
            "component": "MRAD_FULL_RES_ADDED_FOR_AUDIT",
            "score_resolution": f"{SIDE}x{SIDE} -> GT mask shape",
            "gt_resolution_for_pixel_metrics": "dataset transformed mask shape, normally 518x518",
            "gt_resize": "none in this audit; target loader transformed mask is thresholded >0.5",
            "score_resize": "bilinear, align_corners=True, chosen to match MoECLIP official test path",
            "metrics": "P-AUROC/P-AP/P-F1max/PRO computed on full resized mask grid",
            "source_file": "experiments/phase_official_pixel_evaluation_resolution_compatibility.py",
        },
    ]


def official_protocol_rows():
    return [
        {
            "method": "MoECLIP",
            "source": "official GitHub + paper",
            "official_url": "https://github.com/CoCoRessa/MoECLIP",
            "input_resolution": "518x518 default for ViT-L/14-336",
            "anomaly_map_resolution": "img_size=518 via F.interpolate",
            "score_interpolation": "bilinear align_corners=True",
            "gt_resolution": "mask resized to img_size using cv2.INTER_NEAREST in test.py; metrics_eval flattens pixel_label and pixel_preds",
            "metric_level": "dataset/category pixel arrays at 518x518, not patch grid",
            "evidence": "test.py lines 101-116; forward_utils.py lines 191-193 and 234-238; paper implementation details lines 286-289/525-528",
        },
        {
            "method": "VisualAD",
            "source": "official GitHub + CVPR OpenAccess/paper",
            "official_url": "https://github.com/7HHHHH/VisualAD",
            "input_resolution": "518 from checkpoint/default",
            "anomaly_map_resolution": "image_size=518 via F.interpolate",
            "score_interpolation": "bilinear align_corners=False in anomaly map generation",
            "gt_resolution": "target_transform Resize((image_size,image_size)) + ToTensor; compute_metrics flattens mask/map",
            "metric_level": "category mean rows over full image-sized pixel arrays, not patch grid",
            "evidence": "test.py lines 36-41,127-148; utils/anomaly_detection.py lines 37-43; utils/metrics.py lines 47-64",
        },
    ]


def naming_audit_rows():
    percat = read_csv(FINAL_OUT / "PER_CATEGORY_STANDARD_METRICS.csv")
    rows = []
    for source, dataset in (("Full VisA", "MVTec"), ("Full MVTec", "VisA"), ("Full MVTec", "BTAD"), ("Full VisA", "BTAD")):
        part = [r for r in percat if r["source"] == source and r["dataset"] == dataset and r["variant"] == "M2_Ours_HardNormal_Median"]
        vals = np.asarray([fnum(r, "P-AP") for r in part], dtype=np.float64)
        rows.append({
            "source": source,
            "dataset": dataset,
            "field_in_bottleneck_report": "P-AP",
            "actual_definition": "category_macro_mean_of_per_category_P-AP",
            "value": float(np.nanmean(vals)) if vals.size else float("nan"),
            "safe_name": "category_macro_P-AP",
            "leaderboard_comparison_allowed": False,
            "note": "Do not compare this macro diagnostic number with dataset-global leaderboard pixel AP.",
        })
    return rows


def sota_gap(full_rows, patch_rows, refs):
    rows = []
    by_patch = {(r["source"], r["dataset"], r["variant"]): r for r in patch_rows}
    for r in full_rows:
        matches = [ref for ref in refs if ref.get("comparison_type") == "DIRECT_COMPARABLE" and ref["training_source"] == r["source"] and ref["target"] == r["dataset"]]
        if not matches:
            rows.append({"source": r["source"], "dataset": r["dataset"], "variant": r["variant"], "pixel_protocol": r["pixel_protocol"], "comparison_type": "REFERENCE_ONLY", "reference": "no_direct_reference_available"})
            continue
        for ref in matches:
            ref_name = f"{ref['paper']} {ref['venue']} {ref['year']}"
            exact = ref["paper"] == "MoECLIP"
            comp = "DIRECT_COMPARABLE" if exact else "RESOLUTION_MISMATCH_REFERENCE_ONLY"
            row = {"source": r["source"], "dataset": r["dataset"], "variant": r["variant"], "reference": ref_name, "pixel_protocol": r["pixel_protocol"], "comparison_type": comp}
            patch = by_patch.get((r["source"], r["dataset"], r["variant"]), {})
            for cur, refk in (("P-AUROC", "P_AUROC"), ("P-AP", "P_AP"), ("P-F1max", "P_F1max"), ("PRO", "PRO"), ("G_pixel", "G_pixel")):
                rv = fnum(ref, refk)
                row[f"FULL_RES_{cur}"] = r.get(cur, "NA")
                row[f"PATCH_GRID_{cur}"] = patch.get(cur, "NA")
                row[f"reference_{cur}"] = ref.get(refk, "NA")
                row[f"Delta_FULL_RES_{cur}"] = fnum(r, cur) - rv if np.isfinite(rv) else "NA"
                row[f"Delta_PATCH_GRID_{cur}"] = fnum(patch, cur) - rv if np.isfinite(rv) else "NA"
            rows.append(row)
    return rows


def write_report(outcome, local_rows, official_rows, full_rows, naming_rows, gaps, runtime):
    lines = ["# Official Pixel Evaluation Resolution Compatibility Audit", "", f"OUTCOME: `{outcome}`", "", "No training, tuning, or method change was performed.", "", "## MRAD Current Protocol", ""]
    for r in local_rows:
        lines.append(f"- {r['component']}: score `{r['score_resolution']}`, GT `{r['gt_resolution_for_pixel_metrics']}`, GT resize `{r['gt_resize']}`.")
    lines.extend(["", "## Official Protocol Findings", ""])
    for r in official_rows:
        lines.append(f"- {r['method']}: anomaly map is evaluated at image-sized resolution; interpolation `{r['score_interpolation']}`; GT `{r['gt_resolution']}`. Source: {r['official_url']}.")
    lines.extend(["", "## Full-Res M2 Direct Rows", "", "| Source | Dataset | P-AUROC | P-AP | P-F1max | PRO | G |", "|---|---|---:|---:|---:|---:|---:|"])
    for r in full_rows:
        if r["variant"] == "M2_Ours_HardNormal_Median" and (r["source"], r["dataset"]) in {("Full VisA", "MVTec"), ("Full VisA", "BTAD"), ("Full VisA", "CVC-ClinicDB"), ("Full MVTec", "VisA")}:
            lines.append(f"| {r['source']} | {r['dataset']} | {fnum(r,'P-AUROC'):.4f} | {fnum(r,'P-AP'):.4f} | {fnum(r,'P-F1max'):.4f} | {fnum(r,'PRO'):.4f} | {fnum(r,'G_pixel'):.4f} |")
    lines.extend(["", "## SOTA Gap Policy", "", "Patch-grid MRAD numbers are not direct-comparable to MoECLIP/VisualAD full image-sized pixel metrics. `SOTA_GAP.csv` therefore keeps patch-grid deltas for traceability but uses the full-res audit rows for direct MoECLIP comparison. VisualAD is marked `RESOLUTION_MISMATCH_REFERENCE_ONLY` here because its official implementation uses bilinear `align_corners=False`, while the MoECLIP-matched audit uses `align_corners=True`.", "", "| Source | Dataset | Reference | Comparison | Delta Full P-AP | Delta Patch P-AP |", "|---|---|---|---|---:|---:|"])
    for g in gaps:
        if g.get("variant") == "M2_Ours_HardNormal_Median" and g.get("reference", "").startswith(("MoECLIP", "VisualAD")) and g.get("dataset") in {"MVTec", "BTAD", "CVC-ClinicDB", "VisA"}:
            lines.append(f"| {g['source']} | {g['dataset']} | {g['reference']} | {g['comparison_type']} | {fnum(g,'Delta_FULL_RES_P-AP'):.4f} | {fnum(g,'Delta_PATCH_GRID_P-AP'):.4f} |")
    lines.extend(["", "## Macro/Global Naming", ""])
    for r in naming_rows:
        lines.append(f"- `{r['source']} -> {r['dataset']}` bottleneck reciprocal value `{fnum(r,'value'):.4f}` is `{r['safe_name']}`, not dataset-global leaderboard AP.")
    lines.append(f"\nRuntime: {runtime:.1f}s.")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    patch_rows = read_csv(FINAL_OUT / "RAW_METRICS.csv")
    local_rows = local_protocol_rows()
    official_rows = official_protocol_rows()
    models = [load_visa_source(device), load_mvtec_source()]
    full_raw, full_percat = [], []
    variants = VARIANTS if args.all_variants else ("M2_Ours_HardNormal_Median",)
    for model in models:
        raw, per = evaluate_full_res(model, device, args.smoke, direct_only=not args.all_targets, variants=variants)
        full_raw.extend(raw)
        full_percat.extend(per)
    refs = sota_reference()
    gaps = sota_gap(full_raw, patch_rows, refs)
    naming_rows = naming_audit_rows()
    outcome = "PIXEL_EVALUATION_RESOLUTION_MISMATCH_FOUND"
    write_csv(OUT / "MRAD_PIXEL_PROTOCOL_AUDIT.csv", local_rows)
    write_csv(OUT / "OFFICIAL_IMPLEMENTATION_AUDIT.csv", official_rows)
    write_csv(OUT / "FULL_RES_METRICS.csv", full_raw)
    write_csv(OUT / "FULL_RES_PER_CATEGORY_METRICS.csv", full_percat)
    write_csv(OUT / "MACRO_GLOBAL_NAMING_AUDIT.csv", naming_rows)
    write_csv(OUT / "SOTA_REFERENCE.csv", refs)
    write_csv(OUT / "SOTA_GAP.csv", gaps)
    write_json(OUT / "OUTCOME.json", {"outcome": outcome, "runtime_sec": time.time() - start, "device": device, "no_train": True, "no_tuning": True, "current_mrad_pixel_resolution": f"{SIDE}x{SIDE}", "official_pixel_resolution": "image_sized_518x518", "full_res_added": True})
    write_report(outcome, local_rows, official_rows, full_raw, naming_rows, gaps, time.time() - start)
    print(f"[PixelProtocol] outcome={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--all-targets", action="store_true", help="Also run full-res audit on READY targets without direct SOTA references.")
    p.add_argument("--all-variants", action="store_true", help="Also run M0/M1 full-res metrics; default formal audit only runs final M2.")
    run(p.parse_args())


if __name__ == "__main__":
    main()
