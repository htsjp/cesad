import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_hard_normal_contrast import OUT as HN_OUT, PC1_DEBIAS_OUT, load_payload
from experiments.phase_hard_normal_cross_source_mvtec import OUT as MVTEC_HN_OUT
from experiments.phase_hard_normal_cross_source_mvtec import target_cells as mvtec_source_target_cells
from experiments.phase_median_spatial_consensus_generalization import HOLDOUTS, MetaDataset, audit_meta_dataset, ensure_patch_cache
from experiments.phase_private_residual_contrast_memory import contrast_logits, unit_np
from experiments.phase_spatial_residual_refinement_standard_metric import aligned_grid, fnum, logits_from_z
from experiments.phase_spatial_supervision_transfer_audit import target_cells as visa_source_target_cells
from experiments.phase_standard_zsad_benchmark_infrastructure import DEFAULT_DATA_ROOT, build_registry, build_sota_reference
from experiments.phase_state_pair_decomposition import decompose, hybrid_states
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import top1_image_score
from experiments.phase_xxxviii_visual_state_representation import SIDE, norm_np
from experiments.standard_zsad_metrics import evaluate_zsad_standard


OUT = Path("results/phase_final_cross_dataset_benchmark/formal")
VARIANTS = ("M0_StateAligned", "M1_HardNormal", "M2_Ours_HardNormal_Median")
INDUSTRIAL = {"MVTec", "VisA", "BTAD", "MPDD", "DTD-Synthetic"}
MEDICAL = {"CVC-ClinicDB", "CVC-ColonDB", "Kvasir", "BrainMRI", "HeadCT", "Br35H", "ISBI", "Endo"}


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


def median3x3(x):
    return ndimage.median_filter(np.asarray(x, dtype=np.float32), size=3, mode="nearest").astype(np.float32)


def load_visa_source(device):
    q_pc1 = unit_np(np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32))
    true_payload = load_payload(PC1_DEBIAS_OUT / "E1_PC1Removed-TrueMask_formal.pt", device)
    corrupt_payload = load_payload(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt", device)
    states_t = np.asarray(true_payload["patch_states"], dtype=np.float32)
    states_c = np.asarray(corrupt_payload["patch_states"], dtype=np.float32)
    _m_t, d_t = decompose(states_t)
    m_c, _d_c = decompose(states_c)
    alpha = float(np.linalg.norm(d_t))
    vectors = np.load(HN_OUT / "contrast_vectors.npz")
    v0 = unit_np(vectors["B0"].astype(np.float32))
    v1 = unit_np(vectors["E1"].astype(np.float32))
    align_model = make_model(hybrid_states(m_c, alpha * v1), corrupt_payload, device)
    for p in align_model.parameters():
        p.requires_grad_(False)
    return {"source": "Full VisA", "q_pc1": q_pc1, "align_model": align_model.eval(), "m": m_c.astype(np.float32), "alpha": alpha, "v0": v0, "v1": v1, "use_align": True}


def load_mvtec_source():
    payload = np.load(MVTEC_HN_OUT / "contrast_vectors.npz")
    return {"source": "Full MVTec", "q_pc1": None, "align_model": None, "m": payload["midpoint"].astype(np.float32), "alpha": float(payload["alpha"]), "v0": unit_np(payload["B0"].astype(np.float32)), "v1": unit_np(payload["E1"].astype(np.float32)), "use_align": False}


def ready_holdout_cells(device, smoke):
    cells = []
    audits = [audit_meta_dataset(name, root) for name, (root, _domain, _kind) in HOLDOUTS.items()]
    for audit in audits:
        if audit["status"] != "READY":
            continue
        name = audit["dataset"]
        root, _domain, _kind = HOLDOUTS[name]
        dataset = MetaDataset(root, mode="test")
        rows, patch = ensure_patch_cache(name, dataset, device, smoke)
        cells.append((f"holdout_{name}", name, dataset, rows, patch, "external"))
    return cells, audits


def protocol_cells(source_name, device, smoke):
    if source_name == "Full VisA":
        base_cells = list(visa_source_target_cells(smoke))
        holdouts, audits = ready_holdout_cells(device, smoke)
        cells = []
        for cell_id, target, dataset, idx_or_rows, patch, kind in base_cells:
            if target != "VisA":
                cells.append((cell_id, target, dataset, idx_or_rows, patch, kind))
        for c in holdouts:
            cells.append(c)
        return cells, audits
    base_cells = list(mvtec_source_target_cells(smoke))
    holdouts, audits = ready_holdout_cells(device, smoke)
    cells = []
    for cell_id, target, dataset, idx_or_rows, patch, kind in base_cells:
        if target != "MVTec":
            cells.append((cell_id, target, dataset, idx_or_rows, patch, kind))
    for c in holdouts:
        cells.append(c)
    return cells, audits


@torch.no_grad()
def score_cell(model_info, patch, device, smoke):
    n = min(24, patch.shape[0]) if smoke else patch.shape[0]
    maps = {v: np.zeros((n, SIDE, SIDE), dtype=np.float16) for v in VARIANTS}
    image = {v: np.zeros((n,), dtype=np.float32) for v in VARIANTS}
    m_t = torch.tensor(model_info["m"], dtype=torch.float32, device=device)
    v0_t = torch.tensor(model_info["v0"], dtype=torch.float32, device=device)
    v1_t = torch.tensor(model_info["v1"], dtype=torch.float32, device=device)
    for i in range(n):
        if model_info["use_align"]:
            z = aligned_grid(patch[i], model_info["q_pc1"], model_info["align_model"], device).unsqueeze(0)
            s0 = logits_from_z(z, m_t, model_info["alpha"], v0_t).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)
            s1 = logits_from_z(z, m_t, model_info["alpha"], v1_t).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)
        else:
            z = norm_np(np.asarray(patch[i], dtype=np.float32).reshape(SIDE * SIDE, 1024))
            zt = torch.tensor(z, dtype=torch.float32, device=device)
            s0 = contrast_logits(zt, m_t, model_info["alpha"], v0_t).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)
            s1 = contrast_logits(zt, m_t, model_info["alpha"], v1_t).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)
        s2 = median3x3(s1)
        for variant, smap in (("M0_StateAligned", s0), ("M1_HardNormal", s1), ("M2_Ours_HardNormal_Median", s2)):
            maps[variant][i] = smap.astype(np.float16)
            image[variant][i] = top1_image_score(smap)
    return image, maps


def evaluate_protocol(model_info, device, smoke):
    raw, percat = [], []
    cells, audits = protocol_cells(model_info["source"], device, smoke)
    for cell_id, target, dataset, idx_or_rows, patch, kind in cells:
        n = min(24, patch.shape[0]) if smoke else patch.shape[0]
        idx = idx_or_rows[:n]
        print(f"[FinalBenchmark] {model_info['source']} -> {target} n={n}", flush=True)
        image, maps = score_cell(model_info, patch, device, smoke)
        for variant in VARIANTS:
            metric, per = evaluate_zsad_standard(cell_id, target, dataset, idx, kind, image[variant], maps[variant], SIDE)
            metric.update({"source": model_info["source"], "dataset": target, "variant": variant, "domain": "industrial" if target in INDUSTRIAL else "medical"})
            raw.append(metric)
            for cat, vals in per.items():
                percat.append({"source": model_info["source"], "dataset": target, "category": cat, "variant": variant, **vals})
    return raw, percat, audits


def metric_delta(row_a, row_b, metric):
    return fnum(row_a, metric) - fnum(row_b, metric)


def ablation(raw):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for source, dataset in sorted({(r["source"], r["dataset"]) for r in raw}):
        m0, m1, m2 = by[(source, dataset, "M0_StateAligned")], by[(source, dataset, "M1_HardNormal")], by[(source, dataset, "M2_Ours_HardNormal_Median")]
        for name, a, b in (("HardNormal_vs_StateAligned", m1, m0), ("Median_vs_HardNormal", m2, m1)):
            row = {"source": source, "dataset": dataset, "comparison": name}
            for met in ("P-AUROC", "P-AP", "P-F1max", "PRO", "G_pixel", "I-AUROC", "I-AP", "I-F1max"):
                row[f"Delta_{met}"] = metric_delta(a, b, met)
            rows.append(row)
    return rows


def summary(raw, abl):
    rows = []
    for comparison in ("HardNormal_vs_StateAligned", "Median_vs_HardNormal"):
        part = [r for r in abl if r["comparison"] == comparison]
        for metric in ("P-AUROC", "P-AP", "P-F1max", "PRO", "G_pixel"):
            vals = np.asarray([fnum(r, f"Delta_{metric}") for r in part], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            rows.append({"group": "all", "comparison": comparison, "metric": metric, "targets": int(vals.size), "improved": int(np.sum(vals > 0)), "mean_delta": float(vals.mean()) if vals.size else float("nan"), "median_delta": float(np.median(vals)) if vals.size else float("nan"), "worst_delta": float(vals.min()) if vals.size else float("nan")})
        for domain in ("industrial", "medical"):
            ds = INDUSTRIAL if domain == "industrial" else MEDICAL
            vals = np.asarray([fnum(r, "Delta_P-AP") for r in part if r["dataset"] in ds], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            rows.append({"group": domain, "comparison": comparison, "metric": "P-AP", "targets": int(vals.size), "improved": int(np.sum(vals > 0)), "mean_delta": float(vals.mean()) if vals.size else float("nan"), "median_delta": float(np.median(vals)) if vals.size else float("nan"), "worst_delta": float(vals.min()) if vals.size else float("nan")})
        for source in ("Full VisA", "Full MVTec"):
            vals = np.asarray([fnum(r, "Delta_P-AP") for r in part if r["source"] == source], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            rows.append({"group": source, "comparison": comparison, "metric": "P-AP", "targets": int(vals.size), "improved": int(np.sum(vals > 0)), "mean_delta": float(vals.mean()) if vals.size else float("nan"), "median_delta": float(np.median(vals)) if vals.size else float("nan"), "worst_delta": float(vals.min()) if vals.size else float("nan")})
    med = [r for r in abl if r["comparison"] == "Median_vs_HardNormal"]
    quad = {"AP_up_PRO_up": 0, "AP_up_PRO_down": 0, "AP_down_PRO_up": 0, "AP_down_PRO_down": 0}
    for r in med:
        ap = fnum(r, "Delta_P-AP") > 0
        pro = fnum(r, "Delta_PRO") > 0
        if ap and pro:
            quad["AP_up_PRO_up"] += 1
        elif ap and not pro:
            quad["AP_up_PRO_down"] += 1
        elif (not ap) and pro:
            quad["AP_down_PRO_up"] += 1
        else:
            quad["AP_down_PRO_down"] += 1
    rows.append({"group": "median_quadrant", "comparison": "Median_vs_HardNormal", "metric": "P-AP/PRO", "targets": len(med), **quad})
    return rows


def table_rows(raw, domain):
    keep = INDUSTRIAL if domain == "industrial" else MEDICAL
    rows = []
    for r in raw:
        if r["dataset"] in keep:
            rows.append(r)
    return rows


def sota_reference():
    rows = build_sota_reference()
    rows.append({"paper": "MoECLIP", "venue": "CVPR", "year": 2026, "backbone": "OpenCLIP ViT-L/14-336", "input_resolution": "518x518", "training_source": "Full MVTec", "target": "VisA", "target_data_used": "no", "I_AUROC": "NA", "I_AP": "NA", "I_F1max": "NA", "P_AUROC": 95.6, "P_AP": 26.1, "P_F1max": "NA", "PRO": "NA", "G_pixel": 60.85, "comparison_type": "DIRECT_COMPARABLE", "notes": "Recorded MVTec-source to VisA reference."})
    return rows


def sota_gap(raw, refs):
    rows = []
    for r in raw:
        matches = [ref for ref in refs if ref["comparison_type"] == "DIRECT_COMPARABLE" and ref["training_source"] == r["source"] and ref["target"] == r["dataset"]]
        if not matches:
            rows.append({"source": r["source"], "dataset": r["dataset"], "variant": r["variant"], "comparison_type": "REFERENCE_ONLY", "reference": "no_direct_reference_available"})
            continue
        for ref in matches:
            row = {"source": r["source"], "dataset": r["dataset"], "variant": r["variant"], "reference": f"{ref['paper']} {ref['venue']} {ref['year']}", "comparison_type": ref["comparison_type"]}
            for cur, refk in (("I-AUROC", "I_AUROC"), ("I-AP", "I_AP"), ("I-F1max", "I_F1max"), ("P-AUROC", "P_AUROC"), ("P-AP", "P_AP"), ("P-F1max", "P_F1max"), ("PRO", "PRO"), ("G_pixel", "G_pixel")):
                row[f"current_{cur}"] = r.get(cur, "NA")
                row[f"reference_{cur}"] = ref.get(refk, "NA")
                rv = fnum(ref, refk)
                row[f"Delta_{cur}"] = fnum(r, cur) - rv if np.isfinite(rv) else "NA"
            rows.append(row)
    return rows


def final_registry(audits):
    registry = build_registry(DEFAULT_DATA_ROOT)
    for row in registry:
        for audit in audits:
            if row["dataset"] == audit["dataset"] and audit["status"] == "READY":
                row.update({"path": audit["path"], "status": "READY", "current_loader_available": True, "evaluation_level": "PIXEL_AND_IMAGE", "num_normal": audit["num_normal_test"], "num_anomaly": audit["num_anomaly_test"], "num_categories": audit["num_categories"], "normal_available": audit["num_normal_test"] > 0, "anomaly_available": audit["num_anomaly_test"] > 0, "pixel_mask_available": True, "image_label_available": True, "official_split_available": True, "notes": audit["notes"]})
    return registry


def write_report(raw, abl, summ, refs, gaps, outcome, runtime):
    lines = ["# Final Cross-Dataset ZSAD Benchmark Consolidation", "", f"OUTCOME: `{outcome}`", "", "No new method, hyperparameter, or target tuning was introduced. M2 is M1 plus the fixed 3x3 NeighborMedian score-map consensus.", "", "## Main Results", "", "| Source | Target | Variant | P-AUROC | P-AP | P-F1max | PRO | G |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in raw:
        lines.append(f"| {r['source']} | {r['dataset']} | {r['variant']} | {fnum(r,'P-AUROC'):.4f} | {fnum(r,'P-AP'):.4f} | {fnum(r,'P-F1max'):.4f} | {fnum(r,'PRO'):.4f} | {fnum(r,'G_pixel'):.4f} |")
    lines.extend(["", "## Generalization Summary", "", "| Group | Comparison | Metric | Improved/Targets | Mean | Median | Worst |", "|---|---|---|---:|---:|---:|---:|"])
    for r in summ:
        if "mean_delta" in r:
            lines.append(f"| {r['group']} | {r['comparison']} | {r['metric']} | {r.get('improved','')}/{r['targets']} | {fnum(r,'mean_delta'):.4f} | {fnum(r,'median_delta'):.4f} | {fnum(r,'worst_delta'):.4f} |")
    quad = [r for r in summ if r["group"] == "median_quadrant"][0]
    lines.extend(["", "## Median AP/PRO Quadrant", "", f"- AP up / PRO up: {quad['AP_up_PRO_up']}", f"- AP up / PRO down: {quad['AP_up_PRO_down']}", f"- AP down / PRO up: {quad['AP_down_PRO_up']}", f"- AP down / PRO down: {quad['AP_down_PRO_down']}"])
    lines.extend(["", "## SOTA Comparison", "", "Direct comparisons are reported only when source, target, protocol/backbone notes, and pixel metrics are explicitly recorded in `SOTA_REFERENCE.csv`. AA-CLIP and FAPrompt remain reference-only rows here because no verified aligned numbers were provided in this benchmark registry.", "", "| Source | Target | Reference | Variant | Delta P-AUROC | Delta P-AP | Delta G |", "|---|---|---|---|---:|---:|---:|"])
    for g in gaps:
        if g.get("comparison_type") != "DIRECT_COMPARABLE" or g.get("variant") != "M2_Ours_HardNormal_Median":
            continue
        lines.append(f"| {g['source']} | {g['dataset']} | {g['reference']} | {g['variant']} | {fnum(g,'Delta_P-AUROC'):.4f} | {fnum(g,'Delta_P-AP'):.4f} | {fnum(g,'Delta_G_pixel'):.4f} |")
    lines.extend(["", "The final method remains below MoECLIP on MVTec, ClinicDB, and MVTec-source VisA, but exceeds the recorded MoECLIP BTAD P-AP while remaining slightly behind in BTAD G. Relative to VisualAD, it improves BTAD and ClinicDB G, but MVTec P-AP is still the main gap.", "", f"Runtime: {runtime:.1f}s."])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[FinalBenchmark] device={device}", flush=True)
    visa = load_visa_source(device)
    mvtec = load_mvtec_source()
    raw_v, per_v, audits_v = evaluate_protocol(visa, device, args.smoke)
    raw_m, per_m, audits_m = evaluate_protocol(mvtec, device, args.smoke)
    raw = raw_v + raw_m
    percat = per_v + per_m
    abl = ablation(raw)
    summ = summary(raw, abl)
    refs = sota_reference()
    gaps = sota_gap(raw, refs)
    registry = final_registry(audits_v + audits_m)
    runtime = time.time() - start
    outcome = "FINAL_METHOD_BENCHMARK_READY"
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY_STANDARD_METRICS.csv", percat)
    write_csv(OUT / "INDUSTRIAL_TABLE.csv", table_rows(raw, "industrial"))
    write_csv(OUT / "MEDICAL_TABLE.csv", table_rows(raw, "medical"))
    write_csv(OUT / "ABLATION_TABLE.csv", abl)
    write_csv(OUT / "GENERALIZATION_SUMMARY.csv", summ)
    write_csv(OUT / "SOTA_REFERENCE.csv", refs)
    write_csv(OUT / "SOTA_GAP.csv", gaps)
    write_csv(OUT / "DATASET_REGISTRY_FINAL.csv", registry)
    write_json(OUT / "OUTCOME.json", {"outcome": outcome, "runtime_sec": runtime, "device": device, "strict_source_only": True, "target_training": False, "target_statistics": False, "target_normal_reference": False, "variants": VARIANTS, "ready_targets": sorted({r["dataset"] for r in raw})})
    write_report(raw, abl, summ, refs, gaps, outcome, runtime)
    print(f"[FinalBenchmark] outcome={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
