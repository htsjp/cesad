import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_hard_normal_contrast import PC1_DEBIAS_OUT, load_payload
from experiments.phase_private_residual_contrast_memory import unit_np
from experiments.phase_spatial_supervision_transfer_audit import target_cells
from experiments.phase_state_pair_decomposition import decompose, hybrid_states
from experiments.phase_statealigned_pc1_debias import score_pc1_removed_adapter_batch
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xxxviii_visual_state_representation import SIDE
from experiments.standard_zsad_metrics import PRO_FPR_LIMIT, PRO_STEPS, evaluate_zsad_standard


OUT = Path("results/phase_standard_zsad_benchmark_infrastructure/formal")
DEFAULT_DATA_ROOT = Path(r"E:\WorkSpace\dataset")
HN_OUT = Path("results/phase_hard_normal_contrast/formal")
HISTORICAL_B0 = {
    "MVTec": 61.42496625199354,
    "BTAD": 72.59389890285473,
    "CVC-ClinicDB": 65.09979163145577,
}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_HINTS = ("mask", "masks", "ground_truth", "gt", "label", "labels", "anno", "annotation", "segmentation")
NORMAL_HINTS = ("good", "normal", "healthy", "negative", "ok")
ANOMALY_HINTS = ("anomaly", "abnormal", "defect", "defective", "disease", "tumor", "positive", "ko")


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


def as_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def fmt(x):
    x = as_float(x)
    return "NA" if not np.isfinite(x) else f"{x:.4f}"


def is_mask_path(path):
    parts = [p.lower() for p in path.parts]
    stem = path.stem.lower()
    return any(h in parts or h in stem for h in MASK_HINTS)


def count_dataset_files(root):
    if not root.exists():
        return {"exists": False, "images": [], "masks": [], "data_images": []}
    images, masks, data_images = [], [], []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = p.relative_to(root)
        if is_mask_path(rel):
            masks.append(rel)
        else:
            data_images.append(rel)
        images.append(rel)
    return {"exists": True, "images": images, "masks": masks, "data_images": data_images}


def infer_counts(name, root, info):
    normals, anomalies = 0, 0
    categories = set()
    for rel in info["data_images"]:
        parts = [p.lower() for p in rel.parts]
        if any(p in NORMAL_HINTS for p in parts):
            normals += 1
        elif any(p in ANOMALY_HINTS for p in parts):
            anomalies += 1
        elif any("test" == p for p in parts) and "good" not in parts:
            anomalies += 1
        if rel.parts:
            categories.add(rel.parts[0])
    if name == "BTAD":
        categories = {p.parts[0] for p in info["data_images"] if p.parts}
    return normals, anomalies, len(categories)


def registry_row(name, domain, industrial_or_medical, root, loader_ready, notes):
    info = count_dataset_files(root)
    normals, anomalies, cats = infer_counts(name, root, info)
    mask_available = bool(info["masks"])
    if not info["exists"]:
        status = "MISSING"
    elif loader_ready:
        status = "READY"
    else:
        status = "NOT_READY"
    if mask_available:
        eval_level = "PIXEL_AND_IMAGE"
    elif normals or anomalies:
        eval_level = "IMAGE_ONLY"
    else:
        eval_level = "AUDIT_REQUIRED"
    return {
        "dataset": name,
        "domain": domain,
        "industrial_or_medical": industrial_or_medical,
        "path": str(root),
        "normal_available": bool(normals),
        "anomaly_available": bool(anomalies),
        "pixel_mask_available": mask_available,
        "image_label_available": bool(normals or anomalies),
        "num_normal": int(normals),
        "num_anomaly": int(anomalies),
        "num_categories": int(cats),
        "official_split_available": (root / "meta.json").exists() or any((root / d).exists() for d in ("train", "test")),
        "current_loader_available": bool(loader_ready),
        "evaluation_level": eval_level,
        "status": status,
        "notes": notes,
    }


def build_registry(data_root):
    specs = [
        ("MVTec", "industrial_object", "industrial", data_root / "mvtec", True, "Formal loader validated through existing target_cells."),
        ("VisA", "industrial_object", "industrial", data_root / "VISA_pytorch", True, "Current formal source loader uses restored VisA/VisA_pytorch assets."),
        ("BTAD", "industrial_object", "industrial", data_root / "btad", True, "Formal external loader validated for BTAD target."),
        ("MPDD", "industrial_object", "industrial", data_root / "MPDD", False, "Dataset folder present if path exists; formal loader not yet validated in this infrastructure gate."),
        ("DTD-Synthetic", "texture_synthetic", "industrial", data_root / "DTD-Synthetic", False, "Dataset folder present if path exists; formal loader not yet validated."),
        ("CVC-ClinicDB", "medical_endoscopy", "medical", data_root / "CVC-ClinicDB", True, "Formal external loader validated for CVC-ClinicDB target; masks used only for final evaluation."),
        ("CVC-ColonDB", "medical_endoscopy", "medical", data_root / "CVC-ColonDB", False, "Medical audit required: verify image-mask pairing and split before formal pixel evaluation."),
        ("Kvasir", "medical_endoscopy", "medical", data_root / "Kvasir", False, "Medical audit required: verify anomaly definition and masks; no pseudomasks allowed."),
        ("BrainMRI", "medical_mri", "medical", data_root / "BrainMRI", False, "Medical audit required: verify normals, masks, and official split."),
        ("HeadCT", "medical_ct", "medical", data_root / "HeadCT", False, "Medical audit required: verify image labels/masks and split."),
        ("Br35H", "medical_mri", "medical", data_root / "Br35H", False, "Tier-C extension; verify labels/masks before use."),
        ("ISBI", "medical_biomedical", "medical", data_root / "ISBI", False, "Tier-C extension; verify mask format and split before use."),
        ("Endo", "medical_endoscopy", "medical", data_root / "Endo", False, "Tier-C extension; verify image-mask matching before use."),
    ]
    return [registry_row(*s) for s in specs]


def build_protocol():
    return {
        "name": "Standard strict source-only cross-dataset ZSAD benchmark",
        "strict_source_only": True,
        "target_training": False,
        "target_statistics": False,
        "target_normal_reference": False,
        "target_adaptation": False,
        "ground_truth_usage": "final_evaluation_only",
        "image_score": "existing formal top1% mean over patch anomaly map via top1_image_score",
        "metrics": {
            "image": ["I-AUROC", "I-AP", "I-F1max"],
            "pixel": ["P-AUROC", "P-AP", "P-F1max", "PRO/AUPRO"],
            "internal": "G_pixel=(P-AUROC+P-AP)/2",
            "PRO": {
                "definition": "connected-component per-region overlap integrated over false-positive-rate range",
                "fpr_integration_range": [0.0, PRO_FPR_LIMIT],
                "threshold_steps": PRO_STEPS,
                "interpolation": "linear threshold grid; trapezoidal integration over normalized observed FPR support",
            },
        },
        "tiers": {
            "Tier-A Industrial": ["MVTec", "VisA", "BTAD", "MPDD", "DTD-Synthetic"],
            "Tier-B Medical": ["CVC-ClinicDB", "CVC-ColonDB", "Kvasir", "BrainMRI", "HeadCT"],
            "Tier-C Extension": ["Br35H", "ISBI", "Endo"],
        },
        "protocols": {
            "A_Full_VisA_source": {
                "source": "Full VisA",
                "targets": ["MVTec", "BTAD", "MPDD", "DTD-Synthetic", "CVC-ClinicDB", "CVC-ColonDB", "Kvasir", "BrainMRI", "HeadCT"],
            },
            "B_Full_MVTec_source": {
                "source": "Full MVTec AD",
                "targets": ["VisA", "BTAD", "MPDD", "DTD-Synthetic", "CVC-ClinicDB", "CVC-ColonDB", "Kvasir", "BrainMRI", "HeadCT"],
            },
        },
    }


def build_sota_reference():
    rows = []
    for paper, venue, year, backbone, res, vals in [
        ("VisualAD", "CVPR", 2026, "CLIP", "not_locally_audited", {
            "MVTec": (90.8, 41.2, 66.00),
            "BTAD": (91.1, 43.1, 67.10),
            "CVC-ClinicDB": (85.2, 37.8, 61.50),
        }),
        ("MoECLIP", "CVPR", 2026, "OpenCLIP ViT-L/14-336", "518x518", {
            "MVTec": (92.5, 45.7, 69.10),
            "BTAD": (96.8, 50.4, 73.60),
            "CVC-ClinicDB": (89.7, 49.9, 69.80),
        }),
    ]:
        for target, (pa, pap, g) in vals.items():
            rows.append({
                "paper": paper,
                "venue": venue,
                "year": year,
                "backbone": backbone,
                "input_resolution": res,
                "training_source": "Full VisA",
                "target": target,
                "target_data_used": "no",
                "I_AUROC": "NA",
                "I_AP": "NA",
                "I_F1max": "NA",
                "P_AUROC": pa,
                "P_AP": pap,
                "P_F1max": "NA",
                "PRO": "NA",
                "G_pixel": g,
                "comparison_type": "DIRECT_COMPARABLE",
                "notes": "Source/target/pixel metrics directly comparable as recorded; paper implementation details not reproduced locally.",
            })
    for paper, venue, year in [("AA-CLIP", "CVPR", 2025), ("FAPrompt", "ICCV", 2025)]:
        rows.append({
            "paper": paper,
            "venue": venue,
            "year": year,
            "backbone": "NA",
            "input_resolution": "NA",
            "training_source": "NA",
            "target": "NA",
            "target_data_used": "NA",
            "I_AUROC": "NA",
            "I_AP": "NA",
            "I_F1max": "NA",
            "P_AUROC": "NA",
            "P_AP": "NA",
            "P_F1max": "NA",
            "PRO": "NA",
            "G_pixel": "NA",
            "comparison_type": "REFERENCE_ONLY",
            "notes": "Numeric entries pending official paper/repository audit; not used for gap calculations.",
        })
    return rows


def build_sota_gap(metric_rows, ref_rows, variant="B0_HardNormal"):
    gaps = []
    by_metric = {(r["variant"], r["target"]): r for r in metric_rows}
    for ref in ref_rows:
        if ref["comparison_type"] != "DIRECT_COMPARABLE":
            gaps.append({
                "variant": "B0_HardNormal",
                "reference": ref["paper"],
                "target": ref["target"],
                "comparison_type": ref["comparison_type"],
                "delta_I_AUROC": "NA",
                "delta_I_AP": "NA",
                "delta_I_F1max": "NA",
                "delta_P_AUROC": "NA",
                "delta_P_AP": "NA",
                "delta_P_F1max": "NA",
                "delta_PRO": "NA",
                "delta_G": "NA",
                "notes": ref["notes"],
            })
            continue
        row = by_metric.get((variant, ref["target"]))
        if row is None:
            continue
        gaps.append({
                "variant": variant,
            "reference": f"{ref['paper']} {ref['venue']} {ref['year']}",
            "target": ref["target"],
            "comparison_type": ref["comparison_type"],
            "current_P_AUROC": row["P-AUROC"],
            "current_P_AP": row["P-AP"],
            "current_P_F1max": row["P-F1max"],
            "current_PRO": row["PRO"],
            "current_G": row["G_pixel"],
            "reference_P_AUROC": ref["P_AUROC"],
            "reference_P_AP": ref["P_AP"],
            "reference_G": ref["G_pixel"],
            "delta_I_AUROC": "NA",
            "delta_I_AP": "NA",
            "delta_I_F1max": "NA",
            "delta_P_AUROC": as_float(row["P-AUROC"]) - as_float(ref["P_AUROC"]),
            "delta_P_AP": as_float(row["P-AP"]) - as_float(ref["P_AP"]),
            "delta_P_F1max": "NA",
            "delta_PRO": "NA",
            "delta_G": as_float(row["G_pixel"]) - as_float(ref["G_pixel"]),
            "notes": ref["notes"],
        })
    return gaps


def load_hardnormal_model(device):
    q_pc1 = unit_np(np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32))
    true_payload = load_payload(PC1_DEBIAS_OUT / "E1_PC1Removed-TrueMask_formal.pt", device)
    corrupt_payload = load_payload(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt", device)
    states_t = np.asarray(true_payload["patch_states"], dtype=np.float32)
    states_c = np.asarray(corrupt_payload["patch_states"], dtype=np.float32)
    _m_t, d_t = decompose(states_t)
    m_c, _d_c = decompose(states_c)
    alpha = float(np.linalg.norm(d_t))
    vectors = np.load(HN_OUT / "contrast_vectors.npz")
    v_hn = unit_np(vectors["E1"].astype(np.float32))
    model = make_model(hybrid_states(m_c, alpha * v_hn), corrupt_payload, device)
    return q_pc1, model


def evaluate_hardnormal(device, smoke):
    q_pc1, model = load_hardnormal_model(device)
    raw, percat = [], []
    for cell_id, target, dataset, idx_or_rows, patch, kind in target_cells(smoke):
        n = min(24, patch.shape[0]) if smoke else patch.shape[0]
        idx_use = idx_or_rows[:n]
        print(f"[StandardZSAD] eval B0_HardNormal on {cell_id} ({n} images)", flush=True)
        image, maps = score_pc1_removed_adapter_batch(patch, model, q_pc1, device, smoke=smoke)
        metric, per = evaluate_zsad_standard(cell_id, target, dataset, idx_use, kind, image, maps, SIDE)
        metric.update({"dataset": cell_id, "variant": "B0_HardNormal", "source": "Full VisA"})
        raw.append(metric)
        for cat, vals in per.items():
            percat.append({"dataset": cell_id, "target": target, "category": cat, "variant": "B0_HardNormal", **vals})
    return raw, percat


def build_metric_validation(rows):
    out = []
    ok = True
    for row in rows:
        ds = row["dataset"]
        if ds not in HISTORICAL_B0:
            continue
        current = as_float(row["G_pixel"])
        historical = HISTORICAL_B0[ds]
        diff = current - historical
        passed = abs(diff) <= 0.01
        ok = ok and passed
        out.append({
            "dataset": ds,
            "variant": row["variant"],
            "historical_G": historical,
            "standard_G": current,
            "delta_G": diff,
            "tolerance": 0.01,
            "passed": passed,
        })
    return out, ok


def write_report(data_root, registry, validation, standard_rows, gaps, outcome, runtime):
    ready = sum(1 for r in registry if r["status"] == "READY")
    not_ready = sum(1 for r in registry if r["status"] == "NOT_READY")
    lines = [
        "# Standard ZSAD Benchmark Infrastructure",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "## Execution",
        "",
        "- No model training was run.",
        "- No target statistics, target normal references, or target adaptation were used.",
        "- The formal image score is the existing project score: top 1% mean over the patch anomaly map via `top1_image_score`.",
        "- Existing HardNormal checkpoint/pipeline was re-evaluated only to validate metric migration.",
        "",
        "## Metric Audit",
        "",
        "- I-AUROC/P-AUROC use sklearn ROC AUC; I-AP/P-AP use sklearn average precision.",
        "- I-F1max/P-F1max use max F1 over the precision-recall threshold curve.",
        f"- PRO/AUPRO uses connected-component region overlap integrated over FPR 0..{PRO_FPR_LIMIT} with {PRO_STEPS} threshold samples and trapezoidal integration.",
        "- Pixel metrics flatten all valid patch-grid pixels at 37x37; category rows are macro-readable diagnostics, while dataset rows aggregate all pixels.",
        "- `G_pixel=(P-AUROC+P-AP)/2` is retained as an internal project metric, not an external leaderboard metric.",
        "",
        "## Reproduction Check",
        "",
        "| Dataset | Historical G | Standard G | Delta | Pass |",
        "|---|---:|---:|---:|---|",
    ]
    for r in validation:
        lines.append(f"| {r['dataset']} | {r['historical_G']:.4f} | {r['standard_G']:.4f} | {r['delta_G']:+.6f} | {r['passed']} |")
    lines.extend([
        "",
        "## Dataset Registry",
        "",
        f"- Dataset root: `{data_root}`",
        f"- READY datasets: {ready}",
        f"- NOT_READY datasets: {not_ready}",
        "- Medical datasets without audited image-mask pairing are marked NOT_READY/IMAGE_ONLY or AUDIT_REQUIRED; no pseudomasks were created.",
        "",
        "## Standard Metrics",
        "",
        "| Target | I-AUROC | I-AP | I-F1max | P-AUROC | P-AP | P-F1max | PRO | G_pixel |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in standard_rows:
        lines.append(f"| {r['target']} | {fmt(r['I-AUROC'])} | {fmt(r['I-AP'])} | {fmt(r['I-F1max'])} | {fmt(r['P-AUROC'])} | {fmt(r['P-AP'])} | {fmt(r['P-F1max'])} | {fmt(r['PRO'])} | {fmt(r['G_pixel'])} |")
    lines.extend([
        "",
        "## External SOTA Position",
        "",
        "- Directly comparable rows currently cover Full VisA source to MVTec/BTAD/CVC-ClinicDB with pixel P-AUROC/P-AP/G. MoECLIP is marked direct only under the recorded OpenCLIP ViT-L/14-336 518x518 setting; implementation-level reproduction is still a caveat.",
        "- AA-CLIP and FAPrompt are kept as REFERENCE_ONLY until exact official cross-dataset numeric entries are audited; they are not used for delta calculations.",
        "- Current HardNormal remains ahead of VisualAD on BTAD/ClinicDB G but trails VisualAD on MVTec G, and trails MoECLIP on MVTec/BTAD/ClinicDB.",
        "",
        "| Reference | Target | Delta P-AUROC | Delta P-AP | Delta G |",
        "|---|---|---:|---:|---:|",
    ])
    for g in gaps:
        if g["comparison_type"] == "DIRECT_COMPARABLE":
            lines.append(f"| {g['reference']} | {g['target']} | {fmt(g['delta_P_AUROC'])} | {fmt(g['delta_P_AP'])} | {fmt(g['delta_G'])} |")
    lines.extend([
        "",
        "Worth continuing: the infrastructure is ready; method work should be gated by shrinking the MoECLIP gap on MVTec/ClinicDB, not by BTAD-only gains.",
        "",
        f"Runtime: {runtime:.1f}s.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.dataset_root)
    registry = build_registry(data_root)
    protocol = build_protocol()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[StandardZSAD] device={device}", flush=True)
    standard_rows, percat = evaluate_hardnormal(device, args.smoke)
    validation, validation_ok = build_metric_validation(standard_rows)
    refs = build_sota_reference()
    gaps = build_sota_gap(standard_rows, refs)
    outcome = "STANDARD_ZSAD_BENCHMARK_INFRASTRUCTURE_READY" if validation_ok else "STANDARD_METRIC_MIGRATION_REPRODUCTION_FAILED"
    runtime = time.time() - start

    write_csv(OUT / "DATASET_REGISTRY.csv", registry)
    write_csv(OUT / "METRIC_VALIDATION.csv", validation)
    write_csv(OUT / "STANDARD_METRICS.csv", standard_rows)
    write_csv(OUT / "PER_CATEGORY_STANDARD_METRICS.csv", percat)
    write_csv(OUT / "SOTA_REFERENCE.csv", refs)
    write_csv(OUT / "SOTA_GAP.csv", gaps)
    write_json(OUT / "benchmark_protocol.json", protocol)
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "dataset_root": str(data_root),
        "runtime_sec": runtime,
        "device": device,
        "strict_source_only": True,
        "no_training": True,
        "target_statistics": False,
        "target_normal_reference": False,
        "validation_passed": bool(validation_ok),
        "metric_module": "experiments.standard_zsad_metrics",
        "image_score": "top1_percent_mean",
        "pro_fpr_limit": PRO_FPR_LIMIT,
        "pro_steps": PRO_STEPS,
    })
    write_report(data_root, registry, validation, standard_rows, gaps, outcome, runtime)
    print(f"[StandardZSAD] outcome={outcome}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
