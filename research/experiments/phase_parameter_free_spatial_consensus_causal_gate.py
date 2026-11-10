import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_spatial_residual_refinement_standard_metric import (
    OUT as SRR_OUT,
    aligned_grid,
    fixed_smooth,
    fnum,
    load_base,
    logits_from_z,
)
from experiments.phase_spatial_supervision_transfer_audit import target_cells
from experiments.phase_standard_zsad_benchmark_infrastructure import build_sota_reference, build_sota_gap
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import top1_image_score
from experiments.phase_xxxviii_visual_state_representation import SIDE
from experiments.standard_zsad_metrics import evaluate_zsad_standard, resize_mask_flat


OUT = Path("results/phase_parameter_free_spatial_consensus_causal_gate/formal")
VARIANTS = (
    "B0_OriginalScore",
    "E1_FeatureSpatialConsensus",
    "E2_ScoreSpatialConsensus",
    "E3_ProbabilitySpatialConsensus",
    "E4_NeighborMedianControl",
    "E5_SpatiallyShuffledControl",
)


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


def safe_corr(a, b, kind="pearson"):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    if kind == "spearman":
        return float(spearmanr(a, b).correlation)
    return float(pearsonr(a, b).statistic)


def smooth2d_np(m):
    x = torch.tensor(np.asarray(m, dtype=np.float32)).view(1, 1, SIDE, SIDE)
    y = 0.5 * x + 0.5 * F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    return y.view(SIDE, SIDE).numpy().astype(np.float32)


def median2d_np(m):
    return ndimage.median_filter(np.asarray(m, dtype=np.float32), size=3, mode="nearest").astype(np.float32)


@torch.no_grad()
def score_all_maps(patch, q_pc1, align_model, m_np, alpha, v_np, device, smoke=False, batch_images=4):
    n = min(24, patch.shape[0]) if smoke else patch.shape[0]
    maps = {v: np.zeros((n, SIDE, SIDE), dtype=np.float16) for v in VARIANTS}
    image = {v: np.zeros((n,), dtype=np.float32) for v in VARIANTS}
    m = torch.tensor(m_np, dtype=torch.float32, device=device)
    v = torch.tensor(v_np, dtype=torch.float32, device=device)
    rng = np.random.default_rng(0)
    for start in range(0, n, batch_images):
        zs = [aligned_grid(patch[i], q_pc1, align_model, device) for i in range(start, min(start + batch_images, n))]
        z = torch.stack(zs, dim=0)
        logits = logits_from_z(z, m, alpha, v).detach().cpu().numpy().astype(np.float32)
        prob = 1.0 / (1.0 + np.exp(-logits))
        zf = fixed_smooth(z)
        logits_f = logits_from_z(zf, m, alpha, v).detach().cpu().numpy().astype(np.float32)
        prob_f = 1.0 / (1.0 + np.exp(-logits_f))
        b = logits.shape[0]
        for bi in range(b):
            li = logits[bi]
            pi = prob[bi]
            variants = {
                "B0_OriginalScore": pi,
                "E1_FeatureSpatialConsensus": prob_f[bi],
                "E2_ScoreSpatialConsensus": smooth2d_np(li),
                "E3_ProbabilitySpatialConsensus": smooth2d_np(pi),
                "E4_NeighborMedianControl": median2d_np(li),
            }
            perm = rng.permutation(SIDE * SIDE)
            shuffled = li.reshape(-1)[perm].reshape(SIDE, SIDE)
            variants["E5_SpatiallyShuffledControl"] = smooth2d_np(shuffled)
            for name, smap in variants.items():
                idx = start + bi
                maps[name][idx] = smap.astype(np.float16)
                image[name][idx] = top1_image_score(smap)
    return image, maps


def evaluate_all(q_pc1, align_model, m_np, alpha, v_np, device, smoke):
    raw, percat = [], []
    diag_rows, comp_rows, map_equiv = [], [], []
    for cell_id, target, dataset, idx_or_rows, patch, kind in target_cells(smoke):
        n = min(24, patch.shape[0]) if smoke else patch.shape[0]
        idx_use = idx_or_rows[:n]
        print(f"[SpatialConsensus] score {cell_id} ({n} images)", flush=True)
        image, maps = score_all_maps(patch, q_pc1, align_model, m_np, alpha, v_np, device, smoke=smoke)
        for variant in VARIANTS:
            metric, per = evaluate_zsad_standard(cell_id, target, dataset, idx_use, kind, image[variant], maps[variant], SIDE)
            metric.update({"dataset": cell_id, "variant": variant, "source": "Full VisA"})
            raw.append(metric)
            for cat, vals in per.items():
                percat.append({"dataset": cell_id, "target": target, "category": cat, "variant": variant, **vals})
        e1 = np.asarray(maps["E1_FeatureSpatialConsensus"], dtype=np.float32).reshape(-1)
        e2 = np.asarray(maps["E2_ScoreSpatialConsensus"], dtype=np.float32).reshape(-1)
        map_equiv.append({
            "dataset": cell_id,
            "comparison": "E1_final_score_vs_E2_final_score",
            "pearson": safe_corr(e1, e2, "pearson"),
            "spearman": safe_corr(e1, e2, "spearman"),
            "mean_abs_difference": float(np.mean(np.abs(e1 - e2))),
        })
        diag_rows.extend(localization_diagnostics(cell_id, target, dataset, idx_use, kind, maps))
        comp_rows.extend(component_diagnostics(cell_id, target, maps))
    return raw, percat, diag_rows, comp_rows, map_equiv


def localization_diagnostics(cell_id, target, dataset, idx_or_rows, kind, maps):
    rows = []
    for variant in VARIANTS:
        buckets = defaultdict(list)
        for pos, ref in enumerate(idx_or_rows):
            item = dataset[int(ref)] if kind == "source_dataset" else dataset[pos]
            if "img_mask" not in item:
                continue
            mask = resize_mask_flat(item, SIDE).reshape(SIDE, SIDE).astype(bool)
            score = np.asarray(maps[variant][pos], dtype=np.float32).reshape(SIDE, SIDE)
            if not np.any(mask):
                buckets["far_normal"].extend(score.reshape(-1).tolist())
                continue
            interior = ndimage.binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
            boundary = mask & ~interior
            near = ndimage.binary_dilation(mask, structure=np.ones((3, 3)), iterations=1) & ~mask
            far = ~(mask | near)
            for name, sel in (
                ("anomaly_interior", interior),
                ("anomaly_boundary", boundary),
                ("normal_near_boundary", near),
                ("far_normal", far),
            ):
                if np.any(sel):
                    buckets[name].extend(score[sel].tolist())
        for name, vals in sorted(buckets.items()):
            arr = np.asarray(vals, dtype=np.float32)
            rows.append({
                "dataset": cell_id,
                "target": target,
                "variant": variant,
                "region": name,
                "mean_score": float(arr.mean()) if arr.size else float("nan"),
                "p90_score": float(np.percentile(arr, 90)) if arr.size else float("nan"),
                "count": int(arr.size),
            })
    return rows


def component_diagnostics(cell_id, target, maps):
    rows = []
    structure = np.ones((3, 3), dtype=np.int8)
    for variant in VARIANTS:
        counts, sizes = [], []
        for smap in maps[variant]:
            s = np.asarray(smap, dtype=np.float32).reshape(SIDE, SIDE)
            flat = s.reshape(-1)
            k = max(1, int(flat.size * 0.01))
            th = np.partition(flat, flat.size - k)[flat.size - k]
            top = s >= th
            labels, n = ndimage.label(top, structure=structure)
            areas = np.bincount(labels.reshape(-1), minlength=n + 1)[1:]
            counts.append(n)
            sizes.append(float(np.mean(areas)) if len(areas) else 0.0)
        rows.append({
            "dataset": cell_id,
            "target": target,
            "variant": variant,
            "top1_component_count_mean": float(np.mean(counts)),
            "top1_component_count_p90": float(np.percentile(counts, 90)),
            "top1_component_size_mean": float(np.mean(sizes)),
        })
    return rows


def feature_score_equivalence(raw, maps_cache=None):
    rows = []
    by = {(r["dataset"], r["variant"]): r for r in raw}
    for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
        e1 = by[(ds, "E1_FeatureSpatialConsensus")]
        e2 = by[(ds, "E2_ScoreSpatialConsensus")]
        rows.append({
            "dataset": ds,
            "comparison": "E1_feature_vs_E2_logit_score",
            "delta_P-AUROC": fnum(e1, "P-AUROC") - fnum(e2, "P-AUROC"),
            "delta_P-AP": fnum(e1, "P-AP") - fnum(e2, "P-AP"),
            "delta_P-F1max": fnum(e1, "P-F1max") - fnum(e2, "P-F1max"),
            "delta_PRO": fnum(e1, "PRO") - fnum(e2, "PRO"),
            "delta_G": fnum(e1, "G_pixel") - fnum(e2, "G_pixel"),
        })
    return rows


def map_equivalence_rows(map_bank):
    rows = []
    for ds, maps in map_bank.items():
        a = np.asarray(maps["E1_FeatureSpatialConsensus"], dtype=np.float32).reshape(-1)
        b = np.asarray(maps["E2_ScoreSpatialConsensus"], dtype=np.float32).reshape(-1)
        rows.append({
            "dataset": ds,
            "comparison": "E1_final_score_vs_E2_final_score",
            "pearson": safe_corr(a, b, "pearson"),
            "spearman": safe_corr(a, b, "spearman"),
            "mean_abs_difference": float(np.mean(np.abs(a - b))),
        })
    return rows


def decide(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    targets = ("MVTec", "BTAD", "CVC-ClinicDB")
    gains = {
        v: np.asarray([fnum(by[(d, v)], "G_pixel") - fnum(by[(d, "B0_OriginalScore")], "G_pixel") for d in targets], dtype=np.float32)
        for v in VARIANTS[1:]
    }
    pap = {
        v: {d: fnum(by[(d, v)], "P-AP") - fnum(by[(d, "B0_OriginalScore")], "P-AP") for d in targets}
        for v in VARIANTS[1:]
    }
    e1_e2_g = gains["E1_FeatureSpatialConsensus"].mean() - gains["E2_ScoreSpatialConsensus"].mean()
    e1_e2_pap = np.mean([pap["E1_FeatureSpatialConsensus"][d] - pap["E2_ScoreSpatialConsensus"][d] for d in targets])
    e4_gain = gains["E4_NeighborMedianControl"]
    e5_gain = gains["E5_SpatiallyShuffledControl"]
    e1_gain = gains["E1_FeatureSpatialConsensus"]
    e2_gain = gains["E2_ScoreSpatialConsensus"]
    best_consensus = "E1_FeatureSpatialConsensus" if e1_gain.mean() >= e2_gain.mean() else "E2_ScoreSpatialConsensus"
    best_gain = gains[best_consensus]
    best_pap_mc = np.mean([pap[best_consensus]["MVTec"], pap[best_consensus]["CVC-ClinicDB"]])
    pf1_or_pro_ok = 0
    for d in targets:
        b0 = by[(d, "B0_OriginalScore")]
        r = by[(d, best_consensus)]
        if fnum(r, "P-F1max") - fnum(b0, "P-F1max") >= 0 or fnum(r, "PRO") - fnum(b0, "PRO") >= 0:
            pf1_or_pro_ok += 1
    retention = int(np.sum(best_gain > 0)) == 3 and float(best_gain.mean()) >= 1.0 and best_pap_mc >= 1.5 and pf1_or_pro_ok >= 2
    stats = {
        "E1_mean_G_gain": float(e1_gain.mean()),
        "E2_mean_G_gain": float(e2_gain.mean()),
        "E3_mean_G_gain": float(gains["E3_ProbabilitySpatialConsensus"].mean()),
        "E4_mean_G_gain": float(e4_gain.mean()),
        "E5_mean_G_gain": float(e5_gain.mean()),
        "E1_minus_E2_mean_G": float(e1_e2_g),
        "E1_minus_E2_mean_P_AP": float(e1_e2_pap),
        "best_consensus": best_consensus,
        "best_MVTec_ClinicDB_mean_P_AP_gain": float(best_pap_mc),
        "performance_retention_gate": bool(retention),
    }
    if abs(e4_gain.mean() - max(e1_gain.mean(), e2_gain.mean())) < 0.7:
        return "GENERIC_LOCAL_DENOISING_SUFFICIENT", stats
    if abs(e1_e2_g) < 0.5 and abs(e1_e2_pap) < 1.0 and int(np.sum(e1_gain > 0)) == 3 and int(np.sum(e2_gain > 0)) == 3:
        return "SCORE_LEVEL_LOCAL_CONSENSUS_EXPLAINS_GAIN", stats
    if e1_e2_g >= 0.5 or e1_e2_pap >= 1.0:
        return "FEATURE_GEOMETRY_CONSENSUS_HAS_EXTRA_SIGNAL", stats
    if int(np.sum(e5_gain > 0)) >= 2:
        return "SPATIAL_NEIGHBORHOOD_NOT_SPECIFIC", stats
    return "SPATIAL_CONSENSUS_INCONCLUSIVE", stats


def write_report(raw, equiv, outcome, stats, runtime):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    lines = [
        "# Parameter-Free Spatial Consensus - Causal Gate",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        f"Performance retention gate: `{stats['performance_retention_gate']}`",
        "",
        "| Variant | Target | P-AUROC | P-AP | P-F1max | PRO | G |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
            r = by[(ds, variant)]
            lines.append(f"| {variant} | {ds} | {fnum(r,'P-AUROC'):.4f} | {fnum(r,'P-AP'):.4f} | {fnum(r,'P-F1max'):.4f} | {fnum(r,'PRO'):.4f} | {fnum(r,'G_pixel'):.4f} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"- E1 mean G gain: {stats['E1_mean_G_gain']:+.4f}",
        f"- E2 mean G gain: {stats['E2_mean_G_gain']:+.4f}",
        f"- E4 mean G gain: {stats['E4_mean_G_gain']:+.4f}",
        f"- E5 mean G gain: {stats['E5_mean_G_gain']:+.4f}",
        f"- E1-E2 mean G / P-AP: {stats['E1_minus_E2_mean_G']:+.4f} / {stats['E1_minus_E2_mean_P_AP']:+.4f}",
        "",
        "## Feature-Score Equivalence",
        "",
        "| Dataset | Delta G E1-E2 | Delta P-AP E1-E2 |",
        "|---|---:|---:|",
    ])
    for r in equiv:
        if r["comparison"] == "E1_feature_vs_E2_logit_score":
            lines.append(f"| {r['dataset']} | {r['delta_G']:+.4f} | {r['delta_P-AP']:+.4f} |")
    lines.extend(["", f"Runtime: {runtime:.1f}s."])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SpatialConsensus] device={device}", flush=True)
    q_pc1, align_model, m_np, alpha, v_np = load_base(device)
    raw, percat, loc_diag, comp_diag, map_equiv = evaluate_all(q_pc1, align_model, m_np, alpha, v_np, device, args.smoke)
    equiv = feature_score_equivalence(raw) + map_equiv
    outcome, stats = decide(raw)
    refs = build_sota_reference()
    sota_gap = []
    for variant in ("B0_OriginalScore", "E1_FeatureSpatialConsensus", "E2_ScoreSpatialConsensus", "E3_ProbabilitySpatialConsensus", "E4_NeighborMedianControl", "E5_SpatiallyShuffledControl"):
        rows = [r for r in raw if r.get("variant") == variant]
        sota_gap.extend(build_sota_gap(rows, refs, variant=variant))
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY_STANDARD_METRICS.csv", percat)
    write_csv(OUT / "FEATURE_SCORE_EQUIVALENCE.csv", equiv)
    write_csv(OUT / "LOCALIZATION_DIAGNOSTICS.csv", loc_diag)
    write_csv(OUT / "COMPONENT_DIAGNOSTICS.csv", comp_diag)
    write_csv(OUT / "SOTA_GAP.csv", sota_gap)
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": stats,
        "runtime_sec": runtime,
        "device": device,
        "strict_source_only": True,
        "target_statistics": False,
        "target_training": False,
        "no_training": True,
    })
    write_report(raw, equiv, outcome, stats, runtime)
    print(f"[SpatialConsensus] outcome={outcome}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
