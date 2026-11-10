import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_hard_normal_cross_source_mvtec import OUT as MVTEC_HN_OUT
from experiments.phase_hard_normal_cross_source_mvtec import target_cells as mvtec_source_target_cells
from experiments.phase_median_spatial_consensus_generalization import (
    HOLDOUTS,
    MetaDataset,
    audit_meta_dataset,
    ensure_patch_cache,
)
from experiments.phase_private_residual_contrast_memory import contrast_logits, unit_np
from experiments.phase_spatial_residual_refinement_standard_metric import aligned_grid, load_base, logits_from_z
from experiments.phase_spatial_supervision_transfer_audit import target_cells as visa_target_cells
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import top1_image_score
from experiments.phase_xxxviii_visual_state_representation import SIDE, norm_np
from experiments.standard_zsad_metrics import evaluate_zsad_standard, resize_mask_flat


OUT = Path("results/phase_feature_guided_weighted_median/formal")
VARIANTS = ("B0_OriginalScore", "M0_PlainMedian", "E1_FeatureGuidedWeightedMedian", "E2_FeatureGuidedWeightedMean", "E3_RandomFeatureWeightedMedian")
M0_PRO_DOWN_DATASETS = {("Full MVTec", "VisA"), ("Full MVTec", "BTAD"), ("Full VisA", "MPDD"), ("Full VisA", "DTD-Synthetic")}


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


def fnum(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def median3x3(x):
    return ndimage.median_filter(np.asarray(x, dtype=np.float32), size=3, mode="nearest").astype(np.float32)


def weighted_maps(score, feat, rng=None):
    score = np.asarray(score, dtype=np.float32).reshape(SIDE, SIDE)
    feat = np.asarray(feat, dtype=np.float32).reshape(SIDE, SIDE, -1)
    if rng is not None:
        flat = feat.reshape(-1, feat.shape[-1]).copy()
        feat = flat[rng.permutation(flat.shape[0])].reshape(SIDE, SIDE, -1)
    pad_s = np.pad(score, ((1, 1), (1, 1)), mode="edge")
    pad_f = np.pad(feat, ((1, 1), (1, 1), (0, 0)), mode="edge")
    neigh_s, neigh_f = [], []
    for dy in range(3):
        for dx in range(3):
            neigh_s.append(pad_s[dy:dy + SIDE, dx:dx + SIDE])
            neigh_f.append(pad_f[dy:dy + SIDE, dx:dx + SIDE])
    neigh_s = np.stack(neigh_s, axis=-1).reshape(-1, 9)
    neigh_f = np.stack(neigh_f, axis=-2).reshape(-1, 9, feat.shape[-1])
    center = feat.reshape(-1, feat.shape[-1])
    w = np.maximum(0.0, np.einsum("nkd,nd->nk", neigh_f, center, optimize=True))
    w[:, 4] = 1.0
    total = np.maximum(w.sum(axis=1), 1e-12)
    mean = ((w * neigh_s).sum(axis=1) / total).reshape(SIDE, SIDE).astype(np.float32)
    order = np.argsort(neigh_s, axis=1)
    sorted_scores = np.take_along_axis(neigh_s, order, axis=1)
    sorted_w = np.take_along_axis(w, order, axis=1)
    choose_rank = (np.cumsum(sorted_w, axis=1) >= (0.5 * total)[:, None]).argmax(axis=1)
    med = sorted_scores[np.arange(sorted_scores.shape[0]), choose_rank].reshape(SIDE, SIDE).astype(np.float32)
    pick = order[np.arange(order.shape[0]), choose_rank]
    center_count = int(np.sum(pick == 4))
    four_count = int(np.sum(np.isin(pick, [1, 3, 5, 7])))
    diag_count = int(pick.size - center_count - four_count)
    n = SIDE * SIDE
    stats = {
        "mean_neighbor_cosine": float(np.mean(w)),
        "std_neighbor_cosine": float(np.mean(np.std(w, axis=1))),
        "selected_center_ratio": center_count / n,
        "selected_4neighbor_ratio": four_count / n,
        "selected_diagonal_ratio": diag_count / n,
    }
    return med, mean, stats


def add_eval(raw, percat, cell_id, target, source, group, dataset, idx_or_rows, kind, image, maps):
    for variant in VARIANTS:
        metric, per = evaluate_zsad_standard(cell_id, target, dataset, idx_or_rows, kind, image[variant], maps[variant], SIDE)
        metric.update({"dataset": target, "source": source, "variant": variant, "evaluation_group": group})
        raw.append(metric)
        for cat, vals in per.items():
            percat.append({"dataset": target, "source": source, "category": cat, "variant": variant, **vals})


def boundary_diag(source, target, dataset, idx_or_rows, kind, maps):
    rows = []
    for variant in ("M0_PlainMedian", "E1_FeatureGuidedWeightedMedian"):
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
            for name, sel in (("anomaly_interior", interior), ("anomaly_boundary", boundary), ("normal_near_boundary", near), ("far_normal", far)):
                if np.any(sel):
                    buckets[name].extend(score[sel].tolist())
        for region, vals in sorted(buckets.items()):
            arr = np.asarray(vals, dtype=np.float32)
            rows.append({"source": source, "dataset": target, "variant": variant, "region": region, "mean_score": float(arr.mean()), "p90_score": float(np.percentile(arr, 90)), "count": int(arr.size)})
    return rows


@torch.no_grad()
def score_pack(source, target, patch, q_pc1, align_model, m_np, alpha, v_np, device, use_align, smoke):
    n = min(24, patch.shape[0]) if smoke else patch.shape[0]
    maps = {v: np.zeros((n, SIDE, SIDE), dtype=np.float16) for v in VARIANTS}
    image = {v: np.zeros((n,), dtype=np.float32) for v in VARIANTS}
    stats_rows = []
    m_t = torch.tensor(m_np, dtype=torch.float32, device=device)
    v_t = torch.tensor(v_np, dtype=torch.float32, device=device)
    rng = np.random.default_rng(0)
    for i in range(n):
        if use_align:
            z = aligned_grid(patch[i], q_pc1, align_model, device)
            logits = logits_from_z(z.unsqueeze(0), m_t, alpha, v_t).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)
            feat = z.detach().cpu().numpy().reshape(SIDE, SIDE, 1024)
        else:
            feat = norm_np(np.asarray(patch[i], dtype=np.float32).reshape(SIDE * SIDE, 1024)).reshape(SIDE, SIDE, 1024)
            logits = contrast_logits(torch.tensor(feat.reshape(-1, 1024), dtype=torch.float32, device=device), m_t, alpha, v_t).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)
        plain = median3x3(logits)
        wgmed, wgmean, st = weighted_maps(logits, feat)
        rmed, _rmean, rst = weighted_maps(logits, feat, rng=rng)
        variants = {
            "B0_OriginalScore": logits,
            "M0_PlainMedian": plain,
            "E1_FeatureGuidedWeightedMedian": wgmed,
            "E2_FeatureGuidedWeightedMean": wgmean,
            "E3_RandomFeatureWeightedMedian": rmed,
        }
        for name, smap in variants.items():
            maps[name][i] = smap.astype(np.float16)
            image[name][i] = top1_image_score(smap)
        st.update({"source": source, "dataset": target, "variant": "E1_FeatureGuidedWeightedMedian", "image_index": i})
        rst.update({"source": source, "dataset": target, "variant": "E3_RandomFeatureWeightedMedian", "image_index": i})
        stats_rows.extend([st, rst])
    return image, maps, stats_rows


def run_cross_source(device, smoke):
    payload = np.load(MVTEC_HN_OUT / "contrast_vectors.npz")
    m_np = payload["midpoint"].astype(np.float32)
    alpha = float(payload["alpha"])
    v_np = unit_np(payload["E1"].astype(np.float32))
    raw, percat, bdiag, fstats = [], [], [], []
    for cell_id, target, dataset, idx_or_rows, patch, kind in mvtec_source_target_cells(smoke):
        n = min(24, patch.shape[0]) if smoke else patch.shape[0]
        idx = idx_or_rows[:n]
        print(f"[FGWM] Full MVTec -> {target} n={n}", flush=True)
        image, maps, stats = score_pack("Full MVTec", target, patch, None, None, m_np, alpha, v_np, device, False, smoke)
        add_eval(raw, percat, cell_id, target, "Full MVTec", "cross_source_replication", dataset, idx, kind, image, maps)
        bdiag.extend(boundary_diag("Full MVTec", target, dataset, idx, kind, maps))
        fstats.extend(stats)
    return raw, percat, bdiag, fstats


def run_visa_source(device, smoke):
    q_pc1, align_model, m_np, alpha, v_np = load_base(device)
    raw, percat, bdiag, fstats = [], [], [], []
    cells = []
    for cell_id, target, dataset, idx_or_rows, patch, kind in visa_target_cells(smoke):
        cells.append((cell_id, target, dataset, idx_or_rows, patch, kind, "discovery_replication"))
    audits = [audit_meta_dataset(name, root) for name, (root, _d, _k) in HOLDOUTS.items()]
    for audit in audits:
        if audit["status"] != "READY":
            continue
        name = audit["dataset"]
        root, _domain, _kind = HOLDOUTS[name]
        dataset = MetaDataset(root, mode="test")
        rows, patch = ensure_patch_cache(name, dataset, device, smoke)
        cells.append((f"FullVisA_to_{name}", name, dataset, rows, patch, "external", "holdout_confirmation"))
    for cell_id, target, dataset, idx_or_rows, patch, kind, group in cells:
        n = min(24, patch.shape[0]) if smoke else patch.shape[0]
        idx = idx_or_rows[:n]
        print(f"[FGWM] Full VisA -> {target} n={n}", flush=True)
        image, maps, stats = score_pack("Full VisA", target, patch, q_pc1, align_model, m_np, alpha, v_np, device, True, smoke)
        add_eval(raw, percat, cell_id, target, "Full VisA", group, dataset, idx, kind, image, maps)
        bdiag.extend(boundary_diag("Full VisA", target, dataset, idx, kind, maps))
        fstats.extend(stats)
    return raw, percat, bdiag, fstats, audits


def deltas(raw, base_variant="M0_PlainMedian"):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for source, dataset, variant in sorted(by):
        if variant in ("B0_OriginalScore", base_variant):
            continue
        b = by[(source, dataset, base_variant)]
        e = by[(source, dataset, variant)]
        rows.append({"source": source, "dataset": dataset, "variant": variant, "base": base_variant, "Delta_P-AP": fnum(e, "P-AP") - fnum(b, "P-AP"), "Delta_P-F1max": fnum(e, "P-F1max") - fnum(b, "P-F1max"), "Delta_PRO": fnum(e, "PRO") - fnum(b, "PRO"), "Delta_G": fnum(e, "G_pixel") - fnum(b, "G_pixel")})
    return rows


def decide(raw):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    keys = sorted({(r["source"], r["dataset"]) for r in raw})
    e1_keep = []
    e1_b0_ap = []
    e1_m0_pro = []
    e1_e3 = []
    pro_down_recovered = 0
    for k in keys:
        b0 = by[(k[0], k[1], "B0_OriginalScore")]
        m0 = by[(k[0], k[1], "M0_PlainMedian")]
        e1 = by[(k[0], k[1], "E1_FeatureGuidedWeightedMedian")]
        e3 = by[(k[0], k[1], "E3_RandomFeatureWeightedMedian")]
        e1_keep.append(fnum(e1, "P-AP") >= fnum(m0, "P-AP") - 0.3)
        e1_b0_ap.append(fnum(e1, "P-AP") - fnum(b0, "P-AP"))
        e1_m0_pro.append(fnum(e1, "PRO") - fnum(m0, "PRO"))
        e1_e3.append(0.5 * ((fnum(e1, "P-AP") - fnum(e3, "P-AP")) + (fnum(e1, "PRO") - fnum(e3, "PRO"))))
        if k in M0_PRO_DOWN_DATASETS and fnum(e1, "PRO") > fnum(m0, "PRO"):
            pro_down_recovered += 1
    keep = int(np.sum(e1_keep))
    ap_b0 = np.asarray(e1_b0_ap, dtype=np.float32)
    pro_m0 = np.asarray(e1_m0_pro, dtype=np.float32)
    comp = np.asarray(e1_e3, dtype=np.float32)
    success = keep >= 7 and float(pro_m0.mean()) >= 0.7 and pro_down_recovered >= 3 and int(np.sum(ap_b0 > 0)) >= 8 and float(ap_b0.mean()) >= 1.5 and float(comp.mean()) > 0
    stats = {"evaluations": len(keys), "E1_keeps_M0_P_AP": keep, "mean_E1_minus_M0_PRO": float(pro_m0.mean()), "M0_PRO_down_recovered": pro_down_recovered, "E1_P_AP_over_B0": int(np.sum(ap_b0 > 0)), "mean_E1_minus_B0_P_AP": float(ap_b0.mean()), "mean_E1_minus_E3_AP_PRO_composite": float(comp.mean())}
    if success:
        return "FEATURE_GUIDED_MEDIAN_PRESERVES_BOUNDARIES", stats
    m0_close = abs(float(pro_m0.mean())) < 0.7 and keep >= 7
    if m0_close:
        return "PLAIN_LOCAL_MEDIAN_ALREADY_SUFFICIENT", stats
    e2_e1 = []
    e3_e1 = []
    for k in keys:
        e1 = by[(k[0], k[1], "E1_FeatureGuidedWeightedMedian")]
        e2 = by[(k[0], k[1], "E2_FeatureGuidedWeightedMean")]
        e3 = by[(k[0], k[1], "E3_RandomFeatureWeightedMedian")]
        e2_e1.append(abs(fnum(e1, "G_pixel") - fnum(e2, "G_pixel")))
        e3_e1.append(abs(fnum(e1, "G_pixel") - fnum(e3, "G_pixel")))
    if float(np.mean(e2_e1)) < 0.5:
        return "FEATURE_GUIDANCE_HELPS_BUT_MEDIAN_NOT_SPECIFIC", stats
    if float(np.mean(e3_e1)) < 0.5:
        return "FEATURE_SIMILARITY_GUIDANCE_NOT_SPECIFIC", stats
    if float(ap_b0.mean()) <= 0 and float(pro_m0.mean()) <= 0:
        return "EDGE_PRESERVING_GUIDANCE_NO_SIGNAL", stats
    return "FEATURE_GUIDED_MEDIAN_INCONCLUSIVE", stats


def sota_gap(raw):
    refs = {
        ("Full VisA", "MVTec"): ("MoECLIP CVPR 2026", 92.5, 45.7, 69.10, "DIRECT_COMPARABLE"),
        ("Full VisA", "BTAD"): ("MoECLIP CVPR 2026", 96.8, 50.4, 73.60, "DIRECT_COMPARABLE"),
        ("Full VisA", "CVC-ClinicDB"): ("MoECLIP CVPR 2026", 89.7, 49.9, 69.80, "DIRECT_COMPARABLE"),
        ("Full MVTec", "VisA"): ("MoECLIP CVPR 2026", 95.6, 26.1, 60.85, "DIRECT_COMPARABLE"),
    }
    rows = []
    for r in raw:
        key = (r["source"], r["dataset"])
        if key not in refs:
            continue
        ref, pa, pap, g, comp = refs[key]
        rows.append({"source": r["source"], "dataset": r["dataset"], "variant": r["variant"], "reference": ref, "comparison_type": comp, "Delta_P-AUROC": fnum(r, "P-AUROC") - pa, "Delta_P-AP": fnum(r, "P-AP") - pap, "Delta_G": fnum(r, "G_pixel") - g})
    return rows


def write_report(raw, outcome, stats, runtime):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    lines = ["# Feature-Guided Weighted Median Consensus", "", f"OUTCOME: `{outcome}`", "", "No training, no target statistics, fixed 3x3 neighborhood.", "", "| Source | Target | Variant | P-AP | P-F1max | PRO | G |", "|---|---|---|---:|---:|---:|---:|"]
    for source, dataset in sorted({(r["source"], r["dataset"]) for r in raw}):
        for variant in VARIANTS:
            r = by[(source, dataset, variant)]
            lines.append(f"| {source} | {dataset} | {variant} | {fnum(r,'P-AP'):.4f} | {fnum(r,'P-F1max'):.4f} | {fnum(r,'PRO'):.4f} | {fnum(r,'G_pixel'):.4f} |")
    lines.extend(["", "## Gate", "", f"- E1 keeps M0 P-AP: {stats['E1_keeps_M0_P_AP']}/{stats['evaluations']}", f"- mean E1-M0 PRO: {stats['mean_E1_minus_M0_PRO']:+.4f}", f"- previous AP-up/PRO-down recovered: {stats['M0_PRO_down_recovered']}/4", f"- E1 P-AP over B0: {stats['E1_P_AP_over_B0']}/{stats['evaluations']}", f"- mean E1-B0 P-AP: {stats['mean_E1_minus_B0_P_AP']:+.4f}", f"- mean E1-E3 AP/PRO composite: {stats['mean_E1_minus_E3_AP_PRO_composite']:+.4f}", "", f"Runtime: {runtime:.1f}s."])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[FGWM] device={device}", flush=True)
    cross_raw, cross_per, cross_bd, cross_fs = run_cross_source(device, args.smoke)
    visa_raw, visa_per, visa_bd, visa_fs, audits = run_visa_source(device, args.smoke)
    raw = cross_raw + visa_raw
    outcome, stats = decide(raw)
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY_STANDARD_METRICS.csv", cross_per + visa_per)
    write_csv(OUT / "BOUNDARY_DIAGNOSTICS.csv", cross_bd + visa_bd)
    write_csv(OUT / "FEATURE_GUIDANCE_STATS.csv", cross_fs + visa_fs + deltas(raw, "M0_PlainMedian") + deltas(raw, "B0_OriginalScore"))
    write_csv(OUT / "SOTA_GAP.csv", sota_gap(raw))
    write_json(OUT / "OUTCOME.json", {"outcome": outcome, "decision_stats": stats, "runtime_sec": runtime, "device": device, "strict_source_only": True, "no_training": True, "target_statistics": False, "median_kernel": "fixed_3x3", "holdout_audit": audits})
    write_report(raw, outcome, stats, runtime)
    print(f"[FGWM] outcome={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
