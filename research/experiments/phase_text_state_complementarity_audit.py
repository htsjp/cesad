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
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mrad_audit_utils import encode_plain_text, tokenize_texts
from experiments.phase_final_cross_dataset_benchmark import load_mvtec_source, load_visa_source, protocol_cells
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import (
    item_at,
    item_meta,
    mask_np,
    torch_resize,
)
from experiments.phase_private_residual_contrast_memory import contrast_logits
from experiments.phase_sgpr import AUDIT_PAIRS, evaluate_maps, fixed_quadrant_boxes, read_csv, stitch_quadrant74
from experiments.phase_spatial_residual_refinement_standard_metric import aligned_grid, fnum, logits_from_z
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import CHECKPOINT, top1_image_score
from experiments.phase_xxxviii_visual_state_representation import IMAGE_SIZE, SIDE, norm_np
from phase_xa.official_task_forward import normalized_patch_feature, visual_outputs
from utils.transforms import get_transform


OUT = Path("results/phase_text_state_complementarity_audit/formal")
NORMAL_PROMPTS = [
    "a normal object",
    "an intact object",
    "a defect-free object",
    "an object with normal appearance",
]
ANOMALY_PROMPTS = [
    "an anomalous object",
    "a defective object",
    "a damaged object",
    "an abnormal object",
    "an object with abnormal appearance",
]
VARIANTS = ("E0_Baseline_State_NTR_FixedLRP", "E1_TextOnly_FixedLRP", "E2_StateFullText_EqualFusion", "E3_StateOrthogonalText")


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


def unit_np(x):
    x = np.asarray(x, dtype=np.float32)
    return x / max(float(np.linalg.norm(x)), 1e-12)


def load_runtime(device):
    preprocess, _target_transform = get_transform(argparse.Namespace(image_size=IMAGE_SIZE))
    clip_model, *_ = load_components(str(CHECKPOINT), device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)
    return preprocess, clip_model


@torch.no_grad()
def text_direction_1024(clip_model, device):
    texts = NORMAL_PROMPTS + ANOMALY_PROMPTS
    tokens = tokenize_texts(texts).to(device)
    feats = encode_plain_text(clip_model, tokens).float()
    feats = F.normalize(feats, dim=-1)
    n = F.normalize(feats[: len(NORMAL_PROMPTS)].mean(dim=0), dim=-1)
    a = F.normalize(feats[len(NORMAL_PROMPTS) :].mean(dim=0), dim=-1)
    d768 = F.normalize(a, dim=-1) - F.normalize(n, dim=-1)
    d768 = F.normalize(d768, dim=-1)
    proj = clip_model.visual.proj.detach().float()
    # Patch projection is z1024 @ visual.proj; lift text direction back to the
    # current Two-State feature space with the frozen CLIP projection matrix.
    d1024 = F.normalize(proj @ d768, dim=0)
    return {
        "t_N_768": n.detach().cpu().numpy().astype(np.float32),
        "t_A_768": a.detach().cpu().numpy().astype(np.float32),
        "d_t_768": d768.detach().cpu().numpy().astype(np.float32),
        "d_t_1024": d1024.detach().cpu().numpy().astype(np.float32),
    }


def source_dirs(model_info, d_text):
    d_s = unit_np(model_info["v1"])
    d_t = unit_np(d_text)
    dot = float(np.dot(d_t, d_s))
    d_parallel = dot * d_s
    d_perp = d_t - d_parallel
    perp_norm = float(np.linalg.norm(d_perp))
    d_perp_u = unit_np(d_perp) if perp_norm > 1e-12 else np.zeros_like(d_t)
    d_e3 = unit_np(d_s + d_perp_u)
    angle = float(math.degrees(math.acos(max(-1.0, min(1.0, dot)))))
    geom = {
        "source": model_info["source"],
        "state_dim": int(d_s.shape[0]),
        "text_dim_original": 768,
        "text_lift": "frozen_clip_visual_proj",
        "cosine_d_s_d_t": dot,
        "angle_deg": angle,
        "d_t_perp_norm_over_d_t_norm": perp_norm / max(float(np.linalg.norm(d_t)), 1e-12),
        "d_t_norm_1024": float(np.linalg.norm(d_t)),
        "d_s_norm": float(np.linalg.norm(d_s)),
    }
    return {"state": d_s, "text": d_t, "text_perp": d_perp_u, "state_plus_perp": d_e3}, geom


@torch.no_grad()
def score_z(model_info, z_np, direction, device):
    m_t = torch.tensor(model_info["m"], dtype=torch.float32, device=device)
    v_t = torch.tensor(direction, dtype=torch.float32, device=device)
    if model_info["use_align"]:
        z = aligned_grid(z_np, model_info["q_pc1"], model_info["align_model"], device).unsqueeze(0)
        return logits_from_z(z, m_t, model_info["alpha"], v_t).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)
    z = norm_np(np.asarray(z_np, dtype=np.float32).reshape(SIDE * SIDE, 1024))
    zt = torch.tensor(z, dtype=torch.float32, device=device)
    return contrast_logits(zt, m_t, model_info["alpha"], v_t).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)


@torch.no_grad()
def crop_patches(clip_model, preprocess, img, boxes, device):
    batch = torch.stack([preprocess(img.crop(tuple(b))) for b in boxes], dim=0).to(device)
    out = visual_outputs(clip_model, batch, adapter=None, grad=False)
    return normalized_patch_feature(out).float().cpu().numpy().astype(np.float32)


def fixed_lrp_from_maps(global37, crop_maps):
    local74 = stitch_quadrant74(crop_maps)
    global74 = torch_resize(global37, (SIDE * 2, SIDE * 2), align_corners=True)
    return (0.5 * global74 + 0.5 * local74).astype(np.float32)


def build_maps(model_info, dirs, clip_model, preprocess, target, dataset, refs, patch, kind, device, smoke):
    n = min(24, patch.shape[0]) if smoke else patch.shape[0]
    maps = {v: [] for v in VARIANTS}
    residual_maps = []
    metas = []
    diag_rows = []
    for pos, ref in enumerate(refs[:n]):
        if pos % 50 == 0:
            print(f"[TextState] {model_info['source']} -> {target} image {pos}/{n}", flush=True)
        item = item_at(dataset, kind, ref, pos)
        cat, label, path = item_meta(item, kind, ref, target)
        gt = mask_np(item)
        img = Image.open(path).convert("RGB")
        w, h = img.size
        boxes = fixed_quadrant_boxes(w, h)

        global_state = score_z(model_info, patch[pos], dirs["state"], device)
        global_text = score_z(model_info, patch[pos], dirs["text"], device)
        global_perp = score_z(model_info, patch[pos], dirs["text_perp"], device)
        global_e3 = score_z(model_info, patch[pos], dirs["state_plus_perp"], device)

        patches = crop_patches(clip_model, preprocess, img, boxes, device)
        crop_state = [score_z(model_info, patches[i], dirs["state"], device) for i in range(4)]
        crop_text = [score_z(model_info, patches[i], dirs["text"], device) for i in range(4)]
        crop_perp = [score_z(model_info, patches[i], dirs["text_perp"], device) for i in range(4)]
        crop_e3 = [score_z(model_info, patches[i], dirs["state_plus_perp"], device) for i in range(4)]

        e0 = fixed_lrp_from_maps(global_state, crop_state)
        e1 = fixed_lrp_from_maps(global_text, crop_text)
        e_perp = fixed_lrp_from_maps(global_perp, crop_perp)
        e2 = (0.5 * e0 + 0.5 * e1).astype(np.float32)
        e3 = fixed_lrp_from_maps(global_e3, crop_e3)

        maps["E0_Baseline_State_NTR_FixedLRP"].append(e0.astype(np.float16))
        maps["E1_TextOnly_FixedLRP"].append(e1.astype(np.float16))
        maps["E2_StateFullText_EqualFusion"].append(e2.astype(np.float16))
        maps["E3_StateOrthogonalText"].append(e3.astype(np.float16))
        residual_maps.append(e_perp.astype(np.float16))
        metas.append({"category": cat, "label": label, "mask": gt, "img_path": path})
        diag_rows.append({"source": model_info["source"], "dataset": target, "image_index": pos, "category": cat, "label": int(label), "image_score_E0": top1_image_score(e0), "image_score_E1": top1_image_score(e1), "image_score_E2": top1_image_score(e2), "image_score_E3": top1_image_score(e3)})
        img.close()
    return maps, residual_maps, metas, diag_rows


def error_rescue(source, target, maps, residual_maps, metas):
    def sample_flat(values, rng, cap):
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.size <= cap:
            return values
        idx = rng.choice(values.size, size=cap, replace=False)
        return values[idx]

    seed = sum(ord(c) for c in f"{source}->{target}")
    rng = np.random.default_rng(seed)
    rows = []
    all_norm, all_anom = [], []
    for e0, meta in zip(maps["E0_Baseline_State_NTR_FixedLRP"], metas):
        s = torch_resize(e0, meta["mask"].shape, align_corners=True)
        m = np.asarray(meta["mask"], dtype=bool)
        if np.any(~m):
            all_norm.append(sample_flat(s[~m], rng, 2048))
        if np.any(m):
            all_anom.append(sample_flat(s[m], rng, 2048))
    if not all_norm or not all_anom:
        return rows
    all_norm = np.concatenate(all_norm) if all_norm else np.empty(0, dtype=np.float32)
    all_anom = np.concatenate(all_anom) if all_anom else np.empty(0, dtype=np.float32)
    norm_p95 = float(np.percentile(all_norm, 95))
    anom_p20 = float(np.percentile(all_anom, 20))
    buckets = {
        "E0_false_positive_normal": {"count": 0, "sum": 0.0, "samples": []},
        "E0_correct_normal": {"count": 0, "sum": 0.0, "samples": []},
        "E0_false_negative_anomaly": {"count": 0, "sum": 0.0, "samples": []},
        "E0_correct_anomaly": {"count": 0, "sum": 0.0, "samples": []},
    }
    for e0, res, meta in zip(maps["E0_Baseline_State_NTR_FixedLRP"], residual_maps, metas):
        s0 = torch_resize(e0, meta["mask"].shape, align_corners=True)
        sr = torch_resize(res, meta["mask"].shape, align_corners=True)
        m = np.asarray(meta["mask"], dtype=bool)
        masks = {
            "E0_false_positive_normal": (~m) & (s0 >= norm_p95),
            "E0_correct_normal": (~m) & (s0 < norm_p95),
            "E0_false_negative_anomaly": m & (s0 <= anom_p20),
            "E0_correct_anomaly": m & (s0 > anom_p20),
        }
        for name, selector in masks.items():
            vals = np.asarray(sr[selector], dtype=np.float32).reshape(-1)
            if vals.size == 0:
                continue
            buckets[name]["count"] += int(vals.size)
            buckets[name]["sum"] += float(vals.sum(dtype=np.float64))
            buckets[name]["samples"].append(sample_flat(vals, rng, 2048))
    stats = {}
    for name, info in buckets.items():
        arr = np.concatenate(info["samples"]) if info["samples"] else np.empty(0, dtype=np.float32)
        mean = info["sum"] / info["count"] if info["count"] else float("nan")
        median = float(np.median(arr)) if arr.size else float("nan")
        p90 = float(np.percentile(arr, 90)) if arr.size else float("nan")
        stats[name] = {"count": int(info["count"]), "mean": float(mean), "median": median, "p90": p90}
        rows.append({"source": source, "dataset": target, "bucket": name, "count": int(info["count"]), "residual_mean": float(mean), "residual_median": median, "residual_p90": p90})
    fn = stats["E0_false_negative_anomaly"]
    ca = stats["E0_correct_anomaly"]
    fp = stats["E0_false_positive_normal"]
    cn = stats["E0_correct_normal"]
    rows.append({"source": source, "dataset": target, "bucket": "FN_minus_correct_anomaly", "count": int(fn["count"] + ca["count"]), "residual_mean": float(fn["mean"] - ca["mean"]) if fn["count"] and ca["count"] else float("nan"), "residual_median": float(fn["median"] - ca["median"]) if fn["count"] and ca["count"] else float("nan"), "residual_p90": float(fn["p90"] - ca["p90"]) if fn["count"] and ca["count"] else float("nan")})
    rows.append({"source": source, "dataset": target, "bucket": "FP_minus_correct_normal", "count": int(fp["count"] + cn["count"]), "residual_mean": float(fp["mean"] - cn["mean"]) if fp["count"] and cn["count"] else float("nan"), "residual_median": float(fp["median"] - cn["median"]) if fp["count"] and cn["count"] else float("nan"), "residual_p90": float(fp["p90"] - cn["p90"]) if fp["count"] and cn["count"] else float("nan")})
    return rows


def decide(raw, geom_rows):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for variant in ("E1_TextOnly_FixedLRP", "E2_StateFullText_EqualFusion", "E3_StateOrthogonalText"):
        vals = []
        non_drop = 0
        improved = 0
        for source, dataset in sorted(AUDIT_PAIRS):
            e0 = by.get((source, dataset, "E0_Baseline_State_NTR_FixedLRP"))
            ev = by.get((source, dataset, variant))
            if not (e0 and ev):
                continue
            d = fnum(ev, "M_FR_P_AP") - fnum(e0, "M_FR_P_AP")
            if not np.isfinite(d):
                continue
            vals.append(d)
            non_drop += int(d >= -0.2)
            improved += int(d > 0)
        rows.append({"variant": variant, "mean_delta_P_AP_vs_E0": float(np.mean(vals)) if vals else float("nan"), "median_delta_P_AP_vs_E0": float(np.median(vals)) if vals else float("nan"), "improved": improved, "non_drop_ge_minus_0_2": non_drop, "total": len(vals)})
    best = max(rows, key=lambda r: fnum(r, "mean_delta_P_AP_vs_E0"))
    geom_redundant = bool(geom_rows and np.mean([abs(fnum(g, "cosine_d_s_d_t")) for g in geom_rows]) >= 0.9)
    orth_weak = bool(geom_rows and np.mean([fnum(g, "d_t_perp_norm_over_d_t_norm") for g in geom_rows]) <= 0.3)
    if fnum(best, "mean_delta_P_AP_vs_E0") >= 0.5 and int(best["improved"]) >= 5 and int(best["non_drop_ge_minus_0_2"]) >= 6:
        outcome = "TEXT_PROVIDES_TRANSFERABLE_COMPLEMENTARY_SIGNAL"
    elif geom_redundant or orth_weak:
        outcome = "TEXT_IS_MOSTLY_REDUNDANT_WITH_TWO_STATE"
    else:
        outcome = "TEXT_DOES_NOT_IMPROVE_CROSS_DATASET_TRANSFER"
    return outcome, rows


def write_report(raw, summary, geom, err, outcome, runtime):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    lines = [
        "# Text-State Complementarity Audit",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "Frozen CLIP text prompts were lifted into the current 1024-d Two-State feature space with the existing frozen visual projection. No prompt learning, adapter, target statistics, or target tuning was used.",
        "",
        "## Geometry",
        "",
        "| Source | cos(d_s,d_t) | angle | ||d_t_perp||/||d_t|| |",
        "|---|---:|---:|---:|",
    ]
    for g in geom:
        lines.append(f"| {g['source']} | {fnum(g,'cosine_d_s_d_t'):.4f} | {fnum(g,'angle_deg'):.2f} | {fnum(g,'d_t_perp_norm_over_d_t_norm'):.4f} |")
    lines += ["", "## Mean Delta", "", "| Variant | mean dP-AP | median dP-AP | improved | non-drop |", "|---|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['variant']} | {fnum(r,'mean_delta_P_AP_vs_E0'):+.4f} | {fnum(r,'median_delta_P_AP_vs_E0'):+.4f} | {r['improved']}/{r['total']} | {r['non_drop_ge_minus_0_2']}/{r['total']} |")
    lines += ["", "## Per Transfer P-AP", "", "| Source | Target | E0 | E1 Text | E2 Fusion | E3 Orth | E2-E0 | E3-E0 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for source, target in sorted(AUDIT_PAIRS):
        e0 = by.get((source, target, "E0_Baseline_State_NTR_FixedLRP"))
        e1 = by.get((source, target, "E1_TextOnly_FixedLRP"))
        e2 = by.get((source, target, "E2_StateFullText_EqualFusion"))
        e3 = by.get((source, target, "E3_StateOrthogonalText"))
        if e0 and e1 and e2 and e3:
            lines.append(f"| {source} | {target} | {fnum(e0,'M_FR_P_AP'):.4f} | {fnum(e1,'M_FR_P_AP'):.4f} | {fnum(e2,'M_FR_P_AP'):.4f} | {fnum(e3,'M_FR_P_AP'):.4f} | {fnum(e2,'M_FR_P_AP')-fnum(e0,'M_FR_P_AP'):+.4f} | {fnum(e3,'M_FR_P_AP')-fnum(e0,'M_FR_P_AP'):+.4f} |")
    lines += ["", "## Error Rescue Diagnostic", "", "Positive `FN_minus_correct_anomaly` means the text-orthogonal residual responds more strongly on E0 false-negative anomaly pixels. Negative `FP_minus_correct_normal` means it suppresses E0 false-positive normal pixels.", "", "| Source | Target | Bucket | residual mean delta | residual median delta |", "|---|---|---|---:|---:|"]
    for r in err:
        if r["bucket"] in ("FN_minus_correct_anomaly", "FP_minus_correct_normal"):
            lines.append(f"| {r['source']} | {r['dataset']} | {r['bucket']} | {fnum(r,'residual_mean'):+.4f} | {fnum(r,'residual_median'):+.4f} |")
    lines += ["", f"Runtime: {runtime:.1f}s."]
    write_text(OUT / "REPORT.md", "\n".join(lines))


def run(smoke=False):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocess, clip_model = load_runtime(device)
    text = text_direction_1024(clip_model, device)
    models = [load_visa_source(device), load_mvtec_source()]
    raw, cats, geom_rows, diag_rows, err_rows = [], [], [], [], []
    for model_info in models:
        dirs, geom = source_dirs(model_info, text["d_t_1024"])
        geom_rows.append(geom)
        cells, _audits = protocol_cells(model_info["source"], device, smoke)
        for cell_id, target, dataset, idx_or_rows, patch, kind in cells:
            if (model_info["source"], target) not in AUDIT_PAIRS:
                continue
            n = min(24, patch.shape[0]) if smoke else patch.shape[0]
            print(f"[TextState] evaluate {model_info['source']} -> {target} n={n}", flush=True)
            maps, residual_maps, metas, diag = build_maps(model_info, dirs, clip_model, preprocess, target, dataset, idx_or_rows[:n], patch[:n], kind, device, smoke)
            rr, cc = evaluate_maps(model_info["source"], target, idx_or_rows[:n], dataset, kind, maps, metas, variants=VARIANTS)
            raw.extend(rr)
            cats.extend(cc)
            diag_rows.extend(diag)
            err_rows.extend(error_rescue(model_info["source"], target, maps, residual_maps, metas))
            if not smoke:
                write_csv(OUT / "RAW_METRICS.partial.csv", raw)
                write_csv(OUT / "ERROR_RESCUE_DIAGNOSTIC.partial.csv", err_rows)
    outcome, sum_rows = decide(raw, geom_rows)
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY_FULL_RES.csv", cats)
    write_csv(OUT / "MEAN_DELTA.csv", sum_rows)
    write_csv(OUT / "TEXT_STATE_GEOMETRY.csv", geom_rows)
    write_csv(OUT / "ERROR_RESCUE_DIAGNOSTIC.csv", err_rows)
    write_csv(OUT / "IMAGE_SCORE_DIAGNOSTIC.csv", diag_rows)
    write_json(OUT / "OUTCOME.json", {"outcome": outcome, "summary": sum_rows, "geometry": geom_rows, "runtime_sec": runtime, "normal_prompts": NORMAL_PROMPTS, "anomaly_prompts": ANOMALY_PROMPTS, "text_lift": "frozen_clip_visual_proj"})
    write_report(raw, sum_rows, geom_rows, err_rows, outcome, runtime)
    print(f"[TextState] OUTCOME={outcome} runtime={runtime:.1f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    run(args.smoke)


if __name__ == "__main__":
    main()
