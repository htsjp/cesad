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
from scipy import ndimage
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_final_cross_dataset_benchmark import load_mvtec_source, load_visa_source, protocol_cells
from experiments.phase_fova_field_of_view_amplification_audit import MAP_CACHE as FOVA_MAP_CACHE, slug as pair_slug
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import item_at, item_meta, mask_np, torch_resize
from experiments.phase_sgpr import AUDIT_PAIRS, evaluate_maps
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xxxviii_visual_state_representation import IMAGE_SIZE, SIDE


OUT = Path("results/phase_spsd_state_preserving_structural_diffusion/formal")
DINO_CACHE = OUT / "DINO_CACHE"
VARIANTS = ("E0_FixedLRP", "E1_DINOAffinityPropagation", "E2_StateGatedDINOPropagation")
K_FIXED = 8
ALPHA_FIXED = 0.25
SIGMA_FIXED = 0.10
DINO_BACKBONE = "dinov2_vitl14"
SAVE_DINO_FEATURE_CACHE = False


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


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fnum(row, key):
    try:
        v = float(row.get(key, float("nan")))
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def finite_mean(vals):
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def finite_median(vals):
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def safe_auc(y, s):
    y = np.asarray(y, dtype=np.uint8).reshape(-1)
    s = np.asarray(s, dtype=np.float32).reshape(-1)
    keep = np.isfinite(s)
    y, s = y[keep], s[keep]
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_s = s[order]
    n = len(sorted_s)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_s[j] == sorted_s[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    pos = y.astype(bool)
    n_pos = float(np.sum(pos))
    n_neg = float(y.size - np.sum(pos))
    return float((np.sum(ranks[pos]) - n_pos * (n_pos + 1.0) / 2.0) / max(n_pos * n_neg, 1.0) * 100.0)


def dino_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def load_dino(device):
    model = torch.hub.load("facebookresearch/dinov2", DINO_BACKBONE, trust_repo=True).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def extract_dino_patch(model, tfm, path, device):
    img = Image.open(path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    out = model.forward_features(x)
    patch = F.normalize(out["x_norm_patchtokens"].float(), dim=-1).squeeze(0).cpu().numpy().astype(np.float32)
    img.close()
    return patch


def load_pair_items(source, target, device, smoke):
    models = {"Full VisA": load_visa_source(device), "Full MVTec": load_mvtec_source()}
    cells, _ = protocol_cells(source, device, smoke)
    for _cell_id, dst, dataset, refs, patch, kind in cells:
        if dst == target:
            return models[source], dataset, refs, kind
    raise KeyError(f"missing protocol cell {source}->{target}")


def pair_meta(source, target, dataset, refs, kind):
    metas, rows = [], []
    for pos, ref in enumerate(refs):
        item = item_at(dataset, kind, ref, pos)
        cat, label, path = item_meta(item, kind, ref, target)
        mask = mask_np(item)
        metas.append({"category": cat, "label": int(label), "mask": mask, "img_path": path})
        rows.append({"source": source, "dataset": target, "category": cat, "image_index": pos, "label": int(label), "img_path": path, **morphology(mask)})
    assign_groups(rows)
    return metas, rows


def morphology(mask):
    mask = np.asarray(mask, dtype=bool)
    pix = int(mask.sum())
    h, w = mask.shape
    if pix == 0:
        return {"num_anomaly_patches": 0, "thinness": 0.0}
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    boundary = mask & ~eroded
    gm = torch.tensor(mask.astype(np.float32)).view(1, 1, h, w)
    patch = F.interpolate(gm, size=(SIDE, SIDE), mode="nearest").view(SIDE, SIDE).numpy() > 0.5
    return {"num_anomaly_patches": int(patch.sum()), "thinness": float(boundary.sum() / max(pix, 1))}


def assign_groups(rows):
    pos = [r for r in rows if int(r["label"]) == 1 and fnum(r, "num_anomaly_patches") > 0]
    patch_vals = [fnum(r, "num_anomaly_patches") for r in pos]
    thin_vals = [fnum(r, "thinness") for r in pos]
    small_thr = float(np.percentile(patch_vals, 25)) if patch_vals else float("nan")
    large_thr = float(np.percentile(patch_vals, 75)) if patch_vals else float("nan")
    thin_thr = float(np.percentile(thin_vals, 75)) if thin_vals else float("nan")
    for r in rows:
        p = fnum(r, "num_anomaly_patches")
        t = fnum(r, "thinness")
        r["group_small"] = bool(np.isfinite(small_thr) and p > 0 and p <= small_thr)
        r["group_large"] = bool(np.isfinite(large_thr) and p >= large_thr)
        r["group_thin"] = bool(np.isfinite(thin_thr) and p > 0 and t >= thin_thr)
        r["group_thick"] = bool(np.isfinite(thin_thr) and p > 0 and t < thin_thr)


def load_e0_maps(source, target, n):
    path = FOVA_MAP_CACHE / f"{pair_slug(source, target)}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing FOVA map cache for {source}->{target}: {path}. Run Phase FOVA first.")
    arr = np.load(path)
    q = np.asarray(arr["q_lrp"][:n], dtype=np.float32)
    if q.ndim != 3 or q.shape[1:] != (SIDE * 2, SIDE * 2):
        raise ValueError(f"unexpected q_lrp shape {q.shape} in {path}")
    return q


def dino_cache_path(source, target, n):
    return DINO_CACHE / f"{pair_slug(source, target)}__n{n}_{DINO_BACKBONE}.npz"


def load_or_extract_dino(source, target, metas, device):
    path = dino_cache_path(source, target, len(metas))
    if SAVE_DINO_FEATURE_CACHE and path.exists():
        try:
            data = np.load(path)
            return np.asarray(data["patch"], dtype=np.float32), {"cache": str(path), "loaded": True}
        except Exception as exc:
            print(f"[SPSD] ignore unreadable DINO cache {path}: {exc}", flush=True)
    DINO_CACHE.mkdir(parents=True, exist_ok=True)
    model = load_dino(device)
    tfm = dino_transform()
    feats = []
    for i, meta in enumerate(metas):
        if i % 50 == 0:
            print(f"[SPSD] DINO {source}->{target} {i}/{len(metas)}", flush=True)
        feats.append(extract_dino_patch(model, tfm, meta["img_path"], device).astype(np.float16))
    patch = np.stack(feats).astype(np.float16)
    if SAVE_DINO_FEATURE_CACHE:
        np.savez_compressed(path, patch=patch)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return patch.astype(np.float32), {"cache": str(path) if SAVE_DINO_FEATURE_CACHE else "", "loaded": False, "saved": SAVE_DINO_FEATURE_CACHE}


def diffuse_one(q74, feat, k=K_FIXED, alpha=ALPHA_FIXED, sigma=SIGMA_FIXED):
    n = feat.shape[0]
    side = int(round(math.sqrt(n)))
    if side * side != n:
        raise ValueError(f"DINO patch count is not square: {n}")
    q = torch_resize(q74, (side, side), align_corners=True).reshape(-1).astype(np.float32)
    x = feat.astype(np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    aff = x @ x.T
    np.fill_diagonal(aff, 1.0)
    kk = min(int(k), max(n, 1))
    top = np.argpartition(-aff, kth=kk - 1, axis=1)[:, :kk]
    row = np.arange(n)[:, None]
    a_top = np.maximum(aff[row, top], 0.0)
    a_norm = a_top / np.maximum(a_top.sum(axis=1, keepdims=True), 1e-12)
    prop = np.sum(a_norm * q[top], axis=1)
    e1 = (1.0 - alpha) * q + alpha * prop

    gate = np.exp(-np.abs(q[:, None] - q[None, :]) / max(float(sigma), 1e-12)).astype(np.float32)
    w = aff * gate
    np.fill_diagonal(w, 1.0)
    top_g = np.argpartition(-w, kth=kk - 1, axis=1)[:, :kk]
    wg = np.maximum(w[row, top_g], 0.0)
    wg = wg / np.maximum(wg.sum(axis=1, keepdims=True), 1e-12)
    prop_g = np.sum(wg * q[top_g], axis=1)
    e2 = (1.0 - alpha) * q + alpha * prop_g
    return (
        torch_resize(e1.reshape(side, side), (SIDE * 2, SIDE * 2), align_corners=True),
        torch_resize(e2.reshape(side, side), (SIDE * 2, SIDE * 2), align_corners=True),
        {
            "dino_grid": f"{side}x{side}",
            "mean_topk_affinity": float(np.mean(a_top)),
            "mean_state_gate": float(np.mean(gate[row, top_g])),
            "mean_topk_gated_weight": float(np.mean(wg)),
        },
    )


def build_maps(source, target, e0_maps, dino_patch):
    maps = {v: [] for v in VARIANTS}
    rows = []
    for i, q in enumerate(e0_maps):
        if i % 100 == 0:
            print(f"[SPSD] diffuse {source}->{target} {i}/{len(e0_maps)}", flush=True)
        e1, e2, stat = diffuse_one(q, dino_patch[i])
        maps["E0_FixedLRP"].append(q.astype(np.float32))
        maps["E1_DINOAffinityPropagation"].append(e1.astype(np.float32))
        maps["E2_StateGatedDINOPropagation"].append(e2.astype(np.float32))
        rows.append({"source": source, "dataset": target, "image_index": i, **stat, "K": K_FIXED, "alpha": ALPHA_FIXED, "sigma": SIGMA_FIXED})
    return maps, rows


def morphology_diagnostics(cat_rows):
    by = {(r["source"], r["dataset"], r["category"], r["variant"], r.get("protocol")): r for r in cat_rows}
    morph_path = OUT / "MORPHOLOGY_GROUPS_USED.csv"
    morph = read_csv(morph_path) if morph_path.exists() else []
    group_by = {(r["source"], r["dataset"], r["category"]): r for r in morph}
    rows = []
    groups = ("small", "large", "thin", "thick")
    for source, target, cat in sorted({(r["source"], r["dataset"], r["category"]) for r in morph}):
        base = by.get((source, target, cat, "E0_FixedLRP", "M"))
        e1 = by.get((source, target, cat, "E1_DINOAffinityPropagation", "M"))
        e2 = by.get((source, target, cat, "E2_StateGatedDINOPropagation", "M"))
        info = group_by.get((source, target, cat), {})
        if not (base and e1 and e2):
            continue
        for g in groups:
            if str(info.get(f"group_{g}", "False")).lower() == "true":
                rows.append({
                    "source": source,
                    "dataset": target,
                    "category": cat,
                    "group": g,
                    "E1_minus_E0_M_FR_P_AP": fnum(e1, "M_FR_P_AP") - fnum(base, "M_FR_P_AP"),
                    "E2_minus_E0_M_FR_P_AP": fnum(e2, "M_FR_P_AP") - fnum(base, "M_FR_P_AP"),
                    "E2_minus_E1_M_FR_P_AP": fnum(e2, "M_FR_P_AP") - fnum(e1, "M_FR_P_AP"),
                    "num_anomaly_patches": fnum(info, "num_anomaly_patches"),
                    "thinness": fnum(info, "thinness"),
                })
    for g in groups:
        part = [r for r in rows if r["group"] == g]
        for variant in ("E1", "E2"):
            key = f"{variant}_minus_E0_M_FR_P_AP"
            vals = [fnum(r, key) for r in part]
            rows.append({
                "source": "SUMMARY",
                "dataset": "ALL",
                "category": "ALL",
                "group": g,
                f"mean_{key}": finite_mean(vals),
                f"median_{key}": finite_median(vals),
                "improved": sum(1 for v in vals if np.isfinite(v) and v > 0),
                "total": sum(1 for v in vals if np.isfinite(v)),
            })
    return rows


def summarize_metrics(raw):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for variant in VARIANTS[1:]:
        vals = []
        improved = 0
        nondrop = 0
        for source, target in sorted(AUDIT_PAIRS):
            b = by.get((source, target, VARIANTS[0]))
            e = by.get((source, target, variant))
            if not (b and e):
                continue
            d = fnum(e, "M_FR_P_AP") - fnum(b, "M_FR_P_AP")
            if not np.isfinite(d):
                continue
            vals.append(d)
            improved += int(d > 0)
            nondrop += int(d >= -0.2)
            rows.append({
                "source": source,
                "dataset": target,
                "variant": variant,
                "E_minus_E0_M_FR_P_AUROC": fnum(e, "M_FR_P_AUROC") - fnum(b, "M_FR_P_AUROC"),
                "E_minus_E0_M_FR_P_AP": d,
                "E_minus_E0_M_FR_P_F1max": fnum(e, "M_FR_P_F1max") - fnum(b, "M_FR_P_F1max"),
                "E_minus_E0_M_FR_PRO": fnum(e, "M_FR_PRO") - fnum(b, "M_FR_PRO"),
            })
        rows.append({
            "source": "SUMMARY",
            "dataset": "ALL",
            "variant": variant,
            "mean_delta_M_FR_P_AP": finite_mean(vals),
            "median_delta_M_FR_P_AP": finite_median(vals),
            "improved": improved,
            "non_drop_ge_minus_0_2": nondrop,
            "total": len(vals),
        })
    e2e1 = []
    for source, target in sorted(AUDIT_PAIRS):
        e1 = by.get((source, target, VARIANTS[1]))
        e2 = by.get((source, target, VARIANTS[2]))
        if e1 and e2:
            d = fnum(e2, "M_FR_P_AP") - fnum(e1, "M_FR_P_AP")
            if np.isfinite(d):
                e2e1.append(d)
                rows.append({"source": source, "dataset": target, "variant": "E2_minus_E1", "E2_minus_E1_M_FR_P_AP": d})
    rows.append({"source": "SUMMARY", "dataset": "ALL", "variant": "E2_minus_E1", "mean_delta_M_FR_P_AP": finite_mean(e2e1), "median_delta_M_FR_P_AP": finite_median(e2e1), "improved": sum(1 for d in e2e1 if d > 0), "non_drop_ge_minus_0_2": sum(1 for d in e2e1 if d >= -0.2), "total": len(e2e1)})
    return rows


def decide(summary):
    s = {(r["variant"], r["source"]): r for r in summary if r["source"] == "SUMMARY"}
    e1 = s.get(("E1_DINOAffinityPropagation", "SUMMARY"), {})
    e2 = s.get(("E2_StateGatedDINOPropagation", "SUMMARY"), {})
    e21 = s.get(("E2_minus_E1", "SUMMARY"), {})
    if fnum(e2, "mean_delta_M_FR_P_AP") >= 1.0 and int(e2.get("improved", 0)) >= 6 and fnum(e21, "mean_delta_M_FR_P_AP") > 0:
        return "STATE_GATING_IMPROVES_DINO_PROPAGATION"
    if fnum(e1, "mean_delta_M_FR_P_AP") >= 1.0 and int(e1.get("improved", 0)) >= 6:
        return "DINO_STRUCTURE_PROVIDES_STRONG_TRANSFER_GAIN"
    if fnum(e1, "mean_delta_M_FR_P_AP") > 0.3 and int(e1.get("improved", 0)) >= 5 and fnum(e21, "mean_delta_M_FR_P_AP") <= 0.2:
        return "DINO_PROPAGATION_HELPS_BUT_STATE_GATING_HAS_NO_ADDED_VALUE"
    return "DINO_STRUCTURAL_REFINEMENT_HAS_NO_COMPETITIVE_VALUE"


def write_report(raw, summary, morph_rows, outcome, runtime):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    lines = [
        "# Phase SPSD: State-Preserving Structural Diffusion",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        f"Frozen DINO backbone: `{DINO_BACKBONE}`. Fixed parameters: K={K_FIXED}, alpha={ALPHA_FIXED}, sigma={SIGMA_FIXED}. These are uniform fixed source-agnostic values; no source calibration sweep or target tuning was used.",
        "",
        "## Fusion Summary",
        "",
        "| Variant | mean dP-AP | median dP-AP | improved | non-drop |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in summary:
        if r["source"] == "SUMMARY":
            lines.append(f"| {r['variant']} | {fnum(r,'mean_delta_M_FR_P_AP'):+.4f} | {fnum(r,'median_delta_M_FR_P_AP'):+.4f} | {r.get('improved','')}/{r.get('total','')} | {r.get('non_drop_ge_minus_0_2','')}/{r.get('total','')} |")
    lines += ["", "## Per-Transfer Full-Resolution Metrics", "", "| Source | Target | Variant | P-AUROC | P-AP | P-F1max | PRO |", "|---|---|---|---:|---:|---:|---:|"]
    for source, target in sorted(AUDIT_PAIRS):
        for v in VARIANTS:
            r = by.get((source, target, v))
            if r:
                lines.append(f"| {source} | {target} | {v} | {fnum(r,'M_FR_P_AUROC'):.4f} | {fnum(r,'M_FR_P_AP'):.4f} | {fnum(r,'M_FR_P_F1max'):.4f} | {fnum(r,'M_FR_PRO'):.4f} |")
    lines += ["", "## P-AP Deltas", "", "| Source | Target | E1-E0 | E2-E0 | E2-E1 |", "|---|---|---:|---:|---:|"]
    for source, target in sorted(AUDIT_PAIRS):
        e0, e1, e2 = by.get((source, target, VARIANTS[0])), by.get((source, target, VARIANTS[1])), by.get((source, target, VARIANTS[2]))
        if e0 and e1 and e2:
            lines.append(f"| {source} | {target} | {fnum(e1,'M_FR_P_AP')-fnum(e0,'M_FR_P_AP'):+.4f} | {fnum(e2,'M_FR_P_AP')-fnum(e0,'M_FR_P_AP'):+.4f} | {fnum(e2,'M_FR_P_AP')-fnum(e1,'M_FR_P_AP'):+.4f} |")
    lines += ["", "## Morphology Diagnostics", "", "| Group | Variant | mean dP-AP | improved |", "|---|---|---:|---:|"]
    for r in morph_rows:
        if r.get("source") == "SUMMARY":
            for var in ("E1", "E2"):
                key = f"mean_{var}_minus_E0_M_FR_P_AP"
                if key in r and r[key] != "":
                    lines.append(f"| {r['group']} | {var} | {fnum(r,key):+.4f} | {r.get('improved','')}/{r.get('total','')} |")
    lines += [
        "",
        "## Notes",
        "",
        "- E1 uses DINO affinity top-K propagation only.",
        "- E2 multiplies DINO affinity by `exp(-abs(q_i-q_j)/sigma)` before top-K normalization.",
        "- DINO is a frozen structural prior and never directly predicts anomaly.",
        "- GT is used only for metrics and post-hoc small/thin diagnostics.",
        f"- Runtime: {runtime:.1f}s.",
    ]
    write_text(OUT / "REPORT.md", "\n".join(lines))


def run(smoke=False):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw, cats, diff_rows, meta_rows_all, dino_audit = [], [], [], [], []
    for source, target in sorted(AUDIT_PAIRS):
        _model_info, dataset, refs, kind = load_pair_items(source, target, device, smoke)
        n = min(24, len(refs)) if smoke else len(refs)
        refs_use = refs[:n]
        print(f"[SPSD] evaluate {source}->{target} n={n}", flush=True)
        metas, meta_rows = pair_meta(source, target, dataset, refs_use, kind)
        e0_maps = load_e0_maps(source, target, n)
        dino_patch, audit = load_or_extract_dino(source, target, metas, device)
        maps, drows = build_maps(source, target, e0_maps, dino_patch)
        rr, cc = evaluate_maps(source, target, refs_use, dataset, kind, maps, metas, variants=VARIANTS)
        raw.extend(rr)
        cats.extend(cc)
        diff_rows.extend(drows)
        meta_rows_all.extend(meta_rows)
        dino_audit.append({"source": source, "dataset": target, "n": n, "dino_backbone": DINO_BACKBONE, **audit})
        if not smoke:
            write_csv(OUT / "RAW_METRICS.partial.csv", raw)
            write_csv(OUT / "DIFFUSION_AUDIT.partial.csv", diff_rows)
    write_csv(OUT / "MORPHOLOGY_GROUPS_USED.csv", meta_rows_all)
    summary = summarize_metrics(raw)
    morph = morphology_diagnostics(cats)
    outcome = decide(summary)
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY_FULL_RES.csv", cats)
    write_csv(OUT / "FUSION_DELTA_SUMMARY.csv", summary)
    write_csv(OUT / "MORPHOLOGY_DIAGNOSTICS.csv", morph)
    write_csv(OUT / "DIFFUSION_AUDIT.csv", diff_rows)
    write_csv(OUT / "DINO_BACKBONE_AUDIT.csv", dino_audit)
    write_json(OUT / "OUTCOME.json", {"outcome": outcome, "runtime_sec": runtime, "variants": VARIANTS, "dino_backbone": DINO_BACKBONE, "K": K_FIXED, "alpha": ALPHA_FIXED, "sigma": SIGMA_FIXED, "no_train": True, "no_target_tuning": True})
    write_report(raw, summary, morph, outcome, runtime)
    print(f"[SPSD] OUTCOME={outcome} runtime={runtime:.1f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    run(args.smoke)


if __name__ == "__main__":
    main()
