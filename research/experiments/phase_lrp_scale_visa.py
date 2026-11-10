from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_final_cross_dataset_benchmark import load_mvtec_source, protocol_cells
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import (
    METRICS,
    PROTOCOLS,
    item_at,
    item_meta,
    mask_np,
    pg_metrics,
    score_feature_grid,
    summarize_category,
    torch_resize,
)
from experiments.phase_sgpr import mask_morphology
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import CHECKPOINT, top1_image_score
from experiments.phase_xxxviii_visual_state_representation import IMAGE_SIZE, SIDE
from phase_xa.official_task_forward import normalized_patch_feature, visual_outputs
from utils.transforms import get_transform


OUT = Path("results/phase_lrp_scale_visa")
SOURCE = "Full MVTec"
TARGET = "VisA"
VARIANTS = ("E0_2x2_LRP", "E1_3x3_LRP")


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fnum(v: Any, key: str | None = None) -> float:
    if key is not None:
        v = v.get(key, "") if isinstance(v, dict) else ""
    try:
        x = float(v)
        return x if np.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.uint8).reshape(-1)
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    return float(roc_auc_score(y, s) * 100.0) if np.unique(y).size == 2 else float("nan")


def safe_ap(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.uint8).reshape(-1)
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    return float(average_precision_score(y, s) * 100.0) if np.unique(y).size == 2 else float("nan")


def grid_boxes(w: int, h: int, grid: int) -> list[tuple[int, int, int, int]]:
    xs = [int(round(i * w / grid)) for i in range(grid + 1)]
    ys = [int(round(i * h / grid)) for i in range(grid + 1)]
    boxes: list[tuple[int, int, int, int]] = []
    for gy in range(grid):
        for gx in range(grid):
            boxes.append((xs[gx], ys[gy], xs[gx + 1], ys[gy + 1]))
    return boxes


@torch.no_grad()
def score_crops(model_info: dict[str, Any], clip_model: torch.nn.Module, preprocess, img: Image.Image, boxes: list[tuple[int, int, int, int]], device: str, chunk: int = 3) -> list[np.ndarray]:
    scores: list[np.ndarray] = []
    for start in range(0, len(boxes), chunk):
        part = boxes[start : start + chunk]
        batch = torch.stack([preprocess(img.crop(box)) for box in part], dim=0).to(device)
        out = visual_outputs(clip_model, batch, adapter=None, grad=False)
        patches = normalized_patch_feature(out).float().cpu().numpy().astype(np.float32)
        for i in range(len(part)):
            scores.append(score_feature_grid(model_info, patches[i], device))
    return scores


def stitch_regular_grid(local_scores: list[np.ndarray], grid: int) -> np.ndarray:
    side = SIDE * grid
    out = np.zeros((side, side), dtype=np.float32)
    k = 0
    for gy in range(grid):
        for gx in range(grid):
            out[gy * SIDE : (gy + 1) * SIDE, gx * SIDE : (gx + 1) * SIDE] = local_scores[k]
            k += 1
    return out


@torch.no_grad()
def local_lrp_map(model_info: dict[str, Any], clip_model: torch.nn.Module, preprocess, img_path: str, grid: int, device: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    boxes = grid_boxes(w, h, grid)
    local_scores = score_crops(model_info, clip_model, preprocess, img, boxes, device)
    rows = []
    for i, b in enumerate(boxes):
        rows.append({"grid": grid, "crop_index": i, "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], "crop_w": b[2] - b[0], "crop_h": b[3] - b[1]})
    img.close()
    return stitch_regular_grid(local_scores, grid), rows


def fuse_global_local(global37: np.ndarray, local_map: np.ndarray, grid: int) -> np.ndarray:
    side = SIDE * grid
    global_up = torch_resize(global37, (side, side), align_corners=True)
    return (0.5 * global_up + 0.5 * np.asarray(local_map, dtype=np.float32)).astype(np.float32)


def image_metric_row(meta: dict[str, Any], variant: str, score: np.ndarray) -> dict[str, Any]:
    mask = meta["mask"]
    full = torch_resize(score, mask.shape, align_corners=True)
    return {
        "source": SOURCE,
        "dataset": TARGET,
        "category": meta["category"],
        "image_index": meta["image_index"],
        "img_path": meta["img_path"],
        "variant": variant,
        "label": int(meta["label"]),
        "score_resolution": f"{score.shape[0]}x{score.shape[1]}",
        "image_score": top1_image_score(score),
        "image_P_AP": safe_ap(mask, full),
        "image_P_AUROC": safe_auc(mask, full),
        "anomaly_pixels": int(np.asarray(mask, dtype=bool).sum()),
    }


def evaluate_maps(maps: dict[str, list[np.ndarray]], metas: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows: list[dict[str, Any]] = []
    cat_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        print(f"[LRP-SCALE] metrics {variant}", flush=True)
        by_cat: dict[str, dict[str, list[Any]]] = {}
        for i, meta in enumerate(metas):
            d = by_cat.setdefault(meta["category"], {"labels": [], "image_scores": [], "masks": [], "maps": []})
            d["labels"].append(meta["label"])
            d["image_scores"].append(top1_image_score(maps[variant][i]))
            d["masks"].append(meta["mask"])
            d["maps"].append(np.asarray(maps[variant][i], dtype=np.float32))
        pg_cat_rows = []
        fr_cat_by_protocol = {p: [] for p in PROTOCOLS}
        for cat, vals in sorted(by_cat.items()):
            pg = pg_metrics(vals["labels"], vals["image_scores"], vals["masks"], vals["maps"])
            pg_row = {"source": SOURCE, "dataset": TARGET, "category": cat, "variant": variant, "protocol": "PG", "aggregation": "CATEGORY", "score_resolution": f"{vals['maps'][0].shape[0]}x{vals['maps'][0].shape[1]}"}
            for met, val in pg.items():
                pg_row[f"PG_{met.replace('-', '_')}"] = val
            cat_rows.append(pg_row)
            pg_cat_rows.append(pg_row)
            for pkey, pcfg in PROTOCOLS.items():
                scores = [torch_resize(s, m.shape, align_corners=pcfg["align_corners"]) for s, m in zip(vals["maps"], vals["masks"])]
                fr = summarize_category(vals["labels"], vals["image_scores"], vals["masks"], scores)
                fr_row = {"source": SOURCE, "dataset": TARGET, "category": cat, "variant": variant, "protocol": pkey, "protocol_name": pcfg["name"], "aggregation": "CATEGORY", "score_resolution": f"{vals['maps'][0].shape[0]}x{vals['maps'][0].shape[1]}"}
                for met, val in fr.items():
                    fr_row[f"{pcfg['prefix']}_{met.replace('-', '_')}"] = val
                cat_rows.append(fr_row)
                fr_cat_by_protocol[pkey].append(fr_row)
        raw = {"source": SOURCE, "dataset": TARGET, "variant": variant, "aggregation": "CATEGORY_MACRO", "score_resolution": f"{maps[variant][0].shape[0]}x{maps[variant][0].shape[1]}"}
        for met in METRICS:
            vals = np.asarray([fnum(r, f"PG_{met.replace('-', '_')}") for r in pg_cat_rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            raw[f"PG_{met.replace('-', '_')}"] = float(vals.mean()) if vals.size else float("nan")
        for pkey, pcfg in PROTOCOLS.items():
            for met in METRICS:
                vals = np.asarray([fnum(r, f"{pcfg['prefix']}_{met.replace('-', '_')}") for r in fr_cat_by_protocol[pkey]], dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                raw[f"{pcfg['prefix']}_{met.replace('-', '_')}"] = float(vals.mean()) if vals.size else float("nan")
        raw_rows.append(raw)
    return raw_rows, cat_rows


def morphology_rows(metas: list[dict[str, Any]], cat_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    img_morph = []
    for meta in metas:
        row = {"source": SOURCE, "dataset": TARGET, "category": meta["category"], "image_index": meta["image_index"], "label": meta["label"], **mask_morphology(meta["mask"])}
        img_morph.append(row)
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for r in img_morph:
        by_cat.setdefault(r["category"], []).append(r)
    mcat = []
    for cat, rows in sorted(by_cat.items()):
        pos = [r for r in rows if int(r["label"]) == 1]
        out = {"source": SOURCE, "dataset": TARGET, "category": cat}
        for key in ("num_anomaly_patches", "anomaly_patch_ratio", "thinness", "boundary_to_area_ratio"):
            vals = [fnum(r, key) for r in pos if np.isfinite(fnum(r, key))]
            out[f"{key}_mean"] = float(np.mean(vals)) if vals else float("nan")
        mcat.append(out)
    perf = {(r["category"], r["variant"]): r for r in cat_rows if r.get("protocol") == "M"}
    deltas = []
    vals_thin = [fnum(r, "thinness_mean") for r in mcat if np.isfinite(fnum(r, "thinness_mean"))]
    vals_patch = [fnum(r, "num_anomaly_patches_mean") for r in mcat if np.isfinite(fnum(r, "num_anomaly_patches_mean")) and fnum(r, "num_anomaly_patches_mean") > 0]
    thin_thr = float(np.percentile(vals_thin, 75)) if vals_thin else float("nan")
    small_thr = 8.0
    large_thr = float(np.percentile(vals_patch, 75)) if vals_patch else float("nan")
    for m in mcat:
        cat = m["category"]
        b = perf.get((cat, "E0_2x2_LRP"))
        e = perf.get((cat, "E1_3x3_LRP"))
        if not (b and e):
            continue
        row = dict(m)
        for met in ("P_AUROC", "P_AP", "P_F1max", "PRO", "G_pixel", "I_AUROC", "I_AP"):
            row[f"E0_M_FR_{met}"] = fnum(b, f"M_FR_{met}")
            row[f"E1_M_FR_{met}"] = fnum(e, f"M_FR_{met}")
            row[f"Delta_M_FR_{met}"] = fnum(e, f"M_FR_{met}") - fnum(b, f"M_FR_{met}")
        row["group_small"] = fnum(m, "num_anomaly_patches_mean") <= small_thr
        row["group_thin"] = np.isfinite(thin_thr) and fnum(m, "thinness_mean") >= thin_thr
        row["group_large"] = np.isfinite(large_thr) and fnum(m, "num_anomaly_patches_mean") >= large_thr
        row["group_non_thin"] = not bool(row["group_thin"])
        deltas.append(row)
    groups = []
    for name, pred in (
        ("all", lambda r: True),
        ("small", lambda r: bool(r["group_small"])),
        ("thin", lambda r: bool(r["group_thin"])),
        ("small_or_thin", lambda r: bool(r["group_small"]) or bool(r["group_thin"])),
        ("large", lambda r: bool(r["group_large"])),
        ("non_thin", lambda r: bool(r["group_non_thin"])),
    ):
        part = [r for r in deltas if pred(r)]
        vals = [fnum(r, "Delta_M_FR_P_AP") for r in part if np.isfinite(fnum(r, "Delta_M_FR_P_AP"))]
        groups.append({"source": SOURCE, "dataset": TARGET, "group": name, "categories": len(part), "mean_delta_M_FR_P_AP": float(np.mean(vals)) if vals else float("nan"), "improved": int(sum(v > 0 for v in vals)), "total": len(vals)})
    return deltas, groups


def save_visualizations(maps: dict[str, list[np.ndarray]], metas: list[dict[str, Any]], image_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import matplotlib.pyplot as plt

    out_dir = OUT / "visualizations"
    candidates = [r for r in image_rows if r["variant"] == "E1_3x3_LRP" and int(r["label"]) == 1 and np.isfinite(fnum(r, "image_P_AP"))]
    e0_by = {(r["image_index"], r["variant"]): r for r in image_rows}
    ranked = []
    for r in candidates:
        b = e0_by.get((r["image_index"], "E0_2x2_LRP"))
        if b:
            rr = dict(r)
            rr["delta_image_P_AP"] = fnum(r, "image_P_AP") - fnum(b, "image_P_AP")
            ranked.append(rr)
    ranked = sorted(ranked, key=lambda x: fnum(x, "delta_image_P_AP"), reverse=True)[:12]
    rows = []
    for rr in ranked:
        i = int(rr["image_index"])
        meta = metas[i]
        img = Image.open(meta["img_path"]).convert("RGB")
        mask = meta["mask"]
        s0 = torch_resize(maps["E0_2x2_LRP"][i], mask.shape, align_corners=True)
        s1 = torch_resize(maps["E1_3x3_LRP"][i], mask.shape, align_corners=True)
        flat = np.concatenate([s0.reshape(-1), s1.reshape(-1)])
        vmin, vmax = float(np.percentile(flat, 1)), float(np.percentile(flat, 99))
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        axes[0].imshow(img)
        axes[0].set_title("Input")
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("GT")
        axes[2].imshow(img)
        axes[2].imshow(s0, cmap="magma", alpha=0.58, vmin=vmin, vmax=vmax)
        axes[2].set_title("2x2 LRP")
        axes[3].imshow(img)
        axes[3].imshow(s1, cmap="magma", alpha=0.58, vmin=vmin, vmax=vmax)
        axes[3].set_title("3x3 LRP")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        out_path = out_dir / f"{i:04d}_{meta['category']}_delta_{fnum(rr,'delta_image_P_AP'):+.3f}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        img.close()
        rows.append({"image_index": i, "category": meta["category"], "img_path": meta["img_path"], "delta_image_P_AP": rr["delta_image_P_AP"], "visualization": str(out_path)})
    return rows


def build_maps(smoke: bool) -> tuple[dict[str, list[np.ndarray]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocess, _ = get_transform(argparse.Namespace(image_size=IMAGE_SIZE))
    clip_model, *_ = load_components(str(CHECKPOINT), device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)
    model_info = load_mvtec_source()
    cells, _audits = protocol_cells(SOURCE, device, smoke)
    cell = next(c for c in cells if c[1] == TARGET)
    _cell_id, target, dataset, refs, patch, kind = cell
    n = min(24, patch.shape[0]) if smoke else patch.shape[0]
    maps = {v: [] for v in VARIANTS}
    metas = []
    crop_rows = []
    sanity_rows = []
    for pos in range(n):
        if pos % 25 == 0:
            print(f"[LRP-SCALE] {SOURCE}->{TARGET} image {pos}/{n}", flush=True)
        ref = refs[pos]
        item = item_at(dataset, kind, ref, pos)
        cat, label, path = item_meta(item, kind, ref, target)
        mask = mask_np(item)
        global37 = score_feature_grid(model_info, patch[pos], device)
        local74, crops2 = local_lrp_map(model_info, clip_model, preprocess, path, 2, device)
        local111, crops3 = local_lrp_map(model_info, clip_model, preprocess, path, 3, device)
        final74 = fuse_global_local(global37, local74, 2)
        final111 = fuse_global_local(global37, local111, 3)
        maps["E0_2x2_LRP"].append(final74.astype(np.float16))
        maps["E1_3x3_LRP"].append(final111.astype(np.float16))
        metas.append({"category": cat, "label": label, "mask": mask, "img_path": path, "image_index": pos})
        for r in crops2 + crops3:
            crop_rows.append({"source": SOURCE, "dataset": TARGET, "category": cat, "image_index": pos, "img_path": path, **r})
        sanity_rows.append({
            "source": SOURCE,
            "dataset": TARGET,
            "category": cat,
            "image_index": pos,
            "global_shape": "37x37",
            "local2_shape": f"{local74.shape[0]}x{local74.shape[1]}",
            "final2_shape": f"{final74.shape[0]}x{final74.shape[1]}",
            "local3_shape": f"{local111.shape[0]}x{local111.shape[1]}",
            "final3_shape": f"{final111.shape[0]}x{final111.shape[1]}",
            "local2_finite": bool(np.isfinite(local74).all()),
            "local3_finite": bool(np.isfinite(local111).all()),
            "final2_finite": bool(np.isfinite(final74).all()),
            "final3_finite": bool(np.isfinite(final111).all()),
        })
    del clip_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return maps, metas, crop_rows, sanity_rows


def report(raw: list[dict[str, Any]], cat_delta: list[dict[str, Any]], group_rows: list[dict[str, Any]], sanity_rows: list[dict[str, Any]], runtime: float) -> str:
    by = {r["variant"]: r for r in raw}
    e0 = by.get("E0_2x2_LRP", {})
    e1 = by.get("E1_3x3_LRP", {})
    delta_pap = fnum(e1.get("M_FR_P_AP")) - fnum(e0.get("M_FR_P_AP"))
    delta_aupro = fnum(e1.get("M_FR_PRO")) - fnum(e0.get("M_FR_PRO"))
    improved_cats = sum(1 for r in cat_delta if fnum(r, "Delta_M_FR_P_AP") > 0)
    total_cats = sum(1 for r in cat_delta if np.isfinite(fnum(r, "Delta_M_FR_P_AP")))
    small = next((r for r in group_rows if r["group"] == "small_or_thin"), None)
    outcome = "SCALE_SIGNAL_SUPPORTED" if delta_pap > 0.5 and improved_cats >= max(1, total_cats // 2) else "SCALE_SIGNAL_NOT_CLEAR"
    lines = [
        "# LRP-SCALE VisA Diagnostic",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "This is an exploratory no-training diagnostic. It freezes the Full MVTec-source CES/NTR checkpoint and changes only the local observation grid from the formal 2x2 LRP to a single 3x3 non-overlapping LRP configuration.",
        "",
        "## Registration / Numeric Check",
        "",
        f"- Smoke/full run images checked: `{len(sanity_rows)}`.",
        "- Expected 2x2 local/final resolution: `74x74`.",
        "- Expected 3x3 local/final resolution: `111x111`.",
        f"- All finite: `{all(bool(r['final2_finite']) and bool(r['final3_finite']) for r in sanity_rows)}`.",
        "",
        "## Full VisA Metrics",
        "",
        "| Variant | I-AUROC | I-AP | P-AUROC | P-AP | P-F1max | AUPRO |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        r = by.get(variant, {})
        lines.append(f"| {variant} | {fnum(r.get('M_FR_I_AUROC')):.4f} | {fnum(r.get('M_FR_I_AP')):.4f} | {fnum(r.get('M_FR_P_AUROC')):.4f} | {fnum(r.get('M_FR_P_AP')):.4f} | {fnum(r.get('M_FR_P_F1max')):.4f} | {fnum(r.get('M_FR_PRO')):.4f} |")
    lines += [
        "",
        f"- Delta 3x3 - 2x2 P-AP: `{delta_pap:+.4f}`.",
        f"- Delta 3x3 - 2x2 AUPRO: `{delta_aupro:+.4f}`.",
        f"- Improved categories by P-AP: `{improved_cats}/{total_cats}`.",
        "",
        "## Morphology Stratification",
        "",
        "| Group | Categories | Mean Delta P-AP | Improved/Total |",
        "|---|---:|---:|---:|",
    ]
    for r in group_rows:
        lines.append(f"| {r['group']} | {r['categories']} | {fnum(r,'mean_delta_M_FR_P_AP'):+.4f} | {int(r['improved'])}/{int(r['total'])} |")
    if small:
        lines += ["", f"- Small/thin category mean Delta P-AP: `{fnum(small, 'mean_delta_M_FR_P_AP'):+.4f}`."]
    lines += [
        "",
        "## Interpretation",
        "",
        "- Evidence for observation-scale sensitivity is based only on the fixed 3x3 diagnostic above.",
        "- Any improvement can indicate that finer local observation gives extra localization evidence, but it does not isolate whether the cause is crop scale, interpolation, source-target morphology, or score calibration.",
        "- If the 3x3 setting has no clear gain, the protocol stops and does not justify further split-count expansion.",
        "",
        f"Runtime: `{runtime:.1f}s`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = ".smoke" if args.smoke else ""
    maps, metas, crop_rows, sanity_rows = build_maps(args.smoke)
    raw, cats = evaluate_maps(maps, metas)
    image_rows = []
    for i, meta in enumerate(metas):
        for variant in VARIANTS:
            image_rows.append(image_metric_row(meta, variant, np.asarray(maps[variant][i], dtype=np.float32)))
    cat_delta, group_rows = morphology_rows(metas, cats)
    viz_rows = save_visualizations(maps, metas, image_rows) if not args.smoke else []
    write_csv(OUT / f"RAW_METRICS{suffix}.csv", raw)
    write_csv(OUT / f"PER_CATEGORY_METRICS{suffix}.csv", cats)
    write_csv(OUT / f"CATEGORY_DELTA_MORPHOLOGY{suffix}.csv", cat_delta)
    write_csv(OUT / f"MORPHOLOGY_GROUP_SUMMARY{suffix}.csv", group_rows)
    write_csv(OUT / f"IMAGE_DIAGNOSTICS{suffix}.csv", image_rows)
    write_csv(OUT / f"CROP_REGISTRATION_AUDIT{suffix}.csv", crop_rows)
    write_csv(OUT / f"NUMERIC_SANITY{suffix}.csv", sanity_rows)
    if viz_rows:
        write_csv(OUT / "REPRESENTATIVE_MAPS.csv", viz_rows)
    runtime = time.time() - start
    (OUT / f"LRP_SCALE_VISA_REPORT{suffix}.md").write_text(report(raw, cat_delta, group_rows, sanity_rows, runtime), encoding="utf-8")
    write_json(OUT / f"OUTCOME{suffix}.json", {"source": SOURCE, "target": TARGET, "smoke": args.smoke, "runtime_sec": runtime})
    print(f"[LRP-SCALE] done smoke={args.smoke} runtime={runtime:.1f}s out={OUT}", flush=True)


if __name__ == "__main__":
    main()
