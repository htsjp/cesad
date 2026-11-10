from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_final_cross_dataset_benchmark import load_visa_source, protocol_cells
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import (
    build_maps as build_ces_maps,
    item_at,
    item_meta,
    mask_np,
    torch_resize,
)
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import load_components as load_clip_components
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import CHECKPOINT, top1_image_score
from experiments.phase_xxxviii_visual_state_representation import IMAGE_SIZE
from utils.transforms import get_transform

import experiments.phase_fbclip_pretrained_headroom_audit as fb
import experiments.phase_visualad_exact_protocol_audit as va


OUT = Path("results/phase_medical_context_probe")
TARGETS = ("CVC-ClinicDB", "CVC-ColonDB", "Kvasir")
SOURCE = "Full VisA"
SEED = 0
SAMPLE_PER_DATASET = 10
SMOKE_PER_DATASET = 2
METHODS = ("CESAD-Full", "FB-CLIP", "VisualAD")


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


def fnum(v: Any) -> float:
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


def fixed_sample_indices(refs: list[Any], n: int, seed_offset: int) -> list[int]:
    rng = np.random.default_rng(SEED + seed_offset)
    total = len(refs)
    if total <= n:
        return list(range(total))
    return sorted(int(x) for x in rng.choice(total, size=n, replace=False))


def get_cell(target: str, device: str, smoke: bool = False):
    cells, _audits = protocol_cells(SOURCE, device, smoke)
    for cell_id, tname, dataset, refs, patch, kind in cells:
        if tname == target:
            return cell_id, dataset, list(refs), patch, kind
    raise KeyError(f"No protocol cell for {SOURCE}->{target}")


def local_contrast_map(q74: np.ndarray) -> np.ndarray:
    x = torch.tensor(np.asarray(q74, dtype=np.float32)).view(1, 1, *q74.shape)
    pad = F.pad(x, (2, 2, 2, 2), mode="reflect")
    b = F.avg_pool2d(pad, kernel_size=5, stride=1).view(*q74.shape).numpy().astype(np.float32)
    return (np.asarray(q74, dtype=np.float32) + 0.5 * (np.asarray(q74, dtype=np.float32) - b)).astype(np.float32)


def resize_to_mask(score: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return torch_resize(np.asarray(score, dtype=np.float32), mask.shape, align_corners=True)


def top_fraction_coverage(mask: np.ndarray, score: np.ndarray, frac: float = 0.01) -> float:
    m = np.asarray(mask, dtype=bool)
    s = np.asarray(score, dtype=np.float32).reshape(-1)
    y = m.reshape(-1)
    k = max(1, int(np.ceil(s.size * frac)))
    idx = np.argpartition(s, s.size - k)[s.size - k :]
    return float(y[idx].sum() / max(1, y.sum()))


def boundary_stats(mask: np.ndarray, score: np.ndarray) -> dict[str, float]:
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return {"boundary_mean": float("nan"), "interior_mean": float("nan"), "background_p95": float("nan")}
    dil = binary_dilation(m, iterations=2)
    ero = binary_erosion(m, iterations=2)
    boundary = dil & ~ero
    interior = ero
    bg = ~m
    s = np.asarray(score, dtype=np.float32)
    return {
        "boundary_mean": float(np.mean(s[boundary])) if np.any(boundary) else float("nan"),
        "interior_mean": float(np.mean(s[interior])) if np.any(interior) else float("nan"),
        "background_p95": float(np.percentile(s[bg], 95)) if np.any(bg) else float("nan"),
    }


def metric_row(dataset: str, image_id: int, category: str, img_path: str, method: str, variant: str, mask: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    score_fr = resize_to_mask(score, mask) if score.shape != mask.shape else np.asarray(score, dtype=np.float32)
    b = boundary_stats(mask, score_fr)
    return {
        "dataset": dataset,
        "image_index": image_id,
        "category": category,
        "img_path": img_path,
        "method": method,
        "variant": variant,
        "P_AP": safe_ap(mask, score_fr),
        "P_AUROC": safe_auc(mask, score_fr),
        "top1pct_anomaly_coverage": top_fraction_coverage(mask, score_fr, 0.01),
        "top5pct_anomaly_coverage": top_fraction_coverage(mask, score_fr, 0.05),
        **b,
    }


def save_overlay(path: Path, image_path: str, mask: np.ndarray, maps: dict[str, np.ndarray]) -> None:
    import matplotlib.pyplot as plt

    img = Image.open(image_path).convert("RGB")
    fig, axes = plt.subplots(1, 2 + len(maps), figsize=(3.0 * (2 + len(maps)), 3.0))
    axes[0].imshow(img)
    axes[0].set_title("Input")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("GT")
    all_vals = []
    for score in maps.values():
        all_vals.append(score.reshape(-1))
    flat = np.concatenate(all_vals)
    vmin, vmax = float(np.percentile(flat, 1)), float(np.percentile(flat, 99))
    for ax, (name, score) in zip(axes[2:], maps.items()):
        ax.imshow(img)
        ax.imshow(score, cmap="magma", alpha=0.55, vmin=vmin, vmax=vmax)
        ax.set_title(name)
    for ax in axes:
        ax.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


@torch.no_grad()
def ces_maps_for_refs(target: str, refs_pos: list[int], device: str):
    cell_id, dataset, refs, patch, kind = get_cell(target, device)
    selected_refs = [refs[i] for i in refs_pos]
    selected_patch = patch[refs_pos]
    preprocess, _ = get_transform(argparse.Namespace(image_size=IMAGE_SIZE))
    clip_model, *_ = load_clip_components(str(CHECKPOINT), device)
    model_info = load_visa_source(device)
    maps, metas, _scale = build_ces_maps(model_info, clip_model, preprocess, selected_patch, selected_refs, dataset, kind, target, device)
    del clip_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cell_id, dataset, selected_refs, maps, metas


@torch.no_grad()
def fb_maps_for_refs(target: str, refs_pos: list[int], device: str):
    cell_id, dataset, refs, _patch, kind = get_cell(target, device)
    selected_refs = [refs[i] for i in refs_pos]
    model, prompt, args, _info = fb.load_fb_model(SOURCE, device)
    args.test_batch_size = 2
    preprocess, _ = fb.fb_get_transform(args)
    maps = {fb.VARIANT: []}
    metas = []
    batch, batch_meta = [], []
    for pos, ref in enumerate(selected_refs):
        item = item_at(dataset, kind, ref, pos)
        cat, label, path = item_meta(item, kind, ref, target)
        img = Image.open(path).convert("RGB")
        batch.append(preprocess(img))
        batch_meta.append((cat, label, path, mask_np(item)))
        if len(batch) == args.test_batch_size or pos == len(selected_refs) - 1:
            images = torch.stack(batch, dim=0).to(device)
            preds = fb.fb_predict_batch(model, prompt, args, images)
            for pred, meta in zip(preds, batch_meta):
                maps[fb.VARIANT].append(pred)
                metas.append({"category": meta[0], "label": meta[1], "mask": meta[3], "img_path": meta[2]})
            batch, batch_meta = [], []
    del model, prompt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return maps, metas


@torch.no_grad()
def visualad_maps_for_refs(target: str, refs_pos: list[int], device: torch.device):
    official = va.ensure_official_imports()
    model, layer_transforms, cross_attn, _checkpoint = va.load_visualad_bundle(SOURCE, "E1c_FullVisualAD", official, device)
    preprocess = va.official_transform(official)
    cell_id, _t, dataset, refs, kind = va.find_cell(SOURCE, target, device, False)
    selected_refs = [refs[i] for i in refs_pos]
    maps = {"E1c_FullVisualAD": []}
    metas = []
    for pos, ref in enumerate(selected_refs):
        item = item_at(dataset, kind, ref, pos)
        cat, label, _path_from_meta = item_meta(item, kind, ref, target)
        meta_ref = ref if kind != "source_dataset" else None
        path = va.resolve_image_path(item, dataset, meta_ref)
        batch = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        final, _stats = va.infer_batch(model, layer_transforms, cross_attn, batch)
        maps["E1c_FullVisualAD"].append(gaussian_filter(final[0], sigma=va.SIGMA).astype(np.float32))
        metas.append({"category": cat, "label": label, "mask": mask_np(item), "img_path": str(path)})
    del model, layer_transforms, cross_attn
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return maps, metas


def run(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    torch_device = torch.device(device)
    preflight = []
    preflight.append({"resource": "CESAD-Full", "status": "READY", "checkpoint": str(CHECKPOINT), "note": "Uses existing Full VisA source state and LRP runner."})
    preflight.append({"resource": "FB-CLIP", "status": "READY" if fb.WEIGHTS[SOURCE].exists() else "MISSING", "checkpoint": str(fb.WEIGHTS[SOURCE])})
    try:
        _ = va.checkpoint_path(SOURCE)
        va_status = "READY"
    except Exception as exc:
        va_status = f"MISSING: {exc}"
    preflight.append({"resource": "VisualAD", "status": va_status, "checkpoint": str(va.checkpoint_path(SOURCE)) if va_status == "READY" else ""})

    audit_rows, diag_rows, local_rows, sample_rows = [], [], [], []
    for ti, target in enumerate(TARGETS):
        cell_id, dataset, refs, _patch, kind = get_cell(target, device)
        smoke_pos = fixed_sample_indices(refs, SMOKE_PER_DATASET, 100 + ti)
        sample_pos = fixed_sample_indices(refs, SAMPLE_PER_DATASET, 200 + ti)
        for phase, positions in [("smoke", smoke_pos), ("diagnostic_30", sample_pos)]:
            print(f"[MCG] {phase} {target} n={len(positions)}", flush=True)
            cell_id, _dataset, selected_refs, ces_maps, ces_metas = ces_maps_for_refs(target, positions, device)
            fb_maps, fb_metas = fb_maps_for_refs(target, positions, device)
            va_maps, va_metas = visualad_maps_for_refs(target, positions, torch_device)
            for i, meta in enumerate(ces_metas):
                image_id = int(positions[i])
                sample_rows.append({"phase": phase, "dataset": target, "sample_order": i, "protocol_position": image_id, "category": meta["category"], "img_path": meta["img_path"]})
                mask = meta["mask"]
                ces_q = np.asarray(ces_maps["E2_GlobalLocalZoomPyramid"][i], dtype=np.float32)
                lc_q = local_contrast_map(ces_q)
                maps_for_vis = {
                    "CES": resize_to_mask(ces_q, mask),
                    "FB-CLIP": resize_to_mask(fb_maps[fb.VARIANT][i], mask),
                    "VisualAD": resize_to_mask(va_maps["E1c_FullVisualAD"][i], mask),
                    "CES+LC": resize_to_mask(lc_q, mask),
                }
                if phase == "smoke" or i < 3:
                    save_overlay(OUT / "visualizations" / f"{phase}_{target}_{i:02d}.png", meta["img_path"], mask, maps_for_vis)
                method_scores = [
                    ("CESAD-Full", "E2_GlobalLocalZoomPyramid", ces_q),
                    ("FB-CLIP", fb.VARIANT, fb_maps[fb.VARIANT][i]),
                    ("VisualAD", "E1c_FullVisualAD", va_maps["E1c_FullVisualAD"][i]),
                    ("CESAD-Full+FixedLocalContrast", "Q_plus_0p5_Q_minus_avg5", lc_q),
                ]
                for method, variant, score in method_scores:
                    row = metric_row(target, image_id, meta["category"], meta["img_path"], method, variant, mask, score)
                    row["phase"] = phase
                    diag_rows.append(row)
                original = metric_row(target, image_id, meta["category"], meta["img_path"], "CESAD-Full", "E2_GlobalLocalZoomPyramid", mask, ces_q)
                local = metric_row(target, image_id, meta["category"], meta["img_path"], "CESAD-Full+FixedLocalContrast", "Q_plus_0p5_Q_minus_avg5", mask, lc_q)
                local_rows.append({
                    "phase": phase,
                    "dataset": target,
                    "image_index": image_id,
                    "category": meta["category"],
                    "CES_P_AP": original["P_AP"],
                    "LocalContrast_P_AP": local["P_AP"],
                    "Delta_P_AP": fnum(local["P_AP"]) - fnum(original["P_AP"]),
                    "CES_top1pct_coverage": original["top1pct_anomaly_coverage"],
                    "LocalContrast_top1pct_coverage": local["top1pct_anomaly_coverage"],
                    "Delta_top1pct_coverage": fnum(local["top1pct_anomaly_coverage"]) - fnum(original["top1pct_anomaly_coverage"]),
                })
                # Pairwise complementarity at a fixed top-1% pixel budget.
                ces_fr = maps_for_vis["CES"].reshape(-1)
                mask_flat = mask.reshape(-1).astype(bool)
                k = max(1, int(np.ceil(mask_flat.size * 0.01)))
                ces_top = np.zeros_like(mask_flat, dtype=bool)
                ces_top[np.argpartition(ces_fr, ces_fr.size - k)[ces_fr.size - k :]] = True
                for method_name in ("FB-CLIP", "VisualAD"):
                    s = maps_for_vis[method_name].reshape(-1)
                    top = np.zeros_like(mask_flat, dtype=bool)
                    top[np.argpartition(s, s.size - k)[s.size - k :]] = True
                    audit_rows.append({
                        "phase": phase,
                        "dataset": target,
                        "image_index": image_id,
                        "comparison": f"CESAD_vs_{method_name}",
                        "pixel_budget_fraction": 0.01,
                        "ces_top_anomaly_coverage": float((ces_top & mask_flat).sum() / max(1, mask_flat.sum())),
                        "specialist_top_anomaly_coverage": float((top & mask_flat).sum() / max(1, mask_flat.sum())),
                        "specialist_correct_ces_missed_pixels": int((top & mask_flat & ~ces_top).sum()),
                        "ces_unique_correct_pixels": int((ces_top & mask_flat & ~top).sum()),
                        "shared_correct_pixels": int((top & mask_flat & ces_top).sum()),
                    })

    write_csv(OUT / "MEDICAL_PAIRED_RAW_MAP_AUDIT.csv", audit_rows)
    write_csv(OUT / "MEDICAL_ERROR_DIAGNOSTICS.csv", diag_rows)
    write_csv(OUT / "LOCAL_CONTEXT_PROBE_METRICS.csv", local_rows)
    write_csv(OUT / "SAMPLED_IMAGES.csv", sample_rows)
    write_csv(OUT / "PREFLIGHT_RESOURCES.csv", preflight)
    (OUT / "PREFLIGHT.md").write_text(
        "# Medical Context Probe Preflight\n\n"
        f"- Source: `{SOURCE}`.\n"
        f"- Targets: `{', '.join(TARGETS)}`.\n"
        "- Smoke: 2 fixed-seed images per dataset.\n"
        "- Diagnostic: 10 fixed-seed images per dataset, 30 total.\n"
        "- No model training, no target tuning, no hyperparameter search.\n"
        "- CES, FB-CLIP, and VisualAD maps are generated on the same sampled images.\n",
        encoding="utf-8",
    )

    # Industrial full regression is intentionally not run here; write current status.
    write_csv(
        OUT / "INDUSTRIAL_REGRESSION_CHECK.csv",
        [
            {
                "status": "NOT_RUN",
                "reason": "This low-cost probe evaluates fixed local contrast on 30 medical samples only. Full industrial regression would require additional map generation; do not infer no-regression from this file.",
                "required_pairs": "Full MVTec->VisA; Full VisA->BTAD; Full VisA->MPDD",
            }
        ],
    )
    write_report(diag_rows, local_rows, audit_rows)
    write_json(OUT / "OUTCOME.json", {"outcome": "MIXED_EVIDENCE", "recommendation": "INSUFFICIENT_EVIDENCE", "sample_count": len(sample_rows)})
    print(OUT)


def mean_by(rows: list[dict[str, Any]], key_method: str, phase: str = "diagnostic_30") -> dict[str, float]:
    vals = [fnum(r["P_AP"]) for r in rows if r.get("phase") == phase and r.get("method") == key_method]
    vals = [v for v in vals if np.isfinite(v)]
    return {"mean_P_AP": float(np.mean(vals)) if vals else float("nan"), "n": len(vals)}


def write_report(diag_rows: list[dict[str, Any]], local_rows: list[dict[str, Any]], audit_rows: list[dict[str, Any]]) -> None:
    ces = mean_by(diag_rows, "CESAD-Full")
    fbm = mean_by(diag_rows, "FB-CLIP")
    vam = mean_by(diag_rows, "VisualAD")
    lcm = mean_by(diag_rows, "CESAD-Full+FixedLocalContrast")
    lc_deltas = [fnum(r["Delta_P_AP"]) for r in local_rows if r.get("phase") == "diagnostic_30"]
    lc_deltas = [x for x in lc_deltas if np.isfinite(x)]
    pair_fb = [r for r in audit_rows if r.get("phase") == "diagnostic_30" and "FB-CLIP" in r.get("comparison", "")]
    pair_va = [r for r in audit_rows if r.get("phase") == "diagnostic_30" and "VisualAD" in r.get("comparison", "")]

    def avg(rows: list[dict[str, Any]], k: str) -> float:
        vals = [fnum(r[k]) for r in rows]
        vals = [x for x in vals if np.isfinite(x)]
        return float(np.mean(vals)) if vals else float("nan")

    fb_gain = avg(pair_fb, "specialist_top_anomaly_coverage") - avg(pair_fb, "ces_top_anomaly_coverage")
    va_gain = avg(pair_va, "specialist_top_anomaly_coverage") - avg(pair_va, "ces_top_anomaly_coverage")
    outcome = "MIXED_EVIDENCE"
    recommendation = "INSUFFICIENT_EVIDENCE"
    if lc_deltas and float(np.mean(lc_deltas)) > 1.0 and min(lc_deltas) > -5:
        outcome = "CONTEXT_SIGNAL_SUPPORTED"
        recommendation = "YES"
    elif lc_deltas and float(np.mean(lc_deltas)) <= 0:
        outcome = "CONTEXT_SIGNAL_NOT_SUPPORTED"
        recommendation = "NO"

    lines = [
        "# Medical Context Probe Report",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "This is a seed-0, no-training, no-tuning diagnostic on 30 fixed sampled medical images. It is not a replacement for full-dataset formal evaluation.",
        "",
        "## Sample Mean Pixel AP",
        "",
        "| Method | n | mean sample P-AP |",
        "|---|---:|---:|",
        f"| CESAD-Full | {ces['n']} | {ces['mean_P_AP']:.4f} |",
        f"| FB-CLIP | {fbm['n']} | {fbm['mean_P_AP']:.4f} |",
        f"| VisualAD | {vam['n']} | {vam['mean_P_AP']:.4f} |",
        f"| CESAD-Full + Fixed Local Contrast | {lcm['n']} | {lcm['mean_P_AP']:.4f} |",
        "",
        "## Pairwise Top-1% Budget Coverage",
        "",
        f"- FB-CLIP minus CESAD anomaly coverage: {fb_gain:+.4f}",
        f"- VisualAD minus CESAD anomaly coverage: {va_gain:+.4f}",
        "",
        "## Fixed Local Contrast",
        "",
        f"- Mean sample Delta P-AP vs CESAD-Full: {float(np.mean(lc_deltas)) if lc_deltas else float('nan'):+.4f}",
        f"- Improved images: {sum(x > 0 for x in lc_deltas)}/{len(lc_deltas)}",
        "",
        "## Final Question",
        "",
        f"是否值得继续研究 Context-Conditioned Compact Experience States？ `{recommendation}`",
        "",
        "Caveat: industrial regression was not run in this low-cost probe; full industrial checks are required before any method claim.",
    ]
    (OUT / "MEDICAL_CONTEXT_PROBE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
