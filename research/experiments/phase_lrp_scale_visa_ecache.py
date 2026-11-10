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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.phase_lrp_scale_visa as scale
from experiments.phase_final_cross_dataset_benchmark import load_mvtec_source
from experiments.phase_mvtec_source_conditioning_control import (
    load_target_visa,
    local_patch_cache_for_mvtec_visa,
    resize_global_batch,
    score_patch_batch,
)
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import item_at, item_meta, mask_np
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import CHECKPOINT
from experiments.phase_xxxviii_visual_state_representation import IMAGE_SIZE, SIDE
from utils.transforms import get_transform


OUT = Path("results/phase_lrp_scale_visa/ecache_corrected")
SOURCE = "Full MVTec"
TARGET = "VisA"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def fnum(v: Any) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def score_ecache_2x2(source_info: dict[str, Any], v: np.ndarray, patch: np.ndarray, local_cache: np.ndarray, device: torch.device, start: int, end: int) -> np.ndarray:
    gb = score_patch_batch(source_info, v, np.asarray(patch[start:end], dtype=np.float32), device)
    lp = np.asarray(local_cache[start:end], dtype=np.float32)
    lb_flat = lp.reshape(-1, SIDE * SIDE, 1024)
    lb_scores = score_patch_batch(source_info, v, lb_flat, device).reshape(end - start, 4, SIDE, SIDE)
    local74 = np.zeros((end - start, SIDE * 2, SIDE * 2), dtype=np.float32)
    local74[:, :SIDE, :SIDE] = lb_scores[:, 0]
    local74[:, :SIDE, SIDE:] = lb_scores[:, 1]
    local74[:, SIDE:, :SIDE] = lb_scores[:, 2]
    local74[:, SIDE:, SIDE:] = lb_scores[:, 3]
    global74 = resize_global_batch(gb)
    return (0.5 * global74 + 0.5 * local74).astype(np.float32)


@torch.no_grad()
def run(smoke: bool = False) -> None:
    start_time = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_info = load_mvtec_source()
    v = source_info["v1"]
    dataset, refs, patch, kind = load_target_visa(device, smoke=smoke)
    n = min(24, len(refs)) if smoke else len(refs)
    refs = refs[:n]
    patch = patch[:n]
    local_cache = local_patch_cache_for_mvtec_visa(n, smoke=smoke)
    if local_cache is None:
        raise RuntimeError("Missing E-cache 2x2 local patch file: E:/MRAD_final_evidence_cache/local_patch_Full_MVTec__VisA.npy")

    preprocess, _ = get_transform(argparse.Namespace(image_size=IMAGE_SIZE))
    clip_model, *_ = load_components(str(CHECKPOINT), str(device))
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    maps = {"E0_2x2_LRP": [], "E1_3x3_LRP": []}
    metas: list[dict[str, Any]] = []
    crop_rows: list[dict[str, Any]] = []
    sanity_rows: list[dict[str, Any]] = []

    batch = 16
    for start in range(0, n, batch):
        end = min(n, start + batch)
        if start % 50 == 0:
            print(f"[LRP-SCALE-ECACHE] 2x2 E-cache score {start}/{n}", flush=True)
        e0 = score_ecache_2x2(source_info, v, patch, local_cache, device, start, end)
        maps["E0_2x2_LRP"].extend([x.astype(np.float16) for x in e0])

    for pos, ref in enumerate(refs):
        if pos % 25 == 0:
            print(f"[LRP-SCALE-ECACHE] 3x3 live local {pos}/{n}", flush=True)
        item = item_at(dataset, kind, ref, pos)
        cat, label, path = item_meta(item, kind, ref, TARGET)
        mask = mask_np(item)
        global37 = scale.score_feature_grid(source_info, patch[pos], str(device))
        local111, crops3 = scale.local_lrp_map(source_info, clip_model, preprocess, path, 3, str(device))
        final111 = scale.fuse_global_local(global37, local111, 3)
        maps["E1_3x3_LRP"].append(final111.astype(np.float16))
        metas.append({"category": cat, "label": label, "mask": mask, "img_path": path, "image_index": pos})
        for r in crops3:
            crop_rows.append({"source": SOURCE, "dataset": TARGET, "category": cat, "image_index": pos, "img_path": path, **r})
        sanity_rows.append({
            "source": SOURCE,
            "dataset": TARGET,
            "category": cat,
            "image_index": pos,
            "e0_source": "E:/MRAD_final_evidence_cache/local_patch_Full_MVTec__VisA.npy",
            "final2_shape": f"{maps['E0_2x2_LRP'][pos].shape[0]}x{maps['E0_2x2_LRP'][pos].shape[1]}",
            "local3_shape": f"{local111.shape[0]}x{local111.shape[1]}",
            "final3_shape": f"{final111.shape[0]}x{final111.shape[1]}",
            "final2_finite": bool(np.isfinite(np.asarray(maps["E0_2x2_LRP"][pos], dtype=np.float32)).all()),
            "final3_finite": bool(np.isfinite(final111).all()),
        })

    scale.OUT = OUT
    raw, cats = scale.evaluate_maps(maps, metas)
    image_rows = []
    for i, meta in enumerate(metas):
        for variant in scale.VARIANTS:
            image_rows.append(scale.image_metric_row(meta, variant, np.asarray(maps[variant][i], dtype=np.float32)))
    cat_delta, group_rows = scale.morphology_rows(metas, cats)
    viz_rows = scale.save_visualizations(maps, metas, image_rows) if not smoke else []

    suffix = ".smoke" if smoke else ""
    write_csv(OUT / f"RAW_METRICS{suffix}.csv", raw)
    write_csv(OUT / f"PER_CATEGORY_METRICS{suffix}.csv", cats)
    write_csv(OUT / f"CATEGORY_DELTA_MORPHOLOGY{suffix}.csv", cat_delta)
    write_csv(OUT / f"MORPHOLOGY_GROUP_SUMMARY{suffix}.csv", group_rows)
    write_csv(OUT / f"IMAGE_DIAGNOSTICS{suffix}.csv", image_rows)
    write_csv(OUT / f"CROP_REGISTRATION_AUDIT{suffix}.csv", crop_rows)
    write_csv(OUT / f"NUMERIC_SANITY{suffix}.csv", sanity_rows)
    if viz_rows:
        write_csv(OUT / "REPRESENTATIVE_MAPS.csv", viz_rows)

    runtime = time.time() - start_time
    report = scale.report(raw, cat_delta, group_rows, sanity_rows, runtime)
    report += "\n## Corrected Baseline Source\n\n`E0_2x2_LRP` is computed from the locked E-drive cache `E:/MRAD_final_evidence_cache/local_patch_Full_MVTec__VisA.npy`, which reproduces the historical formal 2x2 LRP artifact (`M_FR_P_AP ~= 21.4606`). `E1_3x3_LRP` is the only new observation-scale diagnostic and is generated with nine non-overlapping local crops.\n"
    (OUT / f"LRP_SCALE_VISA_REPORT{suffix}.md").write_text(report, encoding="utf-8")
    write_json(OUT / f"OUTCOME{suffix}.json", {"source": SOURCE, "target": TARGET, "smoke": smoke, "runtime_sec": runtime, "e0_cache": "E:/MRAD_final_evidence_cache/local_patch_Full_MVTec__VisA.npy"})
    print(f"[LRP-SCALE-ECACHE] done smoke={smoke} runtime={runtime:.1f}s out={OUT}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(smoke=args.smoke)


if __name__ == "__main__":
    main()
