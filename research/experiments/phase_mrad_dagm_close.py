import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_core_specialist_complementarity_oracle import pair_slug
from experiments.phase_dagm_visa_eval import (
    DAGMTargetDataset,
    DAGM_EVAL_SIDE,
    TARGET,
    evaluate_maps_grid,
    resize_score_grid,
)
from experiments.phase_mrad_dtd_close import (
    SOURCE,
    audit_assets,
    checkpoint_for_variant,
    method_name,
    protocol_rows,
)
from experiments.phase_mva_mrad_same_protocol_visa_factorization import (
    M1,
    M2,
    M3,
    MRAD_VARIANTS,
    OFFICIAL_DEFAULTS,
    load_mrad_bundle,
    mrad_infer_one,
    source_config,
)
from experiments.phase_xa_smoke import write_csv


OUT = ROOT / "results" / "paper_assets" / "mrad_dagm_close"
MAP_CACHE = OUT / "MAP_CACHE"


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


def load_dagm_metas(dataset, smoke=False):
    n = min(24, len(dataset)) if smoke else len(dataset)
    metas = []
    for i in range(n):
        item = dataset[i]
        metas.append({
            "category": item["cls_name"],
            "label": int(item["anomaly"]),
            "img_path": item["img_path"],
            "mask": item["img_mask"].numpy().astype(np.uint8),
        })
    return metas


def mrad_cache_path(variant):
    return MAP_CACHE / f"{pair_slug(SOURCE, TARGET)}__{variant}_grid{DAGM_EVAL_SIDE}.npz"


def save_cache(variant, maps, image_scores, retrieval_rows):
    path = mrad_cache_path(variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        maps=np.asarray(maps, dtype=np.float16),
        image_scores=np.asarray(image_scores, dtype=np.float32),
        retrieval_rows=np.asarray(retrieval_rows, dtype=object),
    )
    return path


def load_cache(variant):
    path = mrad_cache_path(variant)
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "maps": np.asarray(data["maps"], dtype=np.float32),
        "image_scores": np.asarray(data["image_scores"], dtype=np.float32),
        "retrieval_rows": list(data["retrieval_rows"]) if "retrieval_rows" in data.files else [],
        "path": path,
    }


def build_maps(variant, metas, device, smoke=False, force=False):
    cached = None if force or smoke else load_cache(variant)
    if cached is not None and cached["maps"].shape[0] >= len(metas):
        print(f"[MRAD-DAGM] cache hit {variant}: {cached['path']}", flush=True)
        return cached

    bundle = load_mrad_bundle(SOURCE, variant, device)
    n = min(24, len(metas)) if smoke else len(metas)
    maps, image_scores, retrieval_rows = [], [], []
    start = time.time()
    for i, meta in enumerate(metas[:n]):
        if i % 50 == 0:
            elapsed = time.time() - start
            rate = elapsed / max(i, 1)
            eta = rate * (n - i)
            print(f"[MRAD-DAGM:{variant}] {i}/{n}, eta {eta/60:.1f} min", flush=True)
        amap, img_score, _proj, retr = mrad_infer_one(bundle, meta["img_path"], device)
        maps.append(resize_score_grid(amap, DAGM_EVAL_SIDE))
        image_scores.append(float(img_score))
        retrieval_rows.append({
            "source": SOURCE,
            "target": TARGET,
            "variant": variant,
            "image_index": i,
            "category": meta["category"],
            **retr,
        })
    path = None
    if not smoke:
        path = save_cache(variant, maps, image_scores, retrieval_rows)
    del bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "maps": np.asarray(maps, dtype=np.float32),
        "image_scores": np.asarray(image_scores, dtype=np.float32),
        "retrieval_rows": retrieval_rows,
        "path": path,
    }


def evaluate_mrad_grid(variant, pack, metas):
    raw, percat = evaluate_maps_grid(SOURCE, TARGET, {variant: pack["maps"]}, metas, (variant,))
    # Preserve MRAD's official image score policy instead of top-1% map score.
    for rows in (raw, percat):
        for row in rows:
            row["image_score_policy"] = "MRAD_official_cache_image_score"
    return raw, percat


def protocol_rows_dagm(metric_rows):
    rows = []
    cfg = source_config(SOURCE)
    for variant in MRAD_VARIANTS:
        rows.append({
            "source": SOURCE,
            "target": TARGET,
            "method": method_name(variant),
            "variant": variant,
            "checkpoint": checkpoint_for_variant(variant),
            "checkpoint_note": "none; TF uses source memory only" if variant == M1 else cfg["checkpoint_note"],
            "cache_assets": "cache_model_visa.pt; cache_patch_model_visa.pt",
            "raw_map_cache": str(mrad_cache_path(variant)),
            "backbone": OFFICIAL_DEFAULTS["backbone"],
            "image_size": OFFICIAL_DEFAULTS["image_size"],
            "features_list": ",".join(map(str, OFFICIAL_DEFAULTS["features_list"])),
            "feature_map_layer": ",".join(map(str, OFFICIAL_DEFAULTS["feature_map_layer"])),
            "sigma": OFFICIAL_DEFAULTS["sigma"],
            "k_image_score": OFFICIAL_DEFAULTS["k"],
            "evaluator": f"DAGM_GRID_{DAGM_EVAL_SIDE}x{DAGM_EVAL_SIDE}",
            "protocol_valid": any(r["method"] == method_name(variant) and r["protocol_valid"] for r in metric_rows),
        })
    return rows


def write_report(metric_rows, asset_rows, outcome, elapsed):
    lines = [
        "# Phase MRAD-DAGM-CLOSE",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "This run evaluates MRAD-TF / MRAD-FT / MRAD-CLIP on `Full VisA -> DAGM` using the same VisA MRAD source memory and checkpoint assets as the previous strict same-protocol MRAD audits.",
        "No MRAD model was retrained, no target statistics were used, and DAGM maps are evaluated on the same memory-safe 74x74 grid as the local DAGM comparison.",
        "",
        "## Unified Metrics",
        "",
        "| Source | Target | Method | I-AUROC | I-AP | P-AUROC | P-AP | P-F1max | AUPRO |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in metric_rows:
        lines.append(
            f"| {r['source']} | {r['target']} | {r['method']} | "
            f"{float(r['I_AUROC']):.4f} | {float(r['I_AP']):.4f} | "
            f"{float(r['P_AUROC']):.4f} | {float(r['P_AP']):.4f} | "
            f"{float(r['P_F1MAX']):.4f} | {float(r['AUPRO']):.4f} |"
        )
    lines += [
        "",
        "## Source Asset Trace",
        "",
        "| Asset | Exists | Path | SHA256 |",
        "|---|---:|---|---|",
    ]
    for r in asset_rows:
        lines.append(f"| {r['asset']} | {r['exists']} | `{r['path']}` | `{r['sha256']}` |")
    lines += [
        "",
        "## Protocol Notes",
        "",
        f"- Pixel metrics use `DAGM_GRID_{DAGM_EVAL_SIDE}x{DAGM_EVAL_SIDE}` for consistency with the local DAGM benchmark.",
        "- MRAD image-level scores use the official MRAD cache/image score returned by `mrad_infer_one`.",
        "- Pixel maps are generated by official MRAD inference and immediately resized to the DAGM grid before caching.",
        "",
        f"Runtime: {elapsed:.1f}s.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_to_dagm_tables(metric_rows):
    comp_path = ROOT / "results" / "paper_assets" / "dagm_visa_eval" / "DAGM_VISA_COMPARISON.csv"
    big_path = ROOT / "results" / "paper_assets" / "dagm_visa_eval" / "BIG_MODEL_COMPARISON_WITH_DAGM.csv"

    def read_csv(path):
        import csv
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    comp = [r for r in read_csv(comp_path) if not str(r.get("method", "")).startswith("MRAD")]
    big = [r for r in read_csv(big_path) if not (r.get("target") == TARGET and str(r.get("method", "")).startswith("MRAD"))]
    for r in metric_rows:
        comp.append({
            "method": r["method"],
            "source": SOURCE,
            "dataset": TARGET,
            "variant": r["variant"],
            "I_AUROC": r["I_AUROC"],
            "I_AP": r["I_AP"],
            "I_F1max": r["I_F1max"],
            "P_AUROC": r["P_AUROC"],
            "P_AP": r["P_AP"],
            "P_F1max": r["P_F1MAX"],
            "AUPRO": r["AUPRO"],
        })
        big.append({
            "method": r["method"],
            "source": SOURCE,
            "target": TARGET,
            "variant": r["variant"],
            "I_AUROC": r["I_AUROC"],
            "I_AP": r["I_AP"],
            "P_AUROC": r["P_AUROC"],
            "P_AP": r["P_AP"],
            "P_F1MAX": r["P_F1MAX"],
            "AUPRO": r["AUPRO"],
            "strict_comparable": "YES",
            "result_source": "local MRAD DAGM VisA-source grid evaluator",
        })
    write_csv(comp_path, comp)
    write_csv(big_path, big)


def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    assets = audit_assets()
    required = [r for r in assets if r["asset"] in {"image_memory_cache", "patch_memory_cache", "visa_ft_clip_checkpoint"}]
    if not all(bool(r["exists"]) for r in required):
        outcome = "MRAD_DAGM_BLOCKED_BY_SOURCE_ASSET"
        write_csv(OUT / "SOURCE_TRACE.csv", assets)
        write_json(OUT / "OUTCOME.json", {"outcome": outcome, "missing": [r for r in required if not r["exists"]]})
        return
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    dataset = DAGMTargetDataset(args.dagm_root)
    metas = load_dagm_metas(dataset, smoke=args.smoke)
    print(f"[MRAD-DAGM] loaded {len(metas)} DAGM metas on {device}", flush=True)
    metric_rows, percat_rows, retrieval_rows = [], [], []
    start = time.time()
    for variant in MRAD_VARIANTS:
        pack = build_maps(variant, metas, device=device, smoke=args.smoke, force=args.force)
        raw_rows, cats = evaluate_mrad_grid(variant, pack, metas)
        raw = raw_rows[0]
        row = {
            "source": SOURCE,
            "target": TARGET,
            "method": method_name(variant),
            "variant": variant,
            "checkpoint": checkpoint_for_variant(variant),
            "I_AUROC": raw["M_FR_I_AUROC"],
            "I_AP": raw["M_FR_I_AP"],
            "I_F1max": raw["M_FR_I_F1max"],
            "P_AUROC": raw["M_FR_P_AUROC"],
            "P_AP": raw["M_FR_P_AP"],
            "P_F1MAX": raw["M_FR_P_F1max"],
            "AUPRO": raw["M_FR_PRO"],
            "map_cache": str(mrad_cache_path(variant)),
            "evaluator": f"DAGM_GRID_{DAGM_EVAL_SIDE}x{DAGM_EVAL_SIDE}",
            "protocol_valid": True,
        }
        metric_rows.append(row)
        for c in cats:
            c = dict(c)
            c["method"] = method_name(variant)
            percat_rows.append(c)
        retrieval_rows.extend(pack.get("retrieval_rows", []))
    elapsed = time.time() - start
    source_trace = protocol_rows_dagm(metric_rows)
    outcome = "MRAD_DAGM_STRICT_COVERAGE_COMPLETE" if len(metric_rows) == 3 else "MRAD_DAGM_BLOCKED_BY_SOURCE_ASSET"
    write_csv(OUT / "MRAD_DAGM_UNIFIED_METRICS.csv", metric_rows)
    write_csv(OUT / "PER_CATEGORY_UNIFIED_METRICS.csv", percat_rows)
    write_csv(OUT / "SOURCE_TRACE.csv", source_trace + assets)
    if retrieval_rows:
        write_csv(OUT / "MRAD_DAGM_RETRIEVAL_DIAGNOSTICS.csv", retrieval_rows)
    append_to_dagm_tables(metric_rows)
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "source": SOURCE,
        "target": TARGET,
        "variants_completed": [r["method"] for r in metric_rows],
        "runtime_sec": elapsed,
        "no_train": True,
        "no_target_tuning": True,
    })
    write_report(metric_rows, assets, outcome, elapsed)
    print(f"[MRAD-DAGM] done: {outcome} in {elapsed/60:.1f} min -> {OUT}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dagm-root", default="E:/WorkSpace/dataset/DAGM_KaggleUpload")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
