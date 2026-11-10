import argparse
import csv
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
from experiments.phase_median_spatial_consensus_generalization import HOLDOUTS, MetaDataset
from experiments.phase_mva_mrad_same_protocol_visa_factorization import (
    M1,
    M2,
    M3,
    MRAD_VARIANTS,
    OFFICIAL_DEFAULTS,
    evaluate_variant_maps,
    load_mrad_bundle,
    mrad_infer_one,
    source_config,
)
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import item_at, item_meta, mask_np
from experiments.phase_xa_smoke import write_csv


OUT = ROOT / "results" / "paper_assets" / "mrad_dtd_close"
MAP_CACHE = OUT / "MAP_CACHE"
SOURCE = "Full VisA"
TARGET = "DTD-Synthetic"


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


def sha256_file(path):
    h = __import__("hashlib").sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dtd_metas(smoke=False):
    root, _domain, _kind = HOLDOUTS[TARGET]
    dataset = MetaDataset(root, mode="test")
    refs = list(range(len(dataset)))
    if smoke:
        refs = refs[:24]
    metas = []
    for pos, ref in enumerate(refs):
        item = item_at(dataset, "source_dataset", ref, pos)
        cat, label, img_path = item_meta(item, "source_dataset", ref, TARGET)
        metas.append(
            {
                "category": cat,
                "label": int(label),
                "img_path": img_path,
                "mask": mask_np(item),
            }
        )
    return metas


def mrad_cache_path(variant):
    return MAP_CACHE / f"{pair_slug(SOURCE, TARGET)}__{variant}.npz"


def save_cache(variant, maps, image_scores, feature_means, retrieval_rows):
    path = mrad_cache_path(variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        maps=np.asarray(maps, dtype=np.float16),
        image_scores=np.asarray(image_scores, dtype=np.float32),
        feature_means=np.asarray(feature_means, dtype=np.float32),
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
        "feature_means": np.asarray(data["feature_means"], dtype=np.float32),
        "retrieval_rows": list(data["retrieval_rows"]) if "retrieval_rows" in data.files else [],
        "path": path,
    }


def build_maps(variant, metas, device, smoke=False, force=False):
    cached = None if force or smoke else load_cache(variant)
    if cached is not None and cached["maps"].shape[0] >= len(metas):
        print(f"[MRAD-DTD] cache hit {variant}: {cached['path']}", flush=True)
        return cached

    bundle = load_mrad_bundle(SOURCE, variant, device)
    n = min(24, len(metas)) if smoke else len(metas)
    maps, image_scores, feature_means, retrieval_rows = [], [], [], []
    start = time.time()
    for i, meta in enumerate(metas[:n]):
        if i % 50 == 0:
            elapsed = time.time() - start
            rate = elapsed / max(i, 1)
            eta = rate * (n - i)
            print(f"[MRAD-DTD:{variant}] {i}/{n}, eta {eta/60:.1f} min", flush=True)
        amap, img_score, proj, retr = mrad_infer_one(bundle, meta["img_path"], device)
        maps.append(amap)
        image_scores.append(img_score)
        feature_means.append(proj.mean(axis=0))
        retrieval_rows.append(
            {
                "source": SOURCE,
                "target": TARGET,
                "variant": variant,
                "image_index": i,
                "category": meta["category"],
                **retr,
            }
        )

    path = None
    if not smoke:
        path = save_cache(variant, maps, image_scores, feature_means, retrieval_rows)
    del bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "maps": np.asarray(maps, dtype=np.float32),
        "image_scores": np.asarray(image_scores, dtype=np.float32),
        "feature_means": np.asarray(feature_means, dtype=np.float32),
        "retrieval_rows": retrieval_rows,
        "path": path,
    }


def method_name(variant):
    return {M1: "MRAD-TF", M2: "MRAD-FT", M3: "MRAD-CLIP"}[variant]


def checkpoint_for_variant(variant):
    cfg = source_config(SOURCE)
    return "" if variant == M1 else str(cfg["checkpoint"])


def audit_assets():
    cfg = source_config(SOURCE)
    assets = [
        {"asset": "image_memory_cache", "path": ROOT / "cache" / "cache_model_visa.pt"},
        {"asset": "patch_memory_cache", "path": ROOT / "cache" / "cache_patch_model_visa.pt"},
        {"asset": "visa_ft_clip_checkpoint", "path": cfg["checkpoint"]},
    ]
    rows = []
    for a in assets:
        p = Path(a["path"])
        rows.append(
            {
                "asset": a["asset"],
                "path": str(p),
                "exists": p.exists(),
                "bytes": p.stat().st_size if p.exists() else "",
                "sha256": sha256_file(p) if p.exists() else "",
            }
        )
    return rows


def protocol_rows(metric_rows):
    rows = []
    cfg = source_config(SOURCE)
    for variant in MRAD_VARIANTS:
        rows.append(
            {
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
                "evaluator": "paper_unified_M_FR_category_macro_full_resolution_align_corners_true",
                "protocol_valid": any(r["method"] == method_name(variant) and r["protocol_valid"] for r in metric_rows),
            }
        )
    return rows


def load_official_reported_reference():
    # Kept separate from final metrics. A local official DTD table was not found in the
    # repository assets during this close-out run, so the sanity reference is explicit NA.
    return [
        {
            "source": SOURCE,
            "target": TARGET,
            "reported_reference": "not_found_in_local_official_assets",
            "used_for_final_metric": False,
            "note": "Final table uses only raw maps evaluated by the frozen paper evaluator.",
        }
    ]


def write_report(metric_rows, source_trace, asset_rows, outcome, elapsed):
    lines = [
        "# Phase MRAD-DTD-CLOSE",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "This close-out run fills the missing strict same-protocol MRAD rows for `Full VisA -> DTD-Synthetic`.",
        "No MRAD model was retrained, no target statistics were used, and all final numbers come from the frozen paper full-resolution evaluator.",
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
        "- Source assets are the same VisA memory/cache/checkpoint family used in the earlier MRAD same-protocol audit.",
        "- Raw anomaly maps are saved under `MAP_CACHE/` in this phase directory.",
        "- Final metrics are category-macro M_FR full-resolution metrics with `align_corners=True`, matching the previous MRAD same-protocol audit.",
        "- Official reported DTD-Synthetic numbers are not injected into the table; local official reference values were not found in repository assets during this run.",
        "",
        f"Runtime: {elapsed:.1f}s.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    audit = [
        "# Protocol Audit",
        "",
        f"- Source: `{SOURCE}`",
        f"- Target: `{TARGET}`",
        "- Training: disabled; no optimizer step.",
        "- Target tuning/statistics: disabled.",
        "- MRAD implementation: reused loader/inference functions from `phase_mva_mrad_same_protocol_visa_factorization.py`.",
        "- Unified evaluator: reused `evaluate_variant_maps`, which resizes raw maps to GT masks and aggregates category-macro full-resolution metrics.",
        "",
        "## Variants",
        "",
    ]
    for r in source_trace:
        audit.append(
            f"- `{r['method']}`: checkpoint `{r['checkpoint'] or 'none'}`, maps `{r['raw_map_cache']}`."
        )
    (OUT / "PROTOCOL_AUDIT.md").write_text("\n".join(audit) + "\n", encoding="utf-8")


def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    setup = audit_assets()
    required = [r for r in setup if r["asset"] in {"image_memory_cache", "patch_memory_cache", "visa_ft_clip_checkpoint"}]
    if not all(bool(r["exists"]) for r in required):
        outcome = "MRAD_DTD_BLOCKED_BY_SOURCE_ASSET"
        write_csv(OUT / "SOURCE_TRACE.csv", setup)
        write_json(OUT / "OUTCOME.json", {"outcome": outcome, "missing": [r for r in required if not r["exists"]]})
        (OUT / "REPORT.md").write_text(f"# Phase MRAD-DTD-CLOSE\n\nOUTCOME: `{outcome}`\n", encoding="utf-8")
        return

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    metas = load_dtd_metas(smoke=args.smoke)
    print(f"[MRAD-DTD] loaded {len(metas)} DTD-Synthetic metas on {device}", flush=True)

    metric_rows = []
    per_category_rows = []
    retrieval_rows = []
    start = time.time()
    for variant in MRAD_VARIANTS:
        pack = build_maps(variant, metas, device=device, smoke=args.smoke, force=args.force)
        raw, cats = evaluate_variant_maps(SOURCE, TARGET, variant, pack["maps"], pack["image_scores"], metas)
        row = {
            "source": SOURCE,
            "target": TARGET,
            "method": method_name(variant),
            "checkpoint": checkpoint_for_variant(variant),
            "I_AUROC": raw["M_FR_I_AUROC"],
            "I_AP": raw["M_FR_I_AP"],
            "P_AUROC": raw["M_FR_P_AUROC"],
            "P_AP": raw["M_FR_P_AP"],
            "P_F1MAX": raw["M_FR_P_F1max"],
            "AUPRO": raw["M_FR_PRO"],
            "map_cache": str(mrad_cache_path(variant)),
            "evaluator": "paper_unified_M_FR_category_macro_full_resolution_align_corners_true",
            "protocol_valid": True,
        }
        metric_rows.append(row)
        for c in cats:
            c = dict(c)
            c["method"] = method_name(variant)
            per_category_rows.append(c)
        retrieval_rows.extend(pack.get("retrieval_rows", []))

    elapsed = time.time() - start
    source_trace = protocol_rows(metric_rows)
    official_ref = load_official_reported_reference()
    outcome = "MRAD_DTD_STRICT_COVERAGE_COMPLETE" if len(metric_rows) == 3 else "MRAD_DTD_BLOCKED_BY_SOURCE_ASSET"

    write_csv(OUT / "MRAD_DTD_UNIFIED_METRICS.csv", metric_rows)
    write_csv(OUT / "PER_CATEGORY_UNIFIED_METRICS.csv", per_category_rows)
    write_csv(OUT / "SOURCE_TRACE.csv", source_trace + setup)
    write_csv(OUT / "MRAD_DTD_OFFICIAL_REPORTED_REFERENCE.csv", official_ref)
    if retrieval_rows:
        write_csv(OUT / "MRAD_DTD_RETRIEVAL_DIAGNOSTICS.csv", retrieval_rows)
    write_json(
        OUT / "OUTCOME.json",
        {
            "outcome": outcome,
            "source": SOURCE,
            "target": TARGET,
            "variants_completed": [r["method"] for r in metric_rows],
            "runtime_sec": elapsed,
            "no_train": True,
            "no_target_tuning": True,
        },
    )
    write_report(metric_rows, source_trace, setup, outcome, elapsed)
    print(f"[MRAD-DTD] done: {outcome} in {elapsed/60:.1f} min -> {OUT}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
