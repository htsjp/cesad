import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import (
    OUT,
    X0A,
    baseline_reproduction_smoke,
    indices_for_categories,
    load_dataset,
    teacher_stats,
    to_device_cache,
    train_variant,
    write_csv,
)
from phase_xa.adapted_cache_builder import build_adapted_task_cache
from phase_xa.cache_alignment import assert_cache_alignment
from phase_xa.config import SMOKE_CONFIG
from phase_xa.random_teacher import permute_memory_values
from phase_xa.source_folds import deterministic_source_folds
from phase_xa.teacher_memory_builder import build_memory_from_indices


VARIANTS = [
    "XA0_OfficialMRADCLIP",
    "XA1_HardLabelAdapter",
    "XA2_CLIPPreservedAdapter",
    "XA3_TaskOnlyResidualAdapter",
    "XA4_DualTeacherFullMemory",
    "XA5_DualTeacherCompactMemory",
    "XA6_RandomMemoryTeacher",
    "XA7_LabelOnlySoftTeacher",
    "XA8_CLIPSelfStyleCropTeacher",
]


def read_existing(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def all_indices_for_categories(dataset, categories):
    return [idx for idx in range(len(dataset)) if dataset[idx]["cls_name"] in categories]


def limited(indices, limit):
    if limit is None or limit <= 0:
        return indices
    return indices[:limit]


def run_key(direction, fold, variant, seed):
    return f"{direction}|{fold}|{variant}|{seed}"


def parse_args():
    parser = argparse.ArgumentParser("Phase X-A Stage 1 source screening")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--folds", default="0,1,2")
    parser.add_argument("--directions", default="D1,D2")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--random-seeds", default="0,1,2,3,4")
    parser.add_argument("--max-train-items", type=int, default=0, help="0 means all source-training items")
    parser.add_argument("--max-val-items", type=int, default=0, help="0 means all held-out source-validation items")
    parser.add_argument("--max-teacher-items", type=int, default=0, help="0 means all source-training items")
    parser.add_argument("--max-cache-items", type=int, default=0, help="0 means all source-training items")
    parser.add_argument("--optimize-items-per-epoch", type=int, default=0, help="0 means all selected source-training items")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt_dir = OUT / "checkpoints" / "stage1"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    x0a_outcome = json.loads((X0A / "OUTCOME.json").read_text(encoding="utf-8")).get("outcome_label")
    if x0a_outcome != "X0_ALIAS_AWARE_GRADIENT_FORWARD_RECONCILED":
        raise SystemExit("XA_K_IMPLEMENTATION_FAILURE: X-0A accepted path is not reconciled")
    baseline = baseline_reproduction_smoke(device)
    if not baseline["pass"]:
        raise SystemExit("XA_L_BASELINE_REPRODUCTION_FAILURE")
    (OUT / "baseline_reproduction.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")

    SMOKE_CONFIG["epochs"] = int(args.epochs)
    SMOKE_CONFIG["learning_rate"] = float(args.lr)
    SMOKE_CONFIG["cache_items_per_epoch"] = None if args.max_cache_items <= 0 else int(args.max_cache_items)
    SMOKE_CONFIG["teacher_items"] = None if args.max_teacher_items <= 0 else int(args.max_teacher_items)
    SMOKE_CONFIG["optimize_items_per_epoch"] = 10**9 if args.optimize_items_per_epoch <= 0 else int(args.optimize_items_per_epoch)

    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    directions = [x.strip() for x in args.directions.split(",") if x.strip()]
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    random_seeds = [int(x) for x in args.random_seeds.split(",") if x.strip()]

    fold_info = {
        "mvtec": deterministic_source_folds("mvtec"),
        "visa": deterministic_source_folds("visa"),
    }
    (OUT / "source_folds.json").write_text(json.dumps(fold_info, indent=2), encoding="utf-8")
    (OUT / "source_protocol.json").write_text(json.dumps({
        "protocol": "checkpoint-conditioned source category-held-out validation",
        "stage": "Stage 1 source screening",
        "target_labels_used": False,
        "target_evaluation_started": False,
        "resume_enabled": True,
    }, indent=2), encoding="utf-8")

    stage1_path = OUT / "stage1_source_metrics.csv"
    fold_path = OUT / "stage1_fold_metrics.csv"
    diag_path = OUT / "stage1_diagnostics.json"
    running_path = OUT / "stage1_running.csv"
    existing = read_existing(stage1_path)
    done = {run_key(r["direction"], r["fold"], r["variant"], r["seed"]) for r in existing if r.get("status") == "COMPLETE"}
    stage_rows = existing
    fold_rows = read_existing(fold_path)
    diagnostics = json.loads(diag_path.read_text(encoding="utf-8")) if diag_path.exists() and diag_path.stat().st_size > 0 else {}

    direction_cfgs = {
        "D1": ("mvtec", "data/mvtec", "checkpoints/test_on_visa.pth"),
        "D2": ("visa", "data/VISA", "checkpoints/test_on_mvtec.pth"),
    }
    for direction in directions:
        dataset_name, data_path, checkpoint = direction_cfgs[direction]
        model, _, image_proj, patch_proj, _ = load_components(checkpoint, device)
        dataset = load_dataset(dataset_name, data_path)
        for fold in folds:
            rotation = fold_info[dataset_name]["rotations"][fold]
            train_indices = all_indices_for_categories(dataset, set(rotation["train_categories"]))
            val_indices = all_indices_for_categories(dataset, set(rotation["val_categories"]))
            train_indices = limited(train_indices, args.max_train_items)
            val_indices = limited(val_indices, args.max_val_items)
            if not train_indices or not val_indices:
                raise SystemExit(f"XA_K_IMPLEMENTATION_FAILURE: empty source split {direction} fold {fold}")
            image_cache, patch_cache = build_memory_from_indices(
                model,
                dataset,
                train_indices,
                dataset_name,
                device,
                adapter=None,
                max_items=None if args.max_teacher_items <= 0 else args.max_teacher_items,
            )
            align = assert_cache_alignment(image_cache, patch_cache)
            image_cache = to_device_cache(image_cache, device)
            patch_cache = to_device_cache(patch_cache, device)
            random_patch = to_device_cache(permute_memory_values({k: patch_cache[k].cpu() if torch.is_tensor(patch_cache[k]) else patch_cache[k] for k in patch_cache}, seed=0), device)
            diagnostics[f"{direction}_fold{fold}_teacher"] = teacher_stats(model, dataset, train_indices[: min(8, len(train_indices))], image_proj, patch_proj, image_cache, patch_cache, device, random_patch)
            for variant in variants:
                seeds = random_seeds if variant.startswith("XA6_") else [0]
                for seed in seeds:
                    key = run_key(direction, fold, variant, seed)
                    if args.resume and key in done:
                        continue
                    torch.manual_seed(seed)
                    start = time.time()
                    if variant.startswith("XA0_"):
                        adapted_image_cache, adapted_patch_cache = build_adapted_task_cache(
                            model,
                            dataset,
                            train_indices,
                            dataset_name,
                            device,
                            adapter=torch.nn.Identity(),
                            max_items=None if args.max_cache_items <= 0 else args.max_cache_items,
                        )
                        cache_alignment = assert_cache_alignment(adapted_image_cache, adapted_patch_cache)
                        diag = {
                            "direction": direction,
                            "fold": fold,
                            "variant": variant,
                            "seed": seed,
                            "trainable_parameters": 0,
                            "cache_alignment": cache_alignment,
                            "official_baseline_reference": True,
                        }
                        metrics = []
                        checkpoint_path = ""
                    else:
                        metrics, diag, adapter = train_variant(
                            direction,
                            variant,
                            model,
                            dataset,
                            train_indices,
                            val_indices,
                            image_proj,
                            patch_proj,
                            image_cache,
                            patch_cache,
                            device,
                            return_adapter=True,
                            random_seed=seed,
                        )
                        checkpoint_path = ckpt_dir / f"{direction}_fold{fold}_{variant}_seed{seed}.pt"
                        torch.save({
                            "adapter": adapter.state_dict(),
                            "direction": direction,
                            "fold": fold,
                            "variant": variant,
                            "seed": seed,
                            "config": vars(args),
                        }, checkpoint_path)
                        write_csv(OUT / f"stage1_metrics_{direction}_fold{fold}_{variant}_seed{seed}.csv", metrics)
                    elapsed = time.time() - start
                    row = {
                        "direction": direction,
                        "source_dataset": dataset_name,
                        "fold": fold,
                        "variant": variant,
                        "seed": seed,
                        "epochs": args.epochs,
                        "train_items": len(train_indices),
                        "val_items": len(val_indices),
                        "teacher_image_keys": image_cache["keys"].shape[0],
                        "teacher_patch_keys": patch_cache["keys"].shape[0],
                        "status": "COMPLETE",
                        "seconds": elapsed,
                        "checkpoint": str(checkpoint_path),
                        "trainable_parameters": diag.get("trainable_parameters", 0),
                        "final_original_clip_cosine": diag.get("final_original_clip_cosine", ""),
                        "rank_collapse_fraction": diag.get("rank_collapse_fraction", ""),
                        "validation_inference_complete": diag.get("validation_inference_complete", True),
                        "official_parameters_unchanged": diag.get("official_parameters_unchanged", True),
                        "self_exclusion_active": diag.get("self_exclusion_active", True),
                    }
                    stage_rows.append(row)
                    fold_rows.append(row.copy())
                    diagnostics[key] = diag
                    write_csv(stage1_path, stage_rows)
                    write_csv(fold_path, fold_rows)
                    write_csv(running_path, stage_rows)
                    diag_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    summary = {
        "stage": "Phase X-A Stage 1",
        "status": "COMPLETE",
        "target_evaluation_started": False,
        "rows": len(stage_rows),
        "directions": directions,
        "folds": folds,
        "variants": variants,
        "random_memory_seeds": random_seeds,
        "note": "This entry point performs source-only Stage 1 training/diagnostics and writes resumable rows after every completed run.",
    }
    (OUT / "stage1_continuation.json").write_text(json.dumps({
        "status": "NOT_ANALYZED",
        "reason": "Stage 1 completed by runner; continuation gate analysis must be run after inspecting source metrics.",
        "target_evaluation_started": False,
    }, indent=2), encoding="utf-8")
    (OUT / "stage1_execution_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
