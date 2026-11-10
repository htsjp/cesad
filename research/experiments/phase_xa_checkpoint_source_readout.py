import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import AnomalyCLIP_lib
from experiments.mrad_audit_utils import evaluate_results, init_results
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_stage1 import VARIANTS, all_indices_for_categories, read_existing
from experiments.phase_xa_smoke import load_dataset, to_device_cache, write_csv
from models.mlp import average_neighbor
from mrad import compute_patch_socre
from phase_xa.adapted_cache_builder import build_adapted_task_cache
from phase_xa.official_task_forward import image_memory_scores, normalized_image_feature, visual_outputs
from phase_xa.shared_adapter import SharedBlock18Adapter
from phase_xa.source_folds import deterministic_source_folds
from phase_xa.teacher_memory_builder import build_memory_from_indices


OUT = Path("results/phase_xa_memory_distillation")


def parse_args():
    parser = argparse.ArgumentParser("Phase X-A checkpoint source-category readout")
    parser.add_argument("--direction", required=True, choices=["D1", "D2"])
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--variants", default="XA0_OfficialMRADCLIP,XA1_HardLabelAdapter,XA2_CLIPPreservedAdapter,XA4_DualTeacherFullMemory,XA5_DualTeacherCompactMemory,XA6_RandomMemoryTeacher")
    parser.add_argument("--random-seeds", default="0,1,2")
    parser.add_argument("--limit-val-items", type=int, default=0)
    parser.add_argument("--compute-pro", action="store_true", default=True)
    parser.add_argument("--no-compute-pro", dest="compute_pro", action="store_false")
    parser.add_argument("--save-map64", action="store_true", help="Save 64x64 validation anomaly maps for output-correlation diagnostics.")
    return parser.parse_args()


def direction_cfg(direction):
    if direction == "D1":
        return "mvtec", "data/mvtec", "checkpoints/test_on_visa.pth"
    return "visa", "data/VISA", "checkpoints/test_on_mvtec.pth"


def load_adapter(row, device):
    if row["variant"].startswith("XA0_"):
        return None, ""
    ckpt = row.get("checkpoint", "").replace("\\", "/")
    path = Path(ckpt)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        path = OUT / "checkpoints" / "stage1" / f"{row['direction']}_fold{row['fold']}_{row['variant']}_seed{row['seed']}.pt"
    payload = torch.load(path, map_location=device)
    adapter = SharedBlock18Adapter().to(device).eval()
    adapter.load_state_dict(payload["adapter"], strict=True)
    return adapter, str(path)


def selected_rows(direction, fold, variants, seeds):
    rows = read_existing(OUT / "stage1_source_metrics.csv")
    out = []
    for row in rows:
        if row.get("status") != "COMPLETE":
            continue
        if row.get("direction") != direction or int(row.get("fold", -1)) != fold:
            continue
        if row.get("variant") not in variants:
            continue
        if row.get("variant") == "XA6_RandomMemoryTeacher" and int(row.get("seed", -1)) not in seeds:
            continue
        if row.get("variant") != "XA6_RandomMemoryTeacher" and int(row.get("seed", -1)) != 0:
            continue
        out.append(row)
    order = {v: i for i, v in enumerate(VARIANTS)}
    out.sort(key=lambda r: (order.get(r["variant"], 99), int(r["seed"])))
    return out


def append_result(results, cls_name, item, image_score, anomaly_map):
    mask = item["img_mask"].detach().cpu()
    mask = (mask > 0.5).float()
    results[cls_name]["imgs_masks"].append(mask)
    results[cls_name]["gt_sp"].append(int(item["anomaly"]))
    results[cls_name]["pr_sp"].append(float(image_score))
    results[cls_name]["anomaly_maps"].append(anomaly_map.detach().cpu())


def evaluate_checkpoint(row, model, prompt_learner, image_proj, patch_proj, prompt_proj, dataset, train_indices, val_indices, dataset_name, device, compute_pro, save_map64=False):
    adapter, resolved_checkpoint = load_adapter(row, device)
    if adapter is None:
        image_cache, patch_cache = build_memory_from_indices(model, dataset, train_indices, dataset_name, device, adapter=None)
    else:
        image_cache, patch_cache = build_adapted_task_cache(model, dataset, train_indices, dataset_name, device, adapter=adapter)
    image_cache = to_device_cache(image_cache, device)
    patch_cache = to_device_cache(patch_cache, device)

    obj_list = sorted({dataset[i]["cls_name"] for i in val_indices})
    results = init_results(obj_list)
    per_image = []
    map64_rows = []
    start = time.time()
    for idx in val_indices:
        item = dataset[idx]
        cls_name = item["cls_name"]
        image = item["img"].unsqueeze(0).to(device)
        cls_id = torch.tensor([int(item["cls_id"])], device=device)
        with torch.no_grad():
            outputs = visual_outputs(model, image, adapter=adapter, grad=False)
            image_feature = normalized_image_feature(outputs)
            image_post, _, _, _ = image_memory_scores(image_feature, image_cache["keys"], image_cache["values"], image_proj)
            cls_memory = image_post[:, 1]

            patch_projection = average_neighbor(outputs[3][3])
            patch_projection = patch_projection / patch_projection.norm(dim=-1, keepdim=True)
            patch_bias = average_neighbor(outputs[1][3])
            patch_bias = patch_bias / patch_bias.norm(dim=-1, keepdim=True)
            seg_logit, patch_bias_tokens, _, _ = compute_patch_socre(
                patch_bias,
                patch_cache["keys"],
                patch_cache["values"],
                device=device,
                proj=patch_proj,
                need_mask=False,
                patch_projection=patch_projection,
                gt_mask=item["img_mask"].unsqueeze(0),
                is_mradft=False,
                use_proj=True,
            )
            bias = prompt_proj(patch_bias_tokens[:, 0, :], patch_bias_tokens[:, 1, :])
            prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None, bias=bias)
            text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            anomaly_map_list = []
            for pidx, patch_feature in enumerate(outputs[3]):
                if pidx >= 3:
                    patch_feature = average_neighbor(patch_feature)
                    patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                    similarity = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features)
                    sim_map = AnomalyCLIP_lib.get_similarity_map(similarity, 518)
                    anomaly_map_list.append((sim_map[..., 1] + 1 - sim_map[..., 0]) / 2.0)
            anomaly_map = torch.stack(anomaly_map_list)
            anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(x, sigma=4)) for x in anomaly_map.detach().cpu()], dim=0).sum(dim=0)
            topk = torch.topk(anomaly_map.flatten(), max(1, int(anomaly_map.numel() * 0.01))).values.mean()
            final_score = 0.3 * topk.to(device) + 0.7 * cls_memory
            append_result(results, cls_name, item, final_score.detach().cpu(), anomaly_map)
            if save_map64:
                map_for_resize = anomaly_map.float()
                if map_for_resize.ndim == 2:
                    map_for_resize = map_for_resize.unsqueeze(0).unsqueeze(0)
                elif map_for_resize.ndim == 3:
                    map_for_resize = map_for_resize.unsqueeze(1)
                small_map = F.interpolate(
                    map_for_resize,
                    size=(64, 64),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze().numpy().astype(np.float32)
                map64_rows.append(small_map)
            per_image.append({
                "direction": row["direction"],
                "fold": row["fold"],
                "variant": row["variant"],
                "seed": row["seed"],
                "idx": idx,
                "category": cls_name,
                "label": int(item["anomaly"]),
                "score": float(final_score.detach().cpu()),
            })
    metrics = evaluate_results(results, obj_list, compute_pro=compute_pro)
    return metrics, per_image, map64_rows, {
        "seconds": time.time() - start,
        "images": len(val_indices),
        "categories": obj_list,
        "resolved_checkpoint": resolved_checkpoint,
        "train_memory_categories": sorted({dataset[i]["cls_name"] for i in train_indices}),
        "val_categories": obj_list,
    }


def metric_value(metrics, key):
    value = metrics["mean"].get(key, float("nan"))
    return float(value) * 100.0 if np.isfinite(value) and value <= 1.5 else float(value)


def main():
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dataset_name, data_path, checkpoint = direction_cfg(args.direction)
    model, prompt_learner, image_proj, patch_proj, prompt_proj = load_components(checkpoint, device)
    dataset = load_dataset(dataset_name, data_path)
    rotation = deterministic_source_folds(dataset_name)["rotations"][args.fold]
    train_categories = set(rotation["train_categories"])
    val_categories = set(rotation["val_categories"])
    if train_categories & val_categories:
        raise SystemExit("SOURCE_SPLIT_LEAKAGE")
    train_indices = all_indices_for_categories(dataset, train_categories)
    val_indices = all_indices_for_categories(dataset, val_categories)
    if args.limit_val_items > 0:
        val_indices = val_indices[: args.limit_val_items]

    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    seeds = [int(x) for x in args.random_seeds.split(",") if x.strip()]
    rows = selected_rows(args.direction, args.fold, variants, seeds)
    if not rows:
        raise SystemExit("No completed checkpoints selected")

    all_summary, all_per_category, all_per_image, diagnostics = [], [], [], []
    for row in rows:
        metrics, per_image, map64_rows, diag = evaluate_checkpoint(
            row,
            model,
            prompt_learner,
            image_proj,
            patch_proj,
            prompt_proj,
            dataset,
            train_indices,
            val_indices,
            dataset_name,
            device,
            args.compute_pro,
            save_map64=args.save_map64,
        )
        summary = {
            "direction": row["direction"],
            "fold": row["fold"],
            "variant": row["variant"],
            "seed": row["seed"],
            "I-AUROC": metric_value(metrics, "I-AUROC"),
            "I-AP": metric_value(metrics, "I-AP"),
            "P-AUROC": metric_value(metrics, "P-AUROC"),
            "PRO": metric_value(metrics, "PRO") if args.compute_pro else "",
            "images": diag["images"],
            "seconds": diag["seconds"],
            "resolved_checkpoint": diag["resolved_checkpoint"],
        }
        all_summary.append(summary)
        for category, vals in metrics.get("per_class", {}).items():
            all_per_category.append({
                "direction": row["direction"],
                "fold": row["fold"],
                "variant": row["variant"],
                "seed": row["seed"],
                "category": category,
                "I-AUROC": metric_value({"mean": vals}, "I-AUROC"),
                "I-AP": metric_value({"mean": vals}, "I-AP"),
                "P-AUROC": metric_value({"mean": vals}, "P-AUROC"),
                "PRO": metric_value({"mean": vals}, "PRO") if args.compute_pro else "",
            })
        all_per_image.extend(per_image)
        if args.save_map64 and map64_rows:
            map_path = OUT / f"source_checkpoint_map64_{args.direction}_fold{args.fold}_{row['variant']}_seed{row['seed']}.npz"
            np.savez_compressed(
                map_path,
                maps=np.stack(map64_rows, axis=0),
                indices=np.asarray(val_indices, dtype=np.int64),
                variant=row["variant"],
                seed=int(row["seed"]),
            )
            diag["map64_path"] = str(map_path)
        diag.update({k: row[k] for k in ("direction", "fold", "variant", "seed")})
        diagnostics.append(diag)
        write_csv(OUT / "source_checkpoint_readout_running.csv", all_summary)

    stamp = f"{args.direction}_fold{args.fold}"
    write_csv(OUT / f"source_checkpoint_readout_{stamp}.csv", all_summary)
    write_csv(OUT / f"per_category_source_readout_{stamp}.csv", all_per_category)
    write_csv(OUT / f"source_checkpoint_scores_{stamp}.csv", all_per_image)
    (OUT / f"source_checkpoint_readout_{stamp}_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(all_summary), "output": str(OUT / f"source_checkpoint_readout_{stamp}.csv")}, indent=2))


if __name__ == "__main__":
    main()
