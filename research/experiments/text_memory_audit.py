from collections import defaultdict
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import AnomalyCLIP_lib
from mrad_audit_utils import (
    build_prototype_memory,
    build_text_memory,
    copy_log,
    default_source_dataset,
    evaluate_results,
    extract_layer_features,
    init_results,
    load_config,
    load_model,
    make_loader,
    parse_args,
    retrieval_logits,
    save_metrics,
    save_run_metadata,
    set_seed,
    summarize_retrieval_stats,
    update_retrieval_stats,
    write_csv,
)


@torch.no_grad()
def run_text_method(model, visual_memory, text_memory, loader, cfg, method, device):
    layer = int(cfg.get("layer", 24))
    stats = defaultdict(list)
    results = init_results(loader.dataset.obj_list)
    image_size = cfg.get("image_size", 518)
    sigma = cfg.get("sigma", 4)
    fusion_k = cfg.get("fusion_k", 0.7)
    text_weight = cfg.get("text_weight", 0.5)

    if method == "visual_text_memory":
        image_keys = torch.cat([visual_memory["image_keys"].to(device), text_memory["keys"].to(device)], dim=0)
        image_values = torch.cat([visual_memory["image_values"].to(device), text_memory["values"].to(device)], dim=0)
        patch_keys = torch.cat([visual_memory["patch_keys"].to(device), text_memory["keys"].to(device)], dim=0)
        patch_values = torch.cat([visual_memory["patch_values"].to(device), text_memory["values"].to(device)], dim=0)
    else:
        image_keys = visual_memory["image_keys"].to(device)
        image_values = visual_memory["image_values"].to(device)
        patch_keys = visual_memory["patch_keys"].to(device)
        patch_values = visual_memory["patch_values"].to(device)

    for items in tqdm(loader, desc=method):
        image = items["img"].to(device)
        cls_name = items["cls_name"][0]
        gt_mask = items["img_mask"]
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name]["imgs_masks"].append(gt_mask)
        results[cls_name]["gt_sp"].extend(torch.as_tensor(items["anomaly"]).detach().cpu())
        cls, patches, _ = extract_layer_features(
            model, image, cfg["features_list"], layer, device, smooth=cfg.get("average_neighbor", True)
        )
        image_logits, image_sim, _ = retrieval_logits(cls, image_keys, image_values)
        patch_logits, patch_sim, _ = retrieval_logits(patches, patch_keys, patch_values)
        update_retrieval_stats(stats, image_sim, image_values, "image", cfg.get("retrieval_top_k", 5))
        update_retrieval_stats(stats, patch_sim, patch_values, "patch", cfg.get("retrieval_top_k", 5))

        amap = AnomalyCLIP_lib.get_similarity_map(patch_logits, image_size)[..., 1]
        amap = torch.stack([torch.from_numpy(gaussian_filter(x, sigma=sigma)) for x in amap.detach().cpu()], dim=0)
        flat = amap.flatten(start_dim=1)
        topn = max(1, int(flat.shape[1] * cfg.get("topk_ratio", 0.01)))
        visual_score = (1 - fusion_k) * torch.topk(flat, topn, dim=1).values.mean(dim=1).to(device) + fusion_k * image_logits[:, 1]

        if method == "visual_minus_text_similarity":
            text_logits_cls, _, _ = retrieval_logits(cls, text_memory["keys"].to(device), text_memory["values"].to(device))
            text_logits_patch, _, _ = retrieval_logits(patches, text_memory["keys"].to(device), text_memory["values"].to(device))
            text_map = AnomalyCLIP_lib.get_similarity_map(text_logits_patch, image_size)[..., 1]
            text_map = torch.stack([torch.from_numpy(gaussian_filter(x, sigma=sigma)) for x in text_map.detach().cpu()], dim=0)
            text_flat = text_map.flatten(start_dim=1)
            text_score = (1 - fusion_k) * torch.topk(text_flat, topn, dim=1).values.mean(dim=1).to(device) + fusion_k * text_logits_cls[:, 1]
            score = (1 - text_weight) * visual_score + text_weight * text_score
            amap = (1 - text_weight) * amap + text_weight * text_map
        else:
            score = visual_score

        results[cls_name]["pr_sp"].extend(score.detach().cpu())
        results[cls_name]["anomaly_maps"].append(amap.detach().cpu())

    metrics = evaluate_results(results, loader.dataset.obj_list, compute_pro=cfg.get("compute_pixel_aupro", True))
    memory_size = {"image": int(image_keys.shape[0]), "patch": int(patch_keys.shape[0])}
    return metrics, summarize_retrieval_stats(stats, memory_size)


def main():
    args = parse_args("MRAD text knowledge audit")
    cfg = load_config(
        args.config,
        {
            "dataset": args.dataset,
            "source_dataset": args.source_dataset,
            "device": args.device,
            "output_dir": args.output_dir,
            "seed": args.seed,
            "limit": args.limit,
        },
    )
    cfg["source_dataset"] = cfg.get("source_dataset") or default_source_dataset(cfg["dataset"])
    set_seed(int(cfg.get("seed", 0)))
    output_dir = Path(cfg.get("output_dir", "results/text_memory"))
    save_run_metadata(output_dir, cfg, int(cfg.get("seed", 0)))
    logs = []

    device = cfg.get("device", "cuda:0")
    model = load_model(cfg, device)
    _, source_loader = make_loader(cfg, cfg["source_dataset"], mode=cfg.get("source_mode", "test"), shuffle=False)
    _, target_loader = make_loader(cfg, cfg["dataset"], mode=cfg.get("target_mode", "test"), shuffle=False)
    visual_memory = build_prototype_memory(model, source_loader, cfg, "mean", device)
    text_memory = build_text_memory(model, cfg, device)

    rows = []
    all_metrics = {}
    all_retrieval = {}
    method_names = {
        "visual": "Visual retrieval only",
        "visual_minus_text_similarity": "Visual score - Text similarity",
        "visual_text_memory": "Visual memory - Attribute text memory",
    }
    for method in cfg["methods"]:
        method_dir = output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        metrics, retrieval = run_text_method(model, visual_memory, text_memory, target_loader, cfg, method, device)
        save_metrics(method_dir, metrics, retrieval)
        all_metrics[method] = metrics
        all_retrieval[method] = retrieval
        mean = metrics["mean"]
        rows.append(
            {
                "Method": method_names.get(method, method),
                "I-AUROC": mean["I-AUROC"],
                "I-AP": mean["I-AP"],
                "P-AUROC": mean["P-AUROC"],
                "PRO": mean.get("PRO"),
            }
        )
        logs.append(f"{method}: {mean}")

    write_csv(output_dir / "text_memory_metrics.csv", rows, ["Method", "I-AUROC", "I-AP", "P-AUROC", "PRO"])
    save_metrics(output_dir, {"text_memory_runs": all_metrics, "summary_rows": rows}, all_retrieval)
    copy_log(output_dir, logs)


if __name__ == "__main__":
    main()
