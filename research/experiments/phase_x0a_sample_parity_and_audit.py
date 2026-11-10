import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import AnomalyCLIP_lib
from models.mlp import AnomalyMLP, MLP, Projector, average_neighbor
from models.prompt_learner import AnomalyCLIP_PromptLearner
from mrad import compute_patch_socre
from phase_x.gradient_visual_forward import forward_visual_with_module, freeze_official_modules
from phase_x.insertion_modules import IdentityInsertion
from phase_x.parity_tests import tensor_stats
from utils.dataset import Dataset
from utils.transforms import get_transform


OUT = Path("results/phase_x0a_alias_parity")
REQUIRED_FILES = [
    "true_snapshot_parity.csv",
    "legacy_alias_diagnostics.csv",
    "storage_pointer_audit.csv",
    "final_tensor_parity.csv",
    "metric_parity.csv",
    "gradient_smoke.json",
    "parameter_hashes.json",
    "historical_v12_v18_alias_audit.json",
    "sanity_checks.json",
    "OUTCOME.json",
    "EXECUTION_SUMMARY.md",
]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def load_components(checkpoint_path, device):
    params = {"Prompt_length": 12, "learnabel_text_embedding_depth": 9, "learnabel_text_embedding_length": 4}
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=params)
    model.eval()
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), params).to(device).eval()
    image_proj = MLP().to(device).eval()
    patch_proj = Projector(1024, 768, length=2).to(device).eval()
    prompt_proj = AnomalyMLP().to(device).eval()
    ckpt = torch.load(checkpoint_path, map_location=device)
    image_proj.load_state_dict(ckpt["image_proj"])
    patch_proj.load_state_dict(ckpt["patch_proj"])
    prompt_learner.load_state_dict(ckpt["prompt_learner"])
    prompt_proj.load_state_dict(ckpt["prompt_proj"])
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=24)
    freeze_official_modules(model, prompt_learner, image_proj, patch_proj, prompt_proj)
    return model, prompt_learner, image_proj, patch_proj, prompt_proj


def select_sample_indices(dataset, categories=4, normals_per_class=2, anomalies_per_class=2):
    selected = []
    counts = {}
    for idx in range(len(dataset)):
        item = dataset[idx]
        cls = item["cls_name"]
        if len(counts) >= categories and cls not in counts:
            continue
        if cls not in counts:
            counts[cls] = {0: 0, 1: 0}
        label = int(item["anomaly"])
        limit = anomalies_per_class if label else normals_per_class
        if counts[cls][label] < limit:
            selected.append(idx)
            counts[cls][label] += 1
        complete = [c for c, v in counts.items() if v[0] >= normals_per_class and v[1] >= anomalies_per_class]
        if len(complete) >= categories:
            keep = set(complete[:categories])
            return [i for i in selected if dataset[i]["cls_name"] in keep], counts
    return selected, counts


def capture_official(model, image):
    snapshots = {}
    handles = []

    def clone_value(output):
        if isinstance(output, list):
            return [o.detach().clone() for o in output]
        return output.detach().clone()

    for layer in [6, 12, 18, 24]:
        def make_hook(layer_idx):
            def hook(_module, _inputs, output):
                snapshots[f"true_block{layer_idx}"] = clone_value(output)
            return hook
        handles.append(model.visual.transformer.resblocks[layer - 1].register_forward_hook(make_hook(layer)))
    with torch.no_grad():
        outputs = model.encode_image(image, [6, 12, 18, 24], DPAM_layer=24)
    for handle in handles:
        handle.remove()
    return outputs, snapshots


def primary(value):
    return value[0] if isinstance(value, list) else value


def compute_task_tensors(outputs, model, prompt_learner, image_proj, patch_proj, prompt_proj, image_cache, patch_cache, cls_id, mask):
    cache_key, cache_value = image_cache
    cache_keys_patch, cache_values_patch = patch_cache
    image_features, patch_features, _, patch_projections = outputs
    patch_projection = average_neighbor(patch_projections[3])
    patch_projection = patch_projection / patch_projection.norm(dim=-1, keepdim=True)
    patch_bias = average_neighbor(patch_features[3])
    patch_bias = patch_bias / patch_bias.norm(dim=-1, keepdim=True)
    seg_logit, patch_bias_tokens, _, _ = compute_patch_socre(
        patch_bias,
        cache_keys_patch,
        cache_values_patch,
        device=image_features.device,
        proj=patch_proj,
        need_mask=False,
        patch_projection=patch_projection,
        gt_mask=mask,
        is_mradft=False,
        use_proj=True,
    )
    bias = prompt_proj(patch_bias_tokens[:, 0, :], patch_bias_tokens[:, 1, :])
    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None, bias=bias)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    norm_image = image_features / image_features.norm(dim=-1, keepdim=True)
    image_features_proj, cache_keys_proj = image_proj(norm_image, cache_key)
    cache_logits = torch.softmax(image_features_proj @ cache_keys_proj.t(), dim=-1) @ cache_value.float()
    cls_memory = cache_logits[:, 1]
    similarity = AnomalyCLIP_lib.compute_similarity(patch_projection, text_features)
    sim_map = AnomalyCLIP_lib.get_similarity_map(similarity, 518)
    prompt_map = (sim_map[..., 1] + 1 - sim_map[..., 0]) / 2.0
    gaussian_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma=4)) for i in prompt_map.detach().cpu()], dim=0).sum(dim=0).to(image_features.device)
    prompt_top1 = torch.topk(gaussian_map.flatten(), int(gaussian_map.numel() * 0.01)).values.mean()
    final_score = 0.3 * prompt_top1 + 0.7 * cls_memory
    return {
        "final_cls": image_features,
        "final_patch_features": patch_features[3],
        "image_projection": image_features_proj,
        "patch_projection": patch_projection,
        "prompt_similarity_map": prompt_map,
        "patch_memory_bias": patch_bias,
        "patch_memory_output": patch_bias_tokens,
        "cls_memory_score": cls_memory,
        "prompt_top1_image_score": prompt_top1.reshape(1),
        "final_k0.7_image_score": final_score.reshape(1),
        "final_gaussian_pixel_map": gaussian_map,
    }


def write_file_audit():
    rows = []
    for name in REQUIRED_FILES:
        path = OUT / name
        rows.append({
            "file": str(path.as_posix()),
            "exists": path.exists(),
            "non_empty": path.exists() and path.stat().st_size > 0,
            "bytes": path.stat().st_size if path.exists() else 0,
        })
    write_csv(OUT / "mandatory_file_existence_audit.csv", rows)


def write_branch_audit():
    audit = {
        "block": 18,
        "dapm_layer": 24,
        "number_of_branches_returned_at_block18": 2,
        "branch_shapes_for_batch_1": ["1370x1x1024", "1370x1x1024"],
        "branch_0": {
            "name": "DAPM/new path x",
            "role": "Patch-oriented DAPM branch using the custom Attention qkv/proj copied from official MHA; with ffn=False it skips the MLP on the new path.",
            "contributes_to": ["out_tokens at returned feature layers", "patch features", "patch projections", "prompt similarity map", "patch-memory bias/output", "pixel map"],
        },
        "branch_1": {
            "name": "original path x_ori",
            "role": "Original CLIP residual path with attention residual and MLP; it supplies the final CLS image feature.",
            "contributes_to": ["final CLS", "image projection", "CLS-memory score", "final k=0.7 image score"],
        },
        "inserted_identity_module_application": {
            "current_safe_forward_semantics": "The insertion hook recursively applies the same inserted module instance to every tensor branch in a list output.",
            "applied_to": "both branch_0 and branch_1 when block output is a list",
            "parameters_shared": True,
            "branch_specific_parameters": False,
        },
    }
    (OUT / "dapm_block18_branch_semantics.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model, prompt_learner, image_proj, patch_proj, prompt_proj = load_components("checkpoints/test_on_visa.pth", device)
    preprocess, target_transform = get_transform(argparse.Namespace(image_size=518))
    dataset = Dataset("data/VISA", preprocess, target_transform, "visa")
    indices, counts = select_sample_indices(dataset)
    image_cache = torch.load("cache/cache_model_mvtec.pt", map_location=device)
    patch_cache = torch.load("cache/cache_patch_model_mvtec.pt", map_location=device)
    image_cache = (image_cache["keys"].to(device), image_cache["values"].to(device))
    patch_cache = (patch_cache["keys"].to(device), patch_cache["values"].to(device))

    rows = []
    sample_rows = []
    for sample_idx in indices:
        item = dataset[sample_idx]
        image = item["img"].unsqueeze(0).to(device)
        mask = item["img_mask"].unsqueeze(0)
        official_outputs, official_snaps = capture_official(model, image)
        safe = forward_visual_with_module(
            model.visual,
            image.type(model.dtype),
            insertion_layer=18,
            inserted_module=IdentityInsertion().to(device),
            return_layers=[6, 12, 18, 24],
            dpam_layer=24,
            snapshot_mode="true_clone",
        )
        safe_outputs = safe["outputs"]
        for layer in [6, 12, 18, 24]:
            stat = tensor_stats(f"true block-{layer} snapshot", primary(official_snaps[f"true_block{layer}"]), primary(safe["snapshots"][f"true_block{layer}"]))
            stat.update({"sample_idx": sample_idx, "class": item["cls_name"], "label": int(item["anomaly"])})
            rows.append(stat)
        official_tensors = compute_task_tensors(official_outputs, model, prompt_learner, image_proj, patch_proj, prompt_proj, image_cache, patch_cache, item["cls_id"], mask)
        safe_tensors = compute_task_tensors(safe_outputs, model, prompt_learner, image_proj, patch_proj, prompt_proj, image_cache, patch_cache, item["cls_id"], mask)
        for name in [
            "final_cls",
            "final_patch_features",
            "image_projection",
            "patch_projection",
            "prompt_similarity_map",
            "patch_memory_bias",
            "patch_memory_output",
            "cls_memory_score",
            "prompt_top1_image_score",
            "final_k0.7_image_score",
            "final_gaussian_pixel_map",
        ]:
            stat = tensor_stats(name, official_tensors[name], safe_tensors[name])
            stat.update({"sample_idx": sample_idx, "class": item["cls_name"], "label": int(item["anomaly"])})
            rows.append(stat)
        sample_rows.append({"sample_idx": sample_idx, "class": item["cls_name"], "label": int(item["anomaly"]), "image_path": item["img_path"]})

    write_csv(OUT / "mandatory_16_sample_parity.csv", rows)
    write_csv(OUT / "mandatory_16_sample_manifest.csv", sample_rows)
    write_file_audit()
    write_branch_audit()
    cost = {
        "subtest": "mandatory_16_sample_safe_vs_official_parity",
        "seconds": time.time() - start,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "batch_size": 1,
        "precision": "official fp32 tensors",
        "images": len(indices),
        "normal_images": sum(1 for r in sample_rows if r["label"] == 0),
        "anomalous_images": sum(1 for r in sample_rows if r["label"] == 1),
        "categories": len(set(r["class"] for r in sample_rows)),
    }
    write_csv(OUT / "mandatory_sample_computational_costs.csv", [cost])


if __name__ == "__main__":
    main()
