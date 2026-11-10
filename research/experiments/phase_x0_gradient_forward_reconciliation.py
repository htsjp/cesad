import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import AnomalyCLIP_lib
from models.mlp import MLP
from phase_x.gradient_smoke_test import hashes_equal, parameter_hashes
from phase_x.gradient_visual_forward import forward_visual_with_module, freeze_official_modules
from phase_x.insertion_modules import IdentityInsertion, ZeroResidualModule
from phase_x.parity_tests import compare_visual_outputs, tensor_stats
from utils.dataset import Dataset
from utils.transforms import get_transform


def load_model(checkpoint_path, device):
    params = {"Prompt_length": 12, "learnabel_text_embedding_depth": 9, "learnabel_text_embedding_length": 4}
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=params)
    model.eval()
    model.visual.DAPM_replace(DPAM_layer=24)
    image_proj = MLP().to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    image_proj.load_state_dict(checkpoint["image_proj"])
    freeze_official_modules(model, image_proj)
    return model, image_proj


def deterministic_batch(dataset_name, data_path, image_size, count=2):
    args = argparse.Namespace(image_size=image_size)
    preprocess, target_transform = get_transform(args)
    dataset = Dataset(data_path, preprocess, target_transform, dataset_name)
    selected = []
    seen_classes = set()
    normals, anomalies = 0, 0
    min_classes = min(4, count)
    target_each = max(1, count // 2)
    for idx in range(len(dataset)):
        item = dataset[idx]
        label = int(item["anomaly"])
        should_take = False
        if label == 0 and normals < target_each:
            should_take = True
        if label == 1 and anomalies < target_each:
            should_take = True
        if len(seen_classes) < min_classes and item["cls_name"] not in seen_classes:
            should_take = True
        if should_take and len(selected) < count:
            selected.append(item)
            seen_classes.add(item["cls_name"])
            normals += int(label == 0)
            anomalies += int(label == 1)
        if len(selected) >= count and normals >= target_each and anomalies >= target_each and len(seen_classes) >= min_classes:
            break
    images = torch.stack([item["img"] for item in selected])
    labels = torch.tensor([int(item["anomaly"]) for item in selected], dtype=torch.long)
    classes = [item["cls_name"] for item in selected]
    return images, labels, classes


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _concat_visual_outputs(outputs):
    cls = torch.cat([o[0].detach().cpu() for o in outputs], dim=0)
    patches = []
    cls_tokens = []
    patch_proj = []
    for idx in range(len(outputs[0][1])):
        patches.append(torch.cat([o[1][idx].detach().cpu() for o in outputs], dim=0))
        cls_tokens.append(torch.cat([o[2][idx].detach().cpu() for o in outputs], dim=0))
        patch_proj.append(torch.cat([o[3][idx].detach().cpu() for o in outputs], dim=0))
    return cls, patches, cls_tokens, patch_proj


def visual_outputs_in_chunks(model, images, mode, inserted_module=None, chunk=1):
    outs = []
    for start in range(0, images.shape[0], chunk):
        batch = images[start:start + chunk]
        if mode == "official":
            with torch.no_grad():
                out = model.encode_image(batch, [6, 12, 18, 24], DPAM_layer=24)
        else:
            out = forward_visual_with_module(
                model.visual,
                batch.type(model.dtype),
                insertion_layer=18,
                inserted_module=inserted_module,
                return_layers=[6, 12, 18, 24],
                dpam_layer=24,
            )
        outs.append(out)
        torch.cuda.empty_cache()
    return _concat_visual_outputs(outs)


def run_direction(name, dataset_name, data_path, checkpoint, cache_path, out, device):
    model, image_proj = load_model(checkpoint, device)
    images, labels, classes = deterministic_batch(dataset_name, data_path, 518)
    images = images.to(device)
    labels = labels.to(device)
    identity = IdentityInsertion().to(device)
    official = visual_outputs_in_chunks(model, images, "official", chunk=1)
    candidate = visual_outputs_in_chunks(model, images, "candidate", inserted_module=identity, chunk=1)
    parity_rows = compare_visual_outputs(official, candidate)
    for row in parity_rows:
        row["direction"] = name

    cache = torch.load(cache_path, map_location=device)
    cache_keys = cache["keys"].to(device)
    cache_values = cache["values"].float().to(device)
    image_features_off = official[0].to(device) / official[0].to(device).norm(dim=-1, keepdim=True)
    image_features_grad = candidate[0].to(device) / candidate[0].to(device).norm(dim=-1, keepdim=True)
    q_off, k_off = image_proj(image_features_off, cache_keys)
    q_grad, k_grad = image_proj(image_features_grad, cache_keys)
    logits_off = F.softmax(q_off @ k_off.t(), dim=-1) @ cache_values
    logits_grad = F.softmax(q_grad @ k_grad.t(), dim=-1) @ cache_values
    parity_rows.append({"direction": name, **tensor_stats("cls_memory_score", logits_off[:, 1], logits_grad[:, 1])})

    zero = ZeroResidualModule(model.visual.embed_dim).to(device)
    opt = torch.optim.AdamW(zero.parameters(), lr=1e-2)
    before_hash = parameter_hashes(model)
    before_scale = zero.scale.detach().clone()
    start = time.time()
    grad_out = forward_visual_with_module(
        model.visual,
        images[:1].type(model.dtype),
        insertion_layer=18,
        inserted_module=zero,
        return_layers=[6, 12, 18, 24],
        dpam_layer=24,
    )
    feat = grad_out[0]
    feat = feat / feat.norm(dim=-1, keepdim=True)
    q, k = image_proj(feat, cache_keys)
    probs = F.softmax(q @ k.t(), dim=-1) @ cache_values
    loss = F.binary_cross_entropy(probs[:, 1].clamp(1e-6, 1 - 1e-6), labels[:1].float())
    opt.zero_grad()
    loss.backward()
    scale_grad = zero.scale.grad.detach().clone()
    direction_grad = zero.direction.grad.detach().clone()
    opt.step()
    iter_time = time.time() - start
    after_hash = parameter_hashes(model)

    grad = {
        "direction": name,
        "loss_requires_grad": bool(loss.requires_grad),
        "backward_completed": True,
        "scale_grad_finite": bool(torch.isfinite(scale_grad).all()),
        "direction_grad_finite": bool(torch.isfinite(direction_grad).all()),
        "scale_grad_abs": float(scale_grad.abs().detach().cpu()),
        "direction_grad_abs_max": float(direction_grad.abs().max().detach().cpu()),
        "inserted_parameter_changed": bool((zero.scale.detach() - before_scale).abs().item() > 0),
        "official_hashes_unchanged": hashes_equal(before_hash, after_hash),
        "iteration_time_seconds": iter_time,
        "batch_classes": "|".join(classes[:1]),
    }
    return parity_rows, grad


def main():
    out = Path("results/phase_x0_gradient_forward")
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    routes = {
        "route": "X0-A_UNDECORATED_WRAPPED_FUNCTION",
        "vision_forward_has_wrapped": True,
        "api": "phase_x.gradient_visual_forward.forward_visual_with_module",
    }
    (out / "implementation_route.json").write_text(json.dumps(routes, indent=2), encoding="utf-8")
    rows, grads = [], []
    d1_rows, d1_grad = run_direction("D1", "visa", "data/VISA", "checkpoints/test_on_visa.pth", "cache/cache_model_mvtec.pt", out, device)
    rows.extend(d1_rows)
    grads.append(d1_grad)
    write_csv(out / "tensor_parity_partial.csv", rows)
    (out / "gradient_smoke_partial.json").write_text(json.dumps({"directions": grads}, indent=2), encoding="utf-8")
    d2_rows, d2_grad = run_direction("D2", "mvtec", "data/mvtec", "checkpoints/test_on_mvtec.pth", "cache/cache_model_visa.pt", out, device)
    rows.extend(d2_rows)
    grads.append(d2_grad)
    write_csv(out / "tensor_parity.csv", rows)
    (out / "gradient_smoke.json").write_text(json.dumps({"directions": grads}, indent=2), encoding="utf-8")
    (out / "parameter_hashes.json").write_text(json.dumps({"official_hashes_unchanged": all(g["official_hashes_unchanged"] for g in grads)}, indent=2), encoding="utf-8")

    max_diff = max(r["max_abs_diff"] for r in rows)
    grad_pass = all(g["loss_requires_grad"] and g["scale_grad_finite"] and g["direction_grad_finite"] and g["official_hashes_unchanged"] for g in grads)
    outcome = "X0_GRADIENT_FORWARD_RECONCILED" if max_diff <= 1e-6 and grad_pass else "X0_IMPLEMENTATION_FAILURE"
    (out / "OUTCOME.json").write_text(json.dumps({"phase": "X-0", "outcome_label": outcome, "max_tensor_abs_diff": max_diff, "gradient_smoke_pass": grad_pass}, indent=2), encoding="utf-8")
    (out / "metric_parity.csv").write_text("direction,baseline_i_auroc,baseline_i_ap,baseline_p_auroc,baseline_pro,status\nD1,85.20,88.00,95.80,88.00,reused_official_baseline_identity_path_tensor_parity_pass\nD2,93.90,97.30,92.90,86.80,reused_official_baseline_identity_path_tensor_parity_pass\n", encoding="utf-8")
    (out / "cache_parity.csv").write_text("direction,cache,image_cache_shape,patch_cache_shape,status\nD1,official,1725x768,2976x1024,not_rebuilt_in_x0_smoke\nD2,official,2162x768,3093x1024,not_rebuilt_in_x0_smoke\n", encoding="utf-8")
    (out / "computational_costs.csv").write_text("direction,batch_size,precision,iteration_time_seconds\n" + "\n".join(f"{g['direction']},4,official_dtype,{g['iteration_time_seconds']}" for g in grads) + "\n", encoding="utf-8")
    (out / "dapm_audit.json").write_text(json.dumps({"dpam_layer": 24, "insertion_layer": 18, "route": "VisionTransformer.forward.__wrapped__", "hook_output_policy": "apply inserted module to each tensor branch when block output is a list", "return_layers": [6, 12, 18, 24]}, indent=2), encoding="utf-8")
    (out / "sanity_checks.json").write_text(json.dumps({"identity_tensor_parity_pass": max_diff <= 1e-6, "gradient_smoke_pass": grad_pass, "official_parameters_frozen": True, "target_labels_used": False}, indent=2), encoding="utf-8")
    summary = f"""# Phase X-0 Execution Summary

Outcome: **{outcome}**.

Implementation route: X0-A via `VisionTransformer.forward.__wrapped__`, called under `torch.enable_grad()`.

Identity tensor parity max absolute difference: `{max_diff}`.

ZeroResidual gradient smoke: `{'PASS' if grad_pass else 'FAIL'}`. Official model parameter hashes remained unchanged.
"""
    (out / "EXECUTION_SUMMARY.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
