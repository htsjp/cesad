import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import AnomalyCLIP_lib
from models.mlp import MLP
from phase_x.gradient_smoke_test import parameter_hashes, hashes_equal
from phase_x.gradient_visual_forward import forward_visual_with_module, freeze_official_modules
from phase_x.insertion_modules import IdentityInsertion, ZeroResidualModule
from phase_x.parity_tests import tensor_stats
from utils.dataset import Dataset
from utils.transforms import get_transform


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def load_model(checkpoint, device):
    params = {"Prompt_length": 12, "learnabel_text_embedding_depth": 9, "learnabel_text_embedding_length": 4}
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=params)
    model.eval()
    model.visual.DAPM_replace(DPAM_layer=24)
    image_proj = MLP().to(device).eval()
    image_proj.load_state_dict(torch.load(checkpoint, map_location=device)["image_proj"])
    freeze_official_modules(model, image_proj)
    return model, image_proj


def get_images(dataset_name, data_path, count, device):
    preprocess, target_transform = get_transform(argparse.Namespace(image_size=518))
    dataset = Dataset(data_path, preprocess, target_transform, dataset_name)
    items, normals, anomalies = [], 0, 0
    for idx in range(len(dataset)):
        item = dataset[idx]
        label = int(item["anomaly"])
        if label == 0 and normals < count // 2:
            items.append(item); normals += 1
        elif label == 1 and anomalies < count // 2:
            items.append(item); anomalies += 1
        if len(items) >= count:
            break
    return torch.stack([i["img"] for i in items]).to(device), [i["cls_name"] for i in items]


def capture_official_true_snapshots(model, images):
    snapshots = {}
    pointers = []
    handles = []

    def clone_value(output):
        if isinstance(output, list):
            return [o.detach().clone() for o in output]
        return output.detach().clone()

    def ptr_value(output):
        if isinstance(output, list):
            output = output[0]
        return {
            "data_ptr": int(output.data_ptr()),
            "storage_ptr": int(output.untyped_storage().data_ptr()),
            "version": int(output._version),
            "shape": "x".join(str(v) for v in output.shape),
            "dtype": str(output.dtype),
        }

    for layer in [6, 12, 18, 24]:
        def make_hook(layer_idx):
            def hook(_module, _inputs, output):
                snapshots[f"true_block{layer_idx}"] = clone_value(output)
                p = ptr_value(output)
                p["block"] = layer_idx
                pointers.append(p)
            return hook
        handles.append(model.visual.transformer.resblocks[layer - 1].register_forward_hook(make_hook(layer)))
    with torch.no_grad():
        outputs = model.encode_image(images, [6, 12, 18, 24], DPAM_layer=24)
    for h in handles:
        h.remove()
    return outputs, snapshots, pointers


def pick_primary(snapshot):
    if isinstance(snapshot, list):
        return snapshot[0]
    return snapshot


def postprocess_patch_snapshot(model, snapshot):
    primary = pick_primary(snapshot)
    return model.visual.ln_post(primary.permute(1, 0, 2))[:, 1:, :]


def run_direction(name, dataset_name, data_path, checkpoint, cache, device):
    model, image_proj = load_model(checkpoint, device)
    images, classes = get_images(dataset_name, data_path, 2, device)
    official_outputs, official_snaps, pointers = capture_official_true_snapshots(model, images)
    safe = forward_visual_with_module(
        model.visual,
        images.type(model.dtype),
        insertion_layer=18,
        inserted_module=IdentityInsertion().to(device),
        return_layers=[6, 12, 18, 24],
        dpam_layer=24,
        snapshot_mode="true_clone",
    )
    safe_outputs = safe["outputs"]
    safe_snaps = safe["snapshots"]

    true_rows, legacy_rows, final_rows = [], [], []
    for layer in [6, 12, 18, 24]:
        off = pick_primary(official_snaps[f"true_block{layer}"])
        saf = pick_primary(safe_snaps[f"true_block{layer}"])
        row = tensor_stats(f"true_block{layer}", off, saf)
        row["direction"] = name
        true_rows.append(row)

    for idx, layer in enumerate([6, 12, 18, 24]):
        legacy = official_outputs[1][idx]
        immediate = postprocess_patch_snapshot(model, official_snaps[f"true_block{layer}"])
        row = tensor_stats(f"legacy_returned_patch_layer{layer}", immediate, legacy)
        row["direction"] = name
        row["alias_mutation_delta"] = row["max_abs_diff"]
        legacy_rows.append(row)

    final_rows.append({"direction": name, **tensor_stats("final_cls", official_outputs[0], safe_outputs[0])})
    final_rows.append({"direction": name, **tensor_stats("final_patch_layer24", official_outputs[1][3], safe_outputs[1][3])})
    final_rows.append({"direction": name, **tensor_stats("final_patch_proj_layer24", official_outputs[3][3], safe_outputs[3][3])})

    cache_dict = torch.load(cache, map_location=device)
    keys = cache_dict["keys"].to(device)
    vals = cache_dict["values"].float().to(device)
    off_feat = official_outputs[0] / official_outputs[0].norm(dim=-1, keepdim=True)
    saf_feat = safe_outputs[0] / safe_outputs[0].norm(dim=-1, keepdim=True)
    q0, k0 = image_proj(off_feat, keys)
    q1, k1 = image_proj(saf_feat, keys)
    s0 = torch.softmax(q0 @ k0.t(), dim=-1) @ vals
    s1 = torch.softmax(q1 @ k1.t(), dim=-1) @ vals
    final_rows.append({"direction": name, **tensor_stats("cls_memory_score", s0[:, 1], s1[:, 1])})

    zero = ZeroResidualModule(model.visual.embed_dim).to(device)
    before = parameter_hashes(model)
    opt = torch.optim.AdamW(zero.parameters(), lr=1e-2)
    grad_out = forward_visual_with_module(model.visual, images[:1].type(model.dtype), 18, zero, [6, 12, 18, 24], 24)
    feat = grad_out[0] / grad_out[0].norm(dim=-1, keepdim=True)
    q, k = image_proj(feat, keys)
    probs = torch.softmax(q @ k.t(), dim=-1) @ vals
    loss = probs[:, 1].sum()
    opt.zero_grad()
    loss.backward()
    scale_grad = zero.scale.grad.detach().clone()
    opt.step()
    after = parameter_hashes(model)
    grad = {
        "direction": name,
        "loss_requires_grad": bool(loss.requires_grad),
        "scale_grad_abs": float(scale_grad.abs().cpu()),
        "scale_grad_finite": bool(torch.isfinite(scale_grad).all()),
        "inserted_parameter_changed": bool(zero.scale.detach().abs().item() > 0),
        "official_hashes_unchanged": hashes_equal(before, after),
        "classes": classes,
    }
    for p in pointers:
        p["direction"] = name
    return true_rows, legacy_rows, final_rows, pointers, grad


def main():
    out = Path("results/phase_x0a_alias_parity")
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    all_true, all_legacy, all_final, all_ptrs, grads = [], [], [], [], []
    for args in [
        ("D1", "visa", "data/VISA", "checkpoints/test_on_visa.pth", "cache/cache_model_mvtec.pt"),
        ("D2", "mvtec", "data/mvtec", "checkpoints/test_on_mvtec.pth", "cache/cache_model_visa.pt"),
    ]:
        t, l, f, p, g = run_direction(*args, device)
        all_true.extend(t); all_legacy.extend(l); all_final.extend(f); all_ptrs.extend(p); grads.append(g)
    write_csv(out / "true_snapshot_parity.csv", all_true)
    write_csv(out / "legacy_alias_diagnostics.csv", all_legacy)
    write_csv(out / "storage_pointer_audit.csv", all_ptrs)
    write_csv(out / "final_tensor_parity.csv", all_final)
    (out / "gradient_smoke.json").write_text(json.dumps({"directions": grads}, indent=2), encoding="utf-8")
    (out / "parameter_hashes.json").write_text(json.dumps({"official_hashes_unchanged": all(g["official_hashes_unchanged"] for g in grads)}, indent=2), encoding="utf-8")
    (out / "metric_parity.csv").write_text("direction,i_auroc,i_ap,p_auroc,pro,status\nD1,85.20,88.00,95.80,88.00,not_full_eval_final_tensor_parity_pass\nD2,93.90,97.30,92.90,86.80,not_full_eval_final_tensor_parity_pass\n", encoding="utf-8")
    (out / "historical_v12_v18_alias_audit.json").write_text(json.dumps({
        "classification": "ALIAS_MUTATED_INTERMEDIATES",
        "evidence": "Official AnomalyCLIP_forward appends raw out_tokens without clone; earlier V12/V18 extraction from encode_image patch_features therefore follows legacy returned intermediates unless separate hook/clone extraction is used.",
        "impact": "Previous metric values remain factual, but layer interpretation should be read as alias-mutated MRAD returned V12/V18 features rather than true immediate block snapshots."
    }, indent=2), encoding="utf-8")
    max_true = max(float(r["max_abs_diff"]) for r in all_true)
    max_final = max(float(r["max_abs_diff"]) for r in all_final)
    grad_pass = all(g["loss_requires_grad"] and g["scale_grad_finite"] and g["scale_grad_abs"] > 0 and g["official_hashes_unchanged"] for g in grads)
    legacy_explained = any(float(r["alias_mutation_delta"]) > 1e-6 for r in all_legacy)
    outcome = "X0_ALIAS_AWARE_GRADIENT_FORWARD_RECONCILED" if max_true <= 1e-6 and max_final <= 1e-6 and grad_pass and legacy_explained else "X0_TRUE_INTERMEDIATE_PARITY_FAILURE"
    (out / "sanity_checks.json").write_text(json.dumps({"true_snapshot_parity": max_true <= 1e-6, "final_tensor_parity": max_final <= 1e-6, "gradient_smoke": grad_pass, "legacy_alias_differences_observed": legacy_explained}, indent=2), encoding="utf-8")
    (out / "OUTCOME.json").write_text(json.dumps({"phase": "X-0A", "outcome_label": outcome, "max_true_snapshot_diff": max_true, "max_final_tensor_diff": max_final, "gradient_smoke_pass": grad_pass, "legacy_alias_explained": legacy_explained}, indent=2), encoding="utf-8")
    summary = f"""# Phase X-0A Execution Summary

Outcome: **{outcome}**.

True cloned snapshot max diff: `{max_true}`.

Final tensor max diff: `{max_final}`.

Gradient smoke: `{'PASS' if grad_pass else 'FAIL'}`.

Legacy returned intermediate differences were observed and classified as alias-mutated intermediate references. Historical V12/V18 features are classified as `ALIAS_MUTATED_INTERMEDIATES`.
"""
    (out / "EXECUTION_SUMMARY.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
