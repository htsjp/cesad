import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_stage1 import VARIANTS, all_indices_for_categories, read_existing, run_key
from experiments.phase_xa_smoke import load_dataset, patch_mask_37, to_device_cache, write_csv
from phase_x.gradient_smoke_test import parameter_hashes
from phase_xa.adapted_cache_builder import build_adapted_task_cache
from phase_xa.cache_alignment import assert_cache_alignment
from phase_xa.config import SMOKE_CONFIG
from phase_xa.distillation_losses import hard_label_loss, memory_distillation_loss, preserve_loss
from phase_xa.memory_teacher_forward import anomaly_margin, teacher_targets
from phase_xa.official_task_forward import (
    image_memory_scores,
    normalized_image_feature,
    normalized_patch_feature,
    patch_memory_scores,
    visual_outputs,
)
from phase_xa.random_teacher import permute_memory_values
from phase_xa.shared_adapter import SharedBlock18Adapter
from phase_xa.source_folds import deterministic_source_folds
from phase_xa.teacher_memory_builder import build_memory_from_indices


OUT = Path("results/phase_xa_memory_distillation")


def sha_tensor_state(state):
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        h.update(key.encode("utf-8"))
        if torch.is_tensor(value):
            arr = value.detach().cpu().contiguous()
            h.update(str(tuple(arr.shape)).encode("utf-8"))
            h.update(str(arr.dtype).encode("utf-8"))
            h.update(arr.numpy().tobytes())
        else:
            h.update(repr(value).encode("utf-8"))
    return h.hexdigest()


def completed_rows():
    rows = read_existing(OUT / "stage1_source_metrics.csv")
    return [r for r in rows if r.get("status") == "COMPLETE"]


def next_row(rows):
    done = {run_key(r["direction"], r["fold"], r["variant"], r["seed"]) for r in rows}
    random_seeds = [0, 1, 2, 3, 4]
    for direction in ["D1", "D2"]:
        for fold in [0, 1, 2]:
            for variant in VARIANTS:
                seeds = random_seeds if variant.startswith("XA6_") else [0]
                for seed in seeds:
                    key = run_key(direction, fold, variant, seed)
                    if key not in done:
                        return {"direction": direction, "fold": fold, "variant": variant, "seed": seed}
    return None


def runner_state():
    # The caller has stopped the long-running process before this audit.
    rows = completed_rows()
    last_row = rows[-1] if rows else None
    local_log = max(OUT.glob("stage1_local_resume_*.log"), key=lambda p: p.stat().st_mtime, default=None)
    current = next_row(rows)
    state = {
        "active_pid": None,
        "process_safely_stopped": False,
        "stop_reason": "active local runner exceeded normal row duration without writing a new row; stopped before Batch A to prevent uncontrolled continuation",
        "last_completed_row": last_row,
        "inferred_current_or_next_row": current,
        "current_row_epoch": None,
        "last_checkpoint": last_row.get("checkpoint") if last_row else None,
        "current_row_files_written": [],
        "latest_local_log": str(local_log) if local_log else None,
        "completed_rows": len(rows),
    }
    (OUT / "COMPUTE_EFFICIENT_CONTINUATION_STATE.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def source_validation_call_path_audit():
    return {
        "stage1_file": "experiments/phase_xa_stage1.py",
        "training_checkpoint_write": "main() -> train_variant(..., return_adapter=True) -> torch.save({'adapter', metadata}, checkpoint_path)",
        "held_out_source_input": "main() builds val_indices from deterministic_source_folds(...)[fold]['val_categories']",
        "adapted_visual_forward": "phase_xa.official_task_forward.visual_outputs(model, image, adapter=adapter, grad=False)",
        "adapted_source_memory_cache": "train_variant() rebuilds image_cache/patch_cache from train_indices only via build_adapted_task_cache() at each epoch",
        "official_prompt_path": "not invoked by Stage-1 source validation; only baseline X-0A prompt parity exists elsewhere",
        "official_image_memory_score": "train_variant() validation smoke calls image_memory_scores() for val_indices[:2]",
        "official_patch_memory_path": "not invoked for full validation metrics in Stage-1",
        "final_image_score": "not computed in Stage-1 source validation",
        "final_pixel_map": "not computed in Stage-1 source validation",
        "source_validation_metrics": "not computed; no I-AUROC/I-AP/P-AUROC/PRO source-category readout is invoked",
        "metric_functions_available": [
            "utils.metrics.image_level_metrics",
            "utils.metrics.pixel_level_metrics",
            "experiments.mrad_audit_utils.evaluate_results",
        ],
        "conclusion": "SOURCE_VALIDATION_METRIC_CODE_NOT_INVOKED_BY_PHASE_XA_STAGE1",
        "required_next_readout": "implement/reuse a non-training checkpoint source readout before deciding continuation",
    }


def fold_leakage_audit():
    rows = []
    summaries = []
    for dataset_name, data_path in [("mvtec", "data/mvtec"), ("visa", "data/VISA")]:
        folds = deterministic_source_folds(dataset_name)
        dataset = load_dataset(dataset_name, data_path)
        for rot in folds["rotations"]:
            train = set(rot["train_categories"])
            val = set(rot["val_categories"])
            train_indices = all_indices_for_categories(dataset, train)
            val_indices = all_indices_for_categories(dataset, val)
            train_item_classes = {dataset[i]["cls_name"] for i in train_indices}
            val_item_classes = {dataset[i]["cls_name"] for i in val_indices}
            overlap = sorted(train & val)
            item_overlap = sorted(train_item_classes & val_item_classes)
            ok = not overlap and not item_overlap
            rows.append({
                "dataset": dataset_name,
                "fold": rot["fold"],
                "train_categories": sorted(train),
                "val_categories": sorted(val),
                "category_overlap": overlap,
                "item_class_overlap": item_overlap,
                "train_items": len(train_indices),
                "val_items": len(val_indices),
                "memory_indices_source": "train_indices only in stage1 build_memory_from_indices/build_adapted_task_cache",
                "leakage_free": ok,
            })
            summaries.append(ok)
    payload = {"pass": all(summaries), "folds": rows}
    (OUT / "source_fold_leakage_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def checkpoint_inventory(rows):
    inv = []
    ckpt_dir = OUT / "checkpoints" / "stage1"
    for r in rows:
        checkpoint = r.get("checkpoint", "")
        row = {
            "direction": r.get("direction"),
            "source_dataset": r.get("source_dataset"),
            "fold": r.get("fold"),
            "variant": r.get("variant"),
            "seed": r.get("seed"),
            "csv_checkpoint": checkpoint,
            "checkpoint_exists": "",
            "checkpoint_non_empty": "",
            "adapter_state_present": "",
            "metadata_matches_csv": "",
            "strict_adapter_load_ok": "",
            "sha256": "",
            "config_epochs": "",
            "config_lr": "",
            "final_cache_rebuild_status": "not_rebuilt_in_batch_a_compute_guard",
        }
        if r.get("variant", "").startswith("XA0_"):
            row.update({
                "checkpoint_exists": "N/A_BASELINE",
                "checkpoint_non_empty": "N/A_BASELINE",
                "adapter_state_present": "N/A_BASELINE",
                "metadata_matches_csv": "N/A_BASELINE",
                "strict_adapter_load_ok": "N/A_BASELINE",
            })
            inv.append(row)
            continue
        norm = checkpoint.replace("\\", "/")
        path = Path(norm)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            alt = ckpt_dir / f"{r['direction']}_fold{r['fold']}_{r['variant']}_seed{r['seed']}.pt"
            path = alt
        row["resolved_checkpoint"] = str(path)
        row["checkpoint_exists"] = path.exists()
        row["checkpoint_non_empty"] = path.exists() and path.stat().st_size > 0
        if path.exists() and path.stat().st_size > 0:
            try:
                payload = torch.load(path, map_location="cpu")
                state = payload.get("adapter", {})
                row["adapter_state_present"] = bool(state)
                row["metadata_matches_csv"] = (
                    str(payload.get("direction")) == str(r.get("direction"))
                    and str(payload.get("fold")) == str(r.get("fold"))
                    and str(payload.get("variant")) == str(r.get("variant"))
                    and str(payload.get("seed")) == str(r.get("seed"))
                )
                adapter = SharedBlock18Adapter()
                adapter.load_state_dict(state, strict=True)
                row["strict_adapter_load_ok"] = True
                row["sha256"] = sha_tensor_state(state)
                cfg = payload.get("config", {})
                row["config_epochs"] = cfg.get("epochs", "")
                row["config_lr"] = cfg.get("lr", "")
            except Exception as exc:
                row["strict_adapter_load_ok"] = False
                row["error"] = repr(exc)
        inv.append(row)
    write_csv(OUT / "existing_checkpoint_inventory.csv", inv)
    return inv


def xa1_xa3_audit():
    return {
        "XA1_XA3_DUPLICATE_BY_PROTOCOL": True,
        "architecture": "Both use SharedBlock18Adapter with identical initialization after torch.manual_seed(seed).",
        "loss_path": "train_variant() starts with hard_label_loss for all variants; XA1 and XA3 do not match preserve_loss or memory_distillation_loss prefixes.",
        "preservation_loss": "not applied to XA1 or XA3",
        "teacher_loss": "not applied to XA1 or XA3",
        "optimizer": "AdamW over same adapter parameters at same lr",
        "sampler": "same train_indices order and optimize_items_per_epoch",
        "seed": "same seed=0 for non-XA6 variants",
        "recommendation": "do not spend further compute running both XA1 and XA3",
    }


def grad_norm(params):
    vals = []
    nz = 0
    total = 0
    flat = []
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach().float().reshape(-1)
        flat.append(g)
        vals.append(float(g.norm().item()))
        nz += int((g.abs() > 0).sum().item())
        total += g.numel()
    vec = torch.cat(flat) if flat else torch.zeros(1)
    return float(vec.norm().item()), (float(nz) / float(total) if total else 0.0), vec


def xa8_gradient_audit(device):
    torch.manual_seed(0)
    model, _, image_proj, patch_proj, _ = load_components("checkpoints/test_on_visa.pth", device)
    dataset = load_dataset("mvtec", "data/mvtec")
    folds = deterministic_source_folds("mvtec")
    train = all_indices_for_categories(dataset, set(folds["rotations"][0]["train_categories"]))
    idx = train[0]
    item = dataset[idx]
    image = item["img"].unsqueeze(0).to(device)
    adapter = SharedBlock18Adapter().to(device)

    teacher_cache_i, teacher_cache_p = build_memory_from_indices(model, dataset, train[:8], "mvtec", device, adapter=None, max_items=8)
    adapted_i, adapted_p = build_adapted_task_cache(model, dataset, train[:8], "mvtec", device, adapter, max_items=8)
    image_cache = to_device_cache(adapted_i, device)
    patch_cache = to_device_cache(adapted_p, device)
    teacher_i = to_device_cache(teacher_cache_i, device)
    teacher_p = to_device_cache(teacher_cache_p, device)

    with torch.no_grad():
        teacher_out = visual_outputs(model, image, adapter=None, grad=False)
    student_out = visual_outputs(model, image, adapter=adapter, grad=True)
    teacher_img = normalized_image_feature(teacher_out)
    teacher_patch = normalized_patch_feature(teacher_out)
    student_img = normalized_image_feature(student_out)
    student_patch = normalized_patch_feature(student_out)
    image_post, _, _, _ = image_memory_scores(student_img, image_cache["keys"], image_cache["values"], image_proj)
    patch_post, _, _, _ = patch_memory_scores(student_patch, patch_cache["keys"], patch_cache["values"], patch_proj)
    task = hard_label_loss(image_post, patch_post, item["anomaly"], patch_mask_37(item, device))
    crop = preserve_loss(student_img, teacher_img, student_patch, teacher_patch)

    adapter.zero_grad(set_to_none=True)
    crop.backward(retain_graph=True)
    crop_norm, crop_frac, crop_vec = grad_norm(adapter.parameters())
    adapter.zero_grad(set_to_none=True)
    task.backward()
    task_norm, task_frac, task_vec = grad_norm(adapter.parameters())
    m = min(crop_vec.numel(), task_vec.numel())
    cosine = float(F.cosine_similarity(crop_vec[:m], task_vec[:m], dim=0).item()) if m else float("nan")
    label = "XA8_ACTIVE_TEACHER" if crop_norm > 1e-8 and crop_frac > 0 else "XA8_NEAR_ZERO_TEACHER"
    return {
        "sample_idx": idx,
        "class": item["cls_name"],
        "crop_loss_value": float(crop.detach().cpu().item()),
        "task_loss_value": float(task.detach().cpu().item()),
        "grad_crop_norm": crop_norm,
        "grad_task_norm": task_norm,
        "nonzero_crop_parameter_gradient_fraction": crop_frac,
        "nonzero_task_parameter_gradient_fraction": task_frac,
        "crop_task_gradient_cosine": cosine,
        "classification": label,
        "note": "XA8 uses preserve_loss in code; no crop augmentation branch is present in train_variant.",
    }


def corr(a, b):
    a = torch.as_tensor(a).float().reshape(-1)
    b = torch.as_tensor(b).float().reshape(-1)
    if a.numel() != b.numel() or a.numel() < 2:
        return float("nan")
    if float(a.std().item()) == 0.0 or float(b.std().item()) == 0.0:
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1].item())


def xa6_randomization_audit(device):
    torch.manual_seed(0)
    model, _, image_proj, patch_proj, _ = load_components("checkpoints/test_on_visa.pth", device)
    dataset = load_dataset("mvtec", "data/mvtec")
    folds = deterministic_source_folds("mvtec")
    train = all_indices_for_categories(dataset, set(folds["rotations"][0]["train_categories"]))
    image_cache, patch_cache = build_memory_from_indices(model, dataset, train[:24], "mvtec", device, adapter=None, max_items=24)
    random_patch = permute_memory_values({
        "keys": patch_cache["keys"],
        "values": patch_cache["values"],
        "provenance": patch_cache["provenance"],
    }, seed=0)
    perm = torch.tensor(random_patch["permutation"])
    changed = float((perm != torch.arange(perm.numel())).float().mean().item())
    labels = patch_cache["values"][:, 1].float()
    rand_labels = random_patch["values"][:, 1].float()
    label_changed = float((labels != rand_labels).float().mean().item())
    perm_hash = hashlib.sha256(bytes(str(random_patch["permutation"]), "utf-8")).hexdigest()
    real_posts, rand_posts, real_margins, rand_margins, agreements = [], [], [], [], []
    real_cache = to_device_cache(patch_cache, device)
    rand_cache = to_device_cache(random_patch, device)
    for idx in train[:4]:
        item = dataset[idx]
        image = item["img"].unsqueeze(0).to(device)
        with torch.no_grad():
            out = visual_outputs(model, image, adapter=None, grad=False)
            patch = normalized_patch_feature(out)
            _, real_logits, _, _ = patch_memory_scores(patch, real_cache["keys"], real_cache["values"], patch_proj)
            _, rand_logits, _, _ = patch_memory_scores(patch, rand_cache["keys"], rand_cache["values"], patch_proj)
            rt = teacher_targets(real_logits[:, :32, :].reshape(-1, real_logits.shape[-1]), real_cache["values"], real_cache["provenance"], idx)
            qt = teacher_targets(rand_logits[:, :32, :].reshape(-1, rand_logits.shape[-1]), rand_cache["values"], rand_cache["provenance"], idx)
            real_posts.append(rt["anomaly_posterior"].detach().cpu())
            rand_posts.append(qt["anomaly_posterior"].detach().cpu())
            real_margins.append(rt["margin"].detach().cpu().reshape(-1))
            rand_margins.append(qt["margin"].detach().cpu().reshape(-1))
            agreements.append(float((rt["top_indices"].detach().cpu() == qt["top_indices"].detach().cpu()).float().mean().item()))
    real_post = torch.cat(real_posts).reshape(-1)
    rand_post = torch.cat(rand_posts).reshape(-1)
    real_margin = torch.cat(real_margins).reshape(-1)
    rand_margin = torch.cat(rand_margins).reshape(-1)
    return {
        "memory_keys": int(patch_cache["keys"].shape[0]),
        "normal_value_ratio": float((labels == 0).float().mean().item()),
        "anomaly_value_ratio": float((labels == 1).float().mean().item()),
        "permutation_hash": perm_hash,
        "percent_key_value_assignments_changed": changed * 100.0,
        "percent_label_values_changed": label_changed * 100.0,
        "teacher_posterior_correlation_real_vs_random": corr(real_post, rand_post),
        "teacher_margin_correlation_real_vs_random": corr(real_margin, rand_margin),
        "topk_identity_agreement": float(sum(agreements) / len(agreements)),
        "random_output_variance": float(rand_post.var().item()),
        "classification": "XA6_RANDOMIZATION_OK" if changed > 0.90 and label_changed > 0.05 else "XA6_RANDOMIZATION_SUSPECT",
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    state = runner_state()
    source_path = source_validation_call_path_audit()
    leakage = fold_leakage_audit()
    rows = completed_rows()
    inventory = checkpoint_inventory(rows)
    xa1_xa3 = xa1_xa3_audit()
    start = time.time()
    xa8 = xa8_gradient_audit(device)
    xa6 = xa6_randomization_audit(device)
    audit_seconds = time.time() - start

    usable = [
        r for r in inventory
        if r.get("strict_adapter_load_ok") is True
        or r.get("strict_adapter_load_ok") == "N/A_BASELINE"
    ]
    issues = []
    if not leakage["pass"]:
        issues.append("source split leakage")
    if any(r.get("strict_adapter_load_ok") is False for r in inventory):
        issues.append("checkpoint load failure")
    if xa6["classification"] != "XA6_RANDOMIZATION_OK":
        issues.append("XA6 randomization suspect")
    if xa8["classification"] == "XA8_DISCONNECTED_IMPLEMENTATION":
        issues.append("XA8 disconnected")
    outcome = "A_PASS_EXISTING_CHECKPOINTS_READABLE" if not issues else "A_IMPLEMENTATION_ISSUE_FOUND"
    if not leakage["pass"]:
        outcome = "A_SOURCE_SPLIT_LEAKAGE_FOUND"

    report = [
        "# Phase X-A Audit Batch A Report",
        "",
        f"Outcome: **{outcome}**",
        "",
        "## 1. Runner State",
        "",
        f"- Active PID after control: `{state['active_pid']}`",
        f"- Completed rows: `{state['completed_rows']}`",
        f"- Last completed row: `{state['last_completed_row']}`",
        f"- Inferred next/current row: `{state['inferred_current_or_next_row']}`",
        f"- Safe checkpoint-boundary stop: `{state['process_safely_stopped']}`",
        f"- Stop reason: {state['stop_reason']}",
        "",
        "## 2. Source-Validation Call Path",
        "",
        f"- Conclusion: `{source_path['conclusion']}`",
        "- Stage-1 currently performs only a two-image validation-inference smoke check, not held-out source I-AUROC/I-AP/P-AUROC/PRO.",
        "- Metric utilities exist and can be reused for a non-training checkpoint readout.",
        "",
        "## 3. Fold Leakage",
        "",
        f"- Leakage-free: `{leakage['pass']}`",
        "- Memory and adapted cache builders in Stage-1 are called with `train_indices` only.",
        "",
        "## 4. Checkpoint Inventory",
        "",
        f"- Completed rows inspected: `{len(rows)}`",
        f"- Usable checkpoint/baseline rows: `{len(usable)}`",
        f"- Inventory file: `existing_checkpoint_inventory.csv`",
        "",
        "## 5. XA1 vs XA3",
        "",
        f"- XA1_XA3_DUPLICATE_BY_PROTOCOL: `{xa1_xa3['XA1_XA3_DUPLICATE_BY_PROTOCOL']}`",
        "- Both use only hard-label loss with the same adapter, optimizer, seed, data order and learning rate.",
        "",
        "## 6. XA8 Gradient",
        "",
        f"- Classification: `{xa8['classification']}`",
        f"- Crop/preserve loss: `{xa8['crop_loss_value']:.6f}`",
        f"- ||grad(L_crop)||: `{xa8['grad_crop_norm']:.6e}`",
        f"- Nonzero parameter-gradient fraction: `{xa8['nonzero_crop_parameter_gradient_fraction']:.6f}`",
        f"- Crop/task gradient cosine: `{xa8['crop_task_gradient_cosine']:.6f}`",
        "- Note: XA8 currently adds `preserve_loss`; no distinct crop augmentation teacher branch is present in `train_variant()`.",
        "",
        "## 7. XA6 Randomization",
        "",
        f"- Classification: `{xa6['classification']}`",
        f"- Memory keys: `{xa6['memory_keys']}`",
        f"- Assignments changed: `{xa6['percent_key_value_assignments_changed']:.2f}%`",
        f"- Label values changed: `{xa6['percent_label_values_changed']:.2f}%`",
        f"- Posterior corr real/random: `{xa6['teacher_posterior_correlation_real_vs_random']:.6f}`",
        f"- Margin corr real/random: `{xa6['teacher_margin_correlation_real_vs_random']:.6f}`",
        f"- Top-k identity agreement: `{xa6['topk_identity_agreement']:.6f}`",
        "",
        "## 8. Cheaply Evaluable Existing Checkpoints",
        "",
    ]
    for r in usable:
        report.append(f"- {r.get('direction')} fold{r.get('fold')} {r.get('variant')} seed{r.get('seed')}")
    report.extend([
        "",
        "## 9. Batch B Cost Estimate",
        "",
        "- Batch B is inference-only and should reuse 8 existing D1 fold0 rows.",
        "- It will need one model load, one training-category memory/cache build per adapter, and held-out source validation inference.",
        "- Expected cost on this local GPU is likely hours, but far below retraining; estimate `2-5 GPU-hours` depending on PRO computation and whether per-category maps are cached.",
        "",
        "## Files Written",
        "",
        "- `COMPUTE_EFFICIENT_CONTINUATION_STATE.json`",
        "- `source_fold_leakage_audit.json`",
        "- `existing_checkpoint_inventory.csv`",
        "- `AUDIT_BATCH_A_REPORT.md`",
        "",
        f"Audit probes runtime: `{audit_seconds:.1f}` seconds.",
    ])
    (OUT / "AUDIT_BATCH_A_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (OUT / "AUDIT_BATCH_A_DETAILS.json").write_text(json.dumps({
        "outcome": outcome,
        "runner_state": state,
        "source_validation_call_path": source_path,
        "source_fold_leakage": leakage,
        "xa1_xa3": xa1_xa3,
        "xa8_gradient": xa8,
        "xa6_randomization": xa6,
        "usable_existing_rows": len(usable),
        "issues": issues,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"outcome": outcome, "report": str(OUT / "AUDIT_BATCH_A_REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
