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

from experiments.phase_x0a_mandatory_gates import evaluate_safe_identity, load_components
from experiments.phase_x0a_sample_parity_and_audit import capture_official, compute_task_tensors, select_sample_indices
from phase_x.gradient_smoke_test import hashes_equal, parameter_hashes
from phase_x.insertion_modules import IdentityInsertion
from phase_x.parity_tests import tensor_stats
from phase_xa.adapted_cache_builder import build_adapted_task_cache
from phase_xa.cache_alignment import assert_cache_alignment
from phase_xa.config import SMOKE_CONFIG
from phase_xa.diagnostics import effective_rank
from phase_xa.distillation_losses import hard_label_loss, memory_distillation_loss, preserve_loss
from phase_xa.memory_teacher_forward import anomaly_margin, teacher_targets
from phase_xa.official_task_forward import (
    image_memory_scores,
    normalized_image_feature,
    normalized_patch_feature,
    normalized_patch_projection,
    patch_memory_scores,
    visual_outputs,
)
from phase_xa.random_teacher import permute_memory_values
from phase_xa.shared_adapter import SharedBlock18Adapter, trainable_parameter_count
from phase_xa.source_folds import deterministic_source_folds
from phase_xa.teacher_memory_builder import build_memory_from_indices
from utils.dataset import Dataset
from utils.transforms import get_transform


OUT = Path("results/phase_xa_memory_distillation")
X0A = Path("results/phase_x0a_alias_parity")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_dataset(dataset_name, data_path):
    preprocess, target_transform = get_transform(argparse.Namespace(image_size=518))
    return Dataset(data_path, preprocess, target_transform, dataset_name)


def indices_for_categories(dataset, categories, limit, require_both=True):
    selected, normal, anomaly = [], 0, 0
    for idx in range(len(dataset)):
        item = dataset[idx]
        if item["cls_name"] not in categories:
            continue
        label = int(item["anomaly"])
        if require_both and label == 0 and normal >= limit // 2:
            continue
        if require_both and label == 1 and anomaly >= limit // 2:
            continue
        selected.append(idx)
        normal += int(label == 0)
        anomaly += int(label == 1)
        if len(selected) >= limit and (not require_both or (normal > 0 and anomaly > 0)):
            break
    if not selected or (require_both and (normal == 0 or anomaly == 0)):
        raise RuntimeError("smoke sample selection failed")
    return selected


def verify_x0a_artifacts():
    required = [
        "true_snapshot_parity.csv",
        "final_tensor_parity.csv",
        "metric_parity.csv",
        "gradient_smoke.json",
        "parameter_hashes.json",
        "sanity_checks.json",
        "OUTCOME.json",
    ]
    rows = []
    for name in required:
        path = X0A / name
        rows.append({"file": str(path), "exists": path.exists(), "non_empty": path.exists() and path.stat().st_size > 0})
    outcome = json.loads((X0A / "OUTCOME.json").read_text(encoding="utf-8"))
    return rows, outcome.get("outcome_label") == "X0_ALIAS_AWARE_GRADIENT_FORWARD_RECONCILED"


def baseline_reproduction_smoke(device):
    refs = {
        "D1": {"image_auroc": 85.206540, "image_ap": 87.973309, "pixel_auroc": 95.813210, "pixel_aupro": 87.970955},
        "D2": {"image_auroc": 93.869501, "image_ap": 97.295042, "pixel_auroc": 92.896640, "pixel_aupro": 86.817999},
    }
    rows = []
    metric_path = X0A / "metric_parity.csv"
    if metric_path.exists():
        with open(metric_path, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r.get("object") == "mean"]
    ok = len(rows) == 2
    report = {}
    for row in rows:
        direction = row["direction"]
        report[direction] = {}
        for key in refs[direction]:
            value = float(row[key])
            delta = value - refs[direction][key]
            report[direction][key] = {"value": value, "reference": refs[direction][key], "delta": delta}
            ok = ok and abs(delta) <= (0.15 if key == "pixel_aupro" else 0.10)
    return {"used_phase_x0a_full_eval": True, "pass": ok, "directions": report}


def run_mvtec_sample_parity(device):
    model, prompt_learner, image_proj, patch_proj, prompt_proj = load_components("checkpoints/test_on_mvtec.pth", device)
    dataset = load_dataset("mvtec", "data/mvtec")
    indices, _ = select_sample_indices(dataset)
    image_cache = torch.load("cache/cache_model_visa.pt", map_location=device)
    patch_cache = torch.load("cache/cache_patch_model_visa.pt", map_location=device)
    image_cache = (image_cache["keys"].to(device), image_cache["values"].to(device))
    patch_cache = (patch_cache["keys"].to(device), patch_cache["values"].to(device))
    rows = []
    for sample_idx in indices:
        item = dataset[sample_idx]
        image = item["img"].unsqueeze(0).to(device)
        mask = item["img_mask"].unsqueeze(0)
        official_outputs, official_snaps = capture_official(model, image)
        safe = visual_outputs(model, image, adapter=IdentityInsertion().to(device), grad=True, snapshot_mode="true_clone")
        safe_outputs = safe["outputs"]
        for name, official, candidate in [
            ("true block-18 state", official_snaps["true_block18"][0] if isinstance(official_snaps["true_block18"], list) else official_snaps["true_block18"], safe["snapshots"]["true_block18"][0] if isinstance(safe["snapshots"]["true_block18"], list) else safe["snapshots"]["true_block18"]),
            ("final branch-0 patches", official_outputs[1][3], safe_outputs[1][3]),
            ("final branch-1 CLS", official_outputs[0], safe_outputs[0]),
            ("patch projection", official_outputs[3][3], safe_outputs[3][3]),
        ]:
            stat = tensor_stats(name, official, candidate)
            stat.update({"sample_idx": sample_idx, "class": item["cls_name"], "label": int(item["anomaly"])})
            rows.append(stat)
        official_tensors = compute_task_tensors(official_outputs, model, prompt_learner, image_proj, patch_proj, prompt_proj, image_cache, patch_cache, item["cls_id"], mask)
        safe_tensors = compute_task_tensors(safe_outputs, model, prompt_learner, image_proj, patch_proj, prompt_proj, image_cache, patch_cache, item["cls_id"], mask)
        for key in ["image_projection", "prompt_similarity_map", "patch_memory_output", "cls_memory_score", "final_k0.7_image_score", "final_gaussian_pixel_map"]:
            stat = tensor_stats(key, official_tensors[key], safe_tensors[key])
            stat.update({"sample_idx": sample_idx, "class": item["cls_name"], "label": int(item["anomaly"])})
            rows.append(stat)
    write_csv(X0A / "mandatory_mvtec_16_sample_parity.csv", rows)
    return {"images": len(indices), "max_abs_diff": max(float(r["max_abs_diff"]) for r in rows)}


def to_device_cache(cache, device):
    return {
        "keys": cache["keys"].to(device),
        "values": cache["values"].to(device),
        "provenance": cache["provenance"],
    }


def patch_mask_37(item, device):
    mask = item["img_mask"].squeeze().to(device)
    mask = (mask > 0.5).float()
    return F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(37, 37), mode="bilinear", align_corners=False).reshape(1, -1)


def teacher_stats(model, dataset, indices, image_proj, patch_proj, image_cache, patch_cache, device, random_patch_cache=None):
    img_posts, patch_posts, margins, top_unique = [], [], [], []
    random_diffs = []
    for idx in indices:
        item = dataset[idx]
        image = item["img"].unsqueeze(0).to(device)
        with torch.no_grad():
            out = visual_outputs(model, image, adapter=None, grad=False)
            img = normalized_image_feature(out)
            patch = normalized_patch_feature(out)
            img_post, img_logits, _, _ = image_memory_scores(img, image_cache["keys"], image_cache["values"], image_proj)
            patch_post, patch_logits, _, _ = patch_memory_scores(patch, patch_cache["keys"], patch_cache["values"], patch_proj)
            img_t = teacher_targets(img_logits, image_cache["values"], image_cache["provenance"], idx)
            patch_t = teacher_targets(patch_logits[:, :32, :].reshape(-1, patch_logits.shape[-1]), patch_cache["values"], patch_cache["provenance"], idx)
            img_posts.append(img_t["anomaly_posterior"].detach().cpu())
            patch_posts.append(patch_t["anomaly_posterior"].detach().cpu())
            margins.append(torch.cat([img_t["margin"].detach().cpu(), patch_t["margin"].detach().cpu().reshape(-1)[:4]]))
            top_unique.append(int(torch.unique(patch_t["top_indices"]).numel()))
            if random_patch_cache is not None:
                rand_post, rand_logits, _, _ = patch_memory_scores(patch, random_patch_cache["keys"], random_patch_cache["values"], patch_proj)
                random_diffs.append(float((rand_post[..., 1] - patch_post[..., 1]).abs().mean().cpu()))
    return {
        "image_posterior_variance": float(torch.cat(img_posts).var().item()),
        "patch_posterior_variance": float(torch.cat(patch_posts).var().item()),
        "margins_finite": bool(torch.isfinite(torch.cat(margins)).all().item()),
        "min_topk_unique_keys": min(top_unique),
        "random_memory_mean_abs_diff": min(random_diffs) if random_diffs else None,
    }


def train_variant(direction, variant, model, dataset, train_indices, val_indices, image_proj, patch_proj, teacher_image_cache, teacher_patch_cache, device, return_adapter=False, random_seed=0):
    adapter = SharedBlock18Adapter().to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=SMOKE_CONFIG["learning_rate"])
    before_hash = parameter_hashes(model)
    metrics = []
    random_patch_cache = to_device_cache(permute_memory_values({k: teacher_patch_cache[k].cpu() if torch.is_tensor(teacher_patch_cache[k]) else teacher_patch_cache[k] for k in teacher_patch_cache}, seed=random_seed), device) if variant.startswith("XA6_") else None
    reference_features, final_features = [], []
    self_excluded = []
    for epoch in range(SMOKE_CONFIG["epochs"]):
        start_cache = time.time()
        image_cache, patch_cache = build_adapted_task_cache(model, dataset, train_indices, direction.lower(), device, adapter, max_items=SMOKE_CONFIG["cache_items_per_epoch"])
        image_cache = to_device_cache(image_cache, device)
        patch_cache = to_device_cache(patch_cache, device)
        assert_cache_alignment(image_cache, patch_cache)
        cache_seconds = time.time() - start_cache
        epoch_losses = []
        optimize_indices = train_indices[: int(SMOKE_CONFIG.get("optimize_items_per_epoch", 4))]
        for idx in optimize_indices:
            item = dataset[idx]
            image = item["img"].unsqueeze(0).to(device)
            optimizer.zero_grad(set_to_none=True)
            adapter.reset_batch_trace()
            with torch.no_grad():
                teacher_out = visual_outputs(model, image, adapter=None, grad=False)
            student_out = visual_outputs(model, image, adapter=adapter, grad=True)
            teacher_img = normalized_image_feature(teacher_out)
            teacher_patch = normalized_patch_feature(teacher_out)
            student_img = normalized_image_feature(student_out)
            student_patch = normalized_patch_feature(student_out)
            image_post, image_logits, _, _ = image_memory_scores(student_img, image_cache["keys"], image_cache["values"], image_proj)
            patch_post, patch_logits, _, _ = patch_memory_scores(student_patch, patch_cache["keys"], patch_cache["values"], patch_proj)
            t_img_post, t_img_logits, _, _ = image_memory_scores(teacher_img, teacher_image_cache["keys"], teacher_image_cache["values"], image_proj)
            teacher_patch_source = random_patch_cache if random_patch_cache is not None else teacher_patch_cache
            t_patch_post, t_patch_logits, _, _ = patch_memory_scores(teacher_patch, teacher_patch_source["keys"], teacher_patch_source["values"], patch_proj)
            t_img = teacher_targets(t_img_logits, teacher_image_cache["values"], teacher_image_cache["provenance"], idx)
            t_patch = teacher_targets(t_patch_logits, teacher_patch_source["values"], teacher_patch_source["provenance"], idx)
            self_excluded.extend([t_img["excluded_keys"], t_patch["excluded_keys"]])
            student_img_margin = anomaly_margin(image_logits, image_cache["values"])
            student_patch_margin = anomaly_margin(patch_logits.reshape(-1, patch_logits.shape[-1]), patch_cache["values"]).mean().reshape(1)
            teacher_patch_margin = t_patch["margin"].mean().reshape(1)
            loss = hard_label_loss(image_post, patch_post, item["anomaly"], patch_mask_37(item, device))
            if variant.startswith(("XA2_", "XA4_", "XA5_", "XA6_", "XA8_")):
                loss = loss + 0.1 * preserve_loss(student_img, teacher_img, student_patch, teacher_patch)
            if variant.startswith(("XA4_", "XA5_", "XA6_", "XA7_")):
                loss = loss + 0.1 * memory_distillation_loss(image_post, t_img["posterior"], patch_post, t_patch_post.detach(), student_img_margin, t_img["margin"], student_patch_margin, teacher_patch_margin)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            reference_features.append(teacher_img.detach().cpu())
            final_features.append(student_img.detach().cpu())
            branch_calls = len(adapter.batch_trace)
            metrics.append({
                "direction": direction,
                "variant": variant,
                "epoch": epoch,
                "sample_idx": idx,
                "loss": epoch_losses[-1],
                "branch_calls": branch_calls,
                "branch0_input_norm": adapter.batch_trace[0]["input_norm"] if branch_calls else "",
                "branch1_input_norm": adapter.batch_trace[1]["input_norm"] if branch_calls > 1 else "",
                "branch0_residual_norm": adapter.batch_trace[0]["residual_norm"] if branch_calls else "",
                "branch1_residual_norm": adapter.batch_trace[1]["residual_norm"] if branch_calls > 1 else "",
                "gate": float(adapter.gate.detach().cpu()),
                "cache_rebuild_seconds": cache_seconds,
            })
    after_hash = parameter_hashes(model)
    official_unchanged = hashes_equal(before_hash, after_hash)
    grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters() if p.requires_grad)
    official_grads_none = all(p.grad is None for p in model.parameters())
    ref = torch.cat(reference_features, dim=0)
    stu = torch.cat(final_features, dim=0)
    rank_ref = effective_rank(ref)
    rank_stu = effective_rank(stu)
    cosine = float(F.cosine_similarity(ref.reshape(-1, ref.shape[-1]), stu.reshape(-1, stu.shape[-1]), dim=-1).mean())
    validation_complete = False
    with torch.no_grad():
        for idx in val_indices[:2]:
            item = dataset[idx]
            image = item["img"].unsqueeze(0).to(device)
            out = visual_outputs(model, image, adapter=adapter, grad=False)
            img = normalized_image_feature(out)
            _ = image_memory_scores(img, image_cache["keys"], image_cache["values"], image_proj)
        validation_complete = True
    diag = {
        "direction": direction,
        "variant": variant,
        "trainable_parameters": trainable_parameter_count(adapter),
        "shared_adapter_branch_calls_ok": all(int(r["branch_calls"]) >= 2 for r in metrics[-len(train_indices[: int(SMOKE_CONFIG.get("optimize_items_per_epoch", 4))]) * SMOKE_CONFIG["epochs"]:]),
        "adapter_gradients_ok": bool(grad_ok),
        "official_parameters_unchanged": official_unchanged,
        "official_parameter_grads_none": official_grads_none,
        "self_exclusion_active": max(self_excluded) > 0,
        "validation_inference_complete": validation_complete,
        "no_nan_inf": all(torch.isfinite(torch.tensor(float(r["loss"]))) for r in metrics),
        "effective_rank_teacher": rank_ref,
        "effective_rank_student": rank_stu,
        "rank_collapse_fraction": 0.0 if rank_ref == 0 else max(0.0, (rank_ref - rank_stu) / rank_ref),
        "final_original_clip_cosine": cosine,
    }
    if return_adapter:
        return metrics, diag, adapter
    return metrics, diag


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checkpoints").mkdir(exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SMOKE_CONFIG["seed"])
    x0a_files, x0a_ok = verify_x0a_artifacts()
    write_csv(OUT / "x0a_artifact_check.csv", x0a_files)
    baseline = baseline_reproduction_smoke(device)
    (OUT / "baseline_reproduction.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    mvtec_parity = run_mvtec_sample_parity(device)

    fold_info = {
        "mvtec": deterministic_source_folds("mvtec"),
        "visa": deterministic_source_folds("visa"),
    }
    (OUT / "source_folds.json").write_text(json.dumps(fold_info, indent=2), encoding="utf-8")
    (OUT / "source_protocol.json").write_text(json.dumps({
        "protocol": "checkpoint-conditioned source category-held-out validation",
        "target_labels_used_before_final_evaluation": False,
        "smoke_gate_only": True,
    }, indent=2), encoding="utf-8")

    all_metrics, diagnostics = [], {}
    cache_alignments = []
    directions = [
        ("D1", "mvtec", "data/mvtec", "checkpoints/test_on_visa.pth"),
        ("D2", "visa", "data/VISA", "checkpoints/test_on_mvtec.pth"),
    ]
    for direction, dataset_name, data_path, ckpt in directions:
        model, _, image_proj, patch_proj, _ = load_components(ckpt, device)
        dataset = load_dataset(dataset_name, data_path)
        rotation = fold_info[dataset_name]["rotations"][0]
        train_indices = indices_for_categories(dataset, set(rotation["train_categories"]), SMOKE_CONFIG["train_items_per_direction"])
        val_indices = indices_for_categories(dataset, set(rotation["val_categories"]), SMOKE_CONFIG["val_items_per_direction"], require_both=False)
        image_cache, patch_cache = build_memory_from_indices(model, dataset, train_indices, dataset_name, device, adapter=None, max_items=SMOKE_CONFIG["teacher_items"])
        alignment = assert_cache_alignment(image_cache, patch_cache)
        alignment.update({"direction": direction, "image_keys": image_cache["keys"].shape[0], "patch_keys": patch_cache["keys"].shape[0]})
        cache_alignments.append(alignment)
        image_cache = to_device_cache(image_cache, device)
        patch_cache = to_device_cache(patch_cache, device)
        random_patch = to_device_cache(permute_memory_values({k: patch_cache[k].cpu() if torch.is_tensor(patch_cache[k]) else patch_cache[k] for k in patch_cache}, seed=0), device)
        tstats = teacher_stats(model, dataset, train_indices[:4], image_proj, patch_proj, image_cache, patch_cache, device, random_patch)
        diagnostics[f"{direction}_teacher"] = tstats
        for variant in ["XA1_HardLabelAdapter", "XA2_CLIPPreservedAdapter", "XA4_DualTeacherFullMemory", "XA6_RandomMemoryTeacher"]:
            rows, diag = train_variant(direction, variant, model, dataset, train_indices, val_indices, image_proj, patch_proj, image_cache, patch_cache, device)
            all_metrics.extend(rows)
            diagnostics[f"{direction}_{variant}"] = diag
    write_csv(OUT / "smoke_metrics.csv", all_metrics)
    write_csv(OUT / "cache_alignment.csv", cache_alignments)
    (OUT / "cache_alignment.json").write_text(json.dumps({"entries": cache_alignments, "pass": all(r["image_cache_aligned"] and r["patch_cache_aligned"] for r in cache_alignments)}, indent=2), encoding="utf-8")

    checks = {
        "x0a_outcome_ok": x0a_ok,
        "baseline_safe_path_reproduces": baseline["pass"],
        "mvtec_sample_parity_pass": mvtec_parity["max_abs_diff"] <= 1e-6,
        "teacher_image_posterior_nonzero_variance": all(v["image_posterior_variance"] > 0 for k, v in diagnostics.items() if k.endswith("_teacher")),
        "teacher_patch_posterior_nonzero_variance": all(v["patch_posterior_variance"] > 0 for k, v in diagnostics.items() if k.endswith("_teacher")),
        "teacher_margins_finite": all(v["margins_finite"] for k, v in diagnostics.items() if k.endswith("_teacher")),
        "teacher_topk_multiple_keys": all(v["min_topk_unique_keys"] > 1 for k, v in diagnostics.items() if k.endswith("_teacher")),
        "random_memory_output_differs": all(v["random_memory_mean_abs_diff"] and v["random_memory_mean_abs_diff"] > 1e-8 for k, v in diagnostics.items() if k.endswith("_teacher")),
        "all_variants_shared_two_branch_adapter": all(v.get("shared_adapter_branch_calls_ok", True) for v in diagnostics.values()),
        "all_variants_adapter_gradients": all(v.get("adapter_gradients_ok", True) for v in diagnostics.values()),
        "official_parameters_unchanged": all(v.get("official_parameters_unchanged", True) for v in diagnostics.values()),
        "official_parameter_grads_none": all(v.get("official_parameter_grads_none", True) for v in diagnostics.values()),
        "self_exclusion_active": all(v.get("self_exclusion_active", True) for v in diagnostics.values()),
        "epoch_adapted_cache_rebuild_succeeds": True,
        "cache_key_value_provenance_alignment_passes": all(r["image_cache_aligned"] and r["patch_cache_aligned"] for r in cache_alignments),
        "validation_inference_completes": all(v.get("validation_inference_complete", True) for v in diagnostics.values()),
        "no_nan_inf": all(v.get("no_nan_inf", True) for v in diagnostics.values()),
        "effective_rank_not_collapsed_gt_30pct": all(v.get("rank_collapse_fraction", 0.0) <= 0.30 for v in diagnostics.values()),
        "final_original_clip_cosine_ge_0.85": all(v.get("final_original_clip_cosine", 1.0) >= 0.85 for v in diagnostics.values()),
    }
    passed = all(checks.values())
    diagnostics["smoke_checks"] = checks
    diagnostics["mvtec_sample_parity"] = mvtec_parity
    diagnostics["outcome"] = "SMOKE_PASS" if passed else "XA_K_IMPLEMENTATION_FAILURE"
    (OUT / "smoke_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    (OUT / "sanity_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    (OUT / "OUTCOME.json").write_text(json.dumps({"phase": "X-A smoke", "outcome_label": diagnostics["outcome"], "stage1_started": False, "target_evaluation_started": False}, indent=2), encoding="utf-8")
    report = [
        "# Phase X-A Smoke Gate Report",
        "",
        f"Outcome: **{diagnostics['outcome']}**.",
        "",
        f"Baseline reproduction reused accepted X-0A full safe-Identity metrics: `{baseline['pass']}`.",
        f"MVTec 16-sample parity max diff: `{mvtec_parity['max_abs_diff']}`.",
        "",
        "Smoke variants run for D1 source MVTec and D2 source VisA, fold 0, seed 0, 2 epochs, no target evaluation:",
        "",
        "- XA1_HardLabelAdapter",
        "- XA2_CLIPPreservedAdapter",
        "- XA4_DualTeacherFullMemory",
        "- XA6_RandomMemoryTeacher",
        "",
        "Mandatory smoke checks:",
    ]
    for key, value in checks.items():
        report.append(f"- {key}: `{value}`")
    report.extend(["", "Stage 1 was not started."])
    (OUT / "SMOKE_GATE_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    Path("docs/phase_xa_memory_distillation_path.md").write_text(
        "# Phase X-A Memory Distillation Path\n\n"
        "- Uses `phase_x.gradient_visual_forward.forward_visual_with_module`.\n"
        "- Inserts one shared block-18 adapter recursively on both DAPM branches.\n"
        "- Reuses frozen official image/patch projection modules; no new projection is introduced.\n"
        "- Teacher memory is built from original frozen source-training categories only.\n"
        "- Adapted student task cache is rebuilt from adapted features and detached each epoch.\n"
        "- Smoke gate uses source category-held-out fold 0 only and does not evaluate target data.\n",
        encoding="utf-8",
    )
    if not passed:
        raise SystemExit("XA_K_IMPLEMENTATION_FAILURE")


if __name__ == "__main__":
    main()
