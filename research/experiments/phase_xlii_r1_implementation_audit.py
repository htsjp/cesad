import ast
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import (
    BATCH_SIZE,
    EPOCHS,
    EXPERTS,
    LR,
    OUT as XLII_OUT,
    PATCH_SAMPLE_PER_CLASS_CATEGORY,
    RANK,
    SEED,
    TOPK,
    WEIGHT_DECAY,
    load_fold,
)
from experiments.phase_xxxviii_visual_state_representation import FEATURE_CACHE, class_category_patch_sample, patch_labels


OUT = XLII_OUT / "r1_implementation_audit"
SCRIPT = ROOT / "experiments" / "phase_xlii_patch_specialized_adaptation.py"
PILOT = [("D1", 0), ("D2", 0)]
DIM = 1024


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def class_lines():
    text = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(text)
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            out[node.name] = {"start": node.lineno, "end": getattr(node, "end_lineno", node.lineno)}
    return out


def param_counts():
    r1_lora = 2 * DIM * RANK
    r1_head = 2 * DIM
    r2_lora = 2 * EXPERTS * DIM * RANK
    r2_router = DIM * EXPERTS + EXPERTS
    r2_head = 2 * DIM
    return {
        "R0_FROZEN_BASELINE": {"complete_model_parameters": None, "trainable_parameters": 0, "lora_parameters": 0, "router_parameters": 0, "extra_head_parameters": 0, "trainable_ratio": None},
        "R1_UNIFORM_LORA": {"complete_model_parameters": None, "trainable_parameters": r1_lora + r1_head, "lora_parameters": r1_lora, "router_parameters": 0, "extra_head_parameters": r1_head, "trainable_ratio": None},
        "R2_NAIVE_ROUTED_MOE": {"complete_model_parameters": None, "trainable_parameters": r2_lora + r2_router + r2_head, "lora_parameters": r2_lora, "router_parameters": r2_router, "extra_head_parameters": r2_head, "trainable_ratio": None},
    }


def source_composition():
    rows = []
    for direction, fold in PILOT:
        _dataset_name, dataset, rotation, train_indices, _held_indices = load_fold(direction, fold)
        cache = np.load(FEATURE_CACHE / f"{direction}_fold{fold}_clip.npz", allow_pickle=True)
        train_patch = cache["train_patch"]
        image_counts = defaultdict(lambda: {"normal_images": 0, "anomaly_images": 0})
        full_patch_counts = defaultdict(lambda: {"normal_patches_available": 0, "anomaly_patches_available": 0})
        for pos, idx in enumerate(train_indices):
            item = dataset[idx]
            cat = str(item["cls_name"])
            if int(item["anomaly"]) == 0:
                image_counts[cat]["normal_images"] += 1
            else:
                image_counts[cat]["anomaly_images"] += 1
            labs = patch_labels(item)
            full_patch_counts[cat]["normal_patches_available"] += int((labs == 0).sum())
            full_patch_counts[cat]["anomaly_patches_available"] += int((labs == 1).sum())
        x, y, cats = class_category_patch_sample(dataset, train_indices, train_patch, PATCH_SAMPLE_PER_CLASS_CATEGORY, seed=SEED)
        sampled = defaultdict(lambda: {"normal_patches_sampled": 0, "anomaly_patches_sampled": 0})
        for label, cat in zip(y.tolist(), cats.tolist()):
            if int(label) == 0:
                sampled[str(cat)]["normal_patches_sampled"] += 1
            else:
                sampled[str(cat)]["anomaly_patches_sampled"] += 1
        for cat in sorted(set(list(image_counts) + list(full_patch_counts) + list(sampled))):
            rows.append({
                "direction": direction,
                "fold": fold,
                "category": cat,
                **image_counts[cat],
                **full_patch_counts[cat],
                **sampled[cat],
                "train_split_member": cat in rotation["train_categories"],
            })
    return rows


def metric_lookup():
    rows = read_csv(XLII_OUT / "PILOT_METRICS.csv")
    return {(r["direction"], int(r["fold"]), r["variant"]): r for r in rows}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lines = class_lines()
    metrics = metric_lookup()
    counts = param_counts()
    src_rows = source_composition()
    write_csv(OUT / "SOURCE_COMPOSITION.csv", src_rows)

    entry = {
        "main_script": str(SCRIPT),
        "config_constants": {"EPOCHS": EPOCHS, "LR": LR, "WEIGHT_DECAY": WEIGHT_DECAY, "RANK": RANK, "EXPERTS": EXPERTS, "TOPK": TOPK, "BATCH_SIZE": BATCH_SIZE, "PATCH_SAMPLE_PER_CLASS_CATEGORY": PATCH_SAMPLE_PER_CLASS_CATEGORY, "SEED": SEED},
        "model_builders": {"UniformLoRA": lines.get("UniformLoRA"), "RoutedLoRA": lines.get("RoutedLoRA")},
        "train_functions": {"R1": lines.get("train_uniform"), "R2": lines.get("train_routed")},
        "eval_functions": {"run_fold": lines.get("run_fold"), "aggregate": lines.get("aggregate"), "evaluate_variant_imported_from": "experiments.phase_xxxix_local_multistate.evaluate_variant"},
        "scoring_functions": {"R0": "score_patch_single_all imported from experiments.phase_xxxix_local_multistate", "R1": lines.get("score_uniform"), "R2": lines.get("score_routed")},
        "checkpoint_paths": [str(XLII_OUT / "checkpoints" / f"{d}_fold{f}_pilot.json") for d, f in PILOT],
        "artifact_paths": {"metrics": str(XLII_OUT / "PILOT_METRICS.csv"), "training_curves": str(XLII_OUT / "TRAINING_CURVES.csv")},
    }

    variant_md = [
        "# XLII-A Variant Difference Audit",
        "",
        "This audit is based on the actual executed script, not the protocol text.",
        "",
        "## R0_FROZEN_BASELINE",
        "- Trainable parameters: none.",
        "- Forward/scoring path: cached CLIP patch features -> XXXVIII K2 learned patch states -> two-state cosine probability via `score_patch_single_all`.",
        "- Text/prompt path: none.",
        "- MRAD memory path: none.",
        "- Post-processing/evaluation: shared `evaluate_variant` from `phase_xxxix_local_multistate`, including 37x37 map upsample and PRO reporting.",
        "",
        "## R1_UNIFORM_LORA",
        "- Trainable parameters: `UniformLoRA.a`, `UniformLoRA.b`, and a new direct binary classifier `head`.",
        "- Forward/scoring path: cached CLIP patch feature `x` -> `normalize(x + B(Ax))` -> cosine/dot difference against trainable `head[1]` and `head[0]` -> sigmoid map.",
        "- Text/prompt path: none.",
        "- XXXVIII states: not used in R1 scoring.",
        "- MRAD memory path: none.",
        "- Loss: XLII-local balanced BCE + Dice on sampled source patch labels.",
        "",
        "## R2_NAIVE_ROUTED_MOE",
        "- Trainable parameters: four LoRA experts `a/b`, router linear layer, and a new direct binary classifier `head`.",
        "- Forward/scoring path: cached CLIP patch feature -> top-2 routed LoRA residual -> trainable direct head -> sigmoid map.",
        "- Text/prompt path: none.",
        "- Loss: same balanced BCE + Dice plus fixed `0.01 * balance_loss`.",
        "",
        "## R0 -> R1 Actual Change",
        "R0 -> R1 does not only insert Uniform LoRA. It also replaces the frozen XXXVIII learned-state scorer with a newly trainable direct binary head and trains it with patch supervision.",
    ]
    (OUT / "VARIANT_DIFF_AUDIT.md").write_text("\n".join(variant_md) + "\n", encoding="utf-8")

    lora_rows = [
        {"variant": "R1_UNIFORM_LORA", "module_full_name": "UniformLoRA.a", "in_dim": DIM, "out_dim": RANK, "rank": RANK, "parameters": DIM * RANK, "initialization": "normal std=0.02", "insertion_location": "cached patch-feature residual, not ViT block18 module"},
        {"variant": "R1_UNIFORM_LORA", "module_full_name": "UniformLoRA.b", "in_dim": RANK, "out_dim": DIM, "rank": RANK, "parameters": DIM * RANK, "initialization": "zeros", "insertion_location": "cached patch-feature residual, not ViT block18 module"},
        {"variant": "R1_UNIFORM_LORA", "module_full_name": "UniformLoRA.head", "in_dim": DIM, "out_dim": 2, "rank": "", "parameters": 2 * DIM, "initialization": "normal std=0.02", "insertion_location": "direct scoring head after adapted cached patch feature"},
        {"variant": "R2_NAIVE_ROUTED_MOE", "module_full_name": "RoutedLoRA.a", "in_dim": DIM, "out_dim": RANK, "rank": RANK, "parameters": 4 * DIM * RANK, "initialization": "normal std=0.02", "insertion_location": "cached patch-feature residual, not ViT block18 module"},
        {"variant": "R2_NAIVE_ROUTED_MOE", "module_full_name": "RoutedLoRA.b", "in_dim": RANK, "out_dim": DIM, "rank": RANK, "parameters": 4 * DIM * RANK, "initialization": "zeros", "insertion_location": "cached patch-feature residual, not ViT block18 module"},
        {"variant": "R2_NAIVE_ROUTED_MOE", "module_full_name": "RoutedLoRA.router", "in_dim": DIM, "out_dim": 4, "rank": "", "parameters": DIM * 4 + 4, "initialization": "torch default Linear", "insertion_location": "router over cached patch feature"},
        {"variant": "R2_NAIVE_ROUTED_MOE", "module_full_name": "RoutedLoRA.head", "in_dim": DIM, "out_dim": 2, "rank": "", "parameters": 2 * DIM, "initialization": "normal std=0.02", "insertion_location": "direct scoring head after routed adapted cached patch feature"},
    ]
    write_csv(OUT / "LORA_MODULE_AUDIT.csv", lora_rows)
    train_rows = []
    for variant, vals in counts.items():
        train_rows.append({"variant": variant, **vals, "complete_model_parameters_note": "CLIP model was not instantiated in actual XLII feature-cache audit; trainable_ratio unavailable relative to full backbone."})
    write_csv(OUT / "TRAINABLE_PARAMETER_AUDIT.csv", train_rows)

    scoring_md = [
        "# Scoring Path Audit",
        "",
        "R1 pixel score does not use CLIP text embeddings, learned prompts, MRAD memory labels, or XXXVIII learned states.",
        "",
        "Actual R1 formula:",
        "",
        "```text",
        "z = normalize(x + B(Ax))",
        "h_N, h_A = normalize(head[0]), normalize(head[1])",
        "logit = z dot h_A - z dot h_N",
        "pixel_score = sigmoid(logit)",
        "```",
        "",
        "R0 formula:",
        "",
        "```text",
        "pixel_score = softmax([cos(x, s_N), cos(x, s_A)])_A",
        "```",
        "",
        "Evaluation uses the shared `evaluate_variant` function imported from `phase_xxxix_local_multistate`.",
    ]
    (OUT / "SCORING_PATH_AUDIT.md").write_text("\n".join(scoring_md) + "\n", encoding="utf-8")

    write_json(OUT / "R0_IDENTITY_AUDIT.json", {
        "identity": "A. XXXVIII K2 Single Learned State",
        "not_official_mrad_clip_local_path": True,
        "not_mrad_visual_instance_memory_path": True,
        "code_path": "R0 uses `score_patch_single_all(held_patch, patch_states)` where `patch_states = states38['K2_patch']` loaded from phase_xxxviii_visual_state_representation/states.",
        "checkpoint_or_state_path_pattern": str(XLII_OUT.parent / "phase_xxxviii_visual_state_representation" / "states" / "{direction}_fold{fold}_states.npz"),
        "metric_evidence": {
            "D1f0_R0": metrics[("D1", 0, "R0_FROZEN_BASELINE")],
            "D2f0_R0": metrics[("D2", 0, "R0_FROZEN_BASELINE")],
        },
    })

    comp = {
        "INSTANCE_MEMORY_COMPARISON_VALID": False,
        "same_target_fold": True,
        "same_evaluation_image_list": True,
        "same_gt": True,
        "same_patch_map_resolution": True,
        "same_interpolation_and_metric_implementation": True,
        "differences": [
            "R1 scoring is a newly trained direct binary head over adapted cached patch features.",
            "Historical K0 InstanceMemory uses instance-memory scoring path from XXXIX/XXXVIII, not the direct head.",
            "R0 in XLII is not K0 InstanceMemory; it is XXXVIII K2 SingleState.",
            "Therefore R1-vs-K0 numeric comparison uses the same evaluator/folds but not the same scorer family or controlled single-variable path.",
        ],
        "k0_metrics_reference": {
            "D1f0": {"P-AUROC": 92.13997652120403, "P-AP": 41.248997972998566},
            "D2f0": {"P-AUROC": 95.86760648691333, "P-AP": 19.967431180581105},
        },
        "r1_metrics": {
            "D1f0": metrics[("D1", 0, "R1_UNIFORM_LORA")],
            "D2f0": metrics[("D2", 0, "R1_UNIFORM_LORA")],
        },
    }
    write_json(OUT / "INSTANCE_MEMORY_COMPARABILITY.json", comp)

    write_json(OUT / "LOSS_AUDIT.json", {
        "loss": "balanced BCE with per-batch inverse class weights + Dice",
        "weights": {"bce": 1.0, "dice": 1.0, "r2_balance": 0.01},
        "normal_anomaly_sampling": f"class_category_patch_sample cap {PATCH_SAMPLE_PER_CLASS_CATEGORY} per class/category",
        "category_balancing": "cap-based balanced source sampling by category/class, not minibatch-uniform category sampler",
        "pixel_sampling": "sampled cached source patch features with mask-derived patch labels",
        "source_masks_used": True,
        "reuses_official_mrad_loss": False,
        "note": "XLII implementation uses a new local BCE+Dice loss, not an official MRAD training call path.",
    })
    write_json(OUT / "CHECKPOINT_AUDIT.json", {
        "max_epoch": EPOCHS,
        "actual_stop_epoch": EPOCHS,
        "checkpoint_selection": "fixed final epoch only",
        "best_checkpoint_used": False,
        "heldout_metric_used_for_selection": False,
        "target_metric_used_for_selection": False,
        "CHECKPOINT_LEAKAGE_RISK": False,
        "saved_model_checkpoint": False,
        "saved_payload_checkpoints": [str(XLII_OUT / "checkpoints" / f"{d}_fold{f}_pilot.json") for d, f in PILOT],
    })
    write_json(OUT / "LEAKAGE_AUDIT.json", {
        "heldout_features_used_for_training": False,
        "heldout_masks_used_for_training": False,
        "heldout_images_used_for_loss": False,
        "heldout_metrics_used_for_checkpoint_selection": False,
        "target_statistics_used_for_normalization_temperature_threshold": False,
        "training_indices_source": "load_fold train_indices from source-held-out protocol",
        "evaluation_indices_source": "load_fold held_indices, used only after training",
        "leakage_risk": False,
    })

    curves = read_csv(XLII_OUT / "TRAINING_CURVES.csv")
    r1_curves = [r for r in curves if r["variant"] == "R1_UNIFORM_LORA"]
    write_csv(OUT / "R1_TRAINING_CURVES.csv", r1_curves)

    report = [
        "# XLII-R1 Implementation Audit",
        "",
        "Outcome: **R1_AUDIT_EXTRA_TRAINABLE_COMPONENTS**",
        "",
        "The actual XLII-A implementation is a cached-feature audit, not a ViT block18 insertion into the official MRAD-CLIP backbone.",
        "",
        "## Main Finding",
        "",
        "R1 is not only Uniform LoRA. It trains both the LoRA residual (`A/B`) and a new direct binary classifier head (`head`). It does not use XXXVIII learned states, CLIP text, prompts, or MRAD memory labels for R1 pixel scoring.",
        "",
        "## R0 Identity",
        "",
        "R0 is XXXVIII K2 Single Learned State scoring over cached patch features, not official MRAD-CLIP local path and not K0 InstanceMemory.",
        "",
        "## LoRA Location",
        "",
        "The LoRA residual is applied after frozen patch features have already been cached. It is not inserted into ViT block18 q/k/v/out projection, attention output, MLP fc1/fc2, or a block residual inside the backbone.",
        "",
        "## Formula",
        "",
        "`z = normalize(x + B(Ax)); score = sigmoid(z dot normalize(w_A) - z dot normalize(w_N))`.",
        "",
        "## Parameter Audit",
        "",
        f"- R1 LoRA parameters: {counts['R1_UNIFORM_LORA']['lora_parameters']}.",
        f"- R1 extra direct-head parameters: {counts['R1_UNIFORM_LORA']['extra_head_parameters']}.",
        f"- R1 total trainable parameters: {counts['R1_UNIFORM_LORA']['trainable_parameters']}.",
        "",
        "## Comparability",
        "",
        "The evaluator/folds/masks/map resolution are aligned with prior cached-feature experiments, but R1 is not directly comparable to K0 InstanceMemory as a single-variable replacement because its scorer family is a new supervised direct head rather than memory scoring.",
        "",
        "## Leakage",
        "",
        "No held-out features or masks are used for training or checkpoint selection. Fixed final epoch 10 is used.",
        "",
        "## Missing Without Recompute",
        "",
        "Feature-change cosine is not available from saved artifacts because trained R1 model weights and adapted features were not saved. Recomputing would require rerunning training/scoring, which this audit forbids.",
    ]
    (OUT / "R1_AUDIT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(OUT / "OUTCOME.json", {
        "phase": "XLII-R1",
        "outcome": "R1_AUDIT_EXTRA_TRAINABLE_COMPONENTS",
        "r1_only_lora": False,
        "extra_trainable_components": ["UniformLoRA.head"],
        "r0_identity": "XXXVIII K2 Single Learned State",
        "instance_memory_comparison_valid": False,
        "checkpoint_leakage_risk": False,
        "data_leakage_risk": False,
    })


if __name__ == "__main__":
    main()
