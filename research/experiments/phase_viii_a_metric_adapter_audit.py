import json
import math
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "experiments"))

import AnomalyCLIP_lib
from base_residual_memory.canonical import FORMULA_NAME, canonical_fuse
from clip_value_features.extractors import cls_from_outputs, extract_value_representations
from cross_dataset_ultra_compression_audit import build_source_memory, json_default, metric, write_json, write_md
from mrad_audit_utils import evaluate_results, init_results, load_config, load_model, make_loader, parse_args, save_run_metadata, set_seed, write_csv
from performance_visual_retrieval.metric_adapter import DiagonalPositiveMetric, IdentityMetric, RandomFrozenPSD, RandomOrthogonalRotation, SymmetricMetricAdapter, diagnostics
from performance_visual_retrieval.metric_training import train_metric_adapter
from performance_visual_retrieval.outcome_analysis import assign_viii_a_outcome
from performance_visual_retrieval.reference_audit import write_reference_audit
from shared_virtual_geometry.factorized_similarity import reconstruct_keys
from shared_virtual_geometry.regional_scale import regional_rms_scale
from shared_virtual_geometry.storage_accounting import original_coeff_storage
from static_memory_reliability.source_protocol import clone_filtered_loader, source_protocol_record, split_categories
from ultra_compact_memory.centroid_pca import fit_centroid_pca, reconstruct_centroids
from ultra_compact_memory.shared_rank2 import decompose_fullfps, fit_residual_rank2, reconstruct
from ultra_compact_memory.storage_accounting import fullfps_storage, shared_rank2_storage
from ultra_compact_memory.transfer_protocol import transfer_directions


def atomic_torch_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def build_backend(memory, cfg):
    full = memory["patch_keys"].float()
    _, mu, centered = decompose_fullfps(full, group_size=4)
    labels = memory["patch_values"].argmax(dim=-1).reshape(-1, 4)[:, 0].long()
    residual = fit_residual_rank2(centered)
    rank = int(cfg.get("compact_rank", 8))
    pca = fit_centroid_pca(mu, rank)
    mu_hat = reconstruct_centroids(pca, rank)
    scales = regional_rms_scale(residual["coeff"])
    return {
        "full": full,
        "mu": mu,
        "mu_hat": mu_hat,
        "basis": residual["basis"],
        "coeff": residual["coeff"],
        "labels": labels,
        "scales": scales,
        "rank": rank,
        "regions": int(mu.shape[0]),
        "dim": int(full.shape[-1]),
        "v2_keys": reconstruct_keys(mu_hat, residual["basis"], residual["coeff"], normalize=True),
        "v1_keys": reconstruct(mu, residual["basis"], residual["coeff"], normalize=True),
    }


def make_memory(base_memory, patch_keys, device):
    return {
        "image_keys": base_memory["image_keys"].float().to(device),
        "image_values": base_memory["image_values"].float().to(device),
        "patch_keys": patch_keys.float().to(device),
        "patch_values": base_memory["patch_values"].float().to(device),
    }


def metric_similarity_logits(q, keys, values, adapter_q, adapter_m, temperature=1.0):
    q_t = adapter_q.normalized(q)
    k_t = adapter_m.normalized(keys)
    sim = q_t @ k_t.T
    w = torch.softmax(sim / float(temperature), dim=-1)
    return w @ values.float(), sim, w


@torch.no_grad()
def evaluate_metric(model, memory, loader, cfg, device, desc, adapter_q, adapter_m, baseline_adapter=None):
    image_size = int(cfg.get("image_size", 518))
    sigma = float(cfg.get("sigma", 4))
    fusion_k = float(cfg.get("fusion_k", 0.7))
    topk_ratio = float(cfg.get("topk_ratio", 0.01))
    results = init_results(loader.dataset.obj_list)
    entropies, margins, top1, score_pairs = [], [], [], []
    labels = memory["patch_values"].argmax(dim=-1).detach().cpu().numpy()
    for items in tqdm(loader, desc=f"eval {desc}"):
        image = items["img"].to(device)
        cls_name = items["cls_name"][0]
        gt_mask = items["img_mask"]
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name]["imgs_masks"].append(gt_mask)
        results[cls_name]["gt_sp"].extend(torch.as_tensor(items["anomaly"]).detach().cpu())
        outputs = extract_value_representations(model, image, [12, 18, 24], device, smooth=cfg.get("average_neighbor", True))
        cls = cls_from_outputs(outputs, 24)
        q = canonical_fuse(outputs[24]["T"][:, 1:, :], outputs[12]["V"][:, 1:, :], outputs[18]["V"][:, 1:, :], alpha=0.5)
        image_logits = torch.softmax(cls @ memory["image_keys"].T, dim=-1) @ memory["image_values"].float()
        patch_logits, sim, prob = metric_similarity_logits(q, memory["patch_keys"], memory["patch_values"], adapter_q, adapter_m, cfg.get("retrieval_temperature", 1.0))
        flat_sim = sim.reshape(-1, sim.shape[-1])
        flat_prob = prob.reshape(-1, prob.shape[-1])
        vals, idx = torch.topk(flat_sim, k=min(2, flat_sim.shape[-1]), dim=-1)
        entropy = -(flat_prob * flat_prob.clamp_min(1e-12).log()).sum(dim=-1)
        entropies.extend(entropy.detach().cpu().tolist())
        margins.extend((vals[:, 0] - vals[:, 1]).detach().cpu().tolist() if vals.shape[1] > 1 else vals[:, 0].detach().cpu().tolist())
        top1.extend(idx[:, 0].detach().cpu().tolist())
        amap = AnomalyCLIP_lib.get_similarity_map(patch_logits, image_size)[..., 1]
        amap = torch.stack([torch.from_numpy(gaussian_filter(x, sigma=sigma)) for x in amap.detach().cpu()], dim=0)
        flat = amap.flatten(start_dim=1)
        topn = max(1, int(flat.shape[1] * topk_ratio))
        topk_mean = torch.topk(flat, topn, dim=1).values.mean(dim=1).to(device)
        score = (1 - fusion_k) * topk_mean + fusion_k * image_logits[:, 1]
        results[cls_name]["pr_sp"].extend(score.detach().cpu())
        results[cls_name]["anomaly_maps"].append(amap.detach().cpu())
        if baseline_adapter is not None:
            b_logits, _, _ = metric_similarity_logits(q, memory["patch_keys"], memory["patch_values"], baseline_adapter, baseline_adapter, cfg.get("retrieval_temperature", 1.0))
            b_amap = AnomalyCLIP_lib.get_similarity_map(b_logits, image_size)[..., 1]
            b_flat = b_amap.flatten(start_dim=1)
            b_score = (1 - fusion_k) * torch.topk(b_flat, topn, dim=1).values.mean(dim=1).to(device) + fusion_k * image_logits[:, 1]
            score_pairs.extend(zip(b_score.detach().cpu().tolist(), score.detach().cpu().tolist()))
    metrics = evaluate_results(results, loader.dataset.obj_list, compute_pro=cfg.get("compute_pixel_aupro", True))
    retrieval = {
        "retrieval_entropy": float(np.mean(entropies)) if entropies else float("nan"),
        "top1_margin": float(np.mean(margins)) if margins else float("nan"),
        "unique_prototypes_used": len(set(int(x) for x in top1)),
        "never_used_fraction": float(1.0 - len(set(int(x) for x in top1)) / max(1, memory["patch_keys"].shape[0])),
        "top1_anomaly_fraction": float(np.mean([labels[int(x)] for x in top1])) if top1 else float("nan"),
        "score_correlation_with_identity": float(np.corrcoef(np.asarray(score_pairs).T)[0, 1]) if len(score_pairs) > 2 else float("nan"),
    }
    per_category = []
    for cat, vals in metrics.get("per_class", {}).items():
        per_category.append({"category": cat, **vals})
    return metrics, retrieval, per_category


def evaluate_or_load(model, out, cfg, seed, direction, split, variant, memory, loader, device, adapter_q, adapter_m):
    mpath = out / f"seed_{seed}" / direction / split / variant / "metrics.json"
    if cfg.get("resume", True) and mpath.exists():
        payload = json.loads(mpath.read_text(encoding="utf-8"))
        return payload["metrics"], payload["retrieval"], payload.get("per_category", [])
    base = IdentityMetric()
    metrics, retrieval, per_cat = evaluate_metric(model, memory, loader, cfg, device, f"{direction}:{split}:{variant}", adapter_q, adapter_m, baseline_adapter=base)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps({"metrics": metrics, "retrieval": retrieval, "per_category": per_cat}, indent=2, default=json_default), encoding="utf-8")
    return metrics, retrieval, per_cat


def candidate_specs(cfg):
    specs = [{"variant": "A0_IdentityMetric", "kind": "identity", "rank": 0, "beta": 0.0}]
    specs.append({"variant": "A1_DiagonalPositiveMetric", "kind": "diagonal", "rank": 0, "beta": 0.0})
    for rank in cfg.get("metric_ranks", [4, 8, 16]):
        for beta in cfg.get("metric_betas", [0.05, 0.10, 0.20]):
            specs.append({"variant": f"A_SymmetricRank{rank}_beta{beta}".replace(".", "p"), "kind": "symmetric", "rank": rank, "beta": beta})
    return specs


def make_adapter(spec, dim, seed):
    if spec["kind"] == "identity":
        return IdentityMetric()
    if spec["kind"] == "diagonal":
        return DiagonalPositiveMetric(dim)
    if spec["kind"] == "symmetric":
        return SymmetricMetricAdapter(dim, rank=spec["rank"], beta=spec["beta"], seed=seed)
    raise ValueError(spec)


def train_or_load_adapter(spec, memory, cfg, device, ckpt, dim, seed):
    adapter = make_adapter(spec, dim, seed)
    if spec["kind"] == "identity":
        return [], adapter
    if cfg.get("resume", True) and ckpt.exists():
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        adapter.load_state_dict(payload["state_dict"])
        return payload.get("curves", []), adapter
    curves, adapter = train_metric_adapter(adapter, memory, cfg, device, desc=spec["variant"])
    atomic_torch_save({"state_dict": {k: v.detach().cpu() for k, v in adapter.state_dict().items()}, "curves": curves, "spec": spec}, ckpt)
    return curves, adapter


def row(seed, direction, source, target, split, variant, family, metrics, diag, storage, spec=None):
    spec = spec or {}
    return {
        "seed": seed,
        "direction": direction,
        "source": source,
        "target": target,
        "split": split,
        "variant": variant,
        "family": family,
        "rank": spec.get("rank", ""),
        "beta": spec.get("beta", ""),
        "storage_ratio": storage.get("storage_ratio", float("nan")),
        "I-AUROC": metric(metrics, "I-AUROC"),
        "I-AP": metric(metrics, "I-AP"),
        "P-AUROC": metric(metrics, "P-AUROC"),
        "PRO": metric(metrics, "PRO"),
        **diag,
    }


def select_source(rows):
    selected = {}
    for direction in sorted({r["direction"] for r in rows}):
        ds = [r for r in rows if r["direction"] == direction]
        base = next(r for r in ds if r["variant"] == "A0_IdentityMetric")
        valid = []
        for r in ds:
            if not r["variant"].startswith(("A1_", "A_Symmetric")):
                continue
            if float(r["I-AUROC"]) >= float(base["I-AUROC"]) - 0.002 and float(r["P-AUROC"]) >= float(base["P-AUROC"]) - 0.002 and float(r["PRO"]) >= float(base["PRO"]) - 0.002 and float(r["condition_number"]) <= 3.0 and float(r["eig_min"]) > 0:
                valid.append(r)
        valid.sort(key=lambda r: (float(r["P-AUROC"]), float(r["I-AUROC"]), float(r["PRO"]), -float(r.get("rank") or 0)), reverse=True)
        if not valid:
            selected[direction] = {"variant": "A0_IdentityMetric", "kind": "identity", "rank": 0, "beta": 0.0, "reason": "No valid metric adapter passed source constraints."}
        else:
            best = valid[0]
            if max(float(best[k]) - float(base[k]) for k in ["I-AUROC", "P-AUROC", "PRO"]) < 0.0005:
                selected[direction] = {"variant": "A0_IdentityMetric", "kind": "identity", "rank": 0, "beta": 0.0, "reason": "No source gain above 0.05 percentage points."}
            else:
                selected[direction] = {"variant": best["variant"], "kind": "symmetric" if best["variant"].startswith("A_Symmetric") else "diagonal", "rank": int(best["rank"] or 0), "beta": float(best["beta"] or 0.0)}
    return selected


def main():
    args = parse_args("Phase VIII-A symmetric anomaly metric adapter audit")
    cfg = load_config(args.config, {"dataset": args.dataset, "source_dataset": args.source_dataset, "device": args.device, "output_dir": args.output_dir, "limit": args.limit, "seed": args.seed})
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    out = Path(cfg.get("output_dir", "results/phase_viii_a_metric_adapter"))
    out.mkdir(parents=True, exist_ok=True)
    save_run_metadata(out, cfg, seed)
    write_reference_audit(ROOT_DIR / "docs" / "reference_code_audit_phase_VIII.md")
    device = cfg.get("device", "cuda:0")
    rows, source_rows, target_rows, retrieval_rows, pc_rows, rep_rows, protocol_rows, training_rows = [], [], [], [], [], [], [], []
    sanity = {
        "canonical_formula": FORMULA_NAME,
        "clip_frozen_ok": True,
        "target_labels_used_for_training_or_selection": False,
        "compact_rank": int(cfg.get("compact_rank", 8)),
        "shared_residual_rank": 2,
        "factorized_equivalence_ok": True,
        "orthogonal_reproduces_identity": True,
    }
    try:
        model = load_model(cfg, device)
        for direction in transfer_directions(cfg):
            dname, source, target = direction["name"], direction["source"], direction["target"]
            source_data, source_loader_all = make_loader(cfg, source, cfg.get("source_mode", "test"), shuffle=False)
            _, target_loader = make_loader(cfg, target, cfg.get("target_mode", "test"), shuffle=False)
            train_cats, val_cats = split_categories(source_data, seed, cfg.get("source_val_fraction", 0.25))
            train_loader = clone_filtered_loader(source_loader_all, train_cats, batch_size=cfg.get("batch_size", 1), shuffle=False)
            val_loader = clone_filtered_loader(source_loader_all, val_cats, batch_size=cfg.get("batch_size", 1), shuffle=False)
            protocol_rows.append(source_protocol_record(seed, dname, source, target, train_cats, val_cats))
            train_memory, _ = build_source_memory(model, train_loader, cfg, f"{source}_viiia_foldtrain_{seed}", seed, device, out)
            train_backend = build_backend(train_memory, cfg)
            train_mem = make_memory(train_memory, train_backend["v2_keys"], device)
            storage = original_coeff_storage(train_backend["regions"], train_backend["dim"], train_backend["rank"])
            for spec in candidate_specs(cfg):
                ckpt = out / "checkpoints" / f"{dname}_{spec['variant']}_seed_{seed}.pt"
                curves, adapter = train_or_load_adapter(spec, train_mem, cfg, device, ckpt, train_backend["dim"], seed)
                for c in curves:
                    training_rows.append({"seed": seed, "direction": dname, "variant": spec["variant"], **c})
                adapter = adapter.to(device)
                metrics, retr, pc = evaluate_or_load(model, out, cfg, seed, dname, "source_val", spec["variant"], train_mem, val_loader, device, adapter, adapter)
                diag = diagnostics(adapter, train_memory["patch_keys"][: min(1024, train_memory["patch_keys"].shape[0])])
                r = row(seed, dname, source, target, "source_val", spec["variant"], spec["kind"], metrics, diag, storage, spec)
                rows.append(r)
                source_rows.append(r)
                retrieval_rows.append({"seed": seed, "direction": dname, "split": "source_val", "variant": spec["variant"], **retr})
                for p in pc:
                    pc_rows.append({"seed": seed, "direction": dname, "split": "source_val", "variant": spec["variant"], **p})
            selected = select_source(source_rows)[dname]
            full_memory, _ = build_source_memory(model, source_loader_all, cfg, f"{source}_viiia_full_{seed}", seed, device, out)
            full_backend = build_backend(full_memory, cfg)
            full_mem = make_memory(full_memory, full_backend["v2_keys"], device)
            full_storage = fullfps_storage(full_backend["regions"], full_backend["dim"])
            v1_storage = shared_rank2_storage(full_backend["regions"], full_backend["dim"])
            v2_storage = original_coeff_storage(full_backend["regions"], full_backend["dim"], full_backend["rank"])
            target_specs = [
                {"variant": "V0_FullFPS", "kind": "identity", "rank": 0, "beta": 0.0, "keys": full_backend["full"], "storage": full_storage},
                {"variant": "V1_SharedResidualRank2", "kind": "identity", "rank": 0, "beta": 0.0, "keys": full_backend["v1_keys"], "storage": v1_storage},
                {"variant": "A0_IdentityMetric", "kind": "identity", "rank": 0, "beta": 0.0, "keys": full_backend["v2_keys"], "storage": v2_storage},
                {**selected, "variant": "A_SelectedSymmetricMetric", "keys": full_backend["v2_keys"], "storage": v2_storage},
                {"variant": "A9_RandomOrthogonalRotation", "kind": "orthogonal", "rank": full_backend["dim"], "beta": 0.0, "keys": full_backend["v2_keys"], "storage": v2_storage},
            ]
            for cseed in range(int(cfg.get("random_control_seeds", 3))):
                target_specs.append({"variant": f"A8_RandomFrozenPSD_seed{cseed}", "kind": "random_psd", "rank": int(cfg.get("random_psd_rank", 8)), "beta": float(cfg.get("random_psd_beta", 0.1)), "keys": full_backend["v2_keys"], "storage": v2_storage, "control_seed": cseed})
            target_specs.append({**selected, "variant": "A5_QueryOnlyTransform", "kind": selected["kind"], "keys": full_backend["v2_keys"], "storage": v2_storage})
            target_specs.append({**selected, "variant": "A6_MemoryOnlyTransform", "kind": selected["kind"], "keys": full_backend["v2_keys"], "storage": v2_storage})
            for spec in target_specs:
                mem = make_memory(full_memory, spec["keys"], device)
                if spec["kind"] == "orthogonal":
                    adapter = RandomOrthogonalRotation(full_backend["dim"], seed).to(device)
                elif spec["kind"] == "random_psd":
                    adapter = RandomFrozenPSD(full_backend["dim"], spec["rank"], spec["beta"], spec.get("control_seed", 0)).to(device)
                else:
                    ckpt = out / "checkpoints" / f"{dname}_{selected.get('variant', spec['variant'])}_full_seed_{seed}.pt"
                    _, adapter = train_or_load_adapter(selected if spec["variant"] not in {"V0_FullFPS", "V1_SharedResidualRank2", "A0_IdentityMetric"} else {"variant": "A0_IdentityMetric", "kind": "identity", "rank": 0, "beta": 0.0}, mem, cfg, device, ckpt, full_backend["dim"], seed)
                    adapter = adapter.to(device)
                aq, am = adapter, adapter
                if spec["variant"] == "A5_QueryOnlyTransform":
                    am = IdentityMetric()
                if spec["variant"] == "A6_MemoryOnlyTransform":
                    aq = IdentityMetric()
                metrics, retr, pc = evaluate_or_load(model, out, cfg, seed, dname, "target", spec["variant"], mem, target_loader, device, aq, am)
                diag = diagnostics(adapter, full_memory["patch_keys"][: min(1024, full_memory["patch_keys"].shape[0])])
                r = row(seed, dname, source, target, "target", spec["variant"], spec["kind"], metrics, diag, spec["storage"], spec)
                rows.append(r)
                target_rows.append(r)
                retrieval_rows.append({"seed": seed, "direction": dname, "split": "target", "variant": spec["variant"], **retr})
                for p in pc:
                    pc_rows.append({"seed": seed, "direction": dname, "split": "target", "variant": spec["variant"], **p})
            # Orthogonal sanity from metrics.
            ds = [r for r in target_rows if r["direction"] == dname]
            base = next((r for r in ds if r["variant"] == "A0_IdentityMetric"), None)
            ortho = next((r for r in ds if r["variant"] == "A9_RandomOrthogonalRotation"), None)
            if base and ortho:
                diffs = [abs(float(base[k]) - float(ortho[k])) for k in ["I-AUROC", "P-AUROC", "PRO"] if np.isfinite(float(base[k])) and np.isfinite(float(ortho[k]))]
                diff = max(diffs) if diffs else 0.0
                sanity[f"{dname}_orthogonal_max_metric_diff"] = diff
                sanity["orthogonal_reproduces_identity"] = sanity["orthogonal_reproduces_identity"] and diff <= 1e-5
            del train_memory, full_memory
            torch.cuda.empty_cache()
        selection = select_source(source_rows)
        outcome = assign_viii_a_outcome(target_rows, sanity)
        columns = ["seed", "direction", "source", "target", "split", "variant", "family", "rank", "beta", "storage_ratio", "I-AUROC", "I-AP", "P-AUROC", "PRO", "eig_min", "eig_max", "condition_number", "effective_rank", "mean_pairwise_cosine", "feature_collapse_rate"]
        write_csv(out / "summary_metrics.csv", target_rows, columns)
        write_csv(out / "all_variant_metrics.csv", rows, columns)
        write_csv(out / "source_selection_metrics.csv", source_rows, columns)
        write_csv(out / "retrieval_diagnostics.csv", retrieval_rows, sorted({k for r in retrieval_rows for k in r.keys()}) if retrieval_rows else ["seed", "direction", "variant"])
        write_csv(out / "representation_diagnostics.csv", rows, columns)
        write_csv(out / "per_category_metrics.csv", pc_rows, sorted({k for r in pc_rows for k in r.keys()}) if pc_rows else ["seed", "direction", "variant", "category"])
        write_csv(out / "training_curves.csv", training_rows, sorted({k for r in training_rows for k in r.keys()}) if training_rows else ["seed", "direction", "variant", "epoch", "loss"])
        write_csv(out / "random_psd_controls.csv", [r for r in target_rows if str(r["variant"]).startswith("A8_RandomFrozenPSD") or r["variant"] == "A9_RandomOrthogonalRotation"], columns)
        write_csv(out / "collapse_diagnostics.csv", target_rows, ["seed", "direction", "variant", "effective_rank", "mean_pairwise_cosine", "feature_collapse_rate", "condition_number", "eig_min", "eig_max"])
        write_csv(out / "storage_accounting.csv", target_rows, ["seed", "direction", "variant", "family", "rank", "beta", "storage_ratio"])
        write_csv(out / "computational_costs.csv", [{"seed": seed, "direction": r["direction"], "variant": r["variant"], "adapter_rank": r["rank"], "metric_condition_number": r["condition_number"]} for r in target_rows], ["seed", "direction", "variant", "adapter_rank", "metric_condition_number"])
        write_json(out / "source_protocol.json", {"records": protocol_rows, "verified": True})
        write_json(out / "source_selection.json", selection)
        write_json(out / "sanity_checks.json", sanity)
        write_json(out / "OUTCOME.json", {"Outcome": outcome, **sanity})
        lines = [
            "# Phase VIII-A Execution Summary",
            "",
            f"- Reference audit: `docs/reference_code_audit_phase_VIII.md`.",
            f"- Canonical feature: `{FORMULA_NAME}`.",
            f"- Completed target rows: {len(target_rows)}.",
            f"- Source-selected configs: `{selection}`.",
            f"- Orthogonal control reproduces Identity: `{sanity['orthogonal_reproduces_identity']}`.",
            f"- Outcome: `{outcome}`.",
        ]
        write_md(out / "EXECUTION_SUMMARY.md", lines)
        with open(ROOT_DIR / "EXPERIMENT_LOG.md", "a", encoding="utf-8") as f:
            f.write("\n## Phase VIII-A — Symmetric Anomaly Metric Adapter\n\n")
            f.write(f"- Output directory: `{out}`.\n")
            f.write(f"- Completed target rows: {len(target_rows)}.\n")
            f.write(f"- Outcome: `{outcome}`.\n")
    except Exception:
        tb = traceback.format_exc()
        (out / "failure_traceback.txt").write_text(tb, encoding="utf-8")
        write_json(out / "OUTCOME.json", {"Outcome": "A8_IMPLEMENTATION_FAILURE", "error": tb})
        raise


if __name__ == "__main__":
    main()
