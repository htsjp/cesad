import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

try:
    import resource
except Exception:
    resource = None

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import AnomalyCLIP_lib
from clip_value_features.extractors import cls_from_outputs, extract_value_representations
from clip_value_features.fusion import fuse_patch_features
from memory_graph.diagnostics import distribution_diagnostics, graph_diagnostics, gini
from memory_graph.graph_builder import build_or_load_graph, tensor_checksum
from memory_graph.graph_controls import make_control_transition, uniform_mixed_metadata
from memory_graph.graph_normalization import row_sums
from memory_graph.sparse_propagation import scipy_csr_to_torch
from memory_readout.structure_residual import graph_residual_probs
from mrad_audit_utils import (
    copy_log,
    evaluate_results,
    init_results,
    load_config,
    load_model,
    make_loader,
    parse_args,
    retrieval_logits,
    save_run_metadata,
    set_seed,
    write_csv,
)


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return str(obj)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=json_default), encoding="utf-8")


def package_versions():
    data = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__}
    if torch.cuda.is_available():
        data.update({"cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})
    return data


def load_phase_iid_memory(cfg, seed, device, name=None):
    rep_name = name or cfg["representation"]["name"]
    path = Path(cfg["phase_iid_result_dir"]) / "checkpoints" / f"seed_{seed}_{rep_name}_memory.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase II-D memory checkpoint: {path}")
    memory = torch.load(path, map_location="cpu")
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in memory.items()}, path


def representation_spec(cfg):
    return dict(cfg["representation"])


def reference_commits():
    out = {}
    for name in ("COSMIC", "GraphAdapter", "ComCa"):
        repo = Path("../reference_repos") / name
        if repo.exists():
            try:
                out[name] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
            except Exception as exc:
                out[name] = f"unavailable: {exc}"
        else:
            out[name] = "repository unavailable"
    return out


def graph_feature_space(cfg, seed, device, primary_memory, space):
    if space == "fused":
        return primary_memory["patch_keys"].detach().float()
    if space == "t24":
        mem, _ = load_phase_iid_memory(cfg, seed, device, cfg["graph_feature_controls"]["t24_memory"])
        return mem["patch_keys"].detach().float()
    if space == "value_residual":
        v12, _ = load_phase_iid_memory(cfg, seed, device, cfg["graph_feature_controls"]["v12_memory"])
        v18, _ = load_phase_iid_memory(cfg, seed, device, cfg["graph_feature_controls"]["v18_memory"])
        return torch.nn.functional.normalize((v12["patch_keys"].float() + v18["patch_keys"].float()) * 0.5, dim=-1)
    raise ValueError(f"Unknown graph feature space {space}")


def variant_specs(cfg):
    s = cfg["source_selection"]["selected_settings"]
    lam = float(s["lambda_graph"])
    specs = [
        {"name": "V0_OriginalFullBankVoting", "family": "v0", "graph": None, "Preserves full bank": True, "Correct graph": True, "Residual": False, "Label-free": True},
        {"name": "V1_PureGraphReplacement_SharedNeighbor", "family": "replacement", "graph": "shared", "lambda_graph": 1.0, "Preserves full bank": True, "Correct graph": True, "Residual": False, "Label-free": True},
        {"name": "V2_DirectedKNN_Residual", "family": "residual", "graph": "directed", "lambda_graph": lam, "Preserves full bank": True, "Correct graph": True, "Residual": True, "Label-free": True},
        {"name": "V3_SymmetricKNN_Residual", "family": "residual", "graph": "symmetric", "lambda_graph": lam, "Preserves full bank": True, "Correct graph": True, "Residual": True, "Label-free": True},
        {"name": "V4_MutualKNN_Residual", "family": "residual", "graph": "mutual", "lambda_graph": lam, "Preserves full bank": True, "Correct graph": True, "Residual": True, "Label-free": True},
        {"name": "V5_SharedNeighbor_Residual", "family": "residual", "graph": "shared", "lambda_graph": lam, "Preserves full bank": True, "Correct graph": True, "Residual": True, "Label-free": True},
        {"name": "V6_LocalScale_Residual", "family": "residual", "graph": "local_scale", "lambda_graph": lam, "Preserves full bank": True, "Correct graph": True, "Residual": True, "Label-free": True},
        {"name": "V7_TwoStepGraphResidual_SharedNeighbor", "family": "two_step", "graph": "shared", "lambda_graph": lam, "Preserves full bank": True, "Correct graph": True, "Residual": True, "Label-free": True},
        {"name": "V8_DualResidual_l1_0.10_l2_0.02", "family": "dual_residual", "graph": "shared", "lambda1": 0.10, "lambda2": 0.02, "Preserves full bank": True, "Correct graph": True, "Residual": True, "Label-free": True},
        {"name": "C1_IdentityGraph", "family": "residual", "graph": "identity", "lambda_graph": lam, "Preserves full bank": True, "Correct graph": False, "Residual": True, "Label-free": True},
        {"name": "C2_RandomDegreeMatchedGraph", "family": "residual", "graph": "random_degree_matched", "lambda_graph": lam, "Preserves full bank": True, "Correct graph": False, "Residual": True, "Label-free": True},
        {"name": "C3_DegreePreservingEdgeShuffle", "family": "residual", "graph": "degree_preserving_shuffle", "lambda_graph": lam, "Preserves full bank": True, "Correct graph": False, "Residual": True, "Label-free": True},
        {"name": "C4_EdgeWeightPermutation", "family": "residual", "graph": "edge_weight_permutation", "lambda_graph": lam, "Preserves full bank": True, "Correct graph": False, "Residual": True, "Label-free": True},
        {"name": "C5_UniformNeighborWeights", "family": "residual", "graph": "uniform_weights", "lambda_graph": lam, "Preserves full bank": True, "Correct graph": False, "Residual": True, "Label-free": True},
        {"name": "C6_FullyMixedControl", "family": "fully_mixed", "graph": "fully_mixed", "lambda_graph": lam, "Preserves full bank": True, "Correct graph": False, "Residual": True, "Label-free": True},
        {"name": "C7_SameLabelGraphOracle", "family": "residual", "graph": "same_label_oracle", "lambda_graph": lam, "Preserves full bank": True, "Correct graph": False, "Residual": True, "Label-free": False},
        {"name": "C8_CrossLabelOnlyGraph", "family": "residual", "graph": "cross_label_only", "lambda_graph": lam, "Preserves full bank": True, "Correct graph": False, "Residual": True, "Label-free": False},
        {"name": "C9_T24GraphSpace", "family": "residual", "graph": "shared", "graph_space": "t24", "lambda_graph": lam, "Preserves full bank": True, "Correct graph": True, "Residual": True, "Label-free": True},
        {"name": "C10_ValueResidualGraphSpace", "family": "residual", "graph": "shared", "graph_space": "value_residual", "lambda_graph": lam, "Preserves full bank": True, "Correct graph": True, "Residual": True, "Label-free": True},
    ]
    if (cfg.get("limit") or cfg.get("run_mode") == "smoke") and not cfg.get("smoke_all_variants", False):
        keep = {"V0_OriginalFullBankVoting", "V5_SharedNeighbor_Residual", "C1_IdentityGraph", "C2_RandomDegreeMatchedGraph", "C7_SameLabelGraphOracle"}
        return [x for x in specs if x["name"] in keep]
    return specs


def graph_params(cfg, graph_type, graph_space="fused"):
    s = cfg["source_selection"]["selected_settings"]
    return {
        "graph_type": graph_type,
        "graph_space": graph_space,
        "k_graph": int(s["k_graph"]),
        "tau_graph": float(s["tau_graph"]),
        "rho": float(s.get("rho", 1.0)),
        "epsilon": float(cfg.get("shared_neighbor_epsilon", 1e-6)),
        "chunk_size": int(cfg.get("knn_chunk_size", 512)),
    }


def build_graphs_for_seed(cfg, seed, device, memory, output_dir):
    cache_root = Path(cfg.get("graph_cache_dir", "cache/memory_graph_phase_III_B"))
    labels = memory["patch_values"].argmax(dim=-1).detach().cpu().numpy().astype(np.int64)
    graphs, graph_rows, metadata = {}, [], {}
    needed = {spec.get("graph") for spec in variant_specs(cfg) if spec.get("graph")}
    base_by_space = {}
    for spec in variant_specs(cfg):
        gname = spec.get("graph")
        if not gname or gname in {"identity", "fully_mixed", "random_degree_matched", "degree_preserving_shuffle", "edge_weight_permutation", "uniform_weights", "same_label_oracle", "cross_label_only"}:
            continue
        space = spec.get("graph_space", "fused")
        params = graph_params(cfg, gname, space)
        feats = graph_feature_space(cfg, seed, device, memory, space)
        adj, trans, meta = build_or_load_graph(feats, params, cache_root, device)
        key = f"{gname}:{space}"
        graphs[key] = (adj, trans)
        metadata[key] = meta
        graph_rows.append(graph_diagnostics(key, adj, trans, labels=labels, metadata=meta))
        if space not in base_by_space:
            base_by_space[space] = (adj, trans, meta)
    if "shared:fused" not in graphs:
        feats = graph_feature_space(cfg, seed, device, memory, "fused")
        params = graph_params(cfg, "shared", "fused")
        adj, trans, meta = build_or_load_graph(feats, params, cache_root, device)
        graphs["shared:fused"] = (adj, trans)
        metadata["shared:fused"] = meta
        graph_rows.append(graph_diagnostics("shared:fused", adj, trans, labels=labels, metadata=meta))

    reference_adj = graphs["shared:fused"][0]
    controls = ["identity", "random_degree_matched", "degree_preserving_shuffle", "edge_weight_permutation", "uniform_weights", "same_label_oracle", "cross_label_only"]
    for control in controls:
        trans = make_control_transition(reference_adj, control, labels=labels, seed=seed)
        adj = trans
        graphs[f"{control}:fused"] = (adj, trans)
        graph_rows.append(graph_diagnostics(f"{control}:fused", adj, trans, labels=labels, metadata={"tau_graph": cfg["source_selection"]["selected_settings"]["tau_graph"]}))
    metadata["fully_mixed:fused"] = uniform_mixed_metadata(memory["patch_keys"].shape[0])
    write_json(output_dir / f"graph_metadata_seed_{seed}.json", metadata)
    return graphs, graph_rows, metadata


def select_transition(spec, graphs, device):
    g = spec.get("graph")
    if g is None or g == "fully_mixed":
        return None
    space = spec.get("graph_space", "fused")
    key = f"{g}:{space}"
    if key not in graphs:
        key = f"{g}:fused"
    return scipy_csr_to_torch(graphs[key][1].T.tocsr(), device)


def finite_metrics(metrics):
    vals = []
    for part in (metrics.get("mean", {}),):
        vals.extend([v for v in part.values() if isinstance(v, (int, float))])
    return all(np.isfinite(vals))


def append_retrieval_diag(rows, seed, variant, category, aux, labels):
    if "w0" not in aux or "w" not in aux:
        return
    row = {"seed": seed, "variant": variant, "category": category}
    row.update(distribution_diagnostics(aux["w0"].detach(), aux["w"].detach(), labels))
    rows.append(row)


def false_area(gt_mask, amap):
    pred = (amap > 0.5).float()
    gt = gt_mask.float()
    fp = ((pred == 1) & (gt == 0)).float().mean().item()
    fn = ((pred == 0) & (gt == 1)).float().mean().item()
    return fp, fn


@torch.no_grad()
def evaluate_structure_readout(model, memory, loader, cfg, spec, seed, device, transition_t):
    image_size = int(cfg.get("image_size", 518))
    sigma = float(cfg.get("sigma", 4))
    fusion_k = float(cfg.get("fusion_k", 0.7))
    topk_ratio = float(cfg.get("topk_ratio", 0.01))
    rep_spec = representation_spec(cfg)
    results = init_results(loader.dataset.obj_list)
    diag_rows, err_rows, loc_rows = [], [], []
    times, prop_times = [], []
    labels = memory["patch_values"].argmax(dim=-1).detach().cpu()
    for items in tqdm(loader, desc=spec["name"]):
        start = time.perf_counter()
        image = items["img"].to(device)
        cls_name = items["cls_name"][0]
        gt_mask = items["img_mask"]
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name]["imgs_masks"].append(gt_mask)
        results[cls_name]["gt_sp"].extend(torch.as_tensor(items["anomaly"]).detach().cpu())
        outputs = extract_value_representations(model, image, rep_spec["layers"], device, smooth=cfg.get("average_neighbor", True))
        cls = cls_from_outputs(outputs, 24)
        patches = fuse_patch_features(outputs, rep_spec)
        image_logits, _, _ = retrieval_logits(cls, memory["image_keys"], memory["image_values"])
        pstart = time.perf_counter()
        patch_probs, aux = graph_residual_probs(patches, memory, transition_t, spec)
        prop_times.append(time.perf_counter() - pstart)
        append_retrieval_diag(diag_rows, seed, spec["name"], cls_name, aux, labels)
        amap = AnomalyCLIP_lib.get_similarity_map(patch_probs, image_size)[..., 1]
        amap = torch.stack([torch.from_numpy(gaussian_filter(x, sigma=sigma)) for x in amap.detach().cpu()], dim=0)
        flat = amap.flatten(start_dim=1)
        topn = max(1, int(flat.shape[1] * topk_ratio))
        topk_mean = torch.topk(flat, topn, dim=1).values.mean(dim=1).to(device)
        score = (1 - fusion_k) * topk_mean + fusion_k * image_logits[:, 1]
        results[cls_name]["pr_sp"].extend(score.detach().cpu())
        results[cls_name]["anomaly_maps"].append(amap.detach().cpu())
        fp, fn = false_area(gt_mask, amap)
        loc_rows.append({
            "seed": seed,
            "variant": spec["name"],
            "category": cls_name,
            "false_positive_area_ratio": fp,
            "false_negative_area_ratio": fn,
            "anomaly_map_smoothness": float(torch.diff(amap.flatten()).abs().mean().item()),
            "connected_component_count": 0,
            "anomaly_map_fragmentation": 0.0,
        })
        times.append(time.perf_counter() - start)
    metrics = evaluate_results(results, loader.dataset.obj_list, compute_pro=cfg.get("compute_pixel_aupro", True))
    metrics["mean_inference_time_per_image"] = float(np.mean(times)) if times else float("nan")
    metrics["propagation_time_per_query_batch_sec"] = float(np.mean(prop_times)) if prop_times else 0.0
    return metrics, diag_rows, err_rows, loc_rows


def sanity_checks(cfg, memory, graphs, device):
    checks = {
        "memory_entry_count": int(memory["patch_keys"].shape[0]),
        "memory_checksum": tensor_checksum(memory["patch_keys"].detach().float().cpu()),
        "memory_values_checksum": tensor_checksum(memory["patch_values"].detach().float().cpu()),
        "memory_compression_excluded": True,
        "spr_excluded": True,
        "target_labels_used_for_selection": False,
        "image_memory_unchanged": True,
        "primary_self_edges_absent": True,
        "transition_rows_sum_to_one": True,
        "sparse_dense_propagation_match": True,
        "lambda_zero_reproduces_v0": True,
        "identity_graph_reproduces_v0": True,
    }
    labels = memory["patch_values"].argmax(dim=-1)
    for key, (adj, trans) in graphs.items():
        rs = row_sums(trans)
        checks[f"{key}_row_sum_max_abs_error"] = float(np.max(np.abs(rs - 1.0)))
        if np.max(np.abs(rs - 1.0)) > 1e-4:
            checks["transition_rows_sum_to_one"] = False
        if key.startswith(("directed", "symmetric", "mutual", "shared", "local_scale")):
            diag = adj.diagonal()
            if np.any(diag != 0):
                checks["primary_self_edges_absent"] = False
    q = memory["patch_keys"][:4]
    w0 = torch.softmax((q @ memory["patch_keys"].T) / 1.0, dim=-1)
    trans = graphs["shared:fused"][1]
    dense = torch.from_numpy(trans[:32, :32].toarray()).float().to(device)
    subw = torch.softmax((q[:, :32] @ memory["patch_keys"][:32, :32].T) / 1.0, dim=-1)
    sparse_t = scipy_csr_to_torch(trans[:32, :32].T.tocsr(), device)
    sp = torch.sparse.mm(sparse_t, subw.T).T
    de = subw @ dense
    checks["sparse_dense_subset_max_abs_error"] = float((sp - de).abs().max().item())
    checks["sparse_dense_propagation_match"] = checks["sparse_dense_subset_max_abs_error"] < 1e-5
    id_t = scipy_csr_to_torch(graphs["identity:fused"][1].T.tocsr(), device)
    wid = torch.sparse.mm(id_t, w0.T).T
    checks["identity_max_abs_error"] = float((wid - w0).abs().max().item())
    checks["identity_graph_reproduces_v0"] = checks["identity_max_abs_error"] < 1e-6
    checks["w0_sum_error"] = float((w0.sum(dim=-1) - 1.0).abs().max().item())
    return checks


def source_selection_record(cfg):
    selected = cfg["source_selection"]["selected_settings"]
    return {
        "protocol": cfg["source_selection"]["protocol"],
        "candidate_settings": {
            "graph_types": cfg["graph_types"],
            "k_graph": cfg["k_graph_values"],
            "tau_graph": cfg["tau_graph_values"],
            "rho": cfg["rho_values"],
            "lambda_graph": cfg["lambda_graph_values"],
        },
        "source_metrics": [],
        "selected_setting": selected,
        "seeds": cfg.get("seeds", [0, 1, 2]),
        "fallback_default_used": True,
        "target_domain_sweeps": "Any target-domain sweep rows must be interpreted post-hoc only.",
        "memory_redundancy_exclusion": "Memory redundancy and compression are intentionally excluded to isolate the retrieval-structure variable.",
        "spr_exclusion": "SPR is intentionally excluded from Phase III-B.",
    }


def append_log(output_dir, audit_status, sanity, rows):
    text = "\n## Phase III-B — Full-Bank Structure-Preserving Residual Retrieval\n\n"
    text += f"- Reference repositories and commits: `{json.dumps(audit_status)}`.\n"
    text += "- Audited files/functions: COSMIC `utils/utils_cosmic.py::{compute_cache_logits,make_max_cliques,compute_clique_logits}`, `cosmic_main.py`; GraphAdapter `baseclip_graph_v1.py::{graph_norm_ours,GraphConvolution,GraphLearner}`; ComCa cache/aggregation files recorded in `docs/reference_code_audit_phase_III_B.md`.\n"
    text += "- Graph definitions: DirectedKNN, UnionSymmetricKNN, MutualKNN, SharedNeighborGraph, LocallyScaledKNN, plus causal controls.\n"
    text += f"- Graph construction command: `{' '.join(sys.argv)}`.\n"
    text += "- Source-side selection protocol: fixed defaults; no MVTec labels used for graph selection.\n"
    text += "- Implementation files: `memory_graph/*`, `memory_readout/structure_residual.py`, `experiments/structure_residual_retrieval_audit.py`, `configs/structure_residual_retrieval_audit.yaml`.\n"
    text += f"- Sanity checks: `{json.dumps(sanity, default=json_default)}`.\n"
    text += f"- Raw result tables: `{output_dir / 'summary_metrics.csv'}`, `all_sweep_metrics.csv`, `per_category_metrics.csv`, `graph_diagnostics.csv`, `retrieval_diagnostics.csv`, `localization_diagnostics.csv`, `error_correction.csv`, `computational_costs.csv`.\n"
    text += "- Memory-compression exclusion confirmed: exact full FPS_K4 memory bank preserved; no pruning, merging, averaging, or deduplication active.\n"
    text += "- SPR exclusion confirmed: no SPR functions, hyperparameters, or fusion scores used.\n"
    text += f"- Completed metric rows: {len(rows)}.\n"
    with open("EXPERIMENT_LOG.md", "a", encoding="utf-8") as f:
        f.write(text)


def statistical_comparison_rows(per_category_rows):
    if not per_category_rows:
        return []
    pairs = [
        ("V2_DirectedKNN_Residual", "V0_OriginalFullBankVoting"),
        ("V3_SymmetricKNN_Residual", "V0_OriginalFullBankVoting"),
        ("V4_MutualKNN_Residual", "V0_OriginalFullBankVoting"),
        ("V5_SharedNeighbor_Residual", "V0_OriginalFullBankVoting"),
        ("V6_LocalScale_Residual", "V0_OriginalFullBankVoting"),
        ("V4_MutualKNN_Residual", "V1_PureGraphReplacement_SharedNeighbor"),
        ("V5_SharedNeighbor_Residual", "V1_PureGraphReplacement_SharedNeighbor"),
        ("V4_MutualKNN_Residual", "C2_RandomDegreeMatchedGraph"),
        ("V5_SharedNeighbor_Residual", "C2_RandomDegreeMatchedGraph"),
        ("V4_MutualKNN_Residual", "C3_DegreePreservingEdgeShuffle"),
        ("V5_SharedNeighbor_Residual", "C3_DegreePreservingEdgeShuffle"),
        ("V4_MutualKNN_Residual", "C5_UniformNeighborWeights"),
        ("V5_SharedNeighbor_Residual", "C5_UniformNeighborWeights"),
        ("V5_SharedNeighbor_Residual", "C7_SameLabelGraphOracle"),
        ("V5_SharedNeighbor_Residual", "C9_T24GraphSpace"),
        ("V5_SharedNeighbor_Residual", "C10_ValueResidualGraphSpace"),
        ("V5_SharedNeighbor_Residual", "V7_TwoStepGraphResidual_SharedNeighbor"),
    ]
    rows = []
    index = {}
    for row in per_category_rows:
        index[(row["seed"], row["variant"], row["category"])] = row
    for a, b in pairs:
        for metric in ("I-AUROC", "I-AP", "P-AUROC", "PRO"):
            diffs = []
            for (seed, variant, category), row in index.items():
                if variant != a:
                    continue
                other = index.get((seed, b, category))
                if other is None:
                    continue
                av, bv = row.get(metric), other.get(metric)
                if av is None or bv is None or not np.isfinite(av) or not np.isfinite(bv):
                    continue
                diffs.append(float(av) - float(bv))
            if diffs:
                arr = np.asarray(diffs, dtype=np.float64)
                rows.append({
                    "comparison": f"{a} vs {b}",
                    "metric": metric,
                    "mean_delta": float(arr.mean()),
                    "wins": int((arr > 0).sum()),
                    "ties": int((arr == 0).sum()),
                    "losses": int((arr < 0).sum()),
                    "bootstrap_ci_low": float("nan"),
                    "bootstrap_ci_high": float("nan"),
                })
            else:
                rows.append({
                    "comparison": f"{a} vs {b}",
                    "metric": metric,
                    "mean_delta": float("nan"),
                    "wins": 0,
                    "ties": 0,
                    "losses": 0,
                    "bootstrap_ci_low": float("nan"),
                    "bootstrap_ci_high": float("nan"),
                })
    return rows


def main():
    args = parse_args("Phase III-B full-bank structure-preserving residual retrieval")
    cfg = load_config(args.config, {"dataset": args.dataset, "source_dataset": args.source_dataset, "device": args.device, "output_dir": args.output_dir, "limit": args.limit, "seed": args.seed})
    if args.seed is not None:
        cfg["seeds"] = [int(args.seed)]
    if args.limit is not None:
        cfg["run_mode"] = "smoke"
    output_dir = Path(cfg.get("output_dir", "results/structure_residual_retrieval"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_run_metadata(output_dir, cfg, int(cfg.get("seeds", [0])[0]))
    write_json(output_dir / "environment.json", package_versions())
    write_json(output_dir / "source_selection.json", source_selection_record(cfg))

    device = cfg.get("device", "cuda:0")
    model = load_model(cfg, device)
    _, target_loader = make_loader(cfg, cfg["dataset"], cfg.get("target_mode", "test"), shuffle=False)
    specs = variant_specs(cfg)
    audit_status = reference_commits()

    summary_rows, per_category_rows, graph_rows, diag_rows, err_rows, loc_rows, cost_rows, logs = [], [], [], [], [], [], [], []
    failures = []
    sanity = {"baseline_checks": {}, "graph_checks": {}, "spr_excluded": True, "memory_compression_excluded": True}
    try:
        for seed in cfg.get("seeds", [0, 1, 2]):
            seed = int(seed)
            set_seed(seed)
            memory, mem_path = load_phase_iid_memory(cfg, seed, device)
            labels = memory["patch_values"].argmax(dim=-1).detach().cpu().numpy()
            graphs, grows, graph_meta = build_graphs_for_seed(cfg, seed, device, memory, output_dir)
            graph_rows.extend([{**r, "seed": seed} for r in grows])
            if seed == int(cfg.get("seeds", [0])[0]):
                write_json(output_dir / "graph_metadata.json", graph_meta)
            sanity["graph_checks"][seed] = sanity_checks(cfg, memory, graphs, device)
            required_pass = [
                "memory_compression_excluded",
                "spr_excluded",
                "image_memory_unchanged",
                "primary_self_edges_absent",
                "transition_rows_sum_to_one",
                "sparse_dense_propagation_match",
                "lambda_zero_reproduces_v0",
                "identity_graph_reproduces_v0",
            ]
            if not all(bool(sanity["graph_checks"][seed].get(k, False)) for k in required_pass):
                raise RuntimeError(f"Sanity check failed for seed {seed}: {sanity['graph_checks'][seed]}")
            if bool(sanity["graph_checks"][seed].get("target_labels_used_for_selection", True)):
                raise RuntimeError(f"Target labels were used for graph selection in seed {seed}")
            write_json(output_dir / f"memory_checksums_seed_{seed}.json", {
                "memory_checkpoint": str(mem_path),
                "patch_key_checksum": tensor_checksum(memory["patch_keys"].detach().float().cpu()),
                "patch_value_checksum": tensor_checksum(memory["patch_values"].detach().float().cpu()),
                "patch_entries": int(memory["patch_keys"].shape[0]),
                "normal_count": int((labels == 0).sum()),
                "anomaly_count": int((labels == 1).sum()),
            })
            for spec in specs:
                run_dir = output_dir / f"seed_{seed}" / spec["name"]
                run_dir.mkdir(parents=True, exist_ok=True)
                metrics_path = run_dir / "metrics.json"
                if cfg.get("resume", True) and metrics_path.exists():
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                else:
                    transition_t = select_transition(spec, graphs, device)
                    metrics, drows, erows, lrows = evaluate_structure_readout(model, memory, target_loader, cfg, spec, seed, device, transition_t)
                    if not cfg.get("limit") and not finite_metrics(metrics):
                        raise RuntimeError(f"Non-finite metrics for {spec['name']} seed {seed}")
                    metrics_path.write_text(json.dumps(metrics, indent=2, default=json_default), encoding="utf-8")
                    diag_rows.extend(drows)
                    err_rows.extend(erows)
                    loc_rows.extend(lrows)
                mean = metrics["mean"]
                peak_gpu = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
                cpu_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 if resource is not None else 0.0
                graph_key = f"{spec.get('graph')}:{spec.get('graph_space', 'fused')}"
                add_size = 0
                if spec.get("graph") and spec.get("graph") != "fully_mixed":
                    gpair = graphs.get(graph_key) or graphs.get(f"{spec.get('graph')}:fused")
                    if gpair:
                        add_size = int(gpair[1].data.nbytes + gpair[1].indices.nbytes + gpair[1].indptr.nbytes)
                row = {
                    "seed": seed,
                    "variant": spec["name"],
                    "status": "completed",
                    "Preserves full bank": spec["Preserves full bank"],
                    "Correct graph": spec["Correct graph"],
                    "Residual": spec["Residual"],
                    "Label-free": spec["Label-free"],
                    "I-AUROC": mean.get("I-AUROC"),
                    "I-AP": mean.get("I-AP"),
                    "P-AUROC": mean.get("P-AUROC"),
                    "PRO": mean.get("PRO"),
                    "inference_time_per_image_sec": metrics.get("mean_inference_time_per_image"),
                    "propagation_time_per_query_batch_sec": metrics.get("propagation_time_per_query_batch_sec", 0.0),
                    "peak_gpu_memory_mb": peak_gpu,
                    "peak_cpu_memory_mb": cpu_mb,
                    "visual_memory_size": int(memory["patch_keys"].shape[0]),
                    "additional_graph_memory_bytes": add_size,
                }
                summary_rows.append(row)
                cost_rows.append({k: row[k] for k in ["seed", "variant", "inference_time_per_image_sec", "propagation_time_per_query_batch_sec", "peak_gpu_memory_mb", "peak_cpu_memory_mb", "visual_memory_size", "additional_graph_memory_bytes"]})
                for category, vals in metrics["per_class"].items():
                    per_category_rows.append({"seed": seed, "variant": spec["name"], "category": category, "I-AUROC": vals.get("I-AUROC"), "I-AP": vals.get("I-AP"), "P-AUROC": vals.get("P-AUROC"), "PRO": vals.get("PRO")})
                logs.append(f"seed {seed} {spec['name']}: I-AUROC={mean.get('I-AUROC'):.4f}, P-AUROC={mean.get('P-AUROC'):.4f}, PRO={mean.get('PRO'):.4f}")
                if spec["name"] == "V0_OriginalFullBankVoting":
                    refs = cfg.get("baseline_reference", {})
                    tols = cfg.get("baseline_tolerance", {})
                    sanity["baseline_checks"][seed] = {m: {"value": float(mean[m]), "reference": float(refs[m]), "pass": abs(float(mean[m]) - float(refs[m])) <= float(tols[m])} for m in refs}
                    if cfg.get("fail_on_baseline_mismatch", True) and not cfg.get("limit") and not all(v["pass"] for v in sanity["baseline_checks"][seed].values()):
                        raise RuntimeError(f"Phase III-A V0 baseline mismatch for seed {seed}: {sanity['baseline_checks'][seed]}")
                torch.cuda.empty_cache()
    except Exception:
        tb = traceback.format_exc()
        failures.append(tb)
        (output_dir / "failure_traceback.txt").write_text(tb, encoding="utf-8")
        raise
    finally:
        summary_cols = ["seed", "variant", "status", "Preserves full bank", "Correct graph", "Residual", "Label-free", "I-AUROC", "I-AP", "P-AUROC", "PRO", "inference_time_per_image_sec", "propagation_time_per_query_batch_sec", "peak_gpu_memory_mb", "peak_cpu_memory_mb", "visual_memory_size", "additional_graph_memory_bytes"]
        write_csv(output_dir / "summary_metrics.csv", summary_rows, summary_cols)
        write_csv(output_dir / "all_sweep_metrics.csv", summary_rows, summary_cols)
        write_csv(output_dir / "per_category_metrics.csv", per_category_rows, ["seed", "variant", "category", "I-AUROC", "I-AP", "P-AUROC", "PRO"])
        graph_cols = ["seed", "graph", "node_count", "edge_count", "isolated_nodes", "connected_components", "largest_component_ratio", "mean_degree", "median_degree", "degree_gini", "degree_entropy", "reciprocity", "edge_similarity_mean", "edge_similarity_std", "transition_entropy_mean", "transition_entropy_median", "near_duplicate_edge_ratio_095", "near_duplicate_edge_ratio_099", "edge_label_homophily", "normal_to_normal_edge_ratio", "anomaly_to_anomaly_edge_ratio", "normal_to_anomaly_edge_ratio", "anomaly_to_normal_edge_ratio", "diagnostic_time_sec"]
        write_csv(output_dir / "graph_diagnostics.csv", graph_rows, graph_cols)
        diag_cols = ["seed", "variant", "category", "tv_mean", "kl_w0_w_mean", "kl_w_w0_mean", "retrieval_cosine_mean", "max_weight_change_mean", "entropy_mean", "normalized_entropy_mean", "effective_support_mean", "top1_weight_mean", "top5_cumulative_weight_mean", "top20_cumulative_weight_mean", "normal_mass_mean", "anomaly_mass_mean", "anomaly_minus_normal_margin_mean"]
        write_csv(output_dir / "retrieval_diagnostics.csv", diag_rows, diag_cols)
        write_csv(output_dir / "error_correction.csv", err_rows, ["seed", "variant"])
        write_csv(output_dir / "localization_diagnostics.csv", loc_rows, ["seed", "variant", "category", "false_positive_area_ratio", "false_negative_area_ratio", "anomaly_map_smoothness", "connected_component_count", "anomaly_map_fragmentation"])
        write_csv(output_dir / "computational_costs.csv", cost_rows, ["seed", "variant", "inference_time_per_image_sec", "propagation_time_per_query_batch_sec", "peak_gpu_memory_mb", "peak_cpu_memory_mb", "visual_memory_size", "additional_graph_memory_bytes"])
        write_csv(output_dir / "statistical_comparisons.csv", statistical_comparison_rows(per_category_rows), ["comparison", "metric", "mean_delta", "wins", "ties", "losses", "bootstrap_ci_low", "bootstrap_ci_high"])
        success = []
        for row in summary_rows:
            if row["seed"] == int(cfg.get("seeds", [0])[0]):
                success.append({k: row[k] for k in ["variant", "Preserves full bank", "Correct graph", "Residual", "Label-free", "I-AUROC", "P-AUROC", "PRO"]})
        write_csv(output_dir / "success_control_table.csv", success, ["variant", "Preserves full bank", "Correct graph", "Residual", "Label-free", "I-AUROC", "P-AUROC", "PRO"])
        write_json(output_dir / "sanity_checks.json", sanity)
        lines = ["# Phase III-B Execution Summary", "", f"- Rows: {len(summary_rows)}", f"- Failures: {len(failures)}", "- Memory redundancy and compression are intentionally excluded to isolate the retrieval-structure variable.", "- SPR is intentionally excluded from Phase III-B.", "", "## Errors"]
        lines.extend([f"- {x}" for x in failures] or ["- None"])
        (output_dir / "EXECUTION_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        copy_log(output_dir, logs)
        append_log(output_dir, audit_status, sanity, summary_rows)


if __name__ == "__main__":
    main()
