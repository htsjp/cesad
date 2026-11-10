import json
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
from performance_visual_retrieval.spatial import aggregate_feature, anomaly_map_sharpness
from shared_virtual_geometry.factorized_similarity import reconstruct_keys
from shared_virtual_geometry.regional_scale import regional_rms_scale
from shared_virtual_geometry.storage_accounting import original_coeff_storage
from static_memory_reliability.source_protocol import clone_filtered_loader, source_protocol_record, split_categories
from ultra_compact_memory.centroid_pca import fit_centroid_pca, reconstruct_centroids
from ultra_compact_memory.shared_rank2 import decompose_fullfps, fit_residual_rank2
from ultra_compact_memory.transfer_protocol import transfer_directions


def build_backend(memory, cfg):
    full = memory["patch_keys"].float()
    _, mu, centered = decompose_fullfps(full, 4)
    residual = fit_residual_rank2(centered)
    rank = int(cfg.get("compact_rank", 8))
    pca = fit_centroid_pca(mu, rank)
    mu_hat = reconstruct_centroids(pca, rank)
    return {"keys": reconstruct_keys(mu_hat, residual["basis"], residual["coeff"], normalize=True), "regions": mu.shape[0], "dim": full.shape[-1], "rank": rank}


def make_memory(base, keys, device):
    return {"image_keys": base["image_keys"].float().to(device), "image_values": base["image_values"].float().to(device), "patch_keys": keys.float().to(device), "patch_values": base["patch_values"].float().to(device)}


def variant_specs(cfg):
    etas = cfg.get("eta_grid", [0.25, 0.5, 1.0])
    specs = [{"variant": "B0_SingleScaleBaseline", "mode": "feature", "eta3": 0.0, "eta5": 0.0, "weights": [1, 0, 0]}]
    for e in etas:
        specs += [
            {"variant": f"B1_Feature3x3_eta{e}".replace(".", "p"), "mode": "feature", "eta3": e, "eta5": 0.0, "weights": [1, 0, 0]},
            {"variant": f"B2_Feature5x5_eta{e}".replace(".", "p"), "mode": "feature", "eta3": 0.0, "eta5": e, "weights": [1, 0, 0]},
            {"variant": f"B5_FeatureFusion_1_3_5_eta{e}".replace(".", "p"), "mode": "feature", "eta3": e, "eta5": e, "weights": [1, 0, 0]},
        ]
    for weights in cfg.get("score_weights", [[0.5, 0.5, 0], [0.5, 0, 0.5], [1 / 3, 1 / 3, 1 / 3]]):
        specs.append({"variant": "B8_ScoreFusion_" + "_".join(str(x).replace(".", "p") for x in weights), "mode": "score", "eta3": 0.5, "eta5": 0.5, "weights": weights})
    return specs


def retrieval(q, memory):
    sim = q @ memory["patch_keys"].T
    w = torch.softmax(sim, dim=-1)
    return w @ memory["patch_values"].float()


@torch.no_grad()
def evaluate_spatial(model, memory, loader, cfg, device, desc, spec):
    image_size, sigma = int(cfg.get("image_size", 518)), float(cfg.get("sigma", 4))
    fusion_k, topk_ratio = float(cfg.get("fusion_k", 0.7)), float(cfg.get("topk_ratio", 0.01))
    results = init_results(loader.dataset.obj_list)
    sharpness = []
    for items in tqdm(loader, desc=f"eval {desc}"):
        image = items["img"].to(device)
        cls_name = items["cls_name"][0]
        gt_mask = items["img_mask"]
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name]["imgs_masks"].append(gt_mask)
        results[cls_name]["gt_sp"].extend(torch.as_tensor(items["anomaly"]).detach().cpu())
        outputs = extract_value_representations(model, image, [12, 18, 24], device, smooth=cfg.get("average_neighbor", True))
        cls = cls_from_outputs(outputs, 24)
        q1 = canonical_fuse(outputs[24]["T"][:, 1:, :], outputs[12]["V"][:, 1:, :], outputs[18]["V"][:, 1:, :], alpha=0.5)
        image_logits = torch.softmax(cls @ memory["image_keys"].T, dim=-1) @ memory["image_values"].float()
        if spec["mode"] == "feature":
            q = aggregate_feature(q1, spec.get("eta3", 0.0), spec.get("eta5", 0.0), cfg.get("boundary", "reflect"))
            patch_logits = retrieval(q, memory)
        else:
            qs = [q1, aggregate_feature(q1, 0.5, 0.0, cfg.get("boundary", "reflect")), aggregate_feature(q1, 0.0, 0.5, cfg.get("boundary", "reflect"))]
            maps = []
            for q in qs:
                maps.append(retrieval(q, memory))
            weights = torch.tensor(spec["weights"], device=device, dtype=maps[0].dtype)
            patch_logits = sum(float(weights[i]) * maps[i] for i in range(3))
        amap = AnomalyCLIP_lib.get_similarity_map(patch_logits, image_size)[..., 1]
        amap = torch.stack([torch.from_numpy(gaussian_filter(x, sigma=sigma)) for x in amap.detach().cpu()], dim=0)
        sharpness.append(anomaly_map_sharpness(amap))
        flat = amap.flatten(start_dim=1)
        topn = max(1, int(flat.shape[1] * topk_ratio))
        score = (1 - fusion_k) * torch.topk(flat, topn, dim=1).values.mean(dim=1).to(device) + fusion_k * image_logits[:, 1]
        results[cls_name]["pr_sp"].extend(score.detach().cpu())
        results[cls_name]["anomaly_maps"].append(amap.detach().cpu())
    metrics = evaluate_results(results, loader.dataset.obj_list, compute_pro=cfg.get("compute_pixel_aupro", True))
    return metrics, {"map_sharpness": float(np.mean(sharpness)) if sharpness else float("nan")}, [{"category": k, **v} for k, v in metrics.get("per_class", {}).items()]


def eval_or_load(model, out, cfg, seed, direction, split, variant, memory, loader, device, spec):
    mpath = out / f"seed_{seed}" / direction / split / variant / "metrics.json"
    if cfg.get("resume", True) and mpath.exists():
        p = json.loads(mpath.read_text(encoding="utf-8"))
        return p["metrics"], p["diagnostics"], p.get("per_category", [])
    metrics, diag, pc = evaluate_spatial(model, memory, loader, cfg, device, f"{direction}:{split}:{variant}", spec)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps({"metrics": metrics, "diagnostics": diag, "per_category": pc}, indent=2, default=json_default), encoding="utf-8")
    return metrics, diag, pc


def select_source(rows):
    out = {}
    for d in sorted({r["direction"] for r in rows}):
        ds = [r for r in rows if r["direction"] == d]
        base = next(r for r in ds if r["variant"] == "B0_SingleScaleBaseline")
        valid = [r for r in ds if float(r["I-AUROC"]) >= float(base["I-AUROC"]) - 0.002]
        valid.sort(key=lambda r: (float(r["P-AUROC"]), float(r["PRO"]), float(r["I-AUROC"])), reverse=True)
        best = valid[0]
        if max(float(best["P-AUROC"]) - float(base["P-AUROC"]), float(best["PRO"]) - float(base["PRO"])) < 0.001:
            best = base
        out[d] = {"variant": best["variant"], "mode": best["mode"], "eta3": best["eta3"], "eta5": best["eta5"], "weights": best["weights"]}
    return out


def main():
    args = parse_args("Phase VIII-B multi-scale spatial aggregation")
    cfg = load_config(args.config, {"dataset": args.dataset, "source_dataset": args.source_dataset, "device": args.device, "output_dir": args.output_dir, "limit": args.limit, "seed": args.seed})
    seed = int(cfg.get("seed", 0)); set_seed(seed)
    out = Path(cfg.get("output_dir", "results/phase_viii_b_spatial")); out.mkdir(parents=True, exist_ok=True); save_run_metadata(out, cfg, seed)
    device = cfg.get("device", "cuda:0")
    rows=[]; source_rows=[]; target_rows=[]; pc_rows=[]; map_rows=[]; protocol=[]
    sanity={"canonical_formula": FORMULA_NAME, "boundary_rule": cfg.get("boundary","reflect"), "target_labels_used_for_selection": False}
    try:
        model = load_model(cfg, device)
        for direction in transfer_directions(cfg):
            dname, source, target = direction["name"], direction["source"], direction["target"]
            source_data, source_loader_all = make_loader(cfg, source, cfg.get("source_mode","test"), shuffle=False)
            _, target_loader = make_loader(cfg, target, cfg.get("target_mode","test"), shuffle=False)
            train_cats, val_cats = split_categories(source_data, seed, cfg.get("source_val_fraction",0.25))
            train_loader = clone_filtered_loader(source_loader_all, train_cats, batch_size=cfg.get("batch_size",1), shuffle=False)
            val_loader = clone_filtered_loader(source_loader_all, val_cats, batch_size=cfg.get("batch_size",1), shuffle=False)
            protocol.append(source_protocol_record(seed,dname,source,target,train_cats,val_cats))
            train_memory,_=build_source_memory(model,train_loader,cfg,f"{source}_viiib_foldtrain_{seed}",seed,device,out)
            backend=build_backend(train_memory,cfg); mem=make_memory(train_memory,backend["keys"],device); storage=original_coeff_storage(backend["regions"],backend["dim"],backend["rank"])
            for spec in variant_specs(cfg):
                metrics,diag,pc=eval_or_load(model,out,cfg,seed,dname,"source_val",spec["variant"],mem,val_loader,device,spec)
                r={"seed":seed,"direction":dname,"source":source,"target":target,"split":"source_val","variant":spec["variant"],"mode":spec["mode"],"eta3":spec["eta3"],"eta5":spec["eta5"],"weights":json.dumps(spec["weights"]),"storage_ratio":storage["storage_ratio"],"I-AUROC":metric(metrics,"I-AUROC"),"I-AP":metric(metrics,"I-AP"),"P-AUROC":metric(metrics,"P-AUROC"),"PRO":metric(metrics,"PRO"),**diag}
                rows.append(r); source_rows.append(r); map_rows.append(r)
                for p in pc: pc_rows.append({"seed":seed,"direction":dname,"split":"source_val","variant":spec["variant"],**p})
            selection=select_source(source_rows)[dname]
            full_memory,_=build_source_memory(model,source_loader_all,cfg,f"{source}_viiib_full_{seed}",seed,device,out)
            full_backend=build_backend(full_memory,cfg); full_mem=make_memory(full_memory,full_backend["keys"],device); full_storage=original_coeff_storage(full_backend["regions"],full_backend["dim"],full_backend["rank"])
            target_specs=[next(s for s in variant_specs(cfg) if s["variant"]=="B0_SingleScaleBaseline"),selection,{"variant":"C5_DuplicateSingleScale","mode":"score","eta3":0,"eta5":0,"weights":[1,0,0]}]
            for spec in target_specs:
                metrics,diag,pc=eval_or_load(model,out,cfg,seed,dname,"target",spec["variant"],full_mem,target_loader,device,spec)
                r={"seed":seed,"direction":dname,"source":source,"target":target,"split":"target","variant":spec["variant"],"mode":spec["mode"],"eta3":spec["eta3"],"eta5":spec["eta5"],"weights":json.dumps(spec["weights"]),"storage_ratio":full_storage["storage_ratio"],"I-AUROC":metric(metrics,"I-AUROC"),"I-AP":metric(metrics,"I-AP"),"P-AUROC":metric(metrics,"P-AUROC"),"PRO":metric(metrics,"PRO"),**diag}
                rows.append(r); target_rows.append(r); map_rows.append(r)
                for p in pc: pc_rows.append({"seed":seed,"direction":dname,"split":"target","variant":spec["variant"],**p})
        outcome="B7_NO_SPATIAL_GAIN"
        gains=[]
        for d in sorted({r["direction"] for r in target_rows}):
            ds=[r for r in target_rows if r["direction"]==d]; b=next(r for r in ds if r["variant"]=="B0_SingleScaleBaseline"); s=next((r for r in ds if r["variant"]!="B0_SingleScaleBaseline" and not r["variant"].startswith("C5")),b)
            gains.append((float(s["P-AUROC"])-float(b["P-AUROC"]), float(s["PRO"])-float(b["PRO"])))
        if gains and all(g[0]>0 or g[1]>0 for g in gains) and (sum(max(g) for g in gains)/len(gains)>0.003): outcome="B1_MULTISCALE_SPATIAL_SUPPORTED"
        elif gains and any(max(g)>0.0015 for g in gains): outcome="B6_DATASET_SPECIFIC_SPATIAL_GAIN"
        cols=["seed","direction","source","target","split","variant","mode","eta3","eta5","weights","storage_ratio","I-AUROC","I-AP","P-AUROC","PRO","map_sharpness"]
        write_csv(out/"summary_metrics.csv",target_rows,cols); write_csv(out/"all_variant_metrics.csv",rows,cols); write_csv(out/"per_category_metrics.csv",pc_rows,sorted({k for r in pc_rows for k in r}) if pc_rows else ["seed","direction","variant"]); write_csv(out/"map_sharpness.csv",map_rows,cols)
        write_csv(out/"spatial_controls.csv",[r for r in rows if r["variant"].startswith("C")],cols); write_csv(out/"defect_size_metrics.csv",[],["seed","direction","variant","defect_size","P-AUROC","PRO"]); write_csv(out/"storage_accounting.csv",[],["seed","direction","variant","storage_ratio"]); write_csv(out/"computational_costs.csv",[],["seed","direction","variant","time"])
        write_json(out/"source_protocol.json",{"records":protocol}); write_json(out/"source_selection.json",select_source(source_rows)); write_json(out/"sanity_checks.json",sanity); write_json(out/"OUTCOME.json",{"Outcome":outcome,**sanity})
        write_md(out/"EXECUTION_SUMMARY.md",["# Phase VIII-B Execution Summary","",f"- Completed target rows: {len(target_rows)}.",f"- Outcome: `{outcome}`."])
        with open(ROOT_DIR/"EXPERIMENT_LOG.md","a",encoding="utf-8") as f: f.write(f"\n## Phase VIII-B — Multi-Scale Spatial Aggregation\n\n- Output directory: `{out}`.\n- Outcome: `{outcome}`.\n")
    except Exception:
        tb=traceback.format_exc(); (out/"failure_traceback.txt").write_text(tb,encoding="utf-8"); write_json(out/"OUTCOME.json",{"Outcome":"B8_IMPLEMENTATION_FAILURE","error":tb}); raise


if __name__ == "__main__":
    main()
