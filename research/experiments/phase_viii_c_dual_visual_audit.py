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
from clip_value_features.extractors import cls_from_outputs, extract_value_representations
from cross_dataset_ultra_compression_audit import build_source_memory, json_default, metric, write_json, write_md
from mrad_audit_utils import evaluate_results, init_results, load_config, load_model, make_loader, parse_args, save_run_metadata, set_seed, write_csv
from performance_visual_retrieval.dual_spaces import branch_features, build_branch_memory, score_standardize
from shared_virtual_geometry.factorized_similarity import reconstruct_keys
from shared_virtual_geometry.storage_accounting import original_coeff_storage
from static_memory_reliability.source_protocol import clone_filtered_loader, source_protocol_record, split_categories
from ultra_compact_memory.centroid_pca import fit_centroid_pca, reconstruct_centroids
from ultra_compact_memory.shared_rank2 import decompose_fullfps, fit_residual_rank2
from ultra_compact_memory.transfer_protocol import transfer_directions


def atomic_torch_save(obj, path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp"); torch.save(obj,tmp); tmp.replace(path)


def build_backend(memory, cfg):
    full=memory["patch_keys"].float(); _,mu,centered=decompose_fullfps(full,4); residual=fit_residual_rank2(centered); rank=int(cfg.get("compact_rank",8)); pca=fit_centroid_pca(mu,rank); mu_hat=reconstruct_centroids(pca,rank)
    return {"keys":reconstruct_keys(mu_hat,residual["basis"],residual["coeff"],normalize=True),"regions":mu.shape[0],"dim":full.shape[-1],"rank":rank}


def make_memory(base, keys, device):
    return {"image_keys":base["image_keys"].float().to(device),"image_values":base["image_values"].float().to(device),"patch_keys":keys.float().to(device),"patch_values":base["patch_values"].float().to(device)}


def retrieve(q, memory):
    sim=q@memory["patch_keys"].T; w=torch.softmax(sim,dim=-1); return w@memory["patch_values"].float()


def score_stats(vals):
    arr=np.asarray(vals,dtype=np.float64); med=float(np.median(arr)) if arr.size else 0.0
    return {"mean":float(arr.mean()) if arr.size else 0.0,"std":float(arr.std()) if arr.size else 1.0,"median":med,"mad":float(np.median(np.abs(arr-med))) if arr.size else 1.0,"temperature":float(arr.std()) if arr.size else 1.0}


@torch.no_grad()
def evaluate_dual(model, memories, loader, cfg, device, desc, spec, collect_scores=False):
    image_size=int(cfg.get("image_size",518)); sigma=float(cfg.get("sigma",4)); fusion_k=float(cfg.get("fusion_k",0.7)); topk_ratio=float(cfg.get("topk_ratio",0.01))
    results=init_results(loader.dataset.obj_list); sem_scores=[]; loc_scores=[]; corr_pairs=[]
    for items in tqdm(loader,desc=f"eval {desc}"):
        image=items["img"].to(device); cls_name=items["cls_name"][0]; gt_mask=items["img_mask"]; gt_mask[gt_mask>0.5],gt_mask[gt_mask<=0.5]=1,0
        results[cls_name]["imgs_masks"].append(gt_mask); results[cls_name]["gt_sp"].extend(torch.as_tensor(items["anomaly"]).detach().cpu())
        outputs=extract_value_representations(model,image,[12,18,24],device,smooth=cfg.get("average_neighbor",True)); cls=cls_from_outputs(outputs,24)
        image_logits=torch.softmax(cls@memories["early"]["image_keys"].T,dim=-1)@memories["early"]["image_values"].float()
        maps={}
        for branch in ["early","semantic","local"]:
            q=branch_features(outputs, "early" if branch=="early" else branch)
            logits=retrieve(q,memories[branch])
            amap=AnomalyCLIP_lib.get_similarity_map(logits,image_size)[...,1]
            amap=torch.stack([torch.from_numpy(gaussian_filter(x,sigma=sigma)) for x in amap.detach().cpu()],dim=0)
            maps[branch]=amap
        flat_sem=maps["semantic"].flatten(start_dim=1); flat_loc=maps["local"].flatten(start_dim=1); flat_early=maps["early"].flatten(start_dim=1)
        topn=max(1,int(flat_early.shape[1]*topk_ratio))
        s_sem=torch.topk(flat_sem,topn,dim=1).values.mean(dim=1); s_loc=torch.topk(flat_loc,topn,dim=1).values.mean(dim=1); s_early=torch.topk(flat_early,topn,dim=1).values.mean(dim=1)
        if collect_scores:
            sem_scores.extend(s_sem.tolist()); loc_scores.extend(s_loc.tolist())
        if spec["variant"]=="C0_EarlyFeatureFusionBaseline":
            patch_score=s_early; amap=maps["early"]
        elif spec["variant"]=="C1_SemanticOnly":
            patch_score=s_sem; amap=maps["semantic"]
        elif spec["variant"]=="C2_LocalOnly":
            patch_score=s_loc; amap=maps["local"]
        elif spec["variant"]=="C8_SemanticImage_LocalPixel":
            patch_score=s_loc; amap=maps["local"]
        else:
            st=spec.get("stats",{}); method=spec.get("calibration","none"); lam=float(spec.get("lambda",0.5))
            ss=torch.tensor([score_standardize(float(x),st.get("semantic",{}),method) for x in s_sem.tolist()])
            sl=torch.tensor([score_standardize(float(x),st.get("local",{}),method) for x in s_loc.tolist()])
            patch_score=lam*ss+(1-lam)*sl
            amap=lam*maps["semantic"]+(1-lam)*maps["local"]
        score=(1-fusion_k)*patch_score.to(device)+fusion_k*image_logits[:,1]
        results[cls_name]["pr_sp"].extend(score.detach().cpu()); results[cls_name]["anomaly_maps"].append(amap.detach().cpu())
        corr_pairs.extend(zip(s_sem.tolist(),s_loc.tolist()))
    if collect_scores:
        return {"semantic":score_stats(sem_scores),"local":score_stats(loc_scores)}
    metrics=evaluate_results(results,loader.dataset.obj_list,compute_pro=cfg.get("compute_pixel_aupro",True))
    diag={"branch_score_correlation":float(np.corrcoef(np.asarray(corr_pairs).T)[0,1]) if len(corr_pairs)>2 else float("nan")}
    return metrics,diag,[{"category":k,**v} for k,v in metrics.get("per_class",{}).items()]


def eval_or_load(model,out,cfg,seed,direction,split,variant,memories,loader,device,spec):
    mpath=out/f"seed_{seed}"/direction/split/variant/"metrics.json"
    if cfg.get("resume",True) and mpath.exists():
        p=json.loads(mpath.read_text(encoding="utf-8")); return p["metrics"],p["diagnostics"],p.get("per_category",[])
    metrics,diag,pc=evaluate_dual(model,memories,loader,cfg,device,f"{direction}:{split}:{variant}",spec)
    mpath.parent.mkdir(parents=True,exist_ok=True); mpath.write_text(json.dumps({"metrics":metrics,"diagnostics":diag,"per_category":pc},indent=2,default=json_default),encoding="utf-8")
    return metrics,diag,pc


def specs(cfg, stats=None):
    out=[{"variant":"C0_EarlyFeatureFusionBaseline","calibration":"none","lambda":0.5},{"variant":"C1_SemanticOnly","calibration":"none","lambda":1.0},{"variant":"C2_LocalOnly","calibration":"none","lambda":0.0}]
    for method,var in [("none","C3_UncalibratedDualScore"),("zscore","C4_ZScoreCalibratedDual"),("temperature","C5_TemperatureCalibratedDual"),("robust","C6_RobustCalibratedDual")]:
        for lam in cfg.get("lambda_grid",[0.25,0.5,0.75]):
            out.append({"variant":f"{var}_lambda{lam}".replace(".","p"),"calibration":method,"lambda":lam,"stats":stats or {}})
    out.append({"variant":"C8_SemanticImage_LocalPixel","calibration":"none","lambda":0.0})
    return out


def select_source(rows):
    sel={}
    for d in sorted({r["direction"] for r in rows}):
        ds=[r for r in rows if r["direction"]==d]; base=next(r for r in ds if r["variant"]=="C0_EarlyFeatureFusionBaseline")
        valid=[r for r in ds if float(r["PRO"])>=float(base["PRO"])-0.002]
        valid.sort(key=lambda r:(float(r["P-AUROC"]),float(r["I-AUROC"]),float(r["PRO"])),reverse=True)
        best=valid[0]
        if max(float(best["I-AUROC"])-float(base["I-AUROC"]),float(best["P-AUROC"])-float(base["P-AUROC"]))<0.001:
            best=base
        sel[d]=best
    return sel


def main():
    args=parse_args("Phase VIII-C dual visual space compact retrieval"); cfg=load_config(args.config,{"dataset":args.dataset,"source_dataset":args.source_dataset,"device":args.device,"output_dir":args.output_dir,"limit":args.limit,"seed":args.seed})
    seed=int(cfg.get("seed",0)); set_seed(seed); out=Path(cfg.get("output_dir","results/phase_viii_c_dual")); out.mkdir(parents=True,exist_ok=True); save_run_metadata(out,cfg,seed); device=cfg.get("device","cuda:0")
    rows=[]; source_rows=[]; target_rows=[]; pc_rows=[]; comp_rows=[]; protocol=[]; sanity={"target_labels_used_for_selection":False,"dual_memory_rank":int(cfg.get("compact_rank",8))}
    try:
        model=load_model(cfg,device)
        for direction in transfer_directions(cfg):
            dname,source,target=direction["name"],direction["source"],direction["target"]
            source_data,source_loader_all=make_loader(cfg,source,cfg.get("source_mode","test"),shuffle=False); _,target_loader=make_loader(cfg,target,cfg.get("target_mode","test"),shuffle=False)
            train_cats,val_cats=split_categories(source_data,seed,cfg.get("source_val_fraction",0.25)); train_loader=clone_filtered_loader(source_loader_all,train_cats,batch_size=cfg.get("batch_size",1),shuffle=False); val_loader=clone_filtered_loader(source_loader_all,val_cats,batch_size=cfg.get("batch_size",1),shuffle=False)
            protocol.append(source_protocol_record(seed,dname,source,target,train_cats,val_cats))
            memories={}
            for br in ["early","semantic","local"]:
                ckpt=out/"checkpoints"/f"{source}_{br}_foldtrain_{seed}.pt"
                if cfg.get("resume",True) and ckpt.exists(): raw=torch.load(ckpt,map_location="cpu",weights_only=False)
                else: raw=build_branch_memory(model,train_loader,cfg,br,seed,device); atomic_torch_save(raw,ckpt)
                backend=build_backend(raw,cfg); memories[br]=make_memory(raw,backend["keys"],device)
            stats=evaluate_dual(model,memories,val_loader,cfg,device,f"{dname}:stats",{"variant":"C0_EarlyFeatureFusionBaseline"},collect_scores=True)
            for sp in specs(cfg,stats):
                metrics,diag,pc=eval_or_load(model,out,cfg,seed,dname,"source_val",sp["variant"],memories,val_loader,device,sp)
                r={"seed":seed,"direction":dname,"source":source,"target":target,"split":"source_val","variant":sp["variant"],"calibration":sp["calibration"],"lambda":sp["lambda"],"I-AUROC":metric(metrics,"I-AUROC"),"I-AP":metric(metrics,"I-AP"),"P-AUROC":metric(metrics,"P-AUROC"),"PRO":metric(metrics,"PRO"),**diag}
                rows.append(r); source_rows.append(r); comp_rows.append(r)
                for p in pc: pc_rows.append({"seed":seed,"direction":dname,"split":"source_val","variant":sp["variant"],**p})
            selected=select_source(source_rows)[dname]
            full_memories={}
            storage_ratio=0.0
            for br in ["early","semantic","local"]:
                ckpt=out/"checkpoints"/f"{source}_{br}_full_{seed}.pt"
                if cfg.get("resume",True) and ckpt.exists(): raw=torch.load(ckpt,map_location="cpu",weights_only=False)
                else: raw=build_branch_memory(model,source_loader_all,cfg,br,seed,device); atomic_torch_save(raw,ckpt)
                backend=build_backend(raw,cfg); full_memories[br]=make_memory(raw,backend["keys"],device); storage_ratio+=original_coeff_storage(backend["regions"],backend["dim"],backend["rank"])["storage_ratio"]
            target_specs=[{"variant":"C0_EarlyFeatureFusionBaseline","calibration":"none","lambda":0.5},selected,{"variant":"D1_DuplicateSemanticBranch","calibration":"none","lambda":1.0},{"variant":"D2_DuplicateLocalBranch","calibration":"none","lambda":0.0}]
            for sp in target_specs:
                metrics,diag,pc=eval_or_load(model,out,cfg,seed,dname,"target",sp["variant"],full_memories,target_loader,device,sp)
                r={"seed":seed,"direction":dname,"source":source,"target":target,"split":"target","variant":sp["variant"],"calibration":sp.get("calibration",""),"lambda":sp.get("lambda",""),"storage_ratio":storage_ratio,"I-AUROC":metric(metrics,"I-AUROC"),"I-AP":metric(metrics,"I-AP"),"P-AUROC":metric(metrics,"P-AUROC"),"PRO":metric(metrics,"PRO"),**diag}
                rows.append(r); target_rows.append(r); comp_rows.append(r)
                for p in pc: pc_rows.append({"seed":seed,"direction":dname,"split":"target","variant":sp["variant"],**p})
        outcome="C8_NO_DUAL_VISUAL_GAIN"; gains=[]
        for d in sorted({r["direction"] for r in target_rows}):
            ds=[r for r in target_rows if r["direction"]==d]; b=next(r for r in ds if r["variant"]=="C0_EarlyFeatureFusionBaseline"); s=next((r for r in ds if r["variant"]!="C0_EarlyFeatureFusionBaseline"),b); gains.append((float(s["I-AUROC"])-float(b["I-AUROC"]),float(s["P-AUROC"])-float(b["P-AUROC"])))
        if gains and all(g[0]>0 or g[1]>0 for g in gains) and sum(max(g) for g in gains)/len(gains)>0.003: outcome="C1_DUAL_VISUAL_RETRIEVAL_SUPPORTED"
        elif gains and any(max(g)>0.0015 for g in gains): outcome="C7_DATASET_SPECIFIC_DUAL_GAIN"
        cols=["seed","direction","source","target","split","variant","calibration","lambda","storage_ratio","I-AUROC","I-AP","P-AUROC","PRO","branch_score_correlation"]
        write_csv(out/"summary_metrics.csv",target_rows,cols); write_csv(out/"all_variant_metrics.csv",rows,cols); write_csv(out/"per_category_metrics.csv",pc_rows,sorted({k for r in pc_rows for k in r}) if pc_rows else ["seed","direction","variant"]); write_csv(out/"branch_complementarity.csv",comp_rows,cols)
        write_csv(out/"calibration_metrics.csv",source_rows,cols); write_csv(out/"defect_size_metrics.csv",[],["seed","direction","variant","defect_size"]); write_csv(out/"storage_accounting.csv",[],["seed","direction","variant","storage_ratio"]); write_csv(out/"computational_costs.csv",[],["seed","direction","variant","time"])
        write_json(out/"source_protocol.json",{"records":protocol}); write_json(out/"source_selection.json",select_source(source_rows)); write_json(out/"sanity_checks.json",sanity); write_json(out/"OUTCOME.json",{"Outcome":outcome,**sanity})
        write_md(out/"EXECUTION_SUMMARY.md",["# Phase VIII-C Execution Summary","",f"- Completed target rows: {len(target_rows)}.",f"- Outcome: `{outcome}`."])
        with open(ROOT_DIR/"EXPERIMENT_LOG.md","a",encoding="utf-8") as f: f.write(f"\n## Phase VIII-C — Dual Visual Space Compact Retrieval\n\n- Output directory: `{out}`.\n- Outcome: `{outcome}`.\n")
    except Exception:
        tb=traceback.format_exc(); (out/"failure_traceback.txt").write_text(tb,encoding="utf-8"); write_json(out/"OUTCOME.json",{"Outcome":"C9_IMPLEMENTATION_FAILURE","error":tb}); raise


if __name__=="__main__":
    main()
