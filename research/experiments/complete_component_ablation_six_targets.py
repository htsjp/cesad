"""Complete corrected seed0 component evaluation without training.

One frozen CLIP local extraction is shared by the no-NTR and NTR scorers.
Historical capacity checkpoints are deliberately not substituted into this run.
"""
import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.unify_corrected_ntr_category_macro import (
    source_scoring_info, load_v, HeadSpec, score_patch_grid_multi,
)
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import quadrant_crops, torch_resize, mask_np
from experiments.phase_ces_linear_head_control import stitch_local_maps
from experiments.phase_dagm_visa_eval import DAGMTargetDataset
from experiments.phase_median_spatial_consensus_generalization import MetaDataset, HOLDOUTS
from experiments.phase_sgpr import safe_ap_binary
from experiments.phase_xlv_cross_domain_transfer import CHECKPOINT
from experiments.phase_x0a_mandatory_gates import load_components
from phase_xa.official_task_forward import normalized_patch_feature, visual_outputs
from utils.transforms import get_transform

RUN = ROOT/'results/component_ablation_six_target_completion'
FINAL = ROOT/'results/paper_component_ablation_final'
VARS=('Trained_CES_Core','Trained_CES_NTR','Trained_CES_LRP','Trained_CES_NTR_LRP')
BASE=ROOT/'results/ces_ntr_visualization/corrected/checkpoints/Full_VisA_seed0_statealigned.pt'
NTR=ROOT/'results/paper_assets/final_ablation_close/checkpoints/Full_VisA_seed0_rho0.10.pt'


def jsave(path,data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
    temp.replace(path)


def file_id(path):
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def target_data(target):
    if target=='DAGM':
        dataset=DAGMTargetDataset(Path('E:/WorkSpace/dataset/DAGM_KaggleUpload'))
        refs=[{'category':x['cls_name'],'label':int(x['anomaly']),'img_path':x['img_path']} for x in dataset.items]
        gp=ROOT/'results/paper_assets/dagm_visa_eval/cache/dagm_global_patch_float16.npy'
        patch=np.load(gp,mmap_mode='r')
    else:
        dataset=MetaDataset(HOLDOUTS[target][0],mode='test')
        gp=ROOT/f'results/phase_median_spatial_consensus_generalization/formal/feature_cache/{target}_clip.npz'
        with np.load(gp,allow_pickle=True) as pack:
            refs=[dict(x) for x in pack['rows']]
            patch=pack['patch']
        assert len(dataset)==len(refs)
        for i,ref in enumerate(refs):
            raw=dataset.data_all[i]
            assert str(dataset.root/raw['img_path']).replace('\\','/').casefold()==ref['img_path'].replace('\\','/').casefold()
            assert int(raw['anomaly'])==int(ref['label'])
    assert len(dataset)==len(patch)==len(refs)
    assert patch.shape[1:]==(1369,1024)
    return dataset,refs,patch,gp


def verified_masks(dataset, ids, target):
    if target != 'DTD-Synthetic':
        return [mask_np(dataset[i]) for i in ids]
    masks=[]
    for i in ids:
        row=dataset.data_all[i]
        image=dataset.root/row['img_path']
        if int(row['anomaly']):
            parts=list(Path(row['img_path']).parts)
            parts[parts.index('test')]='ground_truth'
            parts[-1]=Path(parts[-1]).stem+'_mask.png'
            gt=dataset.root.joinpath(*parts)
            assert gt.is_file() and gt.resolve()!=image.resolve(),gt
            with Image.open(gt) as raw:
                raw=raw.convert('L')
                assert np.asarray(raw).max()>0,gt
                with Image.open(image) as im: assert raw.size==im.size,(gt,raw.size,im.size)
                # Keep the formal loader's target_transform and >0.5 binarization.
                masks.append((dataset.target_transform(raw).numpy().squeeze()>0.5).astype(np.uint8))
        else:
            with Image.open(image) as im:
                blank=Image.new('L',im.size,0)
                masks.append((dataset.target_transform(blank).numpy().squeeze()>0.5).astype(np.uint8))
    return masks


@torch.inference_mode()
def extract(path,model,preprocess):
    images=torch.stack([preprocess(c) for c in quadrant_crops(path)]).cuda()
    result=normalized_patch_feature(visual_outputs(model,images,adapter=None,grad=False)).float().cpu().numpy()
    assert result.shape==(4,1369,1024) and np.isfinite(result).all()
    return result


def target_run(target,args,info,specs):
    out=RUN/('smoke' if args.smoke else 'formal')/target
    out.mkdir(parents=True,exist_ok=True)
    dataset,refs,global_patch,gp=target_data(target)
    n=min(8,len(refs)) if args.smoke else len(refs)
    refs=refs[:n]
    identity={'target':target,'source':'Full VisA','seed':0,'no_NTR':file_id(BASE),'NTR':file_id(NTR),
        'global_cache':str(gp),'n_images':n,'manifest_hash':hashlib.sha256(json.dumps(refs,sort_keys=True).encode()).hexdigest(),
        'score':'canonical trained CES; NTR rho0.10; no score normalization',
        'LRP':'2x2 nonoverlap;518crop;37patch;0.5global+0.5local74;align_cornersTrue',
        'evaluation':'same corrected M_FR category macro; loader GT masks; no extra smoothing; P_AP only'}
    cp=out/'CONFIG.json'
    if cp.exists(): assert json.loads(cp.read_text(encoding='utf-8'))==identity,'Resume identity mismatch'
    else:
        jsave(cp,identity)
        pd.DataFrame(refs).to_csv(out/'IMAGE_MANIFEST.csv',index=False)
    status=out/'PROGRESS.json'
    done=json.loads(status.read_text())['completed_images'] if status.exists() else 0
    maps={}
    for variant in VARS:
        side=74 if variant.endswith('_LRP') else 37
        path=out/f'{variant}_maps.npy'
        maps[variant]=np.load(path,mmap_mode='r+') if path.exists() else np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(n,side,side))
        assert maps[variant].shape==(n,side,side)
    start=time.time();initial=done
    if done<n:
        preprocess,_=get_transform(argparse.Namespace(image_size=518))
        model,*_=load_components(str(CHECKPOINT),'cuda')
        model.eval()
        for p in model.parameters():p.requires_grad_(False)
        local_cache=None
        candidate=Path('E:/MRAD_final_evidence_cache/local_patch_Full_MVTec__MPDD.npy')
        if target=='MPDD' and candidate.exists():
            cache=np.load(candidate,mmap_mode='r')
            checks=[]
            assert cache.shape==(len(dataset),4,1369,1024)
            for i in np.linspace(0,len(dataset)-1,min(6,n),dtype=int):
                fresh=extract(dataset[i]['img_path'],model,preprocess)
                difference=np.abs(np.asarray(cache[i],dtype=np.float32)-fresh)
                checks.append({'index':int(i),'mean_abs_error':float(difference.mean()),'max_abs_error':float(difference.max())})
            valid=all(x['mean_abs_error']<2e-4 and x['max_abs_error']<0.005 for x in checks)
            jsave(out/'LOCAL_CACHE_VERIFICATION.json',{'path':str(candidate),'checks':checks,'reused':valid,
                'order_evidence':'same sorted MetaDataset test order; producer local_feature_cache_for_cell stores frozen normalized patches before source transform'})
            if valid:local_cache=cache
        with torch.inference_mode():
            for i in range(done,n):
                local=np.asarray(local_cache[i],dtype=np.float32) if local_cache is not None else extract(refs[i]['img_path'],model,preprocess)
                gs=score_patch_grid_multi(info,specs,global_patch[i],'cuda')
                ls={v:[] for v in specs}
                for quadrant in local:
                    for v,score in score_patch_grid_multi(info,specs,quadrant,'cuda').items():ls[v].append(score)
                for v in specs:
                    maps[v][i]=gs[v]
                    lrp='Trained_CES_LRP' if v=='Trained_CES_Core' else 'Trained_CES_NTR_LRP'
                    maps[lrp][i]=.5*torch_resize(gs[v],(74,74),align_corners=True)+.5*stitch_local_maps(np.stack(ls[v]))
                assert all(np.isfinite(a[i]).all() for a in maps.values())
                if (i+1)%25==0 or i+1==n:
                    for a in maps.values():a.flush()
                    elapsed=time.time()-start
                    eta=elapsed/max(1,i+1-initial)*(n-i-1)
                    jsave(status,{'completed_images':i+1,'total_images':n,'stage':'scoring','elapsed_seconds':elapsed,'eta_seconds':eta})
                    print(f'[Components {target}] {i+1}/{n} elapsed={elapsed/60:.1f}min eta={eta/60:.1f}min',flush=True)
        del model
        gc.collect();torch.cuda.empty_cache()
    if args.smoke:
        jsave(out/'OUTCOME.json',{'outcome':'SMOKE_PASSED','finite_maps':True,'n_images':n})
        print(f'[Components {target}] SMOKE_PASSED',flush=True)
        return
    if target=='DTD-Synthetic':
        jsave(out/'GT_PAIRING_CORRECTION.json',{
            'reason':'947 anomaly meta rows point mask_path at the input image',
            'correct_mask':'category/ground_truth/defect/image_stem_mask.png',
            'original_metadata_modified':False,
            'previous_partial_metrics':'PER_CATEGORY_P_AP.csv; INVALID_GT_PAIRING; retained for provenance',
            'new_metrics':'PER_CATEGORY_P_AP_TRUE_GT.csv',
            'evaluator_resize_aggregation_changed':False})
    catpath=out/('PER_CATEGORY_P_AP_TRUE_GT.csv' if target=='DTD-Synthetic' else 'PER_CATEGORY_P_AP.csv')
    rows=pd.read_csv(catpath).to_dict('records') if catpath.exists() else []
    for category in sorted({r['category'] for r in refs}):
        ids=[i for i,r in enumerate(refs) if r['category']==category]
        masks=verified_masks(dataset,ids,target)
        y=np.concatenate([m.reshape(-1) for m in masks])
        for variant in VARS:
            if any(r['category']==category and r['variant']==variant for r in rows):continue
            print(f'[Components {target}] M_FR AP {category} {variant}',flush=True)
            scores=np.concatenate([torch_resize(maps[variant][i],m.shape,align_corners=True).reshape(-1) for i,m in zip(ids,masks)])
            ap=float(safe_ap_binary(y,scores))
            assert np.isfinite(scores).all()
            assert np.isfinite(ap) or np.unique(y).size<2
            rows.append({'source':'Full VisA','target':target,'seed':0,'variant':variant,'category':category,'M_FR_P_AP':ap,'n_images':len(ids),'GT_shape':str(masks[0].shape)})
            pd.DataFrame(rows).to_csv(catpath,index=False)
            del scores;gc.collect()
        del masks,y;gc.collect()
    raw=[]
    for variant in VARS:
        values=[r['M_FR_P_AP'] for r in rows if r['variant']==variant]
        assert len(values)==len({r['category'] for r in refs})
        finite=[float(x) for x in values if np.isfinite(x)]
        assert finite
        raw.append({'source':'Full VisA','target':target,'seed':0,'variant':variant,'M_FR_P_AP':float(np.mean(finite)),
            'aggregation':'CATEGORY_MACRO','checkpoint':str(NTR if 'NTR' in variant else BASE),'n_images':n,
            'n_categories_total':len(values),'n_categories_valid_AP':len(finite),
            'GT_pairing':'verified ground_truth paths; corrected erroneous metadata' if target=='DTD-Synthetic' else 'existing loader'})
    pd.DataFrame(raw).to_csv(out/'RAW_METRICS.csv',index=False)
    jsave(out/'OUTCOME.json',{'outcome':'COMPONENT_TARGET_COMPLETE','n_images':n,'training':False,'metrics':raw})
    jsave(status,{'completed_images':n,'total_images':n,'stage':'complete'})
    print(pd.DataFrame(raw).to_string(index=False),flush=True)


def consolidate():
    original=ROOT/'results/ces_ntr_visualization/corrected_metric_unification/CORRECTED_ABLATION_CATEGORY_MACRO_RAW.csv'
    d=pd.read_csv(original)
    data={}
    for target,source in [('MVTec','Full VisA'),('VisA','Full MVTec'),('BTAD','Full VisA')]:
        subset=d[(d.source==source)&(d.dataset==target)&(d.seed==0)]
        assert set(subset.variant)==set(VARS)
        data[target]={r.variant:float(r.M_FR_P_AP) for r in subset.itertuples()}
    for target in ('MPDD','DTD-Synthetic','DAGM'):
        path=RUN/'formal'/target/'RAW_METRICS.csv'
        if not path.exists():return
        subset=pd.read_csv(path)
        assert set(subset.variant)==set(VARS) and set(subset.seed)=={0}
        data[target]={r.variant:float(r.M_FR_P_AP) for r in subset.itertuples()}
    columns=['MVTec','VisA','BTAD','MPDD','DTD-Synthetic','DAGM']
    entries=[]
    names=['Trained CES','CES + NTR','CES + LRP','CESAD-Full']
    flags=[(1,0,0),(1,1,0),(1,0,1),(1,1,1)]
    for variant,name,flag in zip(VARS,names,flags):
        vals=[data[t][variant] for t in columns]
        entries.append({'CES':flag[0],'NTR':flag[1],'LRP':flag[2],'configuration':name,'seed':0,
            **{('DTD-Syn.' if t=='DTD-Synthetic' else t):v for t,v in zip(columns,vals)},'Mean':float(np.mean(vals))})
    FINAL.mkdir(exist_ok=True,parents=True)
    for name in ['COMPONENT_ABLATION_FINAL.csv','COMPONENT_ABLATION_FINAL.md']:
        path=FINAL/name
        backup=RUN/f'PRE_COMPLETION_{name}'
        if path.exists() and not backup.exists():backup.write_bytes(path.read_bytes())
    pd.DataFrame(entries).to_csv(FINAL/'COMPONENT_ABLATION_FINAL.csv',index=False,float_format='%.6f')
    lines=['**Table X. Component ablation of CESAD across six target datasets (category-macro P-AP, %).**','',
        '| CES | NTR | LRP | MVTec | VisA | BTAD | MPDD | DTD-Syn. | DAGM | Mean |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in entries:
        lines.append('| '+' | '.join(['✓' if row[k] else '—' for k in ('CES','NTR','LRP')]+[f'{row[k]:.2f}' for k in ('MVTec','VisA','BTAD','MPDD','DTD-Syn.','DAGM','Mean')])+' |')
    lines+=['','All cells use matched trained corrected seed0 source weights. Full MVTec source is used only for VisA; Full VisA source for every other target. No training or target-based checkpoint selection. No-NTR is trained CES, never init_v_fallback. Fixed 2x2 LRP uses equal global/local fusion. Only P-AP was recomputed for missing cells; category-macro M_FR uses the existing loader GT, bilinear resize with align_corners=True, and the formal safe_ap_binary helper. Means are computed from six unrounded transfer values. These are single-seed results, without a multiseed SD.','',
        f'Existing three targets: `{original}`. Completed MPDD/DTD/DAGM raw metrics and per-category evidence: `{RUN}/formal/<target>/`. Full VisA no-NTR checkpoint: `{BASE}`; NTR checkpoint: `{NTR}`. Per-target CONFIG.json stores their hashes and image ordering.',
        '', 'The old DAGM main-table 54.823526 is dataset-global AP at 74x74 GT grid; it is not comparable to this table. The capacity K2 value 63.619092 is full-resolution category-macro, but uses the historical NTR direction instead of the corrected rho0.10 checkpoint. Neither value was substituted into this table. Existing MVTec/BTAD main-table differences are documented in the pre-completion table retained with the run.',
        '', 'DTD-Synthetic metadata incorrectly assigned all 947 anomalous mask paths to the input images. This completion uses the existing paired category/ground_truth/defect/stem_mask.png masks with the same formal loader mask transform and evaluator. The original metadata and invalid partial metric file are retained unchanged. Historical DTD pixel results using that metadata are not directly comparable until their GT pairing is corrected. See GT_PAIRING_CORRECTION.json.','']
    (FINAL/'COMPONENT_ABLATION_FINAL.md').write_text('\n'.join(lines),encoding='utf-8')
    jsave(RUN/'OUTCOME.json',{'outcome':'SIX_TARGET_COMPONENT_ABLATION_COMPLETE','seed':0,'training':False,'results':entries})
    print('SIX_TARGET_COMPONENT_ABLATION_COMPLETE',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--targets',nargs='+',default=['MPDD','DTD-Synthetic','DAGM'],choices=['MPDD','DTD-Synthetic','DAGM'])
    args=parser.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for bounded-cost local extraction')
    info=source_scoring_info('Full VisA',torch.device('cuda'))
    specs={v:HeadSpec('CES',{'m':info['m'],'alpha':info['alpha'],'v':load_v(str(path))},'Full VisA',seed=0)
        for v,path in [('Trained_CES_Core',BASE),('Trained_CES_NTR',NTR)]}
    for target in args.targets:target_run(target,args,info,specs)
    if not args.smoke:consolidate()


if __name__=='__main__':main()
