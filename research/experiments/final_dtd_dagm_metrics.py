"""Frozen-checkpoint metric reconciliation. No optimization or target tuning.

Final AP is sklearn average_precision_score, including tied-score grouping.
DTD/DAGM use paired native GT, and all metrics are category-macro.
"""
import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
FINAL=ROOT/'results/final_dtd_dagm_metric_unification'
WORK=ROOT/'results/final_dtd_dagm_metric_unification_work'
SOURCE='Full VisA'
EVALUATOR='sklearn.average_precision_score;standard_zsad_metrics.safe_auc/f1max;lzp.fast_aupro_vectorized(FPRcap0.30,steps200,existing_normalization);M_FR;bilinear_align_corners=True;category_macro'
AP_EVALUATOR='standard_zsad_metrics.safe_ap=sklearn.average_precision_score;ties_grouped;category_macro'
VARIANTS=('Trained_CES_Core','Trained_CES_NTR','Trained_CES_LRP','Trained_CES_NTR_LRP')
MODELS=('CESAD','AnomalyCLIP','FAPrompt','MRAD-TF','MRAD-FT','MRAD-CLIP','VisualAD','FB-CLIP')


def jsave(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    temp.replace(path)


def signature(path):
    p=Path(path)
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def catalogue(target):
    if target=='DAGM':
        from experiments.phase_dagm_visa_eval import DAGMTargetDataset
        ds=DAGMTargetDataset('E:/WorkSpace/dataset/DAGM_KaggleUpload')
        rows=[{'img_path':r['img_path'],'gt_path':r['mask_path'],'category':r['cls_name'],'label':int(r['anomaly'])} for r in ds.items]
    elif target=='DTD-Synthetic':
        from experiments.phase_median_spatial_consensus_generalization import HOLDOUTS
        root=HOLDOUTS[target][0]
        data=json.loads((root/'meta.json').read_text(encoding='utf-8'))['test']
        rows=[]
        for category in sorted(data):
            for r in data[category]:
                gt=''
                if int(r['anomaly']):
                    parts=list(Path(r['img_path']).parts)
                    parts[parts.index('test')]='ground_truth'
                    parts[-1]=Path(parts[-1]).stem+'_mask.png'
                    gt=str(root.joinpath(*parts))
                    assert Path(gt).is_file() and Path(gt).resolve()!=(root/r['img_path']).resolve()
                rows.append({'img_path':str(root/r['img_path']),'gt_path':gt,'category':category,'label':int(r['anomaly'])})
    else:raise ValueError(target)
    return rows


def native_mask(row):
    with Image.open(row['img_path']) as im:size=im.size
    if row['label']:
        assert row['gt_path'] and Path(row['gt_path']).is_file()
        with Image.open(row['gt_path']) as im:
            assert im.size==size,(row['img_path'],im.size,size)
            mask=(np.asarray(im.convert('L'))>0).astype(np.uint8)
    else:mask=np.zeros((size[1],size[0]),dtype=np.uint8)
    return mask


def mask_for(row):
    return row['_mask'] if '_mask' in row else native_mask(row)


def checked_order(rows,manifest,path_column='img_path'):
    norm=lambda p:str(p).replace('\\','/').casefold()
    saved=pd.read_csv(manifest) if isinstance(manifest,Path) else manifest
    assert len(rows)==len(saved)
    paths=saved[path_column].tolist() if isinstance(saved,pd.DataFrame) else [r[path_column] for r in saved]
    assert [norm(r['img_path']) for r in rows]==[norm(x) for x in paths], 'Image ordering mismatch'


class ArrayStore:
    def __init__(self,path,key=None):
        self.path=Path(path)
        if key:
            with np.load(self.path,allow_pickle=True) as z:self.maps=z[key]
        else:self.maps=np.load(self.path,mmap_mode='r')
    def __getitem__(self,index):return np.asarray(self.maps[index],dtype=np.float32)
    def __len__(self):return len(self.maps)


class ChunkStore:
    def __init__(self,path):
        self.path=Path(path);self.cache=OrderedDict()
    def __getitem__(self,index):
        k=index//32
        if k not in self.cache:
            with np.load(self.path/f'chunk_{k:05d}.npz') as z:self.cache[k]=z['maps']
            while len(self.cache)>2:self.cache.popitem(last=False)
        return self.cache[k][index%32].astype(np.float32)


def evaluate(job,target,rows,store,full=False,extra=None):
    from experiments.standard_zsad_metrics import safe_ap,safe_auc,f1max
    from experiments.phase_parameter_free_local_zoom_pyramid_fullres import torch_resize,fast_aupro_vectorized
    out=WORK/job;out.mkdir(parents=True,exist_ok=True)
    path=out/'categories.csv'
    prior=pd.read_csv(path).to_dict('records') if path.exists() else []
    for category in sorted({r['category'] for r in rows}):
        if any(r['category']==category for r in prior):continue
        ids=[i for i,r in enumerate(rows) if r['category']==category]
        masks=[mask_for(rows[i]) for i in ids]
        sizes=[m.size for m in masks]
        # Expanded scores are temporary, not retained copies of predictions.
        scratch=out/'expanded_scores.npy'
        arr=np.lib.format.open_memmap(scratch,mode='w+',dtype=np.float32,shape=(sum(sizes),))
        views=[];offset=0
        for i,m in zip(ids,masks):
            raw=store[i]
            assert raw.ndim==2 and np.isfinite(raw).all()
            count=m.size
            arr[offset:offset+count]=torch_resize(raw,m.shape,align_corners=True).reshape(-1)
            views.append(arr[offset:offset+count].reshape(m.shape));offset+=count
        arr.flush();y=np.concatenate([m.reshape(-1) for m in masks])
        print(f'[metrics] {job} {category}, pixels={len(y)} full={full}',flush=True)
        result={'category':category,'n_images':len(ids),'n_positive_pixels':int(y.sum()),'n_pixels':int(len(y)),
            'P_AP':safe_ap(y,arr),'GT_shapes':';'.join(sorted({str(m.shape) for m in masks}))}
        gc.collect()
        if full:
            result['P_AUROC']=safe_auc(y,arr);gc.collect()
            result['P_F1max']=f1max(y,arr);gc.collect()
            result['AUPRO']=fast_aupro_vectorized(masks,views)
        prior.append(result)
        pd.DataFrame(prior).to_csv(path,index=False)
        del y,masks,views,arr;gc.collect()
        scratch.unlink(missing_ok=True)
    assert len(prior)==len({r['category'] for r in rows})
    metrics=('P_AP','P_AUROC','P_F1max','AUPRO') if full else ('P_AP',)
    result={'source':SOURCE,'target':target,'n_images':len(rows),'n_categories':len(prior),
        'evaluator':EVALUATOR if full else AP_EVALUATOR,'status':'COMPLETE',
        'GT_resolution':'native paired GT' if target in ('DTD-Synthetic','DAGM') else 'existing formal loader GT',
        **(extra or {})}
    for metric in metrics:
        values=np.asarray([x[metric] for x in prior],dtype=float)
        result[metric]=float(np.mean(values[np.isfinite(values)])) if np.isfinite(values).any() else None
        result[f'{metric}_valid_categories']=int(np.isfinite(values).sum())
    jsave(out/'result.json',result)
    return result


def capacity_prepare(smoke=False):
    from experiments.complete_component_ablation_six_targets import target_data,extract
    from experiments.phase_cmab_compact_multi_anomaly_basis import feature_grid,score_from_grid,state_pair_from_model_info
    from experiments.phase_final_cross_dataset_benchmark import load_visa_source
    from experiments.phase_parameter_free_local_zoom_pyramid_fullres import torch_resize
    from experiments.phase_ces_linear_head_control import stitch_local_maps
    from experiments.phase_xlv_cross_domain_transfer import CHECKPOINT
    from experiments.phase_x0a_mandatory_gates import load_components
    from utils.transforms import get_transform
    out=WORK/('capacity_DTD_smoke' if smoke else 'capacity_DTD_predictions');out.mkdir(parents=True,exist_ok=True)
    ds,refs,patch,gp=target_data('DTD-Synthetic')
    cat=catalogue('DTD-Synthetic');checked_order(cat,refs)
    n=min(4,len(cat)) if smoke else len(cat)
    info=load_visa_source('cuda');s_n,s_a=state_pair_from_model_info(info)
    states={2:(s_n,s_a[None])}
    ckpts={2:ROOT/'results/phase_hard_normal_contrast/formal/contrast_vectors.npz'}
    for k,var in [(3,'E1_K2_CMAB'),(5,'E2_K4_CMAB')]:
        path=ROOT/f'results/phase_cmab_compact_multi_anomaly_basis/formal/Full_VisA_{var}_formal.pt'
        p=torch.load(path,map_location='cpu',weights_only=False);assert p['s_a'].shape==(k-1,1024)
        states[k]=(p['s_n'],p['s_a']);ckpts[k]=path
    identity={'n':n,'images_hash':hashlib.sha256(json.dumps(cat[:n],sort_keys=True).encode()).hexdigest(),
        'checkpoints':{k:signature(v) for k,v in ckpts.items()},'LRP':'fixed2x2;0.5global+0.5local;518crop;74map'}
    cp=out/'identity.json'
    if cp.exists():assert json.loads(cp.read_text())==json.loads(json.dumps(identity))
    else:jsave(cp,identity)
    maps={}
    for k in states:
        path=out/f'K{k}.npy'
        maps[k]=np.load(path,mmap_mode='r+') if path.exists() else np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(n,74,74))
    progress=out/'progress.json'
    done=json.loads(progress.read_text())['done'] if progress.exists() else 0
    if done<n:
        preprocess,_=get_transform(argparse.Namespace(image_size=518))
        model=load_components(str(CHECKPOINT),'cuda')[0].eval()
        for p in model.parameters():p.requires_grad_(False)
        start=time.time();initial=done
        with torch.inference_mode():
            for i in range(done,n):
                raw=extract(cat[i]['img_path'],model,preprocess)
                zg=feature_grid(info,patch[i],'cuda');zl=[feature_grid(info,z,'cuda') for z in raw]
                for k,state in states.items():
                    global37=score_from_grid(zg,state,'cuda')
                    local=stitch_local_maps(np.stack([score_from_grid(z,state,'cuda') for z in zl]))
                    maps[k][i]=.5*torch_resize(global37,(74,74),True)+.5*local
                    assert np.isfinite(maps[k][i]).all()
                if (i+1)%25==0 or i+1==n:
                    for a in maps.values():a.flush()
                    elapsed=time.time()-start
                    jsave(progress,{'done':i+1,'total':n,'eta_seconds':elapsed/max(1,i+1-initial)*(n-i-1)})
                    print(f'[capacity DTD] {i+1}/{n}',flush=True)
    print('CAPACITY_DTD_SMOKE_OK' if smoke else 'CAPACITY_DTD_PREDICTIONS_COMPLETE',flush=True)


def component_prepare(target,smoke=False):
    if target in ('MPDD','DTD-Synthetic','DAGM'):return
    from experiments.phase_ces_linear_head_control import get_target_cell,HeadSpec
    from experiments.unify_corrected_ntr_category_macro import source_scoring_info,load_v,score_patch_grid_multi,lrp_maps_multi
    from experiments.phase_parameter_free_local_zoom_pyramid_fullres import item_at,item_meta,mask_np
    source='Full MVTec' if target=='VisA' else 'Full VisA'
    info=source_scoring_info(source,torch.device('cuda'))
    _,_,dataset,refs,patch,kind=get_target_cell(source,target,torch.device('cuda'),False)
    root=ROOT/'results/ces_ntr_visualization/corrected_metric_unification/CORRECTED_ABLATION_CATEGORY_MACRO_RAW.csv'
    raw=pd.read_csv(root);meta=raw[(raw.source==source)&(raw.dataset==target)&(raw.seed==0)]
    specs={}
    for variant in VARIANTS[:2]:
        row=meta[meta.variant==variant].iloc[0]
        specs[variant]=HeadSpec('CES',{'m':info['m'],'alpha':info['alpha'],'v':load_v(row.checkpoint)},source,seed=0)
    lp=Path(f'E:/MRAD_final_evidence_cache/local_patch_{source.replace(" ","_")}__{target}.npy')
    assert lp.exists(),lp
    local=np.load(lp,mmap_mode='r');n=min(4,len(refs)) if smoke else len(refs)
    assert len(local)==len(refs)
    out=WORK/f'component_{target}_predictions';out.mkdir(exist_ok=True,parents=True)
    maps={}
    for v in VARIANTS:
        side=74 if v.endswith('_LRP') else 37;path=out/f'{v}.npy'
        maps[v]=np.load(path,mmap_mode='r+') if path.exists() else np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(refs),side,side))
    pp=out/'progress.json';done=json.loads(pp.read_text())['done'] if pp.exists() else 0
    manifest=[]
    with torch.inference_mode():
        for i,ref in enumerate(refs):
            if i>=n:break
            if i<done:continue
            gs=score_patch_grid_multi(info,specs,patch[i],'cuda')
            ls=lrp_maps_multi(info,specs,patch[i],np.asarray(local[i],dtype=np.float32),'cuda')
            for v in specs:
                maps[v][i]=gs[v]
                maps['Trained_CES_LRP' if v==VARIANTS[0] else VARIANTS[3]][i]=ls[v]
            if (i+1)%100==0 or i+1==n:
                for a in maps.values():a.flush()
                jsave(pp,{'done':i+1,'total':len(refs)});print(f'[component {target}] {i+1}/{len(refs)}',flush=True)
    jsave(out/'checkpoints.json',{v:str(meta[meta.variant==v].iloc[0].checkpoint) for v in VARIANTS})


def component_catalogue(target):
    if target in ('DTD-Synthetic','DAGM'):return catalogue(target)
    from experiments.phase_mva_mrad_same_protocol_visa_factorization import load_metas_direct
    source='Full MVTec' if target=='VisA' else 'Full VisA'
    meta=load_metas_direct(source,target,'cpu',False)
    return [{'img_path':m['img_path'],'category':m['category'],'label':m['label'],'_mask':m['mask']} for m in meta]


def baseline_cache(target,method):
    if target=='DTD-Synthetic' and method=='FAPrompt':
        path=ROOT/'results/paper_assets/baseline_close/map_cache/FAPrompt_Full_VisA__DTD-Synthetic.npz'
        manifest=path.with_suffix('.json')
        if path.exists() and manifest.exists():
            checked_order(catalogue(target),json.loads(manifest.read_text()))
            return ArrayStore(path,'maps'),str(path)
    if target=='DTD-Synthetic' and method.startswith('MRAD-'):
        var={'MRAD-TF':'M1_MRAD_TF','MRAD-FT':'M2_MRAD_FT','MRAD-CLIP':'M3_MRAD_CLIP'}[method]
        path=ROOT/f'results/paper_assets/mrad_dtd_close/MAP_CACHE/Full_VisA__DTD-Synthetic__{var}.npz'
        if path.exists():
            with np.load(path,allow_pickle=True) as p:
                rr=list(p['retrieval_rows']);rows=catalogue(target)
                assert len(rr)==len(rows)
                assert all(int(x['image_index'])==i and x['category']==rows[i]['category'] for i,x in enumerate(rr))
            return ArrayStore(path,'maps'),str(path)
    return None,None


def purge(prefixes):
    for name in list(sys.modules):
        if any(name==p or name.startswith(p+'.') for p in prefixes):del sys.modules[name]


def predictor(method):
    from scipy.ndimage import gaussian_filter
    if method=='AnomalyCLIP':
        purge(('AnomalyCLIP_lib','prompt_ensemble','utils'))
        import experiments.phase_anomalyclip_same_protocol_audit as ac
        model,prompt,text,args=ac.load_model(SOURCE,'cuda')
        preprocess=ac.ac_get_transform(args)[0]
        return preprocess,lambda images:ac.predict_batch(model,text,args,images),str(ac.WEIGHTS[SOURCE])
    if method=='FB-CLIP':
        purge(('prompt_ensemble',))
        import experiments.phase_fbclip_pretrained_headroom_audit as fb
        model,prompt,args,_=fb.load_fb_model(SOURCE,'cuda')
        preprocess=fb.fb_get_transform(args)[0]
        return preprocess,lambda images:fb.fb_predict_batch(model,prompt,args,images),str(fb.WEIGHTS[SOURCE])
    if method=='VisualAD':
        import experiments.phase_visualad_exact_protocol_audit as va
        official=va.ensure_official_imports()
        model,layers,attention,_=va.load_visualad_bundle(SOURCE,'E1c_FullVisualAD',official,torch.device('cuda'))
        def predict(images):
            maps,_=va.infer_batch(model,layers,attention,images)
            return [gaussian_filter(x,sigma=va.SIGMA).astype(np.float32) for x in maps]
        return va.official_transform(official),predict,str(va.checkpoint_path(SOURCE))
    if method=='FAPrompt':
        purge(('AnomalyCLIP_lib','prompt_ensemble','utils','FAPrompt'))
        import experiments.phase_paper_baseline_close as fa
        ckpt=fa.FAPROMPT_CHECKPOINTS[SOURCE]
        model,prompt=fa.load_faprompt('cuda',ckpt)
        preprocess=fa.fa_get_transform(argparse.Namespace(image_size=518))[0]
        return preprocess,lambda images:[fa.faprompt_map(model,prompt,x[None],'cuda')[0] for x in images],str(ckpt)
    raise ValueError(method)


def checkpoint_name(method):
    if method=='CESAD':return str(ROOT/'results/phase_hard_normal_contrast/formal/contrast_vectors.npz')+'::E1'
    if method=='AnomalyCLIP':return str(ROOT/'_external/AnomalyCLIP_official/checkpoints/9_12_4_multiscale/epoch_15.pth')
    if method=='FAPrompt':return str(ROOT/'results/paper_assets/baseline_close/official_checkpoints/train_on_visa.pth')
    if method=='VisualAD':return str(ROOT/'_external/VisualAD/weight/train_on_visa/CLIP.pth')
    if method=='FB-CLIP':return str(ROOT/'_external/FB-CLIP/official_weights/train_on_visa/visa_epoch_2_model.pth')
    if method.startswith('MRAD-'):return ('none (TF)' if method=='MRAD-TF' else str(ROOT/'checkpoints/test_on_mvtec.pth'))+';cache_model_visa.pt;cache_patch_model_visa.pt'
    return 'UNKNOWN'


def baseline_predict(target,method,smoke=False):
    cached,_=baseline_cache(target,method)
    if cached is not None:
        print('REUSABLE_NATIVE_MAP_CACHE',method,target,flush=True)
        return
    rows=catalogue(target);n=min(2,len(rows)) if smoke else len(rows)
    out=WORK/f'baseline_{method}_{target}';out.mkdir(parents=True,exist_ok=True)
    # This run-owned directory is disposable scratch; retained maps are only
    # existing artifacts or compact CES predictions. Never remove old caches.
    chunks=out/('_smoke_predictions' if smoke else '_temporary_predictions');chunks.mkdir(exist_ok=True)
    if method.startswith('MRAD-'):
        from experiments.phase_mva_mrad_same_protocol_visa_factorization import load_mrad_bundle,mrad_infer_one
        var={'MRAD-TF':'M1_MRAD_TF','MRAD-FT':'M2_MRAD_FT','MRAD-CLIP':'M3_MRAD_CLIP'}[method]
        bundle=load_mrad_bundle(SOURCE,var,'cuda')
        def predict_paths(paths):return [mrad_infer_one(bundle,p,'cuda')[0] for p in paths]
        ckpt=checkpoint_name(method)
    else:
        preprocess,predict,ckpt=predictor(method)
        def predict_paths(paths):
            images=torch.stack([preprocess(Image.open(p).convert('RGB')) for p in paths]).cuda()
            return predict(images)
    identity={'source':SOURCE,'target':target,'method':method,'checkpoint':ckpt,'n':n,
        'order_hash':hashlib.sha256(json.dumps(rows[:n],sort_keys=True).encode()).hexdigest(),'precision':'float32 compressed; no new smoothing/normalization'}
    ipath=out/('smoke_identity.json' if smoke else 'identity.json')
    if ipath.exists():assert json.loads(ipath.read_text())==identity,'Checkpoint/image identity changed before resume'
    else:jsave(ipath,identity)
    start=time.time();processed=0
    with torch.inference_mode():
        for c,startpos in enumerate(range(0,n,32)):
            path=chunks/f'chunk_{c:05d}.npz'
            if path.exists():continue
            end=min(n,startpos+32);maps=[]
            batch=1 if method.startswith('MRAD-') or method=='FAPrompt' else 2
            for i in range(startpos,end,batch):
                preds=predict_paths([x['img_path'] for x in rows[i:min(end,i+batch)]])
                for p in preds:
                    a=np.squeeze(np.asarray(p,dtype=np.float32))
                    assert a.shape==(518,518) and np.isfinite(a).all(),a.shape
                    maps.append(a)
            np.savez_compressed(path,maps=np.stack(maps))
            processed+=len(maps);elapsed=time.time()-start
            jsave(out/'progress.json',{'done':end,'total':n,'elapsed_seconds':elapsed,'eta_seconds':elapsed/max(1,processed)*(n-end)})
            print(f'[baseline {method} {target}] {end}/{n}',flush=True)
    jsave(out/('smoke_pass.json' if smoke else 'predictions_complete.json'),identity)


def metrics_worker(group,target,value):
    if group=='capacity':
        k=int(value)
        path=WORK/f'capacity_DTD_predictions/K{k}.npy' if target=='DTD-Synthetic' else ROOT/f'results/paper_fig5_dagm_capacity/formal/K{k}_LRP_maps.npy'
        store=ArrayStore(path);rows=catalogue(target)
        if target=='DAGM':checked_order(rows,ROOT/'results/paper_fig5_dagm_capacity/formal/IMAGE_MANIFEST.csv',path_column='path')
        kpath=checkpoint_name('CESAD') if k==2 else str(ROOT/f'results/phase_cmab_compact_multi_anomaly_basis/formal/Full_VisA_{"E1_K2_CMAB" if k==3 else "E2_K4_CMAB"}_formal.pt')
        evaluate(f'capacity_{target}_K{k}',target,rows,store,extra={'K':k,'checkpoint':kpath,'prediction_cache':str(path)})
    elif group=='component':
        variant=value;rows=component_catalogue(target)
        if target in ('MPDD','DTD-Synthetic','DAGM'):
            base=ROOT/f'results/component_ablation_six_target_completion/formal/{target}'
            path=base/f'{variant}_maps.npy';checked_order(rows,base/'IMAGE_MANIFEST.csv')
            cp=json.loads((base/'CONFIG.json').read_text())['NTR' if 'NTR' in variant else 'no_NTR']['path']
        else:
            base=WORK/f'component_{target}_predictions';path=base/f'{variant}.npy'
            cp=json.loads((base/'checkpoints.json').read_text())[variant]
        result=evaluate(f'component_{target}_{variant}',target,rows,ArrayStore(path),extra={'variant':variant,'seed':0,'checkpoint':cp,'prediction_cache':str(path)})
        if target=='VisA':
            result['source']='Full MVTec';jsave(WORK/f'component_{target}_{variant}/result.json',result)
    elif group=='sota':
        method=value;rows=catalogue(target)
        if method=='CESAD':
            path=WORK/'capacity_DTD_predictions/K2.npy' if target=='DTD-Synthetic' else ROOT/'results/paper_fig5_dagm_capacity/formal/K2_LRP_maps.npy'
            store=ArrayStore(path);cache=str(path)
        else:
            store,cache=baseline_cache(target,method)
            if store is None:
                path=WORK/f'baseline_{method}_{target}'
                assert (path/'predictions_complete.json').exists()
                store=ChunkStore(path/'_temporary_predictions');cache=str(store.path)
        result=evaluate(f'sota_{method}_{target}',target,rows,store,True,{'method':method,'checkpoint':checkpoint_name(method),'prediction_cache':cache})
        if isinstance(store,ChunkStore):
            # Scratch predictions are new files owned solely by this stage.
            # Retain their identities and per-category metrics before cleanup.
            parent=store.path.resolve();expected=(WORK/f'baseline_{method}_{target}/_temporary_predictions').resolve()
            assert parent==expected and parent.is_relative_to(WORK.resolve())
            store.cache.clear()
            hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in parent.glob('chunk_*.npz')}
            jsave(parent.parent/'prediction_chunk_hashes.json',hashes)
            for p in parent.glob('chunk_*.npz'):p.unlink()
            parent.rmdir()
            result['prediction_cache']='temporary FP32 chunks evaluated then removed; chunk hashes retained'
            jsave(WORK/f'sota_{method}_{target}/result.json',result)


def result_for(job):
    p=WORK/job/'result.json'
    if p.exists():return json.loads(p.read_text())
    error=WORK/job/'FAILED.json'
    return {'status':'N/A','reason':json.loads(error.read_text()).get('error','failed')} if error.exists() else {'status':'PENDING'}


def write_tables():
    FINAL.mkdir(parents=True,exist_ok=True)
    capacity=[]
    old=pd.read_csv(ROOT/'results/paper_protocol_reconciliation/FIG5_CAPACITY_PROVENANCE.csv')
    for row in old.itertuples():
        capacity.append({'source':row.source,'target':row.target,'K':row.K,'P_AP':row.P_AP,'checkpoint':row.checkpoint,
            'evaluator':AP_EVALUATOR,'GT_resolution':'unchanged existing main-table loader GT',
            'status':'REUSED_VERIFIED_SKLEARN','raw_source':row.raw_result,'seed_policy':row.actual_rng_seed})
    for target in ('DTD-Synthetic','DAGM'):
        for k in (2,3,5):
            r=result_for(f'capacity_{target}_K{k}')
            capacity.append({'source':SOURCE,'target':target,'K':k,'P_AP':r.get('P_AP'),
                'checkpoint':r.get('checkpoint',checkpoint_name('CESAD') if k==2 else str(ROOT/f'results/phase_cmab_compact_multi_anomaly_basis/formal/Full_VisA_{"E1_K2_CMAB" if k==3 else "E2_K4_CMAB"}_formal.pt')),
                'evaluator':AP_EVALUATOR,'GT_resolution':'native paired GT','status':r['status'],'reason':r.get('reason',''),
                'seed_policy':'historical nominal0; K3 actual2; K5 actual4'})
    for k in (2,3,5):
        vals=[r.get('P_AP') for r in capacity if r['K']==k]
        capacity.append({'source':'MVTec+VisA','target':'Mean','K':k,'P_AP':float(np.mean(vals)) if len(vals)==6 and all(v is not None and np.isfinite(v) for v in vals) else None,
            'checkpoint':'same per-source capacity-family checkpoints; see six target rows','evaluator':AP_EVALUATOR,'status':'COMPLETE' if len(vals)==6 and all(v is not None for v in vals) else 'N/A'})
    pd.DataFrame(capacity).to_csv(FINAL/'FINAL_CAPACITY_6_DATASETS.csv',index=False,float_format='%.6f',na_rep='N/A')
    components=[]
    for v in VARIANTS:
        for target in ('MVTec','VisA','BTAD','MPDD','DTD-Synthetic','DAGM'):
            r=result_for(f'component_{target}_{v}')
            components.append({'source':'Full MVTec' if target=='VisA' else SOURCE,'target':target,'configuration':v,'seed':0,
                'P_AP':r.get('P_AP'),'checkpoint':r.get('checkpoint','PENDING'),
                'evaluator':AP_EVALUATOR,'GT_resolution':r.get('GT_resolution','native paired GT' if target in ('DTD-Synthetic','DAGM') else 'unchanged formal loader GT'),
                'status':r['status'],'reason':r.get('reason','')})
        vals=[r.get('P_AP') for r in components if r['configuration']==v]
        components.append({'source':'MVTec+VisA','target':'Mean','configuration':v,'seed':0,
            'P_AP':float(np.mean(vals)) if len(vals)==6 and all(x is not None for x in vals) else None,
            'checkpoint':'matched trained corrected per-source seed0 pair; see six rows','evaluator':AP_EVALUATOR,
            'status':'COMPLETE' if len(vals)==6 and all(x is not None for x in vals) else 'N/A'})
    pd.DataFrame(components).to_csv(FINAL/'FINAL_COMPONENT_ABLATION_6_DATASETS.csv',index=False,float_format='%.6f',na_rep='N/A')
    sota=[]
    for target in ('DTD-Synthetic','DAGM'):
        for model in MODELS:
            r=result_for(f'sota_{model}_{target}')
            sota.append({'source':SOURCE,'target':target,'method':model,'checkpoint':r.get('checkpoint',checkpoint_name(model)),
                'P_AUROC':r.get('P_AUROC'),'P_F1max':r.get('P_F1max'),'P_AP':r.get('P_AP'),'AUPRO':r.get('AUPRO'),
                'evaluator':EVALUATOR,'GT_resolution':'native paired GT','status':r['status'],'reason':r.get('reason',''),
                'note':'CESAD official historical main checkpoint; not substituted corrected component checkpoint' if model=='CESAD' else 'official source-specific wrapper; official smoothing retained; no extra normalization'})
    pd.DataFrame(sota).to_csv(FINAL/'FINAL_SOTA_DTD_DAGM.csv',index=False,float_format='%.6f',na_rep='N/A')


def run_child(stage,target=None,value=None,smoke=False,job=None):
    args=[sys.executable,'-u',str(Path(__file__).resolve()),'--stage',stage]
    if target:args+=['--target',target]
    if value:args+=['--value',str(value)]
    if smoke:args+=['--smoke']
    log=WORK/((job or stage)+('_smoke' if smoke else '')+'.log')
    print('RUN',args,flush=True)
    with log.open('a',encoding='utf-8') as f:
        p=subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:
        error={'error':f'{stage} failed exit={p.returncode}; see {log}','log':str(log)}
        jsave(WORK/(job or stage)/'FAILED.json',error)
        return False
    return True


def coordinator():
    WORK.mkdir(parents=True,exist_ok=True);write_tables()
    tasks=[('capacity-prepare',None,None,'prepare_capacity_DTD')]
    tasks += [('capacity',target,str(k),f'capacity_{target}_K{k}') for target in ('DTD-Synthetic','DAGM') for k in (2,3,5)]
    tasks += [('component-prepare',target,None,f'prepare_component_{target}') for target in ('MVTec','VisA','BTAD')]
    tasks += [('component',target,v,f'component_{target}_{v}') for target in ('DTD-Synthetic','DAGM','MVTec','VisA','BTAD','MPDD') for v in VARIANTS]
    for stage,target,value,job in tasks:
        if (WORK/job/'result.json').exists():continue
        run_child(stage,target,value,job=job);write_tables()
    # Evaluate existing native-map baseline caches first; re-infer other released
    # checkpoints only when cached model output cannot be recovered faithfully.
    for target in ('DTD-Synthetic','DAGM'):
        for method in MODELS:
            job=f'sota_{method}_{target}'
            if (WORK/job/'result.json').exists():continue
            if method!='CESAD':
                ok=run_child('baseline-predict',target,method,job=job+'_predict')
                if not ok:
                    jsave(WORK/job/'FAILED.json',{'error':'checkpoint inference/cache unavailable; see prediction log'})
                    write_tables();continue
            run_child('sota',target,method,job=job);write_tables()
    write_tables();jsave(WORK/'FINISHED.json',{'finished':True,'training':False,'output':str(FINAL)})
    print('FINAL_TABLES_WRITTEN',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',default='run')
    p.add_argument('--target');p.add_argument('--value');p.add_argument('--smoke',action='store_true')
    args=p.parse_args();WORK.mkdir(exist_ok=True,parents=True);torch.set_num_threads(4)
    if args.stage=='run':coordinator()
    elif args.stage=='tables':write_tables()
    elif args.stage=='capacity-prepare':capacity_prepare(args.smoke)
    elif args.stage=='component-prepare':component_prepare(args.target,args.smoke)
    elif args.stage=='baseline-predict':baseline_predict(args.target,args.value,args.smoke)
    elif args.stage in ('capacity','component','sota'):metrics_worker(args.stage,args.target,args.value)
    else:raise ValueError(args.stage)


if __name__=='__main__':main()
