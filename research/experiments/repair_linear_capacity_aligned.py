"""Complete linear/capacity controls using the corrected component source pools.

The negative MVTec recipe probe does not replace the original source path.
Reuse verified MVTec controls; repair affected VisA controls only. All capacity
models have TOTAL 30 head updates, with the existing multi-state definition.
"""
from pathlib import Path
import gc
import math
import os
import sys
import traceback
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import phase_ces_unified_ablation_final as u
from experiments import phase_ces_lhc_paired_repeat as p
from experiments import final_unified_evaluation as official

META = ROOT/'results/linear_capacity_original_cache_aligned/formal'
WORK = Path('E:/MRAD_linear_capacity_original_cache_aligned/20261027')
COMP = ROOT/'results/component_ablation_original_cache_repair/formal'
POOLS = Path('E:/MRAD_component_original_cache_repair/20261027/source_pools')
OLD_HEAD = ROOT/'results/phase_ces_lhc_paired_repeat/formal'
OLD_K = ROOT/'results/phase_ces_capacity_matched30/formal'
TARGETS = ('VisA','MVTec','BTAD','MPDD','DTD-Synthetic','DAGM')
CONTROLS = ('LinearHead','NormalizedLinearHead','K3_30','K5_30')
SEEDS = (0,1,2)


def status(phase,**kw):p.status(META,phase,**kw)


def pool(source,seed):
    folder=POOLS/f'{source.replace(" ","_")}_seed{seed}'
    meta=official.readjson(folder/'COMPLETE.json')
    proof=official.readjson(folder/'E_CACHE_RECONSTRUCTION_CHECK.json')
    assert proof['status']=='PASS' and proof['max_abs_error']<2e-6
    z=np.load(folder/'features.npy',mmap_mode='r');y=np.load(folder/'labels.npy')
    batches=np.load(folder/'batch_indices.npy');probe=np.load(folder/'probe_indices.npy')
    for key,arr in [('feature_array_hash',z),('labels_hash',y),('batch_stream_hash',batches),('probe_indices_hash',probe)]:
        assert p.digest(arr)==meta[key]
    assert p.file_hash(folder/'PATCH_MANIFEST.csv')==meta['manifest_hash']
    assert len(batches)==30 and meta['epochs']==10
    return z,y,batches,probe,meta


def setup():
    META.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available()
    u.OUT=WORK;u.status=status
    decision=official.readjson(ROOT/'results/phase_mvtec_exact_visa_recipe_checked/formal/STATUS.json')
    assert decision['phase']=='COMPLETE' and decision['delta_P_AP']<0
    config=dict(source_path='retain original MVTec L2; original frozen VisA PC1/E2 transform',
        reason='complete VisA recipe probe negative, no adoption; not source-only hyperparameter validation',
        exploratory_recipe_probe=decision, seeds=list(SEEDS),targets=list(TARGETS),
        source_pools=str(POOLS),epochs=10,total_updates_all_K=30,
        optimizer=dict(name='AdamW',lr=.001,weight_decay=0.,betas=[.9,.999],eps=1e-8),
        rho=.1,LRP_global_weight=.5,normalized_linear='source-fixed c0 times unit direction, no bias',
        capacity='canonical K2; K3/K5 free states from PRETRAINING geometry with existing perturbation, raw LSE and diversity1',
        evaluator_id=official.evaluator_hash(),checkpoint_selection='last update30, no target checkpoint selection',
        component_raw=dict(path=str(COMP/'CORRECTED_COMPONENT_RAW.csv'),sha256=p.file_hash(COMP/'CORRECTED_COMPONENT_RAW.csv')),
        code=[dict(path=str(q),sha256=p.file_hash(q)) for q in (Path(__file__),Path(p.__file__),Path(u.__file__),Path(u.cmab.__file__))])
    u.lock(META/'CONFIG.json',config)
    (META/'SOURCE_PATH_DECISION.md').write_text('# Frozen source path\n\n'
        'The seed-0 MVTec-to-VisA complete-recipe probe yielded 20.711684 P-AP versus '
        '21.367497 for its matched nearest-mask unconditioned control. The added recipe is not adopted. '
        'Retain the ORIGINAL formal MVTec bilinear-label/L2 source path, not the probe control checkpoint. '
        'The probe is exploratory target-informed evidence, not source-only calibration. '
        'All subsequent controls share corrected component pools, actual batches and K2 anchors. '
        'Do not combine higher cells from different checkpoint families.\n',encoding='utf-8')
    return config


def train_multi(source,seed,variant,z,y,batches,probe,meta,config):
    K=3 if variant=='K3_30' else 5
    path=WORK/'checkpoints'/f'{source.replace(" ","_")}__seed{seed}__{variant}.pt'
    if path.exists():
        saved=torch.load(path,map_location='cpu',weights_only=False)
        assert saved['config_hash']==p.json_hash(config)
        return saved,path
    init=p.init_from_meta(meta);v=p.hist.unit_np(init['init_v'])
    n0=np.asarray(init['m'],np.float32)-init['alpha']*v
    a0=np.asarray(init['m'],np.float32)+init['alpha']*v
    torch.manual_seed(seed)
    model=u.MultiState(n0,a0,K-1,seed).to(u.DEVICE)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    # K=1 anomaly logit must agree with the canonical initial CES scorer.
    with torch.no_grad():
        x=torch.tensor(np.asarray(z[probe]),device=u.DEVICE)
        q=p.new_module('CES',init,u.DEVICE)(x)
        single=u.cmab.multi_state_logits(x,torch.tensor(n0,device=u.DEVICE),torch.tensor(a0[None],device=u.DEVICE))
        assert float(abs(q-single).max())<1e-6
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.,betas=(.9,.999),eps=1e-8)
    logs=[]
    for step,ids in enumerate(batches):
        x=torch.tensor(np.asarray(z[ids]),device=u.DEVICE);yy=torch.tensor(y[ids],device=u.DEVICE)
        q=model(x);base,bce,dice=p.balanced_loss(q,yy)
        tail=p.hist.ntr_loss_from_logits(q,yy,.1);div=u.cmab.diversity_loss(model.s_a)
        loss=base+tail+div;assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward()
        grad=float(torch.sqrt(sum((v.grad.detach()**2).sum() for v in model.parameters())))
        assert math.isfinite(grad)
        opt.step();assert all(torch.isfinite(v).all() for v in model.parameters())
        assert {int(v['step']) for v in opt.state.values()}=={step+1}
        logs.append(dict(source=source,seed=seed,K=K,step=step+1,epoch=step//3+1,
            batch_hash=p.digest(ids),actual_batch=len(ids),loss=float(loss.detach()),
            bce=float(bce.detach()),dice=float(dice.detach()),ntr=float(tail.detach()),
            diversity=float(div.detach()),grad_norm=grad,optimizer_step=step+1))
    saved=dict(source=source,seed=seed,K=K,total_updates=30,realized_updates=30,
        initial_state=initial,initial_geometry=init,initialization_stage='PRE_FINAL_TRAINING geometry, no trained K2 initialization',
        s_n=model.s_n.detach().cpu().numpy(),s_a=model.s_a.detach().cpu().numpy(),
        optimizer_state=opt.state_dict(),logs=logs,config_hash=p.json_hash(config),
        batch_stream_hash=meta['batch_stream_hash'],feature_array_hash=meta['feature_array_hash'],
        labels_hash=meta['labels_hash'],checkpoint_selection='LAST_UPDATE30')
    path.parent.mkdir(parents=True,exist_ok=True);torch.save(saved,path)
    p.write_csv(WORK/'training_logs'/f'{path.stem}.csv',logs)
    return saved,path


def prepare(config):
    seal=official.readjson(COMP/'SOURCE_TRAINING_COMPLETE.json')
    bases=[e for e in seal['models'] if e['variant']=='CES_NTR']
    assert len(bases)==6
    old_multi=official.readjson(OLD_K/'SOURCE_TRAINING_COMPLETE.json')['models']
    models=list(bases);proofs=[]
    for source in ('Full MVTec','Full VisA'):
        for seed in SEEDS:
            z,y,batches,probe,meta=pool(source,seed)
            base=next(e for e in bases if e['source']==source and e['seed']==seed)
            assert base['feature_hash']==meta['feature_array_hash'] and base['batch_hash']==meta['batch_stream_hash']
            assert p.file_hash(base['checkpoint'])==base['checkpoint_sha256']
            p.equivalence(z,meta,u.DEVICE)
            for variant in CONTROLS:
                status('CHECK_OR_TRAIN',source=source,seed=seed,variant=variant)
                if source=='Full MVTec':
                    if variant.startswith('K'):
                        old=next(e for e in old_multi if e['source']==source and e['seed']==seed and e['variant']==variant)
                        cp=Path(old['checkpoint'])
                    else:cp=OLD_HEAD/'checkpoints'/f'Full_MVTec_{variant}_seed{seed}_primary_lr0.001.pt'
                    saved=torch.load(cp,map_location='cpu',weights_only=False)
                    origin='VERIFIED_REUSE_MVTEC'
                elif variant.startswith('K'):
                    saved,cp=train_multi(source,seed,variant,z,y,batches,probe,meta,config)
                    origin='RETRAINED_CORRECTED_VISA_POOL'
                else:
                    saved,cp=p.train_one(WORK,variant,source,seed,z,y,batches,probe,meta,.001,
                        'corrected_source_pool',p.json_hash(config),u.DEVICE)
                    origin='RETRAINED_CORRECTED_VISA_POOL'
                assert saved['realized_updates']==30
                assert saved['feature_array_hash']==meta['feature_array_hash']
                assert saved['batch_stream_hash']==meta['batch_stream_hash']
                key='batch_hash' if variant.startswith('K') else 'batch_ids_hash'
                assert [r[key] for r in saved['logs']]==[p.digest(ids) for ids in batches]
                for key in ('m','alpha','init_v'):
                    np.testing.assert_array_equal(saved['initial_geometry'][key],p.init_from_meta(meta)[key])
                entry=dict(source=source,seed=seed,variant=variant,ntr=True,kind='multi' if variant.startswith('K') else 'scalar',
                    model_id=f'{source.replace(" ","_")}__seed{seed}__{variant}',checkpoint=str(cp),
                    checkpoint_sha256=p.file_hash(cp),feature_hash=meta['feature_array_hash'],
                    source_pool_hash=meta['manifest_hash'],batch_hash=meta['batch_stream_hash'],total_updates=30,origin=origin)
                models.append(entry);proofs.append(entry)
    assert len(models)==30
    p.write_csv(META/'MODEL_REUSE_AND_TRAINING.csv',proofs)
    u.lock(META/'SOURCE_TRAINING_COMPLETE.json',dict(models=models,config_hash=p.json_hash(config),new_heads=12,reused_heads=18))
    return models


def reuse_mvtec(models,config):
    linear=official.readcsv(ROOT/'results/phase_ces_unified_ablation_final/formal/evaluations/VisA_CATEGORIES.csv')
    capacity=official.readcsv(OLD_K/'evaluations/VisA_CATEGORIES.csv')
    rows=[]
    for e in models:
        if e['source']!='Full MVTec' or e['variant']=='CES_NTR':continue
        multi=e['variant'].startswith('K');K=3 if e['variant']=='K3_30' else 5
        candidates=[r for r in (capacity if multi else linear) if int(r['seed'])==e['seed'] and
            (int(r['K'])==K if multi else r['variant']==e['variant'] and r['lrp']=='True')]
        assert len(candidates)==12
        for r in candidates:
            assert r['checkpoint_id']==e['checkpoint_sha256'] and r['evaluator_id']==config['evaluator_id']
            rows.append(dict(source=e['source'],target='VisA',category=r['category'],seed=e['seed'],
                variant=e['variant'],p_ap=float(r['p_ap']),checkpoint=e['checkpoint'],
                checkpoint_id=e['checkpoint_sha256'],evaluator_id=config['evaluator_id'],
                provenance='REUSED_IDENTICAL_CHECKPOINT_NATIVE_GT_RESULT'))
    p.write_csv(META/'VisA_REUSED_CATEGORIES.csv',rows)


def smoke(config,plans):
    target='MVTec';rows,a,local,mapping,_=u.target_features(target)
    assert local is not None
    scorer=u.Scorer(plans[target]['outputs'],'Full VisA');checks=[]
    for label in (0,1):
        i=next(i for i,r in enumerate(rows) if int(r['label'])==label);ix=mapping[i]
        raw=np.concatenate((np.asarray(a[ix],np.float32)[None],np.asarray(local[ix],np.float32)))
        got=scorer(raw)
        for e in plans[target]['outputs']:
            zz=p.transformed(raw,scorer.transform,u.DEVICE)
            cp=torch.load(e['checkpoint'],map_location='cpu',weights_only=False)
            with torch.no_grad():
                if e['variant'].startswith('K'):
                    q=u.cmab.multi_state_logits(zz,torch.tensor(cp['s_n'],device=u.DEVICE),torch.tensor(cp['s_a'],device=u.DEVICE))
                else:
                    spec=p.packed_spec(cp)
                    q=zz@torch.tensor(spec.weight(),device=u.DEVICE)+spec.bias()
            q=q.view(5,37,37).cpu().numpy()
            ref=.5*p.lzp.torch_resize(q[0],(74,74),True)+.5*p.hist.stitch_local_maps(q[1:])
            error=float(abs(ref-got[e['output_id']]).max());assert error<2e-6
            checks.append(dict(sample=rows[i]['sample_id'],model=e['model_id'],max_error=error,status='PASS'))
    p.write_csv(META/'SMOKE_CHECKS.csv',checks)
    p.write_json(META/'SMOKE_COMPLETE.json',dict(status='PASS',actual_samples=2,model_sample_checks=len(checks),
        scope='new source-corrected heads, real cached features, independent scorer parity; frontend inherited verified component caches'))


def summarize(config,models):
    raw=[];component=official.readcsv(COMP/'CORRECTED_COMPONENT_RAW.csv')
    for target in TARGETS:
        rr=official.readcsv(META/'VisA_REUSED_CATEGORIES.csv') if target=='VisA' else official.readcsv(WORK/'evaluations'/f'{target}_CATEGORIES.csv')
        for e in models:
            if e['source']!=u.source_for(target):continue
            if e['variant']=='CES_NTR':
                matches=[r for r in component if r['target']==target and int(r['seed'])==e['seed'] and r['ntr']=='True' and r['lrp']=='True']
                assert len(matches)==1
                r=matches[0];assert p.file_hash(r['checkpoint'])==e['checkpoint_sha256']
                value=float(r['P_AP'])
            else:
                matches=[r for r in rr if r['variant']==e['variant'] and int(r['seed'])==e['seed']]
                assert len(matches)==len({r['category'] for r in u.canonical(target)})
                assert all(r['checkpoint_id']==e['checkpoint_sha256'] for r in matches)
                value=float(np.mean([float(r['p_ap']) for r in matches]))
            raw.append(dict(source=e['source'],target=target,seed=e['seed'],variant=e['variant'],P_AP=value,
                total_updates=30,checkpoint=e['checkpoint'],checkpoint_sha256=e['checkpoint_sha256'],
                evaluator_id=config['evaluator_id'],feature_hash=e['feature_hash'],batch_hash=e['batch_hash']))
    assert len(raw)==90
    p.write_csv(META/'ALIGNED_RAW.csv',raw)
    for name,variants in [('LINEAR_HEAD',('LinearHead','NormalizedLinearHead','CES_NTR')),
                          ('STATE_CAPACITY',('CES_NTR','K3_30','K5_30'))]:
        summary=[];lines=['| Source -> Target | '+' | '.join(variants)+' |','|---|'+'---:|'*3]
        for target in (*TARGETS,'MEAN'):
            row=dict(target=target)
            for variant in variants:
                vv=[np.mean([r['P_AP'] for r in raw if r['seed']==seed and r['variant']==variant and (target=='MEAN' or r['target']==target)]) for seed in SEEDS]
                row[variant]=float(np.mean(vv));row[variant+'_SD']=float(np.std(vv,ddof=1))
            summary.append(row)
            label='Six-transfer mean' if target=='MEAN' else ('MVTec -> VisA' if target=='VisA' else 'VisA -> '+target)
            lines.append('| '+label+' | '+' | '.join(f'{row[v]:.2f} +/- {row[v+"_SD"]:.2f}' for v in variants)+' |')
        p.write_csv(META/f'{name}_SUMMARY.csv',summary)
        (META/f'{name}_TABLE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    p.write_csv(META/'COMPONENT_REFERENCE.csv',official.readcsv(COMP/'CORRECTED_COMPONENT_SUMMARY.csv'))
    (META/'REPORT.md').write_text('# Aligned decision/capacity controls\n\n'
        'All tables use the same corrected component source pools, physical patch identities, labels, '
        'actual batches and K2 checkpoints, with native GT/category-macro exact AP. '
        'Original MVTec source processing is retained; the negative complete-recipe probe is not adopted. '
        'K2/K3/K5 all have total30 updates. K3/K5 still differ in free-state parameterization, '
        'perturbation, raw logsumexp aggregation and diversity, so this is not a count-only causal ablation. '
        'Normalized Linear retains source-fixed c0, no bias. No main-table/Word/historical result overwrite.\n',encoding='utf-8')
    status('COMPLETE',unique_cells=90,shared_component_anchor=True)


def run():
    config=setup()
    seal=META/'SOURCE_TRAINING_COMPLETE.json'
    if seal.exists():
        saved=official.readjson(seal);assert saved['config_hash']==p.json_hash(config);models=saved['models']
        for e in models:assert p.file_hash(e['checkpoint'])==e['checkpoint_sha256']
    else:models=prepare(config)
    reuse_mvtec(models,config)
    plans={}
    for target in TARGETS[1:]:
        entries=[dict(e,lrp=True,output_id=e['model_id']+'__LRP') for e in models if e['source']=='Full VisA' and e['variant']!='CES_NTR']
        plans[target]=dict(target=target,outputs=entries,reuse={},new_outputs=[e['output_id'] for e in entries],n=len(u.canonical(target)))
    u.lock(META/'PREDICTION_PLAN.json',plans)
    if not (META/'SMOKE_COMPLETE.json').exists():smoke(config,plans)
    encoder=None
    for target in TARGETS[1:]:
        status('TARGET_RUNNING',target=target)
        encoder=u.predict(target,plans[target],encoder)
        u.evaluate(target,config,plans[target])
    summarize(config,models)


if __name__=='__main__':
    try:run()
    except BaseException:
        META.mkdir(parents=True,exist_ok=True)
        (META/'ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
        status('FAILED',error=traceback.format_exc())
        raise
