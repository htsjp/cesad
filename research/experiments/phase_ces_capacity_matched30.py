"""Equal TOTAL 30-update higher-state comparison on reciprocal MVTec/VisA.

K3/K5 start from the unoptimized source CES geometry, not a trained K2 head.
Keep the existing CMAB parameterization/LSE/diversity; do not tune by target.
"""
import argparse
import gc
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import phase_ces_unified_ablation_final as u

OUT=ROOT/'results/phase_ces_capacity_matched30/formal'
FROZEN=u.OUT
SEEDS=(0,1,2)
TARGETS=('VisA','MVTec')


def js(p,x):u.js(p,x)
def csv(p,x):u.csv(p,x)
def read(p):return u.read(p)
def status(phase,**kw):u.paired.status(OUT,phase,**kw)
def md(name,s):(OUT/name).write_text(s.strip()+'\n',encoding='utf-8')


def prepare():
    u.setup();OUT.mkdir(parents=True,exist_ok=True)
    oldconfig=read(FROZEN/'UNIFIED_CONFIG.json')
    oldmodels=read(FROZEN/'SOURCE_TRAINING_COMPLETE.json')['models']
    bases=[e for e in oldmodels if e['variant']=='CES_NTR']
    assert len(bases)==6
    assets=[];contexts=[]
    for source in u.SOURCES:
        for seed in SEEDS:
            folder,z,y,b,probe,meta=u.pool(source,seed)
            base=next(e for e in bases if e['source']==source and e['seed']==seed)
            assert u.paired.file_hash(base['checkpoint'])==base['checkpoint_sha256']
            p=torch.load(base['checkpoint'],map_location='cpu',weights_only=False)
            assert p['realized_updates']==30 and p['batch_stream_hash']==meta['batch_stream_hash']
            assert p['feature_array_hash']==meta['feature_array_hash']
            init=u.paired.init_from_meta(meta)
            for k in ('m','alpha','init_v'):np.testing.assert_array_equal(p['initial_geometry'][k],init[k])
            contexts.append(dict(source=source,seed=seed,pool_hash=meta['manifest_hash'],
                feature_hash=meta['feature_array_hash'],labels_hash=meta['labels_hash'],batch_hash=meta['batch_stream_hash'],
                original_geometry_hash=meta['initial_geometry_hash'],updates=30,epochs=10))
            assets.extend(u.ident(folder/n) for n in ('COMPLETE.json','PATCH_MANIFEST.csv','initial_geometry.npz','batch_indices.npy','probe_indices.npy'))
            assets.append(u.ident(base['checkpoint']))
    config=dict(task='Matched TOTAL 30 optimizer updates; reciprocal MVTec/VisA only',
        sources=list(u.SOURCES),targets=list(TARGETS),seeds=list(SEEDS),total_states=[2,3,5],
        epochs=10,total_updates=30,additional_pretrained_head_updates=0,
        initialization='source INITIAL m,alpha,v0 from StepB initial geometry; NO trained CES direction for K3/K5',
        K2='reuse verified canonical CES+NTR30 checkpoint',
        K3_K5='raw n0=m-alpha*unit(v0), a0=m+alpha*unit(v0), then existing projected noise basis',
        perturbation='existing cmab.init_anomaly_basis; seed=seed+(K-1); scale0.01*norm(a0)',
        optimizer=dict(name='AdamW',lr=.001,weight_decay=0.,betas=[.9,.999],eps=1e-8),
        loss='BCE+Dice+NTR rho0.10; high-state configurations additionally retain cosine-diversity weight1',
        aggregation='existing raw logsumexp of K-1 anomaly cosine logits minus normal cosine; tau1; no subtract logK',
        source_contexts=contexts,source_inputs=assets,
        evaluator_id=u.official.evaluator_hash(),GT='paired ORIGINAL native masks; equal category macro AP; exact ties',
        LRP='same fixed global+four locals;0.5/0.5;74map;bilinear align_cornersTrue to native GT',
        checkpoint_selection='final step30; NO target-based model selection',
        config_comparison_not_K_only=True,upstream_assets=[a for a in oldconfig['inputs'] if 'phase_statealigned_pc1_debias' in a['path']],
        code=[u.ident(__file__),u.ident(u.__file__),u.ident(u.cmab.__file__),u.ident(u.official.__file__)])
    u.lock(OUT/'CONFIG.json',config)
    md('PRE_RUN_VALIDATION.md','''# Equal total training budget

K2/K3/K5 all have TOTAL30 final-head updates (10 epochs), not30 versus60. K3/K5 are newly initialized from the SAME pre-final-training source m/alpha/v0 used by K2. They do not load any trained CES head as an initializer, and are not truncated versions of the historical60-step models.

Same source feature/physical-patch/label hashes, actual batch IDs, AdamW settings, NTR and frozen feature transforms. Only MVTec→VisA and VisA→MVTec, seeds0/1/2. K2 weights/native category results can be reused after matching source geometry, batches, checkpoint and evaluator identity.

The existing high-state free-parameter construction, logsumexp and diversity remain unchanged. Thus this fixes TOTAL training budget and trained-head initialization confounds, but does not turn the comparison into a count-only causal ablation. No target-based hyperparameter choice. Original60-step results and all other datasets stay untouched.
''')
    models=list(bases);diagnostics=[];alllogs=[]
    for source in u.SOURCES:
        for seed in SEEDS:
            _,z,y,batches,probe,meta=u.pool(source,seed)
            init=u.paired.init_from_meta(meta)
            v0=u.paired.hist.unit_np(init['init_v'])
            n0=np.asarray(init['m'],np.float32)-init['alpha']*v0
            a0=np.asarray(init['m'],np.float32)+init['alpha']*v0
            baseline=u.paired.hist.CESHead(init['m'],init['alpha'],v0).to(u.DEVICE)
            with torch.no_grad():
                xx=torch.tensor(np.asarray(z[probe]),device=u.DEVICE)
                q0=baseline(xx)
                one=u.cmab.multi_state_logits(xx,torch.tensor(n0,device=u.DEVICE),torch.tensor(a0[None],device=u.DEVICE))
                error=float(abs(q0-one).max());assert error<1e-6,error
            for K in (3,5):
                variant=f'K{K}_30';path=OUT/'checkpoints'/f'{u.mid(source,seed,variant)}.pt'
                if path.exists():
                    p=torch.load(path,map_location='cpu',weights_only=False)
                    assert p['config_hash']==u.paired.json_hash(config)
                    assert p['total_updates']==p['realized_updates']==30
                else:
                    torch.manual_seed(seed)
                    model=u.MultiState(n0,a0,K-1,seed).to(u.DEVICE)
                    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    np.testing.assert_array_equal(initial['s_n'].numpy(),n0)
                    np.testing.assert_array_equal(initial['s_a'].numpy(),u.cmab.init_anomaly_basis(a0,n0,K-1,seed+K-1))
                    first=u.probe_score(model,z,y,probe,True,True)
                    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.,betas=(.9,.999),eps=1e-8)
                    logs=[]
                    for step,ids in enumerate(batches):
                        x=torch.tensor(np.asarray(z[ids]),device=u.DEVICE);labels=torch.tensor(y[ids],device=u.DEVICE)
                        q=model(x);base,bce,dice=u.paired.balanced_loss(q,labels)
                        tail=u.paired.hist.ntr_loss_from_logits(q,labels,.1);div=u.cmab.diversity_loss(model.s_a)
                        loss=base+tail+div;assert torch.isfinite(loss)
                        optimizer.zero_grad(set_to_none=True);loss.backward()
                        grad=float(torch.sqrt(sum((v.grad.detach()**2).sum() for v in model.parameters())))
                        assert math.isfinite(grad);optimizer.step()
                        assert all(torch.isfinite(v).all() for v in model.parameters())
                        counters={int(v['step']) for v in optimizer.state.values()};assert counters=={step+1}
                        logs.append(dict(source=source,seed=seed,K=K,step=step+1,epoch=step//3+1,
                            batch_hash=u.paired.digest(ids),actual_batch=len(ids),loss=float(loss.detach()),
                            bce=float(bce.detach()),dice=float(dice.detach()),ntr=float(tail.detach()),
                            diversity=float(div.detach()),grad_norm=grad,optimizer_step=step+1,
                            probe=u.probe_score(model,z,y,probe,True,True)))
                    p=dict(source=source,seed=seed,K=K,total_updates=30,realized_updates=30,
                        initial_state=initial,initial_probe=first,initial_geometry=init,
                        initialization_stage='PRE_FINAL_TRAINING source geometry; NO trained base checkpoint',
                        s_n=model.s_n.detach().cpu().numpy(),s_a=model.s_a.detach().cpu().numpy(),
                        optimizer_state=optimizer.state_dict(),logs=logs,config_hash=u.paired.json_hash(config),
                        batch_stream_hash=meta['batch_stream_hash'],feature_array_hash=meta['feature_array_hash'],
                        labels_hash=meta['labels_hash'],checkpoint_selection='LAST_UPDATE30')
                    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');torch.save(p,temp);temp.replace(path)
                assert [r['batch_hash'] for r in p['logs']]==[u.paired.digest(ids) for ids in batches]
                csv(OUT/'training_logs'/f'{u.mid(source,seed,variant)}.csv',p['logs']);alllogs.extend(p['logs'])
                models.append(dict(model_id=u.mid(source,seed,variant),source=source,seed=seed,variant=variant,
                    checkpoint=str(path),checkpoint_sha256=u.paired.file_hash(path),ntr=True,lrp=True,
                    total_updates=30,feature_hash=meta['feature_array_hash'],batch_hash=meta['batch_stream_hash'],kind='multi'))
                diagnostics.append(dict(source=source,seed=seed,K=K,total_updates=30,inherited_final_head_updates=0,
                    same_batch_hash=meta['batch_stream_hash'],singleton_initial_logit_error=error,
                    initial_probe=p['initial_probe'],final_probe=p['logs'][-1]['probe']))
                status('HEAD_TRAINED',source=source,seed=seed,K=K,total_updates=30)
    assert len(models)==18
    csv(OUT/'INITIALIZATION_AND_TRAINING_DIAGNOSTICS.csv',diagnostics)
    csv(OUT/'TRAINING_STEPS.csv',alllogs)
    u.lock(OUT/'SOURCE_TRAINING_COMPLETE.json',dict(config_hash=u.paired.json_hash(config),models=models,
        new_training_runs=12,reused_K2=6,total_updates_per_configuration=30))
    copy=OUT/'code_snapshot';copy.mkdir(exist_ok=True)
    for x in config['code']:
        dest=copy/Path(x['path']).name
        if not dest.exists():shutil.copyfile(x['path'],dest)
        assert u.paired.file_hash(dest)==x['sha256']
    status('TRAINING_COMPLETE',new_heads=12,reused_K2=6)


def predict(target,models):
    # Only output destinations are rebound; all old feature/GT assets stay read-only.
    original=u.OUT;u.OUT=OUT
    try:rows,a,loc,mapping,dest=u.target_features(target)
    finally:u.OUT=original
    assert loc is not None,'Expected already-verified frozen local cache; no CLIP re-extraction'
    entries=[dict(e,output_id=e['model_id']+'__LRP',lrp=True) for e in models if e['source']==u.source_for(target) and e['variant']!='CES_NTR']
    assert len(entries)==6
    scorer=u.Scorer(entries,u.source_for(target))
    expected=dict(models=entries,GT_manifest=u.ident(u.official.WORK/'manifests'/f'{target}.csv'),n=len(rows))
    u.lock(dest/'IDENTITY.json',expected)
    path=dest/'maps74.npy';pp=dest/'PROGRESS.json'
    done=read(pp)['done'] if pp.exists() else 0
    if done==len(rows):return
    arrays=np.load(path,mmap_mode='r+') if path.exists() else np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(rows),6,74,74))
    assert arrays.shape==(len(rows),6,74,74)
    cache={};checks=[];start=time.time()
    for i in range(done,len(rows)):
        j=mapping[i];raw=np.concatenate((np.asarray(a[j],np.float32)[None],np.asarray(loc[j],np.float32)))
        maps=scorer(raw)
        for k,e in enumerate(entries):arrays[i,k]=maps[e['output_id']]
        if i==0:
            # Real cached-feature scoring parity against one-model original CMAB.
            for k,e in enumerate(entries):
                payload=torch.load(e['checkpoint'],map_location='cpu',weights_only=False)
                zz=u.paired.transformed(raw,scorer.transform,u.DEVICE)
                with torch.no_grad():q=u.cmab.multi_state_logits(zz,torch.tensor(payload['s_n'],device=u.DEVICE),torch.tensor(payload['s_a'],device=u.DEVICE)).view(5,37,37).cpu().numpy()
                reference=.5*u.paired.lzp.torch_resize(q[0],(74,74),True)+.5*u.paired.hist.stitch_local_maps(q[1:])
                err=float(abs(reference-arrays[i,k]).max());assert err<2e-6
                checks.append(dict(target=target,model=e['model_id'],map_parity_error=err))
            csv(dest/'SCORE_PARITY.csv',checks)
        if (i+1)%50==0 or i+1==len(rows):
            arrays.flush();js(pp,dict(done=i+1,total=len(rows)))
            status('PREDICTING',target=target,done=i+1,total=len(rows),elapsed_seconds=round(time.time()-start,1))
    arrays.flush();del arrays
    js(dest/'COMPLETE.json',dict(**expected,maps=u.ident(path)))


def evaluate(target,models):
    rows=u.canonical(target);dest=OUT/'predictions'/target
    complete=read(dest/'COMPLETE.json')
    assert u.paired.file_hash(dest/'maps74.npy')==complete['maps']['sha256']
    maps=np.load(dest/'maps74.npy',mmap_mode='r');entries=complete['models']
    old=u.rcsv(FROZEN/'evaluations'/f'{target}_CATEGORIES.csv')
    results=[];scratch=OUT/'scratch';scratch.mkdir(exist_ok=True)
    for category in sorted({r['category'] for r in rows}):
        ids=[i for i,r in enumerate(rows) if r['category']==category]
        for seed in SEEDS:
            e=next(x for x in models if x['source']==u.source_for(target) and x['seed']==seed and x['variant']=='CES_NTR')
            saved=next(x for x in old if x['category']==category and int(x['seed'])==seed and x['variant']=='CES_NTR' and x['lrp']=='True')
            assert saved['checkpoint_id']==e['checkpoint_sha256'] and saved['evaluator_id']==u.official.evaluator_hash()
            results.append(dict(source=e['source'],target=target,category=category,seed=seed,K=2,p_ap=float(saved['p_ap']),
                checkpoint=e['checkpoint'],checkpoint_id=e['checkpoint_sha256'],evaluator_id=saved['evaluator_id'],
                total_updates=30,provenance='verified native-GT K2 result reused'))
        for col,e in enumerate(entries):
            cp=OUT/'evaluations'/target/f'{category}__{e["model_id"]}.json'
            if cp.exists():
                r=read(cp);assert r['checkpoint_id']==e['checkpoint_sha256'];results.append(r);continue
            n=sum(int(rows[i]['gt_height'])*int(rows[i]['gt_width']) for i in ids)
            assert shutil.disk_usage(OUT).free>n*5+200_000_000,'Insufficient scratch; no automatic old-cache deletion'
            yp=scratch/'y.npy';sp=scratch/'s.npy'
            y=np.lib.format.open_memmap(yp,mode='w+',dtype=np.bool_,shape=(n,))
            s=np.lib.format.open_memmap(sp,mode='w+',dtype=np.float32,shape=(n,));off=0
            for i in ids:
                mask=u.official.native_mask(rows[i]);score=u.official.resize(maps[i,col],mask.shape)
                end=off+mask.size;y[off:end]=mask.reshape(-1);s[off:end]=score.reshape(-1);off=end
            y.flush();s.flush();assert off==n
            status('EVALUATING_NATIVE_AP',target=target,category=category,model=e['model_id'])
            value=u.official.weighted_pixel_metrics(y,s)['P_AP'];assert value is not None and math.isfinite(value)
            r=dict(source=e['source'],target=target,category=category,seed=e['seed'],K=int(e['variant'][1]),
                p_ap=value,checkpoint=e['checkpoint'],checkpoint_id=e['checkpoint_sha256'],evaluator_id=u.official.evaluator_hash(),
                total_updates=30,provenance='new same-total-budget evaluation')
            js(cp,r);results.append(r)
            del y,s;gc.collect();yp.unlink();sp.unlink()
    csv(OUT/'evaluations'/f'{target}_CATEGORIES.csv',results)


def summarize():
    rr=[]
    for target in TARGETS:
        path=OUT/'evaluations'/f'{target}_CATEGORIES.csv'
        if not path.exists():return False
        rows=u.rcsv(path)
        for seed in SEEDS:
            for K in (2,3,5):
                cells=[r for r in rows if int(r['seed'])==seed and int(r['K'])==K]
                assert len(cells)==len({r['category'] for r in u.canonical(target)})
                assert len({r['checkpoint_id'] for r in cells})==1
                rr.append(dict(source=u.source_for(target),target=target,seed=seed,K=K,total_updates=30,
                    p_ap=float(np.mean([float(r['p_ap']) for r in cells])),checkpoint=cells[0]['checkpoint'],
                    checkpoint_id=cells[0]['checkpoint_id'],evaluator_id=u.official.evaluator_hash()))
    assert len(rr)==18
    csv(OUT/'MATCHED30_RAW.csv',rr)
    summaries=[];lines=['| Source → Target | K=2 (30) | K=3 (30) | K=5 (30) |','|---|---:|---:|---:|']
    paired=[]
    for target in (*TARGETS,'TWO_TRANSFER_MEAN'):
        values={K:np.array([np.mean([r['p_ap'] for r in rr if r['seed']==seed and r['K']==K and (target=='TWO_TRANSFER_MEAN' or r['target']==target)]) for seed in SEEDS]) for K in (2,3,5)}
        label='MVTec → VisA' if target=='VisA' else ('VisA → MVTec' if target=='MVTec' else 'Two-transfer mean')
        lines.append('| '+label+' | '+' | '.join(f'{values[K].mean():.2f} ± {values[K].std(ddof=1):.2f}' for K in (2,3,5))+' |')
        for K in (2,3,5):
            summaries.append(dict(target=target,K=K,total_updates=30,mean=float(values[K].mean()),sample_SD=float(values[K].std(ddof=1)),
                **{f'seed{s}':float(values[K][s]) for s in SEEDS}))
        for K in (3,5):
            delta=values[K]-values[2]
            paired.append(dict(target=target,comparison=f'K{K}−K2',mean_delta=float(delta.mean()),sample_SD=float(delta.std(ddof=1)),
                **{f'delta_seed{s}':float(delta[s]) for s in SEEDS}))
    csv(OUT/'MATCHED30_SUMMARY.csv',summaries);csv(OUT/'MATCHED30_PAIRED.csv',paired)
    table='\n'.join(lines)
    md('MATCHED30_TABLE.md','# Higher-state configurations: matched TOTAL30 updates\n\n'+table+'\n\nNative-GT category-macro P-AP (%), mean ± sample SD over seeds0/1/2. Mean is over the two transfers within each seed.')
    md('REPORT.md','''# 同总预算状态配置对照

本表修复原30/60步表无法回答的同预算问题。K2/K3/K5均为总30次最终head更新（10epochs），无先训练30步再新增30步。K2复用正式30步模型；K3/K5从同一未优化source CES几何m/alpha/v0构造初始state，再保留原高状态扰动/LSE/diversity训练30步。全部共享真实源特征、标签、actual batch IDs、AdamW、NTR与Fixed LRP，只做MVTec↔VisA，不修改原60步结果及其他数据集。

'''+table+'''

完整未舍入逐种子结果见MATCHED30_RAW.csv，配对差值见MATCHED30_PAIRED.csv。该表不预设K2更强，所有结果都必须报告。高状态的自由state参数化、LSE聚合与diversity仍与canonical K2不同，因此是同预算配置对照，不是只改变K的严格因果消融。
''')
    js(OUT/'OUTCOME.json',dict(status='COMPLETE',raw_rows=18,total_updates_all_K=30,targets=list(TARGETS),
        new_heads=12,reused_K2=6,old_results_modified=False))
    status('COMPLETE',raw_rows=18,total_updates=30);return True


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('prepare','run'),default='run')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True);u.setup()
    if args.stage=='prepare':prepare();return
    config=read(OUT/'CONFIG.json');assert config['code'][0]['sha256']==u.paired.file_hash(__file__)
    models=read(OUT/'SOURCE_TRAINING_COMPLETE.json')['models']
    for e in models:assert u.paired.file_hash(e['checkpoint'])==e['checkpoint_sha256']
    for target in TARGETS:predict(target,models);evaluate(target,models)
    assert summarize()


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OUT.exists():js(OUT/'LAST_ERROR.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
