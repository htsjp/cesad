"""Final paired internal evidence; immutable historical inputs, native GT AP.

Run preflight -> train -> smoke -> run. Source fits never read target metrics.
Use lossless compressed score chunks; no frozen feature cache is regenerated.
All writes are contained in the new OUT directory. No baseline/Word edits.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import phase_ces_lhc_paired_repeat as paired
from experiments import phase_ces_geometry_center_control as geo
from experiments import final_unified_evaluation as official
from experiments import phase_cmab_compact_multi_anomaly_basis as cmab
from experiments.run_ces_geometry_center_control import ResolutionEstablishedEncoder

OUT=ROOT/'results/phase_ces_unified_ablation_final/formal'
OLD=ROOT/'results/phase_ces_lhc_paired_repeat/formal'
GEOM=ROOT/'results/phase_ces_geometry_center_control/formal'
SOURCES=('Full MVTec','Full VisA')
SEEDS=(0,1,2)
TARGETS=('VisA','MVTec','BTAD','MPDD','DTD-Synthetic','DAGM')
NEW=('CES_noNTR','CES_NTR_continue','K3_config','K5_config')
SCALAR=('LinearHead','NormalizedLinearHead','CES_NTR','CES_noNTR','CES_NTR_continue','E1_Center')
DEVICE='cuda'


def js(path,value):paired.write_json(path,value)
def csv(path,value):paired.write_csv(path,value)
def read(path):return official.readjson(path)
def rcsv(path):return official.readcsv(path)
def md(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix('.tmp');temp.write_text(value.strip()+'\n',encoding='utf-8');temp.replace(p)
def status(phase,**kw):paired.status(OUT,phase,**kw)
def ident(p):
    p=Path(p);return dict(path=str(p.resolve()),sha256=paired.file_hash(p),bytes=p.stat().st_size)
def lock(p,value):
    value=json.loads(json.dumps(value,default=str))
    if p.exists():assert read(p)==value,f'Identity changed: {p}'
    else:js(p,value)
def slug(source):return source.replace(' ','_')
def mid(source,seed,variant):return f'{slug(source)}__seed{seed}__{variant}'
def source_for(target):return 'Full MVTec' if target=='VisA' else 'Full VisA'
def target_list(source):return ('VisA',) if source=='Full MVTec' else TARGETS[1:]
def setup():
    OUT.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(4)
    assert torch.cuda.is_available(),'No implicit CPU training fallback'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)


def pool(source,seed):
    p=OLD/'source_pools'/f'{slug(source)}_seed{seed}'
    meta=read(p/'COMPLETE.json')
    z=np.load(p/'features.npy',mmap_mode='r');y=np.load(p/'labels.npy')
    b=np.load(p/'batch_indices.npy');probe=np.load(p/'probe_indices.npy')
    for k,a in [('feature_array_hash',z),('labels_hash',y),('batch_stream_hash',b),('probe_indices_hash',probe)]:
        assert paired.digest(a)==meta[k],(source,seed,k)
    assert paired.file_hash(p/'PATCH_MANIFEST.csv')==meta['manifest_hash']
    assert b.shape==(30,8184 if source=='Full VisA' else 8190)
    assert meta['epochs']==10 and len(z)==len(y)
    return p,z,y,b,probe,meta


def spec_for(entry):
    p=torch.load(entry['checkpoint'],map_location='cpu',weights_only=False)
    if entry['variant'].startswith('K'):
        return p
    method=entry['variant'] if entry['variant'] in ('LinearHead','NormalizedLinearHead') else 'CES'
    return paired.hist.HeadSpec(method,p['params'],entry['source'],entry['seed'])


def param_identity(entry):
    spec=spec_for(entry)
    if isinstance(spec,dict):return paired.digest(np.concatenate((spec['s_n'].reshape(-1),spec['s_a'].reshape(-1))))
    return paired.digest(np.concatenate((spec.weight().reshape(-1),np.array([spec.bias()],np.float32))))


def validate_reused(source,seed,method,p,meta,batches):
    saved=torch.load(p,map_location='cpu',weights_only=False)
    assert saved['source']==source and saved['seed']==seed and saved['method']==method
    assert saved['lr']==.001 and saved['realized_updates']==30 and saved['run_type']=='primary'
    assert saved['feature_array_hash']==meta['feature_array_hash']
    assert saved['batch_stream_hash']==meta['batch_stream_hash']
    assert [r['batch_ids_hash'] for r in saved['logs']]==[paired.digest(ids) for ids in batches]
    assert [r['optimizer_step_min'] for r in saved['logs']]==list(range(1,31))
    assert [r['optimizer_step_max'] for r in saved['logs']]==list(range(1,31))
    init=paired.init_from_meta(meta)
    for key in ('m','alpha','init_v','w0','c0'):
        np.testing.assert_array_equal(saved['initial_geometry'][key],init[key])
    return saved


def preflight():
    setup();assets=[];models=[];audit=[];initial=[]
    transform=paired.load_transform('Full VisA',DEVICE)
    e1=torch.load(paired.PAYLOAD/'E1_PC1Removed-TrueMask_formal.pt',map_location='cpu',weights_only=False)
    e2=torch.load(paired.PAYLOAD/'E2_PC1Removed-CorruptedMask_formal.pt',map_location='cpu',weights_only=False)
    assert np.asarray(e1['patch_states']).shape==np.asarray(e2['patch_states']).shape==(2,1024)
    for p in (e1,e2):
        assert Counter(r.get('stage') for r in p['curves'])==dict(state=20,adapter=10)
    for source in SOURCES:
        for seed in SEEDS:
            folder,z,y,b,probe,meta=pool(source,seed)
            initial.extend(paired.equivalence(z,meta,DEVICE))
            assets.extend(folder/n for n in ('COMPLETE.json','features.npy','labels.npy','batch_indices.npy',
                         'probe_indices.npy','PATCH_MANIFEST.csv','SOURCE_IMAGE_MANIFEST.csv','initial_geometry.npz'))
            for method,var in [('CES','CES_NTR'),('LinearHead','LinearHead'),('NormalizedLinearHead','NormalizedLinearHead')]:
                cp=OLD/'checkpoints'/f'{slug(source)}_{method}_seed{seed}_primary_lr0.001.pt'
                saved=validate_reused(source,seed,method,cp,meta,b)
                entry=dict(model_id=mid(source,seed,var),source=source,seed=seed,variant=var,
                    checkpoint=str(cp),checkpoint_sha256=paired.file_hash(cp),origin='REUSED_STEPB_PRIMARY',
                    feature_hash=meta['feature_array_hash'],source_pool_hash=meta['manifest_hash'],
                    batch_hash=meta['batch_stream_hash'],ntr=True,base_updates=30,extra_updates=0,total_updates=30,
                    initialization='matched source effective logits',kind='scalar')
                models.append(entry);assets.append(cp)
                audit.append(dict(Asset=str(cp),Source=source,Seed=seed,Feature_Hash=meta['feature_array_hash'],
                    Source_Pool_Hash=meta['manifest_hash'],Initialization='source matched logits',
                    Training='BCE+Dice+NTR; AdamW1e-3 wd0;30 updates; common actual batches',
                    Checkpoint_Hash=entry['checkpoint_sha256'],Reusable='DIRECT_CHECKPOINT_REUSE; native GT reevaluation required'))
            if source=='Full VisA':
                hybrid=GEOM/'checkpoints'/f'Hybrid_CES_seed{seed}.pt'
                center=GEOM/'checkpoints'/f'E1_Center_CES_seed{seed}.pt'
                hp=torch.load(hybrid,map_location='cpu',weights_only=False)
                ref=torch.load(models[-3]['checkpoint'],map_location='cpu',weights_only=False)
                for key in ('m','alpha','v'):assert torch.equal(hp['raw_model_state'][key],ref['raw_model_state'][key])
                ep=torch.load(center,map_location='cpu',weights_only=False)
                assert ep['feature_array_hash']==meta['feature_array_hash'] and ep['batch_stream_hash']==meta['batch_stream_hash']
                assert [r['batch_ids_hash'] for r in ep['logs']]==[paired.digest(ids) for ids in b]
                assert ep['realized_updates']==30
                assert torch.equal(ep['raw_v_trajectory'][0],hp['raw_v_trajectory'][0])
                assert torch.equal(ep['raw_model_state']['alpha'],hp['raw_model_state']['alpha'])
                assert np.array_equal(np.asarray(hp['params']['m'],np.float32),transform['m'])
                m_e1=.5*(np.asarray(e1['patch_states'])[0]+np.asarray(e1['patch_states'])[1])
                np.testing.assert_array_equal(ep['params']['m'],m_e1)
                entry=dict(model_id=mid(source,seed,'E1_Center'),source=source,seed=seed,variant='E1_Center',
                    checkpoint=str(center),checkpoint_sha256=paired.file_hash(center),origin='REUSED_PAIRED_GEOMETRY',
                    feature_hash=meta['feature_array_hash'],source_pool_hash=meta['manifest_hash'],batch_hash=meta['batch_stream_hash'],
                    ntr=True,base_updates=30,extra_updates=0,total_updates=30,initialization='only reference center differs',kind='scalar')
                models.append(entry);assets.extend((hybrid,center))
                audit.append(dict(Asset=str(center),Source=source,Seed=seed,Feature_Hash=meta['feature_array_hash'],
                    Source_Pool_Hash=meta['manifest_hash'],Initialization='E1 center only; same E2 feature transform',
                    Training='matched complete center-control fit',Checkpoint_Hash=entry['checkpoint_sha256'],Reusable='DIRECT_CHECKPOINT_AND_NATIVE_RESULT_REUSE'))
                audit.append(dict(Asset=str(hybrid),Source=source,Seed=seed,Feature_Hash=meta['feature_array_hash'],
                    Source_Pool_Hash=meta['manifest_hash'],Initialization='hybrid',Training='duplicate exact raw CES m/alpha/v verified',
                    Checkpoint_Hash=paired.file_hash(hybrid),Reusable='PARAMETER_IDENTICAL_ALIAS_TO_STEPB; do not add another reference'))
            for var in NEW:
                audit.append(dict(Asset=f'PLANNED:{mid(source,seed,var)}',Source=source,Seed=seed,
                    Feature_Hash=meta['feature_array_hash'],Source_Pool_Hash=meta['manifest_hash'],
                    Initialization='common initial geometry' if var=='CES_noNTR' else 'same trained CES_NTR base',
                    Training='30 actual paired updates',Checkpoint_Hash='',Reusable='NEW_HEAD_TRAINING_REQUIRED'))
    # Retain legacy incompatible assets as evidence, never as paired outcomes.
    for source in SOURCES:
        for name in ('E1_K2_CMAB','E2_K4_CMAB'):
            cp=ROOT/f'results/phase_cmab_compact_multi_anomaly_basis/formal/{slug(source)}_{name}_formal.pt'
            assert cp.exists();payload=torch.load(cp,map_location='cpu',weights_only=False)
            assets.append(cp)
            audit.append(dict(Asset=str(cp),Source=source,Seed='NOT_SERIALIZED',Feature_Hash='NOT_AUTHENTICATED',
                Source_Pool_Hash='legacy pool differs/not batch paired',Initialization='trained legacy CES + projected perturbation',
                Training='10 extra epochs; no retained optimizer/batch stream',Checkpoint_Hash=paired.file_hash(cp),
                Reusable='STRUCTURE_REFERENCE_ONLY; RETRAIN_MATCHED_CONFIGURATIONS'))
        cp=ROOT/f'results/ces_ntr_visualization/corrected/checkpoints/{slug(source)}_seed0_statealigned.pt'
        assert cp.is_file();assets.append(cp)
        audit.append(dict(Asset=str(cp),Source=source,Seed=0,Feature_Hash='not authenticated to current StepB pool',
            Source_Pool_Hash='legacy sampler; no matched realized StepB stream',Initialization='historical trained noNTR',
            Training='not established paired to StepB',Checkpoint_Hash=paired.file_hash(cp),Reusable='NOT_PAIRED; train six new noNTR heads'))
    assets.extend(paired.PAYLOAD/n for n in ('E1_PC1Removed-TrueMask_formal.pt','E2_PC1Removed-CorruptedMask_formal.pt','q_pc1_statealigned1024.npy'))
    assets.extend([OLD/'LOCKED_CONFIG.json',GEOM/'GEOMETRY_CONTROL_CONFIG.json',GEOM/'SOURCE_TRAINING_COMPLETE.json'])
    for target in TARGETS:
        mp=official.WORK/'manifests'/f'{target}.csv';rows=rcsv(mp);assets.append(mp)
        assert all(Path(r['img_path']).is_file() and (not int(r['label']) or Path(r['gt_path']).is_file()) for r in rows)
        assert all(not int(r['label']) or official.normpath(r['img_path'])!=official.normpath(r['gt_path']) for r in rows)
    scripts=[Path(__file__),Path(paired.__file__),Path(paired.hist.__file__),Path(cmab.__file__),Path(official.__file__),
             Path(geo.__file__),ROOT/'experiments/run_ces_geometry_center_control.py',ROOT/'phase_xa/official_task_forward.py',
             ROOT/'experiments/unified_ablation_final_reporting.py']
    status('HASHING_LOCKED_SOURCE_ASSETS',count=len(set(assets)))
    source_ids=[ident(p) for p in dict.fromkeys(assets)]
    config=dict(phase='CESAD unified final internal ablation',seeds=list(SEEDS),sources=list(SOURCES),
        targets={t:source_for(t) for t in TARGETS},primary_metric='native GT equal category-macro P-AP percent',
        evaluator_id=official.evaluator_hash(),evaluator_policy=official.POLICY,
        optimizer=dict(name='AdamW',lr=.001,weight_decay=0.,betas=[.9,.999],eps=1e-8),
        fit=dict(epochs=10,updates=30,batches='exact saved StepB 30-step stream within source/seed',
                 loss='balanced BCE+Dice; +normal-top0.10 softplus iff NTR',rho=.1,checkpoint='last update',scheduler=None),
        higher_state=dict(total_states=[2,3,5],counts='1N plus K-1 anomaly',base='the canonical shared trained CES_NTR checkpoint',
            raw_init='m +/- alpha*unit(v), before state normalization',perturbation='existing cmab.init_anomaly_basis; seed=training_seed+(K-1)',
            aggregation='raw logsumexp(cos(z,sAk))-cos(z,sN); tau1; no logK correction',
            regularizer='existing mean ordered off-diagonal anomaly cosine; coefficient1',
            extra_epochs=10,extra_updates=30,total_updates=60,optimizer='fresh AdamW1e-3 wd0',
            K2_continuation='canonical raw_v continuation, m/alpha frozen; fresh AdamW; same extra30 batches',
            deliberate_unification='same physical StepB pools and same source/seed batch stream for all extra-stage controls; no K-dependent batch sampling',
            not_count_only=True),
        lrp='1 global+4 nonoverlap crops518;37grid TL/TR/BL/BR;0.5global+0.5local74;bilinear align_cornersTrue;no smoothing',
        global_only='37x37 raw map resized directly to native GT; no intermediate74 resize',
        reused_distinct_models=21,new_training_runs=24,final_distinct_models=45,
        planned_unique_evaluations=177,planned_logical_result_rows=228,
        target_use='final evaluation only; no LR selection or early stopping; all source heads sealed first',
        conditional_LR='historical diagnostic retained only; no new sweep',
        inputs=source_ids,code=[ident(p) for p in scripts],
        dtype='FP32 head/logits; frozen caches retain documented storage dtype; no autocast/TF32',
        storage='lossless compressed float32 new score chunks; existing legitimate map caches referenced, not copied',
        software=dict(torch=torch.__version__,numpy=np.__version__,python=sys.version,GPU=torch.cuda.get_device_name(0)))
    lock(OUT/'UNIFIED_CONFIG.json',config)
    csv(OUT/'UNIFIED_ASSET_REGISTRY.csv',audit);csv(OUT/'UNIFIED_INITIALIZATION_EQUIVALENCE.csv',initial)
    lock(OUT/'REUSED_MODELS.json',models)
    snapshots=OUT/'code_snapshot';snapshots.mkdir(exist_ok=True)
    for p in scripts:
        dest=snapshots/p.name
        if dest.exists():assert paired.file_hash(dest)==paired.file_hash(p)
        else:shutil.copyfile(p,dest)
    md('UNIFIED_PREFLIGHT_AUDIT.md','''# 统一预审

PASS。StepB共同学习率18个checkpoint逐一核对源特征hash、源池manifest、初始化、30个真实batch hash和Adam step。几何分支3个E1-center可直接复用；3个Hybrid已逐元素验证m/alpha/raw_v与共享StepB CES完全一致，只作为缓存别名，不再建立另一套Full参考。

旧no-NTR与K3/K5不具备当前StepB物理池/实际batch匹配证据，不能标记严格配对。旧高状态仅保留初始化、logsumexp与diversity定义。新统一高状态会从每source/seed共享CES_NTR出发，在统一池和统一额外batch流上训练；相对旧版改变的是统一源输入/采样身份，不静默称为历史复现或纯K因果消融。

E1/E2使用Stage-I后patch states，特征变换保留Stage-II冻结E2 adapter。最终训练标签为真实source loader mask，不使用辅助shifted-mask。所有源资产、初始参数、优化器和评价器版本在JSON中锁定。最终目标GT必须为主表native paired masks，DTD禁止使用异常输入图像冒充GT。

NormalizedLinearHead保持StepB真实定义：fixed c0 * unit(u)^T z，无bias、c0不训练。保留固定初始logit尺度，不声称它无尺度或有更弱的表达维数。

可直接复用21个独特模型，另有3个Hybrid相同参数别名；必须训练24个head，不训练任何CLIP/PC1/E1/E2/adapter。六迁移中不含历史MVTec→MPDD。source-selected LR不进入本次共同LR主实验。
''')
    md('UNIFIED_EXECUTION_PLAN.md','''# 最终执行计划

1. 复用18个StepB共同LR模型与3个E1-center模型；21个独特checkpoint。3个Hybrid为参数完全一致的缓存别名。
2. 每个Source×Seed训练no-NTR、K2 continuation、K3、K5：共24次新head fit，每次30步。加上复用模型，总45个独特训练权重。
3. 六迁移177个独特model/inference评价，映射为四表228行逻辑结果；重复Full参考只评价一次。几何表只含五个VisA-source目标。
4. K2 base30步，K2 continuation/K3/K5均含base30+extra30步。K3/K5保留既有多状态参数化、扰动、LSE、cosine penalty；源输入/extra batch统一。不是仅改变K的纯因果对照。
5. 先封存全部源模型，再做12张smoke和正式预测/评价。可复用的geometry native结果与StepB原始maps按hash/参数身份验证后复用；绝不复制旧loader-GT指标。
6. 所有新图预测为无损float32压缩chunk，读取旧global/local特征；DTD/DAGM必要的四crop编码在所有模型之间共享，不创建巨大额外特征库。每类逐模型评价以限制内存。
7. 更新预算、代码快照、输入/权重/采样hash、完成标记和原子结果全部位于本独立目录。若某分支出现科学匹配冲突，停止该分支，不补造结果。

暂无需要人工定义的新算法冲突。磁盘容量在预测计划阶段按实际缺失输出的未压缩上界复核，保留scratch余量。旧表、论文与历史数据均只读。任务结束后冻结，不开启下一轮消融。
''')
    status('PREFLIGHT_PASSED',reused=21,new_head_fits=24,planned_results=228)
    return config,models


class MultiState(torch.nn.Module):
    def __init__(self,n,a,k,seed):
        super().__init__()
        self.s_n=torch.nn.Parameter(torch.tensor(n,dtype=torch.float32))
        self.s_a=torch.nn.Parameter(torch.tensor(cmab.init_anomaly_basis(a,n,k,seed+k),dtype=torch.float32))
    def forward(self,z):return cmab.multi_state_logits(z,self.s_n,self.s_a)


def probe_score(model,z,y,probe,ntr,div=False):
    with torch.no_grad():
        q=model(torch.tensor(np.asarray(z[probe]),device=DEVICE));labels=torch.tensor(y[probe],device=DEVICE)
        base,bce,dice=paired.balanced_loss(q,labels)
        tail=paired.hist.ntr_loss_from_logits(q,labels,.1) if ntr else q.new_zeros(())
        diversity=cmab.diversity_loss(model.s_a) if div else q.new_zeros(())
    return dict(loss=float(base+tail+diversity),bce=float(bce),dice=float(dice),ntr=float(tail),diversity=float(diversity),
                mean_normal=float(q[labels<.5].mean()),mean_anomaly=float(q[labels>.5].mean()))


def train(config,models):
    output=list(models);alllogs=[]
    for source in SOURCES:
        for seed in SEEDS:
            folder,z,y,batches,probe,meta=pool(source,seed)
            baseentry=next(e for e in models if e['source']==source and e['seed']==seed and e['variant']=='CES_NTR')
            baseline=torch.load(baseentry['checkpoint'],map_location='cpu',weights_only=False)
            for variant in NEW:
                cp=OUT/'checkpoints'/f'{mid(source,seed,variant)}.pt'
                if cp.exists():
                    saved=torch.load(cp,map_location='cpu',weights_only=False)
                    assert saved['config_hash']==paired.json_hash(config) and saved['batch_stream_hash']==meta['batch_stream_hash']
                    assert saved['realized_updates']==30
                else:
                    torch.manual_seed(seed)
                    if variant.startswith('K'):
                        k=int(variant[1])-1
                        p=baseline['params'];v=np.asarray(p['v'],np.float32)
                        n=np.asarray(p['m'],np.float32)-p['alpha']*v
                        a=np.asarray(p['m'],np.float32)+p['alpha']*v
                        model=MultiState(n,a,k,seed).to(DEVICE)
                        extra=True;ntr=True
                    else:
                        init=paired.init_from_meta(meta)
                        model=paired.hist.CESHead(init['m'],init['alpha'],init['init_v']).to(DEVICE)
                        if variant=='CES_NTR_continue':model.load_state_dict(baseline['raw_model_state'],strict=True)
                        extra=variant=='CES_NTR_continue';ntr=variant!='CES_noNTR'
                    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.,betas=(.9,.999),eps=1e-8)
                    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    initial_probe=probe_score(model,z,y,probe,ntr,variant.startswith('K'))
                    logs=[]
                    for step,ids in enumerate(batches):
                        x=torch.tensor(np.asarray(z[ids]),device=DEVICE);labels=torch.tensor(y[ids],device=DEVICE)
                        q=model(x);state,bce,dice=paired.balanced_loss(q,labels)
                        tail=paired.hist.ntr_loss_from_logits(q,labels,.1) if ntr else q.new_zeros(())
                        div=cmab.diversity_loss(model.s_a) if variant.startswith('K') else q.new_zeros(())
                        loss=state+tail+div;assert torch.isfinite(loss)
                        optimizer.zero_grad(set_to_none=True);loss.backward()
                        grad=float(torch.sqrt(sum((p.grad.detach()**2).sum() for p in model.parameters())))
                        assert math.isfinite(grad);optimizer.step()
                        assert all(torch.isfinite(p).all() for p in model.parameters())
                        counters=[int(st['step']) for st in optimizer.state.values()];assert set(counters)=={step+1}
                        logs.append(dict(source=source,seed=seed,variant=variant,step=step+1,epoch=step//3+1,
                            actual_batch=len(ids),batch_ids_hash=paired.digest(ids),loss=float(loss.detach()),
                            bce=float(bce.detach()),dice=float(dice.detach()),ntr=float(tail.detach()),diversity=float(div.detach()),
                            grad_norm=grad,optimizer_step=step+1,
                            probe=probe_score(model,z,y,probe,ntr,variant.startswith('K'))))
                    state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}
                    saved=dict(source=source,seed=seed,variant=variant,method='MultiState' if variant.startswith('K') else 'CES',
                        initial_state=initial,raw_model_state=state_dict,optimizer_state=optimizer.state_dict(),
                        logs=logs,initial_probe=initial_probe,realized_updates=30,
                        feature_array_hash=meta['feature_array_hash'],labels_hash=meta['labels_hash'],
                        source_manifest_hash=meta['manifest_hash'],batch_stream_hash=meta['batch_stream_hash'],
                        initial_base_checkpoint=baseentry['checkpoint'] if extra else None,
                        initial_base_hash=baseentry['checkpoint_sha256'] if extra else None,
                        total_updates=60 if extra else 30,config_hash=paired.json_hash(config),checkpoint_selection='LAST_UPDATE')
                    if variant.startswith('K'):
                        saved.update(s_n=state_dict['s_n'].numpy(),s_a=state_dict['s_a'].numpy())
                    else:saved['params']=model.to_spec(source).params
                    cp.parent.mkdir(parents=True,exist_ok=True);tmp=cp.with_suffix('.tmp');torch.save(saved,tmp);tmp.replace(cp)
                    csv(OUT/'training_logs'/f'{mid(source,seed,variant)}.csv',logs)
                assert [r['batch_ids_hash'] for r in saved['logs']]==[paired.digest(ids) for ids in batches]
                alllogs.extend(saved['logs'])
                extra=variant!='CES_noNTR'
                entry=dict(model_id=mid(source,seed,variant),source=source,seed=seed,variant=variant,
                    checkpoint=str(cp),checkpoint_sha256=paired.file_hash(cp),origin='NEW_PAIRED_FIT',
                    feature_hash=meta['feature_array_hash'],source_pool_hash=meta['manifest_hash'],batch_hash=meta['batch_stream_hash'],
                    ntr=variant!='CES_noNTR',base_updates=30 if extra else 0,extra_updates=30 if extra else 0,
                    total_updates=60 if extra else 30,initialization='shared CES_NTR base' if extra else 'shared original CES initializer',
                    kind='multi' if variant.startswith('K') else 'scalar')
                output.append(entry)
                status('HEAD_READY',source=source,seed=seed,variant=variant,total_updates=entry['total_updates'])
    assert len(output)==45 and len({e['model_id'] for e in output})==45
    lock(OUT/'SOURCE_TRAINING_COMPLETE.json',dict(config_hash=paired.json_hash(config),models=output,
         all_target_scores_unopened_during_training=True,checkpoint_selection='last fixed update'))
    csv(OUT/'UNIFIED_CHECKPOINT_REGISTRY.csv',output);csv(OUT/'UNIFIED_TRAINING_STEPS.csv',alllogs)
    md('UNIFIED_TRAINING_AUDIT.md','''# 统一配对训练核验

18个StepB主模型与3个E1-center模型完成身份核验后复用。所有24个新head每组30个实际更新，完整保存raw参数、初始化参数、optimizer state、逐步BCE/Dice/NTR/diversity、梯度、固定源probe与batch hash。source/seed内实际源特征、物理patch manifest、监督标签与30步采样数组共用。

no-NTR仅删除NTR项，完整训练基础BCE+Dice。Global/LRP共用同一个训练权重，不分别训练。Full CES参考唯一来自StepB共同LR CES_NTR，不因表格不同重新选一个更高模型。

高状态从同一个已训练CES_NTR原始构造状态m±alpha*unit(v)出发，保留原有CMAB perturbation/LSE/cosine penalty。K3/K5及canonical K2 continuation共用30步extra batch和新AdamW，共60步总预算；K2 base为30步。扰动seed=training_seed+anomaly_count，仅扰动RNG随K变化，实际batch流不随K变化。不是纯状态数量因果消融。

SOURCE_TRAINING_COMPLETE.json在目标预测/GT指标评价前封存45个模型hash。没有source-selected LR搜索或目标选模。所有同一source的五目标是重复训练后的相关评价，不是额外源训练重复。三种子统计使用ddof=1，先每seed聚合迁移再计算SD。
''')
    return output


def outputs(models,target):
    entries=[e for e in models if e['source']==source_for(target)]
    result=[]
    for e in entries:
        result.append(dict(**e,output_id=e['model_id']+'__LRP',lrp=True))
        if e['variant'] in ('CES_NTR','CES_noNTR'):
            result.append(dict(**e,output_id=e['model_id']+'__Global',lrp=False))
    assert len(result)==(27 if target=='VisA' else 30)
    return result


def canonical(target):return rcsv(official.WORK/'manifests'/f'{target}.csv')


def reuse_predictions(target,out_entries):
    rows=canonical(target);norm=official.normpath
    refs={};files={}
    # Read existing StepB MAPS only: its loader-GT metric/mask caches are NOT used.
    old=OLD/'predictions'/paired.hist.pair_slug(source_for(target),target)
    if (old/'COMPLETE.json').exists() and target in ('VisA','MVTec','BTAD'):
        done=read(old/'COMPLETE.json');manifest=rcsv(old/'TARGET_MANIFEST.csv')
        mapping={norm(r['image']):int(r['position']) for r in manifest}
        assert set(mapping)=={norm(r['img_path']) for r in rows}
        files[str(old/'maps74.npy')]=done['maps_hash']
        for e in out_entries:
            if not e['lrp']:continue
            matches=[(i,x) for i,x in enumerate(done['models']) if x['checkpoint_hash']==e['checkpoint_sha256']]
            if len(matches)==1:
                refs[e['output_id']]=dict(kind='historical_map',path=str(old/'maps74.npy'),column=matches[0][0],
                    index=[mapping[norm(r['img_path'])] for r in rows],proof='same checkpoint SHA256 + physical target image order')
    if target!='VisA':
        gp=GEOM/'predictions'/target;done=read(gp/'COMPLETE.json');manifest=rcsv(gp/'TARGET_MANIFEST.csv')
        mapping={norm(r['img_path']):i for i,r in enumerate(manifest)}
        assert set(mapping)=={norm(r['img_path']) for r in rows}
        files[str(gp/'maps74.npy')]=done['maps']['sha256']
        for e in out_entries:
            if not e['lrp'] or e['variant'] not in ('CES_NTR','E1_Center'):continue
            label=f"{'Hybrid_CES' if e['variant']=='CES_NTR' else 'E1_Center_CES'}_seed{e['seed']}"
            matches=[(i,x) for i,x in enumerate(done['models']) if x['model_id']==label];assert len(matches)==1
            col,oldentry=matches[0]
            oldspec=paired.hist.HeadSpec('CES',torch.load(oldentry['checkpoint'],map_location='cpu',weights_only=False)['params'],e['source'],e['seed'])
            spec=spec_for(e)
            np.testing.assert_array_equal(spec.weight(),oldspec.weight());assert spec.bias()==oldspec.bias()
            refs[e['output_id']]=dict(kind='geometry_map',path=str(gp/'maps74.npy'),column=col,
                index=[mapping[norm(r['img_path'])] for r in rows],proof='exact effective weight/bias + matching source/frozen transform/batch + physical target identity',
                geometry_variant='Hybrid_CES' if e['variant']=='CES_NTR' else 'E1_Center_CES')
    return refs,files


def build_prediction_plan(config,models):
    plans={};filehash={};raw_bytes=0
    for target in TARGETS:
        oo=outputs(models,target);refs,files=reuse_predictions(target,oo);filehash.update(files)
        needed=[e['output_id'] for e in oo if e['output_id'] not in refs]
        n=len(canonical(target))
        raw_bytes+=sum(n*(74 if e['lrp'] else 37)**2*4 for e in oo if e['output_id'] in needed)
        plans[target]=dict(target=target,outputs=oo,reuse=refs,new_outputs=needed,n=n)
    for p,h in filehash.items():assert paired.file_hash(p)==h,('Old prediction cache changed',p)
    free=shutil.disk_usage(OUT).free
    # Existing chunks reduce remaining work on resume; use full upper bound on first run.
    assert (OUT/'PREDICTION_PLAN.json').exists() or free>raw_bytes+1_500_000_000, ('Insufficient safe disk budget; no old cache deletion',free,raw_bytes)
    lock(OUT/'PREDICTION_PLAN.json',plans)
    js(OUT/'DISK_AND_CACHE_AUDIT.json',dict(free_at_check=free,new_raw_float32_upper_bound=raw_bytes,
         lossless_compression=True,scratch_margin_bytes=1_500_000_000,reused_map_files=filehash,
         unique_output_evaluations=sum(len(v['outputs']) for v in plans.values()),
         outputs_needing_new_prediction=sum(len(v['new_outputs']) for v in plans.values())))
    return plans


def target_features(target):
    rows=canonical(target)
    if target=='VisA':
        _,ds,tr,held=paired.light_fold('D2',0)
        gp=paired.hist.LOCAL_CACHE_DIR/'global_patch_VisA.npy'
        paths=[str(Path(ds.root)/ds.data_all[i]['img_path']) for i in tr+held]
        a=np.load(gp,mmap_mode='r')
    else:a,paths,gp=geo.global_inputs(target)
    paths=official.canonical_paths(target,paths,rows)
    lookup={official.normpath(p):i for i,p in enumerate(paths)}
    assert len(lookup)==len(rows)==len(a)
    mapping=[lookup[official.normpath(r['img_path'])] for r in rows]
    lp=paired.hist.local_cache_path(source_for(target),target)
    if target=='MPDD':lp=geo.local_path(target)
    loc=np.load(lp,mmap_mode='r') if lp.exists() else None
    if loc is not None:assert loc.shape==(len(rows),4,1369,1024)
    dest=OUT/'predictions'/target;dest.mkdir(parents=True,exist_ok=True)
    cp=dest/'INPUT_IDENTITY.json'
    if not cp.exists():
        for r in rows:
            assert paired.file_hash(r['img_path'])==r['image_sha256'],r['sample_id']
            if int(r['label']):assert paired.file_hash(r['gt_path'])==r['gt_sha256'],r['sample_id']
        js(cp,dict(global_cache=ident(gp),local_cache=ident(lp) if loc is not None else 'shared fresh CLIP crops',
            manifest=ident(official.WORK/'manifests'/f'{target}.csv'),
            ordered_cache_images_hash=paired.json_hash(paths),image_and_GT_bytes='MATCH_FROZEN_NATIVE_MAIN_TABLE',
            input_float_dtype=str(a.dtype),source_transform='same locked StepB source transform'))
    else:
        old=read(cp)
        assert old['ordered_cache_images_hash']==paired.json_hash(paths)
        assert old['manifest']['sha256']==paired.file_hash(official.WORK/'manifests'/f'{target}.csv')
        assert old['global_cache']['sha256']==paired.file_hash(gp)
        if loc is not None:assert old['local_cache']['sha256']==paired.file_hash(lp)
    return rows,a,loc,mapping,dest


class Scorer:
    def __init__(self,entries,source):
        self.entries=entries;self.transform=paired.load_transform(source,DEVICE)
        self.scalar=[];self.multi={}
        ww=[];bb=[]
        for i,e in enumerate(entries):
            s=spec_for(e)
            if isinstance(s,dict):
                self.multi[i]=(torch.tensor(s['s_n'],device=DEVICE),torch.tensor(s['s_a'],device=DEVICE))
            else:self.scalar.append(i);ww.append(s.weight());bb.append(s.bias())
        self.w=torch.tensor(np.stack(ww,axis=1),device=DEVICE) if ww else None
        self.bias=torch.tensor(bb,device=DEVICE)
    @torch.inference_mode()
    def __call__(self,raw):
        z=paired.transformed(raw,self.transform,DEVICE)
        q=torch.empty((len(z),len(self.entries)),device=DEVICE,dtype=torch.float32)
        if self.scalar:q[:,self.scalar]=z@self.w+self.bias
        for i,(n,a) in self.multi.items():q[:,i]=cmab.multi_state_logits(z,n,a)
        q=q.view(5,37,37,len(self.entries)).permute(3,0,1,2)
        answer={}
        for i,e in enumerate(self.entries):
            if not e['lrp']:answer[e['output_id']]=q[i,0].cpu().numpy();continue
            g=F.interpolate(q[i:i+1,0:1],size=(74,74),mode='bilinear',align_corners=True)[0,0]
            local=torch.empty_like(g)
            local[:37,:37]=q[i,1];local[:37,37:]=q[i,2]
            local[37:,:37]=q[i,3];local[37:,37:]=q[i,4]
            answer[e['output_id']]=(.5*g+.5*local).cpu().numpy()
        assert all(np.isfinite(a).all() for a in answer.values())
        return answer


def read_reused(ref,index,cache):
    path=ref['path']
    if path not in cache:cache[path]=np.load(path,mmap_mode='r')
    return np.asarray(cache[path][ref['index'][index],ref['column']],np.float32)


def smoke(config,models,plans):
    encoder=ResolutionEstablishedEncoder();checks=[];cache={}
    for target in TARGETS:
        rows,a,loc,mapping,dest=target_features(target)
        scorer=Scorer(plans[target]['outputs'],source_for(target))
        ids=[next(i for i,r in enumerate(rows) if int(r['label'])==label) for label in (0,1)]
        for i in ids:
            j=mapping[i];r=rows[i]
            freshg=encoder.extract(r['img_path'],False);freshl=encoder.extract(r['img_path'])
            gd=abs(freshg-np.asarray(a[j],np.float32))
            assert gd.mean()<2e-4 and gd.max()<.005,(target,'global cache failed',gd.max())
            local=np.asarray(loc[j],np.float32) if loc is not None else freshl
            ld=abs(freshl-local);assert ld.mean()<2e-4 and ld.max()<.005
            maps=scorer(np.concatenate((np.asarray(a[j],np.float32)[None],local)))
            maxerr=0.
            for oid,ref in plans[target]['reuse'].items():
                error=float(abs(maps[oid]-read_reused(ref,i,cache)).max());maxerr=max(maxerr,error)
                assert error<2e-6,(target,oid,error)
            mask=official.native_mask(r)
            first=maps[plans[target]['outputs'][0]['output_id']]
            resized=official.resize(first,mask.shape)
            metrics=official.weighted_pixel_metrics(mask.reshape(-1),resized.reshape(-1))
            direct=float(paired.lzp.safe_ap(mask.reshape(-1),resized.reshape(-1))) if mask.any() and not mask.all() else None
            if direct is not None:assert abs(direct-metrics['P_AP'])<1e-10
            checks.append(dict(target=target,sample_id=r['sample_id'],raw_grid=[37,37],
                global_error=float(gd.max()),local_error=float(ld.max()),reuse_map_error=maxerr,
                native_GT_shape=list(mask.shape),direct_AP=direct,official_AP=metrics['P_AP'],status='PASS'))
    after=encoder.checksum();assert after==encoder.before
    csv(OUT/'SMOKE_CHECKS.csv',checks)
    lock(OUT/'SMOKE_COMPLETE.json',dict(config_hash=paired.json_hash(config),status='PASS',
         visual_before=encoder.before,visual_after=after,samples=12,checks=checks))
    del encoder,cache;gc.collect();torch.cuda.empty_cache();status('SMOKE_PASSED',samples=12)


def chunk_write(path,arrays):
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,**arrays)
    temp.replace(path)


def predict(target,plan,encoder=None):
    rows,a,loc,mapping,dest=target_features(target)
    selected=[e for e in plan['outputs'] if e['output_id'] in plan['new_outputs']]
    scorer=Scorer(selected,source_for(target));chunk_size=25
    lock(dest/'MODELS.json',plan)
    if loc is None and encoder is None:encoder=ResolutionEstablishedEncoder()
    begin=time.time();processed=0
    for start in range(0,len(rows),chunk_size):
        stop=min(start+chunk_size,len(rows));path=dest/f'chunk_{start:05d}.npz';cp=path.with_suffix('.json')
        if cp.exists():
            saved=read(cp);assert saved['sha256']==paired.file_hash(path) and saved['outputs']==plan['new_outputs'];continue
        arrays={f'm{k}':np.empty((stop-start,74 if e['lrp'] else 37,74 if e['lrp'] else 37),np.float32) for k,e in enumerate(selected)}
        for pos in range(start,stop):
            ix=mapping[pos]
            ll=np.asarray(loc[ix],np.float32) if loc is not None else encoder.extract(rows[pos]['img_path'])
            raw=np.concatenate((np.asarray(a[ix],np.float32)[None],ll))
            scores=scorer(raw)
            for k,e in enumerate(selected):arrays[f'm{k}'][pos-start]=scores[e['output_id']]
        chunk_write(path,arrays)
        js(cp,dict(start=start,stop=stop,outputs=plan['new_outputs'],sha256=paired.file_hash(path),dtype='lossless float32',chunk_size=chunk_size))
        processed+=stop-start;elapsed=time.time()-begin
        js(dest/'PROGRESS.json',dict(done=stop,total=len(rows),seconds_this_session=elapsed,
            eta_seconds=elapsed/max(processed,1)*(len(rows)-stop)))
        status('PREDICTING',target=target,done=stop,total=len(rows),new_outputs=len(selected),
               eta_minutes=round(elapsed/max(processed,1)*(len(rows)-stop)/60,1))
    js(dest/'COMPLETE.json',dict(n=len(rows),plan_hash=paired.json_hash(plan),chunk_size=chunk_size,
                              done=True,new_outputs=len(selected),reused_outputs=len(plan['reuse'])))
    return encoder


class MapReader:
    def __init__(self,target,plan):
        self.target=target;self.plan=plan;self.cache={};self.chunk=None;self.pack=None
        self.current_key=None;self.current_array=None
    def get(self,oid,index):
        if oid in self.plan['reuse']:return read_reused(self.plan['reuse'][oid],index,self.cache)
        start=index//25*25
        if self.chunk!=start:
            if self.pack is not None:self.pack.close()
            self.pack=np.load(OUT/'predictions'/self.target/f'chunk_{start:05d}.npz',allow_pickle=False)
            self.chunk=start;self.current_key=None;self.current_array=None
        k=self.plan['new_outputs'].index(oid)
        if self.current_key!=k:
            self.current_array=self.pack[f'm{k}'];self.current_key=k
        return self.current_array[index-start]
    def close(self):
        if self.pack is not None:self.pack.close()
        self.cache.clear()


def native_geometry_result(target,entry,ref):
    if ref.get('kind')!='geometry_map':return None
    cp=GEOM/'evaluations'/target/f"PLACEHOLDER__{ref['geometry_variant']}_seed{entry['seed']}.json"
    return cp


def evaluate(target,config,plan):
    assert (OUT/'predictions'/target/'COMPLETE.json').exists()
    rows=canonical(target);reader=MapReader(target,plan);results=[]
    scratch=OUT/'scratch';scratch.mkdir(exist_ok=True)
    for category in sorted({r['category'] for r in rows}):
        ids=[i for i,r in enumerate(rows) if r['category']==category]
        for entry in plan['outputs']:
            oid=entry['output_id'];rp=OUT/'evaluations'/target/f'{category}__{oid}.json'
            if rp.exists():
                r=read(rp);assert r['checkpoint_id']==entry['checkpoint_sha256'] and r['evaluator_id']==config['evaluator_id'];results.append(r);continue
            ref=plan['reuse'].get(oid,{})
            provenance='NEW_NATIVE_GT_EVALUATION'
            if ref.get('kind')=='geometry_map':
                oldpath=GEOM/'evaluations'/target/f"{category}__{ref['geometry_variant']}_seed{entry['seed']}.json"
                old=read(oldpath);assert old['evaluator_hash']==config['evaluator_id']
                value=old['P_AP'];n=old['pixels'];provenance='REUSED_EXACT_NATIVE_GT_GEOMETRY_RESULT; parameter-identical model verified'
            else:
                n=sum(int(rows[i]['gt_height'])*int(rows[i]['gt_width']) for i in ids)
                assert shutil.disk_usage(OUT).free>n*5+200_000_000,'Insufficient new scratch capacity; no old cache deletion'
                yp=scratch/'labels.npy';sp=scratch/'scores.npy'
                y=np.lib.format.open_memmap(yp,mode='w+',dtype=np.bool_,shape=(n,))
                s=np.lib.format.open_memmap(sp,mode='w+',dtype=np.float32,shape=(n,));offset=0
                for i in ids:
                    mask=official.native_mask(rows[i]);score=official.resize(reader.get(oid,i),mask.shape)
                    end=offset+mask.size;y[offset:end]=mask.reshape(-1);s[offset:end]=score.reshape(-1);offset=end
                y.flush();s.flush();assert offset==n
                status('EVALUATING_NATIVE_AP',target=target,category=category,model=oid,pixels=n)
                value=official.weighted_pixel_metrics(y,s)['P_AP']
                del y,s;gc.collect();yp.unlink();sp.unlink()
            assert value is not None and math.isfinite(value),(target,category,oid,'No finite native AP')
            r=dict(source=entry['source'],target=target,category=category,seed=entry['seed'],variant=entry['variant'],
                ntr=entry['ntr'],lrp=entry['lrp'],model_id=entry['model_id'],output_id=oid,p_ap=value,
                checkpoint=entry['checkpoint'],checkpoint_id=entry['checkpoint_sha256'],evaluator_id=config['evaluator_id'],
                pixels=n,images=len(ids),provenance=provenance)
            js(rp,r);results.append(r)
    reader.close();csv(OUT/'evaluations'/f'{target}_CATEGORIES.csv',results)
    return results


def aggregate(config,models,plans):
    values={};registry=[]
    for target in TARGETS:
        path=OUT/'evaluations'/f'{target}_CATEGORIES.csv'
        if not path.exists():return False
        rr=rcsv(path)
        for e in plans[target]['outputs']:
            r=[x for x in rr if x['output_id']==e['output_id']]
            assert len(r)==len({x['category'] for x in canonical(target)})
            v=float(np.mean([float(x['p_ap']) for x in r]))
            record=dict(source=e['source'],target=target,seed=e['seed'],variant=e['variant'],ntr=e['ntr'],lrp=e['lrp'],
                checkpoint_id=e['checkpoint_sha256'],evaluator_id=config['evaluator_id'],p_ap=v,checkpoint=e['checkpoint'],
                output_id=e['output_id'],feature_hash=e['feature_hash'],source_pool_hash=e['source_pool_hash'],
                batch_hash=e['batch_hash'],total_updates=e['total_updates'],category_count=len(r))
            values[(target,e['seed'],e['variant'],e['lrp'])]=record;registry.append(record)
    assert len(values)==177
    raw=[]
    definitions={
        'A_DecisionParameterization': [('LinearHead','LinearHead',True),('NormalizedLinearHead','NormalizedLinearHead',True),('CES','CES_NTR',True)],
        'B_ReferenceCenter':[('Hybrid-CES','CES_NTR',True),('E1-Center-CES','E1_Center',True)],
        'C_NTRxLRP':[('CES','CES_noNTR',False),('CES+NTR','CES_NTR',False),('CES+LRP','CES_noNTR',True),('CES+NTR+LRP','CES_NTR',True)],
        'D_HigherStates':[('K2 base (30)','CES_NTR',True),('K2 continuation (60)','CES_NTR_continue',True),('K3 configuration (60)','K3_config',True),('K5 configuration (60)','K5_config',True)]}
    for exp,defs in definitions.items():
        for target in TARGETS:
            if exp.startswith('B') and target=='VisA':continue
            for seed in SEEDS:
                for label,var,lrp in defs:
                    raw.append(dict(experiment=exp,**values[(target,seed,var,lrp)],configuration=label))
    assert len(raw)==228
    csv(OUT/'UNIFIED_ABLATION_RAW.csv',raw)
    csv(OUT/'UNIFIED_CHECKPOINT_REGISTRY.csv',registry)
    from experiments.unified_ablation_final_reporting import generate
    generate(OUT,raw,definitions)
    # Recheck immutable small source/model identities. No old file is rewritten.
    for e in models:assert paired.file_hash(e['checkpoint'])==e['checkpoint_sha256']
    for p in config['inputs']:
        if p['bytes']<20_000_000:assert paired.file_hash(p['path'])==p['sha256']
    js(OUT/'OUTCOME.json',dict(status='COMPLETE_FROZEN',unique_evaluations=177,logical_rows=228,
        reused_models=21,new_training_runs=24,independent_source_training_seeds=[0,1,2],
        evaluator_id=config['evaluator_id'],no_paper_or_historical_result_modified=True,
        no_next_experiment=True))
    manifest=[ident(p) for p in OUT.rglob('*') if p.is_file() and p.suffix in ('.json','.md','.csv','.py','.pt') and p.name!='FINAL_FILE_HASHES.csv']
    csv(OUT/'FINAL_FILE_HASHES.csv',manifest)
    status('COMPLETE_FROZEN',raw_rows=228,unique_evaluations=177)
    return True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('preflight','train','smoke','run','summarize'),default='run')
    args=parser.parse_args();setup()
    if args.stage=='preflight':preflight();return
    config=read(OUT/'UNIFIED_CONFIG.json')
    assert config['code'][0]['sha256']==paired.file_hash(__file__),'Locked scientific runner changed'
    if args.stage=='train':train(config,read(OUT/'REUSED_MODELS.json'));return
    models=read(OUT/'SOURCE_TRAINING_COMPLETE.json')['models']
    for e in models:assert paired.file_hash(e['checkpoint'])==e['checkpoint_sha256']
    plans=read(OUT/'PREDICTION_PLAN.json') if (OUT/'PREDICTION_PLAN.json').exists() else build_prediction_plan(config,models)
    if args.stage=='smoke':smoke(config,models,plans);return
    if args.stage=='summarize':assert aggregate(config,models,plans);return
    if not (OUT/'SMOKE_COMPLETE.json').exists():smoke(config,models,plans)
    assert read(OUT/'SMOKE_COMPLETE.json')['config_hash']==paired.json_hash(config)
    encoder=None
    for target in TARGETS:
        encoder=predict(target,plans[target],encoder)
        evaluate(target,config,plans[target])
    if encoder is not None:
        after=encoder.checksum();assert encoder.before==after
        js(OUT/'FROZEN_VISUAL_CHECKSUM.json',dict(before=encoder.before,after=after,unchanged=True))
        del encoder;gc.collect();torch.cuda.empty_cache()
    assert aggregate(config,models,plans)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OUT.exists():js(OUT/'LAST_ERROR.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
