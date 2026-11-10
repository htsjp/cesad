"""Evaluate the SAME matched-total30 checkpoints on the four remaining targets.

NO training. D stores metadata/tables; independent E worker stores new lossless
maps and native-GT scratch to avoid exhausting D. Historical outputs stay intact.
"""
import argparse
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import phase_ces_unified_ablation_final as u

OLD=ROOT/'results/phase_ces_capacity_matched30/formal'
META=ROOT/'results/phase_ces_capacity_matched30_extended/formal'
WORK=Path('E:/MRAD_capacity_matched30_extended_cache/20261026')
TARGETS=('BTAD','MPDD','DTD-Synthetic','DAGM')
ALL_TARGETS=('VisA','MVTec',*TARGETS)


def status(phase,**kw):u.paired.status(META,phase,**kw)
def js(p,v):u.js(p,v)
def read(p):return u.read(p)
def csv(p,v):u.csv(p,v)
def md(name,text):(META/name).write_text(text.strip()+'\n',encoding='utf-8')


def bind_worker():
    u.OUT=WORK
    u.TARGETS=TARGETS
    WORK.mkdir(parents=True,exist_ok=True)
    META.mkdir(parents=True,exist_ok=True)
    u.setup()


def plans_for(models):
    plans={}
    for target in TARGETS:
        entries=[dict(e,output_id=e['model_id']+'__LRP',lrp=True,ntr=True) for e in models
                 if e['source']=='Full VisA']
        assert len(entries)==9
        refs,files=u.reuse_predictions(target,entries)
        assert len(refs)==3
        for path,h in files.items():assert u.paired.file_hash(path)==h
        plans[target]=dict(target=target,outputs=entries,reuse=refs,
            new_outputs=[e['output_id'] for e in entries if e['output_id'] not in refs],n=len(u.canonical(target)))
    return plans


def prepare():
    bind_worker()
    assert read(OLD/'OUTCOME.json')['status']=='COMPLETE'
    seal=read(OLD/'SOURCE_TRAINING_COMPLETE.json');models=seal['models']
    assert len(models)==18
    poolproof=[]
    for seed in (0,1,2):
        folder,z,y,b,probe,meta=u.pool('Full VisA',seed)
        poolproof.append(dict(seed=seed,feature_hash=meta['feature_array_hash'],batch_hash=meta['batch_stream_hash'],
                              label_hash=meta['labels_hash'],manifest_hash=meta['manifest_hash']))
    for e in models:
        assert u.paired.file_hash(e['checkpoint'])==e['checkpoint_sha256']
        p=torch.load(e['checkpoint'],map_location='cpu',weights_only=False)
        assert p['realized_updates']==30
        if e['variant']!='CES_NTR':
            assert p['total_updates']==30
            assert p['initialization_stage'].startswith('PRE_FINAL_TRAINING')
    config=dict(task='Extend existing equal-total30 states; evaluation only',targets=list(TARGETS),
        original_targets=['VisA','MVTec'],seeds=[0,1,2],source_for_new_targets='Full VisA',
        total_head_updates=30,epochs=10,new_training_runs=0,models=models,
        original_config=u.ident(OLD/'CONFIG.json'),original_raw=u.ident(OLD/'MATCHED30_RAW.csv'),
        original_training_seal=u.ident(OLD/'SOURCE_TRAINING_COMPLETE.json'),source_pool_proof=poolproof,
        evaluator_id=u.official.evaluator_hash(),evaluation='paired original GT; exact tie category-macro pixel AP',
        inference='same fixed E2/PC1, same CLIP/DAPM features, same2x2 LRP,0.5global+0.5local74',
        heavy_cache=str(WORK),new_target_model_evaluations=36,new_score_outputs=24,
        new_CLIP_inference='only necessary local crop extraction where no reusable feature cache exists; shared across all six new high-state heads',
        code=[u.ident(__file__),u.ident(u.__file__),u.ident(u.official.__file__)])
    u.lock(META/'CONFIG.json',config)
    js(META/'HEAVY_CACHE_LOCATION.json',dict(path=str(WORK),reason='D only~2.54GB free; keep heavy predictions/scratch on E',
                                           metadata=str(META),old_files_deleted=False))
    plans=plans_for(models)
    u.lock(META/'PREDICTION_PLAN.json',plans)
    assert shutil.disk_usage(WORK).free>4_000_000_000,'Insufficient E space; no automatic old-cache cleanup'
    md('PRE_RUN_VALIDATION.md','''# 同30步权重：扩展目标评价

只使用已完成的matched30 checkpoint，不训练任何head或上游模块。新增四个VisA-source目标：BTAD、MPDD、DTD-Synthetic、DAGM。每个K2/K3/K5使用原来三个种子、同一冻结源变换和Fixed LRP。K3/K5仍是从未进行最终CES优化的source geometry出发训练的总30步模型，不改用旧60步权重。

K2的native-GT预测/评价按相同checkpoint或严格相同有效参数、源变换、样本ID和评价器复用。K3/K5只补必要的新目标评分，24个新model-target输出。所有模型共享每张图的frozen特征；缺失DTD/DAGM局部特征时仅做必要的冻结CLIP提取，不重新训练。

主表同一原始配对GT、bilinear align_corners=True、exact score ties和category-macro AP。原有两迁移结果及旧表只读；完成后独立汇总为六迁移matched30表。E盘保存新增无损预测及scratch，项目目录保存配置、原始指标、表格和报告。
''')
    snapshot=META/'code_snapshot';snapshot.mkdir(exist_ok=True)
    for item in config['code']:
        dest=snapshot/Path(item['path']).name
        if not dest.exists():shutil.copyfile(item['path'],dest)
        assert u.paired.file_hash(dest)==item['sha256']
    status('PREFLIGHT_PASSED',new_training=0,new_targets=list(TARGETS),new_prediction_outputs=24)


def aggregate(config,plans):
    original=u.rcsv(OLD/'MATCHED30_RAW.csv')
    assert len(original)==18
    rows=[]
    for r in original:
        rows.append(dict(source=r['source'],target=r['target'],seed=int(r['seed']),K=int(r['K']),total_updates=30,
                         p_ap=float(r['p_ap']),checkpoint=r['checkpoint'],checkpoint_id=r['checkpoint_id'],
                         evaluator_id=r['evaluator_id'],provenance='unchanged original two-transfer matched30 result'))
    for target in TARGETS:
        path=WORK/'evaluations'/f'{target}_CATEGORIES.csv'
        if not path.exists():return False
        rr=u.rcsv(path);csv(META/f'{target}_PER_CATEGORY.csv',rr)
        for e in plans[target]['outputs']:
            cells=[r for r in rr if r['output_id']==e['output_id']]
            assert len(cells)==len({r['category'] for r in u.canonical(target)})
            assert {r['checkpoint_id'] for r in cells}=={e['checkpoint_sha256']}
            K=2 if e['variant']=='CES_NTR' else int(e['variant'][1])
            rows.append(dict(source=e['source'],target=target,seed=e['seed'],K=K,total_updates=30,
                             p_ap=float(np.mean([float(r['p_ap']) for r in cells])),checkpoint=e['checkpoint'],
                             checkpoint_id=e['checkpoint_sha256'],evaluator_id=config['evaluator_id'],
                             provenance='same matched30 checkpoint evaluated on additional target'))
    assert len(rows)==len({(r['target'],r['seed'],r['K']) for r in rows})==54
    csv(META/'MATCHED30_SIX_TARGET_RAW.csv',rows)
    summaries=[];paired=[]
    lines=['| Source → Target | K=2 (30) | K=3 (30) | K=5 (30) |','|---|---:|---:|---:|']
    for target in (*ALL_TARGETS,'SIX_TRANSFER_MEAN'):
        values={K:np.array([np.mean([r['p_ap'] for r in rows if r['K']==K and r['seed']==seed and
                                  (target=='SIX_TRANSFER_MEAN' or r['target']==target)]) for seed in (0,1,2)]) for K in (2,3,5)}
        label='Six-transfer mean' if target=='SIX_TRANSFER_MEAN' else ('MVTec → VisA' if target=='VisA' else 'VisA → '+target)
        lines.append('| '+label+' | '+' | '.join(f'{values[K].mean():.2f} ± {values[K].std(ddof=1):.2f}' for K in (2,3,5))+' |')
        for K,a in values.items():
            summaries.append(dict(target=target,K=K,mean=float(a.mean()),sample_SD=float(a.std(ddof=1)),
                                  seed0=float(a[0]),seed1=float(a[1]),seed2=float(a[2]),total_updates=30))
        for K in (3,5):
            d=values[K]-values[2]
            paired.append(dict(target=target,comparison=f'K{K}−K2',mean_delta=float(d.mean()),sample_SD=float(d.std(ddof=1)),
                               delta_seed0=float(d[0]),delta_seed1=float(d[1]),delta_seed2=float(d[2])))
    csv(META/'MATCHED30_SIX_TARGET_SUMMARY.csv',summaries);csv(META/'MATCHED30_SIX_TARGET_PAIRED.csv',paired)
    table='\n'.join(lines)
    md('MATCHED30_SIX_TARGET_TABLE.md','# Higher-state configurations: equal TOTAL30 updates\n\n'+table+'''

Native-GT category-macro P-AP (%), seeds0/1/2, mean ± sample SD(ddof=1). The mean row first averages six transfers within each seed. Every K has30 final-head updates, not30 versus60. Higher-state configurations retain free-state/LSE/diversity differences; this is not a count-only causal comparison.
''')
    md('REPORT.md','''# 同总预算30步：六目标扩展

本次只复用上轮matched30权重，新增训练为零。只补BTAD、MPDD、DTD-Synthetic、DAGM，原MVTec↔VisA指标不修改。下面所有K2/K3/K5均为总30次最终head更新（10epochs），三种子。没有根据新增目标结果重选checkpoint、参数或训练预算。

'''+table+'''

原始54行见MATCHED30_SIX_TARGET_RAW.csv，全部未舍入配对差值见MATCHED30_SIX_TARGET_PAIRED.csv。高状态的LSE/diversity和自由state参数化仍保留，与canonical CES不同；不能据此宣称两状态普遍最优。无论排名如何，完整保留所有目标和种子。不覆盖旧实验、主表或论文，不开始新一轮优化。
''')
    for e in config['models']:assert u.paired.file_hash(e['checkpoint'])==e['checkpoint_sha256']
    js(META/'OUTCOME.json',dict(status='COMPLETE',all_rows=54,new_training_runs=0,total_updates_all_K=30,
                              targets=list(ALL_TARGETS),heavy_cache=str(WORK),old_results_unchanged=True))
    status('COMPLETE',raw_rows=54,new_training=0);return True


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('prepare','smoke','run'),default='run')
    args=parser.parse_args();bind_worker()
    if args.stage=='prepare':prepare();return
    config=read(META/'CONFIG.json');assert u.paired.file_hash(__file__)==config['code'][0]['sha256']
    plans=read(META/'PREDICTION_PLAN.json');models=config['models']
    for e in models:assert u.paired.file_hash(e['checkpoint'])==e['checkpoint_sha256']
    if args.stage=='smoke' or not (WORK/'SMOKE_COMPLETE.json').exists():
        status('SMOKE_RUNNING',targets=list(TARGETS));u.smoke(config,models,plans)
        js(META/'SMOKE_COMPLETE.json',read(WORK/'SMOKE_COMPLETE.json'))
        if args.stage=='smoke':return
    assert read(WORK/'SMOKE_COMPLETE.json')['config_hash']==u.paired.json_hash(config)
    encoder=None
    for target in TARGETS:
        status('TARGET_RUNNING',target=target)
        encoder=u.predict(target,plans[target],encoder)
        u.evaluate(target,config,plans[target])
    if encoder is not None:
        after=encoder.checksum();assert after==encoder.before
        js(META/'FROZEN_VISUAL_CHECKSUM.json',dict(before=encoder.before,after=after,unchanged=True))
    assert aggregate(config,plans)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        META.mkdir(parents=True,exist_ok=True)
        js(META/'LAST_ERROR.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
