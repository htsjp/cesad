"""Aggregate only newly unified native-GT result rows; no metric recomputation."""
import csv
import json
from pathlib import Path
import numpy as np


def generate(out,raw,definitions):
    out=Path(out)
    def md(name,text):
        p=out/name;tmp=p.with_suffix('.tmp');tmp.write_text(text.strip()+'\n',encoding='utf-8');tmp.replace(p)
    def csvout(name,rows):
        p=out/name;tmp=p.with_suffix('.tmp')
        keys=list(dict.fromkeys(k for r in rows for k in r))
        with tmp.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
        tmp.replace(p)
    targets=('VisA','MVTec','BTAD','MPDD','DTD-Synthetic','DAGM')
    def series(exp,target,label):
        return np.array([np.mean([r['p_ap'] for r in raw if r['experiment']==exp and r['seed']==seed and
                       r['configuration']==label and (target=='MEAN' or r['target']==target)]) for seed in (0,1,2)])
    fmt=lambda a:f'{a.mean():.2f} ± {a.std(ddof=1):.2f}'
    titles=dict(A_DecisionParameterization='Decision Parameterization',B_ReferenceCenter='Source-Derived Reference Center',
                C_NTRxLRP='NTR × LRP',D_HigherStates='Higher-State Configurations')
    tables=[];summaries=[]
    for exp,defs in definitions.items():
        labels=[d[0] for d in defs];tt=targets[1:] if exp.startswith('B') else targets
        table=[f'## {titles[exp]}','', '| Source → Target | '+' | '.join(labels)+' |', '|---|'+'---:|'*len(labels)]
        for target in (*tt,'MEAN'):
            names='Five-target mean' if target=='MEAN' and exp.startswith('B') else ('Six-transfer mean' if target=='MEAN' else ('MVTec → VisA' if target=='VisA' else 'VisA → '+target))
            vv=[series(exp,target,label) for label in labels]
            assert all(np.isfinite(x).all() and len(x)==3 for x in vv)
            table.append('| '+names+' | '+' | '.join(fmt(x) for x in vv)+' |')
            for label,v in zip(labels,vv):
                summaries.append(dict(experiment=exp,target=target,configuration=label,mean=float(v.mean()),sample_SD=float(v.std(ddof=1)),
                    seed0=float(v[0]),seed1=float(v[1]),seed2=float(v[2]),n_seeds=3))
        table+=['','P-AP (%), paired training seeds 0/1/2; sample SD (ddof=1). The mean row averages transfers within each seed before estimating SD.']
        if exp.startswith('D'):
            table+=['K2 base uses 30 updates. K2 continuation and K3/K5 use 30 base + 30 additional updates. K3/K5 retain free-state/log-sum-exp/diversity differences; this is not a count-only causal ablation.']
        if exp.startswith('A'):
            table+=['Normalized Linear uses its source-fixed initial logit scale c0, unit direction, and no bias; c0 is not trained. All initial effective logits and realized minibatches are matched.']
        tables.append('\n'.join(table))
    md('UNIFIED_ABLATION_TABLES.md','# Unified internal ablations\n\nNative paired GT, equal category-macro AP, no target calibration.\n\n'+'\n\n'.join(tables))
    csvout('UNIFIED_ABLATION_SUMMARY.csv',summaries)
    comparisons={
        'A_DecisionParameterization':[('CES−Linear','CES','LinearHead'),('CES−Normalized','CES','NormalizedLinearHead')],
        'B_ReferenceCenter':[('Hybrid−E1Center','Hybrid-CES','E1-Center-CES')],
        'C_NTRxLRP':[('NTR at global','CES+NTR','CES'),('NTR at LRP','CES+NTR+LRP','CES+LRP'),
                     ('LRP without NTR','CES+LRP','CES'),('LRP with NTR','CES+NTR+LRP','CES+NTR')],
        'D_HigherStates':[('Continuation−Base','K2 continuation (60)','K2 base (30)'),
                         ('K3−Base','K3 configuration (60)','K2 base (30)'),('K5−Base','K5 configuration (60)','K2 base (30)'),
                         ('K3−Continuation','K3 configuration (60)','K2 continuation (60)'),
                         ('K5−Continuation','K5 configuration (60)','K2 continuation (60)')]}
    paired=[];overview=[];english=[]
    for exp,comparisons_exp in comparisons.items():
        tt=targets[1:] if exp.startswith('B') else targets
        overview+=['',f'## {titles[exp]}','']
        for name,left,right in comparisons_exp:
            d=series(exp,'MEAN',left)-series(exp,'MEAN',right)
            target_delta={t:float((series(exp,t,left)-series(exp,t,right)).mean()) for t in tt}
            wins=sum(v>0 for v in target_delta.values());losses=sum(v<0 for v in target_delta.values())
            overview.append(f'- {name}：总体配对差 {d.mean():+.4f} ± {d.std(ddof=1):.4f} 点；按目标均值 {wins}/{len(tt)} 正增益、{losses}/{len(tt)} 负增益。')
            overview.append('  各目标：'+'；'.join(f'{t} {v:+.4f}' for t,v in target_delta.items())+'。')
            english.append(f'{name}: the mean paired change is {d.mean():+.2f} ± {d.std(ddof=1):.2f} percentage points, with positive target-mean changes on {wins}/{len(tt)} transfers.')
            for target in (*tt,'MEAN'):
                lv=series(exp,target,left);rv=series(exp,target,right);delta=lv-rv
                for seed in (0,1,2):
                    paired.append(dict(experiment=exp,target=target,comparison=name,level='seed',seed=seed,
                        left_p_ap=float(lv[seed]),right_p_ap=float(rv[seed]),delta=float(delta[seed]),mean='',sample_SD=''))
                paired.append(dict(experiment=exp,target=target,comparison=name,level='summary',seed='ALL',
                    left_p_ap=float(lv.mean()),right_p_ap=float(rv.mean()),delta=float(delta.mean()),
                    mean=float(delta.mean()),sample_SD=float(delta.std(ddof=1)),positive_seeds=int((delta>0).sum()),n_seeds=3))
        if exp=='C_NTRxLRP':
            for target in (*tt,'MEAN'):
                d=(series(exp,target,'CES+NTR+LRP')-series(exp,target,'CES+NTR'))-(series(exp,target,'CES+LRP')-series(exp,target,'CES'))
                paired.append(dict(experiment=exp,target=target,comparison='NTR×LRP additive interaction',level='summary',seed='ALL',
                    delta=float(d.mean()),mean=float(d.mean()),sample_SD=float(d.std(ddof=1)),
                    seed0_delta=float(d[0]),seed1_delta=float(d[1]),seed2_delta=float(d[2]),n_seeds=3))
    csvout('UNIFIED_PAIRED_STATISTICS.csv',paired)
    report='''# CESAD 最终统一消融报告

## 核心结论与控制范围

以下判断完全由本次统一native-GT原始结果生成，保留全部正负变化；未根据目标选择checkpoint、LR或纳入的结果。所有45个训练权重先于本轮目标评分封存。六迁移228行逻辑结果对应177个唯一model/inference评价，同一Full参考跨表复用而不重复训练。

统计重复单位是三个source-training seed；五/六个目标不是五/六次独立源训练。mean±sample SD先每seed聚合迁移再计算，ddof=1。不把这些SD解释为目标域总体不确定性，也不自动声称统计显著。
'''
    report+='\n'.join(overview)+'\n\n## 全部正式表格\n\n'+'\n\n'.join(tables)
    report+='''

## 身份统一与历史差异

复用18个StepB共同LR主模型及3个E1-center模型。3个旧Hybrid只是与StepB CES参数逐元素相同的缓存别名。新训练6个no-NTR、6个canonical K2 continuation和12个K3/K5配置。旧no-NTR与高状态的池/批次不匹配，未作为严格配对复用。Global-only与LRP使用同一个已训练head。

本轮采用主表原始配对GT和exact score-tie AP，旧StepB的518 loader-GT结果不得直接复制进来。旧中心实验的native-GT结果仅在模型参数、源变换、目标物理ID、预测及evaluator身份均验证后复用。旧六目标SOTA表本轮不重训、不覆盖。不同训练checkpoint导致本轮Full不必等于旧SOTA Full，不能为了使数值一致改数据。

## 论证边界

参数化对照检验的是相同冻结特征空间中的优化/几何约束，不是更强的非线性表示能力。中心对照只改变归一化前m，不能归因为E2 adapter的独立收益。NTR与LRP的配对变化按目标完整披露。高状态与额外预算K2并列，仍改变状态自由度、LSE与cosine regularization，不证明两状态的普遍最优性。

## 冻结结论

完整性、配对和评价身份检查通过后，可冻结本次统一内部消融证据。公平性上的旧不匹配资产已经排除；高状态保留的有意多因素差异已明确标注，不能称为纯K消融。没有自动修订Word、主表、旧CSV或图像，也不开展后续探索。
'''
    md('UNIFIED_ABLATION_FINAL_REPORT.md',report)
    plan='''# Paper replacement plan (suggestions only; no Word changes)

## Recommended scientific order

Source-derived geometry → constrained CES learning → normal-tail training and local observation → unseen-target discrimination.

## Table 3: Decision Parameterization

Replace the historical four-transfer loader-GT table with unified six-transfer native-GT results. Preserve the matched initialization, c0 convention, LR and actual batch hashes. The source-selected LR analysis remains a separately scoped historical diagnostic; do not substitute it for the primary shared-LR comparison.

## Table 4: Source-Derived Reference Center

Use five VisA-source transfers only. Hybrid is the identical shared Full reference used by the other tables. Do not invent an MVTec E1/E2 center comparison or interpret this as an E2 adapter ablation.

## Table 5: NTR × LRP

Use the paired three-seed 2×2 table, with full trained no-NTR models and the same trained head used for global and LRP inference. Retain target-specific negative effects. Do not insert legacy fallback or use a different SOTA checkpoint to increase a component score.

## Table 6: Higher-State Configurations

Include K2 base30, K2 continuation60, K3 configuration60 and K5 configuration60. The extra-budget control distinguishes a continuation effect from the richer configurations; free-state geometry, aggregation and regularization remain deliberate differences. State that the source pools and extra batches have been unified rather than calling this an exact replay of older CMAB experiments.

## Historical scores

Historical39.54, StepB39.11, the five-target center mean54.22 and any six-transfer mean use different aggregation sets and/or GT/checkpoint identities. Do not transfer one mean to another table. New scores are authoritative only for this explicitly locked internal protocol. Six-target SOTA main results remain untouched.

## Result-grounded English draft

All values below are generated from unrounded paired results, not selected for a desired narrative.

'''
    plan+='\n\n'.join(english)
    plan+='''

These observations concern the tested source-defined feature spaces and optimization configurations. Any target on which a comparator is stronger must remain in the analysis. The capacity comparison concerns complete higher-state configurations rather than universal optimality of two states. No claim of stronger nonlinear expressivity or independently established adapter mechanism should be added.
'''
    md('UNIFIED_PAPER_REPLACEMENT_PLAN.md',plan)
