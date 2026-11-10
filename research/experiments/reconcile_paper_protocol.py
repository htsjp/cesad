"""Extract existing evidence; never train, infer, or replace historical results."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/paper_protocol_reconciliation'
PAIRS = [('Full MVTec', 'VisA'), ('Full VisA', 'MVTec'),
         ('Full VisA', 'BTAD'), ('Full VisA', 'MPDD')]
CAP = ROOT / 'results/paper_assets/final_evidence'
CMAB = ROOT / 'results/phase_cmab_compact_multi_anomaly_basis/formal'
HEAD = ROOT / 'results/phase_ces_linear_head_control'
COMP = ROOT / 'results/ces_ntr_visualization/corrected_metric_unification'


def read(path):
    return pd.read_csv(path)


def save(name, rows):
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(OUT / name, index=False, float_format='%.6f', na_rep='MISSING')


def line(path, token):
    path = ROOT / path
    for i, s in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if token in s:
            return f'{path}:{i}'
    return f'{path}:UNRESOLVED'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old = read(CAP / 'STATE_CAPACITY_METRICS.csv')
    cmab = read(CMAB / 'RAW_METRICS.csv')
    cat = read(CMAB / 'PER_CATEGORY_STANDARD_METRICS.csv')
    curve = read(CMAB / 'TRAINING_CURVES.csv')
    variants = {2: 'E0_K1_Baseline', 3: 'E1_K2_CMAB', 5: 'E2_K4_CMAB'}
    tables, provenance = [], []
    for source, target in PAIRS:
        entry = {'source': source, 'target': target}
        for k, var in variants.items():
            rows = old[(old.source == source) & (old.target == target) & (old.variant == var)]
            path = CAP / 'STATE_CAPACITY_METRICS.csv'
            if rows.empty:
                rows = cmab[(cmab.source == source) & (cmab.dataset == target) & (cmab.variant == var)]
                path = CMAB / 'RAW_METRICS.csv'
            assert len(rows) <= 1, (source, target, k)
            value = float(rows.iloc[0].M_FR_P_AP) if len(rows) else np.nan
            entry[f'K={k}'] = value
            if k == 2:
                folder = 'phase_hard_normal_contrast' if source == 'Full VisA' else 'phase_hard_normal_cross_source_mvtec'
                ckpt = ROOT / f'results/{folder}/formal/contrast_vectors.npz'
                status = 'DEPLOYED_HISTORICAL_NTR_DIRECTION; not init_v_fallback; run identity not embedded'
                seed, actual = 'nominal source seed0; not stored in vector artifact', 'UNRESOLVED_FROM_PAYLOAD'
                training = line('experiments/phase_final_cross_dataset_benchmark.py', 'def load_visa_source')
            else:
                ckpt = CMAB / f'{source.replace(" ", "_")}_{var}_formal.pt'
                payload = torch.load(ckpt, map_location='cpu', weights_only=False)
                assert payload['s_a'].shape == (k-1, 1024)
                epochs = [int(x['epoch']) for x in payload['curves']]
                status = f'TRAINED_CHECKPOINT; last_epoch={max(epochs)}; states verified; seed absent from payload'
                assert max(epochs) == 10
                seed, actual = 'nominal SEED=0 configuration', str(k-1)
                training = line('experiments/phase_cmab_compact_multi_anomaly_basis.py', 'torch.manual_seed(SEED + k)')
            checks = cat[(cat.source == source) & (cat.dataset == target) & (cat.variant == var) & (cat.protocol == 'M')]
            verify = 'MISSING' if not np.isfinite(value) else 'RECORDED_CATEGORY_MACRO'
            if len(checks):
                assert np.isclose(checks.M_FR_P_AP.mean(), value, atol=1e-4)
                verify = f'VERIFIED_FROM_{len(checks)}_CATEGORY_ROWS'
            provenance.append(dict(K=k, source=source, target=target, P_AP=value,
                seed=seed, actual_rng_seed=actual, checkpoint=str(ckpt), checkpoint_exists=ckpt.exists(),
                raw_result=str(path), raw_csv_line=int(rows.index[0])+2 if len(rows) else 'MISSING',
                evaluation_script='experiments/phase_paper_final_evidence.py + phase_cmab_compact_multi_anomaly_basis.py',
                evaluator=line('experiments/phase_cmab_compact_multi_anomaly_basis.py','def evaluate_maps'),
                aggregation='category-macro M_FR; AP computed within category then category mean',
                NTR='top10% batch normal logits; softplus mean; weight1 (K>2); historical NTR vector for K2',
                LRP='fixed2x2; global/local0.5/0.5; 74x74; bilinear align_corners=True',
                preprocessing='VisA source PC1 removal + frozen rank4 alignment; MVTec normalized patches',
                training_status=status, training_trace=training, verification=verify))
        tables.append(entry)
    tables.append({'source':'Mean', 'target':'four specified transfers',
                   **{f'K={k}':float(np.mean([x[f'K={k}'] for x in tables]))
                      if all(np.isfinite(x[f'K={k}']) for x in tables) else np.nan for k in variants}})
    save('FIG5_CAPACITY_TRANSFER_TABLE.csv', tables)
    save('FIG5_CAPACITY_PROVENANCE.csv', provenance)
    src = read(CAP / 'SOURCE_FIT_METRICS.csv')
    src['source_patch_AP_percent'] = src.source_patch_AP * 100
    src['artifact'] = str(CAP / 'SOURCE_FIT_METRICS.csv')
    src['same_patch_id_verification'] = 'UNRESOLVED: equal counts only; no saved patch IDs/responses or generator located'
    src['label_preprocessing_verification'] = 'UNRESOLVED for source-fit CSV; CMAB loader uses source real masks and source preprocessing'
    src['aggregation'] = 'Recorded pooled source-patch AP in [0,1]; NOT target category-macro AP'
    src['actual_rng_seed'] = src.total_states.map({2:'UNRESOLVED',3:'2 (code-derived)',5:'4 (code-derived)'})
    save('FIG5_SOURCE_PATCH_AP.csv', src)

    h = read(HEAD / 'HEAD_CONTROL_RAW_METRICS.csv')
    c = read(COMP / 'CORRECTED_ABLATION_CATEGORY_MACRO_RAW.csv')
    hc, cc = [], []
    for source, target in PAIRS:
        for method in ('CES', 'LinearHead', 'NormalizedLinearHead'):
            q = h[(h.source==source)&(h.target==target)&(h.method==method)&(h.seed==0)]
            hc.append(dict(source=source,target=target,method=method,seed=0,
                P_AP=float(q.iloc[0].P_AP) if len(q) else np.nan,
                status='RECORDED' if len(q) else 'MISSING',
                checkpoint=str(q.iloc[0].checkpoint) if len(q) else str(HEAD/f'checkpoints/{source.replace(" ","_")}_{method}_seed0.pt'),
                result=str(HEAD/'HEAD_CONTROL_RAW_METRICS.csv'),
                dependency='Missing local_patch_Full_VisA__MPDD.npy' if not len(q) else 'existing result'))
        for seed in (0,1,2):
            for variant in ('Trained_CES_Core','Trained_CES_NTR','Trained_CES_LRP','Trained_CES_NTR_LRP'):
                q=c[(c.source==source)&(c.dataset==target)&(c.seed==seed)&(c.variant==variant)]
                cc.append(dict(source=source,target=target,variant=variant,seed=seed,
                    P_AP=float(q.iloc[0].M_FR_P_AP) if len(q) else np.nan,
                    status='RECORDED' if len(q) else 'MISSING_MATCHED_CORRECTED_RESULT',
                    checkpoint=str(q.iloc[0].checkpoint) if len(q) else 'existing source checkpoint reusable; see MISSING_EXPERIMENTS.md',
                    result=str(COMP/'CORRECTED_ABLATION_CATEGORY_MACRO_RAW.csv')))
    save('PARAMETERIZATION_PROTOCOL_CHECK.csv',hc)
    save('COMPONENT_ABLATION_PROTOCOL_CHECK.csv',cc)
    # Preserve provenance for both historical statistics, never relabel old cells.
    seed_stats=[]
    for variant,group in c.groupby('variant'):
        means=group.groupby('seed').M_FR_P_AP.mean()
        seed_stats.append(dict(protocol='HISTORICAL includes MVTec->MPDD',variant=variant,
            mean=float(means.mean()),sample_std_ddof1=float(means.std(ddof=1)),
            seed0=float(means[0]),seed1=float(means[1]),seed2=float(means[2])))
    save('HISTORICAL_COMPONENT_SEED_STATISTICS.csv',seed_stats)

    differences=[]
    for source,folder in [('Full VisA','phase_hard_normal_contrast'),('Full MVTec','phase_hard_normal_cross_source_mvtec')]:
        oldpath=ROOT/f'results/{folder}/formal/contrast_vectors.npz'
        newpath=ROOT/f'results/paper_assets/final_ablation_close/checkpoints/{source.replace(" ","_")}_seed0_rho0.10.pt'
        with np.load(oldpath) as z:
            a=np.asarray(z['E1'],dtype=np.float64)
        payload=torch.load(newpath,map_location='cpu',weights_only=False)
        b=np.asarray(payload['v'],dtype=np.float64)
        a/=np.linalg.norm(a);b/=np.linalg.norm(b)
        differences.append(dict(source=source,historical_path=str(oldpath),formal_path=str(newpath),
            new_seed=payload['seed'],new_rho=payload['rho'],cosine=float(a@b),max_abs_difference=float(np.max(np.abs(a-b)))))
    save('STATE_DIRECTION_ARTIFACT_COMPARISON.csv',differences)
    rho=read(ROOT/'results/paper_assets/final_ablation_close/RHO_SENSITIVITY.csv')
    rho=rho[(rho.rho==0.10)&(rho.variant=='Base+NTR+LRP')&(rho.seed==0)]
    historical=[]
    for _,r in old[old.capacity=='K2_CES'].iterrows():
        q=rho[(rho.source==r.source)&(rho.target==r.target)].iloc[0]
        historical.append(dict(source=r.source,target=r.target,
            capacity_K2_P_AP=r.M_FR_P_AP,rho010_seed0_P_AP=q.P_AP,
            delta=r.M_FR_P_AP-q.P_AP,
            capacity_result=str(CAP/'STATE_CAPACITY_METRICS.csv'),
            sensitivity_result=str(ROOT/'results/paper_assets/final_ablation_close/RHO_SENSITIVITY.csv'),
            sensitivity_checkpoint=q.get('checkpoint','UNKNOWN')))
    save('HISTORICAL_NUMERIC_RECONCILIATION.csv',historical)
    config=[
        ('CLIP/features','Frozen ViT-L/14@336px;518;final patch1024','Same intended feature pipeline','source loader'),
        ('Source preprocessing','VisA PC1+frozen rank4;MVTec normalized','Same source feature transform','phase_final_cross_dataset_benchmark.py'),
        ('State parameterization','fixed midpoint/alpha; optimize normalized direction','free s_N and independent anomaly states','phase_cmab_compact_multi_anomaly_basis.py:train_cmab_variant'),
        ('Anomaly aggregation','normalized anomaly-normal cosine difference','logsumexp anomaly cosines minus normal cosine; no subtract log(K-1)','multi_state_logits'),
        ('Initialization','historical trained NTR source states','start from historical NTR states +1% orthogonal random perturbations','init_anomaly_basis'),
        ('Seed','historical nominal0;artifact has no seed','SEED+(K-1):2 or4;payload has no seed','train_cmab_variant'),
        ('Optimizer/budget','original NTR source training','additional10epochs AdamW lr0.001 wd0;batch8192','train_cmab_variant'),
        ('Loss/NTR','source state loss+NTR','balanced BCE+Dice +top10% normal softplus+diversity','train_cmab_variant'),
        ('Anti-collapse','None','mean off-diagonal anomaly cosine;weight1','diversity_loss'),
        ('LRP/evaluator','fixed2x2;lambda0.5;M_FR category-macro','same fixed LRP/evaluator','build_pair_maps/evaluate_maps'),
    ]
    save('CAPACITY_CONFIGURATION_COMPARISON.csv',[dict(item=a,K2=b,K3_K5=c,evidence=d) for a,b,c,d in config])
    # Log the bounded global search, including checkpoint/prediction artifacts.
    inventory=[]
    for base in ('results','experiments','cache','checkpoints'):
        for path in (ROOT/base).rglob('*'):
            if path.is_file() and path.suffix.lower() in ('.csv','.json','.npz','.pt','.py','.md'):
                rel=str(path.relative_to(ROOT))
                if any(x in rel.lower() for x in ('cmab','capacity','head_control','linear_head','corrected','final_evidence','mpdd')):
                    inventory.append(dict(path=str(path),bytes=path.stat().st_size))
    save('SEARCH_INVENTORY.csv',inventory)
    missing='''# 缺失实验及可复用资产

未执行训练或目标推理。容量新四迁移的12个分数已存在；无须补训K3/K5。

## 参数化：VisA->MPDD

CES、LinearHead、NormalizedLinearHead在该迁移上的匹配seed0结果缺失。
已存在 `results/phase_ces_linear_head_control/checkpoints/Full_VisA_{CES,LinearHead,NormalizedLinearHead}_seed0.pt`。
不能用旧CMAB的40.348571冒充这里CES的同checkpoint结果。

## 组件：VisA->MPDD

四种组合×seed0/1/2的12条匹配corrected记录缺失。已有Full VisA no-NTR及NTR source checkpoints，原则上只需评估，不需要补训。
seed0 no-NTR在 `results/ces_ntr_visualization/corrected/checkpoints/Full_VisA_seed0_statealigned.pt`；
seed0 NTR在 `results/paper_assets/final_ablation_close/checkpoints/Full_VisA_seed0_rho0.10.pt`；
seeds1/2在 `results/phase_paper_ready_final_validation/formal/checkpoints/Full_VisA_seed{1,2}_{statealigned,ntr}.pt`。

## 依赖与命令

`E:/MRAD_final_evidence_cache/local_patch_Full_VisA__MPDD.npy` 未找到。另一个source的MPDD local cache不能仅凭文件名推定可互换，必须核实image顺序和特征是否未经source变换。
已有74x74 anomaly maps不足以重建另一个head或checkpoint的feature响应。

补齐并验证global/local frozen patch features与测试image ID顺序后，可运行本目录的只评估入口：

```powershell
Set-Location D:/ProjectPython/MRAD-main/MRAD-main
python experiments/reevaluate_reconciliation_mpdd.py --family heads --local-manifest E:/MRAD_final_evidence_cache/local_patch_Full_VisA__MPDD_paths.json
python experiments/reevaluate_reconciliation_mpdd.py --family components --local-manifest E:/MRAD_final_evidence_cache/local_patch_Full_VisA__MPDD_paths.json
```

该入口复用现有scorer与category-macro M_FR evaluator，缺少local feature cache会立即停止；不会训练权重。manifest应为逐图img_path字符串列表，必须从真实特征提取顺序取得，不得人工假定。
当前不提供无根据的运行分钟数：需458张MPDD测试图，每图global+4local观察（如需补feature）；feature FP16 local缓存约5.14GB，计算开销取决于是否能核实复用已有raw缓存。

## Source Patch AP

数值CSV存在，但未定位生成这六条数值的可执行函数、逐patch ID及逐patch预测。
相同样本数不证明相同样本身份。应补充原始source-fit预测/生成日志，或在明确授权下只对现有state重评同一固定source pool，不训练。
'''
    (OUT/'MISSING_EXPERIMENTS.md').write_text(missing,encoding='utf-8')
    table=pd.DataFrame(tables).to_string(index=False,float_format=lambda x:f'{x:.6f}')
    report=f'''# 第四章正式协议复核

本次仅读取代码、权重及已有数值并重新汇总，没有训练、目标推理或修改论文。

## 正式四迁移容量表

```text
{table}
```

前三迁移引用final_evidence容量CSV，VisA->MPDD引用CMAB formal RAW_METRICS；均为category-macro M_FR。
这一表保留同一历史CMAB状态家族，不能将部分K2单元格换成另一个rho0.10 checkpoint以追求数值统一。
新均值是正式四迁移上历史高状态配置的结果，不是多种子均值。

## 核验新发现

1. K指总状态数：1normal+(K-1)anomaly。K3/K5 checkpoint分别有s_a[2,1024]/[4,1024]及10epoch训练曲线。
2. 不能写成同seed纯容量控制：代码固定SEED=0，但实际用SEED+k（k是anomaly数量）初始化和采样，即2和4。checkpoint不保存seed，标为代码推导。
3. K2沿用历史NTR训练方向；K3/K5从这些状态开始再优化free normal/anomaly states10epochs，加入LSE和diversity。是higher-state configuration comparison。
4. Source-fit CSV六条数值真实存在，但生成函数和patch ID证据不足；样本数一致不能证明同一批patch。不得据此声称已严格验证相同source fit protocol。

## 39.62与39.54

39.617968来自历史四迁移容量记录；39.536029来自head/sensitivity seed0；39.540225来自corrected三种子平均。三者都包含MVTec->MPDD，而非新VisA->MPDD。
容量与sensitivity在VisA->BTAD分别58.357748/58.022826，在VisA->MVTec分别44.289840/44.297010。
本次直接读权重核实：历史VisA E1方向与rho0.10 seed0方向cosine={differences[0]['cosine']:.12f}，max差={differences[0]['max_abs_difference']:.12f}；MVTec方向一致。
因此39.62差异不是单种子与多种子均值造成。两套VisA方向确实不同；具体历史训练执行日志不足以进一步解释形成原因。

## 参数化与组件

新协议heads缺VisA->MPDD的3条seed0结果；组件缺四组合×3seed的12条匹配结果。
其余三迁移已有记录可以保留。历史27.88/35.43/31.63/39.54只能标为历史固定四迁移结果。
已存在source checkpoints，当前无需补训；需补feature cache及评估。新四迁移结果不全时不得计算完整新mean±std。
历史corrected标准差已按要求重新整理为先每seed四迁移mean，再对三个seed计算sample std(ddof=1)，见HISTORICAL_COMPONENT_SEED_STATISTICS.csv。
旧summary中的±0.05采用平均各迁移population std，不是该统计口径。

## 论文处理建议

新Fig.5 target容量柱可依据本目录正式四迁移表重画，但caption须限定历史higher-state configuration；不能标同seed纯K消融。
Source fit面板暂不能作为已完整核验的公平对照定稿。最终Fig.5暂不应直接定稿。
参数化和组件主文若统一到新四迁移，需补VisA->MPDD后重汇总；否则明确披露旧四迁移范围，不改标签。
旧表、原始结果和论文均保持不变。没有按目标性能选择source/checkpoint。
'''
    (OUT/'PAPER_PROTOCOL_AUDIT.md').write_text(report,encoding='utf-8')
    print(table)
    print(src[['source','capacity','source_patch_AP_percent']].to_string(index=False))
    print('Missing heads:',sum(x['status']=='MISSING' for x in hc))
    print('Missing components:',sum(x['status']!='RECORDED' for x in cc))
    print('Saved:',OUT)


if __name__=='__main__':
    main()
