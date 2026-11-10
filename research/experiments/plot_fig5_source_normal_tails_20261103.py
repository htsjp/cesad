"""Two-panel source-normal diagnostic, using certified cached features only.

No training, CLIP encoding, target evaluation, KDE, binning or score calibration.
One locked sample: 50 source images/category x 20 GT-normal patches/image.
"""
from pathlib import Path
import sys
import json
import math
import hashlib
import inspect
import argparse
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from experiments import final_unified_evaluation as ev
from experiments import source_build_traceable_visa as build
from experiments.phase_xxxviii_visual_state_representation import norm_np
from experiments.phase_xliii_adapter_head_decomposition import AdapterToState
from experiments.phase_ces_linear_head_control import CESHead

QA=Path('C:/Users/Administrator/.codex/skills/nature-figure/scripts')
sys.path.insert(0,str(QA))
from audit_panel_alignment import require_matplotlib_panel_alignment
import audit_pdf_text
import audit_figure_collisions

OUT=ROOT/'results/paper_fig5_source_normal_tails_20261103'
RUN=ROOT/'results/ntr_lrp_reciprocal_aligned_20261101'
CERT=ROOT/'results/visa_source_traceable_rebuild/formal/source_feature_certification/SOURCE_FEATURE_CERTIFICATE.json'
UPSTREAM=RUN/'formal/UPSTREAM_IDENTITY_VALIDATION.json'
CATEGORIES=('candle','capsules','cashew','chewinggum','fryum','macaroni1',
            'macaroni2','pcb1','pcb2','pcb3','pcb4','pipe_fryum')
SLATE='#5B6F82'
TEAL='#217F78'
SPEC=dict(version=1,source='Full VisA',training_seed=0,sampling_seed=0,
          categories=list(CATEGORIES),images_per_category=50,patches_per_image=20,
          normal_patches_per_category=1000,total_normal_patches=12000,
          candidate_pool='all certified source images; all true-label==0 patches, including normal regions of anomalous images',
          image_order='ascending physical source_position before seeded uniform sampling; selected images sorted again',
          within_image='uniform without replacement among GT-normal patch IDs; sorted sampled IDs',
          hard_fraction=.10,hard_selection='within category: top100 trained-no-NTR raw scores, ties by source_position then patch ID',
          score='CESHead.forward: z @ (normalize(m+alpha*normalize(v))-normalize(m-alpha*normalize(v)))',
          feature_path='certified raw FP32 -> PC1 removal -> norm_np -> frozen E2 adapter.adapt (includes L2)',
          endpoint='update_030 only; trained CES versus trained CES+NTR rho=.10',
          survival='exact P(q>t), searchsorted side=right, equal category quotas',
          tail_display_max=.20,display_xmin='min of the two empirical 80th percentiles minus 4% tail-span',
          target_data=False,training=False,CLIP_inference=False,no_resampling_after_scores=True)


def identity(p):
    p=Path(p)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=ev.sha(p))


def check(record):
    got=identity(record['path'])
    assert got['sha256']==record['sha256'],record['path']
    if 'bytes' in record:assert got['bytes']==record['bytes']
    return got


def trace(fn):
    return dict(file=inspect.getsourcefile(fn),function=fn.__qualname__,line=inspect.getsourcelines(fn)[1])


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    preset=OUT/'LOCKED_SAMPLING_CONFIG.json'
    if preset.exists():assert ev.readjson(preset)==SPEC
    else:ev.savejson(preset,SPEC)
    if (OUT/'DATA_SEALED.json').exists():
        seal=ev.readjson(OUT/'DATA_SEALED.json')
        assert seal['sampling_config_sha256']==ev.sha(preset)
        for record in seal['input_identities']+seal['data_files']:check(record)
        print('Verified existing sealed data; no resampling.',flush=True)
        return
    torch.set_num_threads(4)
    inputs=[identity(CERT),identity(UPSTREAM),identity(Path(__file__))]
    cert=ev.readjson(CERT)
    for key in ('image_feature_manifest','source_labels','patch_features'):
        print('VERIFY',key,flush=True)
        inputs.append(check(cert[key]))
    upstream=ev.readjson(UPSTREAM)
    assert upstream['status']=='PASS'
    stages={r['stage']:r for r in upstream['files']}
    for r in stages.values():inputs.append(check(r))
    config_path=RUN/'formal/LOCKED_CONFIG.json'
    source_seal_path=RUN/'formal/SOURCE_SEALED.json'
    config=ev.readjson(config_path);sealed=ev.readjson(source_seal_path)
    inputs.extend([identity(config_path),identity(source_seal_path)])
    assert sealed['config_sha256']==ev.sha(config_path)
    models=[];initial=[];ckpaths=[]
    for rho in (None,.10):
        entry=next(e for e in sealed['models'] if e['source']=='Full VisA' and e['target']=='MVTec' and e['seed']==0 and e['rho']==rho)
        inputs.append(check(entry['checkpoint']));ckpaths.append(entry['checkpoint']['path'])
        cp=torch.load(entry['checkpoint']['path'],map_location='cpu',weights_only=False)
        assert cp['step']==30 and cp['seed']==0 and cp['rho']==rho
        assert cp['config_sha256']==ev.sha(config_path)
        ip=Path(entry['checkpoint']['path']).with_name('update_000.pt')
        inputs.append(identity(ip))
        initial.append(torch.load(ip,map_location='cpu',weights_only=False)['raw_model'])
        models.append(cp)
    assert models[0]['source_input_sha256']==models[1]['source_input_sha256']
    assert np.array_equal(models[0]['batch_indices'],models[1]['batch_indices'])
    assert all(torch.equal(initial[0][k],initial[1][k]) for k in initial[0])
    assert all(torch.equal(models[0]['raw_model'][k],models[1]['raw_model'][k]) for k in ('m','alpha'))
    assert models[0]['optimizer']['param_groups']==models[1]['optimizer']['param_groups']
    assert config['updates']==30
    sourceinput=RUN/'formal/source_inputs/MVTec_seed0.json'
    assert ev.sha(sourceinput)==models[0]['source_input_sha256']
    inputs.append(identity(sourceinput))
    for rec in ev.readjson(sourceinput)['files']:inputs.append(check(rec))
    comparison=[dict(item='Full VisA source inputs',status='PASS'),
        dict(item='frozen CLIP/PC1/E2 feature chain',status='PASS'),
        dict(item='initial raw state and frozen m/alpha',status='PASS'),
        dict(item='actual batch IDs and 30 updates',status='PASS'),
        dict(item='optimizer parameter groups',status='PASS'),
        dict(item='base loss',status='PASS: shared balanced BCE + Dice training implementation'),
        dict(item='NTR',status='only intended difference: disabled versus rho=0.10'),
        dict(item='LRP',status='N/A: source global patch diagnostic, not a fused target map')]
    ev.writecsv(OUT/'MODEL_COMPARISON.csv',comparison)
    images=ev.readcsv(cert['image_feature_manifest']['path'])
    labels=np.load(cert['source_labels']['path'],mmap_mode='r')
    raw=np.load(cert['patch_features']['path'],mmap_mode='r')
    assert raw.dtype==np.float32 and raw.shape==(len(images),1369,1024)
    assert labels.shape==raw.shape[:2]
    assert set(r['category'] for r in images)==set(CATEGORIES)
    assert all(int(r['source_position'])==i and int(r['patch_feature_row'])==i for i,r in enumerate(images))
    rng=np.random.default_rng(0);sample=[]
    for category in CATEGORIES:
        eligible=[int(r['source_position']) for r in images if r['category']==category
                  and int(np.sum(labels[int(r['source_position'])]==0))>=20]
        assert len(eligible)>=50,(category,len(eligible))
        positions=np.sort(rng.choice(sorted(eligible),size=50,replace=False))
        for pos in positions:
            row=images[int(pos)]
            assert ev.sha(row['image'])==row['image_sha256']
            assert hashlib.sha256(np.asarray(raw[int(pos)]).tobytes()).hexdigest()==row['patch_sha256']
            ids=np.sort(rng.choice(np.flatnonzero(labels[int(pos)]==0),size=20,replace=False))
            for pid in ids:
                sample.append(dict(category=category,source_position=int(pos),patch_index=int(pid),
                    patch_row=int(pid)//37,patch_col=int(pid)%37,label=0,
                    patch_id=f'VisA/source_row{int(pos):04d}/patch{int(pid):04d}',
                    image_path=row['image'],image_sha256=row['image_sha256'],
                    image_label=int(row['image_label']),raw_feature_row_sha256=row['patch_sha256']))
        print('LOCKED SAMPLE',category,'50 images x 20 normal patches',flush=True)
    assert len(sample)==12000 and len({r['patch_id'] for r in sample})==12000
    assert set(Counter(r['category'] for r in sample).values())=={1000}
    assert set(Counter(r['source_position'] for r in sample).values())=={20}
    ev.writecsv(OUT/'SAMPLE_MANIFEST_LOCKED_BEFORE_SCORING.csv',sample)
    q=np.load(stages['PC1']['path'])
    pairs=np.array([(r['source_position'],r['patch_index']) for r in sample],dtype=np.int64)
    # Call the exact source-build PC1-removal/NumPy-L2 function, not a plotting substitute.
    x=build.pool_features(raw,dict(physical_pairs=pairs),q,norm_np)
    adapter_payload=torch.load(stages['E2']['path'],map_location='cpu',weights_only=False)
    adapter=AdapterToState(1024,adapter_payload['patch_states'])
    adapter.load_state_dict(adapter_payload['model'],strict=True);adapter.eval()
    assert tuple(adapter.a.weight.shape)==(4,1024) and tuple(adapter.b.weight.shape)==(1024,4)
    for p in adapter.parameters():p.requires_grad_(False)
    heads=[]
    for cp in models:
        state=cp['raw_model'];head=CESHead(state['m'].numpy(),float(state['alpha']),state['v'].numpy())
        head.load_state_dict(state,strict=True);head.eval()
        for p in head.parameters():p.requires_grad_(False)
        heads.append(head)
    with torch.inference_mode():
        z=adapter.adapt(torch.from_numpy(x))
        scores=torch.stack([head(z) for head in heads],dim=1).numpy()
        for j,head in enumerate(heads):
            direction=torch.nn.functional.normalize(head.v,dim=0)
            sa=torch.nn.functional.normalize(head.m+head.alpha*direction,dim=0)
            sn=torch.nn.functional.normalize(head.m-head.alpha*direction,dim=0)
            assert torch.allclose(head(z),z@(sa-sn),atol=1e-7,rtol=0)
    assert np.isfinite(scores).all() and np.max(np.abs(z.norm(dim=1).numpy()-1))<2e-6
    # Independently compare the shared physical patches against the actual CES training pool.
    pool=RUN/'work/source_pools/VisA_seed0'
    plan=np.load(pool/'POOL_PLAN.npz');oldz=np.load(pool/'features.npy',mmap_mode='r')
    lookup={tuple(p):i for i,p in enumerate(plan['physical_pairs'])}
    common=[(i,lookup[tuple(p)]) for i,p in enumerate(pairs) if tuple(p) in lookup]
    assert common,'No common patches available for independent transformed-feature parity'
    parity=max(float(np.max(np.abs(z[i].numpy()-oldz[j]))) for i,j in common)
    assert parity<2e-6,parity
    df=pd.DataFrame(sample);df['q_ces']=scores[:,0];df['q_ntr']=scores[:,1]
    df['hard_normal']=False;summary=[]
    for category in CATEGORIES:
        ids=np.flatnonzero(df.category.to_numpy()==category)
        # Primary sort descending CES q; exact ties resolved by physical ID, no jitter.
        order=np.lexsort((pairs[ids,1],pairs[ids,0],-scores[ids,0]))
        hard=ids[order[:100]];df.loc[hard,'hard_normal']=True
        assert np.all(labels[pairs[hard,0],pairs[hard,1]]==0)
        before=float(scores[hard,0].mean(dtype=np.float64));after=float(scores[hard,1].mean(dtype=np.float64))
        summary.append(dict(category=category,n_normal=1000,n_images=50,n_hard=100,
            hard_mean_ces=before,hard_mean_ntr=after,hard_delta_ntr_minus_ces=after-before,
            normal_mean_ces=float(scores[ids,0].mean(dtype=np.float64)),
            normal_mean_ntr=float(scores[ids,1].mean(dtype=np.float64)),
            hard_patch_decrease_fraction=float(np.mean(scores[hard,1]<scores[hard,0]))))
    df.to_csv(OUT/'FIG5_PATCH_SCORES.csv',index=False,float_format='%.10g')
    ev.writecsv(OUT/'FIG5_CATEGORY_PAIRED.csv',summary)
    # The ECDF is exact at every observed score; strict > includes all ties on the low side.
    thresholds=np.unique(scores.reshape(-1)).astype(np.float64)
    thresholds=np.r_[np.nextafter(thresholds[0],-np.inf),thresholds]
    surv=np.stack([(12000-np.searchsorted(np.sort(scores[:,j].astype(np.float64)),thresholds,side='right'))/12000 for j in (0,1)],axis=1)
    assert np.all(np.diff(surv,axis=0)<=0) and np.all(surv[0]==1) and np.all(surv[-1]==0)
    pd.DataFrame(dict(threshold=thresholds,survival_ces=surv[:,0],survival_ntr=surv[:,1])).to_csv(OUT/'FIG5_SURVIVAL_CURVES.csv',index=False,float_format='%.17g')
    np.savez_compressed(OUT/'SAMPLED_TRANSFORMED_FEATURES.npz',z=z.numpy(),physical_pairs=pairs,scores=scores)
    source_trace=[trace(build.pool_features),trace(norm_np),trace(AdapterToState.adapt),trace(CESHead.forward)]
    audit=dict(source='Full VisA',seed=0,trained_CES=ckpaths[0],trained_CES_NTR=ckpaths[1],
        n_categories=12,n_normal_patches=12000,images_per_category=50,patches_per_image=20,
        distinct_images=600,normal_patches_from_anomalous_images=int(df.image_label.sum()),
        hard_normal_total=1200,category_hard_means_decreased=sum(r['hard_delta_ntr_minus_ces']<0 for r in summary),
        same_patch_ids=True,same_hard_subset=True,trained_not_fallback=True,
        training_budget=30,source_feature_dtype=str(raw.dtype),scoring_dtype=str(scores.dtype),
        source_pool_parity_patches=len(common),source_pool_transform_parity_max_abs=parity,
        feature_chain=SPEC['feature_path'],score=SPEC['score'],source_trace=source_trace,
        empirical_quantile80_ces=float(np.quantile(scores[:,0],.8)),
        empirical_quantile80_ntr=float(np.quantile(scores[:,1],.8)),
        hard_macro_mean_ces=float(np.mean([r['hard_mean_ces'] for r in summary])),
        hard_macro_mean_ntr=float(np.mean([r['hard_mean_ntr'] for r in summary])),
        interpretation='descriptive source-normal score suppression only; no claim about anomaly ranking or target localization',
        sampling_config_sha256=ev.sha(preset))
    ev.savejson(OUT/'DATA_AUDIT.json',audit)
    names=('FIG5_PATCH_SCORES.csv','FIG5_CATEGORY_PAIRED.csv','FIG5_SURVIVAL_CURVES.csv',
           'SAMPLE_MANIFEST_LOCKED_BEFORE_SCORING.csv','SAMPLED_TRANSFORMED_FEATURES.npz','DATA_AUDIT.json','MODEL_COMPARISON.csv')
    ev.savejson(OUT/'DATA_SEALED.json',dict(sampling_config_sha256=ev.sha(preset),
        input_identities=inputs,data_files=[identity(OUT/n) for n in names]))
    print('DATA AUDIT',json.dumps(audit,ensure_ascii=False),flush=True)


def plot():
    seal=ev.readjson(OUT/'DATA_SEALED.json')
    for rec in seal['data_files']:check(rec)
    df=pd.read_csv(OUT/'FIG5_PATCH_SCORES.csv')
    paired=pd.read_csv(OUT/'FIG5_CATEGORY_PAIRED.csv')
    ecdf=pd.read_csv(OUT/'FIG5_SURVIVAL_CURVES.csv')
    assert paired.category.tolist()==list(CATEGORIES)
    assert len(df)==12000 and df.hard_normal.sum()==1200
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'pdf.fonttype':42,
        'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.7,'legend.frameon':False,'axes.labelsize':10,'xtick.labelsize':8,
        'ytick.labelsize':8})
    fig,axes=plt.subplots(1,2,figsize=(10.6,4.15))
    fig.subplots_adjust(left=.075,right=.985,bottom=.31,top=.87,wspace=.27)
    a,b=axes
    a.step(ecdf.threshold,ecdf.survival_ces,where='post',color=SLATE,lw=1.7,label='Trained CES')
    a.step(ecdf.threshold,ecdf.survival_ntr,where='post',color=TEAL,lw=1.7,label='CES + NTR')
    xmin=min(float(np.quantile(df.q_ces,.8)),float(np.quantile(df.q_ntr,.8)))
    xmax=float(max(df.q_ces.max(),df.q_ntr.max()));pad=.04*(xmax-xmin)
    a.set(xlim=(xmin-pad,xmax+pad),ylim=(0,.20),xlabel='Anomaly score q',ylabel='Normal-patch exceedance')
    a.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    a.set_yticks([0,.05,.10,.15,.20]);a.legend(loc='upper right',fontsize=8)
    x=np.arange(12)
    for i,r in paired.iterrows():
        b.plot([i-.07,i+.07],[r.hard_mean_ces,r.hard_mean_ntr],color='#B8BEC4',lw=.9,zorder=1)
    b.scatter(x-.07,paired.hard_mean_ces,s=28,c=SLATE,label='Trained CES',zorder=3)
    b.scatter(x+.07,paired.hard_mean_ntr,s=28,c=TEAL,marker='s',label='CES + NTR',zorder=3)
    b.set(xlim=(-.65,11.65),ylabel='Mean anomaly score q',xlabel='VisA source category')
    b.set_xticks(x,CATEGORIES,rotation=52,ha='right',rotation_mode='anchor')
    b.legend(loc='upper right',fontsize=8)
    for ax in axes:
        ax.grid(axis='y',color='#D5DADF',lw=.5,alpha=.55)
        ax.set_axisbelow(True)
    a.set_title('(a) Normal-score upper tails',fontsize=10,pad=8)
    b.set_title('(b) Fixed hard-normal subsets',fontsize=10,pad=8)
    fig.suptitle('Effect of NTR on source-normal score tails',fontsize=12,y=.98)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=OUT/'FIG5_ALIGNMENT.json',strict=True)
    stem=OUT/'fig5_source_normal_tails'
    fig.savefig(stem.with_suffix('.png'),dpi=600,bbox_inches='tight')
    fig.savefig(stem.with_suffix('.pdf'),bbox_inches='tight')
    fig.savefig(stem.with_suffix('.svg'),bbox_inches='tight')
    fig.savefig(OUT/'fig5_source_normal_tails_preview.png',dpi=300,bbox_inches='tight')
    plt.close(fig)
    text=audit_pdf_text.audit_pdf(stem.with_suffix('.pdf').read_bytes(),minimum_pt=5.)
    collisions=audit_figure_collisions.audit_pdf(stem.with_suffix('.pdf'))
    ev.savejson(OUT/'FIG5_PDF_QA.json',dict(text=text,collisions=collisions))
    assert text['auditable'] and text['below_minimum_count']==0
    assert collisions['verdict']=='PASS',collisions
    audit=ev.readjson(OUT/'DATA_AUDIT.json')
    caption='''Fig. 5. Effect of NTR on source-normal score tails. Source-normal patches are sampled equally from all 12 VisA categories (50 source images per category and 20 GT-normal patches per image; 1,000 patches per category; sampling seed 0). Both trained seed-0 models use the same frozen feature preparation, initialization, batches and 30-update budget, with NTR disabled or enabled. (a) Empirical normal-score exceedance, Pr(q > t | y = 0), displayed over the upper 20% probability range without density smoothing. Equal category quotas give each category equal weight. (b) Mean scores of the fixed top-10% normal patches selected within each category by the trained model without NTR; the same patch IDs are scored by both models. Lines connect paired category means. The 20% probability range is a display setting, and the category-wise 10% selection is a post-hoc diagnostic, not a change to training. These source-normal responses do not by themselves establish improved anomalous-patch discrimination or target localization.'''
    (OUT/'CAPTION.md').write_text(caption+'\n',encoding='utf-8')
    notes=f'''# Fig. 5 source-normal tail diagnostic

## Data and model identity
Use the completed 20261101 certified VisA-source reciprocal component run, seed0.
Trained CES: `{audit['trained_CES']}`.
Trained CES+NTR: `{audit['trained_CES_NTR']}`.
Both are final update30 checkpoints, not init_v fallback or before/after states of one run.
The common physical source input, initial raw states, m/alpha, actual batches and optimizer
groups are checked. Base loss is shared BCE+Dice. Only the intended NTR training switch differs.
This is the current certified component family, not silently the old historical Table IV family.

## Locked sample and true feature chain
12 categories x 50 uniformly selected eligible source images x 20 uniformly selected normal
patches = 12,000 patches. No replacement and no per-image quota adjustment. Seed0, category
order and all image/patch identities were locked before extracting scores. Source-normal means
true label0; {audit['normal_patches_from_anomalous_images']} patches are normal regions of
source images with anomalies. No target data are consumed. All 600 selected image hashes
and raw feature-row hashes are checked against the fresh source certificate, as are the full
source feature file and labels. The exact source-build pool_features/normalizer and E2 adapt
functions are called: raw normalized FP32 feature -> PC1 removal -> NumPy L2 -> frozen rank4
residual adapter -> output L2 -> CESHead.forward. No raw untransformed feature is scored.
On {audit['source_pool_parity_patches']} common physical patches, transformed features agree
with the actual training pool within max absolute error {audit['source_pool_transform_parity_max_abs']:.10g}.
Function file/line traces and all input hashes are in DATA_AUDIT.json and DATA_SEALED.json.

## Statistical and display definition
Panel(a) is exact strict empirical exceedance using searchsorted(side=right), no KDE/bins.
Equal category quotas make pooled survival equal to category-macro survival. Display y=0-20%;
x starts at the lesser empirical 80th percentile with a fixed 4% tail-span margin, and extends
past the maximum observed score. No high-score observation is omitted or truncated. The full
unrestricted ECDF is saved in FIG5_SURVIVAL_CURVES.csv.
Panel(b) selects exactly 100/1,000 normal patches per category using only trained no-NTR q;
ties resolve by source image position then patch index. Those exact IDs are shared. The
category-wise diagnostic selection is not the batch-wise training NTR implementation.
There are no error bars or p values: dots are descriptive means on a fixed conditional set,
not independent training replicates. All 12 categories are retained in their fixed order.
{audit['category_hard_means_decreased']}/12 category hard-normal means decrease. Macro hard-normal
mean: {audit['hard_macro_mean_ces']:.8f} -> {audit['hard_macro_mean_ntr']:.8f}.
Raw logit scores are not shifted, rescaled, sigmoid-transformed or calibrated.

## Interpretation boundary
The figure describes source-normal score tails and category-wise conditional score changes.
It does not prove anomaly rankings improve, anomalies are unaffected, or causal suppression
of target-domain false positives. Target localization evidence remains in the matched component
ablation and localization images. No anomalous-score control is claimed in this figure.
No training, CLIP encoding, benchmark evaluation, main-table or Word change was performed.
PNG600/PDF/SVG, the locked sample, all scores, paired means, exact ECDF and source audit are saved.
'''
    (OUT/'FIG5_AUDIT_AND_DEFINITIONS.md').write_text(notes,encoding='utf-8')
    ev.savejson(OUT/'FIG5_DELIVERY.json',dict(status='PASS',normal_patch_ids=12000,category_count=12,
        hard_patch_ids=1200,category_hard_means_decreased=audit['category_hard_means_decreased'],
        data_seal=identity(OUT/'DATA_SEALED.json'),
        figure_files=[identity(stem.with_suffix(ext)) for ext in ('.png','.pdf','.svg')],
        manuscript_changed=False,training=False))
    print('FIG 5 COMPLETE',OUT,flush=True)
    print(paired.to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('all','prepare','plot'),default='all')
    args=parser.parse_args()
    if args.stage in ('all','prepare'):prepare()
    if args.stage in ('all','plot'):plot()
