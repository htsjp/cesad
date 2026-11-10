"""Final layout for the immutable all-category source-normal tail data.

The initial data-generator file is kept unchanged because its hash is sealed.
This revision changes layout and explicit upper-20%-window drawing only.
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import plot_fig5_source_normal_tails_20261103 as data
from experiments import final_unified_evaluation as ev


def main():
    if not (data.OUT/'DATA_SEALED.json').is_file():data.prepare()
    seal=ev.readjson(data.OUT/'DATA_SEALED.json')
    for rec in seal['data_files']:data.check(rec)
    for rec in seal['input_identities']:
        if rec['path']==str(Path(data.__file__)):data.check(rec)
    df=pd.read_csv(data.OUT/'FIG5_PATCH_SCORES.csv')
    paired=pd.read_csv(data.OUT/'FIG5_CATEGORY_PAIRED.csv')
    curves=pd.read_csv(data.OUT/'FIG5_SURVIVAL_CURVES.csv')
    audit=ev.readjson(data.OUT/'DATA_AUDIT.json')
    assert paired.category.tolist()==list(data.CATEGORIES)
    assert len(df)==12000 and df.hard_normal.sum()==1200
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'pdf.fonttype':42,
        'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.7,'axes.labelsize':10,'xtick.labelsize':8,
        'ytick.labelsize':8,'legend.frameon':False})
    fig,axes=plt.subplots(1,2,figsize=(11.2,4.8))
    fig.subplots_adjust(left=.075,right=.985,bottom=.31,top=.80,wspace=.27)
    a,b=axes;colors=[data.SLATE,data.TEAL];names=['Trained CES','CES + NTR']
    handles=[]
    for column,color,name in zip(('survival_ces','survival_ntr'),colors,names):
        # Only omit points outside the declared probability display window, not observations.
        # Full scores and the exact unrestricted empirical curves remain in the sealed CSVs.
        visible=curves[column]<=.20
        h,=a.step(curves.loc[visible,'threshold'],curves.loc[visible,column],where='post',
                  lw=1.7,color=color,label=name)
        handles.append(h)
    xmin=min(float(np.quantile(df.q_ces,.8)),float(np.quantile(df.q_ntr,.8)))
    xmax=float(max(df.q_ces.max(),df.q_ntr.max()));pad=.04*(xmax-xmin)
    a.set(xlim=(xmin-pad,xmax+pad),ylim=(0,.20),xlabel='Anomaly score q',ylabel='Normal-patch exceedance')
    a.yaxis.set_major_formatter(PercentFormatter(1,decimals=0));a.set_yticks([0,.05,.10,.15,.20])
    x=np.arange(12)
    for i,r in paired.iterrows():
        b.plot([i-.07,i+.07],[r.hard_mean_ces,r.hard_mean_ntr],color='#B8BEC4',lw=.9,zorder=1)
    b.scatter(x-.07,paired.hard_mean_ces,s=28,color=data.SLATE,zorder=3)
    b.scatter(x+.07,paired.hard_mean_ntr,s=28,color=data.TEAL,marker='s',zorder=3)
    b.set(xlim=(-.65,11.65),ylabel='Mean anomaly score q',xlabel='VisA source category')
    b.set_xticks(x,data.CATEGORIES,rotation=90,ha='center')
    for ax in axes:
        ax.grid(axis='y',color='#D5DADF',lw=.5,alpha=.55);ax.set_axisbelow(True)
    a.set_title('(a) Normal-score upper tails',fontsize=10,pad=8)
    b.set_title('(b) Fixed hard-normal subsets',fontsize=10,pad=8)
    fig.suptitle('Effect of NTR on source-normal score tails',fontsize=12,y=.98)
    fig.legend(handles,names,loc='upper center',bbox_to_anchor=(.53,.925),ncol=2,fontsize=9)
    fig.canvas.draw()
    data.require_matplotlib_panel_alignment(fig,json_out=data.OUT/'FIG5_ALIGNMENT.json',strict=True)
    stem=data.OUT/'fig5_source_normal_tails'
    fig.savefig(stem.with_suffix('.png'),dpi=600,bbox_inches='tight')
    fig.savefig(stem.with_suffix('.pdf'),bbox_inches='tight')
    fig.savefig(stem.with_suffix('.svg'),bbox_inches='tight')
    fig.savefig(data.OUT/'fig5_source_normal_tails_preview.png',dpi=300,bbox_inches='tight')
    plt.close(fig)
    text=data.audit_pdf_text.audit_pdf(stem.with_suffix('.pdf').read_bytes(),minimum_pt=5.)
    collisions=data.audit_figure_collisions.audit_pdf(stem.with_suffix('.pdf'))
    ev.savejson(data.OUT/'FIG5_PDF_QA.json',dict(text=text,collisions=collisions))
    assert text['auditable'] and text['below_minimum_count']==0
    assert collisions['verdict']=='PASS',collisions
    caption='''Fig. 5. Effect of NTR on source-normal score tails. The analysis uses 1,000 GT-normal patches per VisA category, sampled from 50 source images per category with 20 patches per image (sampling seed 0). Both trained seed-0 models share frozen source feature preparation, initialization, actual batches and a 30-update budget; NTR is disabled or enabled. (a) Empirical exceedance Pr(q > t | y = 0), displayed over the upper 20% probability range without density smoothing. Equal category quotas give each category equal weight. (b) Mean scores on the fixed top-10% normal subset selected within each category by the trained CES model without NTR. Both models score exactly the same patch IDs; lines connect paired category means. Normal patches include normal regions of source images containing anomalies. These descriptive source-normal responses do not by themselves establish anomalous-patch discrimination or target localization improvement.'''
    (data.OUT/'CAPTION.md').write_text(caption+'\n',encoding='utf-8')
    notes=f'''# Fig. 5 all-category source-normal tail diagnostic

## Model and source identities
Trained CES: `{audit['trained_CES']}`.
Trained CES+NTR: `{audit['trained_CES_NTR']}`.
Both use certified Full VisA preparation, seed0, final update30, shared initial raw states,
frozen m/alpha, optimizer parameter groups, actual batch IDs and source inputs. Shared BCE+Dice
is used, with NTR disabled or rho=0.10. Neither model is init_v fallback. This is the current
20261101 reciprocal component family, not automatically an older historical Table IV family.
No LRP is applied to source patch responses; no target maps are being compared in this figure.

## Locked sample
12 categories x 50 images x 20 GT-normal patches = 12,000 patches from 600 distinct source images.
Normal labels come from the certified physical nearest-ID-mask mapping; anomalous-image normal
regions are eligible ({audit['normal_patches_from_anomalous_images']} sampled patches). Quotas,
seed0, candidate eligibility and category order were fixed before scoring. Images are uniformly
sampled per category without replacement; within each image 20 true-normal IDs are uniformly
sampled without replacement. No source image can contribute more than 20 selected patches.
Both models score the same physical IDs. Full source file and label hashes, selected physical
image hashes and feature-row hashes are validated. The pre-score sample manifest is preserved.

## Correct feature path and exact scoring
Call source_build_traceable_visa.pool_features with the actual norm_np, then the loaded frozen
rank4 E2 AdapterToState.adapt, then actual CESHead.forward. Thus raw normalized FP32 CLIP patches
undergo source PC1 removal -> NumPy L2 -> E2 residual adapter -> output L2 before CES scoring.
No raw untransformed feature is scored; no sigmoid, rescaling, shifting or calibration.
On {audit['source_pool_parity_patches']} overlapping physical patches, transformed features match
the actual CES training pool within max absolute error {audit['source_pool_transform_parity_max_abs']:.10g}.
Exact function/file/line traces and model/input hashes appear in DATA_AUDIT.json/DATA_SEALED.json.

## Panel definitions
(a) Strict empirical survival P(q>t), implemented by searchsorted(side=right). No KDE/bins.
Equal category quotas make pooled exceedance identical to category-macro exceedance. The
unrestricted exact curve is saved. The figure shows only y in [0,0.20], starting at the lesser
empirical 80th percentile (fixed 4% span margin) and ending beyond the maximum observed score.
No high-score observation is omitted. The 20% value is only a display rule.
(b) Per category, choose exactly 100/1000 patches by descending trained-no-NTR q; exact ties
resolve by source_position then patch_index. The fixed 1,200 IDs are reused for both models.
This category-wise post-hoc diagnostic is not the training batch-wise NTR selection algorithm.
All 12 categories are shown without reordering by improvement. Dots are descriptive conditional
means, not independent training replicates, and do not warrant seed-SD bars or significance tests.

## Observed result and boundary
Hard-normal means decrease for {audit['category_hard_means_decreased']}/12 categories.
Equal-category hard-normal mean: {audit['hard_macro_mean_ces']:.8f} -> {audit['hard_macro_mean_ntr']:.8f}.
This describes source-normal tail scores, not proof of unaffected anomaly scores, improved
rankings, causal mechanisms or deployed false-alarm rates. Target localization is evaluated
by the matched component ablation and the localization figures, not inferred from this plot.
No anomaly-score control is claimed; no new training, CLIP extraction, target evaluation,
paper-table changes or Word modification occurred.

## Reproduction and QA
Run `experiments/redraw_fig5_source_normal_tails_20261103.py` to reproduce from sealed CSVs.
The original data-generator file is preserved unchanged because its hash belongs to the data
seal; its initial layout failed collision QA. This final renderer only changes plot layout and
explicit drawing of the declared probability window. All saved scores/samples are unchanged.
The final PDF passes >=5pt glyph, collision and 1.5pt panel-alignment checks.
'''
    (data.OUT/'FIG5_AUDIT_AND_DEFINITIONS.md').write_text(notes,encoding='utf-8')
    ev.savejson(data.OUT/'FIG5_DELIVERY.json',dict(status='PASS',normal_patch_ids=12000,
        category_count=12,hard_patch_ids=1200,category_hard_means_decreased=audit['category_hard_means_decreased'],
        data_seal=data.identity(data.OUT/'DATA_SEALED.json'),renderer=data.identity(Path(__file__)),
        figure_files=[data.identity(stem.with_suffix(ext)) for ext in ('.png','.pdf','.svg')],
        manuscript_changed=False,training=False))
    print('FIG5 PASS',data.OUT,flush=True)
    print(paired.to_string(index=False),flush=True)


if __name__=='__main__':main()
