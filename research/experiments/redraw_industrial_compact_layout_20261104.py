"""Layout-only revision: category names, compact groups, captions underneath."""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import redraw_industrial_comparison_per_image_20261104 as display
from experiments import final_unified_evaluation as ev

BASE=display.original
OUT=display.OUT/'compact_layout'/'english_labels'
CN=FontProperties(fname='C:/Windows/Fonts/simsun.ttc',size=12)
DISPLAY_CATEGORY_NAMES={('VisA','pcb1'):'PCB',('BTAD','01'):'Ring'}


def category_panel(fig,spec,rows,controlled=False):
    headers=['','Input','GT','Trained CES','CES + NTR'] if controlled else ['','Input','GT',*BASE.METHODS]
    grid=spec.subgridspec(4,len(headers),width_ratios=[.75]+[1]*(len(headers)-1),
                         wspace=.018,hspace=.022)
    axes=[]
    for i,row in enumerate(rows):
        first=row[0] if controlled else row['CESAD']
        label=fig.add_subplot(grid[i,0]);label.set_axis_off()
        record=first['record']
        category=DISPLAY_CATEGORY_NAMES.get((record['target'],record['category']),
                                            record['category'].replace('_',' '))
        label.text(.98,.5,category,transform=label.transAxes,ha='right',va='center',fontsize=9)
        arrays=[first['image'],first['mask']]
        arrays += [BASE.overlay(s) for s in row] if controlled else [BASE.overlay(row[m]) for m in BASE.METHODS]
        for j,array in enumerate(arrays,1):
            ax=fig.add_subplot(grid[i,j]);BASE.base.show(ax,array,'mask' if j==2 else 'image')
            if i==0:ax.set_title(headers[j],fontsize=8.5,pad=4)
            axes.append(ax)
    return axes


def render(comparison,ntr,mode,english=False):
    if mode=='combined':
        fig=plt.figure(figsize=(18.4,5.35))
        fig.subplots_adjust(left=.005,right=.997,bottom=.09,top=.93)
        outer=fig.add_gridspec(1,2,width_ratios=[9.75,4.75],wspace=.008)
        axes=category_panel(fig,outer[0],comparison)+category_panel(fig,outer[1],ntr,True)
        specs=[outer[0],outer[1]];roles=['comparison','ntr']
    else:
        fig=plt.figure(figsize=(12.0,5.35) if mode=='comparison' else (6.25,5.35))
        fig.subplots_adjust(left=.005,right=.997,bottom=.09,top=.93)
        outer=fig.add_gridspec(1,1)
        axes=category_panel(fig,outer[0],comparison if mode=='comparison' else ntr,mode!='comparison')
        specs=[outer[0]];roles=[mode]
    labels_cn={'comparison':'(a) 与模型对比的可视化','ntr':'(b) NTR组件的可视化'}
    labels_en={'comparison':'(a) Visualization of model comparisons','ntr':'(b) Visualization of the NTR component'}
    for spec,role in zip(specs,roles):
        pos=spec.get_position(fig);center=.5*(pos.x0+pos.x1)
        if english:
            fig.text(center,.016,labels_en[role],ha='center',va='bottom',fontsize=12)
        else:
            fig.text(center,.016,labels_cn[role],ha='center',va='bottom',fontproperties=CN)
    fig.canvas.draw()
    suffix='_en' if english else ''
    name='industrial_sota_ntr_'+mode+'_compact'+suffix
    BASE.require_matplotlib_panel_alignment(fig,axes=axes,json_out=OUT/f'{name}.alignment.json',strict=True)
    fig.savefig(OUT/f'{name}.png',dpi=600,bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/f'{name}.svg',bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/f'{name}_preview.png',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    text=BASE.audit_pdf_text.audit_pdf((OUT/f'{name}.pdf').read_bytes(),minimum_pt=5.)
    collisions=BASE.audit_figure_collisions.audit_pdf(OUT/f'{name}.pdf')
    ev.savejson(OUT/f'{name}.pdf_qa.json',dict(text=text,collisions=collisions))
    assert text['auditable'] and text['below_minimum_count']==0 and collisions['verdict']=='PASS',collisions
    print('COMPLETE',name,flush=True)
    return dict(name=name,files=[BASE.base.identity(OUT/f'{name}.{ext}') for ext in ('png','pdf','svg')])


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    display.OUT=OUT
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    comparison,ntr=display.load()
    # Check the display arrays against the preceding version, not just raw-score identity.
    for group in comparison:
        for sample in group.values():
            q=sample['native'];assert sample['limits']==(float(q.min()),float(q.max()))
    figures=[render(comparison,ntr,mode,english=True) for mode in ('combined','comparison','ntr')]
    ev.savejson(OUT/'LAYOUT_ONLY_VERIFICATION.json',dict(status='PASS',
        category_labels_only=True,dataset_names_in_plot=False,sample_IDs_in_plot=False,
        dataset_category_header=False,explanatory_footer=False,
        panel_captions='English only, below panels',comparison_per_image_display_unchanged=True,
        display_labels={'VisA/pcb1':'PCB','BTAD/01':'Ring'},
        ring_label='visual description only; official source calls the category Product 1',
        NTR_shared_raw_scale_unchanged=True,raw_predictions_unchanged=True,
        colormap=BASE.CMAP,overlay_alpha=BASE.ALPHA,training=False,model_inference=False,
        script=BASE.base.identity(Path(__file__)),figures=figures))
    (OUT/'LAYOUT_NOTES.md').write_text('''# Compact figure layout

Only category names remain in the first text column; dataset names, sample IDs and
the Dataset/category header were removed. Category metadata remains in the CSV.
The explanatory footer is removed from the image, not from the required figure caption:
the main comparison still applies per-image min-max identically to all methods;
NTR pairs still share the original raw-logit ranges, with fixed LRP on in both arms.
The previous display-policy caption remains applicable and should accompany publication.
No score, image, GT, selected sample, colormap or opacity changed.
The two groups have a narrower gap and the (a)/(b) captions are below the panels.
All three final figures use English panel captions only. The PCB display label replaces
the internal VisA pcb1 category code. Ring is a visual description of the annular BTAD
sample, not an official component identity: the BTAD author repository names it only
Product 1 (https://github.com/pankajmishra000/VT-ADL). No bearing/washer identity is claimed.
Internal sample/category IDs stay unchanged in the manifest. Old figures are preserved.
''',encoding='utf-8')


if __name__=='__main__':main()
