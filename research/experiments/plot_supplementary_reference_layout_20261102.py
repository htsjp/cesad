"""Reference-style 4 rows x 8 samples, using unchanged saved case data only."""
from pathlib import Path
import sys
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import final_unified_evaluation as ev
from experiments import plot_supplementary_qualitative_20261102 as base
sys.path.insert(0,str(base.QA))
from audit_panel_alignment import require_matplotlib_panel_alignment
import audit_pdf_text
import audit_figure_collisions

SOURCE=ROOT/'results/supplementary_qualitative_expanded_20261102'
OUT=SOURCE/'reference_layout'
CMAP='jet'  # Match the supplied reference's display palette, not its predictions.
ALPHA=.40


def blend(image,q,norm,mask=None):
    color=plt.get_cmap(CMAP)(norm(q))[...,:3]
    rgb=(1-ALPHA)*image.astype(float)/255.+ALPHA*color
    if mask is not None:
        from scipy.ndimage import binary_erosion
        rgb[mask & ~binary_erosion(mask,structure=np.ones((3,3)))]=1.
    return np.clip(rgb,0,1)


def draw(name,rows):
    assert len(rows)==8 and len({r['sample_id'] for r in rows})==8
    family=rows[0]['figure']
    if family=='S2_NTR':
        rowlabels=['Input','GT Mask','Trained CES\n(LRP on)','CES + NTR\n(LRP on)']
    elif family=='S2_LRP':
        rowlabels=['Input','GT Mask','Global only\n(NTR on)','Global + local\n(NTR on)']
    elif family=='S3':
        rowlabels=['Input','GT Mask','CESAD overlay','GT outline\noverlay']
    else:
        rowlabels=['Input','GT Mask','CESAD logits','Overlay']
    fig,axes=plt.subplots(4,8,figsize=(12.6,6.75))
    fig.subplots_adjust(left=.052,right=.995,bottom=.105,top=.925,wspace=.025,hspace=.025)
    image_axes=list(axes.ravel())
    for j,row in enumerate(rows):
        assert ev.sha(row['image_path'])==row['image_sha256']
        assert ev.sha(row['gt_path'])==row['gt_sha256']
        image=np.asarray(Image.open(row['image_path']).convert('RGB'))
        with np.load(row['selected_cache']) as z:
            assert str(z['sample_id'])==row['sample_id']
            mask=z['mask'];native=z['native_maps'];raw=z['maps74']
        assert np.array_equal(mask,np.asarray(Image.open(row['gt_path']).convert('L'))>0)
        assert mask.any() and image.shape[:2]==mask.shape
        assert all(np.array_equal(ev.resize(a,mask.shape),b) for a,b in zip(raw,native))
        norm=Normalize(float(row['vmin']),float(row['vmax']),clip=False)
        if family.startswith('S2'):
            arrays=[image,mask,blend(image,native[0],norm),blend(image,native[1],norm)]
            kinds=['image','mask','image','image']
        elif family=='S3':
            arrays=[image,mask,blend(image,native[0],norm),blend(image,native[0],norm,mask)]
            kinds=['image','mask','image','image']
        else:
            arrays=[image,mask,native[0],blend(image,native[0],norm)]
            kinds=['image','mask','map','image']
        for i,(array,kind) in enumerate(zip(arrays,kinds)):
            # The shared image helper preserves aspect ratio and never crops defects.
            if kind=='map':
                old=base.CMAP;base.CMAP=CMAP
                base.show(axes[i,j],array,kind,norm)
                base.CMAP=old
            else:base.show(axes[i,j],array,kind,norm)
        axes[0,j].set_title(f"{row['target']} / {row['category']}\n{Path(row['image_path']).stem}",
                            fontsize=7,pad=5)
        row['display_column']=j+1
        row['display_row_roles']=' | '.join(rowlabels)
        row['display_cmap']=CMAP;row['overlay_alpha']=ALPHA
        # A narrow shared scale below each sample explicitly discloses column limits.
    for i,label in enumerate(rowlabels):
        axes[i,0].set_ylabel(label,fontsize=8,labelpad=9)
    fig.canvas.draw()
    for j,row in enumerate(rows):
        p=axes[3,j].get_position()
        ca=fig.add_axes([p.x0+.07*p.width,.060,.86*p.width,.009])
        lo,hi=float(row['vmin']),float(row['vmax'])
        cb=fig.colorbar(ScalarMappable(norm=Normalize(lo,hi),cmap=CMAP),cax=ca,orientation='horizontal')
        cb.set_ticks([lo,hi]);cb.ax.set_xticklabels([f'{lo:.3f}',f'{hi:.3f}'],fontsize=6)
        cb.ax.tick_params(length=1.5,width=.4,pad=1);cb.outline.set_linewidth(.35)
    fig.text(.53,.008,'Raw-logit limits; shared within each column comparison',
             ha='center',fontsize=7)
    require_matplotlib_panel_alignment(fig,axes=image_axes,
        json_out=OUT/f'{name}.alignment.json',strict=True)
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/f'{name}.svg',bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/f'{name}.png',dpi=600,bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/f'{name}_preview.png',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    text=audit_pdf_text.audit_pdf((OUT/f'{name}.pdf').read_bytes(),minimum_pt=5.)
    collision=audit_figure_collisions.audit_pdf(OUT/f'{name}.pdf')
    assert text['auditable'] and text['below_minimum_count']==0
    assert collision['verdict']=='PASS',collision
    return dict(name=name,rows=4,columns=8,text_audit=text,collision_audit=collision,
                files=[base.identity(OUT/f'{name}.{ext}') for ext in ('png','pdf','svg')])


def main():
    import torch
    torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Arial','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
    rows=ev.readcsv(SOURCE/'CASE_MANIFEST.csv');assert len(rows)==48
    plates={}
    for row in rows:plates.setdefault(row['plate'],[]).append(dict(row))
    assert len(plates)==6
    checks=[];manifest=[]
    for name,group in plates.items():
        group.sort(key=lambda r:int(r['row_number']))
        checks.append(draw(name,group));manifest.extend(group)
        print('COMPLETE',name,'4 rows x 8 columns',flush=True)
    ev.writecsv(OUT/'DISPLAY_MANIFEST.csv',manifest)
    ev.savejson(OUT/'LAYOUT_VERIFICATION.json',dict(status='PASS',checks=checks,
        cases=48,samples_unchanged=True,training=False,inference=False,
        score_transforms='NONE; visualization normalization uses saved shared limits only',
        input_manifest=base.identity(SOURCE/'CASE_MANIFEST.csv'),script=base.identity(Path(__file__))))
    (OUT/'README.md').write_text('''# Reference-style supplementary figures

Six plates, each 4 rows x 8 sample columns. These are transpositions of the same
48 saved case records, not new selected examples or model predictions.

S1 parts1-3: Input / GT / CESAD raw-logit heatmap / image overlay.
S2 NTR: Input / GT / trained no-NTR overlay / trained NTR overlay (LRP on in both).
S2 LRP: Input / GT / global-only overlay / global-plus-local overlay (same NTR head).
S3: Input / GT / CESAD overlay / overlay with true GT outline.

Jet is used only to match the supplied reference display palette. No sample, GT,
logit, inference setting or shared scale has changed. Overlay is fixed 0.4.
Column scales are printed: both compared models always share the same raw-logit
range within that column. S1/S3 additionally retain all earlier source-family
scale sharing. Different displayed ranges must not be read as absolute inter-model
calibration. No per-image min-max, score offsets, sigmoid or percentile clipping.
Aspect ratios are preserved without GT-driven crops; whitespace is letterboxing.

All historical source/checkpoint limitations and explicit post-hoc failure-case
selection from the parent README_AND_CAPTIONS.md remain applicable. S2 uses the
certified reciprocal rerun, not automatically an older Table IV checkpoint family.
The printed pixel AP and identities remain in the parent CASE_MANIFEST.csv.
All final PDFs pass text-size, collision and image-panel alignment checks.
''',encoding='utf-8')


if __name__=='__main__':main()
