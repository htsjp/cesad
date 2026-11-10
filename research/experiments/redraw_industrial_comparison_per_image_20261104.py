"""Common per-image display min-max for every SOTA method; raw NTR pairs unchanged.

This is a visualization variant, never a new prediction/evaluation normalization.
"""
from pathlib import Path
import sys
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import plot_industrial_sota_ntr_composite_20261104 as original
from experiments import final_unified_evaluation as ev

SOURCE=original.OUT
OUT=SOURCE/'per_image_display'


def load():
    records=ev.readcsv(SOURCE/'CASE_METHOD_MANIFEST.csv')
    assert len(records)==36
    items={};scales=[];inputs=[];unchanged_ntr=0
    for r in records:
        inputs.append(original.base.identity(r['selected_map_cache']))
        assert ev.sha(r['image_path'])==r['image_sha256']
        assert ev.sha(r['gt_path'])==r['gt_sha256']
        with np.load(r['selected_map_cache']) as z:
            assert str(z['sample_id'])==r['sample_id']
            raw=z['raw_map'];native=z['native_map'];mask=z['mask']
        assert np.array_equal(ev.resize(raw,mask.shape),native)
        assert np.array_equal(np.asarray(Image.open(r['gt_path']).convert('L'))>0,mask)
        image=np.asarray(Image.open(r['image_path']).convert('RGB'))
        assert image.shape[:2]==mask.shape and mask.any()
        old_limits=(float(r['display_vmin']),float(r['display_vmax']))
        limits=(float(native.min()),float(native.max())) if r['section']=='comparison' else old_limits
        assert limits[1]>limits[0]
        item=dict(image=image,mask=mask,native=native,limits=limits,record=r)
        if r['section']=='NTR':
            before=original.overlay(dict(item,limits=old_limits))
            after=original.overlay(item)
            assert np.array_equal(before,after)
            unchanged_ntr+=1
        r.update(display_vmin=limits[0],display_vmax=limits[1],
            display_normalization='per-image min-max; all comparison methods identically' if r['section']=='comparison'
                                  else 'unchanged shared source-family raw-logit range',
            scoring_or_evaluation_normalization='UNCHANGED',raw_prediction_unchanged=True)
        scales.append(dict(section=r['section'],method=r['method'],sample_id=r['sample_id'],
            source=r['source'],vmin=limits[0],vmax=limits[1],policy=r['display_normalization']))
        items[(r['section'],r['sample_id'],r['method'])]=item
    comparison=[{method:items[('comparison',sid,method)] for method in original.METHODS}
                for sid in original.COMPARISON_IDS]
    ntr=[[items[('NTR',sid,name)] for name in ('Trained CES','CES + NTR')]
         for sid in original.NTR_IDS]
    assert unchanged_ntr==8
    ev.writecsv(OUT/'CASE_METHOD_MANIFEST.csv',records)
    ev.writecsv(OUT/'COLOR_SCALES.csv',scales)
    ev.savejson(OUT/'REUSED_PREDICTION_IDENTITIES.json',inputs)
    return comparison,ntr


def render(comparison,ntr,mode):
    if mode=='combined':
        fig=plt.figure(figsize=(19.4,6.1));fig.subplots_adjust(left=.006,right=.995,bottom=.105,top=.875)
        outer=fig.add_gridspec(1,2,width_ratios=[10.12,5.12],wspace=.035)
        axes=original.panel(fig,outer[0],comparison)+original.panel(fig,outer[1],ntr,True)
        fig.text(.33,.975,'(a) Industrial cross-dataset comparison',ha='center',va='top',fontsize=12,fontweight='bold')
        fig.text(.825,.975,'(b) NTR comparison (LRP on in both)',ha='center',va='top',fontsize=12,fontweight='bold')
    else:
        fig=plt.figure(figsize=(12.6,6.1) if mode=='comparison' else (7.1,6.1))
        fig.subplots_adjust(left=.008,right=.995,bottom=.105,top=.875)
        outer=fig.add_gridspec(1,1)
        axes=original.panel(fig,outer[0],comparison if mode=='comparison' else ntr,mode!='comparison')
        fig.text(.53,.975,'Industrial cross-dataset comparison' if mode=='comparison' else 'NTR comparison (LRP on in both)',
                 ha='center',va='top',fontsize=12,fontweight='bold')
    note=('Per-image min-max for localization display, applied identically to all comparison methods. NTR pairs retain shared raw-logit limits.'
          if mode!='ntr' else 'Both trained heads use fixed 2x2 LRP; paired maps share source-family raw-logit limits.')
    fig.text(.53,.025,note,ha='center',fontsize=7.5)
    fig.canvas.draw();name='industrial_sota_ntr_'+mode
    original.require_matplotlib_panel_alignment(fig,axes=axes,json_out=OUT/f'{name}.alignment.json',strict=True)
    fig.savefig(OUT/f'{name}.png',dpi=600,bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/f'{name}.svg',bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/f'{name}_preview.png',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    text=original.audit_pdf_text.audit_pdf((OUT/f'{name}.pdf').read_bytes(),minimum_pt=5.)
    collision=original.audit_figure_collisions.audit_pdf(OUT/f'{name}.pdf')
    ev.savejson(OUT/f'{name}.pdf_qa.json',dict(text=text,collision=collision))
    assert text['auditable'] and text['below_minimum_count']==0 and collision['verdict']=='PASS'
    print('RENDERED',mode,flush=True)
    return dict(name=name,files=[original.base.identity(OUT/f'{name}.{ext}') for ext in ('png','pdf','svg')])


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    comparison,ntr=load()
    figures=[render(comparison,ntr,mode) for mode in ('combined','comparison','ntr')]
    (OUT/'DISPLAY_POLICY_AND_CAPTION.md').write_text('''# Display-only restoration

The previous main comparison fixed one display range per method across four images.
This variant instead maps every displayed comparison heatmap by
d(x) = (q(x) - min(q)) / (max(q) - min(q)).
The exact same rule applies to CESAD AND all six baselines, including maps that were already
normalized within their original inference pipeline. No manual intensity, percentile clipping,
extra smoothing, sigmoid, image substitution or per-method special treatment is introduced.
All min/max values are saved. Input/GT/raw/native prediction arrays remain identical.
This makes relative within-image localization visible; it does not establish absolute score
calibration, comparable response magnitudes or a fixed false-positive operating point.

NTR is deliberately NOT individually normalized: both trained models use fixed LRP and
the original shared raw-logit range within each source family. All eight NTR overlay arrays
were checked to be pixel-identical to the previous display implementation. Independent
normalization there would hide the response suppression that the comparison is meant to show.

All original case/checkpoint/source identities remain unchanged. This is a new display
directory; the old figure, predictions, checkpoints, tables and manuscript are untouched.
No model forward, training, inference or benchmark metric calculation was performed.

Suggested caption addition: Heatmaps in the method-comparison panel are independently
min-max normalized per image for localization display, using the same rule, colormap and
opacity for every method. Consequently, colors denote relative within-image response, not
calibrated anomaly scores across methods. NTR comparisons retain shared raw-logit color
limits within each source family, with fixed LRP enabled in both arms.
''',encoding='utf-8')
    ev.savejson(OUT/'DISPLAY_VERIFICATION.json',dict(status='PASS',samples_unchanged=True,
        raw_and_native_predictions_unchanged=True,all_methods_same_display_rule=True,
        NTR_overlay_arrays_unchanged=8,colormap='jet',overlay_alpha=.40,
        model_inference=False,training=False,metrics_unchanged=True,
        source_manifest=original.base.identity(SOURCE/'CASE_METHOD_MANIFEST.csv'),
        script=original.base.identity(Path(__file__)),figures=figures))
    print('COMPLETE',OUT,flush=True)


if __name__=='__main__':main()
