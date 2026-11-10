"""Pinned industrial cases: existing method-specific predictions plus matched NTR.

No training, model forward, sample search, target tuning or benchmark recomputation.
Color normalization is display-only, fixed per method across all four named examples.
NTR maps share raw-logit limits within each source family, with LRP on in both arms.
"""
from pathlib import Path
import sys
import json
import gc
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import final_unified_evaluation as ev
from experiments import plot_supplementary_qualitative_20261102 as base
sys.path.insert(0,str(base.QA))
from audit_panel_alignment import require_matplotlib_panel_alignment
import audit_pdf_text
import audit_figure_collisions

OUT=ROOT/'results/industrial_sota_ntr_composite_20261104'
RECIP=ROOT/'results/ntr_lrp_reciprocal_aligned_20261101'
TRACE=ROOT/'results/visa_source_traceable_rebuild/formal'
WORK=Path('E:/MRAD_visa_source_traceable_rebuild/target_evaluation')
METHODS=('AnomalyCLIP','FAPrompt','MRAD-FT','MRAD-CLIP','VisualAD','FB-CLIP','CESAD')
COMPARISON_IDS=('VisA/pcb1/Data/Images/Anomaly/070.JPG',
    'MVTec/wood/test/scratch/010.png','BTAD/01/test/ko/0047.bmp',
    'MPDD/tubes/test/anomalous/008.png')
NTR_IDS=('VisA/candle/Data/Images/Anomaly/079.JPG',
    'MVTec/metal_nut/test/color/020.png','VisA/chewinggum/Data/Images/Anomaly/039.JPG',
    'MVTec/grid/test/broken/009.png')
CMAP='jet'
ALPHA=.40
assets={};records=[]


def asset(path,expected=None):
    path=Path(path);key=str(path.resolve())
    if key not in assets:assets[key]=base.identity(path)
    if expected is not None:assert assets[key]['sha256']==expected,path
    return assets[key]


def image_mask(row):
    asset(row['img_path'],row['image_sha256']);asset(row['gt_path'],row['gt_sha256'])
    im=np.asarray(Image.open(row['img_path']).convert('RGB'));mask=ev.native_mask(row)
    assert mask.any() and im.shape[:2]==mask.shape,row['sample_id']
    return im,mask


def row_for(rows,sid):
    matching=[r for r in rows if r['sample_id']==sid]
    assert len(matching)==1,sid
    return matching[0]


def save_map(section,method,row,q,cp,cache,index,postprocessing,seed='UNKNOWN',extra=None):
    assert np.isfinite(q).all() and np.asarray(q).ndim==2
    im,mask=image_mask(row);native=ev.resize(q,mask.shape)
    path=OUT/'selected_data'/f'{section}_{row["dataset"]}_{method.replace(" ","_").replace("+","plus")}.npz'
    # A target may occur twice in NTR; include the physical sample identifier in the filename.
    path=path.with_name(path.stem+'_'+str(row['index'])+'.npz')
    np.savez_compressed(path,raw_map=q,native_map=native,mask=mask,sample_id=row['sample_id'])
    rec=dict(section=section,method=method,source=ev.SOURCE[row['dataset']],target=row['dataset'],
        category=row['category'],sample_id=row['sample_id'],image_path=row['img_path'],
        image_sha256=row['image_sha256'],gt_path=row['gt_path'],gt_sha256=row['gt_sha256'],
        positive_pixels=int(mask.sum()),checkpoint=cp,seed=seed,cache_path=str(cache),cache_index=index,
        original_map_shape=str(np.asarray(q).shape),native_shape=str(mask.shape),
        resize='official unified FP32 bilinear align_corners=True to original paired GT',
        original_postprocessing=postprocessing,extra_smoothing='NONE',
        selected_map_cache=str(path),raw_min=float(np.min(q)),raw_max=float(np.max(q)),
        display_cmap=CMAP,overlay_alpha=ALPHA,**(extra or {}))
    records.append(rec)
    return dict(image=im,mask=mask,native=native,record=rec)


def paired_maps(sid,section,need_no_ntr):
    target=sid.split('/')[0];folder=RECIP/'work/predictions'/target
    done=ev.readjson(folder/'COMPLETE.json')
    for key in ('maps','manifest','head_order'):asset(done[key]['path'],done[key]['sha256'])
    assert done['source_seal_sha256']==ev.sha(RECIP/'formal/SOURCE_SEALED.json')
    rows=ev.readcsv(folder/'MANIFEST.csv');row=row_for(rows,sid);idx=int(row['index'])
    entries=ev.readjson(folder/'HEAD_ORDER.json')
    yes=next(i for i,e in enumerate(entries) if e['seed']==0 and e['rho']==.1)
    no=next(i for i,e in enumerate(entries) if e['seed']==0 and e['rho'] is None)
    cps=[];initial=[]
    for i in ((no,yes) if need_no_ntr else (yes,)):
        cp=entries[i]['checkpoint'];asset(cp['path'],cp['sha256'])
        state=torch.load(cp['path'],map_location='cpu',weights_only=False)
        assert state['step']==30 and state['seed']==0 and state['rho']==entries[i]['rho']
        assert state['config_sha256']==ev.sha(RECIP/'formal/LOCKED_CONFIG.json')
        cps.append(state)
        init=Path(cp['path']).with_name('update_000.pt');asset(init)
        initial.append(torch.load(init,map_location='cpu',weights_only=False)['raw_model'])
    if need_no_ntr:
        assert np.array_equal(cps[0]['batch_indices'],cps[1]['batch_indices'])
        assert cps[0]['source_input_sha256']==cps[1]['source_input_sha256']
        assert cps[0]['optimizer']['param_groups']==cps[1]['optimizer']['param_groups']
        assert all(torch.equal(initial[0][k],initial[1][k]) for k in initial[0])
        assert all(torch.equal(cps[0]['raw_model'][k],cps[1]['raw_model'][k]) for k in ('m','alpha'))
    maps=np.load(folder/'global_local74.npy',mmap_mode='r')
    result=[]
    for i,label in ((no,'Trained CES'),(yes,'CES + NTR')) if need_no_ntr else ((yes,'CESAD'),):
        global74,local74=maps[idx,i];q=.5*global74+.5*local74
        result.append(save_map(section,label,row,q,entries[i]['checkpoint']['path'],
            folder/'global_local74.npy',idx,'fixed 2x2 LRP, 0.5 global + 0.5 local; no Gaussian',0,
            dict(lrp=True,ntr=entries[i]['rho'],head_index=i,weights_family='certified 20261101 reciprocal rerun')))
    return result


def certified_ces(sid):
    target=sid.split('/')[0];folder=WORK/'predictions'/target
    done=ev.readjson(folder/'COMPLETE.json');asset(folder/'COMPLETE.json')
    assert done['stamp']['endpoint_order']==['initialization','update30']
    asset(done['maps']['path'],done['maps']['sha256'])
    manifest=done['stamp']['manifest'];asset(manifest['path'],manifest['sha256'])
    rows=ev.readcsv(manifest['path']);row=row_for(rows,sid);i=int(row['index'])
    cfg=ev.readjson(TRACE/'target_evaluation_review/TARGET_EVALUATION_CONFIG.json')
    cp=cfg['source_inputs']['endpoints']['update30'];asset(cp['path'],cp['sha256'])
    maps=np.load(done['maps']['path'],mmap_mode='r');q=np.asarray(maps[i,1]).copy()
    return save_map('comparison','CESAD',row,q,cp['path'],done['maps']['path'],i,
        'fixed 2x2 LRP 0.5 global + 0.5 local; no Gaussian',0,
        dict(lrp=True,ntr=.1,endpoint='update30',weights_family='certified rebuilt Full VisA source'))


def extract():
    registry_path=ev.WORK/'PREDICTION_REGISTRY.json';asset(registry_path)
    registry=ev.readjson(registry_path)
    comparison=[]
    for sid in COMPARISON_IDS:
        target=sid.split('/')[0];group={}
        for method in METHODS[:-1]:
            e=next(e for e in registry if e['method']==method and e['target']==target)
            assert e['source']==ev.SOURCE[target]
            asset(e['manifest'],e['manifest_sha256']);asset(e['checkpoint_file'],e['checkpoint_sha256'])
            rows=ev.readcsv(e['manifest']);row=row_for(rows,sid);i=int(row['index'])
            cp=Path(e['cache_path']);ci=e['canonical_to_cache_index'][i]
            if cp.is_file():asset(cp,e['cache_sha256'])
            else:
                complete=ev.readjson(cp/'COMPLETE.json');asset(cp/'COMPLETE.json')
                assert complete['checkpoint_sha256']==e['checkpoint_sha256']
                assert complete['manifest_sha256']==e['manifest_sha256']
                chunk=cp/f'chunk_{ci//16:05d}.npz'
                asset(chunk,complete['chunk_hashes'][chunk.name])
            store=ev.Store(e);q=store[i].copy();del store;gc.collect()
            group[method]=save_map('comparison',method,row,q,e['checkpoint'],e['cache_path'],ci,
                                  e['postprocessing'])
            print('CACHED',method,sid,flush=True)
        group['CESAD']=(paired_maps(sid,'comparison',False)[0] if target in ('VisA','MVTec')
                        else certified_ces(sid))
        assert len({x['record']['image_sha256'] for x in group.values()})==1
        assert len({x['record']['gt_sha256'] for x in group.values()})==1
        comparison.append(group)
    ntr=[paired_maps(sid,'NTR',True) for sid in NTR_IDS]
    # All current VisA-source NTR predictions share the exact sealed update30 raw states.
    parity=ev.readjson(RECIP/'formal/VISA_SOURCE_ANCHOR_PARITY.json')
    assert parity['max_raw_parameter_error']==0
    asset(parity['reference']['path'],parity['reference']['sha256'])
    asset(parity['new']['path'],parity['new']['sha256'])
    scales=[]
    for method in METHODS:
        lo=min(float(g[method]['native'].min()) for g in comparison)
        hi=max(float(g[method]['native'].max()) for g in comparison)
        assert hi>lo
        scales.append(dict(section='comparison',method=method,source='both fixed source models',
                           vmin=lo,vmax=hi,policy='one method-wise full-extrema range across all four user-pinned cases; display only'))
        for g in comparison:
            g[method]['limits']=(lo,hi);g[method]['record'].update(display_vmin=lo,display_vmax=hi)
    for src in ('Full MVTec','Full VisA'):
        family=[s for pair in ntr for s in pair if s['record']['source']==src]
        lo=min(float(s['native'].min()) for s in family);hi=max(float(s['native'].max()) for s in family)
        scales.append(dict(section='NTR',method='both trained heads',source=src,vmin=lo,vmax=hi,
            policy='shared full-extrema raw-logit range across both arms and both pinned cases of this source'))
        for s in family:s['limits']=(lo,hi);s['record'].update(display_vmin=lo,display_vmax=hi)
    ev.writecsv(OUT/'COLOR_SCALES.csv',scales)
    ev.writecsv(OUT/'CASE_METHOD_MANIFEST.csv',records)
    ev.savejson(OUT/'INPUT_IDENTITIES.json',list(assets.values()))
    return comparison,ntr


def overlay(sample):
    lo,hi=sample['limits'];norm=Normalize(lo,hi,clip=False)
    color=plt.get_cmap(CMAP)(norm(sample['native']))[...,:3]
    return np.clip((1-ALPHA)*sample['image']/255.+ALPHA*color,0,1)


def panel(fig,spec,rows,controlled=False):
    headers=['Category','Input','GT','Trained CES','CES + NTR'] if controlled else ['Category','Input','GT',*METHODS]
    grid=spec.subgridspec(4,len(headers),width_ratios=[1.12]+[1]*(len(headers)-1),wspace=.025,hspace=.035)
    image_axes=[]
    for i,row in enumerate(rows):
        first=row[0] if controlled else row['CESAD'];r=first['record']
        values=[first['image'],first['mask']]
        values += [overlay(s) for s in row] if controlled else [overlay(row[m]) for m in METHODS]
        text=fig.add_subplot(grid[i,0]);text.set_axis_off()
        text.text(.97,.5,f"{r['target']}\n{r['category']}\nID {Path(r['image_path']).stem}",
                  transform=text.transAxes,ha='right',va='center',fontsize=8.5)
        if i==0:text.set_title('Dataset /\ncategory',fontsize=8,pad=5)
        for j,array in enumerate(values,1):
            ax=fig.add_subplot(grid[i,j]);base.show(ax,array,'mask' if j==2 else 'image')
            if i==0:ax.set_title(headers[j],fontsize=8.5,pad=5)
            image_axes.append(ax)
    return image_axes


def render(comparison,ntr,mode):
    if mode=='combined':
        fig=plt.figure(figsize=(19.4,6.1));fig.subplots_adjust(left=.006,right=.995,bottom=.105,top=.875)
        outer=fig.add_gridspec(1,2,width_ratios=[10.12,5.12],wspace=.035)
        axes=panel(fig,outer[0],comparison)+panel(fig,outer[1],ntr,True)
        fig.text(.33,.975,'(a) Industrial cross-dataset comparison',ha='center',va='top',fontsize=12,fontweight='bold')
        fig.text(.825,.975,'(b) NTR comparison (LRP on in both)',ha='center',va='top',fontsize=12,fontweight='bold')
    else:
        fig=plt.figure(figsize=(12.6,6.1) if mode=='comparison' else (7.1,6.1))
        fig.subplots_adjust(left=.008,right=.995,bottom=.105,top=.875)
        outer=fig.add_gridspec(1,1)
        axes=panel(fig,outer[0],comparison if mode=='comparison' else ntr,mode!='comparison')
        fig.text(.53,.975,'Industrial cross-dataset comparison' if mode=='comparison' else 'NTR comparison (LRP on in both)',
                 ha='center',va='top',fontsize=12,fontweight='bold')
    note=('Same colormap and overlay opacity. Method-wise color limits are fixed across cases; colors are not score-calibrated across methods.'
          if mode!='ntr' else 'Both trained heads use fixed 2x2 LRP; paired maps share source-family raw-logit limits.')
    fig.text(.53,.025,note,ha='center',fontsize=7.5)
    fig.canvas.draw()
    name='industrial_sota_ntr_'+mode
    require_matplotlib_panel_alignment(fig,axes=axes,json_out=OUT/f'{name}.alignment.json',strict=True)
    fig.savefig(OUT/f'{name}.png',dpi=600,bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/f'{name}.svg',bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/f'{name}_preview.png',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    text=audit_pdf_text.audit_pdf((OUT/f'{name}.pdf').read_bytes(),minimum_pt=5.)
    collision=audit_figure_collisions.audit_pdf(OUT/f'{name}.pdf')
    ev.savejson(OUT/f'{name}.pdf_qa.json',dict(text=text,collision=collision))
    assert text['auditable'] and text['below_minimum_count']==0
    assert collision['verdict']=='PASS',collision
    print('RENDERED',mode,flush=True)
    return dict(name=name,files=[base.identity(OUT/f'{name}.{ext}') for ext in ('png','pdf','svg')])


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'selected_data').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    ev.savejson(OUT/'LOCKED_CASES.json',dict(comparison=COMPARISON_IDS,ntr=NTR_IDS,
        methods=METHODS,colormap=CMAP,overlay_alpha=ALPHA,training=False,new_model_inference=False,
        comparison_CESAD='current certified source model, not uncertified historical VisA head',
        same_LRP_in_both_NTR_arms=True))
    comparison,ntr=extract()
    results=[render(comparison,ntr,mode) for mode in ('combined','comparison','ntr')]
    note='''# Industrial SOTA and matched NTR qualitative comparison

## Fixed cases and models
All eight cases are explicitly requested by the author; no sample search or ranking is used.
MVTec wood010 is the scratch/010.png image, not a different defect with the same basename.
The six comparators are AnomalyCLIP, FAPrompt, MRAD-FT, MRAD-CLIP, VisualAD and FB-CLIP.
Each reads its own existing checkpoint-linked predictions. Registry, checkpoint, selected chunk
or complete cache hashes and canonical-to-cache indices are checked. No baseline inference is
needed, and one method's predictions are never substituted for another's.
Source matching is Full MVTec -> VisA and Full VisA -> MVTec/BTAD/MPDD.
Original method-specific fixed postprocessing is retained; no new smoothing is added.

CESAD uses the current certified seed0 trained source family. VisA/MVTec examples read the
20261101 reciprocal prediction cache; BTAD/MPDD read the independently sealed traceable source
update30 endpoint. The reciprocal VisA-source NTR states match that endpoint exactly, as
verified by the sealed zero-error anchor record. These replace uncertified historical VisA-head
identity for this new figure only; no historical main table or cached result is overwritten.

The NTR block uses the same trained no-NTR/NTR pair, shared actual source batches, initialization,
optimizer settings, m/alpha and 30-update endpoint. Both arms have LRP ON: exactly 0.5 global74 +
0.5 stitched 2x2 local74. No fallback initialization, image substitution or no-LRP arm is used.

## Geometry and color
Every input/GT hash and sample ID is checked. All masks are nonempty and at their original
paired resolution. Original method maps are bilinearly resized directly to native GT using
the unified FP32 align_corners=True rule; no intermediate 74-grid resize is inserted for a
baseline whose original output has a different resolution. Original aspect ratios are kept,
with white letterboxing only. No GT contours, zoom, defect-oriented crop or contrast edit.

All overlays use jet and alpha=0.40. The same palette does not imply calibrated raw scales:
the main block fixes one method-specific full-extrema display range across all four named
examples, without per-image min-max. These ranges are selected only for display, do not use
GT pixels and never feed inference, evaluation, hyperparameters or checkpoint selection.
NTR shares one raw-logit range within each source family across both arms and its two samples.
No percentile clipping or score rescaling is applied to the saved raw/native maps.
Absolute brightness is not comparable across different method display ranges. All numerical
limits are in COLOR_SCALES.csv and each map's record. This is descriptive visualization only.

## Layout and outputs
Combined: left 4 samples x (category + Input + GT + 6 baselines + CESAD), right 4 samples x
(category + Input + GT + trained CES + CES+NTR). Separate publication blocks are also exported
to avoid unreadable text when the combined wide figure is reduced to one paper column.
PNG600, PDF and editable-text SVG are provided. No Word, main-table metrics or weights changed.

## Suggested caption
Industrial cross-dataset qualitative comparisons and controlled NTR examples. (a) Four fixed
industrial examples with released GT and predictions from six comparison methods and CESAD,
using their source-matched checkpoints. (b) Four fixed examples comparing trained CES without
and with NTR; fixed LRP is enabled in both arms. All overlays use the same colormap and opacity.
Display ranges are fixed per method across the comparison examples and shared within each
source family for NTR pairs; colors are not calibrated anomaly probabilities across methods.
'''
    (OUT/'README_AND_CAPTION.md').write_text(note,encoding='utf-8')
    ev.savejson(OUT/'DELIVERY_VERIFICATION.json',dict(status='PASS',main_sample_count=4,
        NTR_sample_count=4,comparison_methods=METHODS,real_map_records=len(records),
        baseline_inference_run=False,training=False,both_NTR_arms_LRP=True,
        identity_files=len(assets),script=base.identity(Path(__file__)),figures=results))
    print('COMPLETE',OUT,flush=True)


if __name__=='__main__':main()
