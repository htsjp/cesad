"""Cache-only completion for the two matched MVTec->VisA recipe controls.

Reuse the frozen paper evaluator and its established six-metric driver.
No training, encoder/head inference, map normalization or official-table edits.
"""
from pathlib import Path
import sys
import os
import gc
import time
import traceback
import json

import numpy as np
import torch
from sklearn.metrics import average_precision_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import final_unified_evaluation as ev
from experiments import evaluate_aligned_ablation_six_metrics as driver

OUT=ROOT/'results/mvtec_recipe_six_metrics_20261109'
WORK=Path('E:/MRAD_mvtec_recipe_six_metrics_20261109')
OLD=ROOT/'results/phase_mvtec_exact_visa_recipe_checked/formal'
CACHE=Path('E:/MRAD_mvtec_exact_visa_recipe_checked/seed0')
METRICS=('I_AUROC','I_AP','P_AUROC','P_F1max','P_AP','AUPRO')


def ident(path):
    path=Path(path)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=ev.sha(path))


class Tee:
    def __init__(self,original,log):self.original=original;self.log=log
    def write(self,text):self.original.write(text);self.log.write(text);self.log.flush()
    def flush(self):self.original.flush();self.log.flush()


def main():
    torch.set_num_threads(4)
    driver.OUT=OUT;driver.WORK=WORK
    OUT.mkdir(exist_ok=True);(WORK/'scratch').mkdir(parents=True,exist_ok=True)
    manifest=ev.WORK/'manifests/VisA.csv'
    rows=ev.readcsv(manifest)
    assert len(rows)==2162 and len({r['category'] for r in rows})==12
    evaluator_id=ev.evaluator_hash()
    assert evaluator_id=='22b3608b28e166fbe7a0a05d307b4777c34008360be59e4e2aa2258f383ec0cb'
    cats=sorted({r['category'] for r in rows})
    print('Verify canonical images and native GT files.',flush=True)
    for r in rows:
        assert ev.sha(r['img_path'])==r['image_sha256']
        if int(r['label']):assert ev.sha(r['gt_path'])==r['gt_sha256']
    known=ev.readcsv(OLD/'RESULTS.csv')
    entries=[];identities=[ident(manifest),ident(OLD/'RESULTS.csv'),ident(__file__),
        ident(ev.__file__),ident(driver.__file__)]
    for old in known:
        variant=old['variant'];folder=CACHE/variant/'predictions/VisA'
        maps_path=folder/'maps74.npy';maps=np.load(maps_path,mmap_mode='r')
        assert maps.shape==(2162,1,74,74) and maps.dtype==np.float32 and np.isfinite(maps).all()
        map_identity=ident(maps_path)
        producer=CACHE/variant/'predictions/Full_MVTec__VisA/COMPLETE.json'
        proof=ev.readjson(producer)
        cp=ident(old['checkpoint'])
        assert map_identity['sha256']==proof['maps_hash']
        assert cp['sha256']==proof['models'][0]['checkpoint_hash']
        rr=ev.readcsv(folder/'TARGET_MANIFEST.csv')
        assert len(rr)==len(rows)
        lookup={ev.normpath(r['img_path']):i for i,r in enumerate(rr)}
        assert len(lookup)==len(rows)
        order=[lookup[ev.normpath(r['img_path'])] for r in rows]
        assert sorted(order)==list(range(len(rows)))
        for r,j in zip(rows,order):
            cached=rr[j]
            for key in ('category','label','gt_height','gt_width','gt_sha256','image_sha256'):
                assert str(r[key])==str(cached[key]),(variant,r['sample_id'],key)
            if int(r['label']):assert ev.normpath(r['gt_path'])==ev.normpath(cached['gt_path'])
        identities.extend([map_identity,cp,ident(folder/'TARGET_MANIFEST.csv'),ident(producer)])
        entries.append(dict(evaluation_id=variant,configuration=variant,source='Full MVTec',
            target='VisA',seed=0,checkpoint=cp['path'],checkpoint_sha256=cp['sha256'],
            folder=str(folder),kind='npy',column=0,lrp=True,order=order,
            previous_P_AP=float(old['P_AP'])))
        del maps
    config=dict(inputs=identities,entries=entries,evaluator_id=evaluator_id,
        policy=ev.POLICY,training=False,model_inference=False,source_seed=0,
        image_score='top55 mean on cached74x74 raw logits, before native resize',
        aggregation='valid-category equal-weight macro for all six metrics',
        scratch=str(WORK/'scratch'))
    driver.lock(OUT/'LOCKED_CONFIG.json',config)
    # Tiny independent tie-handling sanity test; never writes evaluator outputs.
    yy=np.array([0,1,0,1,1,0],dtype=bool)
    ss=np.array([.1,.1,.3,.2,.3,.1],dtype=np.float32)
    assert abs(ev.weighted_pixel_metrics(yy,ss)['P_AP']-average_precision_score(yy,ss)*100)<1e-9
    print('PREFLIGHT PASS: map/weight hashes, all image/mask identities and2162-row bijections.',flush=True)
    raw=[];per=[]
    for entry in entries:
        driver.status('IMAGE_METRICS',evaluation_id=entry['evaluation_id'])
        image=driver.image_metrics(entry,evaluator_id)
        pixel=[]
        reference={r['category']:r for r in ev.readcsv(CACHE/entry['evaluation_id']/'evaluations/VisA_PER_CATEGORY.csv')}
        for category in cats:
            r=driver.pixel_category(entry,category,evaluator_id)
            assert abs(r['P_AP']-float(reference[category]['P_AP']))<1e-6,(category,r['P_AP'],reference[category]['P_AP'])
            pixel.append(r)
            i=next(x for x in image if x['category']==category)
            per.append(dict(source='Full MVTec',target='VisA',seed=0,variant=entry['evaluation_id'],
                category=category,checkpoint=entry['checkpoint'],checkpoint_sha256=entry['checkpoint_sha256'],
                I_AUROC=i['I_AUROC'],I_AP=i['I_AP'],**{k:r[k] for k in METRICS[2:]},
                P_AP_reference=float(reference[category]['P_AP']),P_AP_parity_error=r['P_AP']-float(reference[category]['P_AP'])))
            ev.writecsv(OUT/'PER_CATEGORY_METRICS.csv',per)
            gc.collect()
        row=dict(source='Full MVTec',target='VisA',seed=0,variant=entry['evaluation_id'],
            checkpoint=entry['checkpoint'],checkpoint_sha256=entry['checkpoint_sha256'])
        row.update({k:driver.finite_mean(image if k.startswith('I_') else pixel,k) for k in METRICS})
        assert abs(row['P_AP']-entry['previous_P_AP'])<1e-6
        row.update(valid_categories=12,n_images=2162,image_grid='74x74',image_top_k=55,
            pixel_resolution='native paired GT',aggregation='category macro',evaluator_id=evaluator_id,
            previous_P_AP=entry['previous_P_AP'],P_AP_parity_error=row['P_AP']-entry['previous_P_AP'],
            map_cache=str(Path(entry['folder'])/'maps74.npy'))
        raw.append(row);ev.writecsv(OUT/'MVTec_TO_VisA_SIX_METRICS.csv',raw)
        print('VARIANT COMPLETE',json.dumps(row),flush=True)
    # Recheck all frozen map/weight/manifest inputs after evaluation.
    for r in identities:
        assert ev.sha(r['path'])==r['sha256'],r['path']
    delta={k:raw[1][k]-raw[0][k] for k in METRICS}
    ev.writecsv(OUT/'DELTA_METRICS.csv',[dict(comparison='Exact_VisA_Recipe minus Original_MVTec',**delta)])
    table='| Configuration | I-AUROC | I-AP | P-AUROC | P-F1max | P-AP | AUPRO |\n|---|---:|---:|---:|---:|---:|---:|\n'
    table+='\n'.join('| '+r['variant']+' | '+' | '.join(f'{r[k]:.6f}' for k in METRICS)+' |' for r in raw)
    table+='\n| Delta (recipe-control) | '+' | '.join(f'{delta[k]:+.6f}' for k in METRICS)+' |\n'
    report='# MVTec to VisA: cache-only six-metric completion\n\n'+table+'\n'
    report+='\nAll values are percentages, seed0 only,12 valid categories. Both trained CES heads have30 updates, NTR rho0.10, and fixed global/local LRP0.5/0.5. Both source arms use nearest37 masks; control is not the historical bilinear-label main-table checkpoint.\n'
    report+='\nImage scores are the official top55 mean on74x74 raw maps, followed by per-category AUROC/AP and equal category mean. Pixel maps use official float32 bilinear align_corners=True resize to paired native GT; exact tie-preserving AP/AUROC/F1 and official200-threshold AUPRO at FPR0.30. No sigmoid, smoothing, per-image normalization or target tuning was added.\n'
    report+='\nAll source/checkpoint/map/sample identities are in LOCKED_CONFIG.json. Every recomputed category P-AP matches the previous evaluation within1e-6. Only isolated output files and temporary category vectors were written; no training/inference/official-table edits. The frozen evaluator and established driver functions were called without algorithm changes.\n'
    (OUT/'REPORT.md').write_text(report,encoding='utf-8')
    driver.status('COMPLETE',results=raw,delta=delta)


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    original_stdout,original_stderr=sys.stdout,sys.stderr
    with (OUT/'run.log').open('a',encoding='utf-8',buffering=1) as log:
        sys.stdout=Tee(sys.stdout,log);sys.stderr=Tee(sys.stderr,log)
        try:main()
        except Exception:
            ev.savejson(OUT/'STATUS.json',dict(phase='FAILED',traceback=traceback.format_exc()))
            traceback.print_exc()
            raise
        finally:
            # Restore streams before the log context closes and Python performs
            # its final interpreter flush; completed metrics are unaffected.
            sys.stdout,sys.stderr=original_stdout,original_stderr
