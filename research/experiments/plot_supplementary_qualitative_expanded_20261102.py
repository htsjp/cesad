"""Expand existing qualitative caches into six 8x4 plates; no model inference."""
from pathlib import Path
import sys
import gc
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import plot_supplementary_qualitative_20261102 as base
from experiments import final_unified_evaluation as ev

OLD = base.OUT
OUT = ROOT/'results/supplementary_qualitative_expanded_20261102'
SLOTS = {
    'MVTec':[('grid',.5),('wood',.5),('metal_nut',.5),('screw',.5)],
    'VisA':[('macaroni2',None),('pcb3',.5),('chewinggum',.5),('capsules',.5)],
    'BTAD':[('01',.5),('02',.5),('03',.5),('01',.75)],
    'MPDD':[('bracket_black',.5),('connector',.5),('metal_plate',.5),('tubes',.5)],
    'DTD-Synthetic':[('Blotchy_099',.5),('Fibrous_183',.5),('Marbled_078',.5),('Matted_069',.5)],
    'DAGM':[('Class1',.5),('Class2',.5),('Class3',.5),('Class4',.5)],
}
CONTROL = {
    'NTR':{'VisA':['pcb1','pcb3','chewinggum','candle'],
           'MVTec':['grid','tile','wood','metal_nut']},
    'LRP':{'VisA':['macaroni2','capsules','pcb2','pipe_fryum'],
           'MVTec':['screw','cable','capsule','grid']},
}


def quantile_case(rows, category, fraction=.5):
    candidates = base.positive(rows, category)
    assert candidates, category
    candidates.sort(key=lambda r:(int(r['positive_pixels']) /
                    (int(r['gt_width'])*int(r['gt_height'])), r['sample_id']))
    return candidates[int((len(candidates)-1)*fraction)]


def origin(entry, row):
    return dict(source=entry['source'], checkpoint=entry['checkpoint'],
                cache_path=entry['cache_path'],
                cache_index=entry['canonical_to_cache_index'][int(row['index'])],
                version='Table I historical prediction registry; no retrospective source-pair certification')


def save_plate(samples, name, titles, controlled=False, outline=False):
    assert len(samples)==8 and len({s['row']['sample_id'] for s in samples})==8
    base.plate(samples, name, titles, controlled=controlled, outline=outline)
    for i,s in enumerate(samples):
        s['record'].update(plate=name, row_number=i+1)
    print('PLATE COMPLETE', name, '8 x 4', flush=True)


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'selected_data').mkdir(exist_ok=True)
    base.OUT=OUT
    base.records.clear();base.assets.clear();base.candidates.clear()
    plt.rcParams.update({'font.family':'Arial','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
    registry=[e for e in ev.readjson(base.REGISTRY) if e['method']=='CESAD']
    base.check_asset(base.REGISTRY)
    old_candidates=ev.readcsv(OLD/'BOUNDARY_SELECTION_CANDIDATES.csv')
    base.check_asset(OLD/'BOUNDARY_SELECTION_CANDIDATES.csv')
    by_category={}
    for r in old_candidates:
        by_category.setdefault(r['category'],[]).append(r)
    for rr in by_category.values():
        rr.sort(key=lambda r:(float(r['pixel_AP']),r['sample_id']))
    s1,s3=[],[]
    for e in registry:
        name=e['target']
        for path,hash_ in [(e['manifest'],e['manifest_sha256']),
                           (e['cache_path'],e['cache_sha256']),
                           (e['checkpoint_file'],e['checkpoint_sha256'])]:
            base.check_asset(path,hash_)
        rows=ev.readcsv(e['manifest']);lookup={r['sample_id']:r for r in rows}
        store=ev.Store(e)
        for category,fraction in SLOTS[name]:
            row=lookup[by_category['macaroni2'][0]['sample_id']] if fraction is None else quantile_case(rows,category,fraction)
            rule=('minimum sample AP within fixed macaroni2 category; explicit posthoc difficult case'
                  if fraction is None else f'GT area-fraction quantile {fraction}; filename tie-break; fixed category')
            s=base.specimen(row,[store[int(row['index'])]],origin(e,row),'S1',rule)
            s1.append(s)
        if name=='VisA':
            for category,count in [('macaroni2',3),('pcb3',3),('chewinggum',2)]:
                for chosen in by_category[category][:count]:
                    row=lookup[chosen['sample_id']]
                    s=base.specimen(row,[store[int(row['index'])]],origin(e,row),'S3',
                        f'lowest {count} sample AP within fixed {category} category; explicit posthoc selection')
                    assert abs(s['record']['sample_AP_left']-float(chosen['pixel_AP']))<1e-9
                    s3.append(s)
        del store
        gc.collect()
        print('S1 DATA',name,4,flush=True)
    for src in ('Full VisA','Full MVTec'):
        group=[s for s in s1 if s['record']['source']==src]
        limits=base.shared_limits(group)
        for s in group:s['limits']=limits
    for page in range(3):
        save_plate(s1[page*8:(page+1)*8],f'supp_fig_s1_part{page+1}',
                   ['Input','GT Mask','CESAD anomaly logits','Overlay'])
    limits=base.shared_limits(s3)
    for s in s3:s['limits']=limits
    save_plate(s3,'supp_fig_s3_visa_boundary_8cases',
               ['Input','GT Mask','CESAD anomaly logits','GT outline overlay'],outline=True)

    controls={'NTR':[],'LRP':[]}
    config_path=base.RECIP/'formal/LOCKED_CONFIG.json'
    base.check_asset(config_path)
    base.check_asset(base.RECIP/'formal/SOURCE_SEALED.json')
    for name in ('VisA','MVTec'):
        folder=base.RECIP/'work/predictions'/name
        done=ev.readjson(folder/'COMPLETE.json')
        for key in ('maps','manifest','head_order'):
            base.check_asset(done[key]['path'],done[key]['sha256'])
        assert done['source_seal_sha256']==ev.sha(base.RECIP/'formal/SOURCE_SEALED.json')
        entries=ev.readjson(folder/'HEAD_ORDER.json')
        no=next(i for i,h in enumerate(entries) if h['seed']==0 and h['rho'] is None)
        yes=next(i for i,h in enumerate(entries) if h['seed']==0 and h['rho']==.1)
        endpoints=[];initial=[]
        for idx in (no,yes):
            cp=entries[idx]['checkpoint']
            base.check_asset(cp['path'],cp['sha256'])
            z=torch.load(cp['path'],map_location='cpu',weights_only=False)
            assert z['step']==30 and z['seed']==0 and z['rho']==entries[idx]['rho']
            assert z['config_sha256']==ev.sha(config_path)
            endpoints.append(z)
            ip=Path(cp['path']).with_name('update_000.pt')
            base.check_asset(ip)
            initial.append(torch.load(ip,map_location='cpu',weights_only=False)['raw_model'])
        assert np.array_equal(endpoints[0]['batch_indices'],endpoints[1]['batch_indices'])
        assert endpoints[0]['source_input_sha256']==endpoints[1]['source_input_sha256']
        assert all(torch.equal(initial[0][k],initial[1][k]) for k in initial[0])
        assert all(torch.equal(endpoints[0]['raw_model'][k],endpoints[1]['raw_model'][k]) for k in ('m','alpha'))
        rows=ev.readcsv(folder/'MANIFEST.csv')
        maps=np.load(folder/'global_local74.npy',mmap_mode='r')
        for component in ('NTR','LRP'):
            for category in CONTROL[component][name]:
                row=quantile_case(rows,category)
                pos=int(row['index']);g0,l0=maps[pos,no];g1,l1=maps[pos,yes]
                pair=[.5*g0+.5*l0,.5*g1+.5*l1] if component=='NTR' else [np.asarray(g1),.5*g1+.5*l1]
                cp=[entries[no]['checkpoint']['path'],entries[yes]['checkpoint']['path']] if component=='NTR' else [entries[yes]['checkpoint']['path']]*2
                org=dict(source=entries[yes]['source'],checkpoint=' | '.join(cp),cache_path=str(folder/'global_local74.npy'),
                         cache_index=pos,version='certified 20261101 reciprocal trained 30-update family; not historical Table IV')
                s=base.specimen(row,pair,org,'S2_'+component,
                    'median positive GT area fraction within fixed category; filename tie-break',
                    dict(component=component,left_lrp=component=='NTR',right_lrp=True,
                         left_rho=0 if component=='NTR' else .1,right_rho=.1,lambda_global=.5,
                         shared_physical_sample=True,actual_batches_identical=True,initialization_identical=True))
                s['limits']=base.shared_limits([s]);s['tag']='MVTec to VisA' if name=='VisA' else 'VisA to MVTec'
                controls[component].append(s)
        del maps
    save_plate(controls['NTR'],'supp_fig_s2a_ntr_8cases',
               ['Input','GT Mask','Trained CES (LRP on)','CES + NTR (LRP on)'],controlled=True)
    save_plate(controls['LRP'],'supp_fig_s2b_lrp_8cases',
               ['Input','GT Mask','Global only (NTR on)','Global + local (NTR on)'],controlled=True)
    assert len(base.records)==48
    ev.writecsv(OUT/'CASE_MANIFEST.csv',base.records)
    ev.writecsv(OUT/'BOUNDARY_SELECTION_CANDIDATES.csv',old_candidates)
    ev.savejson(OUT/'INPUT_IDENTITIES.json',base.assets)
    ev.savejson(OUT/'BUILD_CONFIG.json',dict(selection=SLOTS,controls=CONTROL,
        figure_rows=8,figure_columns=4,figures=6,case_records=48,
        source_script=base.identity(Path(__file__)),renderer=base.identity(Path(base.__file__)),
        training=False,inference=False,raw_logit_normalization=False,clipping='NONE'))
    report='''# Expanded supplementary qualitative plates

Six 8x4 plates, 48 case records. Existing smaller plates are preserved unchanged.

- S1 part1: MVTec / VisA, four cases each.
- S1 part2: BTAD / MPDD, four cases each.
- S1 part3: DTD-Synthetic / DAGM, four cases each.
- S2(a): NTR, four MVTec-to-VisA and four VisA-to-MVTec cases.
- S2(b): LRP, four MVTec-to-VisA and four VisA-to-MVTec cases.
- S3: eight VisA boundary cases (three macaroni2, three pcb3, two chewinggum).

Source families and original-prediction identity are inherited exactly from the first delivery.
S1/S3 use the historical Table I registry; this is not retrospective source-pair certification.
Historical seed is UNKNOWN where the registry lacks it. S2 uses certified 20261101 seed0 heads,
not an initialization fallback or automatically the same checkpoint family as historical Table IV.
No new model inference or training, no changes to formal results or the manuscript.

Selection is predeclared in BUILD_CONFIG.json and the script: positive GT-area medians within
fixed categories; the fourth BTAD case is the 75th-percentile 01 case because there are only three
BTAD categories. S1 retains the previously selected lowest-AP macaroni2 difficult case. S3 uses
the lowest three/three/two sample APs in the fixed categories, reusing the earlier candidate audit.
This explicit post-hoc difficult-case selection must be disclosed, not represented as random.
No selected sample was replaced because of its model comparison outcome.

Every image and original GT hash is checked, and every defect mask is nonempty. Raw FP32 maps
are bilinearly resized to original GT with align_corners=True. No sigmoid, independent min-max,
Gaussian smoothing, percentile clipping or geometry distortion. All S1 VisA-source examples
share one full-extrema color range across all three pages; MVTec-source VisA has its own range.
S3 shares one range. Each S2 row shares full extrema across its two compared arms. Printed color
limits differ between model families/pairs: do not compare brightness as absolute logits across
different ranges. Overlay alpha is 0.4. White outline is GT only, not a predicted boundary.

Suggested captions:
S1. Cross-dataset anomaly localization examples from six industrial benchmarks, four per target.
Each row shows the input, released GT, cached CESAD anomaly logits, and overlay. Parts 1-3
show MVTec/VisA, BTAD/MPDD, and DTD-Synthetic/DAGM, respectively. Predictions and source-matched
checkpoints follow the Table I prediction registry; color limits are printed per row.
S2. Controlled NTR and LRP visualization under reciprocal transfers, four examples per direction.
(a) Matched trained no-NTR/NTR heads with LRP fixed on. (b) Global and global-plus-local maps
from the same trained NTR scorer. The sample and raw-logit limits are shared within each row.
S3. Eight post-hoc VisA boundary examples selected by lowest sample AP within fixed categories.
The examples illustrate localization limitations, not a causal mechanism or failure prevalence.
'''
    (OUT/'README_AND_CAPTIONS.md').write_text(report,encoding='utf-8')
    print('COMPLETE',OUT,'48 case records; six 8x4 plates',flush=True)


if __name__=='__main__':
    main()
