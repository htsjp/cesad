"""User-requested replacement of five displayed IDs, with no new inference."""
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import final_unified_evaluation as ev
from experiments import plot_supplementary_qualitative_20261102 as base
from experiments import plot_supplementary_reference_layout_20261102 as layout

OLD=ROOT/'results/supplementary_qualitative_expanded_20261102'
NEW=ROOT/'results/supplementary_qualitative_reselected_20261102'
BANNED={'BTAD':{'0177'},'MPDD':{'007','010'},'VisA':{'082','037'}}


def forbidden(row):
    return Path(row['image_path']).stem in BANNED.get(row['target'],set())


def main():
    torch.set_num_threads(4)
    NEW.mkdir(parents=True,exist_ok=True)
    (NEW/'selected_data').mkdir(exist_ok=True)
    base.OUT=NEW;base.records.clear();base.assets.clear()
    original=ev.readcsv(OLD/'CASE_MANIFEST.csv')
    used={r['sample_id'] for r in original}
    registry={e['target']:e for e in ev.readjson(base.REGISTRY) if e['method']=='CESAD'}
    candidates=ev.readcsv(OLD/'BOUNDARY_SELECTION_CANDIDATES.csv')
    chosen={};stores={};replacements=[];result=[]
    for old in original:
        if not forbidden(old):
            result.append(dict(old));continue
        target=old['target'];category=old['category'];entry=registry[target]
        rows=ev.readcsv(entry['manifest'])
        key=(old['sample_id'],'boundary' if old['figure']=='S3' else 'regular')
        if key not in chosen:
            pool=[r for r in base.positive(rows,category) if r['sample_id'] not in used
                  and Path(r['img_path']).stem not in BANNED.get(target,set())]
            assert pool,(target,category)
            if old['figure']=='S3':
                scores={r['sample_id']:float(r['pixel_AP']) for r in candidates if r['category']==category}
                pool=[r for r in pool if r['sample_id'] in scores]
                row=min(pool,key=lambda r:(scores[r['sample_id']],r['sample_id']))
                rule='user-requested replacement; next unused lowest sample AP within same fixed category'
            else:
                pool.sort(key=lambda r:(int(r['positive_pixels'])/(int(r['gt_width'])*int(r['gt_height'])),r['sample_id']))
                row=pool[int(.75*(len(pool)-1))]
                rule='user-requested replacement; 75th GT-area percentile among unused same-category nonempty masks; no prediction-based selection'
            chosen[key]=(row,rule);used.add(row['sample_id'])
        row,rule=chosen[key]
        if old['figure'].startswith('S2'):
            folder=base.RECIP/'work/predictions'/target
            done=ev.readjson(folder/'COMPLETE.json')
            for k in ('maps','manifest','head_order'):base.check_asset(done[k]['path'],done[k]['sha256'])
            heads=ev.readjson(folder/'HEAD_ORDER.json')
            no=next(i for i,h in enumerate(heads) if h['seed']==0 and h['rho'] is None)
            yes=next(i for i,h in enumerate(heads) if h['seed']==0 and h['rho']==.1)
            paired=ev.readcsv(folder/'MANIFEST.csv')
            pos=next(i for i,r in enumerate(paired) if r['sample_id']==row['sample_id'])
            assert paired[pos]['image_sha256']==row['image_sha256']
            maps=np.load(folder/'global_local74.npy',mmap_mode='r')
            g0,l0=maps[pos,no];g1,l1=maps[pos,yes]
            scores=[.5*g0+.5*l0,.5*g1+.5*l1]
            org=dict(source=old['source'],checkpoint=old['checkpoint'],cache_path=old['cache_path'],cache_index=pos,version=old['source_version'])
            extra={k:old[k] for k in ('component','left_lrp','right_lrp','left_rho','right_rho','lambda_global',
                   'shared_physical_sample','actual_batches_identical','initialization_identical')}
        else:
            if target not in stores:
                base.check_asset(entry['cache_path'],entry['cache_sha256'])
                base.check_asset(entry['manifest'],entry['manifest_sha256'])
                stores[target]=ev.Store(entry)
            scores=[stores[target][int(row['index'])]]
            org=dict(source=old['source'],checkpoint=old['checkpoint'],cache_path=old['cache_path'],
                cache_index=entry['canonical_to_cache_index'][int(row['index'])],version=old['source_version'])
            extra={}
        sample=base.specimen(row,scores,org,old['figure'],rule,extra)
        fresh=sample['record']
        fresh.update(plate=old['plate'],row_number=old['row_number'],cmap='magma',overlay_alpha=.4,clipping='NONE; full extrema')
        if old['figure'].startswith('S2'):
            fresh['vmin'],fresh['vmax']=base.shared_limits([sample])
        else:
            # Preserve the existing source-family display scale, even after replacement.
            fresh['vmin'],fresh['vmax']=old['vmin'],old['vmax']
            lo=min(float(a.min()) for a in sample['native']);hi=max(float(a.max()) for a in sample['native'])
            fresh['_required_min']=lo;fresh['_required_max']=hi
        result.append(fresh)
        replacements.append(dict(figure=old['figure'],plate=old['plate'],target=target,category=category,
            old_sample_id=old['sample_id'],new_sample_id=row['sample_id'],selection_rule=rule,
            new_sample_AP_left=fresh['sample_AP_left'],new_sample_AP_right=fresh['sample_AP_right']))
        print('REPLACE',old['figure'],old['sample_id'],'->',row['sample_id'],flush=True)
    assert len(result)==48 and not any(forbidden(r) for r in result)
    # If needed, expand one shared family range; never clip or separately stretch a map.
    for family in ('S1','S3'):
        for source in ('Full VisA','Full MVTec'):
            group=[r for r in result if r['figure']==family and r['source']==source]
            if not group:continue
            lo=min(min(float(r['vmin']),float(r.get('_required_min',r['vmin']))) for r in group)
            hi=max(max(float(r['vmax']),float(r.get('_required_max',r['vmax']))) for r in group)
            for r in group:r['vmin']=lo;r['vmax']=hi
    for r in result:r.pop('_required_min',None);r.pop('_required_max',None)
    ev.writecsv(NEW/'CASE_MANIFEST.csv',result)
    ev.writecsv(NEW/'REPLACEMENTS.csv',replacements)
    ev.savejson(NEW/'REPLACEMENT_INPUT_IDENTITIES.json',base.assets)
    text=(OLD/'README_AND_CAPTIONS.md').read_text(encoding='utf-8')
    text+='\n## User-requested sample replacement\nSeven displayed occurrences of five old image IDs were replaced. See REPLACEMENTS.csv. All requested old IDs are absent. Regular replacements use the 75th positive GT-area percentile among unused same-category examples, not a target-performance ranking. The S3 replacement follows the next unused low-AP candidate to retain its boundary-case role. Original and expanded deliveries are preserved. Shared source-family color limits are kept or expanded for all relevant columns together if a replacement exceeds the previous range.\n'
    (NEW/'README_AND_CAPTIONS.md').write_text(text,encoding='utf-8')
    layout.SOURCE=NEW;layout.OUT=NEW/'reference_layout'
    layout.main()


if __name__=='__main__':main()
