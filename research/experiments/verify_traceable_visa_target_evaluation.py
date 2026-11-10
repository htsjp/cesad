"""Read completed results/seal; no model, training, AP/AUROC/PRO recomputation.

Checks raw prediction identities, shared endpoint/view order, recorded image
scores, category coverage, and arithmetic aggregation of already saved metrics.
"""
from pathlib import Path
import hashlib
import json
import math
import sys
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import evaluate_traceable_visa_targets as job

OUT=job.audit.OUT/'target_evaluation_acceptance'


def number(value):
    return None if value is None or str(value) in ('','None','nan','N/A') else float(value)


def same_number(actual,expected):
    a,b=number(actual),number(expected)
    if a is None or b is None:
        assert a is None and b is None,(actual,expected)
    else:
        assert math.isfinite(a) and math.isfinite(b) and abs(a-b)<1e-9,(a,b)


def run():
    from experiments import final_unified_evaluation as ev
    seal_path=job.OUT/'EVALUATION_SEALED.json'
    if not seal_path.is_file():
        raise RuntimeError('Evaluation is not complete/sealed; no acceptance report generated')
    seal=job.audit.read(seal_path)
    manifest,config,manifest_id,approval_id=job.gate()
    job.validate_roles(config)
    assert seal['complete'] and seal['endpoint_roles']==job.ROLES and seal['selection']=='NONE_REPORT_BOTH'
    assert seal['training'] is False
    assert seal['context']['input_manifest']==manifest_id and seal['context']['approval']==approval_id
    assert seal['context']['source_seal_sha256']==manifest['source_seal_sha256']
    assert ev.evaluator_hash()==config['evaluator']['hash']
    for record in seal['result_files']:job.audit.require_identity(record)
    codes=job.audit.read(job.REVIEW/'EVALUATION_CODE_IDENTITY.json')
    with zipfile.ZipFile(job.REVIEW/'EVALUATION_CODE_SNAPSHOT.zip') as z:
        for item in codes:
            job.audit.require_identity(item)
            member=str(Path(item['path']).relative_to(ROOT)).replace('\\','/')
            assert hashlib.sha256(z.read(member)).hexdigest()==item['sha256']
    source_encoder=job.audit.read(config['encoder']['reference_identity']['path'])
    assert seal['encoder_hash']==source_encoder['visual_hash_after_position_setup']
    categories=ev.readcsv(job.OUT/'CATEGORY_METRICS.csv')
    summaries=ev.readcsv(job.OUT/'TARGET_METRICS.csv')
    deltas=ev.readcsv(job.OUT/'PAIRED_ENDPOINT_DELTAS.csv')
    assert len(summaries)==12 and len(deltas)==6 and len(categories)==92
    assert len({(r['target'],r['endpoint'],r['category']) for r in categories})==92
    checks=[];total_images=0;total_views=0;max_smoke=0.
    for target in job.TARGETS:
        rows=ev.readcsv(job.REVIEW/'manifests'/f'{target}.csv')
        folder=job.WORK/'predictions'/target
        done=job.audit.read(folder/'COMPLETE.json')
        assert done['stamp']['context']==seal['context']
        assert done['stamp']['endpoint_order']==list(job.ENDPOINTS)
        assert done['stamp']['encoder_hash']==seal['encoder_hash']
        assert done['stamp']['scorer_hash']==seal['scorer_hash']
        for key in ('maps','row_journal','smoke_checks'):job.audit.require_identity(done[key])
        matching=[r for r in seal['predictions'] if r['maps']['path']==done['maps']['path']]
        assert len(matching)==1 and matching[0]==done
        maps=np.load(done['maps']['path'],mmap_mode='r')
        assert maps.dtype==np.float32 and maps.shape==(len(rows),2,74,74)
        with Path(done['row_journal']['path']).open(encoding='utf-8') as f:
            count=0
            for i,line in enumerate(f):
                rec=json.loads(line);row=rows[i]
                assert rec['index']==i and rec['sample_id']==row['sample_id']
                assert rec['image_sha256']==row['image_sha256'] and rec['stamp']==done['stamp']
                assert rec['map_sha256']==hashlib.sha256(maps[i].tobytes()).hexdigest()
                assert np.isfinite(maps[i]).all()
                views=rec['views']
                assert views['view_order']==['global','TL','TR','BL','BR']
                expected=job.crop_boxes(int(row['image_width']),int(row['image_height']))
                assert views['crop_boxes']==[list(b) for b in expected]
                assert len(views['tensor_sha256'])==5 and all(len(s)==64 for s in views['tensor_sha256'])
                assert len(views['raw_feature_sha256'])==64
                if i<2:
                    assert rec['smoke_cpu_parity_max_abs'] is not None
                    assert rec['smoke_cpu_parity_max_abs']<=2e-6
                    max_smoke=max(max_smoke,rec['smoke_cpu_parity_max_abs'])
                else:assert rec['smoke_cpu_parity_max_abs'] is None
                count+=1
        assert count==len(rows)==done['images']
        names=sorted({r['category'] for r in rows})
        for column,endpoint in enumerate(job.ENDPOINTS):
            image_scores=ev.readcsv(job.OUT/'evaluations'/target/f'{endpoint}_IMAGE_SCORES.csv')
            assert len(image_scores)==len(rows)
            for i,(image,row) in enumerate(zip(image_scores,rows)):
                assert image['sample_id']==row['sample_id'] and image['category']==row['category']
                assert int(image['label'])==int(row['label'])
                # This verifies the saved top55 scalar, not a dataset metric.
                same_number(image['score'],ev.topmean(maps[i,column]))
            rr=[r for r in categories if r['target']==target and r['endpoint']==endpoint]
            assert sorted(r['category'] for r in rr)==names
            for r in rr:
                assert r['source']=='Full VisA' and int(r['seed'])==0 and r['role']==job.ROLES[endpoint]
                expected_pixels=sum(int(row['gt_height'])*int(row['gt_width']) for row in rows if row['category']==r['category'])
                assert int(r['native_pixels'])==expected_pixels
                original=job.audit.read(job.OUT/'evaluations'/target/f'{r["category"]}__{endpoint}.json')
                assert original['stamp']['context']==seal['context']
                assert original['stamp']['prediction_sha256']==done['maps']['sha256']
                for metric in job.METRICS:same_number(r[metric],original[metric])
            summary=next(r for r in summaries if r['target']==target and r['endpoint']==endpoint)
            assert summary['role']==job.ROLES[endpoint] and summary['checkpoint']==config['source_inputs']['endpoints'][endpoint]['path']
            assert summary['checkpoint_sha256']==config['source_inputs']['endpoints'][endpoint]['sha256']
            for metric in job.METRICS:
                values=[number(r[metric]) for r in rr]
                valid=[v for v in values if v is not None and math.isfinite(v)]
                same_number(summary[metric],float(np.mean(valid)) if valid else None)
                assert int(summary[metric+'_valid_categories'])==len(valid)
        total_images+=len(rows);total_views+=len(rows)*5
        checks.append(dict(target=target,images=len(rows),categories=len(names),endpoints=2,
            raw_map_and_row_hashes='PASS',view_order_and_geometry='PASS',saved_image_score_policy='PASS',
            saved_category_macro_arithmetic='PASS',main_endpoint='update30'))
        del maps
        print('TARGET VERIFIED',target,len(rows),flush=True)
    assert total_images==12278 and total_views==61390
    for endpoint in job.ENDPOINTS:
        cells=[r for r in summaries if r['endpoint']==endpoint and r['target'] in job.TARGETS]
        assert len(cells)==5
        mean=next(r for r in summaries if r['endpoint']==endpoint and r['target']=='Five-transfer mean')
        for metric in job.METRICS:
            vals=[number(r[metric]) for r in cells]
            same_number(mean[metric],float(np.mean(vals)) if all(v is not None for v in vals) else None)
    for row in deltas:
        a=next(r for r in summaries if r['target']==row['target'] and r['endpoint']=='initialization')
        b=next(r for r in summaries if r['target']==row['target'] and r['endpoint']=='update30')
        assert row['delta']=='update30_minus_initialization'
        for metric in job.METRICS:
            x,y=number(a[metric]),number(b[metric])
            same_number(row[metric],y-x if x is not None and y is not None else None)
    assert not list((job.WORK/'scratch').glob('*.npy')),'Unexpected retained native scratch'
    OUT.mkdir(parents=True,exist_ok=True)
    ev.writecsv(OUT/'TARGET_ACCEPTANCE_CHECKS.csv',checks)
    verification=dict(status='PASS_COMPLETED_TARGET_EVALUATION',created=job.audit.now(),
        source='Full VisA',seed=0,images=total_images,encoded_views_recorded=total_views,
        categories=46,category_endpoint_cells=92,endpoint_roles=job.ROLES,selection='NONE_REPORT_BOTH',
        max_first_two_image_CPU_parity_error=max_smoke,evaluation_seal=job.audit.ident(seal_path),
        code_snapshot_files_verified=len(codes),all_result_prediction_and_input_hashes_verified=True,
        model_forwards_run_by_this_audit=0,training_run_by_this_audit=False,
        AP_AUROC_F1_PRO_recomputed_by_this_audit=False,
        arithmetic_and_saved_top55_score_checks=True,audit_code=job.audit.ident(__file__),
        limitation='Encoder forwards were not independently repeated. No manuscript or historical results were changed.')
    job.audit.js(OUT/'VERIFICATION.json',verification)
    text=['# Five-target evaluation acceptance','',
        '**PASS_COMPLETED_TARGET_EVALUATION**','',
        'Verified the source/input/approval/code seals, raw FP32 map and per-image row hashes, five-view geometry, '
        'shared endpoint order, first-two-image numerical smoke records, saved top55 image-score scalars, '
        'category coverage and arithmetic aggregation of already saved metrics. '
        'No model forward, training or AP/AUROC/F1/PRO recomputation was performed by this auditor.','',
        '| Target | Images | Categories | Endpoints | Checks |','|---|---:|---:|---:|---|',
        *[f'| {r["target"]} | {r["images"]} | {r["categories"]} | 2 | PASS |' for r in checks],'',
        'The trained main model remains update30, even if initialization is better on a target. '
        'Both endpoints and their paired differences are retained. Five-transfer means include all five targets; '
        'there is no seed-level SD for this seed0-only run.','',
        'Source/encoder immutability is supported by the frozen input files and the approved runner\'s per-target '
        'and final checksum assertions. This audit did not independently rerun the encoder. '
        'Historical model pairing was not retrospectively certified. The manuscript and old tables were not modified.','']
    (OUT/'TARGET_EVALUATION_RUNTIME_AUDIT.md').write_text('\n'.join(text),encoding='utf-8')
    print(json.dumps(verification,indent=2),flush=True)


if __name__=='__main__':run()
