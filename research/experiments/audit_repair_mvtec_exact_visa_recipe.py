"""Audit the MVTec exact-recipe control; isolate any required mask repair."""
from pathlib import Path
import argparse
import gc
import os
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import phase_mvtec_exact_visa_recipe as job
from experiments.phase_defect_query_transfer_pilot import resize_mask_ids

P = job.paired
OLD = job.OUT
OUT = ROOT / 'results/phase_mvtec_exact_visa_recipe_checked/formal'
WORK = Path('E:/MRAD_mvtec_exact_visa_recipe_checked/seed0')


def audit():
    OUT.mkdir(parents=True, exist_ok=True)
    job.torch.set_num_threads(4)
    # Metadata-only fold construction avoids decoding thousands of RGB images
    # merely to recover the already-saved source order.
    name, ds, train, held = P.source_dataset('Full MVTec')
    ids = train + held
    cls = np.load(job.old.CACHE/'D1_fold0_cls.npy',mmap_mode='r')
    patch = np.load(job.old.CACHE/'D1_fold0_patch.npy',mmap_mode='r')
    assert len(cls) == len(patch) == len(ids)
    original = Path('E:/MRAD_final_evidence_cache/global_patch_MVTec.npy')
    a = np.load(original, mmap_mode='r')
    assert a.shape == patch.shape
    mismatch = 0
    for start in range(0, len(a), 32):
        mismatch += int(np.count_nonzero(np.asarray(a[start:start+32]) != np.asarray(patch[start:start+32])))
    assert mismatch == 0, 'Source cache differs; do not start an unverified repair'
    manifest = P.read_csv(OLD / 'SOURCE_IMAGE_MANIFEST.csv')
    assert len(manifest) == len(ids)
    for row, idx in zip(manifest, ids):
        assert int(row['dataset_index']) == int(idx)
        assert row['img_path'] == ds.data_all[idx]['img_path']
    rows = []
    changed = 0
    for pos, idx in enumerate(ids):
        rec = ds.data_all[idx]
        old_y = P.mask_only(ds, idx)
        new_y = (resize_mask_ids(Path(ds.root) / rec['mask_path']) > 0).astype(np.int64) if int(rec['anomaly']) else np.zeros(1369, np.int64)
        count = int(np.count_nonzero(old_y != new_y))
        changed += count
        rows.append(dict(source_position=pos, dataset_index=idx,
            image=str(Path(ds.root) / rec['img_path']), category=rec['cls_name'],
            old_positive_patches=int(old_y.sum()), nearest_positive_patches=int(new_y.sum()),
            changed_patch_labels=count))
    P.write_csv(OUT / 'SOURCE_MASK_AUDIT.csv', rows)
    canonical = job.official.readcsv(job.official.WORK / 'manifests/VisA.csv')
    path_rows = []
    pool_rows = []
    for variant in ('Original_MVTec', 'Exact_VisA_Recipe'):
        folder = job.WORK / variant / 'source_pools/Full_MVTec_seed0'
        pool = P.read_csv(folder / 'PATCH_MANIFEST.csv')
        z = np.load(folder / 'features.npy', mmap_mode='r')
        if variant == 'Original_MVTec':
            transform = dict(use_align=False, q_pc1=None, align_model=None)
        else:
            import torch
            e1, e2 = [torch.load(OLD / 'source_assets' / f'{n}_formal.pt', map_location='cpu', weights_only=False)
                       for n in ('E1_PC1Removed-TrueMask', 'E2_PC1Removed-CorruptedMask')]
            st, sc = [np.asarray(p['patch_states'], np.float32) for p in (e1,e2)]
            model = job.make_model(sc, e2, 'cpu').eval()
            transform = dict(use_align=True, q_pc1=np.load(OLD/'source_assets/q_pc1_statealigned1024_formal.npy'), align_model=model)
        err = 0.
        for start in range(0, len(pool), 2048):
            chunk = pool[start:start+2048]
            raw = np.stack([a[int(r['source_position']), int(r['patch_flat_index'])] for r in chunk]).astype(np.float32)
            got = P.transformed(raw, transform, 'cpu').numpy()
            err = max(err, float(np.max(abs(got-z[start:start+len(chunk)]))))
        assert err < 2e-6, (variant,err)
        pool_rows.append(dict(variant=variant,patches=len(pool),max_abs_reconstruction_error=err,
                              source_identity='PASS', mask_rule='bilinear518_to37 threshold0.5, NOT original VisA nearest-ID rule'))
        saved = P.read_csv(job.WORK / variant / 'predictions/VisA/TARGET_MANIFEST.csv')
        assert len(saved) == len(canonical)
        lookup = {r['sample_id']:r for r in canonical}
        assert len({r['sample_id'] for r in saved}) == len(canonical)
        for r in saved:
            c = lookup[r['sample_id']]
            assert r['image_sha256'] == c['image_sha256'] and r['gt_sha256'] == c['gt_sha256']
        path_rows.append(dict(variant=variant,images=len(saved),native_GT='PASS', image_identity='PASS'))
    P.write_csv(OUT/'OLD_SOURCE_POOL_RECONSTRUCTION.csv',pool_rows)
    P.write_csv(OUT/'OLD_TARGET_IDENTITY.csv',path_rows)
    result = dict(source_order='PASS; MVTec has no VisA carrier row-order bug',
        source_features_differing_values=mismatch, source_images=len(ids),
        source_images_with_mask_difference=sum(r['changed_patch_labels']>0 for r in rows),
        source_patch_labels_different=changed, target_identity='PASS',
        exact_VisA_mask_recipe='FAIL' if changed else 'PASS',
        required_action='rebuild E1/E2 and both matched CES heads under nearest masks' if changed else 'none')
    P.write_json(OUT/'AUDIT.json',result)
    (OUT/'AUDIT.md').write_text('# MVTec to VisA exact-recipe audit\n\n'
        'MVTec cached feature rows and saved source IDs match the original E-drive cache. '
        'Both saved source pools reconstruct from those physical patch IDs (see CSV). '
        'Target image/native-GT manifests match the frozen formal test set. '
        'However the prior clone used bilinear thresholded patch masks in E1/E2 and final CES. '
        'Original VisA E1/E2 uses nearest downsampling of raw ID masks. '
        f'There are {changed} changed patch labels across {result["source_images_with_mask_difference"]} source images. '
        'Thus the previous result is not an exact mask-recipe replication. '
        'The isolated repair uses nearest labels for both arms to keep supervision matched; '
        'its unconditioned arm is a matched nearest-mask control, not the unchanged formal MVTec baseline. '
        'No existing artifact is overwritten.\n',encoding='utf-8')
    print(result,flush=True)
    return result


def run(wait_pid):
    result = job.official.readjson(OUT/'AUDIT.json')
    if not result['source_patch_labels_different']:
        P.status(OUT,'NO_RERUN_REQUIRED');return
    if wait_pid:
        import psutil
        parent = psutil.Process(wait_pid) if psutil.pid_exists(wait_pid) else None
        creation = parent.create_time() if parent else None
        P.status(OUT,'QUEUED',waiting_for_pid=wait_pid,reason='avoid concurrent native-resolution evaluation memory pressure')
        while psutil.pid_exists(wait_pid):
            try:
                if psutil.Process(wait_pid).create_time() != creation:break
            except psutil.NoSuchProcess:break
            time.sleep(30)
    WORK.mkdir(parents=True,exist_ok=True)
    import torch
    torch.set_num_threads(4)
    cache = WORK/'source_cache';cache.mkdir(exist_ok=True)
    for name, origin in [
        ('D1_fold0_patch.npy',Path('E:/MRAD_final_evidence_cache/global_patch_MVTec.npy')),
        ('D1_fold0_cls.npy',job.old.CACHE/'D1_fold0_cls.npy')]:
        dest = cache/name
        if not dest.exists():
            if origin.drive.casefold()==dest.drive.casefold():os.link(origin,dest)
            else:
                import shutil
                shutil.copy2(origin,dest)
    job.OUT, job.WORK = OUT, WORK
    job.old.CACHE = cache
    original_mask = P.mask_only
    def nearest_mask(ds,idx):
        row=ds.data_all[int(idx)]
        if not int(row['anomaly']):return np.zeros(1369,np.int64)
        return (resize_mask_ids(Path(ds.root)/row['mask_path'])>0).astype(np.int64)
    P.mask_only=nearest_mask
    def physical_arrays(direction):
        assert direction=='D1'
        _,ds,tr,held=P.source_dataset('Full MVTec')
        path=Path('E:/MRAD_final_evidence_cache/global_patch_MVTec.npy')
        a=np.load(path,mmap_mode='r')
        assert len(a)==len(tr)+len(held)
        return [a[:len(tr)],a[len(tr):]],[path,path]
    P.source_arrays=physical_arrays
    original_predict=P.predict_pair
    def checked_predict(*args,**kwargs):
        pred=original_predict(*args,**kwargs)
        rows=P.read_csv(pred/'TARGET_MANIFEST.csv')
        if not (pred/'UNRELOCATED_TARGET_MANIFEST.csv').exists():
            P.write_csv(pred/'UNRELOCATED_TARGET_MANIFEST.csv',rows)
        canonical=job.official.readcsv(job.official.WORK/'manifests/VisA.csv')
        paths=job.official.canonical_paths('VisA',[r['image'] for r in rows],canonical)
        for row,path in zip(rows,paths):row['image']=path
        P.write_csv(pred/'TARGET_MANIFEST.csv',rows)
        done=job.official.readjson(pred/'COMPLETE.json')
        done['manifest_hash']=P.file_hash(pred/'TARGET_MANIFEST.csv')
        done['path_canonicalization']='official byte-identical image verification'
        P.write_json(pred/'COMPLETE.json',done)
        return pred
    P.predict_pair=checked_predict
    P.write_json(OUT/'REPAIR_RUNTIME.json',dict(seed=0,source='Full MVTec',target='VisA',
        stage_and_CES_mask_rule='raw GT nearest37 then >0',
        source_cache='E:/MRAD_final_evidence_cache/global_patch_MVTec.npy',
        baseline='unconditioned matched nearest-mask control, not untouched formal baseline',
        code=str(Path(__file__)),sha256=P.file_hash(__file__),upstream_old_assets_modified=False))
    job.run()
    with (OUT/'REPORT.md').open('a',encoding='utf-8') as f:
        f.write('\nBoth arms use nearest source masks. Original_MVTec denotes the matched '
                'unconditioned control in this repair, not the historical bilinear-label baseline. '
                'The old E1/E2 and CES checkpoints were not replaced.\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('audit','run'),default='audit')
    parser.add_argument('--wait-pid',type=int,default=0)
    args=parser.parse_args()
    try:
        if args.stage=='audit':audit()
        else:run(args.wait_pid)
    except BaseException:
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/'ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
        P.status(OUT,'FAILED',error=traceback.format_exc())
        raise
