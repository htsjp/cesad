"""Seed-0 MVTec source replication of the VisA E1/E2 source recipe.

Reuse the actual Stage-I/II functions; redirect all mutable output globals only
inside this process. No historical source asset is modified.
"""
from pathlib import Path
import sys
import gc
import traceback
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import phase_statealigned_pc1_debias as stage
from experiments import phase_mvtec_source_conditioning_control as old
from experiments import phase_ces_lhc_paired_repeat as paired
from experiments import phase_spatial_supervision_transfer_audit as spatial
from experiments import phase_ces_geometry_center_control as geometry
from experiments import final_unified_evaluation as official
from experiments.phase_statealigned_state_alignment_swap import make_model

OUT = ROOT / 'results/phase_mvtec_exact_visa_recipe/formal'
WORK = Path('E:/MRAD_mvtec_exact_visa_recipe/seed0')
DEVICE = 'cuda'


def status(phase, **kw):
    paired.status(OUT, phase, **kw)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    status('PREFLIGHT')
    # Require the frozen cache; do not silently regenerate the visual encoder.
    for name in ('D1_fold0_cls.npy', 'D1_fold0_patch.npy'):
        assert (old.CACHE / name).is_file(), old.CACHE / name
    name, ds, indices, cls, patch = old.ensure_direction_cache('D1', DEVICE, False)
    assert name.lower() == 'mvtec'
    assert len(indices) == len(patch) == len(cls)
    records = []
    true_labels = {}
    for pos, idx in enumerate(indices):
        y = paired.mask_only(ds, idx).astype(np.float32)
        normal = int(ds.data_all[idx]['anomaly']) == 0
        if normal:
            assert not y.any()
        records.append(dict(pos=pos, idx=idx, mask=y, normal=normal))
        true_labels[pos] = y.astype(np.int64)
    corrupted, corruption_audit = spatial.corrupt_masks(records)
    shifted_labels = {r['pos']: np.asarray(m > .5, np.int64)
                      for r, m in zip(records, corrupted)}
    paired.write_csv(OUT / 'SHIFTED_MASK_AUDIT.csv', corruption_audit)
    paired.write_csv(OUT / 'SOURCE_IMAGE_MANIFEST.csv',
                     [dict(source_position=p, dataset_index=int(i), **ds.data_all[i])
                      for p, i in enumerate(indices)])
    config = dict(source='Full MVTec', target='VisA', seed=0,
        recipe='same VisA Stage-I and Stage-II functions; source domain substituted only',
        state_epochs=stage.STATE_EPOCHS, adapter_epochs=stage.ADAPTER_EPOCHS,
        state_lr=stage.STATE_LR, adapter_lr=stage.ADAPTER_LR,
        mask_corruption_seed=spatial.SEED, rank=4, rho=.1,
        CES_epochs=paired.EPOCHS, CES_lr=.001, LRP_global_weight=.5,
        evaluator_hash=official.evaluator_hash(),
        source_cache=str(old.CACHE), source_shape=list(patch.shape),
        code_hashes={str(p.relative_to(ROOT)): paired.file_hash(p) for p in
            (Path(__file__), Path(stage.__file__), Path(paired.__file__),
             Path(spatial.__file__), Path(official.__file__))})
    cp = OUT / 'CONFIG.json'
    if cp.exists():
        assert official.readjson(cp) == config
    else:
        paired.write_json(cp, config)
    (OUT / 'PRE_RUN_VALIDATION.md').write_text(
        '# MVTec exact VisA recipe\n\n'
        'Seed 0 only. Source PC1 uses MVTec normal images only. E1 uses true '
        'source masks; E2 uses the original circular-shift function. Both use '
        'the original state/adapter training functions and hyperparameters. '
        'Final CES labels are true masks. E1 supplies half-difference direction '
        'and scale; E2 supplies midpoint and frozen adapter. A freshly trained '
        'unconditioned MVTec CES is the control. Physical patch IDs and batch '
        'indices must match. Fixed 2x2 LRP and native paired GT evaluator. '
        'All outputs isolated; no target selection.\n', encoding='utf-8')
    stage.OUT = OUT / 'source_assets'
    stage.load_full_visa_source_lazy = lambda smoke: (
        'Full MVTec', ds, indices, cls, patch,
        np.array([int(ds.data_all[i]['anomaly']) for i in indices]),
        np.array([str(ds.data_all[i]['cls_name']) for i in indices]))
    stage.source_labels = lambda smoke, corrupt: (
        ds, indices, records, shifted_labels if corrupt else true_labels,
        corruption_audit if corrupt else [])
    q = stage.compute_statealigned_pc1(ds, indices, patch, False)
    payloads = []
    for variant, corrupt in [('E1_PC1Removed-TrueMask', False),
                             ('E2_PC1Removed-CorruptedMask', True)]:
        status('SOURCE_ASSET_TRAINING', variant=variant)
        stage.train_pc1_removed_statealigned(variant, corrupt, q, DEVICE, False)
        payloads.append(torch.load(stage.OUT / f'{variant}_formal.pt',
                                  map_location='cpu', weights_only=False))
    st, sc = [np.asarray(p['patch_states'], np.float32) for p in payloads]
    adapter = make_model(sc, payloads[1], DEVICE).eval()
    for p in adapter.parameters():
        p.requires_grad_(False)
    d = .5 * (st[1] - st[0])
    hybrid = dict(use_align=True, q_pc1=q, align_model=adapter,
                  m=.5 * (sc[0] + sc[1]), alpha=float(np.linalg.norm(d)),
                  init_v=paired.hist.unit_np(d))
    baseline = dict(use_align=False, q_pc1=None, align_model=None)
    entries, manifests, batches_seen = [], [], []
    for variant, transform in [('Original_MVTec', baseline), ('Exact_VisA_Recipe', hybrid)]:
        folder = WORK / variant
        z, y, batches, probe, meta = paired.prepare_source(
            folder, 'Full MVTec', 0, transform, DEVICE, False)
        manifests.append(paired.read_csv(folder / 'source_pools/Full_MVTec_seed0/PATCH_MANIFEST.csv'))
        batches_seen.append(paired.digest(batches))
        if len(manifests) == 2:
            assert manifests[0] == manifests[1]
            assert batches_seen[0] == batches_seen[1]
        saved, path = paired.train_one(folder, 'CES', 'Full MVTec', 0,
            z, y, batches, probe, meta, .001, variant, paired.json_hash(config), DEVICE)
        entries.append(dict(id=variant, saved=saved, path=path, role=variant,
                            transform=transform))
    paired.write_json(OUT / 'SOURCE_TRAINING_COMPLETE.json', dict(
        models=[dict(variant=e['id'], checkpoint=str(e['path']),
                     sha256=paired.file_hash(e['path'])) for e in entries],
        same_patch_manifest=True, same_batch_hash=batches_seen[0]))
    # Produce low-resolution maps from existing global/local features, no CLIP run.
    all_results = []
    for e in entries:
        status('TARGET_SCORING', variant=e['id'])
        pred = paired.predict_pair(WORK / e['id'], 'Full MVTec', 'VisA',
                                   [e], e['transform'], DEVICE, False)
        loader_rows = paired.read_csv(pred / 'TARGET_MANIFEST.csv')
        canonical = official.readcsv(official.WORK / 'manifests/VisA.csv')
        lookup = {official.normpath(r['img_path']): r for r in canonical}
        native = [lookup[official.normpath(r['image'])] for r in loader_rows]
        dest = WORK / e['id'] / 'predictions/VisA'
        dest.mkdir(parents=True, exist_ok=True)
        paired.write_csv(dest / 'TARGET_MANIFEST.csv', native)
        # Reuse the exact native AP evaluator without copying the map payload.
        target_map = dest / 'maps74.npy'
        if not target_map.exists():
            import os
            os.link(pred / 'maps74.npy', target_map)
        paired.write_json(dest / 'COMPLETE.json', dict(status='READY'))
        geometry.OUT = WORK / e['id']
        geometry.SOURCE = 'Full MVTec'
        model = dict(model_id=e['id'], variant=e['id'], seed=0,
                     checkpoint=str(e['path']), checkpoint_sha256=paired.file_hash(e['path']))
        rows = geometry.evaluate('VisA', config, [model])
        all_results.append(dict(source='Full MVTec', target='VisA', seed=0,
            variant=e['id'], P_AP=float(np.mean([r['P_AP'] for r in rows])),
            checkpoint=str(e['path']), evaluator='native GT category-macro exact AP',
            evaluator_hash=official.evaluator_hash(), map_cache=str(target_map)))
        paired.write_csv(OUT / 'RESULTS.csv', all_results)
        gc.collect()
    delta = all_results[1]['P_AP'] - all_results[0]['P_AP']
    (OUT / 'REPORT.md').write_text('# MVTec source exact VisA recipe\n\n'
        + '\n'.join(f"- {r['variant']}: P-AP {r['P_AP']:.6f}%" for r in all_results)
        + f'\n- Paired delta: {delta:+.6f} percentage points.\n'
        + 'Seed-0 exploratory source-processing comparison, not an independent '
          'PC1/adapter/center ablation. No target tuning or formal table replacement.\n', encoding='utf-8')
    status('COMPLETE', results=all_results, delta_P_AP=delta)


if __name__ == '__main__':
    try:
        run()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / 'ERROR.txt').write_text(traceback.format_exc(), encoding='utf-8')
        status('FAILED', error=traceback.format_exc())
        raise
