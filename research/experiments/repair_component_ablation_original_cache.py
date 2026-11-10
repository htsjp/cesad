"""Restore physical source identities and original VisA supervision.

Only downstream CES heads are fitted. Upstream parameters, old predictions,
historical ablations and the main comparison table are immutable inputs.
"""
from pathlib import Path
import sys
import os
import gc
import traceback
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import phase_ces_unified_ablation_final as u
from experiments import phase_ces_lhc_paired_repeat as p
from experiments import final_unified_evaluation as official
from experiments import phase_pc1_orthogonal_anomaly_sufficiency as labels_old

META = ROOT / 'results/component_ablation_original_cache_repair/formal'
WORK = Path('E:/MRAD_component_original_cache_repair/20261027')
AUDIT = ROOT / 'results/ablation_main_cache_identity_20261027'
SOURCES = ('Full MVTec', 'Full VisA')
SEEDS = (0, 1, 2)
TARGETS = ('VisA', 'MVTec', 'BTAD', 'MPDD', 'DTD-Synthetic', 'DAGM')
DEVICE = 'cuda'


def status(phase, **kw):
    p.status(META, phase, **kw)


def setup():
    META.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available()
    # Redirect only new runtime outputs; no existing runner is edited.
    u.OUT = WORK
    u.status = status
    checks = official.readcsv(AUDIT / 'LIVE_SOURCE_IMAGE_PARITY.csv')
    assert len(checks) == 2
    assert all(float(r['E_cache_max_abs']) < 2e-4 and float(r['carrier_max_abs']) > .01 for r in checks)
    config = dict(seeds=list(SEEDS), targets=list(TARGETS), source_feature_root='E:/MRAD_final_evidence_cache',
        restore='physical image/cache row alignment and original VisA nearest-ID-mask supervision',
        source_preprocessing='MVTec L2; VisA fixed original PC1 -> L2 -> frozen E2 adapter -> L2',
        upstream_parameters_retrained=False, epochs=10, lr=.001, weight_decay=0., rho=.1,
        NTR_weight=1., LRP_global_weight=.5, checkpoint_selection='last fixed-budget update',
        evaluator_id=official.evaluator_hash(), main_table_modified=False,
        code=[dict(path=str(q), sha256=p.file_hash(q)) for q in
              (Path(__file__), Path(p.__file__), Path(u.__file__), Path(labels_old.__file__))])
    dest = META / 'CONFIG.json'
    if dest.exists():
        assert official.readjson(dest) == config, 'Locked repair configuration changed'
    else:
        p.write_json(dest, config)
    return config


def source_readers():
    # E-drive arrays were produced directly in current light_fold image order;
    # unlike the carrier arrays, there is no old metadata-order substitution.
    original_mask = p.mask_only
    nearest, _, _ = labels_old.source_label_maps(False)
    _, visa_ds, train, held = p.source_dataset('Full VisA')
    inverse = {int(idx): pos for pos, idx in enumerate(train + held)}

    def exact_arrays(direction):
        source = 'Full MVTec' if direction == 'D1' else 'Full VisA'
        name = 'MVTec' if direction == 'D1' else 'VisA'
        _, ds, tr, hd = p.source_dataset(source)
        path = Path('E:/MRAD_final_evidence_cache') / f'global_patch_{name}.npy'
        a = np.load(path, mmap_mode='r')
        assert a.shape == (len(tr) + len(hd), 1369, 1024)
        return [a[:len(tr)], a[len(tr):]], [path, path]

    def exact_mask(ds, idx):
        if str(ds.root).replace('\\', '/').casefold() == str(visa_ds.root).replace('\\', '/').casefold():
            return np.asarray(nearest.get(inverse[int(idx)], np.zeros(1369)), np.int64)
        return original_mask(ds, idx)

    p.source_arrays = exact_arrays
    p.mask_only = exact_mask


def fit(config):
    models = []
    base_objective = p.objective
    config_hash = p.json_hash(config)
    for source in SOURCES:
        transform = p.load_transform(source, DEVICE)
        for seed in SEEDS:
            status('SOURCE_POOL', source=source, seed=seed)
            z, y, batches, probe, meta = p.prepare_source(WORK, source, seed, transform, DEVICE, False)
            assert batches.shape == (30, 8190 if source == 'Full MVTec' else 8184)
            # Verify saved pool reconstruction against the physical E-drive rows,
            # independently of the helper that originally wrote the pool.
            folder = WORK / 'source_pools' / f'{source.replace(" ", "_")}_seed{seed}'
            manifest = p.read_csv(folder / 'PATCH_MANIFEST.csv')
            a = np.load(Path('E:/MRAD_final_evidence_cache') / f'global_patch_{"MVTec" if source=="Full MVTec" else "VisA"}.npy', mmap_mode='r')
            error = 0.
            for start in range(0, len(manifest), 2048):
                records = manifest[start:start+2048]
                raw = np.stack([a[int(r['source_position']), int(r['patch_flat_index'])] for r in records]).astype(np.float32)
                actual = p.transformed(raw, transform, DEVICE).cpu().numpy()
                error = max(error, float(np.max(abs(actual - z[start:start+len(records)]))))
            assert error < 2e-6
            p.write_json(folder / 'E_CACHE_RECONSTRUCTION_CHECK.json', dict(max_abs_error=error,
                patches=len(manifest), labels_hash=meta['labels_hash'], batch_hash=meta['batch_stream_hash'], status='PASS'))
            for variant, ntr in [('CES_noNTR', False), ('CES_NTR', True)]:
                def no_tail(logits, yy):
                    base, bce, dice = p.balanced_loss(logits, yy)
                    return base, bce, dice, logits.new_tensor(0.)
                p.objective = base_objective if ntr else no_tail
                saved, cp = p.train_one(WORK, 'CES', source, seed, z, y, batches, probe, meta,
                                        .001, variant, config_hash, DEVICE)
                assert saved['realized_updates'] == 30
                assert saved['batch_stream_hash'] == meta['batch_stream_hash']
                models.append(dict(source=source, seed=seed, variant=variant, ntr=ntr,
                    model_id=f'{source.replace(" ", "_")}__seed{seed}__{variant}',
                    checkpoint=str(cp), checkpoint_sha256=p.file_hash(cp),
                    feature_hash=meta['feature_array_hash'], source_pool_hash=meta['manifest_hash'],
                    batch_hash=meta['batch_stream_hash'], total_updates=30))
            p.objective = base_objective
            gc.collect()
    p.write_json(META / 'SOURCE_TRAINING_COMPLETE.json', dict(models=models, config_hash=config_hash,
        independent_training_seeds=list(SEEDS), target_used_for_selection=False))
    return models


def summarize(config, plans):
    raw = []
    for target in TARGETS:
        categories = official.readcsv(WORK / 'evaluations' / f'{target}_CATEGORIES.csv')
        for e in plans[target]['outputs']:
            group = [r for r in categories if r['output_id'] == e['output_id']]
            assert len(group) == len({r['category'] for r in u.canonical(target)})
            raw.append(dict(source=e['source'], target=target, seed=e['seed'], ntr=e['ntr'], lrp=e['lrp'],
                variant=e['variant'], P_AP=float(np.mean([float(r['p_ap']) for r in group])),
                checkpoint=e['checkpoint'], evaluator_id=config['evaluator_id']))
    assert len(raw) == 72
    p.write_csv(META / 'CORRECTED_COMPONENT_RAW.csv', raw)
    summary = []
    for ntr in (False, True):
        for lrp in (False, True):
            name = 'CES' + ('+NTR' if ntr else '') + ('+LRP' if lrp else '')
            row = dict(configuration=name)
            for target in TARGETS:
                vv = [r['P_AP'] for r in raw if r['target']==target and r['ntr']==ntr and r['lrp']==lrp]
                assert len(vv)==3
                row[target]=float(np.mean(vv));row[target+'_SD']=float(np.std(vv,ddof=1))
            seed_means=[np.mean([r['P_AP'] for r in raw if r['seed']==seed and r['ntr']==ntr and r['lrp']==lrp]) for seed in SEEDS]
            row['Mean']=float(np.mean(seed_means));row['Mean_SD']=float(np.std(seed_means,ddof=1))
            summary.append(row)
    p.write_csv(META / 'CORRECTED_COMPONENT_SUMMARY.csv', summary)
    (META / 'REPORT.md').write_text('# Original-cache component repair\n\n'
        'The prior VisA carrier arrays contained the same feature values but a different image-row order. '
        'This run reads the original E-drive global features in verified physical image order; '
        'the original VisA nearest-ID-mask source labels are restored. MVTec source labels retain their original bilinear rule. '
        'Both training arms share physical patch IDs, features, labels, initialization and actual batches within each seed. '
        'Only NTR differs during training; LRP is a fixed inference toggle. '
        'All source heads are sealed before native GT evaluation. The fixed source PC1/E1/E2/adapter are not refitted. '
        'These new checkpoints are not silently substituted into the frozen main table. '
        'Legacy paired Linear/Normalized/K-state/center results using the affected VisA pool also require correction '
        'before they can be described as physical-patch-matched evidence; they are not rerun automatically here.\n', encoding='utf-8')
    status('COMPLETE', rows=72, summary=summary)


def run():
    config = setup()
    source_readers()
    if (META / 'SOURCE_TRAINING_COMPLETE.json').exists():
        sealed = official.readjson(META / 'SOURCE_TRAINING_COMPLETE.json')
        assert sealed['config_hash'] == p.json_hash(config)
        models = sealed['models']
        for e in models:
            assert p.file_hash(e['checkpoint']) == e['checkpoint_sha256']
    else:
        models = fit(config)
    plans = {}
    encoder = None
    for target in TARGETS:
        entries = []
        for model in models:
            if model['source'] != u.source_for(target):
                continue
            for lrp in (False, True):
                entries.append(dict(**model, lrp=lrp, output_id=model['model_id'] + ('__LRP' if lrp else '__Global')))
        plan = dict(target=target, outputs=entries, reuse={}, new_outputs=[e['output_id'] for e in entries], n=len(u.canonical(target)))
        plans[target] = plan
        status('PREDICTING', target=target, heads=6, configurations=12)
        encoder = u.predict(target, plan, encoder)
        u.evaluate(target, config, plan)
    summarize(config, plans)


if __name__ == '__main__':
    try:
        run()
    except BaseException:
        META.mkdir(parents=True, exist_ok=True)
        (META / 'ERROR.txt').write_text(traceback.format_exc(), encoding='utf-8')
        status('FAILED', error=traceback.format_exc())
        raise
