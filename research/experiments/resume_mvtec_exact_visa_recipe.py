"""Resume sealed heads after relocation-aware target manifest validation."""
from pathlib import Path
import sys
import os
import traceback
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import phase_mvtec_exact_visa_recipe as job


def run():
    p, official, geo = job.paired, job.official, job.geometry
    out, work = job.OUT, job.WORK
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = official.readjson(out / 'CONFIG.json')
    sealed = official.readjson(out / 'SOURCE_TRAINING_COMPLETE.json')
    assert len(sealed['models']) == 2
    assets = out / 'source_assets'
    e1, e2 = [torch.load(assets / name, map_location='cpu', weights_only=False)
              for name in ('E1_PC1Removed-TrueMask_formal.pt',
                           'E2_PC1Removed-CorruptedMask_formal.pt')]
    st, sc = [np.asarray(a['patch_states'], np.float32) for a in (e1, e2)]
    q = np.load(assets / 'q_pc1_statealigned1024_formal.npy')
    adapter = job.make_model(sc, e2, job.DEVICE).eval()
    for param in adapter.parameters():
        param.requires_grad_(False)
    d = .5 * (st[1] - st[0])
    transforms = {
        'Original_MVTec': dict(use_align=False, q_pc1=None, align_model=None),
        'Exact_VisA_Recipe': dict(use_align=True, q_pc1=q, align_model=adapter,
            m=.5 * (sc[0] + sc[1]), alpha=float(np.linalg.norm(d)),
            init_v=p.hist.unit_np(d))}
    canonical = official.readcsv(official.WORK / 'manifests/VisA.csv')
    lookup = {official.normpath(r['img_path']): r for r in canonical}
    results = []
    for seal in sealed['models']:
        variant, cp = seal['variant'], Path(seal['checkpoint'])
        assert p.file_hash(cp) == seal['sha256'], 'Sealed trained checkpoint changed'
        saved = torch.load(cp, map_location='cpu', weights_only=False)
        entry = dict(id=variant, path=cp, saved=saved, role=variant)
        job.status('RESUMED_TARGET_SCORING', variant=variant, retraining=False)
        pred = p.predict_pair(work / variant, 'Full MVTec', 'VisA',
                              [entry], transforms[variant], job.DEVICE, False)
        loader = p.read_csv(pred / 'TARGET_MANIFEST.csv')
        paths = official.canonical_paths('VisA', [r['image'] for r in loader], canonical)
        native = [lookup[official.normpath(path)] for path in paths]
        assert len(native) == len(canonical) == len({r['sample_id'] for r in native})
        for old, new in zip(loader, native):
            assert old['category'] == new['category']
            assert int(old['label']) == int(new['label'])
        p.write_csv(out / f'{variant}_PATH_ALIGNMENT.csv', [dict(
            cache_position=i, cached_image=old['image'], canonical_image=new['img_path'],
            sample_id=new['sample_id'], image_sha256=new['image_sha256'], status='PASS')
            for i, (old, new) in enumerate(zip(loader, native))])
        dest = work / variant / 'predictions/VisA'
        dest.mkdir(parents=True, exist_ok=True)
        p.write_csv(dest / 'TARGET_MANIFEST.csv', native)
        target_map = dest / 'maps74.npy'
        if not target_map.exists():
            os.link(pred / 'maps74.npy', target_map)
        p.write_json(dest / 'COMPLETE.json', dict(status='READY', path_alignment='byte verified'))
        geo.OUT, geo.SOURCE = work / variant, 'Full MVTec'
        model = dict(model_id=variant, variant=variant, seed=0,
                     checkpoint=str(cp), checkpoint_sha256=seal['sha256'])
        rows = geo.evaluate('VisA', config, [model])
        results.append(dict(source='Full MVTec', target='VisA', seed=0, variant=variant,
            P_AP=float(np.mean([r['P_AP'] for r in rows])), checkpoint=str(cp),
            evaluator='native GT category-macro exact AP', evaluator_hash=official.evaluator_hash(),
            map_cache=str(target_map)))
        p.write_csv(out / 'RESULTS.csv', results)
    delta = results[1]['P_AP'] - results[0]['P_AP']
    (out / 'REPORT.md').write_text('# MVTec exact VisA recipe, seed 0\n\n'
        + '\n'.join(f"- {r['variant']}: P-AP {r['P_AP']:.6f}%" for r in results)
        + f'\n- Paired delta: {delta:+.6f} percentage points.\n'
        + '\nThe original run stopped after training, at absolute-path matching. '
          'This resume reuses both sealed trained heads and existing maps. '
          'Relocated images are verified by the official canonical_paths helper '
          'using SHA256, with category/label and unique coverage assertions. '
          'No retraining, target selection or formal-result overwrite.\n', encoding='utf-8')
    job.status('COMPLETE', results=results, delta_P_AP=delta)


if __name__ == '__main__':
    try:
        run()
    except Exception:
        job.status('FAILED_RESUME', error=traceback.format_exc())
        raise
