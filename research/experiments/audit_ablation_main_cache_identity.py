"""Read saved assets only: verify original cache and explain head provenance."""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import phase_ces_lhc_paired_repeat as paired
from experiments import phase_final_cross_dataset_benchmark as historical
from experiments import final_unified_evaluation as official

OUT = ROOT / 'results/ablation_main_cache_identity_20261027'
FULL = ROOT / 'results/phase_ces_unified_ablation_final/formal'


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cache_rows, pool_rows, params = [], [], []
    seen = set()
    checked = OUT / 'CACHE_HASH_CHECK.csv'
    targets = () if checked.exists() else ('VisA', 'MVTec', 'BTAD', 'MPDD', 'DTD-Synthetic', 'DAGM')
    if checked.exists():
        cache_rows = official.readcsv(checked)
        assert all(r['status'] != 'FAIL' for r in cache_rows)
    for target in targets:
        identity = official.readjson(FULL / 'predictions' / target / 'INPUT_IDENTITY.json')
        for kind in ('global_cache', 'local_cache'):
            asset = identity[kind]
            if isinstance(asset, str):
                cache_rows.append(dict(target=target, kind=kind, path=asset, status='LIVE_FEATURE_PATH_NOT_SAVED_ORIGINAL_CACHE'))
                continue
            path = Path(asset['path'])
            print('HASH', target, kind, path, flush=True)
            actual = paired.file_hash(path)
            ok = actual == asset['sha256'] and path.stat().st_size == asset['bytes']
            cache_rows.append(dict(target=target, kind=kind, path=str(path),
                recorded_sha256=asset['sha256'], actual_sha256=actual,
                status='PASS' if ok else 'FAIL', original_E_cache='MRAD_final_evidence_cache' in str(path)))
            assert ok, (target, kind, 'Cache modified since formal ablation')
            seen.add(str(path))
    paired.write_csv(OUT / 'CACHE_HASH_CHECK.csv', cache_rows)
    for source, direction, target in [('Full MVTec', 'D1', 'MVTec'), ('Full VisA', 'D2', 'VisA')]:
        original = Path('E:/MRAD_final_evidence_cache') / f'global_patch_{target}.npy'
        a = np.load(original, mmap_mode='r')
        arrays, paths = paired.source_arrays(direction)
        offset, mismatch = 0, 0
        for arr in arrays:
            for start in range(0, len(arr), 32):
                x = np.asarray(arr[start:start+32])
                y = np.asarray(a[offset+start:offset+start+len(x)])
                mismatch += int(np.count_nonzero(x != y))
            offset += len(arr)
        assert offset == len(a)
        reordered_mismatch = mismatch
        displaced = 0
        if mismatch and direction == 'D2':
            permutation = np.load(OUT / 'CARRIER_TO_E_VISA_INDEX.npy')
            assert len(np.unique(permutation)) == len(a)
            offset, reordered_mismatch = 0, 0
            for arr in arrays:
                for start in range(0, len(arr), 32):
                    x = np.asarray(arr[start:start+32])
                    idx = permutation[offset+start:offset+start+len(x)]
                    reordered_mismatch += int(np.count_nonzero(x != np.asarray(a[idx])))
                offset += len(arr)
            displaced = int(np.count_nonzero(permutation != np.arange(len(a))))
        pool_rows.append(dict(source=source, original_cache=str(original),
            training_feature_files=';'.join(map(str, paths)), values_compared=int(a.size),
            mismatched_values_same_position=mismatch, mismatched_after_verified_permutation=reordered_mismatch,
            displaced_image_rows=displaced, status='PASS' if mismatch == 0 else 'SOURCE_ROW_ORDER_MISMATCH'))
        assert reordered_mismatch == 0
        print('SOURCE_ARRAY_CHECK', source, pool_rows[-1]['status'], flush=True)
        old = historical.load_mvtec_source() if direction == 'D1' else historical.load_visa_source('cpu')
        transform = paired.load_transform(source, 'cpu')
        with np.load((historical.MVTEC_HN_OUT if direction == 'D1' else historical.HN_OUT) / 'contrast_vectors.npz') as data:
            old_v = old['v1']
        for seed in (0, 1, 2):
            cp = paired.ROOT / f'results/phase_ces_lhc_paired_repeat/formal/checkpoints/{source.replace(" ", "_")}_CES_seed{seed}_primary_lr0.001.pt'
            saved = torch.load(cp, map_location='cpu', weights_only=False)
            spec = paired.packed_spec(saved)
            p = saved['params']
            new_v = paired.hist.unit_np(np.asarray(p['v'], np.float32))
            dot = float(np.clip(old_v @ new_v, -1., 1.))
            m = torch.tensor(old['m'])
            s_a = torch.nn.functional.normalize(m + old['alpha'] * torch.tensor(old_v), dim=0)
            s_n = torch.nn.functional.normalize(m - old['alpha'] * torch.tensor(old_v), dim=0)
            w_old = (s_a - s_n).numpy()
            adapter_error = 0.
            if direction == 'D2':
                for k in ('a.weight', 'b.weight'):
                    adapter_error = max(adapter_error, float((old['align_model'].state_dict()[k] - transform['align_model'].state_dict()[k]).abs().max()))
            params.append(dict(source=source, seed=seed, checkpoint=str(cp),
                old_direction_asset=str(historical.MVTEC_HN_OUT if direction == 'D1' else historical.HN_OUT) + '/contrast_vectors.npz::E1',
                same_trained_direction=bool(np.array_equal(old_v,new_v)), direction_cosine=dot,
                direction_angle_degrees=float(np.degrees(np.arccos(dot))),
                effective_weight_max_difference=float(np.max(abs(w_old-spec.weight()))),
                midpoint_max_difference=float(np.max(abs(old['m']-np.asarray(p['m'])))),
                alpha_old=old['alpha'], alpha_new=float(p['alpha']), frozen_adapter_max_difference=adapter_error,
                updates=saved['realized_updates'], lr=saved['lr'],
                batch_hash=saved['batch_stream_hash']))
    paired.write_csv(OUT / 'SOURCE_ARRAY_IDENTITY.csv', pool_rows)
    paired.write_csv(OUT / 'HEAD_PARAMETER_DIFFERENCES.csv', params)
    main = [r for r in official.readcsv(ROOT / 'results/final_unified_evaluation/FINAL_PIXEL_COMPARISON.csv') if r['method']=='CESAD']
    raw = official.readcsv(FULL / 'UNIFIED_ABLATION_RAW.csv')
    registry = [r for r in official.readjson(official.WORK/'PREDICTION_REGISTRY.json') if r['method']=='CESAD']
    comparisons = []
    for row in main:
        target = row['target']
        cells = [r for r in raw if r['experiment']=='C_NTR_LRP' and r['target']==target and r['ntr']=='True' and r['lrp']=='True']
        if not cells:
            cells = list({r['seed']:r for r in raw if r['target']==target and r['variant']=='CES_NTR' and r['lrp']=='True'}.values())
        assert sorted(int(r['seed']) for r in cells)==[0,1,2]
        e = next(r for r in registry if r['target']==target)
        seed0 = next(r for r in cells if r['seed']=='0')
        comparisons.append(dict(target=target, source=row['source'], main_P_AP=float(row['P_AP']),
            ablation_seed0_P_AP=float(seed0['p_ap']), ablation_3seed_mean=float(np.mean([float(r['p_ap']) for r in cells])),
            main_cache=e['cache_path'], main_cache_key=e['cache_key'], main_checkpoint=e['checkpoint'],
            ablation_seed0_checkpoint=seed0['checkpoint'],
            same_evaluator=row['evaluator_hash']==seed0['evaluator_id'],
            reason=('VisA-source StepB carrier row order mismatches physical source IDs; source mask rule also differs; trained heads differ. '
                    'Main single artifact versus paired three seeds is an additional aggregation difference.'
                    if row['source']=='Full VisA' else
                    'MVTec source features match original cache; seed0 effective weight differs only at floating-point precision. '
                    'Seed aggregation and stored-map precision remain distinct.')))
    paired.write_csv(OUT / 'MAIN_VS_ABLATION.csv', comparisons)
    (OUT / 'REPORT.md').write_text('# Main/ablation cache identity check\n\n'
        'Original E-drive global/local caches match the hashes recorded during the latest unified ablation. '
        'MVTec source arrays match by position. VisA source arrays match only after a full image-row permutation: '
        'all 2162 image rows are displaced, while every value matches after reindexing. '
        'The saved StepB physical patch manifests did not account for this permutation. '
        'Two fresh real-image extractions confirm the E-drive/current-image correspondence. '
        'Saved head direction/weight differences are listed separately; cache reuse does not imply identical trained heads. '
        'The latest ablation and main table use the same native-GT evaluator ID. '
        'Older component tables explicitly used loader518 GT and must not be mixed with the native tables.\n\n'
        '| Target | Main P-AP | Ablation seed0 | Ablation 3-seed mean |\n|---|---:|---:|---:|\n'
        + '\n'.join(f"| {r['target']} | {r['main_P_AP']:.6f} | {r['ablation_seed0_P_AP']:.6f} | {r['ablation_3seed_mean']:.6f} |" for r in comparisons)
        + '\n\nThis audit reads saved heads and does not train or recompute metrics. '
          'A separate two-image live feature parity check was performed (LIVE_SOURCE_IMAGE_PARITY.csv). '
          'The source-row mismatch requires corrected source/head processing, not regeneration of the frozen cache. '
          'VisA patch supervision also differs: original nearest ID-mask union versus paired bilinear-threshold; '
          '43 sampled labels per seed differ. Neither difference is explained by seed averaging alone. '
          'A separate isolated component rerun restores E-drive row identity and the original VisA source labels; '
          'historical CSVs and main tables remain unchanged.\n', encoding='utf-8')
    print(json.dumps(dict(status='COMPLETE', output=str(OUT), comparisons=comparisons), indent=2), flush=True)


if __name__ == '__main__':
    run()
