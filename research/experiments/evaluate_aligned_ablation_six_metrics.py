"""Cache-only six-metric evaluation of the repaired controls on MVTec/VisA.

No encoder, training module, target selection, or checkpoint loading is needed.
All three tables share exactly the same component Full prediction/checkpoint.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import final_unified_evaluation as ev

OUT = ROOT / 'results/aligned_ablation_six_metrics_mvtec_visa/formal'
WORK = Path('E:/MRAD_aligned_ablation_six_metrics/20261028')
COMP = ROOT / 'results/component_ablation_original_cache_repair/formal'
ALIGNED = ROOT / 'results/linear_capacity_original_cache_aligned/formal'
COMP_MAPS = Path('E:/MRAD_component_original_cache_repair/20261027/predictions')
CONTROL_MAPS = Path('E:/MRAD_linear_capacity_original_cache_aligned/20261027/predictions')
OLD_HEAD = ROOT / 'results/phase_ces_lhc_paired_repeat/formal/predictions/Full_MVTec__VisA'
OLD_K = ROOT / 'results/phase_ces_capacity_matched30/formal/predictions/VisA'
TARGETS = ('VisA', 'MVTec')
METRICS = ('I_AUROC', 'I_AP', 'P_AUROC', 'P_AP', 'P_F1max', 'AUPRO')
GROUPS = {
    'COMPONENT': ('Trained CES', 'CES + NTR', 'CES + LRP', 'CESAD-Full'),
    'LINEAR_HEAD': ('LinearHead', 'NormalizedLinearHead', 'CESAD-Full'),
    'STATE_CAPACITY': ('CESAD-Full', 'K3_30', 'K5_30'),
}
T0 = time.time()


def status(phase, **kw):
    obj = dict(phase=phase, pid=os.getpid(), local_time=time.strftime('%Y-%m-%d %H:%M:%S'),
               session_seconds=time.time() - T0, **kw)
    ev.savejson(OUT / 'STATUS.json', obj)
    print(json.dumps(obj), flush=True)


def lock(path, value):
    if path.exists():
        assert ev.readjson(path) == value, f'Refuse changed resume inputs: {path}'
    else:
        ev.savejson(path, value)


def identity(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=ev.sha(path))


def component_name(r):
    ntr = r['ntr'] in (True, 'True')
    lrp = r['lrp'] in (True, 'True')
    return {(False, False): 'Trained CES', (True, False): 'CES + NTR',
            (False, True): 'CES + LRP', (True, True): 'CESAD-Full'}[ntr, lrp]


def image_order(folder, target, rows, historical):
    if historical:
        manifest = folder / 'TARGET_MANIFEST.csv'
        rr = ev.readcsv(manifest)
        assert len(rr) == len(rows)
        assert [int(r['position']) for r in rr] == list(range(len(rr)))
        canonical = ev.canonical_paths(target, [r['image'] for r in rr], rows)
        positions = {ev.normpath(p): i for i, p in enumerate(canonical)}
        assert len(positions) == len(rows)
        order = [positions[ev.normpath(r['img_path'])] for r in rows]
        for i, j in enumerate(order):
            assert rows[i]['category'] == rr[j]['category']
            assert int(rows[i]['label']) == int(rr[j]['label'])
        return order, identity(manifest)
    return list(range(len(rows))), None


def registry():
    result = []
    components = ev.readcsv(COMP / 'CORRECTED_COMPONENT_RAW.csv')
    controls = ev.readcsv(ALIGNED / 'ALIGNED_RAW.csv')
    for target in TARGETS:
        rows = ev.readcsv(ev.WORK / 'manifests' / f'{target}.csv')
        component_folder = COMP_MAPS / target
        plan = ev.readjson(component_folder / 'MODELS.json')
        proof = ev.readjson(component_folder / 'INPUT_IDENTITY.json')
        assert proof['manifest']['sha256'] == ev.sha(ev.WORK / 'manifests' / f'{target}.csv')
        assert ev.readjson(component_folder / 'COMPLETE.json')['n'] == len(rows)
        for r in components:
            if r['target'] != target:
                continue
            match = [e for e in plan['outputs'] if e['checkpoint'] == r['checkpoint']
                     and e['lrp'] == (r['lrp'] == 'True')]
            assert len(match) == 1
            e = match[0]
            result.append(dict(target=target, source=r['source'], seed=int(r['seed']),
                configuration=component_name(r), checkpoint=e['checkpoint'],
                checkpoint_sha256=e['checkpoint_sha256'], previous_P_AP=float(r['P_AP']),
                total_updates=e['total_updates'], lrp=e['lrp'], kind='chunks',
                folder=str(component_folder), column=plan['new_outputs'].index(e['output_id']),
                output_id=e['output_id'], order=list(range(len(rows)))))
        for r in controls:
            if r['target'] != target or r['variant'] == 'CES_NTR':
                continue
            if target == 'MVTec':
                folder = CONTROL_MAPS / target
                plan2 = ev.readjson(folder / 'MODELS.json')
                assert ev.readjson(folder / 'INPUT_IDENTITY.json')['manifest']['sha256'] == proof['manifest']['sha256']
                assert ev.readjson(folder / 'COMPLETE.json')['n'] == len(rows)
                selected = [e for e in plan2['outputs'] if e['checkpoint_sha256'] == r['checkpoint_sha256']]
                assert len(selected) == 1
                e = selected[0]
                column = plan2['new_outputs'].index(e['output_id'])
                kind = 'chunks'
                order = list(range(len(rows)))
                manifest = None
            else:
                historical = not r['variant'].startswith('K')
                folder = OLD_HEAD if historical else OLD_K
                done = ev.readjson(folder / 'COMPLETE.json')
                key = 'checkpoint_hash' if historical else 'checkpoint_sha256'
                matches = [(j, e) for j, e in enumerate(done['models']) if e[key] == r['checkpoint_sha256']]
                assert len(matches) == 1, (r, matches)
                column, e = matches[0]
                if not historical:
                    assert done['GT_manifest']['sha256'] == proof['manifest']['sha256']
                order, manifest = image_order(folder, target, rows, historical)
                kind = 'npy'
            result.append(dict(target=target, source=r['source'], seed=int(r['seed']),
                configuration=r['variant'], checkpoint=r['checkpoint'],
                checkpoint_sha256=r['checkpoint_sha256'], previous_P_AP=float(r['P_AP']),
                total_updates=int(r['total_updates']), lrp=True, kind=kind, folder=str(folder),
                column=column, order=order, order_manifest=manifest))
    assert len(result) == 48
    assert len({(e['target'], e['seed'], e['configuration']) for e in result}) == 48
    for e in result:
        assert e['total_updates'] == 30
        assert ev.sha(e['checkpoint']) == e['checkpoint_sha256']
        e['evaluation_id'] = f"{e['target']}__seed{e['seed']}__{e['configuration'].replace(' ', '').replace('+', '_')}"
    return result


def verify_maps(entries):
    checked = {}
    for e in entries:
        folder = Path(e['folder'])
        if str(folder) in checked:
            continue
        done = ev.readjson(folder / 'COMPLETE.json')
        checks = []
        if e['kind'] == 'chunks':
            for start in range(0, done['n'], 25):
                path = folder / f'chunk_{start:05d}.npz'
                cp = ev.readjson(path.with_suffix('.json'))
                item = identity(path)
                assert item['sha256'] == cp['sha256'], path
                assert cp['start'] == start and cp['stop'] == min(start + 25, done['n'])
                checks.append(item)
        else:
            item = identity(folder / 'maps74.npy')
            expected = done.get('maps_hash') or done['maps']['sha256']
            assert item['sha256'] == expected
            checks.append(item)
        checked[str(folder)] = checks
    return checked


class Reader:
    def __init__(self, entry):
        self.e = entry
        self.chunk = None
        self.pack = None
        self.array = np.load(Path(entry['folder']) / 'maps74.npy', mmap_mode='r') if entry['kind'] == 'npy' else None

    def get(self, i):
        i = self.e['order'][i]
        if self.e['kind'] == 'npy':
            a = np.asarray(self.array[i, self.e['column']], dtype=np.float32)
        else:
            start = i // 25 * 25
            if start != self.chunk:
                if self.pack is not None:
                    self.pack.close()
                self.pack = np.load(Path(self.e['folder']) / f'chunk_{start:05d}.npz')
                self.array = self.pack[f"m{self.e['column']}"]
                self.chunk = start
            a = self.array[i - start]
        expected = (74, 74) if self.e['lrp'] else (37, 37)
        assert a.dtype == np.float32 and a.shape == expected and np.isfinite(a).all()
        return a

    def close(self):
        if self.pack is not None:
            self.pack.close()
        self.array = None


def smoke(entries, evaluator_id):
    checks = []
    for target in TARGETS:
        e = next(e for e in entries if e['target'] == target and e['configuration'] == 'Trained CES' and e['seed'] == 0)
        rows = ev.readcsv(ev.WORK / 'manifests' / f'{target}.csv')
        ids = [next(i for i, r in enumerate(rows) if int(r['label']) == label) for label in (0, 1)]
        reader = Reader(e)
        masks = [ev.native_mask(rows[i]) for i in ids]
        scores = [ev.resize(reader.get(i), m.shape) for i, m in zip(ids, masks)]
        y = np.concatenate([a.ravel() for a in masks])
        s = np.concatenate([a.ravel() for a in scores])
        got = ev.weighted_pixel_metrics(y, s)
        direct = average_precision_score(y, s) * 100
        assert abs(got['P_AP'] - direct) < 1e-8
        pro = ev.aupro(masks, scores)
        assert pro is not None and math.isfinite(pro)
        checks.append(dict(target=target, samples=[rows[i]['sample_id'] for i in ids],
            exact_AP_error=abs(got['P_AP'] - direct), AUPRO=pro, evaluator_id=evaluator_id))
        reader.close()
        del y, s, masks, scores
        gc.collect()
    ev.savejson(OUT / 'SMOKE_CHECKS.json', checks)


def image_metrics(entry, evaluator_id):
    path = OUT / 'image_categories' / f"{entry['evaluation_id']}.json"
    if path.exists():
        saved = ev.readjson(path)
        assert saved['checkpoint_sha256'] == entry['checkpoint_sha256'] and saved['evaluator_id'] == evaluator_id
        return saved['categories']
    rows = ev.readcsv(ev.WORK / 'manifests' / f"{entry['target']}.csv")
    reader = Reader(entry)
    rr = []
    for i, r in enumerate(rows):
        score = ev.topmean(ev.resize(reader.get(i), (74, 74)))
        rr.append(dict(sample_id=r['sample_id'], category=r['category'], label=int(r['label']),
            image_score=score, image_grid='74x74', top_k=55))
    reader.close()
    ev.writecsv(OUT / 'image_scores' / f"{entry['evaluation_id']}.csv", rr)
    categories = []
    for category in sorted({r['category'] for r in rr}):
        cell = [r for r in rr if r['category'] == category]
        y = [r['label'] for r in cell]
        s = [r['image_score'] for r in cell]
        valid = len(set(y)) == 2
        categories.append(dict(category=category, images=len(cell),
            I_AUROC=float(roc_auc_score(y, s) * 100) if valid else None,
            I_AP=float(average_precision_score(y, s) * 100) if valid else None))
    ev.savejson(path, dict(checkpoint_sha256=entry['checkpoint_sha256'], evaluator_id=evaluator_id, categories=categories))
    return categories


def pixel_category(entry, category, evaluator_id):
    path = OUT / 'pixel_categories' / entry['evaluation_id'] / f'{category}.json'
    if path.exists():
        saved = ev.readjson(path)
        assert saved['checkpoint_sha256'] == entry['checkpoint_sha256'] and saved['evaluator_id'] == evaluator_id
        return saved
    rows = ev.readcsv(ev.WORK / 'manifests' / f"{entry['target']}.csv")
    ids = [i for i, r in enumerate(rows) if r['category'] == category]
    n = sum(int(rows[i]['gt_height']) * int(rows[i]['gt_width']) for i in ids)
    assert shutil.disk_usage(WORK).free > n * 5 + 512_000_000, 'Insufficient isolated scratch space'
    status('PIXEL_CATEGORY', evaluation_id=entry['evaluation_id'], category=category, pixels=n)
    yp = WORK / 'scratch' / 'labels.npy'
    sp = WORK / 'scratch' / 'scores.npy'
    y = np.lib.format.open_memmap(yp, mode='w+', dtype=np.bool_, shape=(n,))
    s = np.lib.format.open_memmap(sp, mode='w+', dtype=np.float32, shape=(n,))
    reader = Reader(entry)
    masks, scores = [], []
    offset = 0
    for i in ids:
        m = ev.native_mask(rows[i])
        a = ev.resize(reader.get(i), m.shape)
        end = offset + m.size
        y[offset:end] = m.ravel()
        s[offset:end] = a.ravel()
        masks.append(y[offset:end].reshape(m.shape))
        scores.append(s[offset:end].reshape(m.shape))
        offset = end
    y.flush()
    s.flush()
    reader.close()
    assert offset == n
    value = ev.weighted_pixel_metrics(y, s)
    status('AUPRO_CATEGORY', evaluation_id=entry['evaluation_id'], category=category, pixels=n)
    value['AUPRO'] = ev.aupro(masks, scores)
    saved = dict(category=category, images=len(ids), pixels=n, checkpoint_sha256=entry['checkpoint_sha256'],
        evaluator_id=evaluator_id, **value)
    ev.savejson(path, saved)
    masks.clear()
    scores.clear()
    del y, s, a, m
    gc.collect()
    yp.unlink()
    sp.unlink()
    return saved


def finite_mean(rows, metric):
    values = [r[metric] for r in rows if r.get(metric) is not None and math.isfinite(r[metric])]
    assert len(values) == len(rows), (metric, 'Unexpected invalid category')
    return float(np.mean(values))


def export(entries, raw):
    ev.writecsv(OUT / 'SIX_METRICS_RAW.csv', raw)
    for group, configs in GROUPS.items():
        subset = [r for r in raw if r['configuration'] in configs]
        ev.writecsv(OUT / f'{group}_RAW.csv', subset)
        summary = []
        for target in (*TARGETS, 'Two-transfer mean'):
            for config in configs:
                values = []
                for seed in (0, 1, 2):
                    cells = [r for r in subset if r['configuration'] == config and r['seed'] == seed
                             and (target == 'Two-transfer mean' or r['target'] == target)]
                    expected = 2 if target == 'Two-transfer mean' else 1
                    if len(cells) != expected:
                        break
                    values.append({m: float(np.mean([r[m] for r in cells])) for m in METRICS})
                if len(values) != 3:
                    continue
                row = dict(target=target, configuration=config, seeds='0/1/2', n_seeds=3)
                for m in METRICS:
                    row[m + '_mean'] = float(np.mean([v[m] for v in values]))
                    row[m + '_sd'] = float(np.std([v[m] for v in values], ddof=1))
                summary.append(row)
        ev.writecsv(OUT / f'{group}_SUMMARY.csv', summary)
        text = [f'# {group}: MVTec / VisA six-metric evaluation', '',
                'All values are category-macro percentages; uncertainty is sample SD over seeds 0/1/2.',
                'Two-transfer means are averaged within seed before computing sample SD.', '',
                '| Target | Configuration | ' + ' | '.join(METRICS) + ' |',
                '|---|---|' + '|'.join(['---:'] * len(METRICS)) + '|']
        for r in summary:
            cells = [f"{r[m + '_mean']:.2f} +/- {r[m + '_sd']:.2f}" for m in METRICS]
            text.append('| ' + r['target'] + ' | ' + r['configuration'] + ' | ' + ' | '.join(cells) + ' |')
        (OUT / f'{group}_TABLE.md').write_text('\n'.join(text) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('preflight', 'smoke', 'run'), default='run')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (WORK / 'scratch').mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    evaluator_id = ev.evaluator_hash()
    assert evaluator_id == '22b3608b28e166fbe7a0a05d307b4777c34008360be59e4e2aa2258f383ec0cb'
    entries = registry()
    config = dict(evaluator_id=evaluator_id, pixel_protocol=ev.POLICY,
        image_score='74x74 bilinear align_corners=True; top ceil(0.01*5476)=55 mean; category macro',
        seeds=[0, 1, 2], targets=list(TARGETS), unique_cells=48, groups=GROUPS,
        training=False, encoder_inference=False, postprocessing='none added; raw cached logits',
        script=identity(Path(__file__)), evaluator=identity(Path(ev.__file__)),
        component_source=identity(COMP / 'CORRECTED_COMPONENT_RAW.csv'),
        aligned_source=identity(ALIGNED / 'ALIGNED_RAW.csv'),
        manifests=[identity(ev.WORK / 'manifests' / f'{t}.csv') for t in TARGETS])
    # JSON roundtrip makes tuple/list representation stable for resumptions.
    lock(OUT / 'CONFIG.json', json.loads(json.dumps(config)))
    lock(OUT / 'REGISTRY.json', entries)
    proof_path = OUT / 'CACHE_HASH_VERIFICATION.json'
    if not proof_path.exists():
        status('VERIFYING_PREDICTION_CACHE_HASHES', unique_cells=len(entries))
        ev.savejson(proof_path, verify_maps(entries))
    status('PREFLIGHT_PASS', unique_cells=len(entries), training=False, encoder_inference=False)
    if args.stage == 'preflight':
        return
    if not (OUT / 'SMOKE_CHECKS.json').exists():
        smoke(entries, evaluator_id)
    if args.stage == 'smoke':
        status('SMOKE_PASS')
        return
    image_results = {}
    for index, e in enumerate(entries):
        image_results[e['evaluation_id']] = image_metrics(e, evaluator_id)
        status('IMAGE_METRICS', complete=index + 1, total=48)
    image_raw = []
    for e in entries:
        image_raw.append({k: e[k] for k in ('source', 'target', 'seed', 'configuration', 'checkpoint', 'checkpoint_sha256')} |
            {m: finite_mean(image_results[e['evaluation_id']], m) for m in METRICS[:2]})
    ev.writecsv(OUT / 'IMAGE_METRICS_RAW.csv', image_raw)
    raw = []
    for index, e in enumerate(entries):
        rows = ev.readcsv(ev.WORK / 'manifests' / f"{e['target']}.csv")
        category_results = [pixel_category(e, c, evaluator_id) for c in sorted({r['category'] for r in rows})]
        result = {k: e[k] for k in ('source', 'target', 'seed', 'configuration', 'checkpoint', 'checkpoint_sha256', 'total_updates', 'lrp')}
        result.update({m: finite_mean(image_results[e['evaluation_id']], m) for m in METRICS[:2]})
        result.update({m: finite_mean(category_results, m) for m in METRICS[2:]})
        result.update(evaluator_id=evaluator_id, map_cache=e['folder'],
            pixel_valid_categories=len(category_results), image_valid_categories=len(image_results[e['evaluation_id']]),
            previous_P_AP=e['previous_P_AP'], P_AP_parity_error=result['P_AP'] - e['previous_P_AP'])
        assert abs(result['P_AP_parity_error']) < 0.001, ('P-AP differs from verified raw record', result)
        raw.append(result)
        export(entries, raw)
        status('CELL_COMPLETE', completed=index + 1, total=48, evaluation_id=e['evaluation_id'],
               P_AP=result['P_AP'], parity_error=result['P_AP_parity_error'])
    assert len(raw) == 48
    (OUT / 'REPORT.md').write_text(
        '# Aligned ablation six-metric evaluation\n\n'
        'Completed 48 unique configuration/transfer/seed cells from immutable raw prediction caches. '
        'No model training or CLIP inference was performed. The component, linear-head and capacity tables '
        'share the exact same CESAD-Full checkpoints and metrics. K3/K5 use 30 total updates and retain '
        'their existing multi-state aggregation/diversity design; this is not a count-only causal ablation.\n\n'
        'Image scoring: float32 bilinear to 74x74 (align_corners=True), top 55 scores averaged in float64; '
        'per-category image AUROC/AP then equal-category mean. Pixel metrics: direct bilinear resize from '
        'the raw 37x37 or 74x74 map to paired native GT, exact-score-tie sklearn AP/AUROC/full-PR F1; '
        '200-threshold AUPRO, 8-connectivity, FPR cap 0.30. No new smoothing or normalization.\n\n'
        'All six metrics are percentages. Summary uncertainty is ddof=1 across three training seeds. '
        'The two-transfer average is computed within each seed before its uncertainty. '
        'Existing six-target paper tables, source assets, predictions and manuscripts are unchanged.\n', encoding='utf-8')
    status('COMPLETE', unique_cells=48, max_P_AP_parity_error=max(abs(r['P_AP_parity_error']) for r in raw))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        error = traceback.format_exc()
        if OUT.exists():
            status('ERROR', traceback=error)
        raise
