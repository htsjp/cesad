"""Validate and format completed metrics; no inference or metric evaluation."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/aligned_ablation_six_metrics_mvtec_visa/formal'
METRICS = ('I_AUROC', 'I_AP', 'P_AUROC', 'P_AP', 'P_F1max', 'AUPRO')
GROUPS = ('COMPONENT', 'LINEAR_HEAD', 'STATE_CAPACITY')
TRANSFERS = {'VisA': 'MVTec -> VisA', 'MVTec': 'VisA -> MVTec',
             'Two-transfer mean': 'Two-transfer mean'}


def readcsv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    status = json.loads((OUT / 'STATUS.json').read_text(encoding='utf-8'))
    assert status['phase'] == 'COMPLETE'
    raw = readcsv(OUT / 'SIX_METRICS_RAW.csv')
    assert len(raw) == 48
    assert len({(r['target'], r['seed'], r['configuration']) for r in raw}) == 48
    assert all(float(r['P_AP_parity_error']) == 0 for r in raw)
    assert all(math.isfinite(float(r[m])) for r in raw for m in METRICS)
    assert all(int(r['total_updates']) == 30 for r in raw)
    category_files = list((OUT / 'pixel_categories').rglob('*.json'))
    assert len(category_files) == 648
    for r in raw:
        expected = 12 if r['target'] == 'VisA' else 15
        assert int(r['pixel_valid_categories']) == int(r['image_valid_categories']) == expected
    tables = {group: readcsv(OUT / f'{group}_SUMMARY.csv') for group in GROUPS}
    anchors = [r for r in tables['COMPONENT'] if r['configuration'] == 'CESAD-Full']
    for group in GROUPS[1:]:
        for anchor in anchors:
            matching = [r for r in tables[group] if r['target'] == anchor['target']
                        and r['configuration'] == 'CESAD-Full']
            assert matching == [anchor], (group, anchor)
    # Verify each raw Full cell, including its checkpoint, is literally shared.
    full = [r for r in raw if r['configuration'] == 'CESAD-Full']
    assert len(full) == 6
    for group in GROUPS:
        group_full = [r for r in readcsv(OUT / f'{group}_RAW.csv')
                      if r['configuration'] == 'CESAD-Full']
        assert group_full == full
    combined = []
    text = ['# Aligned ablations: six metrics on MVTec and VisA', '',
            'Completed: 48 unique model/transfer/seed cells and 648 category-level pixel evaluations.',
            'All values are percentages, reported as mean +/- sample SD over seeds 0/1/2 (ddof=1).',
            'Image: category macro, 74x74 grid, Top-1% mean (55 positions).',
            'Pixel: paired native-resolution GT, category macro, frozen unified evaluator.',
            'Two-transfer means are formed within each seed before calculating SD.',
            'An SD displayed as 0.00 is rounded, not necessarily zero; see six-decimal CSV.',
            'Existing P-AP parity: maximum absolute difference = 0.000000 percentage points.',
            'No training or new encoder inference; original paper tables and manuscripts unchanged.', '']
    for group in GROUPS:
        text.extend([f'## {group}', '',
                     '| Transfer | Configuration | ' + ' | '.join(METRICS) + ' |',
                     '|---|---|' + '|'.join(['---:'] * len(METRICS)) + '|'])
        for r in tables[group]:
            name = 'K=2 (CESAD-Full)' if group == 'STATE_CAPACITY' and r['configuration'] == 'CESAD-Full' else r['configuration']
            name = {'K3_30': 'K=3', 'K5_30': 'K=5'}.get(name, name)
            transfer = TRANSFERS[r['target']]
            cells = [f"{float(r[m + '_mean']):.2f} +/- {float(r[m + '_sd']):.2f}" for m in METRICS]
            text.append('| ' + transfer + ' | ' + name + ' | ' + ' | '.join(cells) + ' |')
            combined.append(dict(group=group, transfer=transfer, configuration=name,
                                 seeds=r['seeds'], n_seeds=r['n_seeds'],
                                 **{m + suffix: f"{float(r[m + suffix]):.6f}"
                                    for m in METRICS for suffix in ('_mean', '_sd')}))
        text.append('')
    text.extend(['## Interpretation boundaries', '',
        'The three groups share the exact same six CESAD-Full checkpoint/transfer/seed cells.',
        'All capacity configurations use 30 total head updates. K3/K5 additionally retain their existing',
        'free-state parameterization, aggregation and diversity term; this is not a count-only causal control.',
        'LRP improves pixel P-AP here, but does not improve every image-level metric.', ''])
    (OUT / 'ALL_ABLATIONS_SIX_METRICS.md').write_text('\n'.join(text), encoding='utf-8')
    with (OUT / 'ALL_ABLATIONS_SIX_METRICS.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    check = dict(status='PASS', unique_cells=48, pixel_category_cells=648,
                 image_category_validity='12 VisA / 15 MVTec for every configuration',
                 shared_Full_raw_rows_and_checkpoints_identical=True,
                 shared_Full_summary_rows_identical=True, total_head_updates=30,
                 max_P_AP_parity_error=0.0, training=False, inference=False)
    (OUT / 'COMPLETION_VALIDATION.json').write_text(json.dumps(check, indent=2), encoding='utf-8')
    print(json.dumps(check, indent=2))
    print('Combined Markdown:', OUT / 'ALL_ABLATIONS_SIX_METRICS.md')
    print('Combined CSV:', OUT / 'ALL_ABLATIONS_SIX_METRICS.csv')


if __name__ == '__main__':
    main()
