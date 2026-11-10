"""Read-only scientific-content and rendered-PDF checks for supplementary plates."""
from pathlib import Path
import sys
import json
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import final_unified_evaluation as ev
from experiments import plot_supplementary_qualitative_20261102 as plot
sys.path.insert(0, str(plot.QA))
import audit_pdf_text
import audit_figure_collisions


def main(out=None, expected_cases=13, figure_names=None, plotting_script=None):
    torch.set_num_threads(4)
    if out is not None:
        plot.OUT = Path(out)
    rows = ev.readcsv(plot.OUT/'CASE_MANIFEST.csv')
    assert len(rows) == expected_cases
    registry = {r['target']:r for r in ev.readjson(plot.REGISTRY) if r['method']=='CESAD'}
    stores = {}
    content = []
    for row in rows:
        with np.load(row['selected_cache']) as data:
            mask = data['mask']
            maps = data['maps74']
            native = data['native_maps']
            assert str(data['sample_id']) == row['sample_id']
            assert mask.any() and native.shape[1:] == mask.shape
            assert np.isfinite(maps).all() and np.isfinite(native).all()
            assert np.array_equal(mask, np.asarray(plot.Image.open(row['gt_path']).convert('L'))>0)
            assert ev.sha(row['image_path']) == row['image_sha256']
            assert ev.sha(row['gt_path']) == row['gt_sha256']
            for a,b in zip(maps,native):
                assert np.array_equal(ev.resize(a, mask.shape), b)
            if row['figure'].startswith('S2'):
                folder = plot.RECIP/'work/predictions'/row['target']
                source = np.load(folder/'global_local74.npy', mmap_mode='r')
                heads = ev.readjson(folder/'HEAD_ORDER.json')
                no = next(i for i,h in enumerate(heads) if h['seed']==0 and h['rho'] is None)
                yes = next(i for i,h in enumerate(heads) if h['seed']==0 and h['rho']==.1)
                pos = int(row['cache_index'])
                image = ev.readcsv(folder/'MANIFEST.csv')[pos]
                assert image['sample_id'] == row['sample_id']
                g,l = source[pos,yes]
                pair = [source[pos,no].mean(axis=0), .5*g+.5*l] if row['component']=='NTR' else [g,.5*g+.5*l]
                assert np.array_equal(maps, np.stack(pair))
                assert float(row['vmin']) == float(native.min())
                assert float(row['vmax']) == float(native.max())
                del source
            else:
                target = row['target']
                if target not in stores:
                    stores[target] = ev.Store(registry[target])
                manifest = ev.readcsv(registry[target]['manifest'])
                canon = next(i for i,m in enumerate(manifest) if m['sample_id']==row['sample_id'])
                assert np.array_equal(maps[0], stores[target][canon])
            content.append(dict(sample_id=row['sample_id'], figure=row['figure'],
                                masks_and_map_identity='PASS', native_resize_exact='PASS'))
    for family in ('S1','S3'):
        for source in ('Full VisA','Full MVTec'):
            group = [r for r in rows if r['figure']==family and r['source']==source]
            assert len({(r['vmin'],r['vmax']) for r in group}) <= 1
    results = []
    names = figure_names or ('supp_fig_s1_cross_dataset','supp_fig_s2a_ntr','supp_fig_s2b_lrp','supp_fig_s3_boundary_cases')
    if expected_cases == 48:
        from collections import Counter
        assert set(Counter(r['plate'] for r in rows).values()) == {8}
        assert len({r['plate'] for r in rows}) == 6
        assert set(Counter(r['target'] for r in rows if r['figure']=='S1').values()) == {4}
        for family in ('S2_NTR', 'S2_LRP'):
            group = [r for r in rows if r['figure']==family]
            assert len(group)==8 and len({r['sample_id'] for r in group})==8
            assert Counter(r['target'] for r in group)=={'VisA':4,'MVTec':4}
    for name in names:
        pdf = plot.OUT/f'{name}.pdf'
        text = audit_pdf_text.audit_pdf(pdf.read_bytes(), minimum_pt=5.)
        collision = audit_figure_collisions.audit_pdf(pdf)
        align = ev.readjson(plot.OUT/f'{name}.alignment.json')
        assert not text['below_minimum_count'] and text['auditable']
        assert collision['verdict']=='PASS' and not align['findings']
        results.append(dict(figure=name, pdf_text=text, collision=collision,
                            panel_alignment=align,
                            files=[plot.identity(plot.OUT/f'{name}.{ext}') for ext in ('png','pdf','svg')]))
    result = dict(status='PASS', cases=content, figures=results,
                  plotting_script=plot.identity(Path(plotting_script or plot.__file__)),
                  predictions_reused=True, model_inference=False, training=False,
                  no_score_rescaling=True, all_selected_masks_nonempty=True,
                  limitations=['Historical Table I source pairing is not retrospectively certified.',
                    'S2 is the certified 20261101 reciprocal run, not automatically an older Table IV.',
                    'Static validator TIFF warning is irrelevant to requested PNG/PDF/SVG exports.',
                    'Single-image plates do not require uncertainty bars; this is not a seed aggregate.',
                    'Sample AP used for explicit post-hoc S3 selection is not a category metric.'])
    ev.savejson(plot.OUT/'FIGURE_VERIFICATION.json', result)
    print(json.dumps(dict(status=result['status'], cases=len(content), figures=len(results)), indent=2))


if __name__=='__main__':
    main()
