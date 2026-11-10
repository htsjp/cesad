"""Existing predictions only: traceable S1/S2/S3 image plates, no model forward.

Selection is fixed before plotting: median positive GT area within named categories;
S1 macaroni2 and S3 use the lowest sample AP in predeclared VisA categories, explicitly
as post-hoc boundary examples. No pixels, scores, checkpoints or GT are modified.
"""
from pathlib import Path
import sys
import json
import gc
import math

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import final_unified_evaluation as ev

QA = Path('C:/Users/Administrator/.codex/skills/nature-figure/scripts')
sys.path.insert(0, str(QA))
from audit_panel_alignment import require_matplotlib_panel_alignment

OUT = ROOT / 'results/supplementary_qualitative_20261102'
RECIP = ROOT / 'results/ntr_lrp_reciprocal_aligned_20261101'
REGISTRY = ev.WORK / 'PREDICTION_REGISTRY.json'
CMAP = 'magma'
records, candidates, assets = [], [], []


def identity(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=ev.sha(path))


def check_asset(path, expected=None):
    item = identity(path)
    if expected is not None:
        assert item['sha256'] == expected, path
    if item not in assets:
        assets.append(item)
    return item


def positive(rows, category):
    return [r for r in rows if r['category'] == category and int(r['label']) == 1
            and int(r['positive_pixels']) > 0]


def median_area(rows, category):
    pool = positive(rows, category)
    assert pool, category
    pool.sort(key=lambda r: (int(r['positive_pixels']) /
                            (int(r['gt_width']) * int(r['gt_height'])), r['sample_id']))
    return pool[(len(pool) - 1) // 2]


def image_and_mask(row):
    check_asset(row['img_path'], row['image_sha256'])
    check_asset(row['gt_path'], row['gt_sha256'])
    image = np.asarray(Image.open(row['img_path']).convert('RGB'))
    mask = ev.native_mask(row)
    assert mask.any(), 'Empty anomaly mask cannot be used as a defect example'
    assert mask.shape == image.shape[:2], 'No implicit image/GT shape correction'
    return image, mask


def sample_ap(row, raw):
    mask = ev.native_mask(row)
    score = ev.resize(raw, mask.shape)
    assert mask.any() and not mask.all() and np.isfinite(score).all()
    return float(average_precision_score(mask.ravel(), score.ravel()) * 100)


def lowest_ap(rows, category, store):
    pool = positive(rows, category)
    values = []
    for row in pool:
        ap = sample_ap(row, store[int(row['index'])])
        values.append((ap, row['sample_id'], row))
        candidates.append(dict(target=row['dataset'], category=category,
                               sample_id=row['sample_id'], pixel_AP=ap,
                               selection='posthoc minimum sample AP within fixed category'))
    assert values
    return min(values, key=lambda x: (x[0], x[1]))[2]


def specimen(row, maps, origin, section, rule, extra=None):
    image, mask = image_and_mask(row)
    assert all(np.isfinite(m).all() and np.asarray(m).shape == (74, 74) for m in maps)
    native = [ev.resize(m, mask.shape) for m in maps]
    n = len(records)
    cache = OUT / 'selected_data' / f'{n:02d}.npz'
    np.savez_compressed(cache, mask=mask, maps74=np.stack(maps),
                        native_maps=np.stack(native), sample_id=row['sample_id'])
    item = dict(figure=section, source=origin['source'], target=row['dataset'],
                category=row['category'], sample_id=row['sample_id'],
                seed=0 if section.startswith('S2') else 'UNKNOWN (historical registry does not record seed)',
                image_path=row['img_path'], image_sha256=row['image_sha256'],
                gt_path=row['gt_path'], gt_sha256=row['gt_sha256'],
                gt_positive_pixels=int(mask.sum()), gt_shape=str(mask.shape),
                checkpoint=origin['checkpoint'], cache_path=origin['cache_path'],
                cache_index=origin['cache_index'], selection_rule=rule,
                source_version=origin['version'], prediction_only=True,
                resize='FP32 bilinear align_corners=True to original GT',
                map_shape='74x74', smoothing='NONE', sigmoid=False,
                individual_minmax=False, selected_cache=str(cache),
                sample_AP_left=sample_ap(row, maps[0]),
                sample_AP_right=sample_ap(row, maps[-1]),
                background_mean_left=float(native[0][~mask].mean()),
                background_mean_right=float(native[-1][~mask].mean()),
                anomaly_mean_left=float(native[0][mask].mean()),
                anomaly_mean_right=float(native[-1][mask].mean()), **(extra or {}))
    records.append(item)
    return dict(row=row, image=image, mask=mask, native=native, raw=maps, record=item)


def shared_limits(samples):
    # Full extrema, no percentile clipping and no per-panel min-max transformation.
    lo = min(float(q.min()) for s in samples for q in s['native'])
    hi = max(float(q.max()) for s in samples for q in s['native'])
    assert math.isfinite(lo) and math.isfinite(hi) and lo < hi
    return lo, hi


def show(ax, array, kind, norm=None):
    # Preserve native aspect ratio; a square axes is the canvas, not a stretched image.
    h, w = array.shape[:2]
    ratio = w / h
    if ratio >= 1:
        extent = (0, 1, .5 + .5 / ratio, .5 - .5 / ratio)
    else:
        extent = (.5 - .5 * ratio, .5 + .5 * ratio, 1, 0)
    kw = dict(extent=extent, origin='upper', interpolation='nearest')
    if kind == 'map':
        ax.imshow(array, cmap=CMAP, norm=norm, **kw)
    elif kind == 'mask':
        ax.imshow(array, cmap='gray', vmin=0, vmax=1, **kw)
    else:
        ax.imshow(array, **kw)
    ax.set(xlim=(0, 1), ylim=(1, 0), aspect='equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def overlay(s, norm, outline=False):
    q = s['native'][-1]
    color = plt.get_cmap(CMAP)(norm(q))[..., :3]
    blend = .60 * s['image'] / 255. + .40 * color
    if outline:
        from scipy.ndimage import binary_erosion
        edge = s['mask'] & ~binary_erosion(s['mask'], structure=np.ones((3, 3)))
        blend[edge] = 1.
    return np.clip(blend, 0, 1)


def plate(samples, name, titles, controlled=False, outline=False):
    n = len(samples)
    fig, axes = plt.subplots(n, 4, figsize=(7.2, 1.77 * n + .25), squeeze=False)
    fig.subplots_adjust(left=.16, right=.91, bottom=.025, top=1-.24/fig.get_figheight(),
                        wspace=.035, hspace=.13)
    caxes = []
    for i, s in enumerate(samples):
        lo, hi = s['limits']
        norm = Normalize(lo, hi, clip=False)
        for key, val in dict(vmin=lo, vmax=hi, clipping='NONE; full extrema',
                             cmap=CMAP, overlay_alpha=.40).items():
            s['record'][key] = val
        if controlled:
            arrays = [s['image'], s['mask'], *s['native']]
            kinds = ['image', 'mask', 'map', 'map']
        else:
            arrays = [s['image'], s['mask'], s['native'][0], overlay(s, norm, outline)]
            kinds = ['image', 'mask', 'map', 'image']
        for j, (arr, kind) in enumerate(zip(arrays, kinds)):
            show(axes[i, j], arr, kind, norm)
            if i == 0:
                axes[i, j].set_title(titles[j], fontsize=8, pad=5)
        row = s['row']
        suffix = Path(row['img_path']).stem
        tag = s.get('tag', row['dataset'])
        axes[i, 0].set_ylabel(f"{tag}\n{row['category']}\nID: {suffix}",
                              fontsize=7, labelpad=7)
    # Plot bounds are measured after equal-aspect layout, before adding excluded bars.
    fig.canvas.draw()
    for i, s in enumerate(samples):
        pos = axes[i, 3].get_position()
        ca = fig.add_axes([pos.x1+.012, pos.y0+.07*pos.height, .010, .86*pos.height])
        lo, hi = s['limits']
        bar = fig.colorbar(ScalarMappable(norm=Normalize(lo, hi), cmap=CMAP), cax=ca)
        bar.set_ticks([lo, (lo+hi)/2, hi])
        bar.ax.set_yticklabels([f'{v:.3f}' for v in [lo, (lo+hi)/2, hi]], fontsize=6)
        bar.outline.set_linewidth(.4)
        bar.ax.tick_params(length=2, width=.4, pad=2)
        caxes.append(ca)
    require_matplotlib_panel_alignment(fig, axes=list(axes.ravel()),
        json_out=OUT/f'{name}.alignment.json', tolerance_pt=1.5,
        gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(OUT/f'{name}.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(OUT/f'{name}.svg', bbox_inches='tight', facecolor='white')
    fig.savefig(OUT/f'{name}.png', dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT/f'{name}_preview.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'selected_data').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'Arial', 'font.size':8,
                         'pdf.fonttype':42, 'svg.fonttype':'none'})
    entries = [e for e in ev.readjson(REGISTRY) if e['method'] == 'CESAD']
    check_asset(REGISTRY)
    categories = dict(MVTec='grid', VisA='macaroni2', BTAD='01',
                      MPDD='bracket_black', **{'DTD-Synthetic':'Blotchy_099', 'DAGM':'Class1'})
    s1, s3 = [], []
    print('Reading Table I registered predictions, without model inference.', flush=True)
    for entry in entries:
        target = entry['target']
        check_asset(entry['manifest'], entry['manifest_sha256'])
        check_asset(entry['checkpoint_file'], entry['checkpoint_sha256'])
        check_asset(entry['cache_path'], entry['cache_sha256'])
        rows = ev.readcsv(entry['manifest'])
        store = ev.Store(entry)
        category = categories[target]
        row = (lowest_ap(rows, category, store) if target == 'VisA'
               else median_area(rows, category))
        origin = dict(source=entry['source'], checkpoint=entry['checkpoint'],
                      cache_path=entry['cache_path'],
                      cache_index=entry['canonical_to_cache_index'][int(row['index'])],
                      version='Table I historical prediction registry; not retrospective source-pair certification')
        rule = ('minimum sample AP in fixed macaroni2 category; posthoc difficult case' if target == 'VisA'
                else 'lower median positive GT area fraction within predeclared category; filename tie-break')
        s = specimen(row, [store[int(row['index'])]], origin, 'S1', rule)
        s1.append(s)
        if target == 'VisA':
            for cat in ('macaroni2', 'pcb3', 'chewinggum'):
                r = row if cat == 'macaroni2' else lowest_ap(rows, cat, store)
                o = {**origin, 'cache_index':entry['canonical_to_cache_index'][int(r['index'])]}
                s3.append(specimen(r, [store[int(r['index'])]], o, 'S3',
                    'minimum sample AP within predeclared category; explicit posthoc boundary example'))
        del store
        gc.collect()
        print('S1 selected', target, row['sample_id'], flush=True)
    for source in ('Full VisA', 'Full MVTec'):
        family = [s for s in s1 if s['record']['source'] == source]
        limits = shared_limits(family)
        for s in family:
            s['limits'] = limits
    plate(s1, 'supp_fig_s1_cross_dataset', ['Input', 'GT Mask', 'CESAD anomaly logits', 'Overlay'])
    lim = shared_limits(s3)
    for s in s3:
        s['limits'] = lim
    plate(s3, 'supp_fig_s3_boundary_cases', ['Input', 'GT Mask', 'CESAD anomaly logits', 'GT outline overlay'], outline=True)

    print('Reading certified reciprocal component predictions.', flush=True)
    s2 = []
    configpath = RECIP/'formal/LOCKED_CONFIG.json'
    cfg = ev.readjson(configpath)
    check_asset(configpath)
    assert cfg['updates'] == 30 and cfg['learning_rate'] == .001
    for target in ('VisA', 'MVTec'):
        folder = RECIP/'work/predictions'/target
        done = ev.readjson(folder/'COMPLETE.json')
        check_asset(folder/'COMPLETE.json')
        for key in ('maps', 'manifest', 'head_order'):
            check_asset(done[key]['path'], done[key]['sha256'])
        assert done['source_seal_sha256'] == ev.sha(RECIP/'formal/SOURCE_SEALED.json')
        entries = ev.readjson(folder/'HEAD_ORDER.json')
        no = next(i for i, e in enumerate(entries) if e['seed'] == 0 and e['rho'] is None)
        yes = next(i for i, e in enumerate(entries) if e['seed'] == 0 and e['rho'] == .1)
        ck = []
        for i in (no, yes):
            entry = entries[i]
            check_asset(entry['checkpoint']['path'], entry['checkpoint']['sha256'])
            state = torch.load(entry['checkpoint']['path'], map_location='cpu', weights_only=False)
            assert state['step'] == 30 and state['seed'] == 0 and state['rho'] == entry['rho']
            assert state['config_sha256'] == ev.sha(configpath)
            ck.append(state)
        assert np.array_equal(ck[0]['batch_indices'], ck[1]['batch_indices'])
        assert ck[0]['source_input_sha256'] == ck[1]['source_input_sha256']
        for key in ('m', 'alpha'):
            assert torch.equal(ck[0]['raw_model'][key], ck[1]['raw_model'][key])
        initial = [torch.load(Path(entries[i]['checkpoint']['path']).with_name('update_000.pt'),
                              map_location='cpu', weights_only=False)['raw_model'] for i in (no, yes)]
        assert all(torch.equal(initial[0][k], initial[1][k]) for k in initial[0])
        maps = np.load(folder/'global_local74.npy', mmap_mode='r')
        rows = ev.readcsv(folder/'MANIFEST.csv')
        for component, category in [('NTR', 'pcb3' if target == 'VisA' else 'grid'),
                                     ('LRP', 'macaroni2' if target == 'VisA' else 'screw')]:
            row = median_area(rows, category)
            pos = int(row['index'])
            g0, l0 = maps[pos, no]
            g1, l1 = maps[pos, yes]
            pair = [np.asarray(.5*g0+.5*l0), np.asarray(.5*g1+.5*l1)] if component == 'NTR' else [np.asarray(g1), np.asarray(.5*g1+.5*l1)]
            paths = [entries[no]['checkpoint']['path'], entries[yes]['checkpoint']['path']] if component == 'NTR' else [entries[yes]['checkpoint']['path']]*2
            origin = dict(source=entries[yes]['source'], checkpoint=' | '.join(paths),
                          cache_path=str(folder/'global_local74.npy'), cache_index=pos,
                          version='20261101 certified reciprocal rerun, seed0 trained heads; separate from historical Table IV family')
            s = specimen(row, pair, origin, f'S2_{component}',
                'lower median positive GT area fraction in fixed category; no model-result selection',
                dict(component=component, left_lrp=component=='NTR', right_lrp=True,
                     left_rho=0 if component=='NTR' else .1, right_rho=.1,
                     lambda_global=.5, shared_physical_sample=True,
                     actual_batches_identical=True, initialization_identical=True))
            s['tag'] = f'{component}: ' + ('MVTec to VisA' if target=='VisA' else 'VisA to MVTec')
            s['limits'] = shared_limits([s])
            s2.append(s)
    ntr = [s for s in s2 if s['record']['component']=='NTR']
    lrp = [s for s in s2 if s['record']['component']=='LRP']
    plate(ntr, 'supp_fig_s2a_ntr', ['Input', 'GT Mask', 'Trained CES (LRP on)', 'CES + NTR (LRP on)'], controlled=True)
    plate(lrp, 'supp_fig_s2b_lrp', ['Input', 'GT Mask', 'Global only (NTR on)', 'Global + local (NTR on)'], controlled=True)

    ev.writecsv(OUT/'CASE_MANIFEST.csv', records)
    ev.writecsv(OUT/'BOUNDARY_SELECTION_CANDIDATES.csv', candidates)
    ev.savejson(OUT/'INPUT_IDENTITIES.json', assets)
    report = '''# Supplementary qualitative figures

## Figure contract
S1 is an image plate describing localization across six targets, not a claim of universal success.
S2 is a matched component comparison. S3 explicitly shows difficult cases, not a causal analysis.
No training, encoder inference or full-benchmark metric recomputation was performed.

## S1 provenance and selection
All six rows read the exact CESAD prediction store and canonical-to-cache index from the frozen
Table I prediction registry. All input and paired original GT hashes, checkpoint-file hashes and
prediction-file hashes were checked. This reuse does not retrospectively certify source pairing
in the historical VisA model; the newly rebuilt model is NOT silently substituted.
Five examples are GT-area medians within predeclared categories. The macaroni2 row is the lowest
sample-AP example in that category, deliberately retaining a difficult case. Selection is post-hoc
illustrative, not additional quantitative generalization evidence. All anomalous masks are nonempty,
including BTAD. Category choices are explicit in the script and not claimed to be a random survey.

## S2 provenance and controls
Use the completed 20261101 reciprocal rerun, not initialization fallback. Both NTR arms have 30
updates, identical physical source input, actual batches, initial raw parameters, m and alpha.
No-NTR rho is null (loss disabled); enabled rho is 0.10. Both NTR maps use fixed 2x2 LRP, 0.5/0.5.
The LRP comparison fixes the trained seed0 rho=0.10 head and contrasts its global74 with its
0.5 global74 + 0.5 local74. Both originate from the same saved feature/views and row.
This is the newly certified reciprocal component family. Do not assert exact historical Table IV
checkpoint identity if that table still contains earlier repaired-pool runs. The numerical table
version must be confirmed by the author; this figure identifies its actual experiment explicitly.

## Color and image integrity
No sigmoid, score offset, smoothing, individual map normalization, percentile clipping or GT crop.
Original 74x74 maps are FP32-bilinearly resized with align_corners=True to the original paired GT.
S1 shares a single full-extrema raw-logit scale across all five displayed VisA-source examples;
the MVTec-source VisA row uses its own scale. S3 shares a single source-family scale. S2 shares
one full-extrema scale between both arms of each row. Each row displays numerical limits.
Color saturation is not comparable across different scales. Overlay alpha is fixed at 0.40.
The image, mask and score retain their native aspect ratios, with white letterboxing only.
Display resizing does not change exported native scores. S3 white contours are GT outlines only.
Full paths, IDs, score scales, precision, selection rules and selected native maps are retained.

## Existing Fig. 6 audit boundary
The available selected-three plotting chain calls build_maps/load_specs, which supplies both
trained CES and CES+NTR to lrp_map using the same local feature cache. The previous split-figure
report also says fixed LRP is enabled for both qualitative columns. Those columns are therefore
not a global-vs-LRP comparison. However, historical plotting weights and preparation family differ
from this new certified rerun. No Word or main Fig. 6 was modified, and the actual current Word
caption in CESAD_TIM_Chapter3_No_E1E2_20261027.docx was read without modifying it: it lists
tile/wood/metal_nut and CES without/with NTR, but does not state the shared LRP setting. The
embedded-image identity was not certified here. Add the shared-LRP qualification only after
matching the final embedded image to the plotting artifact. Do not reuse a fallback script.

## Captions
Supplementary Fig. S1. Cross-dataset anomaly localization examples from six industrial benchmarks.
Each example shows the input image, released ground-truth annotation, CESAD anomaly-logit map,
and image overlay. Predictions are read from the source-matched fixed Table I prediction caches.
Color limits are shared within each displayed source-model family and are printed per row.
The VisA macaroni2 example is a post-hoc difficult case; other examples use a fixed GT-area rule.

Supplementary Fig. S2. Controlled visualization of NTR and LRP under reciprocal transfers.
(a) Trained no-NTR and NTR heads with fixed LRP. (b) Global and global-plus-local predictions
of the same trained NTR head. Source preparation, initialization, training budget, sample and
color limits are matched within each comparison; only the indicated component is varied.
Seed0 checkpoints and maps are from the certified 20261101 reciprocal rerun.

Supplementary Fig. S3. Post-hoc boundary examples from MVTec to VisA. The minimum sample-AP
case within each predeclared category is shown. These cases illustrate localization limitations;
they do not establish a failure mechanism or a production false-positive rate.
'''
    (OUT/'FIGURE_PROVENANCE_AND_CAPTIONS.md').write_text(report, encoding='utf-8')
    print('Saved', OUT, 'cases=', len(records), flush=True)


if __name__ == '__main__':
    main()
