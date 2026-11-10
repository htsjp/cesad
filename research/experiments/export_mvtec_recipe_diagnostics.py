"""Export saved-head diagnostics; no fitting, target inference or target evaluation."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/phase_mvtec_exact_visa_recipe_checked/diagnostics_20261029'
BASE = Path('E:/MRAD_mvtec_exact_visa_recipe_checked/seed0')
ASSETS = ROOT / 'results/phase_mvtec_exact_visa_recipe_checked/formal/source_assets'
PATHS = {
    'Formal_MVTec': Path('E:/MRAD_component_original_cache_repair/20261027/checkpoints/Full_MVTec_CES_seed0_CES_NTR_lr0.001.pt'),
    'Nearest_Control': BASE / 'Original_MVTec/checkpoints/Full_MVTec_CES_seed0_Original_MVTec_lr0.001.pt',
    'Full_Recipe': BASE / 'Exact_VisA_Recipe/checkpoints/Full_MVTec_CES_seed0_Exact_VisA_Recipe_lr0.001.pt',
}


def unit(x):
    x = np.asarray(x, np.float64)
    return x / max(np.linalg.norm(x), 1e-12)


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(unit(a) @ unit(b), -1, 1))))


def states(m, alpha, v):
    return unit(m - alpha * unit(v)), unit(m + alpha * unit(v))


def ident(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def writecsv(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    geometry, probes, weights, inputs = [], [], {}, []
    for name, path in PATHS.items():
        cp = torch.load(path, map_location='cpu', weights_only=False)
        inputs.append(ident(path))
        g = cp['initial_geometry']
        m, alpha, v0 = np.asarray(g['m'], np.float64), float(g['alpha']), g['init_v']
        vf = cp['params']['v']
        n0, a0 = states(m, alpha, v0)
        nf, af = states(m, alpha, vf)
        wi, wf = a0 - n0, af - nf
        weights[name] = (wi, wf)
        geometry.append(dict(model=name, checkpoint=str(path), seed=cp['seed'],
            m_norm=float(np.linalg.norm(m)), alpha=alpha, m_dot_v0=float(m @ unit(v0)),
            initial_normal_state_norm=float(np.linalg.norm(n0)),
            initial_anomaly_state_norm=float(np.linalg.norm(a0)),
            initial_effective_weight_norm=float(np.linalg.norm(wi)),
            final_effective_weight_norm=float(np.linalg.norm(wf)),
            own_state_weight_rotation_deg=angle(wi, wf), raw_v_rotation_deg=angle(v0, vf),
            updates=cp['realized_updates'], batch_hash=cp['batch_stream_hash']))
        initial = cp['initial_probe']
        for step in (0, 1, 10, 20, 24, 30):
            row = initial if step == 0 else cp['logs'][step - 1]
            probes.append(dict(model=name, checkpoint=str(path), step=step,
                source='Full MVTec', split='fixed saved training-pool probe; not held-out validation',
                probe_loss=row['probe_loss'], probe_bce=row['probe_bce'], probe_dice=row['probe_dice'],
                probe_ntr=row['probe_ntr'], probe_mean_normal=row['probe_mean_normal'],
                probe_mean_anomaly=row['probe_mean_anomaly'], grad_norm=row.get('grad_norm'),
                last6_relative_loss_drop=cp['recent_relative_source_loss_drop'],
                saved_diagnostic_trigger=cp['diagnostic_trigger']))
    ppath = ASSETS / 'q_pc1_statealigned1024_formal.npy'
    apath = ASSETS / 'E2_PC1Removed-CorruptedMask_formal.pt'
    inputs += [ident(ppath), ident(apath), ident(Path(__file__))]
    p = np.load(ppath).astype(np.float64)
    adapter = torch.load(apath, map_location='cpu', weights_only=False)['model']
    A, B = adapter['a.weight'].numpy().astype(np.float64), adapter['b.weight'].numpy().astype(np.float64)

    def raw_numerator(w):
        t = w + A.T @ (B.T @ w)
        return t - p * (p @ t)

    wi, wf = weights['Nearest_Control']
    ti, tf = [raw_numerator(w) for w in weights['Full_Recipe']]
    angles = [dict(comparison=label, angle_degrees=value, space='common raw normalized CLIP feature space; score numerator only')
              for label, value in (
                  ('Control_vs_Recipe_initial', angle(wi, ti)),
                  ('Control_vs_Recipe_final', angle(wf, tf)),
                  ('Control_initial_vs_final', angle(wi, wf)),
                  ('Recipe_initial_vs_final', angle(ti, tf)))]
    folder = BASE / 'Original_MVTec/source_pools/Full_MVTec_seed0'
    raw = np.load(folder / 'features.npy', mmap_mode='r').astype(np.float64)
    labels = np.load(folder / 'labels.npy')
    transformed = np.load(BASE / 'Exact_VisA_Recipe/source_pools/Full_MVTec_seed0/features.npy', mmap_mode='r')
    x = raw - np.outer(raw @ p, p)
    h = x + (x @ A.T) @ B.T
    den = np.linalg.norm(h, axis=1)
    reconstructed = raw @ tf / den
    reference = transformed @ weights['Full_Recipe'][1]
    error = float(np.max(abs(reconstructed - reference)))
    assert error < 1e-6
    extra = dict(raw_scoring_reconstruction_max_error=error,
        denominator_quantiles=dict(zip(('min', 'p05', 'median', 'p95', 'max'), np.quantile(den, [0, .05, .5, .95, 1]).tolist())),
        denominator_normal_mean=float(den[labels == 0].mean()),
        denominator_anomaly_mean=float(den[labels == 1].mean()),
        relative_loss_drop_units='fraction; multiply by100 for percent',
        training=False, target_inference=False, target_metric_recomputation=False)
    writecsv('INITIALIZATION_GEOMETRY.csv', geometry)
    writecsv('RAW_SPACE_DIRECTION_ANGLES.csv', angles)
    writecsv('SOURCE_PROBE_TRACE.csv', probes)
    for fp in (folder / 'COMPLETE.json', folder / 'PATCH_MANIFEST.csv', folder / 'probe_indices.npy',
               BASE / 'Exact_VisA_Recipe/source_pools/Full_MVTec_seed0/COMPLETE.json'):
        inputs.append(ident(fp))
    evidence = dict(inputs=inputs, geometry=geometry, raw_space_direction_angles=angles,
                    source_probe_trace=probes, checks=extra)
    (OUT / 'DIAGNOSTICS.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    (OUT / 'README.md').write_text(
        '# Saved-head MVTec recipe diagnostics\n\n'
        'These diagnostics were first computed interactively in the chat and are now exported with provenance. '
        'They are new analysis artifacts, not previously existing formal experiment reports.\n\n'
        'INITIALIZATION_GEOMETRY.csv reconstructs normalized states from saved m/alpha/v. '
        'The reported 0.363 and1.132 refer to norm(sA-sN), NOT to the norm of either state (both are1).\n\n'
        'SOURCE_PROBE_TRACE.csv copies checkpoint initial_probe and logs fields; it does not rerun optimization. '
        'The probe comes from the saved training pool, not independent source validation. '
        'Last-six relative loss drops are fractions (.000688 = .0688%). '
        'Gradients and saved heuristic triggers do not prove convergence or insufficient optimization.\n\n'
        'For the adapter, A is4x1024 and B is1024x4. Put P=I-pp^T and R=I+BA. '
        'For column raw feature z, transformed scoring is q(z)=z^T P R^T w / norm(R P z). '
        'Intermediate L2 normalization cancels because the residual adapter is bias-free and linear. '
        'RAW_SPACE_DIRECTION_ANGLES.csv compares P R^T w with the untransformed control weight. '
        'It is not a comparison of raw v parameters in different feature spaces. '
        'Angles describe score numerators; the patch-dependent denominator also affects ranking. '
        'DIAGNOSTICS.json includes a source-cache reconstruction check and input SHA256 hashes.\n\n'
        'The exact-recipe/control arms share physical patch IDs and actual batches. '
        'The formal bilinear-label arm is provided for context, not a mask-matched intervention. '
        'Target category AP values remain in the original VisA_PER_CATEGORY.csv files under each recipe arm. '
        'No target metrics, formal tables, original checkpoints or source pools were changed.\n\n'
        'Reproduce: D:/Anaconda/python.exe experiments/export_mvtec_recipe_diagnostics.py\n', encoding='utf-8')
    print(json.dumps(dict(output=str(OUT), angles=angles, checks=extra), indent=2))


if __name__ == '__main__':
    main()
