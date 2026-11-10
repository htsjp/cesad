"""Paired source reference-center control; isolated outputs, no upstream fitting.

Both centers share the frozen E2 feature transform and the saved Step-B physical
patch/batch streams. Targets are opened only after all six heads are sealed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
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
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import phase_ces_lhc_paired_repeat as paired
from experiments import final_unified_evaluation as official

OUT = ROOT / 'results/phase_ces_geometry_center_control/formal'
OLD = ROOT / 'results/phase_ces_lhc_paired_repeat/formal'
VARIANTS = ('Hybrid_CES', 'E1_Center_CES')
SEEDS = (0, 1, 2)
TARGETS = ('MVTec', 'BTAD', 'MPDD', 'DTD-Synthetic', 'DAGM')
SOURCE = 'Full VisA'
DEVICE = 'cuda'


def js(path, value):
    paired.write_json(path, value)


def csv(path, rows):
    paired.write_csv(path, rows)


def md(name, text):
    (OUT / name).write_text(text.strip() + '\n', encoding='utf-8')


def say(phase, **kw):
    paired.status(OUT, phase, **kw)


def identity(p):
    p = Path(p)
    return dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=paired.file_hash(p))


def lock(path, data):
    data = json.loads(json.dumps(data, default=str))
    if path.exists():
        assert official.readjson(path) == data, f'Locked identity changed: {path}'
    else:
        js(path, data)


def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), 'CUDA required; no silent CPU substitution'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def geometry():
    folder = paired.PAYLOAD
    e1 = torch.load(folder / 'E1_PC1Removed-TrueMask_formal.pt', map_location='cpu', weights_only=False)
    e2 = torch.load(folder / 'E2_PC1Removed-CorruptedMask_formal.pt', map_location='cpu', weights_only=False)
    st = np.asarray(e1['patch_states'], np.float32)
    sc = np.asarray(e2['patch_states'], np.float32)
    assert st.shape == sc.shape == (2, 1024)
    for p in (e1, e2):
        assert sum(r.get('stage') == 'state' for r in p['curves']) == 20
        assert sum(r.get('stage') == 'adapter' for r in p['curves']) == 10
    transform = paired.load_transform(SOURCE, DEVICE)
    d = np.float32(.5) * (st[1] - st[0])
    alpha = float(np.linalg.norm(d))
    v0 = paired.hist.unit_np(d)
    centers = dict(Hybrid_CES=np.float32(.5) * (sc[0] + sc[1]),
                   E1_Center_CES=np.float32(.5) * (st[0] + st[1]))
    assert np.array_equal(centers['Hybrid_CES'], transform['m'])
    assert np.array_equal(v0, transform['init_v']) and alpha == transform['alpha']
    return transform, centers, v0, alpha


def pool(seed):
    directory = OLD / 'source_pools' / f'Full_VisA_seed{seed}'
    meta = official.readjson(directory / 'COMPLETE.json')
    z = np.load(directory / 'features.npy', mmap_mode='r')
    y = np.load(directory / 'labels.npy')
    batches = np.load(directory / 'batch_indices.npy')
    probe = np.load(directory / 'probe_indices.npy')
    for key, a in [('feature_array_hash', z), ('labels_hash', y),
                   ('batch_stream_hash', batches), ('probe_indices_hash', probe)]:
        assert paired.digest(a) == meta[key], key
    assert paired.file_hash(directory / 'PATCH_MANIFEST.csv') == meta['manifest_hash']
    assert batches.shape == (30, 8184) and meta['epochs'] == 10
    assert len(z) == len(y) == 17415 and set(np.unique(y)) == {0, 1}
    return directory, z, y, batches, probe, meta


def global_inputs(target):
    """Return mmap plus physical image order from its immutable producer."""
    if target == 'MVTec':
        _, ds, tr, held = paired.light_fold('D1', 0)
        p = paired.hist.LOCAL_CACHE_DIR / 'global_patch_MVTec.npy'
        paths = [str(Path(ds.root) / ds.data_all[i]['img_path']) for i in tr + held]
    elif target == 'BTAD':
        p0 = ROOT / 'results/phase_xlv_cross_domain_transfer/feature_cache/BTAD_clip.npz'
        with np.load(p0, allow_pickle=True) as pack:
            paths = [str(r['img_path']) for r in pack['rows']]
        candidates = list((ROOT / 'results/phase_memory_support_modes/array_cache').glob('BTAD_clip_*_patch.npy'))
        assert len(candidates) == 1
        p = candidates[0]
    elif target == 'DAGM':
        p = ROOT / 'results/paper_assets/dagm_visa_eval/cache/dagm_global_patch_float16.npy'
        rows = official.readcsv(ROOT / 'results/paper_fig5_dagm_capacity/formal/IMAGE_MANIFEST.csv')
        paths = [r['path'] for r in rows]
    else:
        archive = ROOT / f'results/phase_median_spatial_consensus_generalization/formal/feature_cache/{target}_clip.npz'
        with np.load(archive, allow_pickle=True) as pack:
            paths = [str(r['img_path']) for r in pack['rows']]
        if target == 'MPDD':
            p = OLD.parent / 'shared_input_cache/MPDD_global_patch.npy'
            assert p.is_file()
        else:
            p = OUT / 'shared_input_cache/DTD_global_patch.npy'
            p.parent.mkdir(parents=True, exist_ok=True)
            paired.extract_existing_npz(archive, 'patch', p)
    a = np.load(p, mmap_mode='r')
    assert a.shape == (len(paths), 1369, 1024)
    return a, paths, p


def local_path(target):
    if target == 'MPDD':
        return Path('E:/MRAD_final_evidence_cache/local_patch_Full_MVTec__MPDD.npy')
    return paired.hist.local_cache_path(SOURCE, target)


def preflight():
    setup()
    transform, centers, v0, alpha = geometry()
    assets = [paired.PAYLOAD / n for n in ('E1_PC1Removed-TrueMask_formal.pt',
              'E2_PC1Removed-CorruptedMask_formal.pt', 'q_pc1_statealigned1024.npy')]
    scripts = [Path(__file__), Path(paired.__file__), Path(paired.hist.__file__),
               Path(official.__file__), ROOT / 'phase_xa/official_task_forward.py',
               ROOT / 'experiments/phase_statealigned_pc1_debias.py',
               ROOT / 'experiments/phase_statealigned_state_alignment_swap.py',
               ROOT / 'experiments/phase_xlii_patch_specialized_adaptation.py']
    sources = []
    for seed in SEEDS:
        directory, z, y, batches, probe, meta = pool(seed)
        assert np.array_equal(np.asarray(meta['init']['m'], np.float32), centers['Hybrid_CES'])
        assert np.array_equal(np.asarray(meta['init']['init_v'], np.float32), v0)
        assert meta['init']['alpha'] == alpha
        sources.append(dict(seed=seed, directory=str(directory), **{k: meta[k] for k in (
            'feature_array_hash', 'labels_hash', 'batch_stream_hash', 'probe_indices_hash', 'manifest_hash')}))
        assets.extend(directory / n for n in ('COMPLETE.json', 'features.npy', 'labels.npy',
                      'batch_indices.npy', 'probe_indices.npy', 'PATCH_MANIFEST.csv', 'SOURCE_IMAGE_MANIFEST.csv'))
    # Source-only checks precede training; target files here are existence/identity,
    # not measured target performance or hyperparameter selection.
    targets = []
    for target in TARGETS:
        mp = official.WORK / 'manifests' / f'{target}.csv'
        assert mp.is_file(), mp
        rows = official.readcsv(mp)
        assert all(Path(r['img_path']).is_file() and (not int(r['label']) or Path(r['gt_path']).is_file()) for r in rows)
        assert all(not int(r['label']) or official.normpath(r['gt_path']) != official.normpath(r['img_path']) for r in rows)
        lp = local_path(target)
        if lp.exists():
            a = np.load(lp, mmap_mode='r')
            assert a.shape == (len(rows), 4, 1369, 1024)
            assets.append(lp)
        targets.append(dict(target=target, manifest=str(mp), images=len(rows),
                            categories=len({r['category'] for r in rows}),
                            local_cache=str(lp) if lp.exists() else None,
                            local_mode='cached frozen features' if lp.exists() else 'necessary frozen CLIP extraction, shared across six heads'))
        assets.append(mp)
    say('HASHING_INPUTS', files=len(assets) + len(scripts))
    config = dict(phase='Effect of Source-Derived Reference Center', source=SOURCE,
                  seeds=list(SEEDS), variants=list(VARIANTS), source_inputs=sources,
                  centers={k: v.tolist() for k, v in centers.items()}, v0=v0.tolist(), alpha=alpha,
                  intended_difference='pre-normalization reference center m ONLY; initial logits need not match',
                  states='unit(m +/- alpha*unit(raw_v)); q=z@(sA-sN); tau=1; bias=0',
                  trainable='raw_v, 1024 FP32 scalars only', optimizer=dict(name='AdamW', lr=.001,
                  betas=[.9, .999], eps=1e-8, weight_decay=0., amsgrad=False),
                  training=dict(epochs=10, updates=30, batch=8184, loss='balanced BCE + Dice + NTR',
                  rho=.10, ntr_weight=1., scheduler=None, warmup=None, autocast=False,
                  dtype='FP32; TF32 off', checkpoint_selection='LAST_UPDATE; all six sealed before target scores'),
                  transform='shared normalized CLIP/DAPM24 final patch -> PC1 removal -> L2 -> frozen E2 rank4 adapter -> L2',
                  labels='saved StepB true source loader masks; bilinear37 align_corners=False >.5; NOT shifted E2 labels',
                  targets=targets, lrp='nonoverlap2x2;518crop;TL/TR/BL/BR;global37->74 bilinear align_corners=True;0.5global+0.5local;no smoothing',
                  evaluation=dict(policy=official.POLICY, evaluator_hash=official.evaluator_hash(),
                  primary='native paired GT; equal valid-category macro P-AP; exact score ties grouped; no target selection'),
                  input_assets=[identity(p) for p in dict.fromkeys(assets)], code=[identity(p) for p in scripts],
                  software=dict(torch=torch.__version__, numpy=np.__version__, python=sys.version,
                                GPU=torch.cuda.get_device_name(0)),
                  outcome_rule='Descriptive paired seed-level evidence, not a significance claim: B if abs(mean delta)<0.2 or seed signs conflict; A/C if consistent seed signs and abs(mean delta)>=0.2. Tie per cell abs(delta)<=0.2 percentage points. Locked before evaluation.')
    lock(OUT / 'GEOMETRY_CONTROL_CONFIG.json', config)
    md('PRE_RUN_VALIDATION.md', f"""# Pre-run validation

PASS: E1/E2 patch_states are the post-Stage-I pairs (20 saved state entries), not initial class means. Each payload has 10 Stage-II adapter epochs. E2 adapter alone is deployed in BOTH variants. No upstream refit is allowed.

Only intended intervention: m_Hybrid=(E2_N+E2_A)/2 versus m_E1=(E1_N+E1_A)/2. Shared alpha={alpha:.10f}, v0 and fixed E2 transform. Each seed shares the SAME existing physical source pool and exact 30x8184 batch array. Both heads are newly trained with BCE+Dice+NTR, real masks, AdamW1e-3/zero WD,10 epochs, last update. Initial logits are not constrained equal.

Source/code trace: phase_ces_lhc_paired_repeat.py::load_transform, transformed, objective, train_one; phase_ces_linear_head_control.py::CESHead; phase_statealigned_pc1_debias.py::train_pc1_removed_statealigned. Geometry formulas above match actual payloads. Source pool labels are final true loader labels, not auxiliary shifted masks.

Target protocol: final_unified_evaluation.py::native_mask, resize, weighted_pixel_metrics, macro_metrics. Use existing hashed canonical manifests and original paired masks. AP uses sklearn average_precision_score with exact ties; macro is equal-weight valid categories. No historical loader518 target GT is used. No target-based LR/model selection. All six checkpoints must be sealed before target scoring.

Hashes, target counts and provenance are in GEOMETRY_CONTROL_CONFIG.json. Numerical initialization and full source-feature parity tests run next. No original checkpoint, table, StepB pool, figure or manuscript is a writable output. New targets lacking local feature caches require only frozen extraction shared across all six heads.

Three seeds are source head training repeats, not fifteen independent full experiments. Descriptive tie tolerance0.2 P-AP points and outcome rule are locked in config before target evaluation, not used to choose a model or tune any parameter.
""")
    say('STATIC_PREFLIGHT_PASSED', config_hash=paired.json_hash(config), targets=targets)
    return config


def init_module(variant, centers, v0, alpha):
    return paired.hist.CESHead(centers[variant], alpha, v0).to(DEVICE)


def numerical_validation(config):
    transform, centers, v0, alpha = geometry()
    arrays, _ = paired.source_arrays('D2')
    _, ds, tr, held = paired.source_dataset(SOURCE)
    indices = tr + held
    checks = []
    for seed in SEEDS:
        directory, z, y, batches, probe, meta = pool(seed)
        manifest = official.readcsv(directory / 'PATCH_MANIFEST.csv')
        assert len(manifest) == len(z)
        reconstructed = np.empty(z.shape, np.float32)
        error = 0.
        for pos in sorted({int(r['source_position']) for r in manifest}):
            ids = [i for i, r in enumerate(manifest) if int(r['source_position']) == pos]
            patch_ids = [int(manifest[i]['patch_flat_index']) for i in ids]
            expected_labels = paired.mask_only(ds, indices[pos])
            row = ds.data_all[indices[pos]]
            for i, pid in zip(ids, patch_ids):
                assert int(manifest[i]['label']) == y[i] == expected_labels[pid]
                assert official.normpath(manifest[i]['image']) == official.normpath(Path(ds.root) / row['img_path'])
            raw = arrays[0][pos] if pos < len(tr) else arrays[1][pos-len(tr)]
            with torch.no_grad():
                reconstructed[ids] = paired.transformed(raw[patch_ids], transform, DEVICE).cpu().numpy()
        error = float(np.max(np.abs(reconstructed - z)))
        assert error < 2e-6, ('Saved transformed source pool differs from exact frozen transform', error)
        initial_logits = []
        for variant in VARIANTS:
            torch.manual_seed(seed)
            model = init_module(variant, centers, v0, alpha)
            assert sum(p.numel() for p in model.parameters()) == 1024
            with torch.no_grad():
                q = model(torch.tensor(np.asarray(z), device=DEVICE)).cpu().numpy()
            initial_logits.append(q)
            dest = OUT / 'initialization' / f'{variant}_seed{seed}'
            dest.mkdir(parents=True, exist_ok=True)
            np.save(dest / 'source_logits.npy', q)
            np.save(dest / 'raw_v0.npy', model.v.detach().cpu().numpy())
            spec = model.to_spec(SOURCE)
            w = spec.weight()
            probe_stats = paired.source_probe(model, z, y, probe, DEVICE)
            checks.append(dict(seed=seed, variant=variant, source_transform_max_abs_error=error,
                feature_hash=meta['feature_array_hash'], label_hash=meta['labels_hash'],
                batch_hash=meta['batch_stream_hash'], probe_hash=meta['probe_indices_hash'],
                alpha=alpha, v0_hash=paired.digest(v0), m_norm=float(np.linalg.norm(centers[variant])),
                m_dot_v0=float(centers[variant] @ v0), state_distance=float(np.linalg.norm(w)),
                effective_weight_norm=float(np.linalg.norm(w)), logits_hash=paired.digest(q),
                logit_min=float(q.min()), logit_max=float(q.max()), **probe_stats))
        checks[-1]['initial_logits_max_abs_difference_from_Hybrid'] = float(abs(initial_logits[1]-initial_logits[0]).max())
    csv(OUT / 'INITIALIZATION_NUMERICAL_CHECK.csv', checks)
    js(OUT / 'NUMERICAL_VALIDATION.json', dict(status='PASS', config_hash=paired.json_hash(config), rows=checks))
    say('NUMERICAL_VALIDATION_PASSED', seeds=3, same_physical_patches=True, same_batches=True)


def train(config):
    assert official.readjson(OUT / 'NUMERICAL_VALIDATION.json')['config_hash'] == paired.json_hash(config)
    _, centers, v0, alpha = geometry()
    entries = []
    training = []
    for seed in SEEDS:
        _, z, y, batches, probe, meta = pool(seed)
        for variant in VARIANTS:
            cp = OUT / 'checkpoints' / f'{variant}_seed{seed}.pt'
            if cp.exists():
                saved = torch.load(cp, map_location='cpu', weights_only=False)
                assert saved['config_hash'] == paired.json_hash(config)
                assert saved['batch_stream_hash'] == meta['batch_stream_hash'] and saved['realized_updates'] == 30
            else:
                torch.manual_seed(seed)
                model = init_module(variant, centers, v0, alpha)
                optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0., betas=(.9,.999), eps=1e-8)
                initial = paired.source_probe(model, z, y, probe, DEVICE)
                snapshots = [model.v.detach().cpu().clone()]
                logs = []
                for step, ids in enumerate(batches):
                    x = torch.tensor(np.asarray(z[ids]), device=DEVICE)
                    labels = torch.tensor(y[ids], device=DEVICE)
                    loss, bce, dice, ntr = paired.objective(model(x), labels)
                    assert torch.isfinite(loss), 'NaN loss; stop, no model substitution'
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad = float(model.v.grad.norm())
                    assert math.isfinite(grad)
                    optimizer.step()
                    assert torch.isfinite(model.v).all()
                    assert torch.equal(model.m.cpu(), torch.from_numpy(centers[variant]))
                    assert float(model.alpha) == float(np.float32(alpha))
                    counter = int(optimizer.state[model.v]['step'])
                    assert counter == step+1
                    spec = model.to_spec(SOURCE)
                    logs.append(dict(source=SOURCE, variant=variant, seed=seed, step=step+1,
                        epoch=step//3+1, batch_count=len(ids), batch_ids_hash=paired.digest(ids),
                        loss=float(loss), bce=float(bce), dice=float(dice), ntr=float(ntr),
                        grad_norm=grad, raw_v_norm=float(model.v.norm()), optimizer_step=counter,
                        effective_weight_norm=float(np.linalg.norm(spec.weight())),
                        **paired.source_probe(model,z,y,probe,DEVICE)))
                    snapshots.append(model.v.detach().cpu().clone())
                saved = dict(variant=variant, source=SOURCE, seed=seed, method='CES',
                    params=spec.params, raw_model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    raw_v_trajectory=torch.stack(snapshots), optimizer_state=optimizer.state_dict(),
                    initial_probe=initial, final_probe=logs[-1], logs=logs, realized_updates=30,
                    config_hash=paired.json_hash(config), feature_array_hash=meta['feature_array_hash'],
                    labels_hash=meta['labels_hash'], batch_stream_hash=meta['batch_stream_hash'],
                    probe_indices_hash=meta['probe_indices_hash'], checkpoint_selection='LAST_UPDATE_NO_TARGET_ACCESS')
                cp.parent.mkdir(parents=True,exist_ok=True)
                torch.save(saved, cp)
                csv(OUT / 'training_logs' / f'{variant}_seed{seed}.csv', logs)
            assert len(saved['raw_v_trajectory']) == 31
            assert [r['batch_ids_hash'] for r in saved['logs']] == [paired.digest(ids) for ids in batches]
            training.extend(saved['logs'])
            entries.append(dict(model_id=f'{variant}_seed{seed}',variant=variant,seed=seed,
                                checkpoint=str(cp),checkpoint_sha256=paired.file_hash(cp)))
            say('HEAD_TRAINED',variant=variant,seed=seed,updates=30,final_source_probe=saved['logs'][-1]['probe_loss'])
    csv(OUT / 'TRAINING_STEPS.csv', training)
    lock(OUT / 'SOURCE_TRAINING_COMPLETE.json',dict(config_hash=paired.json_hash(config), models=entries,
         target_labels_used_for_training=False, checkpoint_selection='last step30', planned_updates=30))
    return entries


class FrozenEncoder:
    def __init__(self):
        from experiments.phase_x0a_mandatory_gates import load_components
        from experiments.phase_xlv_cross_domain_transfer import CHECKPOINT
        from utils.transforms import get_transform
        self.model, *unused = load_components(str(CHECKPOINT), DEVICE)
        del unused
        self.model.eval()
        assert all(not p.requires_grad for p in self.model.parameters())
        self.preprocess = get_transform(argparse.Namespace(image_size=518))[0]
        self.checkpoint = Path(CHECKPOINT)
        self.before = self.checksum()

    def checksum(self):
        h = hashlib.sha256()
        for name, p in self.model.visual.state_dict().items():
            h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    @torch.inference_mode()
    def extract(self, path, local=True):
        from phase_xa.official_task_forward import normalized_patch_feature, visual_outputs
        from PIL import Image
        if local:
            images = paired.lzp.quadrant_crops(path)
        else:
            with Image.open(path) as im:images = [im.convert('RGB')]
        x = torch.stack([self.preprocess(im) for im in images]).to(DEVICE)
        z = normalized_patch_feature(visual_outputs(self.model,x,adapter=None,grad=False)).float().cpu().numpy()
        assert z.shape == (4 if local else 1,1369,1024) and np.isfinite(z).all()
        return z if local else z[0]


def target_cell(target):
    canonical = official.readcsv(official.WORK / 'manifests' / f'{target}.csv')
    a, paths, gp = global_inputs(target)
    canonical_paths = official.canonical_paths(target, paths, canonical)
    lookup = {official.normpath(p): i for i,p in enumerate(canonical_paths)}
    assert len(lookup) == len(canonical) == len(a)
    mapping = [lookup[official.normpath(r['img_path'])] for r in canonical]
    lp = local_path(target)
    local = np.load(lp,mmap_mode='r') if lp.exists() else None
    if local is not None:assert local.shape == (len(a),4,1369,1024)
    out = OUT / 'predictions' / target
    out.mkdir(parents=True,exist_ok=True)
    csv(out / 'TARGET_MANIFEST.csv',[dict(r,raw_feature_index=mapping[i]) for i,r in enumerate(canonical)])
    return canonical,a,local,mapping,gp,lp,out


def weights(entries):
    columns=[]
    for entry in entries:
        assert paired.file_hash(entry['checkpoint']) == entry['checkpoint_sha256']
        saved=torch.load(entry['checkpoint'],map_location='cpu',weights_only=False)
        columns.append(paired.hist.HeadSpec('CES',saved['params'],SOURCE,entry['seed']).weight())
    return torch.tensor(np.stack(columns,axis=1),device=DEVICE)


@torch.inference_mode()
def fused_maps(raw, transform, w):
    z=paired.transformed(raw,transform,DEVICE)
    q=(z@w).view(5,37,37,len(w[0])).permute(3,0,1,2)
    g=F.interpolate(q[:,0:1],size=(74,74),mode='bilinear',align_corners=True)[:,0]
    local=torch.empty_like(g)
    local[:,:37,:37]=q[:,1];local[:,:37,37:]=q[:,2]
    local[:,37:,:37]=q[:,3];local[:,37:,37:]=q[:,4]
    return (.5*g+.5*local).cpu().numpy()


def smoke(config,entries):
    transform,*_=geometry()
    w=weights(entries)
    encoder=FrozenEncoder()
    checks=[]
    for target in TARGETS:
        rows,a,local,mapping,gp,lp,dest=target_cell(target)
        ids=[next(i for i,r in enumerate(rows) if int(r['label'])==lab) for lab in (0,1)]
        for i in ids:
            ix=mapping[i];row=rows[i]
            freshg=encoder.extract(row['img_path'],False)
            freshl=encoder.extract(row['img_path'])
            cg=np.asarray(a[ix],np.float32)
            gd=np.abs(freshg-cg)
            assert gd.mean()<2e-4 and gd.max()<.005,(target,'global frozen cache mismatch',gd.mean(),gd.max())
            ld=None
            if local is not None:
                cl=np.asarray(local[ix],np.float32);ld=np.abs(freshl-cl)
                assert ld.mean()<2e-4 and ld.max()<.005,(target,'local cache mismatch',ld.mean(),ld.max())
            else:cl=freshl
            raw=np.concatenate((cg[None],cl))
            maps=fused_maps(raw,transform,w)
            assert maps.shape==(6,74,74) and np.isfinite(maps).all()
            # Exact historical feature/scorer/fusion route, but not historical GT.
            saved=torch.load(entries[0]['checkpoint'],map_location='cpu',weights_only=False)
            spec=paired.hist.HeadSpec('CES',saved['params'],SOURCE,entries[0]['seed'])
            ref=paired.hist.lrp_map(transform,spec,cg,cl,DEVICE)
            err=float(abs(maps[0]-ref).max());assert err<2e-6,err
            mask=official.native_mask(row)
            resized=official.resize(maps[0],mask.shape)
            assert resized.shape==mask.shape
            direct=float(average_precision_score(mask.reshape(-1),resized.reshape(-1))*100) if mask.any() and not mask.all() else None
            exact=official.weighted_pixel_metrics(mask.reshape(-1),resized.reshape(-1))['P_AP']
            if direct is not None:assert abs(direct-exact)<1e-10,(direct,exact)
            np.savez_compressed(dest/f'smoke_{i}.npz',maps74=maps,mask=mask)
            checks.append(dict(target=target,index=i,sample_id=row['sample_id'],
                global_mean_abs_error=float(gd.mean()),global_max_abs_error=float(gd.max()),
                local_mean_abs_error=float(ld.mean()) if ld is not None else 0.,
                local_max_abs_error=float(ld.max()) if ld is not None else 0.,
                global_dtype=str(a.dtype),local_cache_reused=local is not None,
                fusion_parity_error=err,direct_sklearn_AP=direct,main_evaluator_AP=exact,
                native_GT_shape=list(mask.shape),status='PASS'))
    after=encoder.checksum();assert encoder.before==after
    csv(OUT/'SMOKE_CHECKS.csv',checks)
    js(OUT/'SMOKE_COMPLETE.json',dict(status='PASS',config_hash=paired.json_hash(config),checks=checks,
        frozen_visual_checksum_before=encoder.before,frozen_visual_checksum_after=after,
        model_input=identity(encoder.checkpoint),target_performance_not_used_for_selection=True))
    del encoder;gc.collect();torch.cuda.empty_cache()
    say('SMOKE_PASSED',images=10,unchanged_visual=True)


def predict(target,config,entries,encoder=None):
    rows,a,local,mapping,gp,lp,dest=target_cell(target)
    transform,*_=geometry();w=weights(entries)
    expected=dict(config_hash=paired.json_hash(config),models=entries,n=len(rows),
                  global_cache=str(gp),global_cache_dtype=str(a.dtype),
                  manifest_sha256=paired.file_hash(dest/'TARGET_MANIFEST.csv'),
                  local_cache=str(lp) if local is not None else 'shared live frozen crop encoding')
    lock(dest/'IDENTITY.json',expected)
    pp=dest/'PROGRESS.json'
    done=official.readjson(pp)['done'] if pp.exists() else 0
    if done==len(rows):return encoder
    if local is None and encoder is None:encoder=FrozenEncoder()
    path=dest/'maps74.npy'
    maps=np.load(path,mmap_mode='r+') if path.exists() else np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(rows),6,74,74))
    assert maps.shape==(len(rows),6,74,74)
    start=time.time();initial=done
    for i in range(done,len(rows)):
        ix=mapping[i]
        loc=np.asarray(local[ix],np.float32) if local is not None else encoder.extract(rows[i]['img_path'])
        raw=np.concatenate((np.asarray(a[ix],np.float32)[None],loc))
        maps[i]=fused_maps(raw,transform,w)
        assert np.isfinite(maps[i]).all()
        if (i+1)%25==0 or i+1==len(rows):
            maps.flush();elapsed=time.time()-start
            js(pp,dict(done=i+1,total=len(rows),elapsed_this_session=elapsed,
                       eta_seconds=elapsed/max(1,i+1-initial)*(len(rows)-i-1)))
            say('PREDICTING',target=target,done=i+1,total=len(rows),
                elapsed_minutes=round(elapsed/60,2),eta_minutes=round(elapsed/max(1,i+1-initial)*(len(rows)-i-1)/60,2))
    maps.flush();del maps
    js(dest/'COMPLETE.json',dict(**expected,maps=identity(path)))
    return encoder


def evaluate(target,config,entries):
    """Call the exact main-table sufficient-statistic AP on native pixels.

    Disk-backed category vectors bound RAM without sampling or quantization.
    Only one head/category is expanded at a time; all predictions remain saved.
    """
    dest=OUT/'predictions'/target
    assert (dest/'COMPLETE.json').is_file()
    rows=official.readcsv(dest/'TARGET_MANIFEST.csv')
    maps=np.load(dest/'maps74.npy',mmap_mode='r')
    results=[]
    scratch=OUT/'evaluation_scratch';scratch.mkdir(exist_ok=True)
    for cat in sorted({r['category'] for r in rows}):
        ids=[i for i,r in enumerate(rows) if r['category']==cat]
        for col,entry in enumerate(entries):
            rp=OUT/'evaluations'/target/f'{cat}__{entry["model_id"]}.json'
            if rp.exists():
                old=official.readjson(rp)
                assert old['checkpoint_sha256']==entry['checkpoint_sha256'] and old['config_hash']==paired.json_hash(config)
                results.append(old);continue
            n=sum(int(rows[i]['gt_height'])*int(rows[i]['gt_width']) for i in ids)
            yp=scratch/'category_y.npy';sp=scratch/'category_score.npy'
            yy=np.lib.format.open_memmap(yp,mode='w+',dtype=np.bool_,shape=(n,))
            ss=np.lib.format.open_memmap(sp,mode='w+',dtype=np.float32,shape=(n,))
            offset=0
            for i in ids:
                mask=official.native_mask(rows[i]);score=official.resize(maps[i,col],mask.shape)
                count=mask.size;yy[offset:offset+count]=mask.reshape(-1);ss[offset:offset+count]=score.reshape(-1);offset+=count
            yy.flush();ss.flush();assert offset==n
            say('EVALUATING_NATIVE_CATEGORY',target=target,category=cat,model=entry['model_id'],pixels=n)
            metrics=official.weighted_pixel_metrics(yy,ss)
            result=dict(source=SOURCE,target=target,category=cat,variant=entry['variant'],seed=entry['seed'],
                        checkpoint=entry['checkpoint'],checkpoint_sha256=entry['checkpoint_sha256'],
                        config_hash=paired.json_hash(config),evaluator_hash=official.evaluator_hash(),
                        images=len(ids),pixels=n,P_AP=metrics['P_AP'])
            js(rp,result);results.append(result)
            del yy,ss;gc.collect()
            # Only new explicit scratch files owned by this run are removed.
            yp.unlink();sp.unlink()
    csv(OUT/'evaluations'/f'{target}_PER_CATEGORY.csv',results)
    return results


def summarize(config,entries):
    raw=[]
    for target in TARGETS:
        path=OUT/'evaluations'/f'{target}_PER_CATEGORY.csv'
        if not path.is_file():return False
        rr=official.readcsv(path)
        for entry in entries:
            values=[float(r['P_AP']) for r in rr if r['variant']==entry['variant'] and int(r['seed'])==entry['seed'] and r['P_AP'] not in ('','None')]
            total=len({r['category'] for r in rr})
            assert len(values)==total and np.isfinite(values).all()
            raw.append(dict(source=SOURCE,target=target,seed=entry['seed'],variant=entry['variant'],
                P_AP=float(np.mean(values)),valid_categories=len(values),checkpoint=entry['checkpoint'],
                checkpoint_sha256=entry['checkpoint_sha256'],evaluator_hash=official.evaluator_hash(),
                GT='paired original native masks',aggregation='equal valid-category macro',
                maps=str(OUT/'predictions'/target/'maps74.npy')))
    assert len(raw)==30
    csv(OUT/'GEOMETRY_CONTROL_RAW.csv',raw)
    def vals(target,variant):
        if target=='FIVE_TARGET_MEAN':
            return np.array([np.mean([r['P_AP'] for r in raw if r['seed']==seed and r['variant']==variant]) for seed in SEEDS])
        return np.array([next(r['P_AP'] for r in raw if r['target']==target and r['seed']==seed and r['variant']==variant) for seed in SEEDS])
    summary=[];table=[]
    for target in TARGETS+('FIVE_TARGET_MEAN',):
        h=vals(target,'Hybrid_CES');e=vals(target,'E1_Center_CES');d=h-e
        summary.append(dict(source=SOURCE,target=target,Hybrid_mean=float(h.mean()),Hybrid_sample_SD=float(h.std(ddof=1)),
            E1Center_mean=float(e.mean()),E1Center_sample_SD=float(e.std(ddof=1)),delta_mean=float(d.mean()),delta_sample_SD=float(d.std(ddof=1)),
            **{f'delta_seed{s}':float(d[s]) for s in SEEDS},Hybrid_better=int((d>.2).sum()),E1Center_better=int((d<-.2).sum()),
            approx_tied=int((abs(d)<=.2).sum()),near_tie_tolerance=.2,replication_unit='three paired source training seeds; not 15 independent trainings'))
        table.append(f'| {target} | {h.mean():.2f} ± {h.std(ddof=1):.2f} | {e.mean():.2f} ± {e.std(ddof=1):.2f} | {d.mean():+.2f} ± {d.std(ddof=1):.2f} |')
    csv(OUT/'GEOMETRY_CONTROL_SUMMARY.csv',summary)
    d=vals('FIVE_TARGET_MEAN','Hybrid_CES')-vals('FIVE_TARGET_MEAN','E1_Center_CES')
    outcome='A' if np.all(d>0) and d.mean()>=.2 else ('C' if np.all(d<0) and d.mean()<=-.2 else 'B')
    meanings=dict(A='Hybrid has a repeatable descriptive advantage under this fixed center control.',
                  B='Centers are close or training-seed directions are not consistent; describe sensitivity without a superiority claim.',
                  C='E1-Center has a repeatable descriptive advantage; revisit the hybrid-center choice.')
    rendered='| Target | Hybrid-CES | E1-Center-CES | Paired Δ (Hybrid − E1) |\n|---|---:|---:|---:|\n'+'\n'.join(table)
    md('GEOMETRY_CONTROL_REPORT.md',f'''# Effect of Source-Derived Reference Center

## Question and controls
Both models use the same fixed E1 direction/scale, frozen E2 adapter/PC1, true source labels, realized source pools and minibatches. Only the pre-normalization construction center differs. Six newly trained heads use 30 final-budget updates, three paired seeds, and no target selection. Full details and hashes are in GEOMETRY_CONTROL_CONFIG.json and the training audit. This is not a complete E1-only model or an adapter ablation.

## Native-GT category-macro P-AP (%)
{rendered}

SD is sample SD (ddof=1) over three training seeds. The final row averages all five targets within each seed before computing SD. Every seed delta is retained in the summary; no target is omitted. Near ties use the pre-evaluation descriptive tolerance0.2 percentage points, not a tuned decision threshold or significance test.

## Scientific conclusion
Outcome {outcome}: {meanings[outcome]} Three seeds support only the reported controlled observation; no population-wide optimality or independent E2-adapter benefit is established. Changed initial logits/state distance and optimization are recorded rather than assumed equal. The reference-center intervention changes the normalized effective scoring direction/scale even with shared raw v0/alpha.

No existing main table, checkpoint, figure, Word manuscript or SOTA baseline was modified. No upstream encoder/adapter/state initializer was retrained. Target GT was used only after all six heads were sealed. This control uses native GT and is not substituted for historical loader-GT StepB numbers.
''')
    md('GEOMETRY_CONTROL_PAPER_DRAFT.md',f'''# Effect of Source-Derived Reference Center

We compare two source-derived reference centers while holding the VisA feature transform, initial contrast direction, separation scale and optimization budget fixed. Hybrid-CES uses the E2 Stage-I midpoint, whereas E1-Center-CES uses the E1 Stage-I midpoint; both retain the same frozen E2 adapter. All heads are independently retrained under three paired seeds with identical realized source minibatches. Native-resolution category-macro pixel AP is evaluated on five unseen targets. The five-target mean paired difference is {d.mean():+.2f} ± {d.std(ddof=1):.2f} percentage points (sample SD across source-training seeds). {meanings[outcome]} This comparison concerns the reference center, not the independent contribution of the E2 adapter or a universal state-capacity optimum.

{rendered}

Values are mean ± sample SD over three paired seeds; the last row aggregates five targets within each seed. Paired Δ is Hybrid minus E1-Center.
''')
    js(OUT/'OUTCOME.json',dict(status='COMPLETE',outcome=outcome,conclusion=meanings[outcome],
        paired_five_target_delta=d.tolist(),mean_delta=float(d.mean()),sample_SD=float(d.std(ddof=1)),
        training_runs=6,raw_evaluations=30,original_paper_modified=False))
    say('COMPLETE',outcome=outcome,mean_paired_delta=float(d.mean()))
    return True


def training_audit(config,entries):
    checks=[]
    for seed in SEEDS:
        models=[torch.load(e['checkpoint'],map_location='cpu',weights_only=False) for e in entries if e['seed']==seed]
        assert len(models)==2
        a,b=models
        for k in ('feature_array_hash','labels_hash','batch_stream_hash','probe_indices_hash'):
            assert a[k]==b[k]
        assert torch.equal(a['raw_v_trajectory'][0],b['raw_v_trajectory'][0])
        assert torch.equal(a['raw_model_state']['alpha'],b['raw_model_state']['alpha'])
        assert not torch.equal(a['raw_model_state']['m'],b['raw_model_state']['m'])
        for p in models:
            assert len(p['logs'])==p['realized_updates']==30
            assert all(math.isfinite(r[k]) for r in p['logs'] for k in ('loss','bce','dice','ntr','grad_norm'))
            checks.append(dict(seed=seed,variant=p['variant'],updates=30,initial=p['initial_probe'],
                              final=p['final_probe'],batch_hash=p['batch_stream_hash'],NaN=False))
    js(OUT/'TRAINING_AUDIT.json',checks)
    md('GEOMETRY_CONTROL_TRAINING_AUDIT.md',f'''# Paired training audit

PASS. Six NEW heads, with raw_v and full optimizer state retained. Each checkpoint contains raw_v at initialization plus every update (31 snapshots), every training objective component/gradient, postupdate fixed-source probe statistics, frozen m/alpha, and step30 Adam states. Both configurations share exact feature/label/batch/probe hashes within each seed; batch identity is checked for every one of30 updates. Initial raw v and alpha are identical; m differs as intended. No initial-logit equivalence is imposed. All training losses/parameters are finite. No LR candidates or early stopping are used.

Physical source identities: read-only StepB PATCH_MANIFEST.csv and SOURCE_IMAGE_MANIFEST.csv for Full_VisA_seed0/1/2. Numerical validation reconstructs every retained source patch through the formal PC1/E2 path and checks actual true-mask labels. INITIALIZATION_NUMERICAL_CHECK.csv records full-pool feature parity, initial logit hashes/ranges and fixed-training-pool BCE/Dice/NTR, state distance and effective contrast norm. TRAINING_AUDIT.json records initial/final source components. The probe is part of the source training pool, not an independent validation set.

SOURCE_TRAINING_COMPLETE.json seals all six model hashes before any target scoring. Three source-training seeds are the repeated unit. Five target evaluations per seed are correlated outcomes, not five more independent training runs. Main-table native masks and exact tied-score AP are used, not old loader-GT metrics. Prediction/evaluation caches belong only to this new directory.

Model/code/input SHA256 are locked in config and prediction identities; frozen visual checksums are recorded by smoke and full-run extraction. Existing source assets must remain unchanged; the runner verifies their locked hashes on completion. No PC1/adapter/upstream state generation is permitted.
''')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('preflight','train','smoke','run','summarize'),default='run')
    args=parser.parse_args();setup()
    if args.stage=='preflight':
        config=preflight();numerical_validation(config);return
    config=official.readjson(OUT/'GEOMETRY_CONTROL_CONFIG.json')
    assert config['code'][0]['sha256']==paired.file_hash(__file__),'Scientific runner changed after lock'
    if args.stage=='train':
        entries=train(config);training_audit(config,entries);return
    entries=official.readjson(OUT/'SOURCE_TRAINING_COMPLETE.json')['models']
    if args.stage=='smoke':smoke(config,entries);return
    if args.stage=='summarize':
        assert summarize(config,entries),'Not all targets finished';return
    if not (OUT/'SMOKE_COMPLETE.json').exists():smoke(config,entries)
    assert official.readjson(OUT/'SMOKE_COMPLETE.json')['config_hash']==paired.json_hash(config)
    encoder=None
    for target in TARGETS:
        encoder=predict(target,config,entries,encoder)
        evaluate(target,config,entries)
    if encoder is not None:
        after=encoder.checksum();assert after==encoder.before
        js(OUT/'FULL_RUN_VISUAL_CHECKSUM.json',dict(before=encoder.before,after=after,unchanged=True))
    for item in config['input_assets']:
        assert paired.file_hash(item['path'])==item['sha256'],f'Input changed during experiment: {item["path"]}'
    assert summarize(config,entries)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OUT.exists():js(OUT/'LAST_ERROR.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
