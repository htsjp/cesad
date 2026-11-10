"""Post-run source-only acceptance; no optimizer, forward pass or target access."""
from pathlib import Path
import csv
import json
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import source_build_traceable_visa as job

OUT = job.OUTPUT/'runtime_audit'


def finite_tree(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value,np.ndarray) and value.dtype.kind in 'fc':
        return bool(np.isfinite(value).all())
    if isinstance(value,dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value,(list,tuple)):
        return all(finite_tree(v) for v in value)
    if isinstance(value,float):
        return math.isfinite(value)
    return True


def same_tree(a,b):
    if torch.is_tensor(a):
        return torch.equal(a,b)
    if isinstance(a,np.ndarray):
        return np.array_equal(a,b)
    if isinstance(a,dict):
        return a.keys() == b.keys() and all(same_tree(a[k],b[k]) for k in a)
    return a == b


def load(path):
    return torch.load(path,map_location='cpu',weights_only=False)


def unit(x):
    x = np.asarray(x,np.float32)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def run():
    seal_path = job.OUTPUT/'SOURCE_SEALED.json'
    seal = job.audit.read(seal_path)
    assert seal['complete'] is True and seal['updates'] == 30 and seal['seed'] == 0
    assert seal['target_access'] is False and seal['evaluation_approved'] is False
    manifest,manifest_id,approval_id = job.approval_gate()
    assert seal['identity']['source_input_manifest'] == manifest_id
    assert seal['identity']['training_approval'] == approval_id
    print('REVIEWED INPUTS AND CODE VERIFIED',flush=True)
    for entry in seal['locked_files']:
        job.audit.require_identity(entry)
    for entry in seal['endpoints'].values():
        job.audit.require_identity(entry)
    config = job.audit.read(job.REVIEW/'SOURCE_BUILD_CONFIG.json')
    expected = dict(E1_stage_I=20,E2_stage_I=20,CES=30,
        E1_stage_II=config['stage_II']['actual_updates']['E1'],
        E2_stage_II=config['stage_II']['actual_updates']['E2'])
    image_plan = np.load(job.REVIEW/'STAGE_I_IMAGE_BATCHES.npy')
    assert image_plan.shape == (20,2162)
    counts,chains = [],{}
    for stage,updates in expected.items():
        directory = job.WORK/'checkpoints'/stage
        paths = sorted(directory.glob('update_*.pt'))
        assert len(paths) == updates+1
        variant = stage[:2]
        if stage == 'CES':
            batches = list(np.load(job.REVIEW/'plans/CES/BATCHES.npy'))
        elif stage.endswith('_I'):
            batches = list(np.load(job.REVIEW/'plans'/variant/'STAGE_I_PATCH_BATCHES.npy'))
        else:
            with np.load(job.REVIEW/'plans'/variant/'STAGE_II_BATCHES.npz') as b:
                batches = [b['flat'][b['offsets'][i]:b['offsets'][i+1]] for i in range(len(b['offsets'])-1)]
        payloads=[]
        for step,path in enumerate(paths):
            assert path.name == f'update_{step:03d}.pt'
            job.audit.require_identity(job.audit.read(path.with_suffix('.sha256.json')))
            p = load(path)
            assert p['stage'] == stage and p['step'] == step
            assert p['identities']['source_input_manifest'] == manifest_id
            assert p['identities']['training_approval'] == approval_id
            assert p['identities']['config_sha256'] == manifest['config_sha256']
            assert finite_tree(p['params']) and finite_tree(p['optimizer_state']) and finite_tree(p['losses'])
            actual = np.load(p['actual_batch']['path'])
            job.audit.require_identity(p['actual_batch'])
            np.testing.assert_array_equal(actual,np.asarray([],np.int32) if step == 0 else batches[step-1])
            if step:
                counters = [int(s['step'].item()) for s in p['optimizer_state']['state'].values()]
                assert counters and all(c == step for c in counters)
                assert p['optimizer_counters'] == counters
                if stage.endswith('_I'):
                    job.audit.require_identity(p['params']['actual_image_batch'])
                    np.testing.assert_array_equal(np.load(p['params']['actual_image_batch']['path']),image_plan[step-1])
            else:
                assert not p['optimizer_state']['state'] and p['optimizer_counters'] == []
            for group in p['optimizer_state']['param_groups']:
                assert group['betas'] == (.9,.999) and group['eps'] == 1e-8 and group['amsgrad'] is False
                expected_lr,expected_wd = (.0001,.0001) if stage.endswith('_II') else (.001,0.)
                assert group['lr'] == expected_lr and group['weight_decay'] == expected_wd
            payloads.append(p)
        chains[stage] = payloads
        counts.append(dict(stage=stage,expected_updates=updates,actual_updates=payloads[-1]['step'],
            checkpoints=len(paths),all_optimizer_counters_match=True,all_actual_batches_match=True,
            finite_losses_parameters_optimizer=True,status='PASS'))
        print('STAGE VERIFIED',stage,updates,flush=True)
    pc1 = np.load(job.WORK/'pc1.npy')
    assert pc1.shape == (1024,) and np.isfinite(pc1).all() and abs(np.linalg.norm(pc1)-1) < 1e-6
    pc1_meta = job.audit.read(job.WORK/'PC1_IDENTITY.json')
    assert pc1_meta['identity'] == seal['identity'] and pc1_meta['normal_images'] == 962
    job.audit.require_identity(pc1_meta['file'])
    cert = job.audit.read(job.CERT)
    images = job.csvread(cert['image_feature_manifest']['path'])
    normals = np.asarray([i for i,r in enumerate(images) if int(r['image_label']) == 0])
    np.testing.assert_array_equal(np.load(job.WORK/'PC1_NORMAL_IMAGE_POSITIONS.npy'),normals)
    e1,e2 = [load(job.WORK/'assets'/f'{v}_formal.pt') for v in ('E1','E2')]
    for variant,asset,supervision in [('E1',e1,'true'),('E2',e2,'shifted')]:
        assert asset['supervision'] == supervision
        np.testing.assert_array_equal(asset['patch_states'],unit(chains[variant+'_stage_I'][-1]['params']['patch'].numpy()))
        np.testing.assert_array_equal(asset['img_states'],unit(chains[variant+'_stage_I'][-1]['params']['image'].numpy()))
        adapter_chain = chains[variant+'_stage_II']
        assert same_tree(asset['model'],adapter_chain[-1]['params']['model'])
        initial_states = adapter_chain[0]['params']['model']['states']
        np.testing.assert_array_equal(initial_states.numpy(),unit(asset['patch_states']))
        assert all(torch.equal(p['params']['model']['states'],initial_states) for p in adapter_chain)
    ces = chains['CES']
    initial,final = ces[0],ces[-1]
    assert seal['endpoints']['initialization']['path'] == str((job.WORK/'checkpoints/CES/update_000.pt').resolve())
    assert seal['endpoints']['update30']['path'] == str((job.WORK/'checkpoints/CES/update_030.pt').resolve())
    midpoint = .5*(e2['patch_states'][0]+e2['patch_states'][1])
    d = .5*(e1['patch_states'][1]-e1['patch_states'][0])
    alpha = float(np.linalg.norm(d))
    frozen = initial['params']['frozen']
    actual_adapter_hash = {k:job.array_hash(v.numpy()) for k,v in e2['model'].items()}
    assert frozen['adapter'] == actual_adapter_hash
    for p in ces:
        raw = p['params']['raw_model']
        assert set(raw) == {'m','alpha','v'} and raw['v'].numel() == 1024
        assert p['params']['frozen'] == frozen
        np.testing.assert_array_equal(raw['m'].numpy(),midpoint)
        assert raw['alpha'].item() == float(np.float32(alpha))
        assert p['params']['alpha'] == alpha
        assert p['identities']['pc1'] == job.audit.ident(job.WORK/'pc1.npy')
        assert p['identities']['E1'] == job.audit.ident(job.WORK/'assets/E1_formal.pt')
        assert p['identities']['E2'] == job.audit.ident(job.WORK/'assets/E2_formal.pt')
        assert len(p['optimizer_state']['param_groups'][0]['params']) == 1
        # Saved spec normalizes on CUDA; verify the formula on CPU with an
        # explicit FP32 tolerance, not a false cross-device bitwise requirement.
        np.testing.assert_allclose(p['params']['spec']['v'],F.normalize(raw['v'],dim=0).numpy(),rtol=1e-6,atol=1e-7)
    np.testing.assert_array_equal(initial['params']['raw_model']['v'].numpy(),d/(np.linalg.norm(d)+1e-12))
    v_initial = initial['params']['raw_model']['v'].numpy()
    v_final = final['params']['raw_model']['v'].numpy()
    cosine = float(np.clip(unit(v_initial)@unit(v_final),-1.,1.))
    angle = float(np.degrees(np.arccos(cosine)))
    endpoints=[]
    for name,p in [('initialization',initial),('update30',final)]:
        raw = p['params']['raw_model']
        v = F.normalize(raw['v'],dim=0)
        s_a = F.normalize(raw['m']+raw['alpha']*v,dim=0)
        s_n = F.normalize(raw['m']-raw['alpha']*v,dim=0)
        endpoints.append(dict(endpoint=name,step=p['step'],raw_direction_norm=float(raw['v'].norm()),
            effective_weight_norm=float((s_a-s_n).norm()),s_N_norm=float(s_n.norm()),s_A_norm=float(s_a.norm()),
            source_probe_loss=p['params']['source_probe']['loss'],
            source_probe_BCE=p['params']['source_probe']['bce'],source_probe_Dice=p['params']['source_probe']['dice'],
            source_probe_NTR=p['params']['source_probe']['ntr'],source_normal_mean=p['params']['source_probe']['normal_mean'],
            source_anomaly_mean=p['params']['source_probe']['anomaly_mean']))
    log = job.csvread(job.OUTPUT/'TRAINING_LOG.csv')
    assert len(log) == sum(v+1 for v in expected.values())
    assert len({(r['stage'],int(r['step'])) for r in log}) == len(log)
    assert job.audit.sha(job.audit.OUT/'PREFLIGHT_MANIFEST.json') == manifest['original_preflight_sha256']
    assert job.audit.sha(job.CERT) == manifest['feature_certificate_sha256']
    OUT.mkdir(parents=True,exist_ok=True)
    job.audit.csvout(OUT/'STAGE_UPDATE_COUNTS.csv',counts)
    job.audit.csvout(OUT/'CES_ENDPOINT_COMPARISON.csv',endpoints)
    result=dict(status='PASS_SOURCE_BUILD_ONLY_TARGET_EVALUATION_LOCKED',created=job.audit.now(),
        source='Full VisA',seed=0,total_optimizer_updates=sum(expected.values()),total_checkpoints=len(log),
        stages=counts,source_seal=job.audit.ident(seal_path),input_manifest=manifest_id,training_approval=approval_id,
        frozen_PC1_E2_adapter_m_alpha='PASS',CES_trainable_scalars=1024,PC1_normal_images=962,
        endpoints=endpoints,raw_direction_initial_final_cosine=cosine,raw_direction_rotation_degrees=angle,
        raw_direction_L2_change=float(np.linalg.norm(v_final-v_initial)),
        original_preflight_unchanged=True,feature_certificate_unchanged=True,all_declared_files_hash_verified=True,
        forward_passes_run_by_this_audit=0,optimizer_updates_run_by_this_audit=0,
        target_evaluation_run=False,target_evaluation_approved=False,
        limits='Source execution integrity only; no target-transfer gain or historical-model certification established.')
    job.audit.js(OUT/'RUNTIME_VERIFICATION.json',result)
    text=['# Source-build runtime acceptance','',
        '**PASS_SOURCE_BUILD_ONLY_TARGET_EVALUATION_LOCKED**','',
        'No source/target forward pass, optimizer update or metric evaluation was performed by this acceptance script. '
        'Saved checkpoints, optimizer states, physical index plans, frozen buffers and all declared file hashes were inspected.','',
        '| Stage | Actual updates | Checkpoints incl. initialization | Batch / optimizer / finite checks |',
        '|---|---:|---:|---|',
        *[f'| {r["stage"]} | {r["actual_updates"]} | {r["checkpoints"]} | PASS |' for r in counts],'',
        'PC1 uses the recorded 962 normal-image positions. E1 uses true patch supervision and E2 uses shifted '
        'patch supervision through both auxiliary stages. Stage-II state buffers remain fixed. '
        'CES m/alpha exactly match the rebuilt E2 midpoint/E1 half-difference, and its E2 adapter hashes '
        'are unchanged across all 30 updates. Raw v is the only final-stage optimized parameter (1024 scalars).','',
        '| CES endpoint | Source probe objective | Effective weight norm |',
        '|---|---:|---:|',
        *[f'| {r["endpoint"]} | {r["source_probe_loss"]:.9f} | {r["effective_weight_norm"]:.9f} |' for r in endpoints],'',
        f'Raw direction change: L2 {result["raw_direction_L2_change"]:.9f}; cosine {cosine:.9f}; rotation {angle:.6f} degrees. '
        'These are source-side execution diagnostics, not target results.','',
        'The execution log contains a PyTorch requires_grad-to-float UserWarning when recording NTR. '
        'No traceback or nonfinite value was found; the warning is not an optimizer failure. '
        'Reviewed training code was not modified to suppress it.','',
        'The original preflight, new source feature certificate and approved input manifest remain unchanged. '
        'Both CES endpoints are retained; no target-based endpoint selection took place. '
        'The complete SOURCE_SEALED.json is present. Target evaluation still requires a separate implementation/release review.','']
    (OUT/'SOURCE_BUILD_RUNTIME_AUDIT.md').write_text('\n'.join(text),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    run()
