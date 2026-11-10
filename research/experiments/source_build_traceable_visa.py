"""Reviewable source-only reconstruction, with a separate explicit training gate.

prepare: hash certified FP32 inputs, make source-only index plans and seal code.
source-build: requires an exact separate training approval; reconstructs all assets.
self-test: tests guards/samplers without optimization or encoder inference.
No target loader/evaluator is invoked. Historical inputs are never rewritten.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import rebuild_traceable_visa_source as audit

REVIEW = audit.OUT / 'source_build_review'
OUTPUT = audit.OUT / 'source_build'
WORK = audit.WORK / 'source_build/seed0'
CERT = audit.OUT / 'source_feature_certification/SOURCE_FEATURE_CERTIFICATE.json'
FEATURE_REVIEW = Path('D:/ProjectPython/Paper/reviews/VisA_Source_Feature_Certification_20261030')
SEED = 0


def csvread(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def array_hash(x):
    return hashlib.sha256(x.tobytes(order='C')).hexdigest()


def refs():
    from experiments import phase_xxxviii_visual_state_representation as states
    from experiments import phase_xlii_patch_specialized_adaptation as adapter
    from experiments.phase_xliii_adapter_head_decomposition import AdapterToState
    from experiments import phase_ces_linear_head_control as head
    from experiments import phase_ces_lhc_paired_repeat as paired
    from experiments import phase_spatial_supervision_transfer_audit as spatial
    return states, adapter, AdapterToState, head, paired, spatial


def fresh_only(cert):
    """Do not resolve old cache names or allow a storage/dimension fallback."""
    import numpy as np
    expected = audit.WORK / 'certified_source'
    for key, name, shape in (
        ('patch_features','source_patch_float32.npy',(2162,1369,1024)),
        ('cls_features','source_cls_float32.npy',(2162,768))):
        path = Path(cert[key]['path']).resolve()
        if path != (expected / name).resolve():
            raise RuntimeError(f'Non-certified source feature path: {key}')
        a = np.load(path, mmap_mode='r')
        if a.dtype != np.float32 or a.shape != shape:
            raise RuntimeError(f'Incorrect source dtype/shape: {key}')


def aux_pool_plan(images, labels):
    """Exact historical collect_pc1_removed_patch_samples index policy, seed 0."""
    import numpy as np
    rng = np.random.default_rng(SEED)
    groups = {}
    for pos, image in enumerate(images):
        for label in (0,1):
            ids = np.flatnonzero(labels[pos] == label)
            parts = groups.setdefault((image['category'],label), [])
            room = 512-len(parts)
            if room <= 0 or len(ids) == 0:
                continue
            if len(ids) > room:
                ids = rng.choice(ids, room, replace=False)
            parts.extend((pos,int(pid)) for pid in ids)
    pairs, ys, cats = [], [], []
    for (cat,label), members in sorted(groups.items()):
        pairs.extend(members); ys.extend([label]*len(members)); cats.extend([cat]*len(members))
    return np.asarray(pairs,np.int32), np.asarray(ys,np.int64), np.asarray(cats)


def ces_pool_plan(images, labels):
    """Exact repaired VisA category/origin prefix policy; true labels only."""
    import numpy as np
    rng = np.random.default_rng(SEED)
    pairs, ys, cats, origins = [], [], [], []
    for cat in sorted({r['category'] for r in images}):
        parts = {k:[] for k in ('normal_image','anomaly_mask_out','anomaly_mask')}
        for pos, image in enumerate(images):
            if image['category'] != cat:
                continue
            y = labels[pos]
            pools = [('normal_image',0,np.flatnonzero(y == 0))] if y.max() == 0 else [
                ('anomaly_mask_out',0,np.flatnonzero(y == 0)), ('anomaly_mask',1,np.flatnonzero(y == 1))]
            for origin, label, ids in pools:
                room = 512-len(parts[origin])
                if room <= 0 or len(ids) == 0:
                    continue
                if len(ids) > room:
                    ids = rng.choice(ids, room, replace=False)
                parts[origin].extend((pos,int(pid),label) for pid in ids)
        for origin, members in parts.items():
            for pos,pid,label in members:
                pairs.append((pos,pid)); ys.append(label); cats.append(cat); origins.append(origin)
    return np.asarray(pairs,np.int32), np.asarray(ys,np.int64), np.asarray(cats), np.asarray(origins)


def state_batches(y, cat):
    import numpy as np
    rng = np.random.default_rng(SEED)
    batches = []
    for _ in range(20):
        parts = []
        for name in sorted(set(cat.tolist())):
            ids = np.flatnonzero(cat == name)
            normal, anomaly = ids[y[ids] == 0], ids[y[ids] == 1]
            if not len(normal) or not len(anomaly):
                continue
            take = min(len(normal),len(anomaly),128)
            parts.extend([rng.choice(normal,take,replace=False),rng.choice(anomaly,take,replace=False)])
        if not parts:
            normal, anomaly = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
            take = min(len(normal),len(anomaly),2048)
            parts = [rng.choice(normal,take,replace=False),rng.choice(anomaly,take,replace=False)]
        batches.append(np.concatenate(parts).astype(np.int32))
    return np.stack(batches)


def adapter_batches(n):
    import numpy as np
    rng = np.random.default_rng(SEED)
    batches, epochs = [], []
    for epoch in range(1,11):
        order = rng.permutation(n)
        for start in range(0,n,8192):
            batches.append(order[start:start+8192].astype(np.int32)); epochs.append(epoch)
    offsets = np.cumsum([0]+[len(ids) for ids in batches]).astype(np.int64)
    return dict(flat=np.concatenate(batches), offsets=offsets, epochs=np.asarray(epochs,np.int32))


def plan_entry(directory, variant, pairs, y, cat, origins=None):
    import numpy as np
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory/'POOL_PLAN.npz', physical_pairs=pairs, labels=y, categories=cat,
             origins=origins if origins is not None else np.asarray(['auxiliary']*len(y)))
    audit.csvout(directory/'PHYSICAL_PATCH_PLAN.csv', [dict(pool_index=i,source_position=int(pos),
        patch_flat_index=int(pid),patch_id=f'visa:{pos}:{pid}',label=int(y[i]),category=str(cat[i]),
        origin=str(origins[i]) if origins is not None else 'auxiliary') for i,(pos,pid) in enumerate(pairs)])
    return dict(variant=variant, patches=len(y), categories=len(set(cat.tolist())),
                labels=array_hash(y), physical_pairs=array_hash(pairs))


def prepare():
    import numpy as np
    import torch
    states, adapter, adapter_class, head, paired, spatial = refs()
    REVIEW.mkdir(parents=True, exist_ok=True)
    if (REVIEW/'SOURCE_INPUT_MANIFEST.json').exists():
        raise RuntimeError('Source input manifest already sealed; refuse overwrite')
    # Preparation stops before PCA, model construction, optimizer construction or fitting.
    cert = audit.read(CERT)
    preflight = audit.OUT/'PREFLIGHT_MANIFEST.json'
    if audit.sha(preflight) != cert['original_preflight_sha256']:
        raise RuntimeError('Original preflight identity mismatch')
    review_path, verification_path = FEATURE_REVIEW/'REVIEW.md', FEATURE_REVIEW/'VERIFICATION.json'
    verification = audit.read(verification_path)
    if verification['decision'] != 'NEW_SOURCE_FEATURE_ARTIFACT_REVIEW_PASSED_ONLY':
        raise RuntimeError('Missing explicit feature-only review')
    if verification['source_feature_certificate_sha256'] != audit.sha(CERT):
        raise RuntimeError('Feature review certificate mismatch')
    if verification['original_preflight_sha256'] != audit.sha(preflight):
        raise RuntimeError('Feature review preflight mismatch')
    fresh_only(cert)
    print('PREPARE: verify certified inputs; NO PCA/TRAINING/ENCODER/TARGET operations', flush=True)
    consumed = [audit.ident(CERT), audit.ident(preflight), audit.ident(review_path),audit.ident(verification_path)]
    for key in ('patch_features','cls_features','source_patch_binding','source_labels','image_feature_manifest',
                'extractor_config','encoder_identity','base_CLIP_weights','validation_summary'):
        audit.require_identity(cert[key]); consumed.append(cert[key])
        print('INPUT VERIFIED',key,flush=True)
    images = csvread(cert['image_feature_manifest']['path'])
    labels = np.load(cert['source_labels']['path'])
    assert len(images) == 2162 and labels.shape == (2162,1369) and set(np.unique(labels)) == {0,1}
    assert all(int(r['source_position']) == i and int(r['patch_feature_row']) == i and int(r['cls_feature_row']) == i
               for i,r in enumerate(images))
    assert sum(int(r['image_label']) == 0 for r in images) == 962
    binding_count = 0
    with gzip.open(cert['source_patch_binding']['path'],'rt',encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            pos, pid = int(r['patch_feature_row']), int(r['patch_flat_index'])
            assert binding_count == pos*1369+pid
            assert int(r['image_ledger_row']) == pos and int(r['label']) == int(labels[pos,pid])
            assert r['patch_id'] == f'visa:{pos}:{pid}' and int(r['patch_y']) == pid//37 and int(r['patch_x']) == pid%37
            binding_count += 1
    assert binding_count == labels.size
    records = [dict(pos=i,idx=int(row['dataset_index']),normal=not bool(labels[i].any()),mask=labels[i].astype(np.float32))
               for i,row in enumerate(images)]
    shifted, shift_audit = spatial.corrupt_masks(records)
    shifted = (np.stack(shifted) > .5).astype(np.uint8)
    np.save(REVIEW/'SHIFTED_LABELS.npy',shifted)
    audit.csvout(REVIEW/'SHIFTED_MASK_AUDIT.csv',shift_audit)
    # Full image-label stream is independent of shifted patch labels.
    np.save(REVIEW/'STAGE_I_IMAGE_BATCHES.npy',np.tile(np.arange(2162,dtype=np.int32),(20,1)))
    plan_summary = []
    for variant, ymap in [('E1',labels),('E2',shifted)]:
        folder = REVIEW/'plans'/variant
        pairs,y,cat = aux_pool_plan(images,ymap)
        plan_summary.append(plan_entry(folder,variant,pairs,y,cat))
        np.save(folder/'STAGE_I_PATCH_BATCHES.npy',state_batches(y,cat))
        ab = adapter_batches(len(y))
        np.savez(folder/'STAGE_II_BATCHES.npz',**ab)
        plan_summary[-1].update(state_updates=20,adapter_updates=len(ab['epochs']),adapter_epochs=10)
    pairs,y,cat,origins = ces_pool_plan(images,labels)
    folder = REVIEW/'plans/CES'
    plan_summary.append(plan_entry(folder,'CES',pairs,y,cat,origins))
    if 10*math.ceil(len(y)/8192) != 30:
        raise RuntimeError('Fixed 10-epoch policy conflicts with required 30 CES updates; needs explicit review')
    rng = np.random.default_rng(SEED)
    batches = np.stack([paired.balanced_indices(y,cat,rng) for _ in range(30)])
    assert batches.shape == (30,8184)
    np.save(folder/'BATCHES.npy',batches)
    np.save(folder/'PROBE.npy',paired.balanced_indices(y,cat,np.random.default_rng(1000000+SEED)))
    plan_summary[-1].update(updates=30,actual_batch=8184)
    paired.write_csv(REVIEW/'SOURCE_POOL_PLAN_SUMMARY.csv',plan_summary)
    assert states.EPOCHS == 20 and states.LR == .001 and adapter.RANK == 4
    assert adapter.EPOCHS == 10 and adapter.LR == .0001 and adapter.WEIGHT_DECAY == .0001
    assert paired.EPOCHS == 10 and paired.RHO == .1
    config = dict(source='Full VisA',seed=0,input_dtype='float32',source_images=2162,
        input_policy='ONLY certified fresh patch, CLS and true labels; no old-cache/payload fallback',
        output_root=str(OUTPUT),work_root=str(WORK),
        pc1=dict(normal_images='image_label == 0, 962 real normal images; no anomaly-image normal regions',
            construction='float64 sum normalized source patch maps -> image mean -> spatial centering -> numpy.linalg.svd(full_matrices=False)',
            direction='vt[0] -> float32 L2; no imposed sign',updates=0),
        auxiliary_pool=dict(cap_per_category_class=512,order='source image order, then cls0/1; output sorted(category,class)',
            sampling='default_rng(0), choice without replacement only if remaining quota oversubscribed',
            feature='remove_pc1 raw FP32 -> norm_np; no adapter at Stage I',plans=plan_summary[:2]),
        shifted_masks=dict(function='phase_spatial_supervision_transfer_audit.corrupt_masks',seed=0,
            grid=[37,37],max_attempts=128,dy_dx=[0,36],zero_shift='skip consumes attempt',
            rule='first IoU<0.1 else lowest attempted IoU; circular wrap; all components shifted together'),
        stage_I=dict(initial_states='unit class means of normalized image CLS and PC1-removed unit patch pool',
            image_supervision='original source image label for both E1/E2',
            patch_supervision='E1 true, E2 shifted',image_batch=2162,patch_take_per_category_class=128,
            loss='weighted image CE + unweighted class-balanced sampled patch CE; each weight 1',
            optimizer='AdamW',lr=.001,weight_decay=0.,betas=[.9,.999],eps=1e-8,amsgrad=False,
            updates=20,trainable='image pair[2,768], patch pair[2,1024]',selection='last update'),
        stage_II=dict(network='AdapterToState: L2(x+B(A(x))), no bias/activation; A[4,1024], B[1024,4]',
            initialization='torch.manual_seed(0); A normal std0.02, B zeros; class constructor unchanged',
            states='frozen unit Stage-I patch states',supervision='E1 true, E2 shifted, never silently reverted',
            inputs='norm_np of Stage-I patch pool (same extra normalization as original)',
            loss='balanced BCE mean + sigmoid Dice with epsilon1, weights1+1',
            optimizer='AdamW',lr=.0001,weight_decay=.0001,betas=[.9,.999],eps=1e-8,amsgrad=False,
            epochs=10,batch=8192,actual_updates={r['variant']:r['adapter_updates'] for r in plan_summary[:2]},
            selection='last update'),
        ces=dict(m='(E2 Stage-I sN+sA)/2',d='(E1 Stage-I sA-sN)/2',alpha='L2 norm(d)',v0='unit(d)',
            states='unit(m +/- alpha*unit(raw_v))',score='z dot (sA-sN), no independent bias or temperature',
            input='PC1 removal -> L2 -> same frozen newly trained E2 adapter -> L2',
            labels='true masks ONLY',trainable='raw_v[1024] ONLY',
            frozen='m, alpha, PC1, E2 adapter; no encoder is loaded by this runner',
            optimizer='AdamW',lr=.001,weight_decay=0.,betas=[.9,.999],eps=1e-8,amsgrad=False,
            epochs=10,updates=30,batch=8184,pool=plan_summary[2],
            loss='balanced BCE + sigmoid Dice + NTR',rho=.1,NTR_weight=1.,
            ntr='ceil(rho * batch normal count), detached topk logit indices, mean softplus of selected live logits',
            endpoints=['initialization','update30'],selection='retain both; no target selection'),
        optimizer_options=dict(foreach=None,fused=None,maximize=False,capturable=False,differentiable=False),
        runtime=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,
            dtype='float32 training',pc1_accumulation='float64',device='cuda',TF32=False,autocast=False,
            deterministic=True,torch_num_threads=4,CUBLAS_WORKSPACE_CONFIG=':4096:8',
            cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
            gpu_driver=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip()),
        persistence='initialization and EVERY optimizer update: raw parameters + optimizer + RNG + actual index files + loss + frozen hashes',
        resumption='only hash-verified new-stage checkpoints bound to this manifest; no historical asset fallback',
        target_evaluation='NOT IMPLEMENTED OR APPROVED; future 5-target evaluation after complete SOURCE_SEALED.json',
        feature_certificate=audit.ident(CERT),original_preflight=audit.ident(preflight),
        feature_review=audit.ident(review_path),feature_verification=audit.ident(verification_path))
    audit.js(REVIEW/'SOURCE_BUILD_CONFIG.json',config)
    code = audit.code_inventory([Path(__file__),Path(audit.__file__),Path(states.__file__),Path(adapter.__file__),
                                Path(head.__file__),Path(paired.__file__),Path(spatial.__file__)])
    # Explicit seeds cover relative imports used by the actual retained adapter class.
    code.extend([Path(inspect.getfile(adapter_class)),ROOT/'tests/test_traceable_visa_source_build.py'])
    code = sorted(set(p.resolve() for p in code))
    code_ids = [audit.ident(p) for p in code]
    audit.js(REVIEW/'TRAINING_CODE_IDENTITY.json',code_ids)
    snapshot = REVIEW/'TRAINING_CODE_SNAPSHOT.zip'
    with zipfile.ZipFile(snapshot,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in code:
            z.write(path,str(path.relative_to(ROOT)))
    tests_path = REVIEW/'SOURCE_BUILD_GUARD_TESTS.json'
    tests = audit.read(tests_path)
    if not tests['passed'] or tests['training_executed'] or tests['runner_sha256'] != audit.sha(__file__):
        raise RuntimeError('Current runner guard tests missing/stale')
    doc = ['# Source-build implementation awaiting separate training review', '',
        '**No PCA, model fitting, optimizer creation, encoder inference or target evaluation has been executed.**', '',
        'The approved new float32 patch/CLS certificate and actual true labels are the only feature inputs. '
        'Historical PC1/E1/E2/adapter/CES files and old carrier features are never loaded by the runner.', '',
        '## Fixed sequence', '',
        '1. PC1: mean the 962 normal-image patch maps in float64, spatially center, full SVD; unit vt[0].',
        '2. E1/E2 Stage I: source-only class-mean initialization and 20 AdamW updates, weighted image CE + sampled patch CE. '
        'E1 patch labels are true; E2 patch labels are the fixed circular-shift labels. Both use real image labels.',
        '3. E1/E2 Stage II: independent rank-4 residual adapters, fixed unit Stage-I states, 10 epochs. '
        'E2 continues using shifted labels. The final scorer retains the E2 adapter only.',
        '4. CES: E1 Stage-I half-difference gives direction/alpha; E2 Stage-I midpoint gives m. '
        'Only raw_v is optimized for 30 updates with true labels, BCE+Dice+NTR(rho=.10).', '',
        'Physical pool plans, planned indices, shuffled adapter batches, fixed CES batches and probe IDs are already sealed. '
        'During future approved execution each actual consumed index array is saved separately and matched to its plan. '
        'Initial and every-step checkpoints retain raw parameters, optimizer state/counters, RNG and stage input identities. '
        'Final source seal must bind PC1, E1/E2, E2 adapter, CES initialization and update30.', '',
        '## Approval contract', '',
        'SOURCE_BUILD_APPROVAL.json is not created by prepare. source-build requires decision '
        'APPROVED_SOURCE_BUILD_ONLY, reviewer, exact source_input_manifest_sha256 and config_sha256. '
        'It rehashes every locked input/code/plan; missing or changed input stops before optimizer creation. '
        'The feature review alone never authorizes training. The original blocked preflight is immutable.', '',
        '## Scope and limits', '',
        'No training numerical parity or convergence has been tested in this preparation-only run. '
        'Sampling/loss definitions reuse the original helpers; training loops are instrumented equivalents with '
        'explicit checkpoints. The implementation remains subject to review. No target entry is released.', '',
        f'New training output: {OUTPUT}', f'New heavy output: {WORK}', '']
    (REVIEW/'SOURCE_BUILD_IMPLEMENTATION.md').write_text('\n'.join(doc),encoding='utf-8')
    locked = consumed+code_ids+[audit.ident(p) for p in sorted(REVIEW.rglob('*')) if p.is_file()]
    locked = list({r['path']:r for r in locked}.values())
    manifest = dict(created=audit.now(),status='IMPLEMENTATION_SEALED_AWAITING_TRAINING_REVIEW',
        original_preflight_sha256=audit.sha(preflight),feature_certificate_sha256=audit.sha(CERT),
        feature_review_sha256=audit.sha(review_path),config_sha256=audit.sha(REVIEW/'SOURCE_BUILD_CONFIG.json'),
        locked_files=locked,plans=plan_summary,training_approved=False,training_executed=False,
        PCA_executed=False,target_evaluation_approved=False,
        training_approval_path=str(REVIEW/'SOURCE_BUILD_APPROVAL.json'))
    audit.js(REVIEW/'SOURCE_INPUT_MANIFEST.json',manifest)
    print(json.dumps(dict(status=manifest['status'],pools=plan_summary,
        source_input_manifest=str(REVIEW/'SOURCE_INPUT_MANIFEST.json'),
        manifest_sha256=audit.sha(REVIEW/'SOURCE_INPUT_MANIFEST.json'),training_executed=False),indent=2),flush=True)


def approval_gate(base=REVIEW):
    """All approval checks precede imports that could construct an optimizer."""
    manifest_path = base/'SOURCE_INPUT_MANIFEST.json'
    manifest = audit.read(manifest_path)
    approval = audit.read(base/'SOURCE_BUILD_APPROVAL.json')
    if approval.get('decision') != 'APPROVED_SOURCE_BUILD_ONLY' or not approval.get('reviewer'):
        raise RuntimeError('Separate explicit source-build approval required')
    if approval.get('source_input_manifest_sha256') != audit.sha(manifest_path):
        raise RuntimeError('Training approval manifest hash mismatch')
    if approval.get('config_sha256') != manifest['config_sha256']:
        raise RuntimeError('Training approval config hash mismatch')
    if manifest['status'] != 'IMPLEMENTATION_SEALED_AWAITING_TRAINING_REVIEW':
        raise RuntimeError('Unrecognized source input manifest status')
    for record in manifest['locked_files']:
        audit.require_identity(record)
    return manifest,audit.ident(manifest_path),audit.ident(base/'SOURCE_BUILD_APPROVAL.json')


def cpu_tree(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value,dict):
        return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):
        return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def save_torch(path,value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() or path.with_suffix('.sha256.json').exists():
        raise RuntimeError(f'Refuse checkpoint overwrite: {path}')
    temp = path.with_suffix('.tmp')
    torch.save(cpu_tree(value),temp)
    temp.replace(path)
    audit.js(path.with_suffix('.sha256.json'),audit.ident(path))


def checkpoint(stage,step,params,opt,batch,losses,identities):
    import numpy as np
    import torch
    actual = WORK/'actual_batches'/stage/f'{step:03d}.npy'
    actual.parent.mkdir(parents=True,exist_ok=True)
    if actual.exists():
        raise RuntimeError(f'Duplicate actual batch output: {actual}')
    np.save(actual,np.asarray(batch,np.int32))
    counters = [int(s['step'].item()) for s in opt.state.values()]
    if step and (not counters or any(s != step for s in counters)):
        raise RuntimeError(f'Optimizer counter mismatch: {stage}/{step}')
    payload = dict(stage=stage,step=step,params=params,optimizer_state=opt.state_dict(),
        actual_batch=audit.ident(actual),losses=losses,identities=identities,
        rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state()),
        optimizer_counters=counters,selection='FIXED_BUDGET_NO_TARGET_ACCESS')
    path = WORK/'checkpoints'/stage/f'update_{step:03d}.pt'
    save_torch(path,payload)
    return payload,path


def resume(stage,identities,expected_batches):
    import numpy as np
    import torch
    directory = WORK/'checkpoints'/stage
    paths = sorted(directory.glob('update_*.pt')) if directory.exists() else []
    if not paths:
        return None
    for expected,path in enumerate(paths):
        if path.name != f'update_{expected:03d}.pt':
            raise RuntimeError(f'Noncontiguous new-stage checkpoint chain: {stage}')
        audit.require_identity(audit.read(path.with_suffix('.sha256.json')))
        saved = torch.load(path,map_location='cpu',weights_only=False)
        if saved['step'] != expected or saved['identities'] != identities:
            raise RuntimeError(f'Checkpoint provenance mismatch: {path}')
        audit.require_identity(saved['actual_batch'])
        expected_ids = np.asarray([],np.int32) if expected == 0 else expected_batches[expected-1]
        if not np.array_equal(np.load(saved['actual_batch']['path']),expected_ids):
            raise RuntimeError(f'Actual batch does not match sealed plan: {path}')
        if 'actual_image_batch' in saved['params']:
            audit.require_identity(saved['params']['actual_image_batch'])
            image_ids = np.load(saved['params']['actual_image_batch']['path'])
            if not np.array_equal(image_ids,np.arange(2162,dtype=np.int32)):
                raise RuntimeError('Stage-I actual image batch changed')
    return saved


def rng_restore(saved):
    import numpy as np
    import torch
    torch.set_rng_state(saved['rng']['torch'])
    torch.cuda.set_rng_state_all(saved['rng']['cuda'])
    np.random.set_state(saved['rng']['numpy'])


def adam(params,lr,wd):
    import torch
    return torch.optim.AdamW(params,lr=lr,weight_decay=wd,betas=(.9,.999),eps=1e-8,
        amsgrad=False,foreach=None,fused=None,maximize=False,capturable=False,differentiable=False)


def pool_features(raw,plan,q,normalizer,per_image=False):
    import numpy as np
    pairs = plan['physical_pairs']
    if per_image:
        # Auxiliary legacy construction projects each full map before selecting
        # patch IDs. Keep that operation grouping, not just its algebraic form.
        result = np.empty((len(pairs),1024),np.float32)
        for pos in np.unique(pairs[:,0]):
            selected = np.flatnonzero(pairs[:,0] == pos)
            image = np.asarray(raw[int(pos)],np.float32)
            residual = image-(image@q)[:,None]*q
            result[selected] = normalizer(residual[pairs[selected,1]])
        return result
    x = np.stack([raw[int(pos),int(pid)] for pos,pid in pairs]).astype(np.float32)
    return normalizer(x-(x@q)[:,None]*q)


def run_stage_I(variant,x,y,cat,cls,image_y,plan,identities,states):
    import numpy as np
    import torch
    import torch.nn.functional as F
    stage = variant+'_stage_I'
    torch.manual_seed(SEED)
    image_initial,patch_initial = states.centroids(cls,image_y),states.centroids(x,y)
    ip = torch.nn.Parameter(torch.tensor(image_initial,device='cuda'))
    pp = torch.nn.Parameter(torch.tensor(patch_initial,device='cuda'))
    opt = adam([ip,pp],.001,0.)
    image_x = torch.tensor(states.norm_np(cls),device='cuda')
    target = torch.tensor(image_y,dtype=torch.long,device='cuda')
    counts = np.bincount(image_y,minlength=2).astype(np.float32)
    weights = torch.tensor((counts.sum()/np.maximum(counts,1.))/2.,device='cuda')
    # Original train_state_vectors applies this additional norm before CE.
    normalized = states.norm_np(x)
    saved = resume(stage,identities,plan)
    if saved is None:
        checkpoint(stage,0,dict(image=ip,patch=pp),opt,[],{},identities)
        start = 0
    else:
        with torch.no_grad():
            ip.copy_(saved['params']['image'].to('cuda')); pp.copy_(saved['params']['patch'].to('cuda'))
        opt.load_state_dict(saved['optimizer_state']); rng_restore(saved); start = saved['step']
    for step in range(start,20):
        ids = plan[step]
        opt.zero_grad()
        image_loss = F.cross_entropy(image_x@F.normalize(ip,dim=-1).T,target,weight=weights)
        px = torch.tensor(normalized[ids],device='cuda')
        py = torch.tensor(y[ids],dtype=torch.long,device='cuda')
        patch_loss = F.cross_entropy(px@F.normalize(pp,dim=-1).T,py)
        loss = image_loss+patch_loss
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite Stage-I loss')
        loss.backward(); opt.step()
        if not torch.isfinite(ip).all() or not torch.isfinite(pp).all():
            raise RuntimeError('Nonfinite Stage-I parameter')
        image_batch_path = WORK/'actual_batches'/stage/f'image_{step+1:03d}.npy'
        image_batch_path.parent.mkdir(parents=True,exist_ok=True)
        if image_batch_path.exists():
            raise RuntimeError('Uncommitted image batch exists; diagnose before resume')
        np.save(image_batch_path,np.arange(2162,dtype=np.int32))
        checkpoint(stage,step+1,dict(image=ip,patch=pp,actual_image_batch=audit.ident(image_batch_path)),opt,ids,
            dict(loss=float(loss.detach()),image_CE=float(image_loss.detach()),patch_CE=float(patch_loss.detach())),identities)
    return states.norm_np(ip.detach().cpu().numpy()),states.norm_np(pp.detach().cpu().numpy())


def run_adapter(variant,x,y,pair,batches,identities,states,adapter,adapter_class):
    import numpy as np
    import torch
    stage = variant+'_stage_II'
    model = adapter_class(1024,pair).to('cuda')
    opt = adam(model.parameters(),.0001,.0001)
    initial_states = model.states.detach().clone()
    x,y = states.norm_np(x),y.astype(np.float32)
    offsets,flat = batches['offsets'],batches['flat']
    actual_plan = [flat[offsets[i]:offsets[i+1]] for i in range(len(offsets)-1)]
    saved = resume(stage,identities,actual_plan)
    if saved is None:
        checkpoint(stage,0,dict(model=model.state_dict()),opt,[],{},identities); start = 0
    else:
        model.load_state_dict(saved['params']['model'],strict=True)
        opt.load_state_dict(saved['optimizer_state']); rng_restore(saved); start = saved['step']
    for step in range(start,len(offsets)-1):
        ids = flat[offsets[step]:offsets[step+1]]
        xx,yy = torch.tensor(x[ids],device='cuda'),torch.tensor(y[ids],device='cuda')
        opt.zero_grad()
        logits,_ = model.logits(xx)
        loss,bce,dice = adapter.balanced_loss(logits,yy)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite adapter loss')
        loss.backward(); opt.step()
        if not torch.equal(initial_states,model.states) or any(not torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError('Adapter states changed or nonfinite parameters')
        checkpoint(stage,step+1,dict(model=model.state_dict()),opt,ids,
            dict(loss=float(loss.detach()),bce=float(bce),dice=float(dice),epoch=int(batches['epochs'][step])),identities)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def run_ces(raw,plan,q,e1,e2,align,identities,states,adapter,head):
    import numpy as np
    import torch
    x = pool_features(raw,plan,q,states.norm_np)
    with torch.no_grad():
        z = np.concatenate([align.adapt(torch.tensor(x[i:i+8192],device='cuda')).cpu().numpy()
                            for i in range(0,len(x),8192)]).astype(np.float32)
    y = plan['labels'].astype(np.float32)
    d = .5*(e1[1]-e1[0]); m = .5*(e2[0]+e2[1]); alpha = float(np.linalg.norm(d)); v0 = head.unit_np(d)
    if not np.isfinite(alpha) or alpha <= 0:
        raise RuntimeError('Invalid source geometry, no fallback')
    torch.manual_seed(SEED)
    model = head.CESHead(m,alpha,v0).to('cuda')
    assert sum(p.numel() for p in model.parameters()) == 1024
    opt = adam(model.parameters(),.001,0.)
    batches = np.load(REVIEW/'plans/CES/BATCHES.npy')
    probe = np.load(REVIEW/'plans/CES/PROBE.npy')
    directory = WORK/'CES_pool'
    directory.mkdir(parents=True,exist_ok=True)
    for name,value in [('features',z),('labels',y)]:
        path = directory/f'{name}.npy'
        if path.exists():
            if not np.array_equal(np.load(path),value):
                raise RuntimeError('CES pool changed on resume')
        else:
            np.save(path,value)
    frozen = dict(m=array_hash(m),alpha=alpha,adapter={k:array_hash(v.detach().cpu().numpy()) for k,v in align.state_dict().items()})
    def probe_scores():
        with torch.no_grad():
            logits = model(torch.tensor(z[probe],device='cuda'))
            labels = torch.tensor(y[probe],device='cuda')
            base,bce,dice = adapter.balanced_loss(logits,labels)
            ntr = head.ntr_loss_from_logits(logits,labels,.1)
            return dict(loss=float(base+ntr),bce=float(bce),dice=float(dice),ntr=float(ntr),
                        normal_mean=float(logits[labels < .5].mean()),anomaly_mean=float(logits[labels > .5].mean()))
    def parameters():
        return dict(raw_model=model.state_dict(),m=m,alpha=alpha,v0=v0,
                    frozen=frozen,spec=model.to_spec('Full VisA').params,source_probe=probe_scores())
    saved = resume('CES',identities,batches)
    if saved is None:
        checkpoint('CES',0,parameters(),opt,[],probe_scores(),identities); start = 0
    else:
        model.load_state_dict(saved['params']['raw_model'],strict=True)
        if saved['params']['frozen'] != frozen:
            raise RuntimeError('Frozen geometry/adapter changed')
        opt.load_state_dict(saved['optimizer_state']); rng_restore(saved); start = saved['step']
    for step in range(start,30):
        ids = batches[step]
        xx,yy = torch.tensor(z[ids],device='cuda'),torch.tensor(y[ids],device='cuda')
        logits = model(xx)
        base,bce,dice = adapter.balanced_loss(logits,yy)
        ntr = head.ntr_loss_from_logits(logits,yy,.1)
        loss = base+ntr
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite CES loss')
        opt.zero_grad(set_to_none=True); loss.backward()
        grad = float(torch.linalg.vector_norm(model.v.grad).detach())
        opt.step()
        if not torch.isfinite(model.v).all():
            raise RuntimeError('Nonfinite CES direction')
        if not np.array_equal(model.m.detach().cpu().numpy(),m) or float(model.alpha.detach().cpu()) != float(np.float32(alpha)):
            raise RuntimeError('Frozen CES geometry changed')
        checkpoint('CES',step+1,parameters(),opt,ids,
            dict(loss=float(loss.detach()),bce=float(bce),dice=float(dice),ntr=float(ntr),gradient_norm=grad),identities)
    if {k:array_hash(v.detach().cpu().numpy()) for k,v in align.state_dict().items()} != frozen['adapter']:
        raise RuntimeError('Frozen E2 adapter changed during CES training')
    return model


def source_build():
    # Approval is checked before imports/constructors/PC1 fitting and before outputs are created.
    manifest,manifest_identity,approval_identity = approval_gate()
    import numpy as np
    import torch
    states,adapter,adapter_class,head,paired,spatial = refs()
    del paired,spatial
    if (OUTPUT/'SOURCE_SEALED.json').exists():
        raise RuntimeError('Source build already sealed; do not overwrite')
    if not torch.cuda.is_available():
        raise RuntimeError('No implicit CPU training fallback')
    config = audit.read(REVIEW/'SOURCE_BUILD_CONFIG.json')
    runtime = config['runtime']
    actual_runtime = dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        gpu_driver=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip())
    for key,value in actual_runtime.items():
        if runtime[key] != value:
            raise RuntimeError(f'Reviewed runtime changed: {key}')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != runtime['CUBLAS_WORKSPACE_CONFIG']:
        raise RuntimeError('Reviewed CUBLAS workspace setting changed')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    if shutil.disk_usage(audit.WORK).free < 1024**3:
        raise RuntimeError('Insufficient source-build scratch space')
    WORK.mkdir(parents=True,exist_ok=True); OUTPUT.mkdir(parents=True,exist_ok=True)
    identity = dict(source_input_manifest=manifest_identity,training_approval=approval_identity,
                    config_sha256=manifest['config_sha256'],seed=SEED)
    audit.js(OUTPUT/'STATUS.json',dict(stage='SOURCE_BUILD_STARTED',time=audit.now(),identity=identity,target_access=False))
    cert = audit.read(CERT); fresh_only(cert)
    raw = np.load(cert['patch_features']['path'],mmap_mode='r')
    cls = np.load(cert['cls_features']['path'])
    images = csvread(cert['image_feature_manifest']['path'])
    image_y = np.asarray([int(r['image_label']) for r in images],np.int64)
    # PC1 consumes only the newly certified normal-image maps, not old auxiliary files.
    pc1path = WORK/'pc1.npy'
    normal_positions = np.flatnonzero(image_y == 0)
    if pc1path.exists():
        pc1meta = audit.read(WORK/'PC1_IDENTITY.json')
        audit.require_identity(pc1meta['file'])
        if pc1meta['identity'] != identity:
            raise RuntimeError('PC1 input identity changed')
        q = np.load(pc1path)
    else:
        total = np.zeros((1369,1024),np.float64)
        for pos in normal_positions:
            total += np.asarray(raw[pos],np.float32).astype(np.float64)
        mean = total/len(normal_positions)
        centered = mean-mean.mean(axis=0,keepdims=True)
        _,s,vt = np.linalg.svd(centered,full_matrices=False)
        q = vt[0].astype(np.float32); q /= max(float(np.linalg.norm(q)),1e-12)
        np.save(pc1path,q)
        np.save(WORK/'PC1_NORMAL_IMAGE_POSITIONS.npy',normal_positions)
        np.save(WORK/'PC1_SINGULAR_VALUES.npy',s)
        audit.js(WORK/'PC1_IDENTITY.json',dict(identity=identity,file=audit.ident(pc1path),normal_images=len(normal_positions)))
    pairs,adapters = {},{}
    for variant in ('E1','E2'):
        directory = REVIEW/'plans'/variant
        with np.load(directory/'POOL_PLAN.npz') as data:
            plan = {k:data[k] for k in data.files}
        x = pool_features(raw,plan,q,states.norm_np,per_image=True)
        y,cat = plan['labels'],plan['categories']
        stage_identity = dict(identity,pc1=audit.ident(pc1path),pool_plan=audit.ident(directory/'POOL_PLAN.npz'))
        img_pair,patch_pair = run_stage_I(variant,x,y,cat,cls,image_y,
            np.load(directory/'STAGE_I_PATCH_BATCHES.npy'),stage_identity,states)
        pairs[variant] = patch_pair
        with np.load(directory/'STAGE_II_BATCHES.npz') as data:
            ab = {k:data[k] for k in data.files}
        last_state = WORK/'checkpoints'/f'{variant}_stage_I'/'update_020.pt'
        adapter_identity = dict(stage_identity,stage_I_checkpoint=audit.ident(last_state))
        model = run_adapter(variant,x,y,patch_pair,ab,adapter_identity,states,adapter,adapter_class)
        adapters[variant] = model
        payload = WORK/'assets'/f'{variant}_formal.pt'
        if payload.exists():
            audit.require_identity(audit.read(payload.with_suffix('.sha256.json')))
            previous = torch.load(payload,map_location='cpu',weights_only=False)
            if previous['identity'] != adapter_identity or not np.array_equal(previous['patch_states'],patch_pair):
                raise RuntimeError('New-stage auxiliary asset identity mismatch')
            for key,value in model.state_dict().items():
                if not torch.equal(previous['model'][key],value.detach().cpu()):
                    raise RuntimeError('New-stage auxiliary adapter differs from checkpoint chain')
        else:
            save_torch(payload,dict(source='Full VisA',seed=0,identity=adapter_identity,
                patch_states=patch_pair,img_states=img_pair,model=model.state_dict(),
                supervision='true' if variant == 'E1' else 'shifted',stage_I_updates=20,
                stage_II_updates=len(ab['offsets'])-1,stage_II_epochs=10))
    with np.load(REVIEW/'plans/CES/POOL_PLAN.npz') as data:
        plan = {k:data[k] for k in data.files}
    ces_identity = dict(identity,pc1=audit.ident(pc1path),E1=audit.ident(WORK/'assets/E1_formal.pt'),
        E2=audit.ident(WORK/'assets/E2_formal.pt'),pool_plan=audit.ident(REVIEW/'plans/CES/POOL_PLAN.npz'))
    run_ces(raw,plan,q,pairs['E1'],pairs['E2'],adapters['E2'],ces_identity,states,adapter,head)
    # Seal both endpoints unconditionally. No target evaluation or endpoint choice occurs here.
    endpoints = dict(initialization=audit.ident(WORK/'checkpoints/CES/update_000.pt'),
                     update30=audit.ident(WORK/'checkpoints/CES/update_030.pt'))
    logs = []
    for p in sorted((WORK/'checkpoints').rglob('update_*.pt')):
        payload = torch.load(p,map_location='cpu',weights_only=False)
        logs.append(dict(stage=payload['stage'],step=payload['step'],checkpoint=str(p),
            actual_batch_hash=payload['actual_batch']['sha256'],**payload['losses']))
    # Use existing general CSV writer to preserve heterogeneous stage loss columns.
    from experiments.phase_ces_lhc_paired_repeat import write_csv
    write_csv(OUTPUT/'TRAINING_LOG.csv',logs)
    files = [audit.ident(p) for p in sorted(WORK.rglob('*')) if p.is_file()]
    files.append(audit.ident(OUTPUT/'TRAINING_LOG.csv'))
    approval_gate()  # Recheck inputs/code/approval before the complete-source seal.
    audit.js(OUTPUT/'SOURCE_SEALED.json',dict(created=audit.now(),complete=True,source='Full VisA',seed=0,
        identity=identity,updates=30,endpoints=endpoints,locked_files=files,target_access=False,evaluation_approved=False))
    audit.js(OUTPUT/'STATUS.json',dict(stage='SOURCE_SEALED_EVALUATION_LOCKED',time=audit.now(),training_complete=True,target_access=False))


def self_test():
    import unittest
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_traceable_visa_source_build.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    REVIEW.mkdir(parents=True,exist_ok=True)
    audit.js(REVIEW/'SOURCE_BUILD_GUARD_TESTS.json',dict(time=audit.now(),passed=result.wasSuccessful(),
        tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        runner_sha256=audit.sha(__file__),training_executed=False,optimizer_constructed=False,
        encoder_forward_executed=False,target_access=False))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('prepare','source-build','self-test'),required=True)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'source-build':
        source_build()
    else:
        self_test()
