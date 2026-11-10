"""Locked five-target evaluation of rebuilt VisA source; preparation is not release.

update30 is the trained main model. Initialization is a paired diagnostic, never
a candidate selected by target results. Both share every query's encoded views.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
import zipfile

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import rebuild_traceable_visa_source as audit
from experiments import source_build_traceable_visa as source

REVIEW = audit.OUT/'target_evaluation_review'
OUT = audit.OUT/'target_evaluation'
WORK = audit.WORK/'target_evaluation'
TARGETS = ('MVTec','BTAD','MPDD','DTD-Synthetic','DAGM')
ENDPOINTS = ('initialization','update30')
ROLES = {'initialization':'paired_diagnostic','update30':'trained_main_model'}
METRICS = ('I_AUROC','I_AP','P_AUROC','P_F1max','P_AP','AUPRO')
RUNTIME_REVIEW = Path('D:/ProjectPython/Paper/reviews/VisA_Source_Runtime_20261031')


def runtime():
    import numpy as np
    import torch
    import torchvision
    import scipy
    import sklearn
    import PIL
    return dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,torchvision=torchvision.__version__,
        scipy=scipy.__version__,sklearn=sklearn.__version__,pillow=PIL.__version__,cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),gpu_driver=subprocess.check_output(
            ['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip())


def crop_boxes(width,height):
    return [(0,0,width//2,height//2),(width//2,0,width,height//2),
            (0,height//2,width//2,height),(width//2,height//2,width,height)]


def fuse_views(q):
    """Input [endpoint, global/TL/TR/BL/BR,37,37], output [endpoint,74,74]."""
    import torch
    import torch.nn.functional as F
    if q.ndim != 4 or tuple(q.shape[1:]) != (5,37,37):
        raise RuntimeError('Wrong fixed-LRP view tensor')
    g = F.interpolate(q[:,0:1],size=(74,74),mode='bilinear',align_corners=True)[:,0]
    local = torch.empty_like(g)
    local[:,:37,:37]=q[:,1]; local[:,:37,37:]=q[:,2]
    local[:,37:,:37]=q[:,3]; local[:,37:,37:]=q[:,4]
    return .5*g+.5*local


def validate_roles(config):
    if tuple(config['endpoint_order']) != ENDPOINTS or config['endpoint_roles'] != ROLES:
        raise RuntimeError('Endpoint roles/order changed; no target-dependent model selection')
    if config['main_model'] != 'update30' or config['selection'] != 'NONE_REPORT_BOTH':
        raise RuntimeError('Trained main model must remain update30')


def identity_value(row,key):
    # Some frozen manifests omit this optional column when there are no notes.
    # All scientific identity fields remain mandatory; nonempty notes still match.
    return str(row.get(key,'')) if key == 'annotation_note' else str(row[key])


def gate(base=REVIEW):
    mp = base/'TARGET_EVALUATION_INPUT_MANIFEST.json'
    manifest = audit.read(mp)
    approval = audit.read(base/'TARGET_EVALUATION_APPROVAL.json')
    if approval.get('decision') != 'APPROVED_TARGET_EVALUATION_ONLY' or not approval.get('reviewer'):
        raise RuntimeError('Separate explicit target evaluation approval required')
    if approval.get('input_manifest_sha256') != audit.sha(mp):
        raise RuntimeError('Target approval input-manifest hash mismatch')
    if approval.get('config_sha256') != manifest['config_sha256']:
        raise RuntimeError('Target approval config hash mismatch')
    if approval.get('source_seal_sha256') != manifest['source_seal_sha256']:
        raise RuntimeError('Target approval source seal mismatch')
    if manifest['status'] != 'PREPARED_TARGET_EVALUATION_LOCKED':
        raise RuntimeError('Unrecognized target input manifest status')
    for item in manifest['locked_files']:
        audit.require_identity(item)
    config = audit.read(base/'TARGET_EVALUATION_CONFIG.json')
    validate_roles(config)
    return manifest,config,audit.ident(mp),audit.ident(base/'TARGET_EVALUATION_APPROVAL.json')


def cache_inventory():
    """Inspect headers/provenance availability only; NONE become execution inputs."""
    import numpy as np
    cache = Path('E:/MRAD_final_evidence_cache')
    paths = {
        'MVTec': [cache/'global_patch_MVTec.npy',cache/'local_patch_Full_VisA__MVTec.npy'],
        'BTAD': list((ROOT/'results/phase_memory_support_modes/array_cache').glob('BTAD_clip_*_patch.npy'))
                +[cache/'local_patch_Full_VisA__BTAD.npy'],
        'MPDD': [ROOT/'results/phase_ces_lhc_paired_repeat/shared_input_cache/MPDD_global_patch.npy',
                 cache/'local_patch_Full_MVTec__MPDD.npy'],
        'DTD-Synthetic': [ROOT/'results/phase_ces_geometry_center_control/formal/shared_input_cache/DTD_global_patch.npy',
                          cache/'local_patch_Full_VisA__DTD-Synthetic.npy'],
        'DAGM': [ROOT/'results/paper_assets/dagm_visa_eval/cache/dagm_global_patch_float16.npy',
                 cache/'local_patch_Full_VisA__DAGM.npy']}
    rows=[]
    for target,entries in paths.items():
        for path in entries:
            shape,dtype='UNKNOWN','UNKNOWN'
            if path.is_file():
                a=np.load(path,mmap_mode='r');shape=str(list(a.shape));dtype=str(a.dtype);del a
            provenance=ROOT/f'results/phase_ces_unified_ablation_final/formal/predictions/{target}/INPUT_IDENTITY.json'
            rows.append(dict(target=target,kind='local' if 'local_patch_' in path.name else 'global',
                candidate_path=str(path),exists=path.is_file(),bytes=path.stat().st_size if path.is_file() else 0,
                observed_shape=shape,observed_dtype=dtype,content_hash='NOT_REHASHED_NOT_CONSUMED',
                legacy_provenance=str(provenance) if provenance.is_file() else '',
                legacy_provenance_sha256=audit.sha(provenance) if provenance.is_file() else '',
                full_encoder_preprocess_precision_identity='UNAUTHENTICATED',
                full_physical_image_and_view_identity='UNAUTHENTICATED',
                certification='UNAUTHENTICATED' if path.is_file() else 'MISSING',
                reuse_approved=False,execution_action='FRESH_FROM_HASHED_ORIGINAL_IMAGES',
                note='Historical file/header/hash/smoke records do not certify all rows and all 4 local views under this locked encoder'))
    return rows


def prepare():
    from experiments import final_unified_evaluation as ev
    import numpy as np
    REVIEW.mkdir(parents=True,exist_ok=True)
    if (REVIEW/'TARGET_EVALUATION_INPUT_MANIFEST.json').exists():
        raise RuntimeError('Target preparation already sealed; refuse overwrite')
    print('PREPARE ONLY: NO ENCODER/SCORER FORWARD OR TARGET METRIC COMPUTATION',flush=True)
    seal_path=source.OUTPUT/'SOURCE_SEALED.json';seal=audit.read(seal_path)
    decision_path=RUNTIME_REVIEW/'REVIEW_DECISION.json';decision=audit.read(decision_path)
    if not decision['source_runtime_accepted'] or decision['source_seal_sha256'] != audit.sha(seal_path):
        raise RuntimeError('Accepted source runtime review does not match this source seal')
    if not seal['complete'] or seal['updates'] != 30 or seal['target_access'] or seal['evaluation_approved']:
        raise RuntimeError('Source seal not complete or wrong release scope')
    consumed=[audit.ident(seal_path),audit.ident(decision_path),audit.ident(RUNTIME_REVIEW/'REVIEW.md')]
    for entry in seal['locked_files']:
        audit.require_identity(entry)
    cert=audit.read(source.CERT)
    for key in ('base_CLIP_weights','ancillary_checkpoint','extractor_config','encoder_identity'):
        audit.require_identity(cert[key]);consumed.append(cert[key])
    consumed.append(audit.ident(source.CERT))
    source_inputs=dict(source_seal=audit.ident(seal_path),PC1=audit.ident(source.WORK/'pc1.npy'),
        E2=audit.ident(source.WORK/'assets/E2_formal.pt'),
        endpoints={name:seal['endpoints'][name] for name in ENDPOINTS})
    consumed.extend([source_inputs['PC1'],source_inputs['E2'],*source_inputs['endpoints'].values()])
    summary=[];physical=[];manifest_paths=[];max_pixels=0;total_images=0
    for target in TARGETS:
        original=ev.WORK/'manifests'/f'{target}.csv'
        recorded=ev.readcsv(original)
        # Re-enumeration is an identity/split/GT check, not a model evaluation.
        current=ev.catalogue(target)
        if len(recorded) != len(current):
            raise RuntimeError(f'Changed main-table test population: {target}')
        keys=('sample_id','category','label','image_sha256','gt_sha256','gt_height','gt_width','positive_pixels','annotation_note')
        for old,new in zip(recorded,current):
            for key in keys:
                if identity_value(old,key) != identity_value(new,key):
                    raise RuntimeError(f'Changed canonical target identity: {target}/{old["sample_id"]}/{key}')
            if ev.normpath(old['img_path']) != ev.normpath(new['img_path']):
                raise RuntimeError('Changed target image path')
            if old['gt_path'] and ev.normpath(old['gt_path']) != ev.normpath(new['gt_path']):
                raise RuntimeError('Changed paired original GT path')
            for key,hashkey in [('img_path','image_sha256'),('gt_path','gt_sha256')]:
                if old[key]:
                    p=Path(old[key]).resolve()
                    physical.append(dict(path=str(p),bytes=p.stat().st_size,sha256=old[hashkey]))
        destination=REVIEW/'manifests'/f'{target}.csv';destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(original,destination)
        manifest_paths.append(audit.ident(destination));consumed.append(audit.ident(original))
        counts=Counter()
        for row in recorded:
            counts[row['category']]+=int(row['gt_height'])*int(row['gt_width'])
        peak=max(counts.values());max_pixels=max(max_pixels,peak);total_images+=len(recorded)
        summary.append(dict(source='Full VisA',target=target,images=len(recorded),categories=len(counts),
            anomalous_images=sum(int(r['label']) for r in recorded),manifest=str(destination),
            manifest_sha256=audit.sha(destination),main_manifest_sha256=audit.sha(original),
            physical_image_GT_hashes='ALL_VERIFIED',max_category_pixels=peak,
            annotation_exceptions=sum(bool(r.get('annotation_note','')) for r in recorded),
            inference='FRESH_GLOBAL_AND_FOUR_LOCAL_VIEWS',endpoint_forward_sharing=True))
        print('TARGET IDENTITY VERIFIED',target,len(recorded),flush=True)
    ev.writecsv(REVIEW/'TARGET_INPUT_SUMMARY.csv',summary)
    ev.writecsv(REVIEW/'TARGET_CACHE_IDENTITY_AUDIT.csv',cache_inventory())
    raw_bytes=total_images*2*74*74*4
    budget=dict(images=total_images,views=total_images*5,raw_logits_bytes=raw_bytes,
        largest_category_native_pixels=max_pixels,peak_native_scratch_bytes=max_pixels*5,
        reserve_bytes=512*1024**2,required_free_bytes=raw_bytes+max_pixels*5+512*1024**2,
        current_E_free_bytes=shutil.disk_usage(audit.WORK).free,
        feature_cache_saved=False,compression_or_quantization=False)
    if budget['current_E_free_bytes'] < budget['required_free_bytes']:
        raise RuntimeError('Insufficient isolated E-drive prediction/scratch reserve; no old files deleted')
    config=dict(source='Full VisA',seed=0,targets=list(TARGETS),endpoint_order=list(ENDPOINTS),
        endpoint_roles=ROLES,main_model='update30',selection='NONE_REPORT_BOTH',
        source_inputs=source_inputs,target_manifests=manifest_paths,
        source_feature_certificate=audit.ident(source.CERT),runtime_acceptance=audit.ident(decision_path),
        feature_policy='FRESH_ONLY; no legacy target features/maps or source state fallback',
        encoder=dict(base=cert['base_CLIP_weights'],ancillary=cert['ancillary_checkpoint'],
            reference_identity=cert['encoder_identity'],model='CLIP ViT-L/14@336px with frozen DAPM_replace(24)',
            input=[518,518],feature_layer=24,feature_shape=[37,37,1024],neighbor_scales=[1,3,5],
            preprocessing='same get_transform and locked source extractor',dtype='float32',autocast=False,TF32=False,
            image_decode='PIL open then convert RGB before creating global/local views; fixed for grayscale inputs too',
            position='official bilinear 24x24->37x37, align_corners=False, preserve CLS row',
            batch_policy='global batch1 then local batch4; no OOM-driven batch fallback',
            shared='one global/local feature pair per image scored for BOTH endpoints in one matrix product'),
        scoring=dict(transform='remove same newly rebuilt PC1 -> NumPy L2 -> frozen rebuilt E2 adapt/L2',
            states='unit(m +/- alpha*unit(raw_v)); read exact source endpoint',
            score='unit aligned patch dot (sA-sN); no bias, sigmoid or independent temperature',
            views='global,TL,TR,BL,BR; boxes based on original integer width//2,height//2',
            fusion='0.5*bilinear(global37->74, align_corners=True)+0.5*stitched local quadrants',
            smoothing=None,normalization=None,output='float32 [N,2,74,74] raw fused logits'),
        evaluator=dict(module='experiments.final_unified_evaluation',hash=ev.evaluator_hash(),policy=ev.POLICY,
            functions=['native_mask','resize','topmean','weighted_pixel_metrics','aupro','macro_metrics'],
            metrics=list(METRICS),image='top55 mean on 74x74 BEFORE native-resolution resize',
            pixels='native paired GT; category-pooled native pixels; independently equal category macro per metric',
            transfer_mean='arithmetic mean of all FIVE target category-macro values, same seed; no partial mean',
            invalid='report null and valid category counts; never fabricate metrics',
            endpoint_delta='update30 minus initialization; descriptive, no checkpoint selection'),
        runtime=runtime(),threads=4,CUBLAS_WORKSPACE_CONFIG=':4096:8',deterministic=True,
        smoke='first two manifest images per target; shape/finite/CPU score parity maxabs<=2e-6; no metric selection',
        prediction_order='locked target order then canonical manifest order; no difficulty filtering',
        storage=budget,outputs=str(OUT),heavy_outputs=str(WORK),
        guard='separate TARGET_EVALUATION_APPROVAL.json; source-build approval is insufficient',
        evaluation_run=False,training=False)
    validate_roles(config);audit.js(REVIEW/'TARGET_EVALUATION_CONFIG.json',config)
    tests=audit.read(REVIEW/'TARGET_EVALUATION_GUARD_TESTS.json')
    if not tests['passed'] or tests['runner_sha256'] != audit.sha(__file__):
        raise RuntimeError('Current evaluation guard tests missing/stale')
    seeds=[Path(__file__),Path(source.__file__),Path(audit.__file__),Path(ev.__file__),
           ROOT/'tests/test_traceable_visa_target_evaluation.py',ROOT/'experiments/phase_x0a_mandatory_gates.py']
    for folder in ('AnomalyCLIP_lib','phase_x','phase_xa','models','utils'):
        seeds.extend((ROOT/folder).rglob('*.py'))
    codes=audit.code_inventory(seeds);code_ids=[audit.ident(p) for p in codes]
    audit.js(REVIEW/'EVALUATION_CODE_IDENTITY.json',code_ids)
    with zipfile.ZipFile(REVIEW/'EVALUATION_CODE_SNAPSHOT.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in codes:archive.write(p,str(p.relative_to(ROOT)))
    documentation=['# Rebuilt VisA five-target evaluation: implementation/input review','',
        '**PREPARED_TARGET_EVALUATION_LOCKED. No target forward pass or target metric was run.**','',
        'The sole trained main model is update30. Initialization is a paired diagnostic. Both endpoints are '
        'reported on every target/category/metric; neither target results nor the five-transfer mean can change their roles.','',
        '## Target inputs and cache decision','',
        'Canonical main-table test manifests, native paired masks, original image/GT byte hashes and exception rules '
        'were independently re-enumerated and matched. This preparation reads identity/annotation metadata only. '
        'Target category names are used only for manifest checking and macro evaluation, not encoder/prompts.','',
        'Every legacy target global/local candidate is UNAUTHENTICATED or MISSING. Headers and available historical '
        'provenance are recorded in TARGET_CACHE_IDENTITY_AUDIT.csv. Their content is not rehashed because none is '
        'an execution input. No full-row/view certificate matching the locked encoder is accepted for them. '
        'All five targets therefore use fresh inference from hashed original images; no silent reuse fallback exists.','',
        '## Fixed inference and metrics','',
        'Per image: one global forward and one 4-crop forward, shared by both heads. Original nonoverlapping '
        'quadrants are resized to518. New PC1 and frozen E2 adapter precede the two unit-state score. '
        'Global37 is resized to74 and fused .5/.5 with the stitched74 local map. Save only FP32 raw logits, '
        'the per-image view/preprocessing/map identities, and encoder/source hashes; no high-dimensional cache.','',
        'The existing frozen evaluator is called directly: native-resolution pixel category pooling and category macro '
        'P-AUROC/P-F1max/P-AP/AUPRO; image top1%=55 mean on74 before native resize and category macro I-AUROC/I-AP. '
        'AP ties, complete PR F1max, and AUPRO200 thresholds/8-connectivity/FPR.30 remain unchanged. '
        'Five-transfer means are produced only with all five cells; no single-seed standard deviation is invented.','',
        '## Release and resumption','',
        'No evaluation approval is generated. run requires APPROVED_TARGET_EVALUATION_ONLY, reviewer, exact '
        'input_manifest_sha256/config_sha256/source_seal_sha256. It verifies all source/encoder/code/target-file '
        'identities before constructing the encoder. Resume verifies committed raw-map row hashes, manifest/config '
        'identity, and completed category results. No old maps, changed crop count, batch-size fallback or retuning.','',
        'Resource checks reserve raw predictions plus the largest native scratch and512MiB. Only explicit '
        'new-run scratch files may be removed after their category is committed; all predictions persist. '
        'Input/source seal and code are rechecked before final result sealing.','',
        f'Images: {total_images}; total encoded views: {total_images*5}; raw logits bytes: {raw_bytes}; '
        f'largest native scratch bytes: {max_pixels*5}. These are capacity calculations, not timed benchmarks.','',
        'The unit tests cover gating, roles, crop/fusion geometry and preparation scope. They are not a target '
        'model smoke test or numerical reproduction. Actual inference/evaluator integration remains to be checked '
        'after separate release. Historical baselines, MVTec-to-VisA, old results and manuscript stay untouched.','']
    (REVIEW/'TARGET_EVALUATION_IMPLEMENTATION.md').write_text('\n'.join(documentation),encoding='utf-8')
    locked=consumed+physical+code_ids+[audit.ident(p) for p in REVIEW.rglob('*') if p.is_file()]
    locked=list({r['path']:r for r in locked}.values())
    manifest=dict(status='PREPARED_TARGET_EVALUATION_LOCKED',created=audit.now(),
        source_seal_sha256=audit.sha(seal_path),config_sha256=audit.sha(REVIEW/'TARGET_EVALUATION_CONFIG.json'),
        endpoint_roles=ROLES,main_model='update30',targets=list(TARGETS),images=total_images,
        locked_files=locked,target_evaluation_approved=False,target_inference_run=False,target_metrics_computed=False,
        approval_path=str(REVIEW/'TARGET_EVALUATION_APPROVAL.json'))
    audit.js(REVIEW/'TARGET_EVALUATION_INPUT_MANIFEST.json',manifest)
    print(json.dumps(dict(status=manifest['status'],images=total_images,views=total_images*5,
        roles=ROLES,legacy_target_caches_reused=0,input_manifest_sha256=audit.sha(REVIEW/'TARGET_EVALUATION_INPUT_MANIFEST.json'),
        config_sha256=manifest['config_sha256']),indent=2),flush=True)


def visual_hash(model):
    h=hashlib.sha256()
    for name,tensor in model.visual.state_dict().items():
        h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Encoder:
    def __init__(self,config):
        import torch
        import torch.nn.functional as F
        import AnomalyCLIP_lib.model_load as loader
        from experiments.phase_x0a_mandatory_gates import load_components
        from utils.transforms import get_transform
        base=config['encoder']['base'];ancillary=config['encoder']['ancillary']
        audit.require_identity(base);audit.require_identity(ancillary)
        original=loader._download
        def pinned(url,cache_dir=None):
            if url != loader._MODELS['ViT-L/14@336px']:
                raise RuntimeError('Unapproved encoder download/fallback')
            return base['path']
        loader._download=pinned
        try:self.model,*unused=load_components(ancillary['path'],'cuda')
        finally:loader._download=original
        del unused
        self.model.eval();assert all(not p.requires_grad for p in self.model.parameters())
        reference=audit.read(config['encoder']['reference_identity']['path'])
        assert visual_hash(self.model) == reference['visual_hash_before_position_setup']
        position=self.model.visual.positional_embedding
        with torch.inference_mode():
            assert position.shape == (577,1024)
            grid=position[1:].reshape(-1,24,24,1024).permute(0,3,1,2)
            grid=F.interpolate(grid,(37,37),mode='bilinear',align_corners=False)
            position.data=torch.cat([position[:1],grid.reshape(-1,1024,1369).transpose(1,2)[0]],dim=0)
        self.before=visual_hash(self.model)
        assert self.before == reference['visual_hash_after_position_setup']
        assert str(self.model.dtype) == reference['model_dtype'] == 'torch.float32'
        self.preprocess=get_transform(argparse.Namespace(image_size=518))[0]

    def extract(self,row):
        import numpy as np
        import torch
        from PIL import Image
        from phase_xa.official_task_forward import visual_outputs,normalized_patch_feature
        content=Path(row['img_path']).read_bytes()
        if hashlib.sha256(content).hexdigest() != row['image_sha256']:
            raise RuntimeError('Target query image changed')
        with Image.open(io.BytesIO(content)) as image:
            image=image.convert('RGB');w,h=image.size;boxes=crop_boxes(w,h)
            views=[image]+[image.crop(b) for b in boxes]
            tensors=torch.stack([self.preprocess(im) for im in views])
        outputs=[]
        with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
            for batch in (tensors[:1],tensors[1:]):
                out=visual_outputs(self.model,batch.to('cuda'),adapter=None,grad=False)
                outputs.append(normalized_patch_feature(out).float().cpu().numpy())
        raw=np.concatenate(outputs)
        assert raw.shape == (5,1369,1024) and np.isfinite(raw).all()
        return raw,dict(crop_boxes=boxes,view_order=['global','TL','TR','BL','BR'],
            tensor_sha256=[hashlib.sha256(t.numpy().tobytes()).hexdigest() for t in tensors],
            raw_feature_sha256=hashlib.sha256(raw.tobytes()).hexdigest())


class Scorer:
    def __init__(self,config):
        import numpy as np
        import torch
        import torch.nn.functional as F
        from experiments.phase_xliii_adapter_head_decomposition import AdapterToState
        inp=config['source_inputs']
        self.q=np.load(inp['PC1']['path']).astype(np.float32)
        e2=torch.load(inp['E2']['path'],map_location='cpu',weights_only=False)
        self.adapter=AdapterToState(1024,e2['patch_states']).to('cuda')
        self.adapter.load_state_dict(e2['model'],strict=True);self.adapter.eval()
        for p in self.adapter.parameters():p.requires_grad_(False)
        weights=[]
        for name in ENDPOINTS:
            saved=torch.load(inp['endpoints'][name]['path'],map_location='cpu',weights_only=False)
            assert saved['step'] == (0 if name == 'initialization' else 30)
            state=saved['params']['raw_model']
            m,alpha,v=[state[k].to('cuda') for k in ('m','alpha','v')]
            v=F.normalize(v,dim=0)
            weights.append(F.normalize(m+alpha*v,dim=0)-F.normalize(m-alpha*v,dim=0))
        self.weights=torch.stack(weights,dim=1)
        self.frozen=self.checksum()

    def checksum(self):
        h=hashlib.sha256(self.q.tobytes()+self.weights.detach().cpu().numpy().tobytes())
        for name,tensor in self.adapter.state_dict().items():
            h.update(name.encode());h.update(tensor.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def __call__(self,raw):
        import numpy as np
        import torch
        z=np.asarray(raw,np.float32).reshape(-1,1024)
        z=z-(z@self.q)[:,None]*self.q
        z=z/np.maximum(np.linalg.norm(z,axis=-1,keepdims=True),1e-12)
        with torch.inference_mode():
            aligned=self.adapter.adapt(torch.tensor(z,device='cuda'))
            q=(aligned@self.weights).reshape(5,37,37,2).permute(3,0,1,2)
            maps=fuse_views(q).cpu().numpy()
        assert maps.shape == (2,74,74) and np.isfinite(maps).all()
        return maps

    def cpu_parity(self,raw,maps):
        """First-two-image math smoke check, not a target performance test."""
        import numpy as np
        import torch
        state=self.adapter.state_dict()
        z=np.asarray(raw,np.float32).reshape(-1,1024)
        z=z-(z@self.q)[:,None]*self.q
        z=z/np.maximum(np.linalg.norm(z,axis=-1,keepdims=True),1e-12)
        adapted=z+(z@state['a.weight'].cpu().numpy().T)@state['b.weight'].cpu().numpy().T
        adapted=adapted/np.maximum(np.linalg.norm(adapted,axis=-1,keepdims=True),1e-12)
        q=(adapted@self.weights.cpu().numpy()).reshape(5,37,37,2).transpose(3,0,1,2).copy()
        reference=fuse_views(torch.from_numpy(q)).numpy()
        error=float(np.max(abs(reference-maps)))
        if error > 2e-6:
            raise RuntimeError(f'Locked score/fusion CPU parity smoke failed: {error}')
        return error


def identity(context,path,kind):
    return dict(context=context,kind=kind,file=audit.ident(path))


def predict(target,config,context,encoder,scorer):
    import numpy as np
    from experiments import final_unified_evaluation as ev
    rows=ev.readcsv(REVIEW/'manifests'/f'{target}.csv')
    dest=WORK/'predictions'/target;dest.mkdir(parents=True,exist_ok=True)
    stamp=dict(context=context,manifest=audit.ident(REVIEW/'manifests'/f'{target}.csv'),
        encoder_hash=encoder.before,scorer_hash=scorer.frozen,endpoint_order=list(ENDPOINTS))
    complete=dest/'COMPLETE.json'
    if complete.exists():
        done=audit.read(complete);assert done['stamp'] == stamp
        audit.require_identity(done['maps']);audit.require_identity(done['row_journal'])
        return done
    path=dest/'maps74.npy';journal=dest/'ROWS.jsonl'
    if path.exists() != journal.exists():
        raise RuntimeError('Uncommitted target allocation/journal; preserve and diagnose')
    new=not path.exists()
    if new:journal.touch(exist_ok=False)
    maps=np.lib.format.open_memmap(path,mode='w+' if new else 'r+',dtype=np.float32,shape=(len(rows),2,74,74))
    assert maps.shape == (len(rows),2,74,74) and maps.dtype == np.float32
    saved=[]
    for line in journal.read_text(encoding='utf-8').splitlines():
        record=json.loads(line);i=record['index'];assert i == len(saved) and record['stamp'] == stamp
        assert record['sample_id'] == rows[i]['sample_id'] and record['image_sha256'] == rows[i]['image_sha256']
        assert record['map_sha256'] == hashlib.sha256(maps[i].tobytes()).hexdigest()
        saved.append(record)
    with journal.open('a',encoding='utf-8') as f:
        for i in range(len(saved),len(rows)):
            raw,views=encoder.extract(rows[i]);maps[i]=scorer(raw);maps.flush()
            smoke=scorer.cpu_parity(raw,np.asarray(maps[i])) if i < 2 else None
            record=dict(index=i,sample_id=rows[i]['sample_id'],image_sha256=rows[i]['image_sha256'],
                smoke_cpu_parity_max_abs=smoke,
                stamp=stamp,views=views,map_sha256=hashlib.sha256(maps[i].tobytes()).hexdigest())
            f.write(json.dumps(record)+'\n');f.flush()
            if (i+1)%25 == 0 or i+1 == len(rows):
                audit.js(OUT/'STATUS.json',dict(stage='TARGET_PREDICTION',target=target,done=i+1,total=len(rows),time=audit.now()))
                print('PREDICT',target,i+1,len(rows),flush=True)
    assert visual_hash(encoder.model) == encoder.before and scorer.checksum() == scorer.frozen
    done=dict(stamp=stamp,maps=audit.ident(path),row_journal=audit.ident(journal),images=len(rows))
    smoke_rows=[json.loads(line) for line in journal.read_text(encoding='utf-8').splitlines()[:2]]
    audit.js(dest/'SMOKE_CHECKS.json',dict(stamp=stamp,checks=[dict(sample_id=r['sample_id'],
        cpu_score_parity_max_abs=r['smoke_cpu_parity_max_abs'],views=r['views']) for r in smoke_rows]))
    done['smoke_checks']=audit.ident(dest/'SMOKE_CHECKS.json')
    audit.js(complete,done)
    return done


def evaluate(target,config,context,prediction):
    import numpy as np
    from sklearn.metrics import roc_auc_score,average_precision_score
    from experiments import final_unified_evaluation as ev
    rows=ev.readcsv(REVIEW/'manifests'/f'{target}.csv')
    maps=np.load(prediction['maps']['path'],mmap_mode='r')
    destination=OUT/'evaluations'/target;destination.mkdir(parents=True,exist_ok=True)
    scratch=WORK/'scratch';scratch.mkdir(parents=True,exist_ok=True)
    all_categories=[]
    for column,name in enumerate(ENDPOINTS):
        image_scores=[dict(sample_id=r['sample_id'],category=r['category'],label=int(r['label']),
                           score=ev.topmean(maps[i,column])) for i,r in enumerate(rows)]
        ev.writecsv(destination/f'{name}_IMAGE_SCORES.csv',image_scores)
        for category in sorted({r['category'] for r in rows}):
            result_path=destination/f'{category}__{name}.json'
            stamp=dict(context=context,prediction_sha256=prediction['maps']['sha256'],endpoint=name,category=category)
            if result_path.exists():
                result=audit.read(result_path);assert result['stamp'] == stamp
                all_categories.append(result);continue
            ids=[i for i,r in enumerate(rows) if r['category'] == category]
            n=sum(int(rows[i]['gt_height'])*int(rows[i]['gt_width']) for i in ids)
            if shutil.disk_usage(WORK).free < n*5+512*1024**2:
                raise RuntimeError('Insufficient isolated native scratch reserve')
            yp,sp=scratch/'labels.npy',scratch/'scores.npy'
            if yp.exists() or sp.exists():
                raise RuntimeError('Uncommitted prior scratch exists; preserve/diagnose before resumption')
            y=np.lib.format.open_memmap(yp,mode='w+',dtype=np.bool_,shape=(n,))
            s=np.lib.format.open_memmap(sp,mode='w+',dtype=np.float32,shape=(n,))
            masks,scores=[],[];offset=0
            for i in ids:
                if rows[i]['gt_path'] and audit.sha(rows[i]['gt_path']) != rows[i]['gt_sha256']:
                    raise RuntimeError('Paired target GT changed')
                mask=ev.native_mask(rows[i]);a=ev.resize(maps[i,column],mask.shape)
                end=offset+mask.size;y[offset:end]=mask.ravel();s[offset:end]=a.ravel()
                masks.append(y[offset:end].reshape(mask.shape));scores.append(s[offset:end].reshape(mask.shape));offset=end
            y.flush();s.flush();assert offset == n
            print('METRICS',target,category,name,n,flush=True)
            audit.js(OUT/'STATUS.json',dict(stage='PIXEL_CATEGORY',target=target,category=category,
                endpoint=name,pixels=n,time=audit.now()))
            metrics=ev.weighted_pixel_metrics(y,s)
            audit.js(OUT/'STATUS.json',dict(stage='AUPRO_CATEGORY',target=target,category=category,
                endpoint=name,pixels=n,time=audit.now()))
            metrics['AUPRO']=ev.aupro(masks,scores)
            image_y=[image_scores[i]['label'] for i in ids];image_s=[image_scores[i]['score'] for i in ids]
            valid=len(set(image_y)) == 2
            metrics.update(I_AUROC=float(roc_auc_score(image_y,image_s)*100) if valid else None,
                           I_AP=float(average_precision_score(image_y,image_s)*100) if valid else None)
            result=dict(source='Full VisA',target=target,seed=0,endpoint=name,role=ROLES[name],category=category,
                images=len(ids),native_pixels=n,stamp=stamp,**metrics)
            audit.js(result_path,result);all_categories.append(result)
            masks.clear();scores.clear();del y,s,mask,a;gc.collect()
            # Only these two explicit, new-run-owned scratch files are removed.
            for p in (yp,sp):
                assert p.resolve().parent == scratch.resolve() and p.is_file()
                p.unlink()
    ev.writecsv(destination/'CATEGORY_METRICS.csv',all_categories)
    return all_categories


def run():
    # No encoder/scorer import or target metric runs before this independent gate.
    manifest,config,manifest_id,approval_id=gate()
    import numpy as np
    import torch
    from experiments import final_unified_evaluation as ev
    if runtime() != config['runtime'] or ev.evaluator_hash() != config['evaluator']['hash']:
        raise RuntimeError('Reviewed runtime/evaluator changed')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != config['CUBLAS_WORKSPACE_CONFIG']:
        raise RuntimeError('Reviewed CUBLAS workspace changed')
    if not torch.cuda.is_available():raise RuntimeError('No implicit CPU inference fallback')
    OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    if (OUT/'EVALUATION_SEALED.json').exists():raise RuntimeError('Evaluation already sealed; no overwrite')
    if shutil.disk_usage(WORK).free < config['storage']['required_free_bytes']:
        raise RuntimeError('Insufficient full prediction/scratch disk budget')
    torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    context=dict(input_manifest=manifest_id,approval=approval_id,config_sha256=manifest['config_sha256'],
                 source_seal_sha256=manifest['source_seal_sha256'])
    encoder=Encoder(config);scorer=Scorer(config)
    categories=[];summaries=[];predictions=[]
    for target in TARGETS:
        predicted=predict(target,config,context,encoder,scorer);predictions.append(predicted)
        rr=evaluate(target,config,context,predicted);categories.extend(rr)
        for name in ENDPOINTS:
            subset=[r for r in rr if r['endpoint'] == name]
            macro=ev.macro_metrics(subset,METRICS)
            summaries.append(dict(source='Full VisA',target=target,seed=0,endpoint=name,role=ROLES[name],
                checkpoint=config['source_inputs']['endpoints'][name]['path'],
                checkpoint_sha256=config['source_inputs']['endpoints'][name]['sha256'],**macro))
    for name in ENDPOINTS:
        cells=[r for r in summaries if r['endpoint'] == name]
        assert len(cells) == 5
        row=dict(source='Full VisA',target='Five-transfer mean',seed=0,endpoint=name,role=ROLES[name])
        for metric in METRICS:
            values=[r[metric] for r in cells]
            row[metric]=float(np.mean(values)) if all(v is not None for v in values) else None
        summaries.append(row)
    deltas=[]
    for target in (*TARGETS,'Five-transfer mean'):
        a=next(r for r in summaries if r['target'] == target and r['endpoint'] == 'initialization')
        b=next(r for r in summaries if r['target'] == target and r['endpoint'] == 'update30')
        deltas.append(dict(target=target,delta='update30_minus_initialization',
            **{metric:b[metric]-a[metric] if a[metric] is not None and b[metric] is not None else None for metric in METRICS}))
    ev.writecsv(OUT/'CATEGORY_METRICS.csv',categories);ev.writecsv(OUT/'TARGET_METRICS.csv',summaries)
    ev.writecsv(OUT/'PAIRED_ENDPOINT_DELTAS.csv',deltas)
    report=['# Rebuilt VisA source: fixed five-target evaluation','',
        'update30 is the trained main model; initialization is a paired diagnostic. '
        'Both endpoints are reported without target-driven selection. All numbers are category-macro percentages; '
        'this is seed0 only, with no invented uncertainty.','',
        '| Target | Endpoint | Role | '+' | '.join(METRICS)+' |',
        '|---|---|---|'+'|'.join(['---:']*len(METRICS))+'|']
    for row in summaries:
        values=['N/A' if row[m] is None else f'{row[m]:.6f}' for m in METRICS]
        report.append('| '+row['target']+' | '+row['endpoint']+' | '+row['role']+' | '+' | '.join(values)+' |')
    report.extend(['','All five targets used newly encoded original-image views shared by both heads. '
        'No historical prediction/feature cache or previous-source state was reused. '
        'The input manifests retain original GT, category definitions and annotation exceptions. '
        'Per-metric valid category counts are retained in TARGET_METRICS.csv; CATEGORY_METRICS.csv retains all categories.','',
        'PAIRED_ENDPOINT_DELTAS.csv reports update30 minus initialization, not a model-selection rule. '
        'The source checkpoint, encoder and evaluator identities are bound in EVALUATION_SEALED.json. '
        'Main-paper tables, historical baselines and MVTec-to-VisA were not modified.',''])
    (OUT/'TARGET_EVALUATION_REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    assert visual_hash(encoder.model) == encoder.before and scorer.checksum() == scorer.frozen
    gate()  # Recheck complete source/code/query/GT identity before sealing results.
    audit.js(OUT/'EVALUATION_SEALED.json',dict(complete=True,time=audit.now(),context=context,
        endpoint_roles=ROLES,selection='NONE_REPORT_BOTH',predictions=predictions,
        result_files=[audit.ident(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ('STATUS.json','EVALUATION_SEALED.json')],
        encoder_hash=encoder.before,scorer_hash=scorer.frozen,training=False))
    audit.js(OUT/'STATUS.json',dict(stage='COMPLETE',time=audit.now(),targets=5,endpoints=2))


def self_test():
    if (REVIEW/'TARGET_EVALUATION_INPUT_MANIFEST.json').exists():
        raise RuntimeError('Tests are sealed; use read-only unittest to repeat, do not overwrite evidence')
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_traceable_visa_target_evaluation.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    REVIEW.mkdir(parents=True,exist_ok=True)
    audit.js(REVIEW/'TARGET_EVALUATION_GUARD_TESTS.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,
        runner_sha256=audit.sha(__file__),model_forward=False,target_metric_computation=False,training=False))
    if not result.wasSuccessful():raise SystemExit(1)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('prepare','run','self-test'),required=True)
    args=parser.parse_args()
    if args.stage == 'prepare':prepare()
    elif args.stage == 'run':run()
    else:self_test()
