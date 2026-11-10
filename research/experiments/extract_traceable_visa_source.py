"""Re-extract source-only VisA patch/CLS features; never fit or evaluate a head.

The original preflight is immutable. A new source-feature certificate references
its hash. Old features are diagnostics only, never accepted as extraction input.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import rebuild_traceable_visa_source as audit

OUT = audit.OUT / 'source_feature_certification'
WORK = audit.WORK / 'certified_source'
BASE = Path('C:/Users/Administrator/.cache/clip/ViT-L-14-336px.pt')
BASE_SHA = '3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02'


def digest(array):
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()


def lock(path, value):
    if path.exists():
        if audit.read(path) != value:
            raise RuntimeError(f'Locked extraction identity changed: {path}')
    else:
        audit.js(path, value)


def status(stage, **values):
    audit.js(OUT / 'STATUS.json', dict(stage=stage, time=audit.now(), training_started=False,
                                     target_inference_started=False, **values))
    print(stage, json.dumps(values), flush=True)


def visual_digest(model):
    h = hashlib.sha256()
    for name, value in model.visual.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run():
    import numpy as np
    import PIL
    from PIL import Image
    import torch
    import torch.nn.functional as F
    import torchvision
    import AnomalyCLIP_lib.model_load as clip_loader
    from experiments.phase_x0a_mandatory_gates import load_components
    from experiments.phase_xlv_cross_domain_transfer import CHECKPOINT
    from phase_xa.official_task_forward import visual_outputs, normalized_patch_feature, normalized_image_feature
    from utils.transforms import get_transform

    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    if (OUT / 'SOURCE_FEATURE_CERTIFICATE.json').exists():
        raise RuntimeError('Source certificate already sealed; no overwrite allowed')
    status('CHECKING_SOURCE_INPUTS')
    preflight_path = audit.OUT / 'PREFLIGHT_MANIFEST.json'
    preflight_sha = audit.sha(preflight_path)
    if audit.read(audit.OUT / 'STATUS.json')['manifest_sha256'] != preflight_sha:
        raise RuntimeError('Original preflight seal does not match')
    preflight = audit.read(preflight_path)
    by_path = {str(Path(r['path']).resolve()): r for r in preflight['locked_files']}
    image_manifest = audit.OUT / 'SOURCE_IMAGE_MANIFEST.csv'
    audit.require_identity(by_path[str(image_manifest.resolve())])
    with image_manifest.open(encoding='utf-8-sig', newline='') as stream:
        images = list(csv.DictReader(stream))
    assert len(images) == 2162
    assert [int(r['source_position']) for r in images] == list(range(2162))
    assert len({r['image'] for r in images}) == len(images)
    base_identity = audit.ident(BASE)
    if base_identity['sha256'] != BASE_SHA:
        raise RuntimeError('Actual CLIP base file is not the locked official SHA256')
    ancillary_identity = audit.ident(CHECKPOINT)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError('No implicit CPU/precision fallback')
    driver = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], text=True).strip()
    preprocess = get_transform(argparse.Namespace(image_size=518))[0]
    # Include relative-import implementation files, not just an absolute import closure.
    seeds = [Path(__file__), Path(audit.__file__)]
    for folder in ('AnomalyCLIP_lib', 'phase_x', 'phase_xa', 'models', 'utils'):
        seeds.extend((ROOT / folder).rglob('*.py'))
    seeds.extend([ROOT / 'experiments/phase_x0a_mandatory_gates.py',
                  ROOT / 'experiments/phase_xlv_cross_domain_transfer.py'])
    codes = [audit.ident(path) for path in audit.code_inventory(seeds)]
    config = dict(source='Full VisA', seed=0, original_preflight=dict(path=str(preflight_path), sha256=preflight_sha),
        source_image_manifest=audit.ident(image_manifest), images=2162, patches_per_image=1369,
        base_CLIP_weights=base_identity,
        ancillary_checkpoint=dict(**ancillary_identity, purpose='load_components prompt/projection modules; NOT CLIP base weights; not used by visual readout'),
        extractor='phase_xa.official_task_forward.visual_outputs, normalized_patch_feature, normalized_image_feature',
        model='AnomalyCLIP_lib ViT-L/14@336px, frozen DAPM_replace(DPAM_layer=24)',
        forward=dict(layers=[6,12,18,24], selected_patch_layer=24, adapter=None, ori_patch=False,
                     proj_use=True, ffn=False, neighbor_scales=[1,3,5]),
        patch='unit average_neighbor(outputs[1][3]); [1369,1024], BEFORE PC1 or source adapter',
        CLS='unit outputs[0], [768]; projected original-stream CLS exactly as historical extract_clip_one',
        preprocessing=dict(repr=repr(preprocess), RGB='require real RGB input; no forced channel reinterpretation',
                           image_size=[518,518], augmentation=None),
        position='official bilinear 24x24 ->37x37 align_corners=False; retain CLS row; once before frozen checksum',
        precision=dict(storage='float32 patch and CLS', model='native load_components dtype recorded in ENCODER_IDENTITY.json',
                       autocast=False, TF32=False, batch_size=1, deterministic=True),
        endpoints='feature extraction only; no PC1 fit, states, adapter, CES, target model or metrics',
        old_cache_policy='diagnostics only; differences are NOT a certification failure or tuning criterion',
        runtime=dict(python=sys.version, torch=torch.__version__, torchvision=torchvision.__version__, numpy=np.__version__,
                     pillow=PIL.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                     gpu=torch.cuda.get_device_name(0), driver=driver), code=codes)
    lock(OUT / 'EXTRACTION_CONFIG.json', config)
    config_sha = audit.sha(OUT / 'EXTRACTION_CONFIG.json')
    snapshot = WORK / 'EXTRACTION_CODE_SNAPSHOT.zip'
    if not snapshot.exists():
        with zipfile.ZipFile(snapshot, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for entry in codes:
                path = Path(entry['path'])
                archive.write(path, str(path.relative_to(ROOT)))

    # Forbid a network fetch or an alternative implicit CLIP base file.
    original_download = clip_loader._download
    def local_base_only(url, cache_dir=None):
        if url != clip_loader._MODELS['ViT-L/14@336px']:
            raise RuntimeError(f'Unapproved model download requested: {url}')
        audit.require_identity(base_identity)
        return str(BASE)
    clip_loader._download = local_base_only
    try:
        model, *unused = load_components(str(CHECKPOINT), 'cuda')
    finally:
        clip_loader._download = original_download
    del unused
    model.eval()
    assert all(not p.requires_grad for p in model.parameters())
    original_visual = visual_digest(model)
    position = model.visual.positional_embedding
    before_position = digest(position.detach().cpu().numpy())
    with torch.inference_mode():
        assert position.shape == (577,1024), position.shape
        grid = position[1:].reshape(-1,24,24,1024).permute(0,3,1,2)
        resized = F.interpolate(grid, (37,37), mode='bilinear', align_corners=False)
        resized = resized.reshape(-1,1024,1369).transpose(1,2)
        position.data = torch.cat([position[:1],resized[0]],dim=0)
    before = visual_digest(model)
    encoder = dict(base_file=base_identity, ancillary_file=ancillary_identity,
        visual_hash_before_position_setup=original_visual, visual_hash_after_position_setup=before,
        position_before=before_position, position_after=digest(position.detach().cpu().numpy()),
        position_operation=config['position'], model_dtype=str(model.dtype),
        visual_parameter_dtypes=dict(Counter(str(p.dtype) for p in model.visual.parameters())),
        all_parameters_frozen=True, preprocess_tensor_dtype='torch.float32',
        input_to_visual_dtype=str(model.dtype), autocast=False, config_sha256=config_sha)
    lock(OUT / 'ENCODER_IDENTITY.json', encoder)
    encoder_sha = audit.sha(OUT / 'ENCODER_IDENTITY.json')
    patch_path, cls_path = WORK / 'source_patch_float32.npy', WORK / 'source_cls_float32.npy'
    required = 2162*(1369*1024+768)*4 + 2*1024**3
    if not patch_path.exists() and shutil.disk_usage(WORK).free < required:
        raise RuntimeError(f'Insufficient E-drive space: require outputs plus 2 GiB reserve ({required} bytes)')
    if patch_path.exists() != cls_path.exists():
        raise RuntimeError('Incomplete allocation; preserve evidence, do not silently recreate')
    mode = 'r+' if patch_path.exists() else 'w+'
    patches = np.lib.format.open_memmap(patch_path, mode=mode, dtype=np.float32, shape=(2162,1369,1024))
    cls = np.lib.format.open_memmap(cls_path, mode=mode, dtype=np.float32, shape=(2162,768))
    assert patches.shape == (2162,1369,1024) and cls.shape == (2162,768)
    journal = OUT / 'EXTRACTION_ROWS.jsonl'
    completed = []
    if journal.exists():
        for line in journal.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            p = int(row['source_position'])
            assert p == len(completed) and row['config_sha256'] == config_sha
            assert row['patch_sha256'] == digest(patches[p]) and row['cls_sha256'] == digest(cls[p])
            assert row['image_sha256'] == images[p]['image_sha256']
            completed.append(row)
    status('EXTRACTING_SOURCE_ONLY', completed=len(completed), total=2162, model_dtype=str(model.dtype))
    started = time.monotonic()
    with journal.open('a', encoding='utf-8') as stream:
        for pos in range(len(completed), len(images)):
            row = images[pos]
            content = Path(row['image']).read_bytes()
            if hashlib.sha256(content).hexdigest() != row['image_sha256']:
                raise RuntimeError(f'Physical source image changed: {row["image"]}')
            if row['mask'] and audit.sha(row['mask']) != row['mask_sha256']:
                raise RuntimeError(f'Physical source mask changed: {row["mask"]}')
            with Image.open(io.BytesIO(content)) as im:
                assert im.mode == 'RGB', (row['image'], im.mode)
                tensor = preprocess(im).unsqueeze(0)
            tensor_sha = digest(tensor.numpy())
            with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
                outputs = visual_outputs(model, tensor.to('cuda'), adapter=None, grad=False)
                pf = normalized_patch_feature(outputs).squeeze(0).float().cpu().numpy()
                cf = normalized_image_feature(outputs).squeeze(0).float().cpu().numpy()
            assert pf.shape == (1369,1024) and cf.shape == (768,)
            assert np.isfinite(pf).all() and np.isfinite(cf).all()
            patch_norm = np.linalg.norm(pf, axis=1)
            assert np.max(abs(patch_norm-1)) < .005 and abs(np.linalg.norm(cf)-1) < .005
            patches[pos], cls[pos] = pf, cf
            # Flush feature bytes before committing the corresponding identity row.
            patches.flush(); cls.flush()
            record = dict(source_position=pos, dataset_index=int(row['dataset_index']), category=row['category'],
                image=row['image'], image_sha256=row['image_sha256'], mask=row['mask'], mask_sha256=row['mask_sha256'],
                image_label=int(row['image_label']), positive_patches=int(row['positive_patches']),
                preprocessing_tensor_sha256=tensor_sha, preprocessing_tensor_shape=list(tensor.shape),
                preprocessing_tensor_dtype=str(tensor.dtype), encoder_identity_sha256=encoder_sha,
                config_sha256=config_sha, patch_feature_row=pos, cls_feature_row=pos,
                patch_sha256=digest(pf), cls_sha256=digest(cf), patch_norm_min=float(patch_norm.min()),
                patch_norm_max=float(patch_norm.max()), cls_norm=float(np.linalg.norm(cf)),
                feature_dtype=str(pf.dtype), status='FRESH_FROM_HASHED_PHYSICAL_IMAGE')
            stream.write(json.dumps(record)+'\n'); stream.flush()
            completed.append(record)
            if (pos+1) % 25 == 0 or pos < 2:
                elapsed = time.monotonic()-started
                status('EXTRACTING_SOURCE_ONLY', completed=pos+1, total=2162, elapsed_seconds=round(elapsed,1),
                       E_free_bytes=shutil.disk_usage(WORK).free)
            if shutil.disk_usage(WORK).free < 512*1024**2:
                raise RuntimeError('E-drive free space below 512 MiB safety floor')
    assert len(completed) == 2162
    status('VERIFYING_FRESH_OUTPUTS', completed=2162)
    rows = []
    for pos, record in enumerate(completed):
        assert record['patch_sha256'] == digest(patches[pos])
        assert record['cls_sha256'] == digest(cls[pos])
        rows.append({k:json.dumps(v) if isinstance(v,list) else v for k,v in record.items()})
    audit.csvout(OUT / 'IMAGE_FEATURE_MANIFEST.csv', rows)
    if visual_digest(model) != before:
        raise RuntimeError('Frozen visual parameters changed after extraction')
    assert all(not p.requires_grad for p in model.parameters())
    # Diagnostics cannot fail certification just because old numeric values differ.
    old_path = Path('E:/MRAD_final_evidence_cache/global_patch_VisA.npy')
    old = np.load(old_path, mmap_mode='r')
    assert old.shape == patches.shape
    diagnostics = []
    for pos in range(2162):
        fresh, prior = np.asarray(patches[pos]), np.asarray(old[pos],np.float32)
        diff = fresh-prior
        diagnostics.append(dict(source_position=pos, category=images[pos]['category'],
            patch_max_abs=float(np.max(abs(diff))), patch_mean_abs=float(np.mean(abs(diff))),
            patch_rmse=float(np.sqrt(np.mean(diff*diff))),
            values_different_float32=int(np.count_nonzero(fresh != prior)),
            values_different_after_float16_cast=int(np.count_nonzero(fresh.astype(np.float16) != old[pos])),
            used_for_acceptance=False))
    audit.csvout(OUT / 'OLD_CACHE_DIFFERENCES.csv', diagnostics)
    old_identity = audit.ident(old_path)
    labels_path = audit.WORK / 'preflight/true_labels_nearest_id_union.npy'
    audit.require_identity(by_path[str(labels_path.resolve())])
    labels = np.load(labels_path)
    source_patches = audit.WORK / 'preflight/SOURCE_PATCH_MANIFEST.csv.gz'
    audit.require_identity(by_path[str(source_patches.resolve())])
    binding_path = WORK / 'PATCH_FEATURE_BINDING.csv.gz'
    count = 0
    with gzip.open(source_patches, 'rt', newline='', encoding='utf-8') as src:
        with gzip.open(binding_path, 'wt', newline='', encoding='utf-8') as dst:
            writer = csv.writer(dst)
            writer.writerow(['patch_id','image_ledger_row','patch_feature_row','patch_flat_index','patch_y','patch_x','label'])
            for record in csv.DictReader(src):
                pos, pid = int(record['source_position']), int(record['patch_flat_index'])
                assert count == pos*1369+pid and int(record['label']) == int(labels[pos,pid])
                assert record['patch_id'] == f'visa:{pos}:{pid}'
                writer.writerow([record['patch_id'],pos,pos,pid,pid//37,pid%37,int(labels[pos,pid])])
                count += 1
    assert count == 2162*1369
    patch_identity = audit.ident(patch_path)
    cls_identity = audit.ident(cls_path)
    # All consumed source bytes and every code file are rechecked before sealing.
    for entry in codes + [base_identity, ancillary_identity]:
        audit.require_identity(entry)
    for row in images:
        assert audit.sha(row['image']) == row['image_sha256']
        if row['mask']:
            assert audit.sha(row['mask']) == row['mask_sha256']
    assert audit.sha(preflight_path) == preflight_sha
    summary = dict(source='Full VisA', images=2162, patches=count, current_live_extraction_rows=2162,
        historical_cache_reused_as_model_input=False, all_physical_hashes_checked_before_and_after=True,
        duplicate_image_paths=0, nonfinite_rows=0, all_patch_and_CLS_rows_verified=True,
        visual_weights_unchanged_after_position_setup=True, training_started=False, target_inference_started=False,
        old_cache_max_abs=max(r['patch_max_abs'] for r in diagnostics),
        old_cache_float16_cast_different_values=sum(r['values_different_after_float16_cast'] for r in diagnostics),
        source_rebuild_release='NOT APPROVED; PC1/E1/E2/adapter and CES reconstruction require separate review')
    audit.js(OUT / 'VALIDATION_SUMMARY.json', summary)
    report = ['# Fresh VisA source feature certification', '',
        '## FEATURE_CERTIFIED; TRAINING AND TARGET EVALUATION REMAIN LOCKED', '',
        f'Original preflight SHA256: `{preflight_sha}`. Original files were not edited.', '',
        'All 2162 source images were read as actual bytes, matched to their frozen image hashes, transformed using '
        'the locked formal preprocessing, and forwarded through the frozen encoder. Each row binds input tensor SHA256, '
        'patch SHA256, CLS SHA256, physical image and mask hashes, configuration and encoder identities.', '',
        'The actual CLIP base file is ViT-L-14-336px.pt with its official SHA256. test_on_mvtec.pth restores only '
        'ancillary prompt/projection modules; those modules are not used by this visual feature readout. '
        'No model download or fallback was permitted.', '',
        f'Actual model/input dtype: {model.dtype}; output storage float32; autocast and TF32 disabled. '
        'The canonical unit patch and projected original-stream unit CLS helpers are used unchanged. '
        'Position embeddings receive the official fixed 24-to-37 bilinear interpolation before the frozen checksum.', '',
        'Each of 2,959,778 patches has a grid position and true source label in PATCH_FEATURE_BINDING.csv.gz. '
        'Join image_ledger_row to IMAGE_FEATURE_MANIFEST.csv to recover the physical input and per-image feature hash.', '',
        'OLD_CACHE_DIFFERENCES.csv is descriptive only. No equality to old float16 features was required and '
        'no preprocessing or precision was adjusted to match old outputs.', '',
        'This certificate authenticates the new extraction, not historical model training. It does not authenticate '
        'the old PC1, E1/E2, adapter or CES head. Those source stages remain unapproved for reconstruction, '
        'and no target images, GT, predictions or metrics were read.', '',
        f'Outputs: {WORK}', '']
    (OUT / 'SOURCE_FEATURE_CERTIFICATION_REPORT.md').write_text('\n'.join(report), encoding='utf-8')
    sealed = dict(created=audit.now(), status='SOURCE_FEATURES_CERTIFIED_ONLY', original_preflight_sha256=preflight_sha,
        feature_inputs='fresh physical source images, never old features',
        patch_features=patch_identity, cls_features=cls_identity,
        source_patch_binding=audit.ident(binding_path), source_labels=audit.ident(labels_path),
        base_CLIP_weights=base_identity, ancillary_checkpoint=ancillary_identity,
        image_feature_manifest=audit.ident(OUT/'IMAGE_FEATURE_MANIFEST.csv'),
        extractor_config=audit.ident(OUT/'EXTRACTION_CONFIG.json'), encoder_identity=audit.ident(OUT/'ENCODER_IDENTITY.json'),
        code_snapshot=audit.ident(snapshot), code=codes, old_diagnostic_cache=old_identity,
        validation_summary=audit.ident(OUT/'VALIDATION_SUMMARY.json'),
        source_build_approved=False, target_evaluation_approved=False)
    audit.js(OUT / 'SOURCE_FEATURE_CERTIFICATE.json', sealed)
    status('SOURCE_FEATURES_CERTIFIED_TRAINING_LOCKED', images=2162, patches=count,
           certificate_sha256=audit.sha(OUT/'SOURCE_FEATURE_CERTIFICATE.json'))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    try:
        run()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        audit.js(OUT/'STATUS.json', dict(stage='FEATURE_EXTRACTION_FAILED', time=audit.now(),
            error=traceback.format_exc(), training_started=False, target_inference_started=False))
        raise
