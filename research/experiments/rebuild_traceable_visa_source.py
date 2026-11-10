"""Source-only preflight for a new traceable VisA reconstruction.

No encoder forward, optimizer, target loader, target GT, or metric computation is
executed here. Future training/evaluation stages are fail-closed and deliberately
unavailable until the preflight has been reviewed and their implementation locked.
"""
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import io
import json
import shutil
import sys
import tempfile
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/visa_source_traceable_rebuild/formal'
WORK = Path('E:/MRAD_visa_source_traceable_rebuild')
AUDIT = ROOT / 'results/ablation_main_cache_identity_20261027'
sys.path.insert(0, str(ROOT))


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def js(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def ident(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def csvout(path, rows):
    rows = list(rows)
    if not rows:
        raise RuntimeError(f'Empty evidence table: {path}')
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def require_identity(record):
    actual = ident(record['path'])
    if actual != record:
        raise RuntimeError(f'Locked input changed: {record["path"]}')


def stage_gate(stage, base=OUT):
    """Approval is a separate review artifact, never produced by preflight."""
    manifest_path = base / 'PREFLIGHT_MANIFEST.json'
    approval_path = base / 'PREFLIGHT_APPROVAL.json'
    manifest = read(manifest_path)
    approval = read(approval_path)
    if manifest['training_eligibility'] != 'ELIGIBLE_FOR_REVIEW':
        raise RuntimeError('Preflight has unresolved physical/source certification blockers')
    if approval.get('decision') != 'APPROVED' or not approval.get('reviewer'):
        raise RuntimeError('An explicit reviewed APPROVED record is required')
    if approval.get('manifest_sha256') != sha(manifest_path):
        raise RuntimeError('Approval does not match the actual preflight manifest')
    for entry in manifest['locked_files']:
        require_identity(entry)
    if stage == 'evaluate':
        seal_path = base / 'SOURCE_SEALED.json'
        seal = read(seal_path)
        if seal.get('preflight_manifest_sha256') != sha(manifest_path):
            raise RuntimeError('Source seal is not bound to the reviewed preflight')
        if seal.get('complete') is not True or seal.get('updates') != 30:
            raise RuntimeError('Incomplete source training seal')
        if set(seal.get('endpoints', {})) != {'initialization', 'update30'}:
            raise RuntimeError('Both locked endpoints are required')
        for entry in seal['locked_files']:
            require_identity(entry)
        for entry in seal['endpoints'].values():
            require_identity(entry)
    raise RuntimeError(f'{stage} is NOT IMPLEMENTED/NOT RELEASED; no training or inference executed')


def self_test():
    """Exercise missing, blocked, stale and changed-input guards in temp space."""
    with tempfile.TemporaryDirectory(prefix='cesad_preflight_guard_') as folder:
        base = Path(folder)
        def rejected(stage, expected):
            try:
                stage_gate(stage, base)
            except (RuntimeError, FileNotFoundError) as error:
                assert expected in str(error), (expected, str(error))
                return
            raise AssertionError('Guard unexpectedly released execution')
        rejected('source-build', 'PREFLIGHT_MANIFEST')
        asset = base / 'input.txt'
        asset.write_text('original', encoding='ascii')
        manifest = dict(training_eligibility='BLOCKED', locked_files=[ident(asset)])
        mp = base / 'PREFLIGHT_MANIFEST.json'
        js(mp, manifest)
        js(base / 'PREFLIGHT_APPROVAL.json', dict(decision='APPROVED', reviewer='test', manifest_sha256=sha(mp)))
        rejected('source-build', 'blockers')
        manifest['training_eligibility'] = 'ELIGIBLE_FOR_REVIEW'
        js(mp, manifest)
        rejected('source-build', 'Approval does not match')
        js(base / 'PREFLIGHT_APPROVAL.json', dict(decision='APPROVED', reviewer='test', manifest_sha256=sha(mp)))
        asset.write_text('tampered', encoding='ascii')
        rejected('source-build', 'Locked input changed')
        asset.write_text('original', encoding='ascii')
        rejected('source-build', 'NOT IMPLEMENTED')
        rejected('evaluate', 'SOURCE_SEALED')
        js(base / 'SOURCE_SEALED.json', dict(preflight_manifest_sha256=sha(mp), complete=False))
        rejected('evaluate', 'Incomplete')
    print('SELF_TEST PASS: 7 fail-closed checks; no project assets changed', flush=True)


def code_inventory(seeds):
    """Freeze the statically resolvable local import closure, including function imports."""
    pending, seen = list(seeds), set()
    while pending:
        path = Path(pending.pop()).resolve()
        if path in seen or not path.is_file() or not path.is_relative_to(ROOT):
            continue
        seen.add(path)
        tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.append(node.module)
                modules.extend(node.module + '.' + alias.name for alias in node.names)
        for module in modules:
            stem = ROOT.joinpath(*module.split('.'))
            for candidate in (stem.with_suffix('.py'), stem / '__init__.py'):
                if candidate.is_file():
                    pending.append(candidate)
    return sorted(seen)


def preflight():
    import numpy as np
    import torch
    from PIL import Image
    from experiments import phase_ces_lhc_paired_repeat as paired
    from experiments import phase_defect_query_transfer_pilot as source
    from experiments import phase_statealigned_pc1_debias as auxiliary
    from experiments import phase_spatial_supervision_transfer_audit as spatial
    from experiments import phase_xxxviii_visual_state_representation as states
    from experiments import phase_xlii_patch_specialized_adaptation as adapt
    from experiments import final_unified_evaluation as evaluator
    from experiments.phase_xlv_cross_domain_transfer import CHECKPOINT

    if (OUT / 'PREFLIGHT_MANIFEST.json').exists():
        raise RuntimeError('Preflight already sealed; refuse to overwrite existing evidence')
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    heavy = WORK / 'preflight'
    heavy.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    js(OUT / 'STATUS.json', dict(stage='PREFLIGHT', started=now(), training_started=False))
    print('SOURCE METADATA ONLY; no encoder forward or target dataset access', flush=True)
    _, ds, tr, held = paired.source_dataset('Full VisA')
    indices = list(tr) + list(held)
    assert len(indices) == len(set(indices)) == len(ds.data_all)
    mapping, _ = source.load_taxonomy()
    taxonomy = ident(source.TAXONOMY)
    meta_path = Path(ds.root) / 'meta.json'
    cache_path = Path('E:/MRAD_final_evidence_cache/global_patch_VisA.npy')
    a = np.load(cache_path, mmap_mode='r')
    assert a.shape == (len(indices), 1369, 1024) and a.dtype == np.float16
    feature_identity = ident(cache_path)
    print('HASHED original E-drive feature cache', flush=True)
    arrays, carrier_paths = paired.source_arrays('D2')
    carrier_ids = [ident(path) for path in carrier_paths]
    carrier_npz = ROOT / 'results/phase_xxiii_carrier_audit/feature_cache/D2_fold0_clip.npz'
    with np.load(carrier_npz) as saved:
        recorded_indices = np.concatenate([saved['train_indices'], saved['held_indices']])
    recorded_indices = recorded_indices.astype(np.int64)
    permutation_path = AUDIT / 'CARRIER_TO_E_VISA_INDEX.npy'
    permutation = np.load(permutation_path)
    assert permutation.shape == (len(a),) and np.array_equal(np.sort(permutation), np.arange(len(a)))
    assert len(recorded_indices) == len(a) == sum(len(x) for x in arrays)
    inverse = np.argsort(permutation)
    csvout(OUT / 'CARRIER_RECORDED_INDICES.csv',
           [dict(carrier_row=i, recorded_dataset_index=int(idx), content_matched_E_row=int(permutation[i]),
                 currently_claimed_dataset_index=int(indices[permutation[i]]),
                 recorded_index_matches_current_claim=bool(idx == indices[permutation[i]]))
            for i, idx in enumerate(recorded_indices)])
    labels = np.zeros((len(a), 1369), np.uint8)
    id_masks = np.zeros((len(a), 1369), np.uint8)
    rows, norms, content_rows = [], [], []
    unknown_ids = []
    records = []
    patch_path = heavy / 'SOURCE_PATCH_MANIFEST.csv.gz'
    # Each of all 2,959,778 patches joins by source_position to the hashed image/mask ledger.
    with gzip.open(patch_path, 'wt', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['patch_id', 'source_position', 'patch_flat_index', 'patch_y', 'patch_x',
                         'label', 'raw_nearest_mask_id', 'E_feature_row', 'carrier_content_row'])
        offset = np.cumsum([0] + [len(x) for x in arrays])
        for pos, idx in enumerate(indices):
            item = ds.data_all[int(idx)]
            image = (Path(ds.root) / item['img_path']).resolve()
            mask = (Path(ds.root) / item['mask_path']).resolve() if item['mask_path'] else None
            normal = int(item['anomaly']) == 0
            if not image.is_file() or (not normal and (mask is None or not mask.is_file())):
                raise RuntimeError(f'Missing physical image/mask at source row {pos}')
            with Image.open(image) as im:
                image_wh = im.size
            if not normal:
                with Image.open(mask) as im:
                    assert im.size == image_wh, (image, mask, 'spatial size mismatch')
                ids = source.resize_mask_ids(mask)
                assert ids.shape == (1369,) and ids.min() >= 0 and ids.max() <= 255
                id_masks[pos] = ids
                for mid in sorted(int(x) for x in np.unique(ids) if x > 0):
                    if mapping.get((item['cls_name'], mid)):
                        labels[pos, ids == mid] = 1
                    else:
                        unknown_ids.append(dict(source_position=pos, category=item['cls_name'], mask_id=mid,
                                                affected_patches=int((ids == mid).sum())))
            raw = np.asarray(a[pos])
            assert np.isfinite(raw).all(), f'Nonfinite frozen feature row {pos}'
            carrier_row = int(inverse[pos])
            part = int(np.searchsorted(offset[1:], carrier_row, side='right'))
            carrier = np.asarray(arrays[part][carrier_row - offset[part]])
            equal = np.array_equal(raw, carrier)
            if not equal:
                raise RuntimeError(f'Verified permutation no longer matches source row {pos}')
            n = np.linalg.norm(raw.astype(np.float32), axis=1)
            norms.append(dict(source_position=pos, minimum=float(n.min()), maximum=float(n.max()), mean=float(n.mean())))
            content_rows.append(dict(E_row=pos, carrier_row=carrier_row, differing_values=0,
                                     row_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                                     finite=True, status='EXACT_CONTENT_MATCH'))
            rows.append(dict(source_position=pos, dataset_index=int(idx), category=item['cls_name'],
                image=str(image), image_sha256=sha(image), mask=str(mask) if mask else '',
                mask_sha256=sha(mask) if mask else '', image_label=int(not normal),
                image_width=image_wh[0], image_height=image_wh[1], E_feature_row=pos,
                carrier_content_row=carrier_row, carrier_recorded_dataset_index=int(recorded_indices[carrier_row]),
                positive_patches=int(labels[pos].sum()),
                physical_feature_binding='CLAIMED_CURRENT_METADATA_ORDER; NOT_ALL_ROWS_LIVE_VERIFIED'))
            records.append(dict(pos=pos, idx=int(idx), mask=labels[pos].astype(np.float32),
                                normal=not bool(labels[pos].any())))
            for pid in range(1369):
                writer.writerow([f'visa:{pos}:{pid}', pos, pid, pid // 37, pid % 37,
                                 int(labels[pos, pid]), int(id_masks[pos, pid]), pos, carrier_row])
            if pos % 200 == 0:
                print(f'PHYSICAL FILES + CONTENT {pos}/{len(a)}', flush=True)
    shifted, shift_audit = spatial.corrupt_masks(records)
    shifted = (np.stack(shifted) > .5).astype(np.uint8)
    np.save(heavy / 'true_labels_nearest_id_union.npy', labels)
    np.save(heavy / 'shifted_labels.npy', shifted)
    np.save(heavy / 'nearest_raw_id_masks.npy', id_masks)
    csvout(OUT / 'SOURCE_IMAGE_MANIFEST.csv', rows)
    csvout(OUT / 'FEATURE_ROW_CONTENT_AUDIT.csv', content_rows)
    csvout(OUT / 'FEATURE_NORMALIZATION_AUDIT.csv', norms)
    csvout(OUT / 'SHIFTED_MASK_AUDIT.csv', shift_audit)
    js(OUT / 'UNMAPPED_MASK_IDS.json', unknown_ids)

    code_files = code_inventory([Path(__file__), Path(paired.__file__), Path(auxiliary.__file__),
        Path(spatial.__file__), Path(evaluator.__file__), ROOT / 'phase_xa/official_task_forward.py',
        ROOT / 'experiments/run_ces_geometry_center_control.py'])
    code_ids = [ident(path) for path in code_files]
    archive = heavy / 'CODE_SNAPSHOT.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for path in code_files:
            z.write(path, str(path.relative_to(ROOT)))
    js(OUT / 'CODE_IDENTITY.json', code_ids)
    historical = auxiliary.OUT
    if not historical.is_absolute():
        historical = ROOT / historical
    assets = [historical / name for name in ('q_pc1_statealigned1024.npy',
        'E1_PC1Removed-TrueMask_formal.pt', 'E2_PC1Removed-CorruptedMask_formal.pt')]
    asset_ids = [ident(path) for path in assets]
    clip_identity = ident(CHECKPOINT)
    live_path = AUDIT / 'LIVE_SOURCE_IMAGE_PARITY.csv'
    with live_path.open(encoding='utf-8-sig', newline='') as f:
        previous_live = list(csv.DictReader(f))
    historical_live_rows = sorted({int(r['position']) for r in previous_live})
    cls_shapes = {}
    with np.load(carrier_npz) as saved:
        for key in ('train_cls', 'held_cls'):
            cls_shapes[key] = list(saved[key].shape)
    configuration = dict(source='Full VisA', seeds=[0], targets=['MVTec', 'BTAD', 'MPDD', 'DTD-Synthetic', 'DAGM'],
        source_image_order='light_fold(D2,0): train_indices followed by held_indices, explicitly manifested',
        source_images=len(rows), source_patches=int(labels.size), dtype='float32 downstream; candidate cache float16',
        upstream=dict(backbone='CLIP ViT-L/14@336px', checkpoint=clip_identity, input_size=[518,518],
            patch_shape=[37,37,1024], dpam_layer=24, feature_layer=24, neighbor_scales=[1,3,5],
            preprocessing='exact get_transform and visual_outputs/normalized_patch_feature, code locked',
            pc1='normal-image mean spatial feature map; center across 1369 locations; SVD vt[0]; unit; sign unforced',
            feature_order='normalized frozen patch -> remove PC1 -> L2 -> E2 rank4 residual adapter -> L2',
            cls_cache_shapes=cls_shapes, cls_cache_certification='UNAUTHENTICATED physical rows'),
        masks=dict(true='raw ID mask PIL L -> nearest 37x37 -> union taxonomy-mapped positive IDs',
            shifted='whole union mask circular np.roll dy then dx', seed=int(spatial.SEED),
            max_attempts=128, dy_dx=[0,36], zero_shift='skip consumes attempt',
            accept='first IoU < 0.1 else lowest attempted IoU', normal='all zero',
            final_CES_labels='TRUE nearest-ID union, never shifted'),
        auxiliary=dict(state_optimizer='AdamW', state_lr=float(auxiliary.STATE_LR),
            state_weight_decay=0., state_updates=int(auxiliary.STATE_EPOCHS),
            state_loss='weighted image CE + category-balanced patch CE; source code snapshot is authoritative',
            patch_cap_per_category_class=int(auxiliary.PATCH_SAMPLE_PER_CLASS_CATEGORY),
            patch_take_per_category_class_per_update=128,
            adapter_rank=int(adapt.RANK), adapter_optimizer='AdamW', adapter_lr=float(auxiliary.ADAPTER_LR),
            adapter_weight_decay=float(auxiliary.WEIGHT_DECAY), adapter_epochs=int(auxiliary.ADAPTER_EPOCHS),
            adapter_batch=int(auxiliary.BATCH_SIZE), adapter_loss='balanced BCE + sigmoid Dice, weights 1+1',
            E1_supervision='true masks for Stage I and Stage II', E2_supervision='shifted masks for both stages'),
        ces=dict(center='mean of E2 Stage-I unit states',
            half_difference='half(E1 Stage-I anomaly minus normal)', alpha='norm(half_difference)',
            v0='unit(half_difference)', states='normalize(m +/- alpha*normalize(raw_v))',
            trainable='raw_v only, 1024 scalars', frozen='CLIP, PC1, E2 adapter, m, alpha',
            optimizer='AdamW', lr=.001, weight_decay=0., epochs=10, actual_updates=30,
            pool_cap_per_category_origin=512, origins=['normal_image','anomaly_mask_out','anomaly_mask'],
            batch='category/class balanced; 8184 actual indices; save all 30 streams',
            base_loss='balanced BCE + Dice', rho=.10, NTR_weight=1.,
            endpoints=['initialization','update30'], selection='fixed endpoints, never target-selected'),
        inference=dict(LRP='2x2 nonoverlapping quadrants; global+four views', crop_size=[518,518],
            logits_grid=[74,74], global_local_weights=[.5,.5], gaussian_smoothing=None),
        evaluation=dict(evaluator_id=evaluator.evaluator_hash(), policy=evaluator.POLICY,
            pixel='original paired GT; float32 bilinear align_corners=True; category-macro exact AP ties',
            image='74x74 grid, ceil top1 percent =55, category-macro',
            target_access='NONE during preflight/source fitting; two endpoints after complete seal'),
        runtime=dict(python=sys.version, numpy=np.__version__, torch=torch.__version__,
                     tf32=False, deterministic=True, autocast=False),
        code_identity=str(OUT / 'CODE_IDENTITY.json'), code_snapshot=str(archive))
    js(OUT / 'LOCKED_CONFIG.json', configuration)
    matching_records = sum(bool(idx == indices[permutation[i]]) for i, idx in enumerate(recorded_indices))
    levels = [
        dict(level='L1_CURRENT_FILE_LEDGER', coverage=len(rows), total=len(rows), status='VERIFIED',
             scope='Current image/mask paths, bytes, spatial dimensions and label derivation; not cache-to-image proof'),
        dict(level='L2_ARRAY_CONTENT', coverage=len(rows), total=len(rows), status='VERIFIED',
             scope='Every E-cache row equals carrier row under recorded permutation; finite values and row SHA256'),
        dict(level='L2_CARRIER_INDEX_AGREEMENT', coverage=matching_records, total=len(rows),
             status='ACTUAL_INCONSISTENCY' if matching_records != len(rows) else 'VERIFIED',
             scope='Saved carrier dataset indices versus current E-row metadata claim; not resolved by shapes'),
        dict(level='L3_HISTORICAL_LIVE_EXTRACTION', coverage=len(historical_live_rows), total=len(rows),
             status='HISTORICAL_PARTIAL_EVIDENCE',
             scope='Prior two-image numeric parity CSV; no original input hashes/full extraction seal in that CSV'),
        dict(level='L3_CURRENT_LIVE_EXTRACTION', coverage=0, total=len(rows), status='NOT_PERFORMED',
             scope='This source-only preflight performs no encoder forward; all-row physical feature binding remains unauthenticated')]
    csvout(OUT / 'CERTIFICATION_LEVELS.csv', levels)
    reuse = [
        dict(asset='CLIP checkpoint', action='REUSE_FROZEN_AFTER_RUNTIME_IDENTITY_CHECK', status='BYTE_IDENTITY_RECORDED', reason='No training; future extractor must checksum base weights and locked positional adaptation'),
        dict(asset='E-drive VisA patch cache', action='HOLD_FOR_PHYSICAL_ROW_VALIDATION', status='PARTIAL', reason='All content rows match; physical image correspondence has only historical 2-row live evidence'),
        dict(asset='carrier image/CLS features', action='REEXTRACT_WITH_PHYSICAL_IMAGE_IDS_UNLESS_FULLY_CERTIFIED', status='UNAUTHENTICATED', reason='No image-CLS physical pairing certificate; patch row matching does not prove CLS row identity'),
        dict(asset='historical PC1', action='REBUILD_FROM_CERTIFIED_NORMAL_IMAGE_POOL', status='UNAUTHENTICATED', reason='Historical normal-image membership/feature-row consumption not independently authenticated; not declared numerically wrong'),
        dict(asset='historical E1 Stage-I states', action='REBUILD_AFTER_SOURCE_CERTIFICATION', status='UNAUTHENTICATED', reason='Historical physical feature-label batch pairing cannot be authenticated'),
        dict(asset='historical E2 Stage-I states', action='REBUILD_AFTER_SOURCE_CERTIFICATION', status='UNAUTHENTICATED', reason='Historical physical feature-shifted-mask batch pairing cannot be authenticated'),
        dict(asset='historical E2 adapter', action='REBUILD_AFTER_CERTIFIED_E2_STAGE_I', status='UNAUTHENTICATED', reason='Depends on E2 states, transformed source rows and shifted labels'),
        dict(asset='historical final CES head', action='REBUILD_FROM_NEW_SEALED_SOURCE_ASSETS', status='HISTORICAL_IDENTITY_ONLY', reason='Do not use legacy fallback or silently reuse old trained head'),
        dict(asset='MVTec-source model and MVTec-to-VisA results', action='KEEP_UNCHANGED', status='OUT_OF_SCOPE', reason='Not rebuilt, evaluated or modified'),
        dict(asset='target caches', action='AUDIT_LATER_AFTER_SOURCE_SEAL', status='NOT_ACCESSED', reason='No target loader/GT/prediction access during this stage')]
    csvout(OUT / 'ASSET_REUSE_REBUILD.csv', reuse)
    disk = {drive:dict(free_bytes=shutil.disk_usage(drive).free) for drive in ('D:/','E:/')}
    js(OUT / 'RESOURCE_SNAPSHOT.json', disk)
    blockers = ['All-row physical feature/image binding is not yet authenticated (0 current live rows; 2 historical rows).',
                'Carrier image/CLS physical row identity must be established or source features re-extracted with IDs.',
                'Historical PC1/E1/E2/adapter cannot be certified for reuse; certified reconstruction required.',
                'Training/evaluation runner is intentionally not released; review and updated implementation lock required.']
    if unknown_ids:
        blockers.append('Taxonomy-unmapped source mask IDs require explicit review; no labels silently changed.')
    report = ['# VisA source reconstruction preflight', '',
        '## Decision: BLOCKED_PENDING_SOURCE_CERTIFICATION_AND_REVIEW', '',
        'No training, encoder forward, target inference, metric computation, or historical-file modification occurred.', '',
        '## Coverage', '',
        *[f'- {r["level"]}: {r["coverage"]}/{r["total"]}; {r["status"]}. {r["scope"]}.' for r in levels], '',
        'A content-equal array permutation is not independent physical image certification. '
        'Missing historical evidence means UNAUTHENTICATED, not proven wrong or approved.', '',
        'SOURCE_PATCH_MANIFEST.csv.gz contains every patch ID, grid index, true label and joined image-row identity; '
        'SOURCE_IMAGE_MANIFEST.csv contains image/mask paths and hashes. shifted_labels.npy plus SHIFTED_MASK_AUDIT.csv '
        'preserve source-only mask processing. None is a target-derived choice.', '',
        '## Asset actions', '',
        *[f'- **{r["asset"]}**: {r["action"]}; {r["status"]}. {r["reason"]}.' for r in reuse], '',
        '## Blockers', '', *[f'- {s}' for s in blockers], '',
        '## Release discipline', '',
        'PREFLIGHT_APPROVAL.json is deliberately NOT generated. A later source-build requires an explicitly reviewed '
        'eligible manifest and exact SHA256 approval. evaluate additionally requires SOURCE_SEALED.json, all asset '
        'hashes, and both initialization/update30 endpoints. Current handlers stop even after guard verification '
        'because training/evaluation have not yet been implemented and frozen.', '',
        'LOCKED_CONFIG.json and CODE_SNAPSHOT.zip freeze the preparation recipe for review, NOT approval to run. '
        'Future source/evaluation implementation changes require a new reviewed lock; no silent substitution is allowed.', '',
        f'Historical live positions: {historical_live_rows}. Unmapped ID records: {len(unknown_ids)}.',
        f'Heavy evidence directory: {heavy}', '']
    (OUT / 'PREFLIGHT_REPORT.md').write_text('\n'.join(report), encoding='utf-8')
    locked = [feature_identity, *carrier_ids, ident(carrier_npz), ident(permutation_path), taxonomy,
              *asset_ids, clip_identity, ident(live_path)]
    if meta_path.is_file():
        locked.append(ident(meta_path))
    locked.extend(ident(path) for path in sorted(OUT.iterdir()) if path.is_file() and path.name != 'STATUS.json')
    locked.extend(ident(path) for path in sorted(heavy.iterdir()) if path.is_file())
    locked.extend(code_ids)
    # Physical files are bound individually, so a same-path image replacement cannot pass a later release check.
    for row in rows:
        for key in ('image','mask'):
            if row[key]:
                path = Path(row[key])
                locked.append(dict(path=str(path), bytes=path.stat().st_size, sha256=row[key+'_sha256']))
    unique = {r['path']:r for r in locked}
    manifest = dict(created=now(), source='Full VisA', seed=0,
        training_eligibility='BLOCKED_PENDING_SOURCE_CERTIFICATION_AND_REVIEW',
        blocks=blockers, source_images=len(rows), patches=int(labels.size),
        current_live_extraction_count=0, historical_live_extraction_count=len(historical_live_rows),
        code_static_import_closure_count=len(code_ids), locked_files=list(unique.values()),
        report=str(OUT / 'PREFLIGHT_REPORT.md'), training_started=False, target_inference_started=False)
    js(OUT / 'PREFLIGHT_MANIFEST.json', manifest)
    js(OUT / 'STATUS.json', dict(stage='PREFLIGHT_COMPLETE_BLOCKED', completed=now(), training_started=False,
        target_inference_started=False, manifest_sha256=sha(OUT / 'PREFLIGHT_MANIFEST.json')))
    print(json.dumps(dict(status='PREFLIGHT_COMPLETE_BLOCKED', source_images=len(rows),
        patch_count=int(labels.size), carrier_recorded_index_matches=matching_records,
        historical_live_rows=historical_live_rows, new_live_rows=0, unmapped_mask_ids=len(unknown_ids),
        report=str(OUT / 'PREFLIGHT_REPORT.md'), assets=str(OUT / 'ASSET_REUSE_REBUILD.csv')), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('preflight','source-build','evaluate','self-test'), required=True)
    args = parser.parse_args()
    if args.stage == 'self-test':
        self_test()
    elif args.stage == 'preflight':
        try:
            preflight()
        except Exception:
            OUT.mkdir(parents=True, exist_ok=True)
            js(OUT / 'STATUS.json', dict(stage='PREFLIGHT_FAILED', time=now(),
                                        traceback=traceback.format_exc(), training_started=False))
            raise
    else:
        stage_gate(args.stage)


if __name__ == '__main__':
    main()
