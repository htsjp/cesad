"""Inference-only runtime guard for the locked reference-center experiment.

The official visual forward deterministically replaces positional_embedding.data
on its first 518 input. Establish exactly that documented shape adaptation before
the frozen checksum; do not confuse it with an optimizer update. The locked
training script and six existing checkpoints remain byte-for-byte unchanged.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import phase_ces_geometry_center_control as core


class ResolutionEstablishedEncoder(core.FrozenEncoder):
    def checksum(self):
        visual = self.model.visual
        p = visual.positional_embedding
        side = int((p.shape[0]-1)**.5)
        if side != 37:
            assert side == 24 and p.shape[1] == 1024
            original = core.paired.digest(p.detach().cpu().numpy())
            # Exactly AnomalyCLIP_lib/AnomalyCLIP.py::VisionTransformer.forward:
            # bilinear with default align_corners=False, CLS row retained.
            with torch.no_grad():
                new = p[1:, :].reshape(-1, side, side, p.shape[-1]).permute(0,3,1,2)
                new = F.interpolate(new, (37,37), mode='bilinear')
                new = new.reshape(-1, p.shape[-1], 37*37).transpose(1,2)
                p.data = torch.cat([p[:1,:],new[0]],0)
            assert not p.requires_grad
            self.position_setup = dict(source_grid=[24,24], inference_grid=[37,37],
                before_hash=original, after_hash=core.paired.digest(p.detach().cpu().numpy()),
                operation='official first-forward bilinear interpolation, no fitted parameter update',
                evidence='AnomalyCLIP_lib/AnomalyCLIP.py:389-401')
        return super().checksum()


original_target_cell = core.target_cell


def checked_target_cell(target):
    cell = original_target_cell(target)
    rows,a,local,mapping,gp,lp,dest = cell
    cp = dest / 'TARGET_INPUT_HASHES.json'
    current = dict(global_features=core.identity(gp),
        target_manifest=core.identity(dest/'TARGET_MANIFEST.csv'),
        local_features=core.identity(lp) if local is not None else 'fresh frozen extraction',
        source_transform_assets='locked GEOMETRY_CONTROL_CONFIG.json')
    if cp.exists():
        old = core.official.readjson(cp)
        assert old['inputs'] == current
    else:
        for r in rows:
            assert core.paired.file_hash(r['img_path']) == r['image_sha256'], ('Image differs from main table',r['sample_id'])
            if int(r['label']):
                assert core.paired.file_hash(r['gt_path']) == r['gt_sha256'], ('GT differs from main table',r['sample_id'])
        core.js(cp, dict(inputs=current, images=len(rows),
                         status='PASS; image and GT bytes match frozen main-table manifest',
                         local_cache_validation='SMOKE_CHECKS.csv; actual fresh frozen features, not source-name assumption'))
    return cell


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('smoke','run'),default='run')
    args = parser.parse_args()
    core.setup()
    config = core.official.readjson(core.OUT/'GEOMETRY_CONTROL_CONFIG.json')
    assert core.paired.file_hash(core.__file__)==config['code'][0]['sha256']
    models = core.official.readjson(core.OUT/'SOURCE_TRAINING_COMPLETE.json')['models']
    for m in models:assert core.paired.file_hash(m['checkpoint'])==m['checkpoint_sha256']
    core.js(core.OUT/'RUNTIME_INFERENCE_PROVENANCE.json',dict(
        locked_training_runner=core.identity(core.__file__), runtime_guard=core.identity(__file__),
        changed_training_functions=False,changed_checkpoints=False,
        reason='The first smoke failed only its pre-resolution vs post-resolution whole-visual checksum; all ten real-feature parity/AP checks completed. Official forward resizes the positional embedding once.',
        solution='Establish exactly the official 24->37 positional interpolation before frozen checksum. Both configurations and all targets keep exactly the same518 input and backbone.',
        frozen_check_scope='whole visual state after deterministic input-resolution setup, before/after forwards',
        target_hashes='Every image and positive GT checked against frozen main-table manifest; feature inputs SHA256 saved per target.'))
    core.FrozenEncoder = ResolutionEstablishedEncoder
    core.target_cell = checked_target_cell
    # Only the inference resolution/checksum guard is substituted; all source
    # geometry, fitting, scoring, crop/fusion and evaluator functions are intact.
    sys.argv = [str(core.__file__),'--stage',args.stage]
    core.main()


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        core.js(core.OUT/'RUNTIME_ERROR.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
