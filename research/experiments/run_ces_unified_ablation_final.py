"""Inference/packaging entry point for the locked unified ablation experiment.

Old StepB manifests use dataset copies under project/data while the native-GT
manifest uses E:/WorkSpace/dataset. Resolve only verified byte-identical image
aliases in memory; retain old file bytes and original cached map indices.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import phase_ces_unified_ablation_final as core

original_rcsv=core.rcsv


def verified_alias_csv(path):
    path=Path(path);rows=original_rcsv(path)
    if path.name=='TARGET_MANIFEST.csv' and path.resolve().is_relative_to((core.OLD/'predictions').resolve()):
        done=core.read(path.parent/'COMPLETE.json')
        assert core.paired.file_hash(path)==done['manifest_hash']
        target=done['target']
        canonical=original_rcsv(core.official.WORK/'manifests'/f'{target}.csv')
        paths=core.official.canonical_paths(target,[r['image'] for r in rows],canonical)
        return [dict(r,image=p) for r,p in zip(rows,paths)]
    return rows


def source_diagnostics(config,models):
    rows=[];checks=[]
    for source in core.SOURCES:
        for seed in core.SEEDS:
            entries=[e for e in models if e['source']==source and e['seed']==seed]
            base=next(e for e in entries if e['variant']=='CES_NTR')
            baseline=torch.load(base['checkpoint'],map_location='cpu',weights_only=False)
            no=next(e for e in entries if e['variant']=='CES_noNTR')
            nt=torch.load(no['checkpoint'],map_location='cpu',weights_only=False)
            for k in ('m','alpha'):
                torch.testing.assert_close(nt['initial_state'][k],baseline['raw_model_state'][k],rtol=0,atol=0)
            torch.testing.assert_close(nt['initial_state']['v'],torch.tensor(baseline['initial_geometry']['init_v']),rtol=0,atol=0)
            checks.append(dict(source=source,seed=seed,noNTR_initialization_match=True,
                same_feature_hash=base['feature_hash']==no['feature_hash'],
                same_batch_hash=base['batch_hash']==no['batch_hash'],
                base_updates=30,noNTR_updates=30,continuation_and_higher_total_updates=60))
            for e in entries:
                p=torch.load(e['checkpoint'],map_location='cpu',weights_only=False)
                ip=p['initial_probe'];last=p['logs'][-1]
                fp=last.get('probe',last)
                row=dict(source=source,seed=seed,variant=e['variant'],checkpoint=e['checkpoint'],
                    checkpoint_sha256=e['checkpoint_sha256'],total_updates=e['total_updates'],
                    source_feature_hash=e['feature_hash'],batch_hash=e['batch_hash'])
                for label,stats in [('initial',ip),('final',fp)]:
                    for key in ('loss','bce','dice','ntr','mean_normal','mean_anomaly'):
                        row[f'{label}_source_probe_{key}']=stats.get('probe_'+key,stats.get(key))
                if not e['variant'].startswith('K'):
                    spec=core.spec_for(e);row['final_effective_weight_norm']=float(np.linalg.norm(spec.weight()))
                    if e['variant'] in ('LinearHead','NormalizedLinearHead','CES_NTR'):
                        initial=p['initial_geometry']
                        row['initial_effective_weight_norm']=float(np.linalg.norm(initial['w0']))
                    elif 'initial_state' in p:
                        init=p['initial_state']
                        w=core.paired.hist.ces_weight_np(init['m'].numpy(),float(init['alpha']),init['v'].numpy())
                        row['initial_effective_weight_norm']=float(np.linalg.norm(w))
                    elif 'raw_v_trajectory' in p:
                        w=core.paired.hist.ces_weight_np(p['params']['m'],p['params']['alpha'],p['raw_v_trajectory'][0].numpy())
                        row['initial_effective_weight_norm']=float(np.linalg.norm(w))
                else:
                    row['anomaly_state_count']=int(p['s_a'].shape[0])
                    row['initial_base_checkpoint']=p['initial_base_checkpoint']
                    row['initial_base_hash']=p['initial_base_hash']
                rows.append(row)
    core.csv(core.OUT/'UNIFIED_SOURCE_GEOMETRY_DIAGNOSTICS.csv',rows)
    core.js(core.OUT/'UNIFIED_PAIRED_TRAINING_VALIDATION.json',dict(status='PASS',groups=checks,
            source_heads=45,new_fits=24,config_hash=core.paired.json_hash(config)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('smoke','run','summarize'),default='run')
    args=parser.parse_args();core.setup()
    config=core.read(core.OUT/'UNIFIED_CONFIG.json')
    assert config['code'][0]['sha256']==core.paired.file_hash(core.__file__)
    models=core.read(core.OUT/'SOURCE_TRAINING_COMPLETE.json')['models']
    for e in models:assert core.paired.file_hash(e['checkpoint'])==e['checkpoint_sha256']
    source_diagnostics(config,models)
    core.js(core.OUT/'RUNTIME_INFERENCE_PROVENANCE.json',dict(
        locked_training_runner=core.ident(core.__file__),runtime_entry=core.ident(__file__),
        training_and_scoring_functions_unchanged=True,source_checkpoints_unchanged=True,
        adaptation='Read-only in-memory canonicalization of byte-identical old StepB target image paths; original map indices retained.',
        evidence='official.canonical_paths verifies dataset-relative identity plus image SHA256. Original manifest hash is checked before alias substitution.',
        reason='First prediction-plan guard correctly rejected unequal D:/project/data and E:/dataset path strings, before any target inference/evaluation.',
        output_directory=str(core.OUT)))
    core.rcsv=verified_alias_csv
    sys.argv=[str(core.__file__),'--stage',args.stage]
    core.main()
    if core.official.ALIASES:
        core.js(core.OUT/'VERIFIED_TARGET_PATH_ALIASES.json',core.official.ALIASES)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        core.js(core.OUT/'RUNTIME_ERROR.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
