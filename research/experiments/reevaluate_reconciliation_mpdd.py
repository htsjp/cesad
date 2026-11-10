"""Optional cache-only MPDD evaluation. No extraction, optimization or fallback.

Not invoked by the audit. Requires verified raw (pre-source-transform) local
features and a matching image-path manifest before any scoring is possible.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=['heads', 'components'], required=True)
    parser.add_argument('--local-cache', type=Path, default=Path('E:/MRAD_final_evidence_cache/local_patch_Full_VisA__MPDD.npy'))
    parser.add_argument('--local-manifest', type=Path, help='JSON list of img_path strings in local-cache order')
    args = parser.parse_args()
    if not args.local_cache.is_file():
        raise FileNotFoundError(f'Missing verified local feature cache: {args.local_cache}')
    if args.local_manifest is None or not args.local_manifest.is_file():
        raise FileNotFoundError('Pass --local-manifest with verified raw local-feature image order')
    from experiments.phase_ces_linear_head_control import HeadSpec
    from experiments.phase_median_spatial_consensus_generalization import HOLDOUTS, MetaDataset
    from experiments.phase_parameter_free_local_zoom_pyramid_fullres import item_at, item_meta, mask_np
    import experiments.unify_corrected_ntr_category_macro as unify
    source, target = 'Full VisA', 'MPDD'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global_path = ROOT/'results/phase_median_spatial_consensus_generalization/formal/feature_cache/MPDD_clip.npz'
    if not global_path.is_file():
        raise FileNotFoundError(global_path)
    with np.load(global_path, allow_pickle=True) as data:
        patch = data['patch']
        refs = [dict(r) for r in data['rows']]
    local = np.load(args.local_cache, mmap_mode='r')
    paths = json.loads(args.local_manifest.read_text(encoding='utf-8'))
    canonical = lambda p: str(p).replace('\\','/').casefold()
    assert len(local) == len(patch) == len(refs) == len(paths)
    assert all(canonical(p)==canonical(r['img_path']) for p,r in zip(paths,refs)), 'Image-order mismatch'
    assert local.shape[1:] in [(4,1369,1024),(4,37,37,1024)], local.shape
    dataset = MetaDataset(HOLDOUTS[target][0], mode='test')
    info = unify.source_scoring_info(source,device)
    output = ROOT/f'results/paper_protocol_reconciliation/reevaluated_{args.family}'
    output.mkdir(parents=True,exist_ok=True)
    raw,percat=[],[]
    for seed in ([0] if args.family=='heads' else [0,1,2]):
        specs,checkpoint={},{}
        if args.family=='heads':
            for method in ('CES','LinearHead','NormalizedLinearHead'):
                path=ROOT/f'results/phase_ces_linear_head_control/checkpoints/Full_VisA_{method}_seed0.pt'
                payload=torch.load(path,map_location='cpu',weights_only=False)
                assert payload['source']==source
                params={k:v for k,v in payload.items() if k not in ('method','source','seed')}
                specs[method]=HeadSpec(method,params,source,seed=seed)
                checkpoint[method]=str(path)
        else:
            for variant in ('Trained_CES_Core','Trained_CES_NTR','Trained_CES_LRP','Trained_CES_NTR_LRP'):
                ntr='NTR' in variant
                if seed==0:
                    path=ROOT/('results/paper_assets/final_ablation_close/checkpoints/Full_VisA_seed0_rho0.10.pt' if ntr else 'results/ces_ntr_visualization/corrected/checkpoints/Full_VisA_seed0_statealigned.pt')
                else:
                    path=ROOT/f'results/phase_paper_ready_final_validation/formal/checkpoints/Full_VisA_seed{seed}_{"ntr" if ntr else "statealigned"}.pt'
                v=unify.load_v(str(path))
                specs[variant]=HeadSpec('CES',{'m':info['m'],'alpha':info['alpha'],'v':v},source,seed=seed)
                checkpoint[variant]=str(path)
        maps={v:[] for v in specs}
        metas=[]
        for pos,ref in enumerate(refs):
            item=item_at(dataset,'external',ref,pos)
            cat,label,path=item_meta(item,'external',ref,target)
            assert canonical(path)==canonical(paths[pos])
            metas.append({'category':cat,'label':int(label),'mask':mask_np(item)})
            core={v:s for v,s in specs.items() if args.family=='components' and not v.endswith('_LRP')}
            lrp={v:s for v,s in specs.items() if v not in core}
            batch=unify.score_patch_grid_multi(info,core,patch[pos],device) if core else {}
            if lrp:
                batch.update(unify.lrp_maps_multi(info,lrp,patch[pos],np.asarray(local[pos],dtype=np.float32),device))
            for variant,score in batch.items():
                maps[variant].append(score)
        unify.VARIANTS=tuple(specs)
        rr,cc=unify.evaluate_pap_category_macro_only(source,target,maps,metas)
        for r in rr:
            r.update(seed=seed,checkpoint=checkpoint[r['variant']],cache=str(args.local_cache))
        for c in cc:
            c['seed']=seed
        raw.extend(rr);percat.extend(cc)
        unify.write_csv(output/'RAW.csv',raw)
        unify.write_csv(output/'PER_CATEGORY.csv',percat)
    print(output)


if __name__=='__main__':
    main()
