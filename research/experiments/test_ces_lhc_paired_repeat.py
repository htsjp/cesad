"""Real-artifact smoke verification, never a replacement for formal results."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import phase_ces_lhc_paired_repeat as paired


def main():
    out=paired.DEFAULT_OUT/"smoke"
    checks=[]
    for source in ("Full MVTec","Full VisA"):
        pool=out/"source_pools"/(source.replace(" ","_")+"_seed0")
        y=np.load(pool/"labels.npy")
        stream=np.load(pool/"batch_indices.npy")
        manifest=paired.read_csv(pool/"PATCH_MANIFEST.csv")
        cat=np.array([r["category"] for r in manifest])
        groups={}
        for name in sorted(np.unique(cat)):
            ids=np.flatnonzero(cat==name)
            groups[name]=dict(x=ids[:,None].astype(np.float32),y=y[ids])
        rng=np.random.default_rng(0)
        for step in range(len(stream)):
            ids,labels=paired.hist.balanced_batch(groups,rng)
            assert np.array_equal(ids.reshape(-1).astype(np.int32),stream[step])
            assert np.array_equal(labels,y[stream[step]])
        ckpts=[torch.load(p,map_location="cpu",weights_only=False) for p in (out/"checkpoints").glob(source.replace(' ','_')+'_*_primary_*.pt')]
        assert len(ckpts)==3
        assert len({p['batch_stream_hash'] for p in ckpts})==1
        assert len({p['feature_array_hash'] for p in ckpts})==1
        assert all(p['realized_updates']==2 for p in ckpts)
        for step in range(2):
            assert len({p['logs'][step]['batch_ids_hash'] for p in ckpts})==1
        checks.append(dict(check="real_source_sampler_replay_and_pairing",source=source,passed=True))
    # Independently invoke the historical full evaluator on real smoke maps.
    dest=out/"predictions"/paired.hist.pair_slug("Full MVTec","VisA")
    maps=np.load(dest/"maps74.npy",mmap_mode='r')
    masks=np.load(dest/"masks518.npy",mmap_mode='r')
    manifest=paired.read_csv(dest/"TARGET_MANIFEST.csv")
    metas=[dict(category=r['category'],label=int(r['label']),mask=np.asarray(masks[i])) for i,r in enumerate(manifest)]
    old=paired.lzp.VARIANTS
    try:
        paired.lzp.VARIANTS=("TEST",)
        rows,_=paired.lzp.evaluate_maps("Full MVTec","VisA",{"TEST":maps[:,0]},metas)
    finally:
        paired.lzp.VARIANTS=old
    key=paired.hist.pair_slug("Full MVTec","VisA")+"__LinearHead_seed0_primary.json"
    expected=json.loads((out/"evaluations"/key).read_text())['row']['P_AP']
    actual=rows[0]['M_FR_P_AP']
    assert abs(actual-expected)<1e-10,(actual,expected)
    checks.append(dict(check="AP_only_path_equals_historical_full_evaluator",passed=True,max_error=abs(actual-expected)))
    paired.write_json(out/"TEST_RESULTS.json",checks)
    print(json.dumps(checks,indent=2))


if __name__=="__main__":
    torch.set_num_threads(4)
    main()
