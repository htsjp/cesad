"""Full VisA -> DAGM capacity evaluation using existing CMAB states only."""
import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.phase_cmab_compact_multi_anomaly_basis import (
    feature_grid, score_from_grid, state_pair_from_model_info,
)
from experiments.phase_final_cross_dataset_benchmark import load_visa_source
from experiments.phase_dagm_visa_eval import DAGMTargetDataset
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import (
    quadrant_crops, torch_resize, score_feature_grid,
)
from experiments.phase_xlv_cross_domain_transfer import CHECKPOINT
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_sgpr import safe_ap_binary
from phase_xa.official_task_forward import normalized_patch_feature, visual_outputs
from utils.transforms import get_transform

OUT = ROOT/'results/paper_fig5_dagm_capacity'
GLOBAL = ROOT/'results/paper_assets/dagm_visa_eval/cache/dagm_global_patch_float16.npy'
STATE = ROOT/'results/phase_cmab_compact_multi_anomaly_basis/formal'


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--root', default='E:/WorkSpace/dataset/DAGM_KaggleUpload')
    args = parser.parse_args()
    out = OUT/('smoke' if args.smoke else 'formal')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing an unexpectedly long CPU inference run')
    device = 'cuda'
    dataset = DAGMTargetDataset(args.root)
    n = min(8, len(dataset)) if args.smoke else len(dataset)
    global_patch = np.load(GLOBAL, mmap_mode='r')
    assert len(global_patch) == len(dataset) == 8050
    assert global_patch.shape[1:] == (1369,1024)
    info = load_visa_source(device)
    normal,anomaly = state_pair_from_model_info(info)
    states = {2:(normal,anomaly[None])}
    checks = []
    for k,var in [(3,'E1_K2_CMAB'),(5,'E2_K4_CMAB')]:
        path = STATE/f'Full_VisA_{var}_formal.pt'
        payload = torch.load(path,map_location='cpu',weights_only=False)
        assert payload['s_a'].shape == (k-1,1024)
        assert max(int(x['epoch']) for x in payload['curves']) == 10
        states[k] = (payload['s_n'],payload['s_a'])
        checks.append({'K':k,'checkpoint':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'actual_rng_seed_code':k-1})
    frozen_preprocessing = ROOT/'results/phase_statealigned_pc1_debias/formal'
    checks.append({'K':2,'checkpoint':str(ROOT/'results/phase_hard_normal_contrast/formal/contrast_vectors.npz'),
        'direction_key':'E1','fallback':False,'preprocessing':str(frozen_preprocessing)})
    manifest = [{'index':i,'path':x['img_path'],'category':x['cls_name'],'label':x['anomaly']} for i,x in enumerate(dataset.items[:n])]
    fingerprint = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    identity = {'source':'Full VisA','target':'DAGM 2007','n_images':n,'checkpoints':checks,
        'image_order_hash':fingerprint,'global_cache':str(GLOBAL),'LRP':'2x2 nonoverlap;518;0.5global+0.5local;74grid',
        'metric':'full-resolution category-macro M_FR P-AP; bilinear align_corners=True; no extra smoothing',
        'train':False,'scope':'same historical capacity configuration family; actual RNG seeds differ'}
    ipath = out/'CONFIG.json'
    if ipath.exists():
        assert json.loads(ipath.read_text(encoding='utf-8')) == identity, 'Resume identity mismatch'
    else:
        write_json(ipath,identity)
        pd.DataFrame(manifest).to_csv(out/'IMAGE_MANIFEST.csv',index=False)
    maps = {}
    for k in states:
        path = out/f'K{k}_LRP_maps.npy'
        if path.exists():
            maps[k] = np.load(path,mmap_mode='r+')
            assert maps[k].shape == (n,74,74)
        else:
            maps[k] = np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(n,74,74))
    progress_path = out/'PROGRESS.json'
    done = json.loads(progress_path.read_text())['completed_images'] if progress_path.exists() else 0
    start = time.time()
    initial = done
    if done < n:
        preprocess,_ = get_transform(argparse.Namespace(image_size=518))
        clip,*_ = load_components(str(CHECKPOINT),device)
        clip.eval()
        for parameter in clip.parameters():
            parameter.requires_grad_(False)
        with torch.inference_mode():
            for i in range(done,n):
                crops = quadrant_crops(dataset.items[i]['img_path'])
                assert len(crops) == 4
                images = torch.stack([preprocess(c) for c in crops]).to(device)
                patch = normalized_patch_feature(visual_outputs(clip,images,adapter=None,grad=False)).float().cpu().numpy()
                zg = feature_grid(info,global_patch[i],device)
                zl = [feature_grid(info,x,device) for x in patch]
                for k,state in states.items():
                    g = score_from_grid(zg,state,device)
                    local = [score_from_grid(z,state,device) for z in zl]
                    stitched = np.block([[local[0],local[1]],[local[2],local[3]]])
                    maps[k][i] = .5*torch_resize(g,(74,74),align_corners=True)+.5*stitched
                    assert np.isfinite(maps[k][i]).all()
                    if i == 0 and k == 2:
                        baseline = score_feature_grid(info,global_patch[i],device)
                        error=float(np.max(np.abs(g-baseline)))
                        assert error < 1e-5,error
                        write_json(out/'SMOKE_PARITY.json',{'K2_global_max_abs_error':error,'quadrants':4,'shape':[74,74]})
                del images,patch,zg,zl
                if (i+1)%25 == 0 or i+1 == n:
                    for arr in maps.values(): arr.flush()
                    elapsed=time.time()-start
                    eta=elapsed/max(i+1-initial,1)*(n-i-1)
                    status={'completed_images':i+1,'total_images':n,'stage':'inference','elapsed_seconds':elapsed,'eta_seconds':eta}
                    write_json(progress_path,status)
                    print(f'[DAGM capacity] {i+1}/{n}, elapsed={elapsed/60:.1f}min eta={eta/60:.1f}min',flush=True)
        del clip
        gc.collect();torch.cuda.empty_cache()
    if args.smoke:
        write_json(out/'OUTCOME.json',{'outcome':'SMOKE_PASSED','n_images':n,'all_maps_finite':True})
        print('SMOKE_PASSED',flush=True)
        return
    # Original-resolution masks and scores, grouped by category exactly as M_FR.
    # Reuse the formal safe_ap_binary helper; never resize GT onto the patch grid.
    catpath=out/'PER_CATEGORY_P_AP.csv'
    rows=pd.read_csv(catpath).to_dict('records') if catpath.exists() else []
    for category in sorted({x['cls_name'] for x in dataset.items}):
        ids=[i for i,x in enumerate(dataset.items) if x['cls_name']==category]
        masks=[dataset[i]['img_mask'].numpy().squeeze().astype(np.uint8) for i in ids]
        y=np.concatenate([m.reshape(-1) for m in masks])
        for k in states:
            if any(r['category']==category and int(r['K'])==k for r in rows): continue
            print(f'[DAGM capacity] full-resolution AP {category} K{k}',flush=True)
            s=np.concatenate([torch_resize(maps[k][i],m.shape,align_corners=True).reshape(-1) for i,m in zip(ids,masks)])
            ap=float(safe_ap_binary(y,s))
            rows.append({'source':'Full VisA','target':'DAGM 2007','category':category,'K':k,'M_FR_P_AP':ap,'n_images':len(ids),'GT_resolution':str(masks[0].shape),'aggregation':'CATEGORY'})
            pd.DataFrame(rows).to_csv(catpath,index=False)
            del s;gc.collect()
        del y,masks;gc.collect()
    raw=[]
    for k in states:
        values=[r['M_FR_P_AP'] for r in rows if int(r['K'])==k]
        assert len(values)==10 and np.isfinite(values).all()
        raw.append({'source':'Full VisA','target':'DAGM 2007','K':k,'M_FR_P_AP':float(np.mean(values)),'aggregation':'CATEGORY_MACRO','n_images':n,'n_categories':10})
    pd.DataFrame(raw).to_csv(out/'DAGM_CAPACITY_METRICS.csv',index=False)
    write_json(out/'OUTCOME.json',{'outcome':'DAGM_CAPACITY_EVALUATION_COMPLETE','training':False,'metrics':raw})
    (out/'REPORT.md').write_text('# DAGM State Capacity\n\nExisting Full VisA CMAB-family states; no training. Full-resolution category-macro M_FR P-AP. See CONFIG.json for checkpoint and RNG qualifications. Historical DAGM_GRID74 pixel results are not interchangeable with these metrics.\n\n'+pd.DataFrame(raw).to_string(index=False)+'\n',encoding='utf-8')
    print(pd.DataFrame(raw).to_string(index=False),flush=True)


if __name__=='__main__':
    main()
