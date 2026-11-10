"""Evaluate fully committed categories while the single GPU producer continues."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import sys,time,gc,json,traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments import rerun_ntr_lrp_reciprocal_20261101 as run
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
torch.set_num_threads(2)
OUT=run.OUT;WORK=run.WORK;ev=run.ev

def status(**v):ev.savejson(OUT/'STREAM_EVALUATION_STATUS.json',dict(time=time.strftime('%Y-%m-%d %H:%M:%S'),**v));print(v,flush=True)
def category(name,cat,rows,exps,maps,scale3=None):
 ids=[i for i,r in enumerate(rows) if r['category']==cat];scratch=WORK/'stream_scratch'/name/cat;scratch.mkdir(parents=True,exist_ok=True)
 n=sum(int(rows[i]['gt_height'])*int(rows[i]['gt_width']) for i in ids);yp=scratch/'labels.npy';sp=scratch/'scores.npy'
 for p in [yp,sp]:
  if p.exists():assert p.parent.resolve()==scratch.resolve();p.unlink()
 y=np.lib.format.open_memmap(yp,mode='w+',dtype=np.bool_,shape=(n,));s=np.lib.format.open_memmap(sp,mode='w+',dtype=np.float32,shape=(n,));offset=0;masks=[];segments=[]
 for i in ids:
  row=rows[i]
  if row['gt_path']:assert ev.sha(row['gt_path'])==row['gt_sha256']
  mask=ev.native_mask(row);end=offset+mask.size;y[offset:end]=mask.ravel();masks.append(y[offset:end].reshape(mask.shape));segments.append((i,offset,end,mask.shape));offset=end
 scores=[s[a:b].reshape(sh) for i,a,b,sh in segments]
 for ex in exps:
  ed=OUT/'evaluations'/name/ex['id'];ed.mkdir(parents=True,exist_ok=True);jp=ed/(cat+'.json')
  if jp.exists():continue
  image_scores=[]
  for i,a,b,sh in segments:
   g,l=maps[i,ex['head_index']]
   if ex.get('grid')==3:
    large=run.F.interpolate(torch.from_numpy(g.copy())[None,None],size=(111,111),mode='bilinear',align_corners=True)[0,0].numpy();fused=.5*large+.5*scale3[i]
   else:fused=ex['lambda_global']*g+(1-ex['lambda_global'])*l if ex['lrp'] else g
   s[a:b]=ev.resize(fused,sh).ravel();image_scores.append(ev.topmean(ev.resize(fused,(74,74))))
  status(stage='CATEGORY_METRIC',target=name,category=cat,experiment=ex['id'],pixels=n)
  metrics=ev.weighted_pixel_metrics(y,s)
  if ex['full_metrics']:metrics['AUPRO']=ev.aupro(masks,scores)
  iy=[int(rows[i]['label']) for i in ids];metrics.update(I_AUROC=float(roc_auc_score(iy,image_scores)*100),I_AP=float(average_precision_score(iy,image_scores)*100))
  ev.savejson(jp,dict(target=name,category=cat,**ex,**metrics));gc.collect()
 masks.clear();scores.clear();del y,s,mask,fused;gc.collect()
 for p in (yp,sp):p.unlink()
 status(stage='CATEGORY_DONE',target=name,category=cat)

try:
 while True:
  primary=ev.readjson(OUT/'STATUS.json')
  if primary['stage'] in ['FAILED','COMPLETE_RECIPROCAL_NTR_LRP']:break
  worked=False
  for name in ('VisA','MVTec'):
   d=WORK/'predictions'/name
   if not (d/'PROGRESS.json').exists():continue
   rows=ev.readcsv(d/'MANIFEST.csv');entries=ev.readjson(d/'HEAD_ORDER.json');maps=np.load(d/'global_local74.npy',mmap_mode='r');progress=ev.readjson(d/'PROGRESS.json')['committed_images'];exps=run.experiments(entries)
   for cat in sorted({r['category'] for r in rows}):
    ids=[i for i,r in enumerate(rows) if r['category']==cat]
    base=[x for x in exps if x.get('grid')!=3 and not (OUT/'evaluations'/name/x['id']/(cat+'.json')).exists()]
    if base and max(ids)<progress:category(name,cat,rows,base,maps);worked=True;run.summarize()
    sg=d/'SCALE3_PROGRESS.json'
    if sg.exists() and max(ids)<ev.readjson(sg)['committed_images']:
     probes=[x for x in exps if x.get('grid')==3 and not (OUT/'evaluations'/name/x['id']/(cat+'.json')).exists()]
     if probes:
      scale=np.load(d/'local111_seed0.npy',mmap_mode='r');category(name,cat,rows,probes,maps,scale);del scale;worked=True;run.summarize()
   del maps
  if not worked:time.sleep(20)
 status(stage='STREAM_WORKER_STOPPED',primary_status=ev.readjson(OUT/'STATUS.json')['stage'])
except Exception as e:status(stage='FAILED',error=str(e),traceback=traceback.format_exc());raise
