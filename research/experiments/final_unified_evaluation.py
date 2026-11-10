"""Predeclared frozen-model, native-GT evaluation. No training or selection.

The six named deliverables alone live in results/final_unified_evaluation.
Manifests, per-category results, logs and new prediction chunks live separately.
Historical results, checkpoints and manuscripts are read-only inputs.
"""
from __future__ import annotations
import argparse, csv, gc, hashlib, inspect, json, math, os, re, subprocess, sys, time, traceback
from collections import Counter, defaultdict, OrderedDict
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
WORK=ROOT/'results/final_unified_evaluation_work'
FINAL=ROOT/'results/final_unified_evaluation'
SCRATCH=Path('C:/Users/Administrator/.cache/cesad_final_unified_20261020')
DATA=Path('E:/WorkSpace/dataset')
METHODS=('AnomalyCLIP','FAPrompt','MRAD-FT','MRAD-CLIP','VisualAD','FB-CLIP','CESAD')
TARGETS=('MVTec','VisA','BTAD','MPDD','DTD-Synthetic','DAGM')
SOURCE={t:('Full MVTec' if t=='VisA' else 'Full VisA') for t in TARGETS}
ROOTS={'MVTec':DATA/'mvtec','VisA':DATA/'VISA','BTAD':DATA/'btad/BTech_Dataset_transformed',
       'MPDD':DATA/'MPDD','DTD-Synthetic':DATA/'DTD-Synthetic/DTD-Synthetic/DTD-Synthetic',
       'DAGM':DATA/'DAGM_KaggleUpload'}
POLICY={'version':2,'sources':SOURCE,'methods':METHODS,'pixel_gt':'paired original mask resolution; normal image-sized zero mask',
 'resize':'torch bilinear align_corners=True; no antialias; float32',
 'image_grid':[74,74],'top_fraction':.01,'top_k':'ceil; minimum 1','image_native':'final method map before evaluator resize',
 'aggregation':'equal-weight valid target categories, independently per metric',
 'validity':'both positive and negative labels required for all AUROC/AP/F1; PRO requires >=1 region and background',
 'AUPRO':{'connectivity':8,'thresholds':200,'threshold_grid':'linear over category score min/max, strict score>threshold',
          'fpr_cap':.30,'endpoints':'include (0,0) and (1,1); preserve vertical segments in threshold order',
          'integration':'piecewise linear interpolation at FPR=0.30; trapezoid on [0,.30] divided by .30'},
 'AP':'sklearn.average_precision_score with exact score-tie multiplicities, no binning/quantization',
 'training':False,'target_selection':False,'manuscript':str(Path('D:/ProjectPython/Paper/CESAD_V3_Final_Alignment_DAGM_20261020.docx'))}

def normpath(x):
 p=Path(x)
 if not p.is_absolute():p=ROOT/p
 return str(p.resolve()).replace('\\','/').casefold()
def readjson(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def savejson(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8');tmp.replace(p)
def writecsv(p,rows,fields=None):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if fields is None:fields=list(dict.fromkeys(k for r in rows for k in r))
 tmp=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp')
 with tmp.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
 tmp.replace(p)
def readcsv(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def lock_policy():
 WORK.mkdir(parents=True,exist_ok=True);SCRATCH.mkdir(parents=True,exist_ok=True)
 p=WORK/'POLICY.json';expected=json.loads(json.dumps(POLICY))
 if p.exists():assert readjson(p)==expected,'Refuse resume under changed policy'
 else:savejson(p,expected)
def native_mask(r):
 if int(r['label']):
  with Image.open(r['gt_path']) as im:m=np.asarray(im.convert('L'))>0
 else:m=np.zeros((int(r['gt_height']),int(r['gt_width'])),dtype=bool)
 assert m.shape==(int(r['gt_height']),int(r['gt_width']))
 return m
def catalogue(t):
 root=ROOTS[t];rows=[]
 if t=='DAGM':
  for c in sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith('Class')],key=lambda p:int(p.name[5:])):
   for ip in sorted((c/'Test').glob('*.PNG')):
    gp=c/'Test/Label'/f'{ip.stem}_label.PNG'
    rows.append({'category':c.name,'img_path':str(ip),'gt_path':str(gp) if gp.exists() else '', 'label':int(gp.exists())})
 else:
  meta=readjson(root/'meta.json')['test']
  assert t!='DTD-Synthetic' or len(meta)==12
  for c,rr in meta.items():
   for r in rr:
    ip=root/r['img_path'];label=int(r['anomaly']);gp=''
    if label:
     if t=='DTD-Synthetic':
      parts=list(Path(r['img_path']).parts);parts[parts.index('test')]='ground_truth';parts[-1]=Path(parts[-1]).stem+'_mask.png'
      gp=str(root.joinpath(*parts))
     elif t=='BTAD' and not r.get('mask_path'):
      matches=list((root/c/'ground_truth/ko').glob(ip.stem+'.*'))
      assert len(matches)==1,(ip,matches)
      gp=str(matches[0])
     else:gp=str(root/r['mask_path'])
    rows.append({'category':c,'img_path':str(ip),'gt_path':gp,'label':label})
 seen=set()
 for i,r in enumerate(rows):
  ip=Path(r['img_path']);assert ip.is_file(),ip
  assert normpath(ip) not in seen,'Duplicate target image';seen.add(normpath(ip))
  with Image.open(ip) as im:iw,ih=im.size
  r.update(dataset=t,index=i,image_height=ih,image_width=iw,sample_id=t+'/'+str(ip.relative_to(root)).replace('\\','/'))
  if r['label']:
   gp=Path(r['gt_path']);assert gp.is_file() and normpath(gp)!=normpath(ip),(ip,gp)
   with Image.open(gp) as im:
    gw,gh=im.size;mask=np.asarray(im.convert('L'))>0
   r.update(gt_height=gh,gt_width=gw,positive_pixels=int(mask.sum()),gt_sha256=sha(gp),
            annotation_note='' if mask.any() else 'ORIGINAL_MASK_EMPTY_WITH_ANOMALY_IMAGE_LABEL')
  else:r.update(gt_height=ih,gt_width=iw,positive_pixels=0,gt_sha256='ZERO_MASK',annotation_note='')
  r['image_sha256']=sha(ip)
 return rows
def build_manifests():
 summary=[]
 for t in TARGETS:
  path=WORK/'manifests'/f'{t}.csv'
  if path.exists():rows=readcsv(path)
  else:
   rows=catalogue(t);writecsv(path,rows)
  cats=sorted({r['category'] for r in rows})
  row={'target':t,'source':SOURCE[t],'images':len(rows),'categories':len(cats),
   'anomalous_images':sum(int(r['label']) for r in rows),'native_shapes':dict(Counter(f"{r['gt_height']}x{r['gt_width']}" for r in rows)),
   'pixel_valid_categories':sum(any(int(r['positive_pixels'])>0 for r in rows if r['category']==c) and any(int(r['positive_pixels'])<int(r['gt_height'])*int(r['gt_width']) for r in rows if r['category']==c) for c in cats),
   'image_valid_categories':sum(len({int(r['label']) for r in rows if r['category']==c})==2 for c in cats),
   'annotation_exceptions':[r['sample_id'] for r in rows if r.get('annotation_note')],
   'manifest':str(path),'manifest_sha256':sha(path)}
  summary.append(row);print('MANIFEST',json.dumps(row),flush=True)
 savejson(WORK/'MANIFEST_SUMMARY.json',summary)
 return summary
def resize(a,shape):
 a=np.asarray(a,dtype=np.float32)
 if a.shape==tuple(shape):return a
 return F.interpolate(torch.from_numpy(a.copy()).view(1,1,*a.shape),size=tuple(map(int,shape)),mode='bilinear',align_corners=True).view(*shape).numpy()
def topmean(a):
 x=np.asarray(a,dtype=np.float32).reshape(-1);k=max(1,math.ceil(.01*len(x)))
 return float(np.mean(np.partition(x,len(x)-k)[-k:],dtype=np.float64))
def evaluator_hash():
 text=json.dumps(POLICY,sort_keys=True)+''.join(inspect.getsource(f) for f in [resize,topmean,weighted_pixel_metrics,integrate_pro_curve,aupro])
 return hashlib.sha256(text.encode()).hexdigest()
def weighted_pixel_metrics(y,s):
 """Exact sufficient statistics per distinct score; AP still uses sklearn.

 No pixels are sampled, no score is rounded and ties are not split. Aggregating
 same-label/equal-score pixels with integer weights preserves all PR/ROC points.
 """
 y=np.asarray(y,dtype=bool);s=np.asarray(s,dtype=np.float32)
 if not y.any() or y.all():return dict(P_AUROC=None,P_F1max=None,P_AP=None)
 unique,total=np.unique(s,return_counts=True)
 posu,posc=np.unique(s[y],return_counts=True)
 pos=np.zeros(len(unique),dtype=np.int64);pos[np.searchsorted(unique,posu)]=posc
 neg=total-pos;ip=pos>0;ineg=neg>0
 yy=np.concatenate((np.ones(ip.sum(),dtype=np.uint8),np.zeros(ineg.sum(),dtype=np.uint8)))
 ss=np.concatenate((unique[ip],unique[ineg]));ww=np.concatenate((pos[ip],neg[ineg])).astype(np.float64)
 del unique,total,posu,posc,pos,neg;gc.collect()
 ap=float(average_precision_score(yy,ss,sample_weight=ww)*100)
 auc=float(roc_auc_score(yy,ss,sample_weight=ww)*100)
 p,r,_=precision_recall_curve(yy,ss,sample_weight=ww,drop_intermediate=False)
 f=np.divide(2*p*r,p+r,out=np.zeros_like(p),where=(p+r)>0)
 return dict(P_AUROC=auc,P_F1max=float(f.max()*100),P_AP=ap)
def integrate_pro_curve(fpr,pro,cap=.3):
 """Do not replace vertical ROC/PRO segments with their upper envelope."""
 xx=np.asarray(fpr,dtype=np.float64);yy=np.asarray(pro,dtype=np.float64)
 assert np.all(np.diff(xx)>=0),'PRO points must follow decreasing thresholds'
 j=int(np.searchsorted(xx,cap,side='right'))
 x=xx[:j];y=yy[:j]
 if x[-1]<cap:
  yc=yy[j-1]+(cap-xx[j-1])/(xx[j]-xx[j-1])*(yy[j]-yy[j-1])
  x=np.r_[x,cap];y=np.r_[y,yc]
 return float(np.trapezoid(y,x)/cap*100)
def aupro(masks,scores):
 """Histogram accumulation reproduces the declared 200 thresholds exactly."""
 mn=min(float(a.min()) for a in scores);mx=max(float(a.max()) for a in scores)
 thresholds=np.linspace(mn,mx,200,dtype=np.float32)
 neg_hist=np.zeros(201,dtype=np.float64);reg_hist=np.zeros(201,dtype=np.float64)
 neg_count=0;regions=0
 for m,s in zip(masks,scores):
  bins=np.searchsorted(thresholds,s,side='left')
  neg=~m;neg_count+=int(neg.sum());neg_hist+=np.bincount(bins[neg],minlength=201)
  labels,n=ndimage.label(m,structure=np.ones((3,3),dtype=np.uint8))
  if n:
   areas=np.bincount(labels.reshape(-1));weights=1.0/areas[labels[m]]
   reg_hist+=np.bincount(bins[m],weights=weights,minlength=201);regions+=n
 if not neg_count or not regions:return None
 fpr=np.cumsum(neg_hist[::-1])[::-1][1:]/neg_count
 pro=np.cumsum(reg_hist[::-1])[::-1][1:]/regions
 fpr=np.r_[0.,fpr[::-1],1.];pro=np.r_[0.,pro[::-1],1.]
 return integrate_pro_curve(fpr,pro)
def selftest():
 rng=np.random.default_rng(70421)
 for n in [19,1000,5000]:
  for ties in [True,False]:
   y=rng.integers(0,2,n,dtype=np.uint8);s=rng.normal(size=n).astype(np.float32)
   if ties:s=np.round(s,1)
   got=weighted_pixel_metrics(y,s);p,r,_=precision_recall_curve(y,s)
   expected=[roc_auc_score(y,s)*100,np.max(np.divide(2*p*r,p+r,out=np.zeros_like(p),where=(p+r)>0))*100,average_precision_score(y,s)*100]
   assert np.allclose(list(got.values()),expected,rtol=1e-10,atol=1e-8),(got,expected)
 # Independent brute-force PRO curve under the SAME predeclared integration.
 m=np.zeros((3,9,11),bool);m[0,1:4,2:4]=1;m[1,6:8,7:10]=1
 s=rng.normal(size=m.shape).astype(np.float32);th=np.linspace(s.min(),s.max(),200,dtype=np.float32)
 labels=[ndimage.label(x,np.ones((3,3)))[0] for x in m];fr=[];pr=[]
 for v in th:
  pred=s>v;fr.append((pred&~m).sum()/(~m).sum());a=[]
  for mm,ll,pp in zip(m,labels,pred):
   for rid in range(1,ll.max()+1):a.append(pp[ll==rid].mean())
  pr.append(np.mean(a))
 fr=np.r_[0,np.asarray(fr)[::-1],1];pr=np.r_[0,np.asarray(pr)[::-1],1]
 keep=fr<=.3;xx=fr[keep];pp=pr[keep]
 if xx[-1]<.3:
  j=np.flatnonzero(~keep)[0]
  pp=np.r_[pp,pp[-1]+(.3-xx[-1])*(pr[j]-pp[-1])/(fr[j]-xx[-1])];xx=np.r_[xx,.3]
 ref=np.trapezoid(pp,xx)/.3*100
 assert abs(aupro(m,s)-ref)<1e-9
 # Hand-computable vertical-segment curve: area=.25, normalized=.25/.30.
 assert abs(integrate_pro_curve([0,0,.1,.1,.3,1],[0,.5,.5,1,1,1])-100*.25/.3)<1e-10
 assert len(np.arange(74*74)[-max(1,math.ceil(.01*74*74)):])==55
 assert topmean(np.arange(74*74,dtype=np.float32).reshape(74,74))==np.arange(74*74)[-55:].mean()
 # Verify cache compression is byte-exact, including signed zero/extremes.
 import io
 a=rng.normal(size=(2,19,23)).astype(np.float32);a.reshape(-1)[:4]=[0.,-0.,np.finfo(np.float32).tiny,np.finfo(np.float32).max]
 bits=a.view(np.uint32).reshape(-1);d=bits.copy();d[1:]^=bits[:-1]
 stream=io.BytesIO();np.savez_compressed(stream,maps_xor=d.view(np.uint8).reshape(-1,4).T.copy(),shape=np.asarray(a.shape),codec='float32_xor_byteplane_v1')
 stream.seek(0)
 with np.load(stream) as z:assert np.array_equal(unpack_chunk(z).view(np.uint32),a.view(np.uint32))
 savejson(WORK/'SELFTEST.json',{'status':'PASS','weighted_AP_AUROC_full_PR_F1':'equal to uncompressed sklearn including ties','AUPRO':'equal to independent brute force','top_k_74':55})
 print('SELFTEST PASS',flush=True)

def legacy_paths(t):
 """Reconstruct the immutable producer's explicit dataset ordering."""
 if t in ('MVTec','VisA'):
  from experiments.phase_defect_query_transfer_pilot import light_fold
  _,ds,tr,held=light_fold('D1' if t=='MVTec' else 'D2',0)
  return [str(Path(ds.root)/ds.data_all[i]['img_path']) for i in list(tr)+list(held)]
 if t=='BTAD':
  from experiments.phase_liv_memory_support_modes import XLV_A
  with np.load(XLV_A/'feature_cache/BTAD_clip.npz',allow_pickle=True) as z:rr=list(z['rows'])
  return [str(r['img_path']) for r in rr]
 if t=='DTD-Synthetic':
  d=readjson(ROOTS[t]/'meta.json')['test']
  return [str(ROOTS[t]/r['img_path']) for c in sorted(d) for r in d[c]]
 return [r['img_path'] for r in readcsv(WORK/'manifests'/f'{t}.csv')]

ALIASES={}
def canonical_paths(t,paths,rows):
 direct={normpath(r['img_path']):r for r in rows}
 rel={str(Path(r['img_path']).relative_to(ROOTS[t])).replace('\\','/').casefold():r for r in rows}
 mapped=[]
 for p in paths:
  key=normpath(p)
  if key in direct:mapped.append(direct[key]['img_path']);continue
  parts=str(p).replace('\\','/').casefold().split('/');matches=[]
  for i in range(len(parts)):
   suffix='/'.join(parts[i:])
   if suffix in rel:matches.append(rel[suffix])
  assert len(matches)==1,(t,p,'Cannot uniquely map dataset-relative sample ID')
  r=matches[0];old=Path(p) if Path(p).is_absolute() else ROOT/p
  assert old.is_file(),f'Cannot verify relocated image bytes: {old}'
  if key not in ALIASES:
   assert sha(old)==r['image_sha256'],f'Image copies differ: {old} versus {r["img_path"]}'
   ALIASES[key]={'canonical':r['img_path'],'sha256':r['image_sha256'],'status':'BYTE_IDENTICAL_IMAGE_COPY'}
  mapped.append(r['img_path'])
 return mapped

def make_registry():
 protocols=readcsv(ROOT/'results/paper_assets/reproduction_details/BASELINE_EVALUATION_PROTOCOL.csv')
 final2=ROOT/'results/final_dtd_dagm_metric_unification_work'
 allrows=[];legacy={}
 for t in TARGETS:
  rows=readcsv(WORK/'manifests'/f'{t}.csv');canon={normpath(r['img_path']):i for i,r in enumerate(rows)}
  for method in METHODS:
   pp=next(p for p in protocols if p['Method']==method and p['Source']==SOURCE[t] and p['Target'].replace(' 2007','')==t)
   cp=pp['Checkpoint'];source=SOURCE[t];slug=source.replace(' ','_')+'__'+t
   entry={'method':method,'source':source,'target':t,'checkpoint':cp,'postprocessing':pp['Postprocessing'],
     'official_training_provenance':'UNKNOWN original released training population/selection history',
     'manifest':str(WORK/'manifests'/f'{t}.csv'),'manifest_sha256':sha(WORK/'manifests'/f'{t}.csv'),
     'cache_status':'MISSING','cache_path':'','cache_key':'maps','order_verification':'','n_samples':len(rows)}
   path=None;paths=None;key='maps';proof=''
   if method=='FAPrompt' and t!='DAGM':
    candidate=ROOT/f'results/paper_assets/baseline_close/map_cache/FAPrompt_{slug}.npz'
    if candidate.exists() and candidate.with_suffix('.json').exists():
     path=candidate;paths=[x['img_path'] for x in readjson(candidate.with_suffix('.json'))];proof='SAVED_IMAGE_PATH_MANIFEST'
   if method in ('MRAD-FT','MRAD-CLIP') and t!='DAGM':
    v='M2_MRAD_FT' if method=='MRAD-FT' else 'M3_MRAD_CLIP'
    base='paper_assets/mrad_dtd_close' if t=='DTD-Synthetic' else 'phase_mva_mrad_same_protocol_visa_factorization/formal'
    candidate=ROOT/f'results/{base}/MAP_CACHE/{slug}__{v}.npz'
    if candidate.exists():
     if t not in legacy:legacy[t]=legacy_paths(t)
     paths=canonical_paths(t,legacy[t],rows);path=candidate;proof='ORIGINAL_PRODUCER_ORDER_RECONSTRUCTED_WITH_PER_IMAGE_INDEX_AND_CATEGORY_CHECK'
     bypath={normpath(r['img_path']):r for r in rows}
     with np.load(path,allow_pickle=True) as z:
      rr=z['retrieval_rows'];assert len(rr)==len(paths)
      assert all(int(r['image_index'])==i and r['category']==bypath[normpath(paths[i])]['category'] for i,r in enumerate(rr))
   if method=='CESAD':
    if t=='DTD-Synthetic':
     candidate=final2/'capacity_DTD_predictions/K2.npy'
     if candidate.exists():
      path=candidate;key='';paths=legacy_paths(t);proof='SAVED_IDENTITY_HASH_AND_ORIGINAL_CATALOGUE'
      identity=readjson(candidate.parent/'identity.json')
      from experiments.final_dtd_dagm_metrics import catalogue as old_catalogue
      hh=hashlib.sha256(json.dumps(old_catalogue(t),sort_keys=True).encode()).hexdigest()
      assert hh==identity['images_hash'],'DTD CESAD source catalogue changed'
    elif t=='DAGM':
     candidate=ROOT/'results/paper_fig5_dagm_capacity/formal/K2_LRP_maps.npy'
     if candidate.exists():
      path=candidate;key='';paths=[x['path'] for x in readcsv(candidate.parent/'IMAGE_MANIFEST.csv')];proof='SAVED_IMAGE_PATH_MANIFEST'
    else:
     candidate=ROOT/f'results/phase_fova_field_of_view_amplification_audit/formal/MAP_CACHE/{slug}.npz'
     if candidate.exists():
      if t not in legacy:legacy[t]=legacy_paths(t)
      path=candidate;key='q_lrp';paths=legacy[t];proof='ORIGINAL_FOVA_PRODUCER_ORDER_RECONSTRUCTED'
   if path:
    paths=canonical_paths(t,paths,rows)
    assert len(paths)==len(rows) and len(set(map(normpath,paths)))==len(rows)
    lookup={normpath(p):i for i,p in enumerate(paths)}
    assert set(lookup)==set(canon),(method,t,'test image set differs')
    entry.update(cache_status='REUSABLE',cache_path=str(path),cache_key=key,
      canonical_to_cache_index=[lookup[normpath(r['img_path'])] for r in rows],order_verification=proof,
      cache_sha256=sha(path))
   else:
    entry.update(cache_path=str(SCRATCH/'predictions'/f'{method}_{t}'),cache_key='',
      canonical_to_cache_index=list(range(len(rows))),order_verification='NEW_INFERENCE_WILL_USE_HASHED_CANONICAL_MANIFEST')
   cp0=cp.split(';')[0].split('::')[0]
   assert Path(cp0).is_file(),(method,t,'Missing formal checkpoint',cp0)
   entry.update(checkpoint_file=cp0,checkpoint_sha256=sha(cp0))
   allrows.append(entry)
   print('REGISTRY',method,t,entry['cache_status'],proof or 'needs frozen-checkpoint inference',flush=True)
 savejson(WORK/'PREDICTION_REGISTRY.json',allrows)
 savejson(WORK/'PATH_ALIAS_VERIFICATION.json',ALIASES)
 return allrows

def unpack_chunk(z):
 if 'maps' in z:return z['maps']
 assert str(z['codec'])=='float32_xor_byteplane_v1'
 packed=z['maps_xor'];shape=tuple(z['shape'])
 bits=np.bitwise_xor.accumulate(packed.T.copy().view(np.uint32).reshape(-1))
 return bits.view(np.float32).reshape(shape)

def save_chunk(p,maps):
 """Lossless float-bit decorrelation, no numerical quantization."""
 maps=np.ascontiguousarray(maps,dtype=np.float32);bits=maps.view(np.uint32).reshape(-1)
 delta=bits.copy();delta[1:]^=bits[:-1]
 packed=delta.view(np.uint8).reshape(-1,4).T.copy()
 with Path(p).open('wb') as f:
  np.savez_compressed(f,maps_xor=packed,shape=np.asarray(maps.shape,dtype=np.int32),codec='float32_xor_byteplane_v1')

class Store:
 def __init__(self,entry):
  self.entry=entry;self.cache=OrderedDict();self.maps=None
  p=Path(entry['cache_path'])
  if p.is_file():
   if entry['cache_key']:
    with np.load(p,allow_pickle=True) as z:self.maps=z[entry['cache_key']]
   else:self.maps=np.load(p,mmap_mode='r')
   assert len(self.maps)==entry['n_samples']
 def __getitem__(self,index):
  i=self.entry['canonical_to_cache_index'][index]
  if self.maps is not None:return np.asarray(self.maps[i],dtype=np.float32)
  chunk=i//16
  if chunk not in self.cache:
   with np.load(Path(self.entry['cache_path'])/f'chunk_{chunk:05d}.npz') as z:self.cache[chunk]=unpack_chunk(z)
   while len(self.cache)>2:self.cache.popitem(last=False)
  return np.asarray(self.cache[chunk][i%16],dtype=np.float32)

def _predict_cell(method,t,limit=None):
 entries=readjson(WORK/'PREDICTION_REGISTRY.json');e=next(x for x in entries if x['method']==method and x['target']==t)
 assert e['cache_status']=='MISSING'
 out=Path(e['cache_path']);out.mkdir(parents=True,exist_ok=True)
 rows=readcsv(e['manifest']);assert sha(e['manifest'])==e['manifest_sha256']
 identity={k:e[k] for k in ['source','method','target','checkpoint','checkpoint_sha256','manifest_sha256']}
 identity.update(precision='float32',chunk_size=16,training=False)
 ip=out/'identity.json'
 if ip.exists():assert readjson(ip)==identity
 else:savejson(ip,identity)
 if (out/'COMPLETE.json').exists():return
 import shutil
 if shutil.disk_usage(out).free<5*1024**3:raise RuntimeError('Prediction disk has less than 5 GiB; no data deleted')
 # Prime the project package before official repos prepend a standalone
 # utils.py; the original catalogue-based runner establishes the same order.
 import utils.dataset, utils.transforms
 from experiments import final_dtd_dagm_metrics as original
 original.SOURCE=e['source']
 if method.startswith('MRAD-'):
  from experiments.phase_mva_mrad_same_protocol_visa_factorization import load_mrad_bundle,mrad_infer_one
  v='M2_MRAD_FT' if method=='MRAD-FT' else 'M3_MRAD_CLIP'
  bundle=load_mrad_bundle(e['source'],v,'cuda')
  def pred(paths):return [mrad_infer_one(bundle,p,'cuda')[0] for p in paths]
  batch=1
 else:
  transform,predict,loaded=original.predictor(method)
  assert normpath(loaded)==normpath(e['checkpoint_file']),(loaded,e['checkpoint_file'])
  def pred(paths):
   ims=[]
   for p in paths:
    with Image.open(p) as im:ims.append(transform(im.convert('RGB')))
   return predict(torch.stack(ims).cuda())
  batch=1 if method=='FAPrompt' else 4
 n=len(rows) if limit is None else min(limit,len(rows));start=time.time();nnew=0
 with torch.inference_mode():
  for c,beg in enumerate(range(0,n,16)):
   p=out/f'chunk_{c:05d}.npz'
   if p.exists():continue
   rr=rows[beg:min(beg+16,n)];maps=[]
   for pos in range(0,len(rr),batch):
    paths=[x['img_path'] for x in rr[pos:pos+batch]]
    try:predicted=pred(paths)
    except torch.cuda.OutOfMemoryError:
     torch.cuda.empty_cache();predicted=[];batch=1
     for single in paths:predicted.extend(pred([single]))
    for a in predicted:
     a=np.squeeze(np.asarray(a,dtype=np.float32));assert a.ndim==2 and np.isfinite(a).all()
     maps.append(a)
   tmp=p.with_suffix('.tmp')
   save_chunk(tmp,np.stack(maps))
   tmp.replace(p);nnew+=len(maps)
   savejson(out/'PROGRESS.json',{'done':beg+len(rr),'total':len(rows),'new_images':nnew,'elapsed_seconds':time.time()-start})
   if c%5==0 or beg+len(rr)==n:print('PREDICT',method,t,beg+len(rr),len(rows),'seconds',round(time.time()-start,1),flush=True)
 assert sha(e['checkpoint_file'])==e['checkpoint_sha256'],'Checkpoint changed during inference'
 if n==len(rows):
  chunks=list(out.glob('chunk_*.npz'));assert len(chunks)==math.ceil(n/16)
  savejson(out/'COMPLETE.json',{**identity,'n':n,'chunk_hashes':{x.name:sha(x) for x in chunks}})

def predict_cell(method,t,limit=None):
 import msvcrt
 e=next(x for x in readjson(WORK/'PREDICTION_REGISTRY.json') if x['method']==method and x['target']==t)
 out=Path(e['cache_path']);out.mkdir(parents=True,exist_ok=True)
 # A Windows byte-range lock prevents duplicate inference across coordinators.
 with (out/'PRODUCER.lock').open('a+b') as lock:
  lock.seek(0,2)
  if lock.tell()==0:lock.write(b'0');lock.flush()
  while True:
   lock.seek(0)
   try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1);break
   except OSError:time.sleep(2)
  try:_predict_cell(method,t,limit)
  finally:
   lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)

def auxiliary_gpu_queue():
 entries=readjson(WORK/'PREDICTION_REGISTRY.json')
 for e in reversed(entries):
  if e['cache_status']!='MISSING' or (Path(e['cache_path'])/'COMPLETE.json').exists():continue
  command=[sys.executable,'-u',str(Path(__file__).resolve()),'--stage','predict','--method',e['method'],'--target',e['target']]
  log=WORK/'logs'/f'aux_predict_{e["method"]}_{e["target"]}.log'
  print('AUX START',e['method'],e['target'],flush=True)
  with log.open('a',encoding='utf-8') as f:
   result=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
    env=dict(os.environ,PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
  print('AUX END',e['method'],e['target'],result.returncode,flush=True)

def evaluate_cell(method,t,image_only=False,category_limit=None):
 entries=readjson(WORK/'PREDICTION_REGISTRY.json');e=next(x for x in entries if x['method']==method and x['target']==t)
 out=WORK/'cells'/f'{method}_{t}';out.mkdir(parents=True,exist_ok=True)
 if e['cache_status']=='MISSING':assert (Path(e['cache_path'])/'COMPLETE.json').exists(),'Predictions incomplete'
 assert sha(e['manifest'])==e['manifest_sha256']
 rows=readcsv(e['manifest']);store=Store(e);cats=sorted({r['category'] for r in rows});start=time.time()
 if image_only:
  scores=[]
  for i,r in enumerate(rows):
   a=store[i];assert np.isfinite(a).all()
   scores.append({'sample_id':r['sample_id'],'category':r['category'],'label':int(r['label']),
    'score_native':topmean(a),'score_74':topmean(resize(a,(74,74))), 'native_shape':str(a.shape)})
  writecsv(out/'IMAGE_SCORES.csv',scores)
  per=[]
  for c in cats:
   cc=[x for x in scores if x['category']==c];y=[x['label'] for x in cc]
   r={'category':c,'n_images':len(y),'n_anomalous_images':sum(y)}
   for mode in ['native','74']:
    s=[x['score_'+mode] for x in cc]
    r['I_AUROC_'+mode]=float(roc_auc_score(y,s)*100) if len(set(y))==2 else None
    r['I_AP_'+mode]=float(average_precision_score(y,s)*100) if len(set(y))==2 else None
   per.append(r)
  writecsv(out/'IMAGE_CATEGORIES.csv',per)
  savejson(out/'IMAGE_COMPLETE.json',{'status':'COMPLETE','seconds':time.time()-start,'evaluator_hash':evaluator_hash()})
  print('IMAGE COMPLETE',method,t,flush=True);return
 for c in cats[:category_limit] if category_limit else cats:
  result=out/'pixel_categories'/f'{c}.json'
  prior=readjson(result) if result.exists() else None
  if prior and prior.get('AUPRO_policy_version')==2:continue
  ids=[i for i,r in enumerate(rows) if r['category']==c];masks=[native_mask(rows[i]) for i in ids]
  sizes=[m.size for m in masks];scratch=SCRATCH/f'expanded_{method}_{t}_{c}.npy'
  arr=np.lib.format.open_memmap(scratch,mode='w+',dtype=np.float32,shape=(sum(sizes),));views=[];offset=0
  for i,m in zip(ids,masks):
   raw=store[i];assert raw.ndim==2 and np.isfinite(raw).all()
   arr[offset:offset+m.size]=resize(raw,m.shape).reshape(-1)
   views.append(arr[offset:offset+m.size].reshape(m.shape));offset+=m.size
  arr.flush();yy=np.concatenate([m.reshape(-1) for m in masks])
  print('METRICS',method,t,c,'pixels',len(yy),flush=True)
  r={'category':c,'n_images':len(ids),'n_pixels':len(yy),'positive_pixels':int(yy.sum()),'evaluator_hash':evaluator_hash(),'AUPRO_policy_version':2}
  if prior:
   assert prior['n_pixels']==r['n_pixels'] and prior['positive_pixels']==r['positive_pixels']
   r.update({k:prior[k] for k in ['P_AUROC','P_F1max','P_AP']})
  else:r.update(weighted_pixel_metrics(yy,arr))
  gc.collect();r['AUPRO']=aupro(masks,views)
  savejson(result,r)
  del yy,masks,views,arr;gc.collect();scratch.unlink()
  print('CATEGORY COMPLETE',method,t,c,{k:r[k] for k in ['P_AP','AUPRO']},'seconds',round(time.time()-start,1),flush=True)
 if all((out/'pixel_categories'/f'{c}.json').exists() for c in cats):savejson(out/'PIXEL_COMPLETE.json',{'status':'COMPLETE','categories':len(cats),'seconds':time.time()-start,'evaluator_hash':evaluator_hash()})

def macro_metrics(rr,keys):
 result={}
 for k in keys:
  vv=[float(r[k]) for r in rr if r.get(k) is not None and str(r[k]) not in ('','None','nan') and np.isfinite(float(r[k]))]
  result[k]=float(np.mean(vv)) if vv else None;result[k+'_valid_categories']=len(vv)
 return result
def ranks(rows,key,target):
 values=sorted({float(r[key]) for r in rows if r['target']==target and r.get(key) is not None},reverse=True)
 return {r['method']:1+values.index(float(r[key])) for r in rows if r['target']==target and r.get(key) is not None}
def old_paper_values():
 from docx import Document
 ref=WORK/'REFERENCE_DOCUMENT.json'
 p=Path(readjson(ref)['approved_reference_document']) if ref.exists() else Path(POLICY['manuscript'])
 if not p.exists():return {},str(p)
 snapshot=WORK/'OLD_REFERENCE_VALUES.json'
 if snapshot.exists():
  saved=readjson(snapshot);assert normpath(saved['document'])==normpath(p)
  return {(r['method'],r['target'],r['metric']):r['value'] for r in saved['values']},str(p)
 out={}
 for table in Document(p).tables:
  headers=[c.text.strip() for c in table.rows[0].cells]
  if len(headers)<7:continue
  aliases={'MVTec':'MVTec','MVTec AD':'MVTec','VisA':'VisA','BTAD':'BTAD','MPDD':'MPDD','DTD-Syn.':'DTD-Synthetic','DTD-Synthetic':'DTD-Synthetic','DAGM':'DAGM','DAGM 2007':'DAGM'}
  for row in table.rows[1:]:
   label=row.cells[0].text.strip();method=next((m for m in METHODS if label==m or label.startswith(m+' ') or label.startswith(m+'(')),None)
   if not method:continue
   for j,header in enumerate(headers[1:],1):
    if header not in aliases:continue
    vv=[float(x) for x in re.findall(r'[-+]?\d+(?:\.\d+)?',row.cells[j].text)]
    keys=['P_AUROC','P_F1max','P_AP','AUPRO'] if len(vv)==4 else ['I_AUROC','I_AP'] if len(vv)==2 else []
    for k,v in zip(keys,vv):out[(method,aliases[header],k)]=v
 assert len(out)==252,f'Expected all 42 old cells / 252 metrics, got {len(out)}'
 savejson(snapshot,{'document':str(p),'document_sha256':sha(p),'purpose':'Frozen old-table values only; no Word modification',
  'values':[{'method':m,'target':t,'metric':k,'value':v} for (m,t,k),v in out.items()]})
 return out,str(p)

def asset_inventory():
 """Hash every component of composite checkpoints, not only the first file."""
 import zipfile
 entries=readjson(WORK/'PREDICTION_REGISTRY.json');hashes={};inventory={}
 for e in entries:
  paths=[Path(p.split('::')[0]) for p in e['checkpoint'].split(';')]
  paths.append(Path('C:/Users/Administrator/.cache/clip/ViT-L-14-336px.pt'))
  if e['method']=='CESAD':
   paths.extend([ROOT/'checkpoints/test_on_mvtec.pth',ROOT/'.cache/clip/ViT-L-14-336px.pt'])
   if e['source']=='Full VisA':paths.append(ROOT/'results/phase_statealigned_pc1_debias/formal/q_pc1_statealigned1024.npy')
  assets=[]
  for p in dict.fromkeys(paths):
   assert p.is_file(),p
   if str(p) not in hashes:hashes[str(p)]=sha(p)
   assets.append({'path':str(p),'sha256':hashes[str(p)],'bytes':p.stat().st_size})
  cp=Path(e['cache_path']);dtype='float32';shape='per-image final maps; see chunk headers'
  if cp.is_file():
   if e['cache_key']:
    with zipfile.ZipFile(cp) as z:
     with z.open(e['cache_key']+'.npy') as f:
      version=np.lib.format.read_magic(f)
      dims,fortran,dt=np.lib.format._read_array_header(f,version)
      dtype=str(dt);shape=str(dims)
   else:
    a=np.load(cp,mmap_mode='r');dtype=str(a.dtype);shape=str(a.shape);del a
  inventory[e['method']+'_'+e['target']]={'assets':assets,'cache_dtype':dtype,'cache_shape':shape,
   'cache_precision_note':'Historical cache preserved without requantization; evaluator casts to float32. Existing float16 ties cannot be reversed.' if dtype=='float16' else 'float32 maps; no evaluator quantization'}
 savejson(WORK/'COMPOSITE_ASSET_INVENTORY.json',inventory)
 base1=str(Path('C:/Users/Administrator/.cache/clip/ViT-L-14-336px.pt'))
 base2=str(ROOT/'.cache/clip/ViT-L-14-336px.pt')
 assert hashes[base1]==hashes[base2],'Runtime and project CLIP base weights differ'
 print('ASSET INVENTORY',len(inventory),'cells',len(hashes),'unique files',flush=True)
def final_outputs():
 registry=readjson(WORK/'PREDICTION_REGISTRY.json');old,reference=old_paper_values();pixels=[];images=[];native=[];provenance=[]
 assets=readjson(WORK/'COMPOSITE_ASSET_INVENTORY.json') if (WORK/'COMPOSITE_ASSET_INVENTORY.json').exists() else {}
 for e in registry:
  method,t=e['method'],e['target'];out=WORK/'cells'/f'{method}_{t}'
  base={'method':method,'source':e['source'],'target':t,'n_samples':e['n_samples'],
        'n_categories_total':len({r['category'] for r in readcsv(e['manifest'])})}
  pc=[readjson(p) for p in sorted((out/'pixel_categories').glob('*.json'))]
  okp=(out/'PIXEL_COMPLETE.json').exists() and len(pc)==base['n_categories_total'] and all(r.get('AUPRO_policy_version')==2 for r in pc)
  oki=(out/'IMAGE_COMPLETE.json').exists()
  rp={**base,**macro_metrics(pc,['P_AUROC','P_F1max','P_AP','AUPRO']),'status':'COMPLETE' if okp else 'PENDING',
      'gt_resolution_policy':POLICY['pixel_gt'],'evaluator_hash':evaluator_hash(),'old_reference_doc':reference}
  if not okp:
   for k in ['P_AUROC','P_F1max','P_AP','AUPRO']:rp[k]=None
  for k in ['P_AUROC','P_F1max','P_AP','AUPRO']:
   ov=old.get((method,t,k));rp['Old_'+k]=ov;rp['Delta_'+k]=(rp[k]-ov) if okp and ov is not None else None
  pixels.append(rp)
  ic=readcsv(out/'IMAGE_CATEGORIES.csv') if oki else []
  mi=macro_metrics(ic,['I_AUROC_74','I_AP_74','I_AUROC_native','I_AP_native'])
  ri={**base,'I_AUROC':mi['I_AUROC_74'],'I_AP':mi['I_AP_74'],
      'I_AUROC_valid_categories':mi['I_AUROC_74_valid_categories'],'I_AP_valid_categories':mi['I_AP_74_valid_categories'],
      'grid':'74x74','top_k':55,'status':'COMPLETE' if oki else 'PENDING','evaluator_hash':evaluator_hash(),'old_reference_doc':reference}
  for k in ['I_AUROC','I_AP']:
   ov=old.get((method,t,k));ri['Old_'+k]=ov;ri['Delta_'+k]=(ri[k]-ov) if oki and ov is not None else None
  images.append(ri)
  shapes=sorted({s['native_shape'] for s in readcsv(out/'IMAGE_SCORES.csv')}) if oki else []
  rn={**base,**mi,'status':ri['status'],'native_resolution_policy':POLICY['image_native'],'native_map_shapes':';'.join(shapes)}
  for k in ['I_AUROC','I_AP']:
   rn['Delta_74_minus_native_'+k]=mi[k+'_74']-mi[k+'_native'] if oki else None
  native.append(rn)
  prov={**base,'checkpoint':e['checkpoint'],'checkpoint_sha256':e['checkpoint_sha256'],
    'composite_assets_and_hashes':json.dumps(assets.get(method+'_'+t,{}).get('assets',[])),
    'cached_prediction_dtype':assets.get(method+'_'+t,{}).get('cache_dtype','UNKNOWN'),
    'cached_prediction_shape':assets.get(method+'_'+t,{}).get('cache_shape','UNKNOWN'),
    'prediction_precision_note':assets.get(method+'_'+t,{}).get('cache_precision_note','UNKNOWN'),
    'prediction_cache_path':e['cache_path'],'prediction_cache_key':e['cache_key'],
    'prediction_cache_sha256':e.get('cache_sha256','see COMPLETE.json chunk hashes'),
    'GT_manifest':e['manifest'],'GT_manifest_sha256':e['manifest_sha256'],'GT_roots':str(ROOTS[t]),
    'pixel_resolution_policy':POLICY['pixel_gt'],'image_score_grid':'74x74 plus native sensitivity; ceil top1%',
    'evaluator_code':str(Path(__file__).resolve()),'evaluator_code_sha256':sha(__file__),'evaluator_function_hash':evaluator_hash(),
    'category_aggregation':'category-macro; valid categories equal weight; two label classes required',
    'pixel_valid_categories':rp['P_AP_valid_categories'],'image_valid_categories':ri['I_AP_valid_categories'],
    'sample_order_verification':e['order_verification'],'method_fixed_postprocessing':e['postprocessing'],
    'official_training_provenance':e['official_training_provenance'],
    'software_versions':f'Python {sys.version.split()[0]};numpy {np.__version__};torch {torch.__version__};sklearn {__import__("sklearn").__version__};scipy {__import__("scipy").__version__}',
    'status':'COMPLETE' if okp and oki else 'PENDING',
    'recompute_missing_inference_command':f'"{sys.executable}" "{Path(__file__).resolve()}" --stage predict --method {method} --target {t}' if e['cache_status']=='MISSING' else 'Existing full cache reused; no supplementary inference needed',
    'recompute_pixel_command':f'"{sys.executable}" "{Path(__file__).resolve()}" --stage pixel --method {method} --target {t}',
    'recompute_image_command':f'"{sys.executable}" "{Path(__file__).resolve()}" --stage image --method {method} --target {t}'}
  if not (okp and oki) and ((WORK/'failures'/f'predict_{method}_{t}.json').exists() or (WORK/'failures'/f'pixel_{method}_{t}.json').exists() or (WORK/'failures'/f'image_{method}_{t}.json').exists()):prov['status']='FAIL; see work/failures'
  provenance.append(prov)
 for t in TARGETS:
  for rr,keys in [(pixels,['P_AUROC','P_F1max','P_AP','AUPRO']),(images,['I_AUROC','I_AP'])]:
   for k in keys:
    rk=ranks(rr,k,t)
    for r in rr:
     if r['target']==t:
      r['Rank_'+k]=rk.get(r['method']);ov=old.get((r['method'],t,k))
      allold=sorted({v for (m,tt,kk),v in old.items() if tt==t and kk==k},reverse=True)
      r['Old_Rank_'+k]=1+allold.index(ov) if ov is not None else None
  for k in ['I_AUROC','I_AP']:
   ra=ranks(native,k+'_74',t);rb=ranks(native,k+'_native',t)
   baselineA=[r[k+'_74'] for r in native if r['target']==t and r['method']!='CESAD' and r.get(k+'_74') is not None]
   baselineB=[r[k+'_native'] for r in native if r['target']==t and r['method']!='CESAD' and r.get(k+'_native') is not None]
   for r in native:
    if r['target']==t:
     r['Rank_74_'+k]=ra.get(r['method']);r['Rank_native_'+k]=rb.get(r['method'])
     if r['method']=='CESAD' and r.get(k+'_74') is not None:
      r['CESAD_gap_to_best_baseline_74_'+k]=r[k+'_74']-max(baselineA) if baselineA else None
      r['CESAD_gap_to_best_baseline_native_'+k]=r[k+'_native']-max(baselineB) if baselineB else None
 writecsv(FINAL/'FINAL_PIXEL_COMPARISON.csv',pixels);writecsv(FINAL/'FINAL_IMAGE_COMPARISON_74.csv',images)
 writecsv(FINAL/'FINAL_IMAGE_NATIVE_SENSITIVITY.csv',native);writecsv(FINAL/'EVALUATION_PROVENANCE.csv',provenance)
 def table(rr,keys):
  lines=['| Method | MVTec | VisA | BTAD | MPDD | DTD-Synthetic | DAGM 2007 |','|---|---|---|---|---|---|---|']
  for method in METHODS:
   cells=[]
   for t in TARGETS:
    r=next(x for x in rr if x['method']==method and x['target']==t);v=[]
    for k in keys:
     if r.get(k) is None:v.append('PENDING');continue
     s=f'{r[k]:.2f}';rank=r.get('Rank_'+k)
     v.append('**'+s+'**' if rank==1 else '<u>'+s+'</u>' if rank==2 else s)
    cells.append(' / '.join(v))
   lines.append('| '+method+' | '+' | '.join(cells)+' |')
  return '\n'.join(lines)
 ncomplete=sum(r['status']=='COMPLETE' for r in provenance)
 tables='# Final unified paper tables\n\n'+f'Completed cells: {ncomplete}/42. Percentage units; ranks use unrounded values; bold=best, underline=second distinct value. Pending rows are never eligible for a freeze.\n\n'
 tables+='## Table 1. Pixel-level comparison\n\nP-AUROC / P-F1max / P-AP / AUPRO. Paired native GT; category macro.\n\n'+table(pixels,['P_AUROC','P_F1max','P_AP','AUPRO'])
 tables+='\n\n## Table 2. Image-level comparison\n\nI-AUROC / I-AP. All maps resized to 74×74, bilinear align_corners=True, ceil Top-1% (55 pixels), category macro.\n\n'+table(images,['I_AUROC','I_AP'])+'\n'
 (FINAL/'FINAL_PAPER_TABLES.md').write_text(tables,encoding='utf-8')
 changes=[]
 for rows,keys in [(pixels,['P_AUROC','P_F1max','P_AP','AUPRO']),(images,['I_AUROC','I_AP'])]:
  for r in rows:
   for k in keys:
    if r.get('Delta_'+k) is not None:changes.append((abs(r['Delta_'+k]),r['method'],r['target'],k,r['Old_'+k],r[k],r['Delta_'+k]))
 changes.sort(reverse=True)
 summary=readjson(WORK/'MANIFEST_SUMMARY.json');exceptions=[(s['target'],s['annotation_exceptions']) for s in summary if s['annotation_exceptions']]
 report=['# Unified evaluation alignment report',f'Completed: **{ncomplete}/42**. No checkpoint changes, training, target calibration or model selection. Exact approved old-value reference: `{reference}`.',
 '## Fixed evaluator','Pixel maps: float32 bilinear align_corners=True to each paired original mask shape; normal images use image-sized zeros. Original GT is never bilinearly resized. AP: sklearn average_precision_score, exact tied-score multiplicities. AUROC: sklearn ROC AUC. F1max: complete sklearn PR path. Equal category-macro means; all AUROC/AP/F1 categories require positive and negative labels.',
 'AUPRO: 8-connected GT regions, equal region weighting, 200 category-wide linear thresholds, strict score>threshold, pixel FPR, endpoints (0,0)/(1,1), preserve vertical segments at duplicate FPR in threshold order, interpolate at FPR=.30 and divide trapezoid area on [0,.30] by .30. Unlike the historical evaluator, the denominator is fixed .30, not the sampled FPR span. During verification, an upper-envelope duplicate-FPR integration defect was corrected (policy v2); every affected AUPRO is recomputed, without changing AP/AUROC/F1 or selecting a policy by rankings. EVALUATOR_CORRECTION.md records the correction and independent tests.',
 'Image A: identical 74×74 bilinear grid and ceil Top-1% (55) mean. Image B: same ceil Top-1% on each final method map before evaluator resize. Method scores retain their original meaning (not all are logits); method-specific fixed smoothing is retained. Image validity is category-local, never dataset-global.',
 '## Samples and GT','| Target | Source | Images | Categories | Pixel/image valid | Annotation exceptions |','|---|---|---:|---:|---|---:|']
 for s in summary:report.append(f"| {s['target']} | {s['source']} | {s['images']} | {s['categories']} | {s['pixel_valid_categories']}/{s['image_valid_categories']} | {len(s['annotation_exceptions'])} |")
 report+=['## Frozen checkpoint alignment (source VisA / source MVTec)',
 '| Method | Formal source-specific assets | Fixed internal processing | Shared GT/pixel/image/macro |',
 '|---|---|---|---|',
 '| AnomalyCLIP | 9_12_4_multiscale/epoch_15.pth / 9_12_4_multiscale_visa/epoch_15.pth | official map + sigma4 | same/same/same/same |',
 '| FAPrompt | train_on_visa.pth / train_on_mvtecad.pth | generic object CAP/DAP + sigma4 | same/same/same/same |',
 '| MRAD-FT | test_on_mvtec.pth + visa caches / test_on_visa.pth + mvtec caches | source memories, neighbor pooling + sigma4 | same/same/same/same |',
 '| MRAD-CLIP | same source-specific asset bundles; distinct official trained-text path | source memories, neighbor pooling + sigma4 | same/same/same/same |',
 '| VisualAD | train_on_visa/CLIP.pth / train_on_mvtec/CLIP.pth | layer fusion + sigma4 | same/same/same/same |',
 '| FB-CLIP | visa_epoch_2_model.pth / mvtec_epoch_1_model.pth | foreground/background, internal align_corners=False, sigma4 | same/same/same/same |',
 '| CESAD | VisA / MVTec contrast_vectors.npz::E1; VisA additionally PC1 + fixed adapter/init assets | fixed 0.5 global/local LRP; no Gaussian | same/same/same/same |']
 report+=['GT roots and per-image image/GT paths, dimensions, labels, SHA256 and sample IDs are retained in the six manifests linked by EVALUATION_PROVENANCE.csv. DTD uses 12 categories and ground_truth/*_mask.png, never anomaly RGB as a mask; DAGM uses the original Test/Label/*_label.PNG masks, not grid GT.',
 '## Assets and fairness','Every method/dataset row identifies the frozen source-specific checkpoint, hash, prediction store, sample-order verification, postprocessing, valid counts and commands in EVALUATION_PROVENANCE.csv. Copies under project/data and the data drive are matched by complete image-byte hashes, not just filenames. Historical producer order is explicitly distinguished from saved path manifests. New inference receives image paths only, under torch.inference_mode; no GT is passed to the model.',
 'Evaluation rules and target sets are unified. Architectures, official fixed smoothing, prompts, memory and CESAD LRP remain method-specific. Reused historical caches include float16 maps; newly inferred missing maps are float32. No map is requantized to improve a ranking, and original cache precision/ties are disclosed per row. Original released checkpoint training population/selection history is UNKNOWN where not supplied; uniform evaluation does not certify identical training conditions.',
 '## Largest changes versus approved old paper','| Method | Target | Metric | Old | New | Delta |','|---|---|---|---:|---:|---:|']
 for _,m,t,k,ov,nv,de in changes[:10]:report.append(f'| {m} | {t} | {k} | {ov:.2f} | {nv:.2f} | {de:+.2f} |')
 if not changes:report.append('No comparable completed values yet; do not copy historical values into new results.')
 rankchanges=[]
 for rr,keys in [(pixels,['P_AP']),(images,['I_AUROC','I_AP'])]:
  for r in rr:
   for k in keys:
    if r.get('Old_Rank_'+k) is not None and r.get('Rank_'+k) is not None and r['Old_Rank_'+k]!=r['Rank_'+k]:rankchanges.append(f"{r['method']} {r['target']} {k}: {r['Old_Rank_'+k]}→{r['Rank_'+k]}")
 report+=['## Rank and grid sensitivity','; '.join(rankchanges[:18]) or 'No rank changes available yet.','All unrounded old/new metric ranks are retained in the two comparison CSVs. Old ranks are necessarily based on printed old-paper values; new ranks use full precision. Native versus 74 score differences, both rankings, and CESAD-minus-best-baseline gaps are retained for every target in FINAL_IMAGE_NATIVE_SENSITIVITY.csv.']
 gridchanges=[]
 for r in native:
  for k in ['I_AUROC','I_AP']:
   de=r.get('Delta_74_minus_native_'+k)
   if de is not None:gridchanges.append((abs(de),r['method'],r['target'],k,de))
 gridchanges.sort(reverse=True)
 report.append('Largest 74-minus-native changes: '+('; '.join(f'{m}/{t}/{k} {v:+.2f}' for _,m,t,k,v in gridchanges[:5]) or 'pending')+'.')
 report+=['## Exceptions and freeze decision',f'Missing cells: {42-ncomplete}. Logs/failures are under the separate run-work directory; no historical cache or CSV is overwritten.']
 if exceptions:report.append('Original annotation exception: BTAD contains 11 image-label-positive samples with empty original masks. All are retained with their original image labels and zero pixel masks for all methods; sample IDs are recorded in MANIFEST_SUMMARY.json. No masks were invented and no samples were dropped. Their source annotation consistency needs explicit author confirmation.')
 report.append('**No unqualified FINAL PASS.** '+('All numerical cells are recomputed, but final freeze remains conditional on the declared annotation/provenance exceptions.' if ncomplete==42 else 'The tables cannot be frozen while cells remain incomplete.'))
 report.append('Tables 3/4 are untouched. Their MVTec/VisA/BTAD/MPDD pixel cells and six-target means require a separately authorized native-GT reevaluation before claiming the same new pixel protocol. DTD/DAGM already use native GT; identical checkpoint/map/AP inputs should agree, but are not substituted into controls here.')
 (FINAL/'EVALUATION_ALIGNMENT_REPORT.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
 return ncomplete

def coordinator():
 import psutil
 selftest()
 registry=readjson(WORK/'PREDICTION_REGISTRY.json')
 # The whole run's complete source is saved mechanically, never editing old code.
 import shutil
 shutil.copy2(__file__,WORK/'evaluator_source_snapshot.py')
 pending_gpu=[e for t in ['MPDD','BTAD','DTD-Synthetic','MVTec','VisA','DAGM'] for e in registry if e['target']==t and e['cache_status']=='MISSING' and not (Path(e['cache_path'])/'COMPLETE.json').exists()]
 active=[];failed=set();last=0;started=time.time()
 def launch(stage,e):
  name=f'{stage}_{e["method"]}_{e["target"]}';log=WORK/'logs'/f'{name}.log';log.parent.mkdir(exist_ok=True)
  command=[sys.executable,'-u',str(Path(__file__).resolve()),'--stage',stage,'--method',e['method'],'--target',e['target']]
  h=log.open('a',encoding='utf-8');env=dict(os.environ,PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
  p=subprocess.Popen(command,cwd=ROOT,stdout=h,stderr=subprocess.STDOUT,env=env)
  active.append((p,h,stage,e,name));print('START',name,'PID',p.pid,flush=True)
 while True:
  for item in active[:]:
   p,h,stage,e,name=item
   code=p.poll()
   if code is None:continue
   h.close();active.remove(item)
   if code:
    failed.add(name);savejson(WORK/'failures'/f'{name}.json',{'exit_code':code,'log':str(WORK/'logs'/f'{name}.log')})
   print('END',name,'exit',code,flush=True)
  if not any(x[2]=='predict' for x in active) and pending_gpu:launch('predict',pending_gpu.pop(0))
  running={x[4] for x in active};cpu=sum(x[2]!='predict' for x in active)
  ready=[e for e in registry if e['cache_status']=='REUSABLE' or (Path(e['cache_path'])/'COMPLETE.json').exists()]
  tasks=[]
  for stage,marker in [('image','IMAGE_COMPLETE.json'),('pixel','PIXEL_COMPLETE.json')]:
   for e in ready:
    name=f'{stage}_{e["method"]}_{e["target"]}'
    cell=WORK/'cells'/f'{e["method"]}_{e["target"]}'
    needed=not (cell/marker).exists()
    if stage=='pixel' and not needed:
     needed=any(readjson(p).get('AUPRO_policy_version')!=2 for p in (cell/'pixel_categories').glob('*.json'))
    if name not in failed and name not in running and needed:tasks.append((stage,e))
  while cpu<2 and tasks:
   if cpu and psutil.virtual_memory().available<10*1024**3:break
   stage,e=tasks.pop(0);launch(stage,e);cpu+=1
  if time.time()-last>45:
   completed=final_outputs();last=time.time()
   print('PROGRESS',completed,'/42','GPU_remaining',len(pending_gpu),'active',[(x[2],x[3]['method'],x[3]['target']) for x in active],
    'failures',len(failed),'elapsed_minutes',round((time.time()-started)/60,1),flush=True)
  if not active and not pending_gpu and not tasks:break
  time.sleep(2)
 complete=final_outputs()
 if complete==42:verify_completed_results();complete=final_outputs()
 savejson(WORK/'RUN_FINISHED.json',{'numerical_complete':complete,'total':42,'failures':sorted(failed),'seconds':time.time()-started,'unqualified_final_pass':False})
 print('RUN FINISHED',complete,'/42',flush=True)

def verify_completed_results():
 """Final identity/aggregation checks, with no model or score changes."""
 registry=readjson(WORK/'PREDICTION_REGISTRY.json');assets=readjson(WORK/'COMPOSITE_ASSET_INVENTORY.json')
 checked_assets={};checked_caches=set();checks=[]
 for e in registry:
  rows=readcsv(e['manifest']);cell=WORK/'cells'/f'{e["method"]}_{e["target"]}'
  assert sha(e['manifest'])==e['manifest_sha256']
  for a in assets[e['method']+'_'+e['target']]['assets']:
   if a['path'] not in checked_assets:
    assert sha(a['path'])==a['sha256'],f'Frozen asset changed: {a["path"]}'
    checked_assets[a['path']]=a['sha256']
  cp=Path(e['cache_path'])
  if cp.is_file():
   if str(cp) not in checked_caches:assert sha(cp)==e['cache_sha256'];checked_caches.add(str(cp))
  else:
   meta=readjson(cp/'COMPLETE.json');assert meta['n']==len(rows) and meta['manifest_sha256']==e['manifest_sha256']
   for name,digest in meta['chunk_hashes'].items():assert sha(cp/name)==digest,(cp,name,'Prediction chunk changed')
  image=readcsv(cell/'IMAGE_SCORES.csv')
  assert [r['sample_id'] for r in image]==[r['sample_id'] for r in rows]
  assert [int(r['label']) for r in image]==[int(r['label']) for r in rows]
  per=readcsv(cell/'IMAGE_CATEGORIES.csv');cats=sorted({r['category'] for r in rows})
  assert sorted(r['category'] for r in per)==cats
  for c in cats:
   rr=[r for r in rows if r['category']==c];metric=readjson(cell/'pixel_categories'/f'{c}.json')
   assert metric['AUPRO_policy_version']==2
   assert metric['n_images']==len(rr)
   assert metric['n_pixels']==sum(int(r['gt_height'])*int(r['gt_width']) for r in rr)
   assert metric['positive_pixels']==sum(int(r['positive_pixels']) for r in rr)
   for key in ['P_AUROC','P_AP','P_F1max','AUPRO']:assert metric[key] is not None and 0<=metric[key]<=100.000001
   ii=[r for r in image if r['category']==c];mm=next(r for r in per if r['category']==c)
   y=[int(r['label']) for r in ii]
   for mode in ['74','native']:
    ss=[float(r['score_'+mode]) for r in ii]
    for key,value in [('I_AUROC',roc_auc_score(y,ss)*100),('I_AP',average_precision_score(y,ss)*100)]:
     assert abs(value-float(mm[key+'_'+mode]))<1e-9
  checks.append({'method':e['method'],'target':e['target'],'images':len(rows),'categories':len(cats),'status':'PASS'})
  print('VERIFIED',e['method'],e['target'],flush=True)
 selftest()
 savejson(WORK/'FINAL_VERIFICATION.json',{'numerical_validation':'PASS','cells':checks,'checked_frozen_assets':checked_assets,'policy_version':2,'code_sha256':sha(__file__),'evaluation_function_hash':evaluator_hash(),'unqualified_final_pass':False,'limitations':['Original released training provenance UNKNOWN','BTAD original empty masks in 11 anomalous images','Historical FP16 versus new FP32 prediction-cache precision disclosed']})

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['manifest','test','registry','predict','pixel','image','run','tables','gpu-aux','finalize','assets'],default='manifest')
 p.add_argument('--method',choices=METHODS);p.add_argument('--target',choices=TARGETS);p.add_argument('--limit',type=int);a=p.parse_args()
 lock_policy();torch.set_num_threads(4)
 if a.stage=='test':selftest()
 elif a.stage=='manifest':build_manifests()
 elif a.stage=='registry':make_registry()
 elif a.stage=='assets':asset_inventory()
 elif a.stage=='predict':predict_cell(a.method,a.target,a.limit)
 elif a.stage=='run':coordinator()
 elif a.stage=='finalize':
  import psutil
  # Drain the two already-running queues before resuming with the corrected
  # evaluator. This is part of this run, not a scheduled future automation.
  for pid in [5104,35224]:
   try:
    process=psutil.Process(pid)
    if 'final_unified_evaluation.py' not in ' '.join(process.cmdline()):continue
    print('WAIT EXISTING COORDINATOR',pid,flush=True)
    while process.is_running():time.sleep(2)
   except psutil.NoSuchProcess:pass
  # Reload the saved evaluator after the long wait so the final repair uses
  # precisely the code archived in the completed run.
  os.execv(sys.executable,[sys.executable,'-u',str(Path(__file__).resolve()),'--stage','run'])
 elif a.stage=='tables':final_outputs()
 elif a.stage=='gpu-aux':auxiliary_gpu_queue()
 else:evaluate_cell(a.method,a.target,a.stage=='image',a.limit)
if __name__=='__main__':main()
