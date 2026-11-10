"""User-authorized reciprocal NTR/LRP rerun. New outputs only, no target selection."""
from pathlib import Path
import os,sys,json,csv,math,hashlib,time,gc,traceback,shutil,argparse
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from experiments import final_unified_evaluation as ev
from experiments import evaluate_traceable_visa_targets as target
from experiments import source_build_traceable_visa as source
from experiments import phase_ces_linear_head_control as head
from experiments import phase_ces_lhc_paired_repeat as paired
from experiments.phase_xliii_adapter_head_decomposition import AdapterToState,balanced_loss
BASE=ROOT/'results/ntr_lrp_reciprocal_aligned_20261101';OUT=BASE/'formal';WORK=BASE/'work';OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(exist_ok=True)
SEEDS=(0,1,2);CORE_RHOS=(.05,.10,.15);EXTRA_RHOS=(.01,.03,.20,.30);LAMBDAS=(.3,.4,.5,.6,.7)
CERTROOT=ROOT/'results/visa_source_traceable_rebuild/formal';BUILDDIR=Path('E:/MRAD_visa_source_traceable_rebuild/source_build/seed0')
METRICS=('I_AUROC','I_AP','P_AUROC','P_AP','P_F1max','AUPRO')
def js(p,v):ev.savejson(p,v)
def read(p):return ev.readjson(p)
def sha(p):return ev.sha(p)
def status(stage,**kw):js(OUT/'STATUS.json',dict(stage=stage,time=time.strftime('%Y-%m-%d %H:%M:%S'),**kw));print(stage,kw,flush=True)
def ident(p):p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def setup():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
 assert torch.cuda.is_available()
def variants():
 return [dict(seed=s,rho=r,name=f'seed{s}_'+('noNTR' if r is None else f'rho{r:.2f}')) for s in SEEDS for r in (None,*CORE_RHOS)]+[dict(seed=0,rho=r,name=f'seed0_rho{r:.2f}') for r in EXTRA_RHOS]
def config():
 p=OUT/'LOCKED_CONFIG.json'
 expected=dict(sources={'VisA':'Full MVTec','MVTec':'Full VisA'},targets=['VisA','MVTec'],seeds=list(SEEDS),rho_core=list(CORE_RHOS),rho_wide_seed0=[.01,.03,.05,.10,.15,.20,.30],lambda_seed0=list(LAMBDAS),scale_seed0=[2,3],variants=variants(),updates=30,learning_rate=.001,weight_decay=0,loss='balanced BCE + Dice + unit-weight NTR when enabled',selection='NONE; final update30; report all locked settings',source_preparation='MVTec repaired pools/L2; VisA certified rebuilt PC1/E1/E2/adapter',evaluation='paired native GT; category macro; fixed evaluator, FP32 predictions',evaluator_sha256=ev.evaluator_hash(),user_authorization='2026-11-01 user requested rerun mismatched NTR/LRP experiments only on reciprocal MVTec/VisA',code=ident(Path(__file__)))
 if p.exists():assert read(p)==expected,'Locked experiment configuration changed'
 else:js(p,expected)
 return expected
def adapter_model():
 state=torch.load(BUILDDIR/'assets/E2_formal.pt',map_location='cpu',weights_only=False)
 a=AdapterToState(1024,state['patch_states']).cuda();a.load_state_dict(state['model'],strict=True);a.eval()
 for p in a.parameters():p.requires_grad_(False)
 return a

def visa_pool(seed):
 folder=WORK/'source_pools'/f'VisA_seed{seed}';done=folder/'COMPLETE.json'
 if done.exists():
  m=read(done)
  for x in m['files']:assert sha(x['path'])==x['sha256']
  return folder,m
 folder.mkdir(parents=True,exist_ok=True)
 cert=read(CERTROOT/'source_feature_certification/SOURCE_FEATURE_CERTIFICATE.json')
 images=ev.readcsv(cert['image_feature_manifest']['path']);labels=np.load(cert['source_labels']['path']);source.SEED=seed
 pairs,y,cats,origins=source.ces_pool_plan(images,labels)
 np.savez(folder/'POOL_PLAN.npz',physical_pairs=pairs,labels=y,categories=cats,origins=origins)
 assert np.array_equal(y,labels[pairs[:,0],pairs[:,1]])
 rng=np.random.default_rng(seed);batches=np.stack([paired.balanced_indices(y,cats,rng) for _ in range(30)])
 np.save(folder/'batch_indices.npy',batches);np.save(folder/'labels.npy',y.astype(np.float32))
 if seed==0:
  saved=np.load(CERTROOT/'source_build_review/plans/CES/POOL_PLAN.npz')
  assert np.array_equal(saved['physical_pairs'],pairs) and np.array_equal(saved['labels'],y)
  assert np.array_equal(batches,np.load(CERTROOT/'source_build_review/plans/CES/BATCHES.npy'))
  z=np.load(BUILDDIR/'CES_pool/features.npy');np.save(folder/'features.npy',z)
 else:
  raw=np.load(cert['patch_features']['path'],mmap_mode='r');q=np.load(BUILDDIR/'pc1.npy');a=adapter_model()
  x=source.pool_features(raw,dict(physical_pairs=pairs),q,source.refs()[0].norm_np)
  with torch.inference_mode():z=np.concatenate([a.adapt(torch.tensor(x[i:i+8192],device='cuda')).cpu().numpy() for i in range(0,len(x),8192)])
  np.save(folder/'features.npy',z.astype(np.float32));del a,raw,x,z;gc.collect();torch.cuda.empty_cache()
 init=torch.load(BUILDDIR/'checkpoints/CES/update_000.pt',map_location='cpu',weights_only=False)['params']['raw_model']
 torch.save(init,folder/'INITIAL_MODEL.pt')
 ev.writecsv(folder/'PATCH_MANIFEST.csv',[dict(pool_index=i,source_position=int(pos),patch_flat_index=int(pid),label=int(y[i]),category=str(cats[i]),origin=str(origins[i])) for i,(pos,pid) in enumerate(pairs)])
 m=dict(source='Full VisA',seed=seed,patches=len(y),batch_shape=list(batches.shape),upstream='certified rebuilt assets fixed for all seeds',files=[ident(p) for p in folder.iterdir() if p.is_file()]);js(done,m)
 return folder,m

def mvtec_pool(seed):
 folder=Path('E:/MRAD_component_original_cache_repair/20261027/source_pools')/f'Full_MVTec_seed{seed}'
 m=read(folder/'COMPLETE.json')
 for key,name in [('feature_array_hash','features'),('labels_hash','labels'),('batch_stream_hash','batch_indices')]:assert paired.digest(np.load(folder/(name+'.npy'),mmap_mode='r'))==m[key]
 assert sha(folder/'PATCH_MANIFEST.csv')==m['manifest_hash'] and read(folder/'E_CACHE_RECONSTRUCTION_CHECK.json')['status']=='PASS'
 return folder,dict(source='Full MVTec',seed=seed,upstream='repaired physical-order MVTec/L2 pools; no VisA upstream assets',files=[ident(folder/n) for n in ['COMPLETE.json','E_CACHE_RECONSTRUCTION_CHECK.json','features.npy','labels.npy','batch_indices.npy','initial_geometry.npz','PATCH_MANIFEST.csv','SOURCE_IMAGE_MANIFEST.csv']])

def train():
 c=config();setup();status('SOURCE_PREPARATION');registry=[]
 for target_name,src in c['sources'].items():
  for seed in SEEDS:
   folder,meta=visa_pool(seed) if target_name=='MVTec' else mvtec_pool(seed)
   dest=OUT/'source_inputs'/f'{target_name}_seed{seed}.json';js(dest,meta)
   x=torch.tensor(np.load(folder/'features.npy'),device='cuda');y=torch.tensor(np.load(folder/'labels.npy'),device='cuda',dtype=torch.float32);batches=np.load(folder/'batch_indices.npy');assert batches.shape==(30,8184 if target_name=='MVTec' else 8190)
   if target_name=='MVTec':initial=torch.load(folder/'INITIAL_MODEL.pt',weights_only=False,map_location='cpu')
   else:
    g=np.load(folder/'initial_geometry.npz');initial=head.CESHead(g['m'],float(g['alpha']),g['init_v']).state_dict()
   for v in [v for v in variants() if v['seed']==seed]:
    slug=f'{target_name}__{v["name"]}';p=WORK/'checkpoints'/slug/'update_030.pt';p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
     final=torch.load(p,map_location='cpu',weights_only=False);assert final['config_sha256']==sha(OUT/'LOCKED_CONFIG.json')
    else:
     module=head.CESHead(initial['m'].numpy(),float(initial['alpha']),initial['v'].numpy()).cuda();module.load_state_dict(initial);opt=source.adam(module.parameters(),.001,0.);trace=[]
     torch.save(dict(raw_model=initial,step=0,config_sha256=sha(OUT/'LOCKED_CONFIG.json')),p.parent/'update_000.pt')
     for step,ids in enumerate(batches,1):
      scores=module(x[ids]);base,bce,dice=balanced_loss(scores,y[ids]);ntr=scores.new_tensor(0.) if v['rho'] is None else head.ntr_loss_from_logits(scores,y[ids],v['rho']);loss=base+ntr
      assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();grad=float(module.v.grad.norm());opt.step()
      assert torch.isfinite(module.v).all();trace.append(dict(step=step,total=float(loss.detach()),bce=float(bce.detach()),dice=float(dice.detach()),ntr=float(ntr.detach()),gradient_norm=grad))
     final=dict(raw_model={k:t.detach().cpu() for k,t in module.state_dict().items()},optimizer=source.cpu_tree(opt.state_dict()),step=30,seed=seed,rho=v['rho'],config_sha256=sha(OUT/'LOCKED_CONFIG.json'),source_input_sha256=sha(dest),batch_indices=batches,rng_state=torch.get_rng_state())
     torch.save(final,p);ev.writecsv(p.parent/'TRAINING_TRACE.csv',trace);del module,opt
    if target_name=='MVTec' and seed==0 and v['rho']==.10:
     reference=torch.load(BUILDDIR/'checkpoints/CES/update_030.pt',map_location='cpu',weights_only=False)['params']['raw_model'];error=max(float((final['raw_model'][k]-reference[k]).abs().max()) for k in reference)
     assert error<2e-7,f'Failed source-only anchor reconstruction {error}'
     js(OUT/'VISA_SOURCE_ANCHOR_PARITY.json',dict(max_raw_parameter_error=error,tolerance=2e-7,reference=ident(BUILDDIR/'checkpoints/CES/update_030.pt'),new=ident(p)))
    registry.append(dict(target=target_name,source=src,**v,checkpoint=ident(p)));status('HEAD_TRAINED',target=target_name,variant=v['name'])
   del x,y;torch.cuda.empty_cache()
 js(OUT/'SOURCE_SEALED.json',dict(config_sha256=sha(OUT/'LOCKED_CONFIG.json'),models=registry,targets_accessed=False));status('SOURCE_SEALED',models=len(registry))

def load_weights(target_name):
 entries=[e for e in read(OUT/'SOURCE_SEALED.json')['models'] if e['target']==target_name];weights=[]
 for e in entries:
  assert sha(e['checkpoint']['path'])==e['checkpoint']['sha256'];v=torch.load(e['checkpoint']['path'],map_location='cpu',weights_only=False)['raw_model'];direction=F.normalize(v['v'],dim=0);weights.append(F.normalize(v['m']+v['alpha']*direction,dim=0)-F.normalize(v['m']-v['alpha']*direction,dim=0))
 return entries,torch.stack(weights,dim=1).cuda()
def transform(raw,target_name,adapter,q):
 z=np.asarray(raw,np.float32).reshape(-1,1024)
 if target_name=='MVTec':z=z-(z@q)[:,None]*q;z=z/np.maximum(np.linalg.norm(z,axis=-1,keepdims=True),1e-12)
 with torch.inference_mode():
  z=torch.tensor(z,device='cuda');return adapter.adapt(z) if target_name=='MVTec' else F.normalize(z,dim=-1)
def pack(q,grid):
 # [heads, views,37,37] -> global resized74, local stitched74 (or111).
 g=F.interpolate(q[:,0:1],(74,74),mode='bilinear',align_corners=True)[:,0]
 l=torch.cat([torch.cat([q[:,1+r*grid+c] for c in range(grid)],dim=-1) for r in range(grid)],dim=-2)
 return g,l

def predict():
 setup();config();assert (OUT/'SOURCE_SEALED.json').exists();cfg=read(CERTROOT/'target_evaluation_review/TARGET_EVALUATION_CONFIG.json');enc=target.Encoder(cfg);a=adapter_model();q=np.load(BUILDDIR/'pc1.npy')
 for name in ('VisA','MVTec'):
  entries,w=load_weights(name);dest=WORK/'predictions'/name;dest.mkdir(parents=True,exist_ok=True)
  srcman=ev.WORK/'manifests'/f'{name}.csv';rows=ev.readcsv(srcman);assert len(rows)==(2162 if name=='VisA' else 1725)
  ev.writecsv(dest/'MANIFEST.csv',rows);js(dest/'HEAD_ORDER.json',entries)
  p=dest/'global_local74.npy';done=dest/'COMPLETE.json';shape=(len(rows),len(entries),2,74,74)
  if done.exists():assert sha(p)==read(done)['maps']['sha256'];continue
  resume=dest/'PROGRESS.json';start=read(resume)['committed_images'] if resume.exists() else 0
  maps=np.lib.format.open_memmap(p,mode='r+' if p.exists() else 'w+',dtype=np.float32,shape=shape)
  logpath=dest/'VIEW_IDENTITIES.jsonl'
  with logpath.open('a',encoding='utf-8') as log:
   for i in range(start,len(rows)):
    raw,views=enc.extract(rows[i])
    with torch.inference_mode():
     z=transform(raw,name,a,q);scores=(z@w).reshape(5,37,37,len(entries)).permute(3,0,1,2);g,l=pack(scores,2);maps[i]=torch.stack([g,l],dim=1).cpu().numpy()
    assert np.isfinite(maps[i]).all();log.write(json.dumps(dict(index=i,sample_id=rows[i]['sample_id'],**views,map_sha256=hashlib.sha256(maps[i].tobytes()).hexdigest()))+'\n');maps.flush();log.flush();js(resume,dict(committed_images=i+1))
    if (i+1)%25==0:status('ENCODING',target=name,images=i+1,total=len(rows))
  del maps;js(done,dict(maps=ident(p),manifest=ident(dest/'MANIFEST.csv'),head_order=ident(dest/'HEAD_ORDER.json'),source_seal_sha256=sha(OUT/'SOURCE_SEALED.json'),encoder_hash=enc.before));status('PREDICTIONS_COMPLETE',target=name)
 # A predeclared seed-0 3x3 scale probe, using this run's same fixed full head.
 for name in ('VisA','MVTec'):
  from PIL import Image
  import io
  from phase_xa.official_task_forward import visual_outputs,normalized_patch_feature
  entries,w=load_weights(name);index=next(i for i,e in enumerate(entries) if e['seed']==0 and e['rho']==.10);weight=w[:,index:index+1]
  dest=WORK/'predictions'/name;rows=ev.readcsv(dest/'MANIFEST.csv');path=dest/'local111_seed0.npy';done=dest/'SCALE3_COMPLETE.json'
  if done.exists():assert sha(path)==read(done)['maps']['sha256'];continue
  progress=dest/'SCALE3_PROGRESS.json';start=read(progress)['committed_images'] if progress.exists() else 0
  maps=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=np.float32,shape=(len(rows),111,111))
  with (dest/'SCALE3_VIEW_IDENTITIES.jsonl').open('a',encoding='utf-8') as log:
   for i in range(start,len(rows)):
    content=Path(rows[i]['img_path']).read_bytes();assert hashlib.sha256(content).hexdigest()==rows[i]['image_sha256']
    with Image.open(io.BytesIO(content)) as image:
     image=image.convert('RGB');width,height=image.size
     boxes=[(c*width//3,r*height//3,(c+1)*width//3,(r+1)*height//3) for r in range(3) for c in range(3)]
     tensors=torch.stack([enc.preprocess(image.crop(box)) for box in boxes])
    out=[]
    with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
     for j in range(0,9,3):out.append(normalized_patch_feature(visual_outputs(enc.model,tensors[j:j+3].cuda(),adapter=None,grad=False)).float().cpu().numpy())
     raw=np.concatenate(out);z=transform(raw,name,a,q);qs=(z@weight).reshape(9,37,37)
     local=torch.cat([torch.cat([qs[r*3+c] for c in range(3)],dim=-1) for r in range(3)],dim=-2);maps[i]=local.cpu().numpy()
    maps.flush();log.write(json.dumps(dict(index=i,sample_id=rows[i]['sample_id'],boxes=boxes,tensor_sha256=hashlib.sha256(tensors.numpy().tobytes()).hexdigest(),map_sha256=hashlib.sha256(maps[i].tobytes()).hexdigest()))+'\n');log.flush();js(progress,dict(committed_images=i+1))
    if (i+1)%25==0:status('ENCODING_SCALE3',target=name,images=i+1,total=len(rows))
  del maps;js(done,dict(maps=ident(path),grid=3,seed=0,rho=.10));status('SCALE3_COMPLETE',target=name)
 assert target.visual_hash(enc.model)==enc.before
 status('PREDICTIONS_SEALED')

def experiments(entries):
 e=[]
 def add(fam,seed,rho,lrp=True,lam=.5,full=False):
  idx=next(i for i,x in enumerate(entries) if x['seed']==seed and x['rho']==rho);label=f'{fam}__seed{seed}__'+('noNTR' if rho is None else f'rho{rho:.2f}')+f'__lrp{int(lrp)}__lambda{lam:.1f}'
  e.append(dict(id=label,family=fam,seed=seed,rho=rho,lrp=lrp,lambda_global=lam,head_index=idx,full_metrics=full))
 for seed in SEEDS:
  for rho in (None,.10):
   for lrp in (False,True):add('component',seed,rho,lrp,full=True)
  for rho in (.05,.15):
   for lrp in (False,True):add('rho_core',seed,rho,lrp)
 for rho in EXTRA_RHOS:add('rho_wide',0,rho)
 for lam in LAMBDAS:
  if lam!=.5:add('lrp_fusion',0,.10,True,lam)
 idx=next(i for i,x in enumerate(entries) if x['seed']==0 and x['rho']==.10)
 e.append(dict(id='lrp_scale3_seed0',family='lrp_scale',seed=0,rho=.10,lrp=True,lambda_global=.5,head_index=idx,full_metrics=True,grid=3))
 return e

def evaluate():
 from sklearn.metrics import roc_auc_score,average_precision_score
 torch.set_num_threads(4);config()
 for name in ('VisA','MVTec'):
  d=WORK/'predictions'/name;assert (d/'COMPLETE.json').exists();entries=read(d/'HEAD_ORDER.json');rows=ev.readcsv(d/'MANIFEST.csv');maps=np.load(d/'global_local74.npy',mmap_mode='r');exps=experiments(entries);scale3=np.load(d/'local111_seed0.npy',mmap_mode='r')
  dest=OUT/'evaluations'/name;dest.mkdir(parents=True,exist_ok=True);scratch=WORK/'scratch'/name;scratch.mkdir(parents=True,exist_ok=True)
  categories=sorted({r['category'] for r in rows})
  for ex in exps:
   ed=dest/ex['id'];ed.mkdir(exist_ok=True)
   def merged(i):
    g,l=maps[i,ex['head_index']]
    if ex.get('grid')==3:
     large=F.interpolate(torch.from_numpy(g.copy())[None,None],size=(111,111),mode='bilinear',align_corners=True)[0,0].numpy();fused=.5*large+.5*scale3[i]
     return fused
    return (ex['lambda_global']*g+(1-ex['lambda_global'])*l) if ex['lrp'] else g
   image_scores=[ev.topmean(ev.resize(merged(i),(74,74))) for i in range(len(rows))]
   for cat in categories:
    jp=ed/(cat+'.json')
    if jp.exists():continue
    ids=[i for i,r in enumerate(rows) if r['category']==cat];n=sum(int(rows[i]['gt_height'])*int(rows[i]['gt_width']) for i in ids)
    yp=scratch/'labels.npy';sp=scratch/'scores.npy'
    # Remove only this new run's uncommitted scratch left after a failed attempt.
    for p in (yp,sp):
     if p.exists():assert p.parent.resolve()==scratch.resolve();p.unlink()
    assert shutil.disk_usage(WORK).free>n*5+512*1024**2
    y=np.lib.format.open_memmap(yp,mode='w+',dtype=np.bool_,shape=(n,));s=np.lib.format.open_memmap(sp,mode='w+',dtype=np.float32,shape=(n,));masks=[];scores=[];offset=0
    for i in ids:
     row=rows[i]
     if row['gt_path']:assert sha(row['gt_path'])==row['gt_sha256']
     mask=ev.native_mask(row);ar=ev.resize(merged(i),mask.shape);end=offset+mask.size;y[offset:end]=mask.ravel();s[offset:end]=ar.ravel();masks.append(y[offset:end].reshape(mask.shape));scores.append(s[offset:end].reshape(mask.shape));offset=end
    status('EVALUATING',target=name,experiment=ex['id'],category=cat,pixels=n)
    met=ev.weighted_pixel_metrics(y,s)
    if ex['full_metrics']:met['AUPRO']=ev.aupro(masks,scores)
    iy=[int(rows[i]['label']) for i in ids];si=[image_scores[i] for i in ids]
    met.update(I_AUROC=float(roc_auc_score(iy,si)*100),I_AP=float(average_precision_score(iy,si)*100))
    js(jp,dict(target=name,category=cat,**ex,**met));masks.clear();scores.clear();del y,s,mask,ar;gc.collect()
    for p in (yp,sp):p.unlink()
   status('EXPERIMENT_EVALUATED',target=name,experiment=ex['id'])
 summarize();status('COMPLETE_RECIPROCAL_NTR_LRP')

def summarize():
 records=[];summaries=[]
 for name in ('VisA','MVTec'):
  for ed in sorted((OUT/'evaluations'/name).iterdir()):
   rr=[read(p) for p in sorted(ed.glob('*.json'))]
   if len(rr)!=(12 if name=='VisA' else 15):continue
   records+=rr;m={k:v for k,v in rr[0].items() if k not in ['category',*METRICS]}
   for key in METRICS:
    vals=[float(r[key]) for r in rr if r.get(key) is not None];m[key]=float(np.mean(vals)) if vals else None
   summaries.append(m)
 ev.writecsv(OUT/'CATEGORY_METRICS.csv',records);ev.writecsv(OUT/'PER_SEED_METRICS.csv',summaries)
 groups={}
 for r in summaries:
  key=(r['target'],r['family'],r['rho'],r['lrp'],r['lambda_global']);groups.setdefault(key,[]).append(r)
 mean=[]
 for key,rr in groups.items():
  row=dict(target=key[0],family=key[1],rho=key[2],lrp=key[3],lambda_global=key[4],seeds=','.join(str(x['seed']) for x in rr),n_seeds=len(rr))
  for k in METRICS:
   vals=[x[k] for x in rr if x[k] is not None];row[k]=float(np.mean(vals)) if vals else None;row[k+'_SD']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
  mean.append(row)
 ev.writecsv(OUT/'MEAN_SD_METRICS.csv',mean)

def tests():
 torch.manual_seed(4)
 # Exact topology, FP32 fusion and no-LRP are tested without training or target data.
 q=torch.arange(5*37*37,dtype=torch.float32).reshape(1,5,37,37);g,l=pack(q,2)
 assert g.shape==l.shape==(1,74,74) and torch.equal(l[:,:37,:37],q[:,1]) and torch.equal(l[:,37:,37:],q[:,4])
 assert len(variants())==16 and len(experiments(variants()))==33
 assert set(x['target'] for x in [])==set()
 for rho in (.01,.05,.1,.3):
  z=torch.tensor([-.2,.1,.3,-.7],requires_grad=True);y=torch.zeros(4);loss=head.ntr_loss_from_logits(z,y,rho);assert torch.isfinite(loss);loss.backward();assert torch.isfinite(z.grad).all()
 js(OUT/'SELF_TEST.json',dict(status='PASS',topology=True,experiment_count=33,head_count_per_source=16,tail_gradients=True))
 print('Self-test passed',flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['self-test','train','predict','evaluate','summarize','all']);args=parser.parse_args()
 try:
  if args.stage=='self-test':tests()
  elif args.stage=='train':train()
  elif args.stage=='predict':predict()
  elif args.stage=='evaluate':evaluate()
  elif args.stage=='summarize':summarize()
  else:tests();train();predict();evaluate()
 except Exception as e:status('FAILED',error=str(e),traceback=traceback.format_exc());raise
