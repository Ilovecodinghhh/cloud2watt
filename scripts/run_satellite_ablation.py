"""Fixed satellite-processing ablation on immutable rolling development splits."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['HF_HUB_OFFLINE']='1'
import json,time,copy,shutil
from pathlib import Path
import numpy as np,pandas as pd,torch,zarr
import train_local_gpu as t
import download_route_b as r
from cloud2watt.satellite_ablation import VARIANTS,crop_stats,make_satellite,fit_transform
from cloud2watt.data.shared_tiles import open_satellite_frames
ROOT=t.ROOT; OUT=ROOT/'outputs/project-v2/satellite-ablation-20261009'; ROLL=ROOT/'outputs/project-v2/rolling-spring-20261008'; AUDIT=ROOT/'outputs/project-v2/satellite-quality-20261008'
FOLDS=['early','middle','extended']; SEEDS=[20261008,20261009,20261010]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(stage,**kw):t.atomic_json(OUT/'status.json',{'stage':stage,'final_labels_read':False,'network_requests':0,**kw})
def load(fold):
 with np.load(ROLL/fold/'dataset.npz') as z:return {k:z[k] for k in z.files}
def check_resources():
 config=read(ROOT/'configs/project-v2/route-b-freeze.json'); used=r.Budget(config).usage()+r.store_bytes(ROOT/'data/processed/route-b-spring-supplement-v1'); disk=shutil.disk_usage(ROOT)
 assert used+1024**3<config['resource_limits']['new_persistent_bytes_cap'];assert disk.free-1024**3>disk.total*.2
 t.atomic_json(OUT/'resource-check.json',{'persistent_bytes':used,'reserved_growth':1024**3,'disk_free_bytes':disk.free,'disk_free_fraction':disk.free/disk.total})
def frame_features():
 path=OUT/'frame-features.parquet'; receipt=OUT/'frame-features-receipt.json'
 if receipt.exists():
  assert t.file_hash(path)==read(receipt)['sha256'];return pd.read_parquet(path).set_index(['site','time'])
 needed=set()
 for fold in FOLDS:
  d=load(fold)
  for h in [-45,-30,-15,0]:needed.update(zip(d['site'],pd.to_datetime(d['time'],utc=True)+pd.Timedelta(minutes=h)))
 audit=pd.read_parquet(AUDIT/'frame-channel-audit.parquet',columns=['site','time','channel','elevation']);solar=audit.loc[audit.channel=='VIS006'].set_index(['site','time']).elevation
 cfg=read(ROOT/'configs/project-v2/local-first.json');tasks=[(part,ROOT/'data/processed/route-b-development-v1'/part,rec) for part,rec in cfg['completed_snapshot'].items()]
 extra=read(ROOT/'outputs/project-v2/spring-supplement-20261008/completion-progress.json')['completed'];tasks += [(part,ROOT/'data/processed/route-b-spring-supplement-v1'/part,rec) for part,rec in extra.items()]
 rows=[];seen=set();sources={}
 for i,(part,path0,rec) in enumerate(sorted(tasks),1):
  assert t.file_hash(path0/'verified.json')==rec['receipt_sha256'];sources[part]=rec['receipt_sha256']
  z=zarr.open_group(str(path0/'satellite_frames.zarr'),mode='r');frames=open_satellite_frames(path0/'satellite_frames.zarr');times=pd.to_datetime(np.asarray(z['time_ns']),utc=True);day=pd.to_datetime(part.split('/')[1],format='%Y%m%d',utc=True);ids=list(z.attrs['site_ids'])
  for ti in np.flatnonzero((times>=day)&(times<day+pd.Timedelta(days=1))):
   local={}
   for si,site in enumerate(ids):
    key=(str(site),times[ti])
    if key not in needed:continue
    assert key not in seen;seen.add(key);origin=tuple(z.attrs['site_origins'][si])
    if origin not in local:local[origin]=crop_stats(np.asarray(frames[si,int(ti)]))
    rows.append({'site':key[0],'time':key[1],'elevation':float(solar.loc[key]),**{str(j):float(v) for j,v in enumerate(local[origin])}})
  write('local_feature_extraction',completed=i,total=len(tasks));print('features',i,len(seen),flush=True)
 # A missing daily partition can still have valid history in its neighbor's context.
 context_count=0
 if seen!=needed:
  from cloud2watt.data.solar import build_solar_features
  for part,path0,rec in sorted(tasks):
   z=zarr.open_group(str(path0/'satellite_frames.zarr'),mode='r');times=pd.to_datetime(np.asarray(z['time_ns']),utc=True);ids=list(z.attrs['site_ids'])
   missing=needed-seen; requests=[(si,ti,str(site),timestamp) for si,site in enumerate(ids) for ti,timestamp in enumerate(times) if (str(site),timestamp) in missing]
   if not requests:continue
   sites=pd.read_parquet(path0/'sites.parquet');sun=build_solar_features(pd.DatetimeIndex(sorted({x[3] for x in requests})),sites).set_index(['ss_id','datetime_GMT']).solar_elevation_deg;frames=open_satellite_frames(path0/'satellite_frames.zarr')
   for si,ti,site,timestamp in requests:
    key=(site,timestamp);v=crop_stats(np.asarray(frames[si,ti]));rows.append({'site':site,'time':timestamp,'elevation':float(sun.loc[key]),**{str(j):float(x) for j,x in enumerate(v)}});seen.add(key);context_count+=1
   if seen==needed:break
 assert seen==needed,('missing feature keys',len(needed-seen))
 f=pd.DataFrame(rows).sort_values(['site','time']).reset_index(drop=True);f.to_parquet(path,index=False)
 t.atomic_json(receipt,{'sha256':t.file_hash(path),'rows':len(f),'context_only_frames':context_count,'source_receipts':sources,'audit_sha256':t.file_hash(AUDIT/'frame-channel-audit.parquet')})
 return f.set_index(['site','time'])
def inputs(d,features):
 issues=pd.to_datetime(d['time'],utc=True);local=[];elev=[]
 for h in [-45,-30,-15,0]:
  key=pd.MultiIndex.from_arrays([d['site'],issues+pd.Timedelta(minutes=h)]);v=features.loc[key];local.append(v[[str(j) for j in range(15)]].to_numpy(dtype='float32'));elev.append(v.elevation.to_numpy(dtype='float32'))
 local=np.stack(local,axis=1);elev=np.stack(elev,axis=1)
 # The control vector must be reconstructed exactly; all variants retain dimensions.
 np.testing.assert_array_equal(make_satellite(d['sat'],local,elev),d['sat'])
 return local,elev

def train(d,variant,seed,folder,local,elev,start):
 day,robust,small=VARIANTS[variant];sat=make_satellite(d['sat'],local,elev,daylight=day,small=small);fit=d['role']=='fit';inner=d['role']=='inner';x,params=fit_transform(d['base'],sat,fit,robust=robust)
 assert x.shape[1]==132
 identity={'parent_sha256':t.file_hash(OUT/'preregistration.json'),'dataset_sha256':t.file_hash(ROLL/folder.parent.name/'dataset.npz'),'frame_features_sha256':read(OUT/'frame-features-receipt.json')['sha256'],'variant':variant,'seed':seed}
 folder.mkdir(parents=True,exist_ok=True);reg=folder/'preregistration.json'
 if reg.exists():assert read(reg)==identity
 else:t.atomic_json(reg,identity)
 if (folder/'complete.json').exists():return
 t.atomic_json(folder/'normalization.json',{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in params.items()})
 t.seed_everything(seed);m=t.model(x.shape[1]);opt=torch.optim.Adam(m.parameters(),lr=.001);gen=torch.Generator().manual_seed(seed)
 x=torch.tensor(x,device='cuda');y=torch.tensor(d['y'],device='cuda');mask=torch.tensor(d['mask'],device='cuda');history=[];best=float('inf');best_state=None;latest=folder/'latest.pt';epoch0=0
 if latest.exists():
  c=torch.load(latest,weights_only=False);assert c['identity']==t.file_hash(reg);m.load_state_dict(c['model']);opt.load_state_dict(c['opt']);t.restore_rng(c['rng'],{});gen.set_state(c['generator']);history=c['history'];best=c['best'];best_state=c['best_state'];epoch0=c['epoch']
 for ep in range(epoch0,20):
  assert time.monotonic()-start<6*3600,'total six-hour guard'
  loss=t.epoch(m,opt,x[fit],y[fit],mask[fit],gen);m.eval()
  with torch.no_grad():pred=m(x[inner]).cpu().numpy()
  daylight=d['mask'][inner]&(d['elevation'][inner]>5);score=float(np.mean([abs(pred[:,h]-d['y'][inner,h])[daylight[:,h]].mean() for h in [1,2,3]]))
  if score<best:best=score;best_state=copy.deepcopy(m.state_dict())
  history.append({'epoch':ep+1,'train_mae':loss,'inner_daylight_primary_mae':score})
  t.atomic_torch(latest,{'identity':t.file_hash(reg),'model':m.state_dict(),'opt':opt.state_dict(),'rng':t.capture_rng({}),'generator':gen.get_state(),'history':history,'best':best,'best_state':best_state,'epoch':ep+1})
  t.atomic_json(folder/'history.json',history);print(folder.parent.name,variant,seed,ep+1,score,flush=True)
 m.load_state_dict(best_state);m.eval()
 with torch.no_grad():prediction=torch.cat([m(v).cpu() for v in x.split(2048)]).numpy()
 t.atomic_torch(folder/'best.pt',{'model':best_state,'input_dim':132,'params':params,'variant':variant,'inner_score':best})
 metrics={}
 for role in ['inner','outer_development','spring_stress_development','spring_outer_development']:
  take=d['role']==role;metrics[role]=t.score_forecast(prediction[take],d['y'][take],d['mask'][take],d['elevation'][take],pd.DataFrame({'site_id':d['site'][take],'issue_time_utc':d['time'][take]}))
 t.atomic_json(folder/'metrics.json',metrics)
 with t.atomic_path(folder/'predictions.npz') as temp:
  with temp.open('wb') as stream:np.savez_compressed(stream,prediction=prediction,target=d['y'],mask=d['mask'],role=d['role'],site=d['site'],time=d['time'])
 t.atomic_json(folder/'complete.json',{'epochs':len(history),'prediction_sha256':t.file_hash(folder/'predictions.npz'),'checkpoint_sha256':t.file_hash(folder/'best.pt'),'final_labels_read':False})

def main():
 OUT.mkdir(exist_ok=True);start=time.monotonic();assert 'raincalib-gpu' in t.sys.executable and torch.cuda.is_available();torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
 with t.run_lock(OUT):
  check_resources()
  reg={'variants':VARIANTS,'folds':read(ROLL/'preregistration.json')['folds'],'seeds':SEEDS,'epochs':20,'dataset_hashes':{fold:t.file_hash(ROLL/fold/'dataset.npz') for fold in FOLDS},'gpu':torch.cuda.get_device_name(0),'script_sha256':t.file_hash(Path(__file__)),'transform_sha256':t.file_hash(ROOT/'src/cloud2watt/satellite_ablation.py'),'decision_sha256':t.file_hash(ROOT/'docs/adr/019-satellite-processing-ablation.md'),'network_requests':0,'final_labels_read':False}
  reg=json.loads(json.dumps(reg))
  if (OUT/'preregistration.json').exists():assert read(OUT/'preregistration.json')==reg
  else:t.atomic_json(OUT/'preregistration.json',reg)
  features=frame_features();done=0
  for fold in FOLDS:
   d=load(fold);local,elev=inputs(d,features)
   for seed in SEEDS:
    old=ROLL/fold/str(seed);assert read(old/'preregistration.json')['dataset_sha256']==reg['dataset_hashes'][fold]
    for kind in ['power_solar','satellite_stats','smart_persistence','persistence']:
     with np.load(old/(kind+'-predictions.npz')) as z:
      for k in ['site','time','role','mask']:np.testing.assert_array_equal(z[k],d[k])
      np.testing.assert_array_equal(z['target'],d['y'])
    for variant in VARIANTS:
     folder=OUT/fold/(variant+'-'+str(seed));write('training',fold=fold,variant=variant,seed=seed,completed=done,total=36,gpu=reg['gpu']);train(d,variant,seed,folder,local,elev,start);done+=1
  write('training_complete',completed=done,total=36,elapsed_seconds=time.monotonic()-start,gpu=reg['gpu']);print('TRAINING_COMPLETE',done,flush=True)
if __name__=='__main__':main()
