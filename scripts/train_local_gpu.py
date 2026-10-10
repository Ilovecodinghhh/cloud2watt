"""Offline GPU diagnostic: paired power/solar vs local satellite statistics."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['HF_HUB_OFFLINE']='1'
import sys,json,time,copy,subprocess,shutil,argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from cloud2watt.run_state import atomic_json,atomic_torch,atomic_path,file_hash,run_lock,capture_rng,restore_rng
from cloud2watt.training import seed_everything,SOLAR_COLUMNS,SITE_COLUMNS
from cloud2watt.data.shared_tiles import open_satellite_frames
from cloud2watt.data.solar import build_solar_features
from cloud2watt.evaluation.forecast import score_forecast
ROOT=Path('D:/code/AI_atm'); OUT=ROOT/'outputs/project-v2/local-gpu-20261008'
DEST=ROOT/'data/processed/route-b-development-v1'
INDEX=ROOT/'outputs/project-v2/local-provenance-20261007/safe-split-index.parquet'
SEED=20261008

def model(dim): return nn.Sequential(nn.Linear(dim,128),nn.SiLU(),nn.Dropout(.1),nn.Linear(128,128),nn.SiLU(),nn.Dropout(.1),nn.Linear(128,6)).cuda()
def epoch(m,opt,x,y,mask,generator):
 m.train(); total=0.; count=0
 for ids in torch.randperm(len(x),generator=generator).split(512):
  ids=ids.cuda(); pred=m(x[ids]); good=mask[ids]; loss=(pred-y[ids]).abs()[good].mean()
  assert torch.isfinite(loss)
  opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True); opt.step()
  total+=loss.item()*int(good.sum()); count+=int(good.sum())
 return total/count

def recovery(which):
 seed_everything(SEED); x=torch.arange(128*8,device='cuda').reshape(128,8).float()/1024; y=x[:,:6]*.3; mask=torch.ones_like(y,dtype=torch.bool)
 m=model(8); opt=torch.optim.Adam(m.parameters(),lr=.001); gen=torch.Generator().manual_seed(SEED)
 if which=='resume':
  c=torch.load(OUT/'e02-split.pt',weights_only=False); m.load_state_dict(c['model']); opt.load_state_dict(c['opt']); restore_rng(c['rng'],{}); gen.set_state(c['generator']); start=1
 else: start=0
 for i in range(start,1 if which=='split' else 2): epoch(m,opt,x,y,mask,gen)
 atomic_torch(OUT/('e02-'+which+'.pt'),{'model':m.state_dict(),'opt':opt.state_dict(),'rng':capture_rng({}),'generator':gen.get_state()})

def prepare(config_path=None, satellite_root=None):
 index=pd.read_parquet(INDEX); cfg=json.loads((config_path or ROOT/'configs/project-v2/local-first.json').read_text(encoding='utf-8'))
 assert file_hash(INDEX)==cfg['local_audit']['sha256']
 cache=OUT/'features'; cache.mkdir(exist_ok=True)
 # Coordinates and future timestamps only; no future measured inputs.
 sites=pd.concat([pd.read_parquet(DEST/p/'sites.parquet') for p in index.partition.unique()]).drop_duplicates('ss_id')
 times=pd.DatetimeIndex(sorted(set(t+pd.Timedelta(minutes=h) for t in index.issue_time_utc.unique() for h in [0,15,30,60,120,180,240])))
 solar=build_solar_features(times,sites).set_index(['ss_id','datetime_GMT'])
 blocks=[]
 for partition,keys in index.groupby('partition',sort=True):
  d=DEST/partition; name=partition.replace('/','_'); dest=cache/(name+'.npz')
  assert file_hash(d/'verified.json')==cfg['completed_snapshot'][partition]['receipt_sha256']
  if dest.exists():
   with np.load(dest) as z: blocks.append({k:z[k] for k in z.files})
   continue
  p=pd.read_parquet(d/'power_15min.parquet'); samples=pd.read_parquet(d/'sample_index.parquet').iloc[keys.sample_row.to_numpy()]; metadata=pd.read_parquet(d/'sites.parquet').set_index('ss_id')
  hist=np.stack(samples.power_history_row_indices); target=np.stack(samples.target_row_indices); mask=np.stack(samples.target_mask).astype(bool)&(target>=0)
  y=p.normalized_power.to_numpy()[np.maximum(target,0)]; y=np.where(mask,y,0).astype('float32')
  history=p.normalized_power.to_numpy()[hist].astype('float32')
  sf=[]; static=[]; ghi0=[]
  for row in samples.itertuples():
   tt=[row.issue_time_utc+pd.Timedelta(minutes=h) for h in [15,30,60,120,180,240]]
   sf.append(solar.loc[[(str(row.site_id),t) for t in tt],list(SOLAR_COLUMNS)].to_numpy())
   static.append(metadata.loc[str(row.site_id),list(SITE_COLUMNS)].to_numpy(dtype=float))
   ghi0.append(solar.loc[(str(row.site_id),row.issue_time_utc),'clear_sky_ghi_wm2'])
  sf=np.asarray(sf,dtype='float32'); static=np.asarray(static,dtype='float32')
  frames=open_satellite_frames((satellite_root/partition if satellite_root is not None else d)/'satellite_frames.zarr'); frame_stats={}
  unique=sorted({(int(row.site_index),int(t)) for row in samples.itertuples() for t in row.satellite_frame_indices},key=lambda x:(x[1],x[0]))
  for site,t in unique:
   a=np.asarray(frames[site,t],dtype='float32'); h,w=a.shape[-2:]
   with np.errstate(invalid='ignore'):
    v=np.concatenate([np.nanmean(a[:,:h//2,:w//2],axis=(1,2)),np.nanmean(a[:,:h//2,w//2:],axis=(1,2)),np.nanmean(a[:,h//2:,:w//2],axis=(1,2)),np.nanmean(a[:,h//2:,w//2:],axis=(1,2)),np.nanstd(a,axis=(1,2))])
   assert np.isfinite(v).all(),(partition,site,t)
   frame_stats[site,t]=v
  sat=np.asarray([[frame_stats[int(row.site_index),int(t)] for t in row.satellite_frame_indices] for row in samples.itertuples()],dtype='float32')
  sat=np.concatenate([sat.reshape(len(sat),-1),np.diff(sat,axis=1).reshape(len(sat),-1)],axis=1)
  base=np.concatenate([history,sf.reshape(len(sf),-1),static],axis=1).astype('float32')
  b={'base':base,'sat':sat,'y':y,'mask':mask,'elevation':sf[:,:,0],'future_ghi':sf[:,:,2],'ghi0':np.asarray(ghi0),'site':samples.site_id.astype(str).to_numpy(dtype=str),'time':samples.issue_time_utc.astype(str).to_numpy(dtype=str),'role':keys.role.to_numpy(dtype=str)}
  assert np.isfinite(base).all() and np.isfinite(y).all()
  with atomic_path(dest) as temp:
   with temp.open('wb') as stream: np.savez_compressed(stream,**b)
  blocks.append(b); atomic_json(OUT/'status.json',{'stage':'local_feature_extraction','last_partition':partition,'completed':len(blocks),'total':index.partition.nunique(),'network_requests':0}); print('features',len(blocks),partition,flush=True)
 return {k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]}

def train(data):
 fit=data['role']=='fit'; inner=data['role']=='inner'; metrics={}
 for kind in ['persistence','smart_persistence','power_solar','satellite_stats']:
  if kind in ['power_solar','satellite_stats']:
   raw=data['base'] if kind=='power_solar' else np.concatenate([data['base'],data['sat']],axis=1)
   mean=raw[fit].mean(0); scale=raw[fit].std(0); scale[scale<1e-6]=1
   atomic_json(OUT/(kind+'-normalization.json'),{'mean':mean.tolist(),'scale':scale.tolist(),'fit_only':True})
   x=torch.tensor((raw-mean)/scale,device='cuda'); y=torch.tensor(data['y'],device='cuda'); mask=torch.tensor(data['mask'],device='cuda')
   seed_everything(SEED); m=model(x.shape[1]); opt=torch.optim.Adam(m.parameters(),lr=.001); gen=torch.Generator().manual_seed(SEED)
   history=[]; best=float('inf'); best_state=None; latest=OUT/(kind+'-latest.pt'); elapsed=0.; start=0
   if latest.exists():
    c=torch.load(latest,weights_only=False); assert c['identity']==file_hash(OUT/'preregistration.json'); m.load_state_dict(c['model']); opt.load_state_dict(c['opt']); restore_rng(c['rng'],{}); gen.set_state(c['generator']); history=c['history']; best=c['best']; best_state=c['best_state']; start=c['epoch']; elapsed=c['elapsed']
   for ep in range(start,20):
    tick=time.monotonic(); loss=epoch(m,opt,x[fit],y[fit],mask[fit],gen); m.eval()
    with torch.no_grad(): pred=m(x[inner]).cpu().numpy()
    day=data['mask'][inner]&(data['elevation'][inner]>5); score=float(np.mean([np.abs(pred[:,h]-data['y'][inner,h])[day[:,h]].mean() for h in [1,2,3]]))
    if score<best: best=score; best_state=copy.deepcopy(m.state_dict())
    elapsed+=time.monotonic()-tick; history.append({'epoch':ep+1,'train_mae':loss,'inner_daylight_primary_mae':score})
    atomic_torch(latest,{'identity':file_hash(OUT/'preregistration.json'),'model':m.state_dict(),'opt':opt.state_dict(),'rng':capture_rng({}),'generator':gen.get_state(),'history':history,'best':best,'best_state':best_state,'epoch':ep+1,'elapsed':elapsed})
    atomic_json(OUT/(kind+'-history.json'),history); print(kind,ep+1,score,flush=True)
    if elapsed>3500: break
   m.load_state_dict(best_state); m.eval()
   with torch.no_grad(): prediction=torch.cat([m(v).cpu() for v in x.split(2048)]).numpy()
   atomic_torch(OUT/(kind+'-best.pt'),{'model':best_state,'input_dim':x.shape[1],'mean':mean,'scale':scale,'inner_score':best})
  else:
   prediction=np.repeat(data['base'][:,0:4][:,-1:],6,axis=1)
   if kind=='smart_persistence': prediction=np.where(data['ghi0'][:,None]>20,prediction*np.clip(data['future_ghi']/np.maximum(data['ghi0'][:,None],20),0,5),prediction)
  metrics[kind]={}
  for role in ['inner','outer_development','spring_stress_development']:
   take=data['role']==role; keys=pd.DataFrame({'site_id':data['site'][take],'issue_time_utc':data['time'][take]})
   metrics[kind][role]=score_forecast(prediction[take],data['y'][take],data['mask'][take],data['elevation'][take],keys)
  with atomic_path(OUT/(kind+'-predictions.npz')) as temp:
   with temp.open('wb') as stream: np.savez_compressed(stream,prediction=prediction,target=data['y'],mask=data['mask'],role=data['role'],site=data['site'],time=data['time'])
  atomic_json(OUT/'metrics.json',metrics)
 atomic_json(OUT/'status.json',{'stage':'complete','models':list(metrics),'gpu':torch.cuda.get_device_name(0),'final_labels_read':False,'network_requests':0,'scope':'single_seed_local_development_diagnostic'})

def main():
 OUT.mkdir(exist_ok=True); torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
 assert 'raincalib-gpu' in sys.executable and torch.cuda.is_available()
 if len(sys.argv)>1: recovery(sys.argv[1]); return
 with run_lock(OUT):
  if (OUT/'status.json').exists() and json.loads((OUT/'status.json').read_text())['stage']=='complete': return
  assert shutil.disk_usage(ROOT).free/shutil.disk_usage(ROOT).total>.2
  if not (OUT/'e02-acceptance.json').exists():
   for mode in ['continuous','split','resume']: subprocess.run([sys.executable,__file__,mode],check=True)
   a=torch.load(OUT/'e02-continuous.pt',weights_only=False); b=torch.load(OUT/'e02-resume.pt',weights_only=False)
   assert all(torch.equal(a['model'][k],b['model'][k]) for k in a['model'])
   assert torch.equal(a['generator'],b['generator'])
   atomic_json(OUT/'e02-acceptance.json',{'passed':True,'cross_process':True,'random_order':True,'dropout':True,'jitter':0,'workers':0,'scope':'this compact-feature trainer only'})
  data=prepare(); train(data)
if __name__=='__main__': main()
