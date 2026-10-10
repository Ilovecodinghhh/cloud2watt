"""Offline frozen-model prospective evaluation, without fitting or tuning."""
import os
os.environ['HF_HUB_OFFLINE']='1'
import json,time,shutil
from pathlib import Path
import numpy as np,pandas as pd,torch,zarr
import train_local_gpu as t
import download_route_b as r
import train_rolling_spring as rr
from train_rolling_spring import rebuild
from cloud2watt.data.role_quality import CONTEXT_BITS
from cloud2watt.data.shared_tiles import open_satellite_frames
from cloud2watt.satellite_ablation import crop_stats,make_satellite,apply_transform
ROOT=t.ROOT;OUT=ROOT/'outputs/project-v2/prospective-evaluation-20261009';SOURCE=ROOT/'data/processed/route-b-spring-validation-v1';STATE=ROOT/'outputs/project-v2/spring-validation-20261009'
SEEDS=[20261008,20261009,20261010];H=[15,30,60,120,180,240]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def stamp(stage,**kw):t.atomic_json(OUT/'status.json',{'stage':stage,'training':False,'final_labels_read':False,'network_requests':0,**kw})
def prepare():
    completed=read(ROOT/'outputs/project-v2/spring-validation-20261009/completion-progress.json')['completed']
    views=OUT/'views'; parts=[]; power=[]; snapshots={}
    for n,(part,receipt) in enumerate(sorted(completed.items()),1):
        d=SOURCE/part; assert t.file_hash(d/'verified.json')==receipt['receipt_sha256']; r.verify_shard(d)
        p=pd.read_parquet(d/'power_15min.parquet'); sites=pd.read_parquet(d/'sites.parquet')
        # All pointwise inputs, without retaining the old context-dependent selection.
        samples=rebuild(d,p,sites); v=views/part; v.mkdir(parents=True,exist_ok=True)
        p.to_parquet(v/'power_15min.parquet',index=False); sites.to_parquet(v/'sites.parquet',index=False); samples.to_parquet(v/'sample_index.parquet',index=False)
        t.atomic_json(v/'verified.json',{'source_receipt_sha256':receipt['receipt_sha256'],'files':{f:t.file_hash(v/f) for f in ['power_15min.parquet','sites.parquet','sample_index.parquet']}})
        snapshots[part]={'receipt_sha256':t.file_hash(v/'verified.json')}
        q=samples[['site_id','issue_time_utc']].copy(); q['partition']=part; q['sample_row']=np.arange(len(q)); q['role']='pending_role_qc'; parts.append(q); power.append(p)
        stamp('rebuild_candidates',completed=n,total=len(completed))
    index=pd.concat(parts,ignore_index=True); assert not index.duplicated(['site_id','issue_time_utc']).any()
    idx=OUT/'candidate-index.parquet'; index.to_parquet(idx,index=False)
    cfg=OUT/'feature-config.json'; t.atomic_json(cfg,{'completed_snapshot':snapshots,'local_audit':{'sha256':t.file_hash(idx)}})
    t.OUT=OUT; t.INDEX=idx; t.DEST=views
    data=t.prepare(cfg,satellite_root=SOURCE)
    p=pd.concat(power,ignore_index=True); p['base_flags']=p.quality_flags.astype('uint16')&np.uint16(65535^CONTEXT_BITS)
    # Overlapping shard contexts must describe the same source observations.
    cols=['normalized_power','reading_count','base_flags']
    assert (p.groupby(['ss_id','datetime_GMT'])[cols].nunique(dropna=False)<=1).all().all()
    p=p.drop_duplicates(['ss_id','datetime_GMT']).sort_values(['ss_id','datetime_GMT']).reset_index(drop=True)
    p.to_parquet(OUT/'spring-power-grid.parquet',index=False)
    return data,p

def ci(diff,mask,dates,length=1):
 days=np.unique(dates); sums=np.array([(diff[dates==d]*mask[dates==d]).sum(0) for d in days]);counts=np.array([mask[dates==d].sum(0) for d in days]);rng=np.random.default_rng(20261009)
 if length==1: ids=rng.integers(len(days),size=(10000,len(days)))
 else:
  starts=rng.integers(len(days),size=(10000,int(np.ceil(len(days)/length))));ids=((starts[:,:,None]+np.arange(length))%len(days)).reshape(10000,-1)[:,:len(days)]
 den=counts[ids].sum(1);good=(den>0).all(1);values=(sums[ids].sum(1)[good]/den[good]).mean(1)
 return {'delta':float((sums.sum(0)/counts.sum(0)).mean()),'ci95':np.quantile(values,[.025,.975]).tolist(),'dates':len(days),'block_length':length,'draws':len(values)}
def score(e,m):
 counts=m.sum(0);good=counts>0
 return float(((e*m).sum(0)[good]/counts[good]).mean()) if good.any() else None

def main():
 start=time.monotonic();OUT.mkdir(exist_ok=True);assert 'raincalib-gpu' in t.sys.executable and torch.cuda.is_available();torch.set_num_threads(4)
 with t.run_lock(OUT):
  freeze=read(STATE/'evaluation-freeze.json')
  for name,digest in freeze['model_hashes'].items():assert t.file_hash(ROOT/name)==digest
  usage=r.Budget(read(r.FREEZE)).usage()+r.store_bytes(ROOT/'data/processed/route-b-spring-supplement-v1')+r.store_bytes(SOURCE);disk=shutil.disk_usage(ROOT);assert usage+1024**3<16*1024**3 and disk.free-1024**3>disk.total*.2
  reg={'model_freeze_sha256':t.file_hash(STATE/'evaluation-freeze.json'),'decision_sha256':t.file_hash(ROOT/'docs/adr/021-frozen-spring-validation-evaluation.md'),'script_sha256':t.file_hash(Path(__file__)),'source_acceptance_sha256':t.file_hash(STATE/'completion-acceptance.json'),'gpu':torch.cuda.get_device_name(0)}
  if (OUT/'preregistration.json').exists():assert read(OUT/'preregistration.json')==reg
  else:t.atomic_json(OUT/'preregistration.json',reg)
  data,p=prepare();rr.OUT=OUT;rr.CURRENT='new_validation';(OUT/'qc/new_validation').mkdir(parents=True,exist_ok=True)
  d,audit=rr.assign(data,p,{'prospective_validation':['2021-04-21','2021-05-05']});t.atomic_json(OUT/'split-audit.json',audit)
  with t.atomic_path(OUT/'dataset.npz') as temp:
   with temp.open('wb') as f:np.savez_compressed(f,**d)
  # Extract the fixed central32 summary without fitting any transform.
  keylookup={(str(site),str(tm)):i for i,(site,tm) in enumerate(zip(d['site'],d['time']))};local=np.full((len(d['site']),4,15),np.nan,np.float32)
  index=pd.read_parquet(OUT/'candidate-index.parquet')
  for n,(part,keys) in enumerate(index.groupby('partition',sort=True),1):
   samples=pd.read_parquet(OUT/'views'/part/'sample_index.parquet');frames=open_satellite_frames(SOURCE/part/'satellite_frames.zarr');cache={}
   for row in samples.itertuples():
    i=keylookup.get((str(row.site_id),str(row.issue_time_utc)))
    if i is None:continue
    for j,ti in enumerate(row.satellite_frame_indices):
     key=(int(row.site_index),int(ti))
     if key not in cache:cache[key]=crop_stats(np.asarray(frames[key[0],key[1]]))
     local[i,j]=cache[key]
   stamp('central32_features',completed=n,total=42)
  assert np.isfinite(local).all();sat32=make_satellite(d['sat'],local,np.zeros((len(local),4)),small=True)
  np.save(OUT/'local32-satellite.npy',sat32)
  dates=np.array([x[:10] for x in d['time']]);mask=d['mask']&(d['elevation']>5);errors={};seed_scores={};predictions={}
  for kind in ['power_solar','satellite_stats','local32']:
   errs=[];seed_scores[kind]=[]
   for seed in SEEDS:
    if kind=='local32':
     folder=ROOT/'outputs/project-v2/satellite-ablation-20261009/extended'/f'local32-{seed}';path=folder/'best.pt';c=torch.load(path,weights_only=False);x=apply_transform(d['base'],sat32,c['params'])
    else:
     folder=ROOT/'outputs/project-v2/rolling-spring-20261008/extended'/str(seed);path=folder/f'{kind}-best.pt';c=torch.load(path,weights_only=False);raw=d['base'] if kind=='power_solar' else np.concatenate([d['base'],d['sat']],axis=1);x=((raw-c['mean'])/c['scale']).astype('float32')
    assert path.relative_to(ROOT).as_posix() in freeze['model_hashes'];assert np.isfinite(x).all()
    net=t.model(x.shape[1]);net.load_state_dict(c['model']);net.eval()
    with torch.inference_mode():pred=torch.cat([net(torch.tensor(v,device='cuda')).cpu() for v in np.array_split(x,max(1,int(np.ceil(len(x)/2048))))]).numpy()
    assert np.isfinite(pred).all();predictions[kind+'-'+str(seed)]=pred;e=abs(pred-d['y']);errs.append(e);seed_scores[kind].append(score(e[:,[1,2,3]],mask[:,[1,2,3]]))
   errors[kind]=np.mean(errs,axis=0)
  last=d['base'][:,3:4];pred=np.where(d['ghi0'][:,None]>20,last*np.clip(d['future_ghi']/np.maximum(d['ghi0'][:,None],20),0,5),np.repeat(last,6,axis=1));predictions['smart_persistence']=pred;errors['smart_persistence']=abs(pred-d['y'])
  np.savez_compressed(OUT/'predictions.npz',**predictions,target=d['y'],mask=d['mask'],site=d['site'],time=d['time'])
  with np.load(ROOT/'outputs/project-v2/rolling-spring-20261008/extended/dataset.npz') as z:
   fit=(z['role']=='fit')&(z['elevation'][:,0]>5);thresholds=np.quantile(np.max(abs(np.diff(z['base'][fit,:4],axis=1)),axis=1),[1/3,2/3]);trainkeys=set(zip(z['site'][z['role']=='fit'],z['time'][z['role']=='fit']))
  assert not trainkeys.intersection(keylookup)
  volatility=np.max(abs(np.diff(d['base'][:,:4],axis=1)),axis=1);bins=np.digitize(volatility,thresholds)
  overall={k:score(e[:,[1,2,3]],mask[:,[1,2,3]]) for k,e in errors.items()};groups={}
  for name,select in [('low_sun',(d['elevation']>5)&(d['elevation']<20)),('medium_sun',(d['elevation']>=20)&(d['elevation']<40)),('high_sun',d['elevation']>=40)]+[(name,np.repeat((bins==i)[:,None],6,axis=1)) for i,name in enumerate(['stable_power','medium_variation','high_variation'])]:
   m=(mask&select)[:,[1,2,3]];groups[name]={'mae':{k:score(e[:,[1,2,3]],m) for k,e in errors.items()},'target_counts':m.sum(0).tolist(),'dates':int(len(np.unique(dates[m.any(1)])))}
  comparisons={k:{'date_blocks':ci((errors[k]-errors['power_solar'])[:,[1,2,3]],mask[:,[1,2,3]],dates),'three_day_blocks':ci((errors[k]-errors['power_solar'])[:,[1,2,3]],mask[:,[1,2,3]],dates,3)} for k in ['local32','satellite_stats','smart_persistence']}
  horizon={str(h):{k:score(e[:,j:j+1],mask[:,j:j+1]) for k,e in errors.items()} for j,h in enumerate(H)}
  daily={day:{k:score(e[dates==day][:,[1,2,3]],mask[dates==day][:,[1,2,3]]) for k,e in errors.items()} for day in sorted(set(dates))}
  result={'overall_daylight_mae':overall,'seed_scores':seed_scores,'paired_vs_power_solar':comparisons,'by_horizon':horizon,'groups_exploratory':groups,'volatility_thresholds_train_only':thresholds.tolist(),'daily':daily,'samples':len(dates),'dates':len(set(dates)),'valid_daylight_targets_primary':mask[:,[1,2,3]].sum(0).tolist(),'training':False,'final_labels_read':False}
  t.atomic_json(OUT/'results.json',result)
  for name,digest in freeze['model_hashes'].items():assert t.file_hash(ROOT/name)==digest
  t.atomic_json(OUT/'acceptance.json',{'passed':True,'models_unchanged':True,'same_samples_targets_masks':True,'no_fit_overlap':True,'dataset_sha256':t.file_hash(OUT/'dataset.npz'),'predictions_sha256':t.file_hash(OUT/'predictions.npz'),'results_sha256':t.file_hash(OUT/'results.json')})
  stamp('complete',samples=len(dates),dates=len(set(dates)),elapsed_seconds=time.monotonic()-start);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
