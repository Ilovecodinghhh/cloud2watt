"""Frozen spring extension: local-only paired CUDA training, final labels locked."""
import json, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import train_local_gpu as t
import download_route_b as r
ROOT=t.ROOT
OUT=ROOT/'outputs/project-v2/spring-gpu-20261008'
DEST=ROOT/'data/processed/route-b-spring-supplement-v1'
STATE=ROOT/'outputs/project-v2/spring-supplement-20261008'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def main():
 OUT.mkdir(exist_ok=True)
 assert 'raincalib-gpu' in t.sys.executable and torch.cuda.is_available()
 torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
 with t.run_lock(OUT):
  assert shutil.disk_usage(ROOT).free/shutil.disk_usage(ROOT).total>.2
  plan=read(STATE/'plan.json'); completed=read(STATE/'completion-progress.json')['completed']
  roles=plan['proposed_future_roles']; rows=[]
  for part,receipt in sorted(completed.items()):
   d=DEST/part
   assert t.file_hash(d/'verified.json')==receipt['receipt_sha256']
   r.verify_shard(d)
   reg=read(d/'preregistration.json'); samples=pd.read_parquet(d/'sample_index.parquet')
   for role,bounds in roles.items():
    a,b=[pd.Timestamp(x,tz='UTC') for x in bounds]
    if pd.Timestamp(reg['support_start_inclusive'])<a or pd.Timestamp(reg['support_end_exclusive'])>b: continue
    keep=(samples.issue_time_utc-pd.Timedelta(hours=1)>=a)&(samples.issue_time_utc+pd.Timedelta(hours=4)<b)
    block=samples.loc[keep,['site_id','issue_time_utc']].copy(); block['sample_row']=np.flatnonzero(keep); block['partition']=part
    block['role']='spring_outer_development' if role=='outer_development' else role; rows.append(block)
  index=pd.concat(rows,ignore_index=True); assert not index.duplicated(['site_id','issue_time_utc']).any()
  idx=OUT/'safe-spring-index.parquet'
  if idx.exists(): pd.testing.assert_frame_equal(pd.read_parquet(idx),index)
  else: index.to_parquet(idx,index=False)
  cfg={'completed_snapshot':completed,'local_audit':{'sha256':t.file_hash(idx)}}
  config=OUT/'feature-config.json'; t.atomic_json(config,cfg)
  t.OUT=OUT; t.DEST=DEST; t.INDEX=idx
  new=t.prepare(config)
  old=[]; cache=ROOT/'outputs/project-v2/local-gpu-20261008/features'
  for p in sorted(cache.glob('*.npz')):
   with np.load(p) as z: old.append({k:z[k] for k in z.files})
  data={k:np.concatenate([b[k] for b in old]+[new[k]]) for k in new}
  keys=pd.DataFrame({'site':data['site'],'time':data['time']}); assert not keys.duplicated().any()
  assert sum(len(b['role']) for b in old)==76199
  counts={str(k):int(v) for k,v in pd.Series(data['role']).value_counts().items()}
  prereg={'roles':roles,'counts':counts,'spring_index_sha256':t.file_hash(idx),'seeds':[20261008,20261009,20261010],'epochs':20,'checkpoint_selection':'pooled winter and spring inner daylight MAE 30/60/120; fit-only normalization','outer_selection':False,'network_requests':0,'final_labels_read':False,'missing_station_10169':'retain valid winter; no spring imputation','qc_policy':'entire partition source support inside split; history -1h, targets +4h','gpu':torch.cuda.get_device_name(0),'old_feature_hashes':{p.name:t.file_hash(p) for p in sorted(cache.glob('*.npz'))}}
  target=OUT/'preregistration.json'
  if target.exists(): assert read(target)==prereg
  else: t.atomic_json(target,prereg)
  print('GPU',prereg['gpu'],'COUNTS',counts,flush=True)
  for seed in prereg['seeds']:
   t.SEED=seed; t.OUT=OUT/str(seed); t.OUT.mkdir(exist_ok=True)
   t.atomic_json(t.OUT/'preregistration.json',prereg)
   if not (t.OUT/'status.json').exists() or read(t.OUT/'status.json').get('stage')!='complete': t.train(data)
   extra={}; take=data['role']=='spring_outer_development'
   for kind in ['persistence','smart_persistence','power_solar','satellite_stats']:
    with np.load(t.OUT/(kind+'-predictions.npz')) as z: pred=z['prediction']
    extra[kind]=t.score_forecast(pred[take],data['y'][take],data['mask'][take],data['elevation'][take],pd.DataFrame({'site_id':data['site'][take],'issue_time_utc':data['time'][take]}))
   t.atomic_json(t.OUT/'spring-outer-metrics.json',extra)
  t.atomic_json(OUT/'status.json',{'stage':'complete','counts':counts,'gpu':prereg['gpu'],'seeds':prereg['seeds'],'final_labels_read':False})
if __name__=='__main__': main()
