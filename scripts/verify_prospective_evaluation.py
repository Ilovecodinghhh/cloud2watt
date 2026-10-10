"""Independent prediction replay and metric/boundary acceptance."""
import json
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from cloud2watt.run_state import file_hash,atomic_json
ROOT=Path('D:/code/AI_atm');OUT=ROOT/'outputs/project-v2/prospective-evaluation-20261009'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
assert read(OUT/'status.json')['stage']=='complete'
freeze=read(ROOT/'outputs/project-v2/spring-validation-20261009/evaluation-freeze.json');accept=read(OUT/'acceptance.json');result=read(OUT/'results.json')
for name,digest in freeze['model_hashes'].items():assert file_hash(ROOT/name)==digest
for name in ['dataset','predictions']:assert file_hash(OUT/(name+'.npz'))==accept[name+'_sha256']
with np.load(OUT/'dataset.npz') as z:d={k:z[k] for k in z.files}
with np.load(OUT/'predictions.npz') as z:pred={k:z[k] for k in z.files}
for key in ['site','time','mask']:np.testing.assert_array_equal(d[key],pred[key])
np.testing.assert_array_equal(d['y'],pred['target'])
issue=pd.DatetimeIndex(pd.to_datetime(d['time'],utc=True));assert (issue-pd.Timedelta(hours=1)>=pd.Timestamp('2021-04-21',tz='UTC')).all();assert (issue+pd.Timedelta(hours=4)<pd.Timestamp('2021-05-05',tz='UTC')).all()
assert len(set(zip(d['site'],d['time'])))==len(issue)
ids=np.arange(0,len(issue),379);sat32=np.load(OUT/'local32-satellite.npy');checked=[]
for kind in ['power_solar','satellite_stats','local32']:
 errors=[]
 for seed in [20261008,20261009,20261010]:
  if kind=='local32':
   c=torch.load(ROOT/'outputs/project-v2/satellite-ablation-20261009/extended'/f'local32-{seed}'/'best.pt',map_location='cpu',weights_only=False);raw=np.concatenate([d['base'][ids],sat32[ids]],axis=1);x=((raw-c['params']['center'])/c['params']['scale']).astype('float32');assert not c['params']['robust']
  else:
   c=torch.load(ROOT/'outputs/project-v2/rolling-spring-20261008/extended'/str(seed)/f'{kind}-best.pt',map_location='cpu',weights_only=False);raw=d['base'][ids] if kind=='power_solar' else np.concatenate([d['base'][ids],d['sat'][ids]],axis=1);x=((raw-c['mean'])/c['scale']).astype('float32')
  net=nn.Sequential(nn.Linear(x.shape[1],128),nn.SiLU(),nn.Dropout(.1),nn.Linear(128,128),nn.SiLU(),nn.Dropout(.1),nn.Linear(128,6));net.load_state_dict(c['model']);net.eval()
  with torch.inference_mode(): replay=net(torch.tensor(x)).numpy()
  np.testing.assert_allclose(replay,pred[f'{kind}-{seed}'][ids],rtol=1e-4,atol=3e-6);checked.append(f'{kind}-{seed}')
  errors.append(abs(pred[f'{kind}-{seed}']-d['y']))
 e=np.mean(errors,axis=0);scores=[]
 for j in [1,2,3]:
  valid=d['mask'][:,j]&(d['elevation'][:,j]>5);scores.append(e[valid,j].mean())
 np.testing.assert_allclose(np.mean(scores),result['overall_daylight_mae'][kind],atol=1e-7)
last=d['base'][:,3:4];base=np.where(d['ghi0'][:,None]>20,last*np.clip(d['future_ghi']/np.maximum(d['ghi0'][:,None],20),0,5),np.repeat(last,6,axis=1));np.testing.assert_array_equal(base,pred['smart_persistence'])
atomic_json(OUT/'independent-acceptance.json',{'passed':True,'cpu_replayed_models':checked,'replay_samples_each':len(ids),'same_targets_masks_keys':True,'metrics_independently_recomputed':True,'boundaries_passed':True,'frozen_model_hashes_passed':True,'results_sha256':file_hash(OUT/'results.json')});print('INDEPENDENT ACCEPTANCE PASSED',len(issue))
