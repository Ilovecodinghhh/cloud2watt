"""Frozen three-seed local validation; no network or final-label access."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json, time, copy, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from cloud2watt.run_state import atomic_json,atomic_torch,file_hash,run_lock
from cloud2watt.training import seed_everything
ROOT=Path('D:/code/AI_atm'); OLD=ROOT/'outputs/project-v2/local-gpu-20261008'; OUT=ROOT/'outputs/project-v2/mvp-v2'
spec=importlib.util.spec_from_file_location('local_train',ROOT/'scripts/train_local_gpu.py'); t=importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
SEEDS=[20261008,20261009,20261010]

def predict(checkpoint, raw):
 c=torch.load(checkpoint,weights_only=False); m=t.model(c['input_dim']); m.load_state_dict(c['model']); m.eval()
 x=torch.tensor((raw-c['mean'])/c['scale'],device='cuda',dtype=torch.float32)
 with torch.no_grad(): return torch.cat([m(v).cpu() for v in x.split(2048)]).numpy()

def blocked_ci(differences, mask, dates, *, draws=10000, seed=813):
 """Mean across seed errors; resample whole dates jointly across sites/horizons."""
 rng=np.random.default_rng(seed); unique=np.unique(dates)
 sums=np.array([(differences[dates==d]*mask[dates==d]).sum(0) for d in unique])
 counts=np.array([mask[dates==d].sum(0) for d in unique])
 point=float(np.mean(sums.sum(0)/counts.sum(0)))
 take=rng.integers(len(unique),size=(draws,len(unique))); den=counts[take].sum(1); good=(den>0).all(1)
 values=(sums[take].sum(1)[good]/den[good]).mean(1)
 return {'delta_mae':point,'ci95':np.quantile(values,[.025,.975]).tolist(),'dates':len(unique),'draws':len(values),'requested_draws':draws,'unit':'UTC issue date; all sites and paired seeds kept together','negative_means_satellite_better':True,'scope':'conditional on three seeds; not an independent final-test inference'}

def main():
 OUT.mkdir(exist_ok=True); torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
 assert torch.cuda.is_available() and 'raincalib-gpu' in __import__('sys').executable
 with run_lock(OUT):
  manifest=OUT/'preregistration.json'; assert manifest.exists()
  blocks=[]
  for p in sorted((OLD/'features').glob('*.npz')):
   with np.load(p) as z: blocks.append({k:z[k] for k in z.files})
  data={k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
  assert len(data['y'])==76199
  # Align old outputs explicitly: never assume npz concatenation order.
  with np.load(OLD/'power_solar-predictions.npz') as z:
   assert np.array_equal(data['site'],z['site']) and np.array_equal(data['time'],z['time'])
   assert np.array_equal(data['mask'],z['mask']) and np.array_equal(data['y'],z['target'])
  runs={SEEDS[0]:OLD}
  for seed in SEEDS[1:]:
   folder=OUT/str(seed); folder.mkdir(exist_ok=True); runs[seed]=folder
   if (folder/'status.json').exists() and json.loads((folder/'status.json').read_text())['stage']=='complete': continue
   atomic_json(folder/'preregistration.json',{'parent_sha256':file_hash(manifest),'seed':seed,'feature_cache_sha256':{p.name:file_hash(p) for p in sorted((OLD/'features').glob('*.npz'))}})
   t.OUT=folder; t.SEED=seed; t.train(data)
  # Additional power-only comparator makes the solar comparison a trained ablation.
  p=OUT/'power_only-best.pt'; fit=data['role']=='fit'; inner=data['role']=='inner'
  raw=data['base'][:,[0,1,2,3,22,23,24,25,26]]
  if not p.exists():
   seed_everything(SEEDS[0]); mean=raw[fit].mean(0); scale=raw[fit].std(0); scale[scale<1e-6]=1
   x=torch.tensor((raw-mean)/scale,dtype=torch.float32,device='cuda'); y=torch.tensor(data['y'],device='cuda'); mask=torch.tensor(data['mask'],device='cuda')
   m=t.model(9); opt=torch.optim.Adam(m.parameters(),lr=.001); gen=torch.Generator().manual_seed(SEEDS[0]); best=float('inf'); history=[]
   for epoch in range(20):
    loss=t.epoch(m,opt,x[fit],y[fit],mask[fit],gen); m.eval()
    with torch.no_grad(): v=m(x[inner]).cpu().numpy()
    day=data['mask'][inner]&(data['elevation'][inner]>5)
    score=float(np.mean([abs(v[:,h]-data['y'][inner,h])[day[:,h]].mean() for h in [1,2,3]]))
    history.append({'epoch':epoch+1,'loss':loss,'inner_score':score})
    if score<best:
     best=score; atomic_torch(p,{'model':copy.deepcopy(m.state_dict()),'input_dim':9,'mean':mean,'scale':scale,'inner_score':best})
   atomic_json(OUT/'power_only-history.json',history)
  result={'seeds':SEEDS,'roles':{},'shuffle':{},'final_labels_read':False,'network_requests':0}
  pairs=[]
  for seed,folder in runs.items():
   with np.load(folder/'power_solar-predictions.npz') as a, np.load(folder/'satellite_stats-predictions.npz') as b:
    for z in [a,b]:
     assert np.array_equal(z['site'],data['site']) and np.array_equal(z['time'],data['time']) and np.array_equal(z['target'],data['y']) and np.array_equal(z['mask'],data['mask'])
    pairs.append((a['prediction'].copy(),b['prediction'].copy()))
  for role in ['outer_development','spring_stress_development']:
   take=data['role']==role; masks=data['mask'][take][:,[1,2,3]]&(data['elevation'][take][:,[1,2,3]]>5)
   diffs=[]; seed_stats=[]
   for seed,(base,sat) in zip(SEEDS,pairs):
    a=abs(base[take][:,[1,2,3]]-data['y'][take][:,[1,2,3]]); b=abs(sat[take][:,[1,2,3]]-data['y'][take][:,[1,2,3]])
    diffs.append(b-a); seed_stats.append({'seed':seed,'power_mae':float((a*masks).sum(0).__truediv__(masks.sum(0)).mean()),'satellite_mae':float((b*masks).sum(0).__truediv__(masks.sum(0)).mean())})
   result['roles'][role]={'seeds':seed_stats,**blocked_ci(np.mean(diffs,axis=0),masks,np.array([str(x)[:10] for x in data['time'][take]]))}
   shuffle=[]; indices=np.flatnonzero(take)
   for seed,folder in runs.items():
    rng=np.random.default_rng(seed+91); scores=[]
    for repeat in range(20):
     perm=np.arange(len(indices))
     # Same site, same role, different UTC date. Preserve each satellite vector.
     clocks=np.array([str(x)[11:16] for x in data['time'][take]])
     labels=np.char.add(np.char.add(data['site'][take],':'),clocks)
     for label in np.unique(labels):
      ix=np.flatnonzero(labels==label)
      if len(ix)>1: perm[ix]=np.roll(ix,int(rng.integers(1,len(ix))))
     assert all(clocks[i]==clocks[j] for i,j in enumerate(perm))
     v=predict(folder/'satellite_stats-best.pt',np.concatenate([data['base'][take],data['sat'][take][perm]],axis=1))
     err=abs(v[:,[1,2,3]]-data['y'][take][:,[1,2,3]]); scores.append(float(((err*masks).sum(0)/masks.sum(0)).mean()))
    shuffle.append({'seed':seed,'repeats':20,'same_clock_and_site':True,'unchanged_singletons':int((perm==np.arange(len(perm))).sum()),'mae_mean':float(np.mean(scores)),'mae_min':min(scores),'mae_max':max(scores),'interpretation':'input permutation sensitivity, not a causal or formal permutation significance test'})
   result['shuffle'][role]=shuffle
  # Few independent dates and known spring baseline weakness: retain experimental status.
  result['satellite_default_enabled']=False
  result['promotion_reason']='Only 6 winter and 5 spring development dates; no independent final test; spring baseline weakness. Satellite stays opt-in regardless of exploratory CI.'
  atomic_json(OUT/'validation.json',result)
  atomic_json(OUT/'status.json',{'stage':'validated','gpu':torch.cuda.get_device_name(0),'seeds':SEEDS,'final_labels_read':False,'network_requests':0})
  print(json.dumps(result),flush=True)
if __name__=='__main__': main()
