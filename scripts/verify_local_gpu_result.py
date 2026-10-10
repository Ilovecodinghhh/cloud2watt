import json
from pathlib import Path
import torch,numpy as np,pandas as pd
from cloud2watt.run_state import atomic_json,file_hash
p=Path('D:/code/AI_atm/outputs/project-v2/local-gpu-20261008')
a=torch.load(p/'e02-continuous.pt',weights_only=False); b=torch.load(p/'e02-resume.pt',weights_only=False)
def equal(a,b):
 if isinstance(a,torch.Tensor): return torch.equal(a,b)
 if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
assert equal(a,b), 'cross-process full state mismatch'
print('Full model, optimizer and RNG states identical')
if (p/'metrics.json').exists() and json.loads((p/'status.json').read_text())['stage']=='complete':
 metrics=json.loads((p/'metrics.json').read_text()); names=list(metrics)
 for role in ['inner','outer_development','spring_stress_development']:
  assert all(metrics[n][role]['comparison']==metrics[names[0]][role]['comparison'] for n in names)
 index=pd.read_parquet('D:/code/AI_atm/outputs/project-v2/local-provenance-20261007/safe-split-index.parquet')
 with np.load(p/'power_solar-predictions.npz') as pred:
  actual=pd.DataFrame({'site_id':pred['site'],'issue_time_utc':pd.to_datetime(pred['time'],utc=True),'role':pred['role']})
 expected=index[['site_id','issue_time_utc','role']].copy(); expected.site_id=expected.site_id.astype(str)
 assert len(actual)==len(expected)==76199
 assert set(map(tuple,actual.to_numpy()))==set(map(tuple,expected.to_numpy()))
 atomic_json(p/'acceptance.json',{'full_recovery_state_equal':True,'all_models_same_keys_targets_masks_daylight':True,'safe_manifest_exact_match':True,'samples':76199,'preregistration_sha256':file_hash(p/'preregistration.json'),'files':{f.name:file_hash(f) for f in p.glob('*') if f.is_file() and f.suffix in ['.pt','.npz']}})
 print('All model comparison identities and frozen sample keys match')
