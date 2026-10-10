"""Independent artifact, split and fitted-checkpoint acceptance."""
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import train_local_gpu as t
from cloud2watt.data.role_quality import within_role
from train_rolling_spring import OUT,FOLDS,SEEDS

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def main():
 assert read(OUT/'status.json')['stage']=='complete'
 assert torch.cuda.is_available() and 'raincalib-gpu' in sys.executable
 torch.set_num_threads(4)
 checked=[]; outer_keys=[]
 for fold,bounds in FOLDS.items():
  folder=OUT/fold
  with np.load(folder/'dataset.npz') as z: d={k:z[k] for k in z.files}
  issues=pd.DatetimeIndex(pd.to_datetime(d['time'],utc=True)); spring=issues.month==4
  for role,dates in bounds.items():
   a,b=[pd.Timestamp(x,tz='UTC') for x in dates]; keep=(d['role']==role)&spring
   assert within_role(issues[keep],a,b).all()
   q=pd.read_parquet(OUT/'qc'/fold/(role+'.parquet')).set_index(['ss_id','datetime_GMT']).is_valid
   for h in [-45,-30,-15,0]:
    keys=pd.MultiIndex.from_arrays([d['site'][keep],issues[keep]+pd.Timedelta(minutes=h)])
    assert q.reindex(keys,fill_value=False).all()
   for col,h in enumerate([15,30,60,120,180,240]):
    keys=pd.MultiIndex.from_arrays([d['site'][keep],issues[keep]+pd.Timedelta(minutes=h)])
    np.testing.assert_array_equal(d['mask'][keep,col],q.reindex(keys,fill_value=False).to_numpy())
  take=d['role']=='spring_outer_development'
  outer_keys.extend(zip(d['site'][take],d['time'][take]))
  fit=d['role']=='fit'
  for seed in SEEDS:
   folder=OUT/fold/str(seed)
   assert read(folder/'preregistration.json')['dataset_sha256']==t.file_hash(OUT/fold/'dataset.npz')
   for kind in ['power_solar','satellite_stats']:
    raw=d['base'] if kind=='power_solar' else np.concatenate([d['base'],d['sat']],axis=1)
    c=torch.load(folder/(kind+'-best.pt'),weights_only=False)
    mean=raw[fit].mean(0); scale=raw[fit].std(0); scale[scale<1e-6]=1
    np.testing.assert_array_equal(c['mean'],mean); np.testing.assert_array_equal(c['scale'],scale)
    h=read(folder/(kind+'-history.json')); assert len(h)==20
    assert c['inner_score']==min(x['inner_daylight_primary_mae'] for x in h)
    with np.load(folder/(kind+'-predictions.npz')) as z:
     for k in ['site','time','role','mask']: np.testing.assert_array_equal(z[k],d[k])
     np.testing.assert_array_equal(z['target'],d['y']); expected=z['prediction'].copy()
    model=t.model(c['input_dim']); model.load_state_dict(c['model']); model.eval()
    ids=np.flatnonzero(take)[::101]
    with torch.no_grad(): pred=model(torch.tensor((raw[ids]-mean)/scale,device='cuda')).cpu().numpy()
    np.testing.assert_allclose(pred,expected[ids],rtol=1e-4,atol=2e-6)
    checked.append({'fold':fold,'seed':seed,'kind':kind,'checkpoint_sha256':t.file_hash(folder/(kind+'-best.pt'))})
 assert len(outer_keys)==len(set(outer_keys))
 result={'passed':True,'verified_checkpoints':checked,'nonoverlapping_outer_samples':len(outer_keys),'fit_only_normalization':True,'inner_only_checkpoint_selection':True,'role_local_qc_and_temporal_support':True,'saved_prediction_replay':True,'final_labels_read':False}
 t.atomic_json(OUT/'independent-acceptance.json',result)
 print(json.dumps({k:v for k,v in result.items() if k!='verified_checkpoints'}))
if __name__=='__main__':main()
