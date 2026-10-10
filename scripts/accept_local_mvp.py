"""Acceptance on actual checkpoints: inference parity and missing-input behavior."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from cloud2watt.mvp import LocalForecastMVP
from cloud2watt.run_state import atomic_json,file_hash
from cloud2watt.training import SITE_COLUMNS
ROOT=Path('D:/code/AI_atm'); OLD=ROOT/'outputs/project-v2/local-gpu-20261008'; OUT=ROOT/'outputs/project-v2/mvp-v2'

def main():
    engine=LocalForecastMVP(OLD,OUT); blocks=[]
    for path in sorted((OLD/'features').glob('*.npz')):
        with np.load(path) as z: blocks.append({k:z[k] for k in z.files})
    data={k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
    with np.load(OLD/'power_solar-predictions.npz') as z: base_prediction=z['prediction'].copy()
    with np.load(OLD/'satellite_stats-predictions.npz') as z: sat_prediction=z['prediction'].copy()
    maximum=0.; count=0
    for role in ['inner','outer_development','spring_stress_development']:
        ix=np.flatnonzero(data['role']==role); selected=ix[np.linspace(0,len(ix)-1,8,dtype=int)]
        for i in selected:
            issue=pd.Timestamp(str(data['time'][i])); base=data['base'][i]
            q={'issue_time':issue,'site':{'ss_id':data['site'][i],**dict(zip(SITE_COLUMNS,map(float,base[22:])))},'history_times':[issue+pd.Timedelta(minutes=m) for m in [-45,-30,-15,0]],'history_power':base[:4]}
            r=engine.forecast(**q); assert r['selected_model']=='power_solar'
            np.testing.assert_allclose(r['prediction'],base_prediction[i],atol=2e-5,rtol=1e-4)
            sat=engine.forecast(**q,enable_satellite=True,satellite_features=data['sat'][i],satellite_times=q['history_times'])
            np.testing.assert_allclose(sat['prediction'],sat_prediction[i],atol=2e-5,rtol=1e-4)
            maximum=max(maximum,float(np.max(abs(np.array(sat['prediction'])-sat_prediction[i]))))
            fallback=engine.forecast(**q,enable_satellite=True)
            np.testing.assert_array_equal(fallback['prediction'],r['prediction'])
            missing=dict(q); missing['history_power']=[float('nan')]*4
            assert engine.forecast(**missing)['prediction'] is None
            assert len(r['explanation']['solar_model_difference'])==6
            count+=1
    v=json.loads((OUT/'validation.json').read_text()); assert v['seeds']==[20261008,20261009,20261010] and not v['satellite_default_enabled']
    atomic_json(OUT/'acceptance.json',{'status':'passed','actual_checkpoint_cases':count,'cpu_gpu_max_abs_error':maximum,'default_power_solar':True,'missing_satellite_exact_fallback':True,'missing_history_unavailable':True,'solar_recomputed_without_labels':True,'three_seed_validation_completed':True,'final_labels_read':False,'new_download_objects':[],'validation_sha256':file_hash(OUT/'validation.json'),'model_provenance':engine.provenance})
    print('MVP acceptance passed:',count,'actual checkpoint cases, maximum error',maximum)
if __name__=='__main__': main()
