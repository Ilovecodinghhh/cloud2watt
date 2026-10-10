"""Run an explicit request JSON, or export one deterministic daylight replay example."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from cloud2watt.mvp import LocalForecastMVP,render_explanation
from cloud2watt.run_state import atomic_json
from cloud2watt.training import SITE_COLUMNS
ROOT=Path('D:/code/AI_atm'); OLD=ROOT/'outputs/project-v2/local-gpu-20261008'; OUT=ROOT/'outputs/project-v2/mvp-v2'

def example_request():
    for path in sorted((OLD/'features').glob('*.npz')):
        with np.load(path) as z:
            eligible=np.flatnonzero((z['role']=='outer_development')&(z['elevation'][:,0]>10))
            if not len(eligible): continue
            i=int(eligible[0]); issue=pd.Timestamp(str(z['time'][i])); base=z['base'][i]
            return {'issue_time':issue.isoformat(),'site':{'ss_id':str(z['site'][i]),**dict(zip(SITE_COLUMNS,map(float,base[22:])))},
                    'history_times':[(issue+pd.Timedelta(minutes=m)).isoformat() for m in [-45,-30,-15,0]],
                    'history_power':base[:4].tolist(),'history_valid':[True]*4,'satellite_features':z['sat'][i].tolist(),
                    'satellite_times':[(issue+pd.Timedelta(minutes=m)).isoformat() for m in [-45,-30,-15,0]],
                    'satellite_valid':[True]*4,'enable_satellite':True}
    raise ValueError('no daylight example')

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--request',type=Path); parser.add_argument('--output',type=Path,default=OUT/'example'); args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    q=json.loads(args.request.read_text(encoding='utf-8')) if args.request else example_request()
    engine=LocalForecastMVP(OLD,OUT); r=engine.forecast(**q)
    atomic_json(args.output/'request.json',q); atomic_json(args.output/'forecast.json',r); render_explanation(r,args.output/'explanation.png')
    print(json.dumps({'status':r['status'],'model':r['selected_model'],'output':str(args.output)}))
if __name__=='__main__': main()
