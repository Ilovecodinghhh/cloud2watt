"""Calendar coverage only; no model scoring or source substitution."""
import json,sys
from pathlib import Path
import pandas as pd
from cloud2watt.run_state import atomic_json,file_hash
assert 'raincalib-gpu' in sys.executable
ROOT=Path('D:/code/AI_atm');OUT=ROOT/'outputs/project-v2/spring-validation-20261009'
plan=json.loads((OUT/'plan.json').read_text()); receipts=json.loads((ROOT/'outputs/project-v2/route-b-download/pv-receipts.json').read_text()); sites=[int(s) for r in plan['regions'] for s in r['site_ids']]
frames=[]
for month in ['04','05']:
 rec=receipts[f'5_minutely/year=2021/month={month}/data.parquet'];assert file_hash(Path(rec['path']))==rec['sha256']
 frames.append(pd.read_parquet(rec['path'],filters=[('ss_id','in',sites)]))
data=pd.concat(frames,ignore_index=True);data.datetime_GMT=pd.to_datetime(data.datetime_GMT,utc=True)
records=[]
for day in pd.date_range('2021-04-21','2021-05-05',inclusive='left',tz='UTC'):
 for sid in sites:
  rows=data.loc[data.ss_id.eq(sid)&data.datetime_GMT.ge(day)&data.datetime_GMT.lt(day+pd.Timedelta(days=1))]
  records.append({'date':str(day.date()),'site':str(sid),'rows':len(rows),'unique_times':rows.datetime_GMT.nunique(),'expected_5min_slots':288,'finite_readings':int(rows.generation_Wh.notna().sum())})
result={'records':records,'whole_day_gaps':[x for x in records if x['rows']==0],'incomplete_site_days':sum(x['unique_times']<288 for x in records),'calendar_only':True,'training':False,'model_scoring':False}
atomic_json(OUT/'raw-power-coverage.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='records'}))
