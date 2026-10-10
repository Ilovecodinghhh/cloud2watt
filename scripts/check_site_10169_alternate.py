"""Check same-provider half-hour source in isolation, never merge resolutions."""
import json,importlib.util
from pathlib import Path
from urllib.request import Request,urlopen
from filelock import FileLock
import pandas as pd
from huggingface_hub import hf_hub_url,get_hf_file_metadata
from cloud2watt.run_state import atomic_json,file_hash
ROOT=Path('D:/code/AI_atm'); OUT=ROOT/'outputs/project-v2/site-10169-recheck'; LIMIT=16*1024**2
spec=importlib.util.spec_from_file_location('route',ROOT/'scripts/download_route_b.py'); r=importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
result=r.read_json(OUT/'result.json'); revision=result['upstream_revision']; name='30_minutely/year=2021/month=04/data.parquet'
remote=get_hf_file_metadata(hf_hub_url('openclimatefix/uk_pv',name,repo_type='dataset',revision=revision),timeout=30)
plan={'object':name,'revision':revision,'expected_bytes':remote.size,'expected_sha256':remote.etag.strip('"'),'purpose':'station 10169 April7-20 missingness retry; isolated alternative resolution','response_cap_bytes':LIMIT,'maximum_attempts':1,'merge_into_training':False,'final_labels_read':False}
atomic_json(OUT/'alternate-plan.json',plan)
assert remote.size < 64*1024**2
with FileLock(r.STATE/'writer.lock',timeout=0):
 path=OUT/'alternate-30min-april.parquet'; ledger_path=OUT/'alternate-transfer.json'
 if not path.exists():
  if ledger_path.exists(): raise ValueError('attempt already registered; do not reset')
  budget=r.Budget(r.read_json(r.FREEZE)); budget.reserve_pv(remote.size)
  atomic_json(ledger_path,{'attempts':1,'reserved_project_bytes':remote.size*4,'status':'attempt_registered'})
  pieces=[]
  for start in range(0,remote.size,8*1024**2):
   end=min(start+8*1024**2,remote.size)-1
   with urlopen(Request(remote.location,headers={'Range':f'bytes={start}-{end}'}),timeout=60) as response:
    assert response.status==206 and response.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/')
    part=response.read(LIMIT+1)
   assert len(part)==end-start+1 and len(part)<=LIMIT
   pieces.append(part)
  payload=b''.join(pieces)
  assert len(payload)==remote.size
  temp=path.with_suffix('.part'); temp.write_bytes(payload)
  assert file_hash(temp)==plan['expected_sha256']; temp.replace(path)
  atomic_json(ledger_path,{'attempts':1,'reserved_project_bytes':remote.size*4,'actual_bytes':len(payload),'sha256':file_hash(path),'status':'verified'})
 else: assert file_hash(path)==plan['expected_sha256']
 frame=pd.read_parquet(path); station=frame.loc[frame.ss_id.astype(str).eq('10169')].copy(); station.datetime_GMT=pd.to_datetime(station.datetime_GMT,utc=True)
 take=station.loc[station.datetime_GMT.ge('2021-04-07T00:00Z')&station.datetime_GMT.lt('2021-04-21T00:00Z')]
 info={'april_station_rows':len(station),'requested_window_rows':len(take),'dates':station.groupby(station.datetime_GMT.dt.strftime('%Y-%m-%d')).size().to_dict(),'object':name,'bytes':remote.size,'sha256':file_hash(path),'merged_into_training':False}
 result['alternate_30min']=info; result['status']='no_requested_records_in_either_resolution' if take.empty else 'alternative_resolution_found_requires_separate_protocol'; result['network_payload_downloaded_bytes']=remote.size
 atomic_json(OUT/'result.json',result); print(json.dumps(info))
