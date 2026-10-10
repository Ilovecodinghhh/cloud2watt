"""Read-only completion audit of the bounded spring batch."""
import importlib.util,json
from pathlib import Path
import pandas as pd
import psutil
from filelock import FileLock
from cloud2watt.run_state import atomic_json,file_hash
ROOT=Path('D:/code/AI_atm'); STATE=ROOT/'outputs/project-v2/spring-supplement-20261008'; DEST=ROOT/'data/processed/route-b-spring-supplement-v1'
spec=importlib.util.spec_from_file_location('route',ROOT/'scripts/download_route_b.py'); r=importlib.util.module_from_spec(spec); spec.loader.exec_module(r)

def main():
 plan=r.read_json(STATE/'plan.json'); execution=r.read_json(STATE/'execution.json'); progress=r.read_json(STATE/'progress.json'); ledger=r.read_json(STATE/'transfer.json')
 assert progress['status']=='batch_complete_stopped_budget_remainder'
 active=[]
 for p in psutil.process_iter(['pid','name','cmdline']):
  if p.info['name'] and p.info['name'].lower()=='python.exe' and any(str(a).endswith('scripts/supplement_spring.py') for a in (p.info['cmdline'] or [])): active.append(p.info)
 assert not active
 with FileLock(r.STATE/'writer.lock',timeout=0),FileLock(STATE/'writer.lock',timeout=0):
  rows=[]; sites=set()
  for tid in execution['partitions']:
   d=DEST/tid; r.verify_shard(d)
   assert file_hash(d/'verified.json')==progress['completed'][tid]['receipt_sha256']
   p=pd.read_parquet(d/'power_15min.parquet'); s=pd.read_parquet(d/'sample_index.parquet'); sat=pd.read_parquet(d/'satellite_quality.parquet')
   core=p.loc[p.datetime_GMT.ge(pd.Timestamp('2021-04-07T00:00Z'))&p.datetime_GMT.lt(pd.Timestamp('2021-04-08T00:00Z'))]
   sites.update(p.ss_id.astype(str)); assert s.issue_time_utc.ge(pd.Timestamp('2021-04-07T00:00Z')).all() and s.issue_time_utc.lt(pd.Timestamp('2021-04-08T00:00Z')).all()
   rows.append({'partition':tid,'samples':len(s),'core_power_rows':len(core),'core_invalid_power_rows':int((~core.is_valid).sum()),'satellite_invalid_rows_including_history':int((~sat.is_valid).sum()),'bytes':r.store_bytes(d),'receipt_sha256':file_hash(d/'verified.json')})
  expected_sites={str(x) for region in plan['regions'] for x in region['site_ids']}
  assert len(expected_sites)==30 and len(rows)==3
  missing=sorted(expected_sites-sites)
  assert missing==['10169']
  pv=r.read_json(r.STATE/'pv-receipts.json')['5_minutely/year=2021/month=04/data.parquet']
  raw=pd.read_parquet(pv['path'],filters=[('ss_id','==',10169)])
  raw.datetime_GMT=pd.to_datetime(raw.datetime_GMT,utc=True)
  assert not ((raw.datetime_GMT>=pd.Timestamp('2021-04-07T00:00Z'))&(raw.datetime_GMT<pd.Timestamp('2021-04-08T00:00Z'))).any()
  for key in execution['source_keys']:
   expected=ledger['receipts'][key]['sha256'] if key in ledger['receipts'] else plan['objects'][key]['sha256']
   assert file_hash(r.CACHE/'objects'/key)==expected
  assert sum(v['bytes'] for v in ledger['receipts'].values())==117485019
  result={'status':'batch_accepted_and_stopped','dates':['2021-04-07'],'planned_sites':30,'sites_with_power':len(sites),'source_missing_sites':missing,'all_30_sites_paired':False,'partitions':rows,'samples':sum(x['samples'] for x in rows),'new_transfer_bytes':ledger['charged_bytes'],'new_objects':len(ledger['receipts']),'elapsed_seconds':progress['elapsed_seconds'],'remaining_dates':['2021-04-08','2021-04-21'],'remaining_end_exclusive':True,'remaining_partitions':39,'two_week_target_complete':False,'reason':'256 MiB supplement cap; next complete 30-site date does not fit','active_download_processes':active,'writer_locks_free':True,'training_started':False,'final_labels_read':False}
  atomic_json(STATE/'acceptance.json',result); print(json.dumps(result))
if __name__=='__main__': main()
