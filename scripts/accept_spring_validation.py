"""Full registered spring completion audit with explicit source missingness."""
import json,importlib.util
from pathlib import Path
import numpy as np
import zarr
import pandas as pd
import psutil
from filelock import FileLock
from cloud2watt.run_state import atomic_json,file_hash
ROOT=Path('D:/code/AI_atm'); STATE=ROOT/'outputs/project-v2/spring-validation-20261009'; DEST=ROOT/'data/processed/route-b-spring-validation-v1'
spec=importlib.util.spec_from_file_location('route',ROOT/'scripts/download_route_b.py'); r=importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
def main():
 progress=r.read_json(STATE/'completion-progress.json'); assert progress['status']=='complete_stopped'
 plan=r.read_json(STATE/'plan.json'); ledger=r.read_json(STATE/'transfer.json'); expected={str(x) for region in plan['regions'] for x in region['site_ids']}
 active=[]
 for p in psutil.process_iter(['pid','name','cmdline']):
  if p.info['name'] and p.info['name'].lower()=='python.exe' and any(str(a).endswith('scripts/download_spring_validation.py') for a in (p.info['cmdline'] or [])): active.append(p.info)
 assert not active
 with FileLock(r.STATE/'writer.lock',timeout=0),FileLock(STATE/'writer.lock',timeout=0):
  records=[]; absent=[]; keys=[]; sites_seen=set()
  for tid in sorted(plan['partition_objects']):
   d=DEST/tid; r.verify_shard(d); assert file_hash(d/'verified.json')==progress['completed'][tid]['receipt_sha256']
   p=pd.read_parquet(d/'power_15min.parquet'); s=pd.read_parquet(d/'sample_index.parquet'); sat=pd.read_parquet(d/'satellite_quality.parquet'); meta=pd.read_parquet(d/'sites.parquet')
   day=pd.to_datetime(tid.split('/')[1],format='%Y%m%d',utc=True); end=day+pd.Timedelta(days=1)
   assert s.issue_time_utc.ge(day).all() and s.issue_time_utc.lt(end).all()
   hist=np.stack(s.power_history_row_indices); target=np.stack(s.target_row_indices); present=target>=0; safe=np.maximum(target,0); mask=np.stack(s.target_mask).astype(bool)
   assert hist.min()>=0 and hist.max()<len(p) and safe.max()<len(p)
   site=p.ss_id.astype(str).to_numpy(); valid=p.is_valid.to_numpy(); times=p.datetime_GMT.dt.as_unit('ns').astype('int64').to_numpy(); issue=s.issue_time_utc.dt.as_unit('ns').astype('int64').to_numpy(); ids=s.site_id.astype(str).to_numpy()
   assert (site[hist]==ids[:,None]).all() and valid[hist].all()
   assert (times[hist]==issue[:,None]+np.array([-45,-30,-15,0])*60*10**9).all()
   assert (site[safe][present]==np.broadcast_to(ids[:,None],safe.shape)[present]).all()
   assert (times[safe][present]==(issue[:,None]+np.array([15,30,60,120,180,240])*60*10**9)[present]).all()
   assert np.array_equal(mask,present&valid[safe])
   z=zarr.open_group(d/'satellite_frames.zarr',mode='r'); frame=np.stack(s.satellite_frame_indices)
   assert np.array_equal(np.asarray(z['time_ns'])[frame],issue[:,None]+np.array([-45,-30,-15,0])*60*10**9)
   assert list(z.attrs['channel_names'])==['VIS006','IR_016','IR_108']
   assert np.array_equal(np.asarray(z.attrs['site_ids'])[s.site_index.to_numpy()],ids)
   sat_lookup=sat.set_index(['ss_id','datetime_GMT']).is_valid
   for offset in [-45,-30,-15,0]:
    index=pd.MultiIndex.from_arrays([ids,pd.to_datetime(issue+offset*60*10**9,utc=True)])
    assert sat_lookup.reindex(index).fillna(False).all()
   assert r.read_json(d/'manifest.json')['data_role']=='prospective_spring_validation'
   core=p.loc[p.datetime_GMT.ge(day)&p.datetime_GMT.lt(end)]; sites_seen.update(core.ss_id.astype(str)); missing=set(meta.ss_id.astype(str))-set(core.ss_id.astype(str))
   absent.extend({'date':str(day.date()),'site':sid} for sid in sorted(missing))
   records.append({'partition':tid,'samples':len(s),'planned_sites':len(meta),'sites_with_core_power':core.ss_id.nunique(),'core_power_rows':len(core),'expected_core_power_rows':len(meta)*96,'missing_core_power_rows':len(meta)*96-len(core),'valid_core_power_rows':int(core.is_valid.sum()),'invalid_core_power_rows':int((~core.is_valid).sum()),'satellite_invalid_with_context':int((~sat.is_valid).sum()),'bytes':r.store_bytes(d)})
   keys.extend(zip(ids,s.issue_time_utc.astype(str)))
  assert len(keys)==len(set(keys)) and len(records)==42
  for key,info in plan['objects'].items():
   expected_hash=ledger['receipts'].get(key,info)['sha256']; assert file_hash(r.CACHE/'objects'/key)==expected_hash
  # Verify whole-day gaps against the already-local original development PV file.
  frames=[]
  for month in ['04','05']:
   pv=r.read_json(r.STATE/'pv-receipts.json')[f'5_minutely/year=2021/month={month}/data.parquet']; assert file_hash(Path(pv['path']))==pv['sha256']
   frames.append(pd.read_parquet(pv['path'],filters=[('ss_id','in',list(map(int,expected)))]))
  raw=pd.concat(frames,ignore_index=True); raw.datetime_GMT=pd.to_datetime(raw.datetime_GMT,utc=True)
  for item in absent:
   day=pd.Timestamp(item['date'],tz='UTC'); n=int((raw.ss_id.eq(int(item['site']))&raw.datetime_GMT.ge(day)&raw.datetime_GMT.lt(day+pd.Timedelta(days=1))).sum()); assert n==0; item['raw_rows']=n
  new=records
  result={'status':'authorized_objects_complete_and_stopped','new_dates':['2021-04-21','2021-05-05'],'end_exclusive':True,'new_partitions':len(new),'new_samples':sum(x['samples'] for x in new),'all_supplement_partitions':42,'all_supplement_samples':sum(x['samples'] for x in records),'planned_sites':30,'sites_observed_any_day':len(sites_seen),'whole_day_source_gaps':absent,'partitions':records,'new_payload_bytes_this_run':sum(v['bytes'] for v in ledger['receipts'].values()),'total_supplement_payload_bytes':sum(v['bytes'] for v in ledger['receipts'].values()),'total_supplement_charged_bytes':ledger['charged_bytes'],'elapsed_seconds':progress['elapsed_seconds'],'source_objects_verified':len(plan['objects']),'all_30_sites_complete_every_day':not bool(absent),'training_started':False,'final_labels_read':False,'active_download_processes':active,'writer_locks_free':True}
  result['expected_core_power_rows']=sum(x['expected_core_power_rows'] for x in records)
  result['valid_core_power_rows']=sum(x['valid_core_power_rows'] for x in records)
  result['missing_core_power_rows']=sum(x['missing_core_power_rows'] for x in records)
  result['invalid_core_power_rows']=sum(x['invalid_core_power_rows'] for x in records)
  result['qc_note']='Offline contextual QC; prospective scores require frozen sample-role boundary handling. Candidate samples are not final evaluation counts.'
  result['pv_new_payload_bytes']=r.read_json(STATE/'pv-transfer.json')['actual_bytes']
  result['evaluation_freeze_sha256']=file_hash(STATE/'evaluation-freeze.json')
  for name,digest in r.read_json(STATE/'evaluation-freeze.json')['model_hashes'].items(): assert file_hash(ROOT/name)==digest
  result['persistent_bytes_after']=r.Budget(r.read_json(r.FREEZE)).usage()+r.store_bytes(ROOT/'data/processed/route-b-spring-supplement-v1')+r.store_bytes(DEST)
  assert result['persistent_bytes_after']<=16*1024**3
  atomic_json(STATE/'completion-acceptance.json',result); print(json.dumps({k:v for k,v in result.items() if k!='partitions'}))
if __name__=='__main__': main()
