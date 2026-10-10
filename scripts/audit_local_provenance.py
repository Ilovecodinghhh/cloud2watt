"""Offline provenance and conservative split audit; never opens final-test data."""
import json, hashlib, importlib.util
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
ROOT=Path('D:/code/AI_atm')
OUT=ROOT/'outputs/project-v2/local-provenance-20261007'
OUT.mkdir(exist_ok=True)
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def save(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
cfg=read(ROOT/'configs/project-v2/local-first.json')
roles={k:tuple(map(pd.Timestamp,v)) for k,v in cfg['split'].items() if isinstance(v,list)}
spec=importlib.util.spec_from_file_location('runner',ROOT/'scripts/download_route_b.py'); runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
errors=[]; cores=[]; sats=[]; sample_parts=[]; region_sites={}; split_counts={}; manifests=[]; checks=Counter()
for tid,receipt in cfg['completed_snapshot'].items():
 d=runner.DEST/tid
 assert digest(d/'verified.json')==receipt['receipt_sha256'],tid
 runner.verify_shard(d)
 reg=read(d/'preregistration.json'); region=reg['region']; region_sites[region]=[int(x) for x in reg['site_ids']]
 p=pd.read_parquet(d/'power_15min.parquet'); s=pd.read_parquet(d/'sample_index.parquet'); q=pd.read_parquet(d/'satellite_quality.parquet')
 lo=pd.to_datetime(tid.split('/')[1],format='%Y%m%d',utc=True); hi=lo+pd.Timedelta(days=1)
 for table,collection in [(p,cores),(q,sats)]:
  core=table.loc[table.datetime_GMT.ge(lo)&table.datetime_GMT.lt(hi)].copy(); core['region']=region; core['partition']=tid; collection.append(core)
 assert len(s)==receipt['samples']
 times=p.datetime_GMT.dt.as_unit('ns').astype('int64').to_numpy(); sites=p.ss_id.astype(str).to_numpy(); valid=p.is_valid.to_numpy()
 issue=s.issue_time_utc.dt.as_unit('ns').astype('int64').to_numpy(); site=s.site_id.astype(str).to_numpy()
 hist=np.stack(s.power_history_row_indices); target=np.stack(s.target_row_indices); mask=np.stack(s.target_mask).astype(bool)
 present=target>=0; safe=np.maximum(target,0)
 assert hist.min()>=0 and hist.max()<len(p) and safe.max()<len(p)
 assert (sites[hist]==site[:,None]).all() and (times[hist]==issue[:,None]+np.array([-45,-30,-15,0])*60*10**9).all()
 assert valid[hist].all()
 assert (sites[safe][present]==np.broadcast_to(site[:,None],safe.shape)[present]).all()
 assert (times[safe][present]==(issue[:,None]+np.array([15,30,60,120,180,240])*60*10**9)[present]).all()
 assert np.array_equal(mask,present&valid[safe])
 import zarr
 z=zarr.open_group(str(d/'satellite_frames.zarr'),mode='r'); ft=np.asarray(z['time_ns']); fi=np.stack(s.satellite_frame_indices)
 assert fi.min()>=0 and fi.max()<len(ft)
 assert (ft[fi]==issue[:,None]+np.array([-45,-30,-15,0])*60*10**9).all()
 checks['samples']+=len(s); checks['partitions']+=1; checks['masked_missing_targets']+=int((~present).sum())
 s=s.copy(); s['partition']=tid; s['sample_row']=np.arange(len(s)); s['role']='excluded'
 supportlo=pd.Timestamp(reg['support_start_inclusive']); supporthi=pd.Timestamp(reg['support_end_exclusive'])
 for role,(a,b) in roles.items():
  candidate=s.issue_time_utc.ge(a)&s.issue_time_utc.lt(b)
  # Keep original QC unchanged only if its entire source window belongs to the role.
  support_ok=supportlo>=a and supporthi<=b
  temporal=s.issue_time_utc.sub(pd.Timedelta(hours=1)).ge(a)&s.issue_time_utc.add(pd.Timedelta(hours=4)).lt(b)
  keep=candidate&temporal&support_ok
  stat=split_counts.setdefault(role,Counter()); stat['candidate']+=int(candidate.sum()); stat['history_or_target_cross_boundary']+=int((candidate&~temporal).sum()); stat['retained']+=int(keep.sum()); stat['excluded_conservative_qc_support']+=int((candidate&temporal&~pd.Series(support_ok,index=s.index)).sum())
  s.loc[keep,'role']=role
  manifests.extend(s.loc[keep,['partition','sample_row','site_id','issue_time_utc','role']].to_dict('records'))
 sample_parts.append(s[['partition','site_id','issue_time_utc','role']])
 print('verified',tid,flush=True)
power=pd.concat(cores,ignore_index=True); satellite=pd.concat(sats,ignore_index=True); allsamples=pd.concat(sample_parts,ignore_index=True)
assert not allsamples.duplicated(['site_id','issue_time_utc']).any()
power.ss_id=power.ss_id.astype(int)
receipts=read(runner.STATE/'pv-receipts.json'); rawparts=[]
for key,v in receipts.items():
 assert v['role']=='development' and ('year=2021/month=01/' in key or 'year=2021/month=04/' in key)
 path=Path(v['path']); assert digest(path)==v['sha256']
 raw=pd.read_parquet(path,filters=[('ss_id','in',sum(region_sites.values(),[]))],columns=['ss_id','datetime_GMT','generation_Wh'])
 raw.datetime_GMT=pd.to_datetime(raw.datetime_GMT,utc=True); rawparts.append(raw)
raw=pd.concat(rawparts,ignore_index=True)
raw_duplicates=int(raw.duplicated(['ss_id','datetime_GMT'],keep=False).sum())
raw=raw.groupby(['ss_id','datetime_GMT'],as_index=False).generation_Wh.mean()
raw['bin']=raw.datetime_GMT.dt.ceil('15min')
agg=raw.groupby(['ss_id','bin']).generation_Wh.agg(raw_count='count',raw_sum='sum',raw_min='min',raw_max='max').reset_index().rename(columns={'bin':'datetime_GMT'})
grids=[]
for window,start,end in [('winter_fit','2021-01-01','2021-01-29'),('spring','2021-04-01','2021-04-07')]:
 for region,ids in region_sites.items():
  g=pd.MultiIndex.from_product([ids,pd.date_range(start,end,freq='15min',inclusive='left',tz='UTC')],names=['ss_id','datetime_GMT']).to_frame(index=False)
  g['region']=region; g['partition']=window+'/'+g.datetime_GMT.dt.strftime('%Y%m%d')+'/'+region
  g['partition_present']=g.partition.isin(cfg['completed_snapshot']); grids.append(g)
grid=pd.concat(grids,ignore_index=True).merge(power.drop(columns=['region','partition']),on=['ss_id','datetime_GMT'],how='left',indicator='derived').merge(agg,on=['ss_id','datetime_GMT'],how='left')
grid.raw_count=grid.raw_count.fillna(0).astype(int)
grid['cause']='stored_valid'
grid.loc[grid.is_valid.eq(False),'cause']='stored_other_qc'
grid.loc[(grid.quality_flags.fillna(0).astype(int)&64)!=0,'cause']='source_daytime_zero'
grid.loc[grid.raw_count.lt(3),'cause']='source_fewer_than_3_readings'
grid.loc[grid.derived.eq('left_only')&grid.raw_count.ge(3),'cause']='derived_absent_raw_complete'
grid.loc[~grid.partition_present,'cause']='partition_not_built'
# Leading season bins can be intentionally trimmed by the builder.
grid.loc[grid.cause.eq('derived_absent_raw_complete')&grid.datetime_GMT.isin(pd.to_datetime(['2021-01-01T00:00Z','2021-01-01T00:15Z','2021-04-01T00:00Z','2021-04-01T00:15Z'])),'cause']='season_boundary_trim'
stored=grid.derived.eq('both'); complete=stored&grid.raw_count.eq(3)&grid.reading_count.eq(3)
checks['raw_count_mismatches']=int((stored&grid.reading_count.fillna(0).ne(grid.raw_count)).sum())
checks['raw_aggregation_mismatches']=int((~np.isclose(grid.loc[complete,'generation_Wh'],grid.loc[complete,'raw_sum'],rtol=1e-6,atol=1e-8)).sum())
checks['stuck_zero_not_supported_by_raw']=int((grid.cause.eq('source_daytime_zero')&(grid.raw_min.ne(0)|grid.raw_max.ne(0))).sum())
zero_runs=[]
for site,g in grid.loc[grid.ss_id.isin([6801,7408,6075,26909])].sort_values(['ss_id','datetime_GMT']).groupby('ss_id'):
 m=(g.quality_flags.fillna(0).astype(int)&64)!=0
 runs=(m.ne(m.shift())|g.datetime_GMT.diff().ne(pd.Timedelta(minutes=15))).cumsum()
 for _,x in g.loc[m].groupby(runs[m]):
  zero_runs.append({'site':int(site),'start':str(x.datetime_GMT.min()),'end':str(x.datetime_GMT.max()),'bins':len(x),'raw_all_zero':bool((x.raw_min.eq(0)&x.raw_max.eq(0)).all())})
manifest=pd.DataFrame(manifests); manifest.to_parquet(OUT/'safe-split-index.parquet',index=False)
grid.to_parquet(OUT/'calendar-provenance.parquet',index=False)
save('zero-runs.json',zero_runs)
summary={'timestamp_utc':str(pd.Timestamp.now(tz='UTC')),'checks':dict(checks),'raw_duplicate_rows':raw_duplicates,'calendar_bins':len(grid),'causes':grid.cause.value_counts().to_dict(),'regions':{r:g.cause.value_counts().to_dict() for r,g in grid.groupby('region')},'focus_sites':{str(s):g.cause.value_counts().to_dict() for s,g in grid.loc[grid.ss_id.isin([6801,7408,6075,26909])].groupby('ss_id')},'splits':{r:dict(c) for r,c in split_counts.items()},'split_policy':'Keep only samples whose entire original partition QC source support belongs to the role; no boundary QC reused across roles. Conservative exclusion, not QC recomputation.','satellite_quality_flags':{str(k):int(v) for k,v in satellite.quality_flags.value_counts().items()},'unexplained_absent':grid.loc[grid.cause.eq('derived_absent_raw_complete'),['ss_id','datetime_GMT','partition']].head(30).to_dict('records'),'final_labels_read':False,'network_requests':0,'formal_training_ready':False,'manifest_sha256':digest(OUT/'safe-split-index.parquet')}
save('summary.json',summary)
print(json.dumps(summary,default=str),flush=True)
