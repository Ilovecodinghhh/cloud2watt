"""Read-only local 2021 satellite audit; no training or network."""
import hashlib,json,time,shutil,warnings
from pathlib import Path
import numpy as np
import pandas as pd
import zarr
from cloud2watt.data.shared_tiles import open_satellite_frames
from cloud2watt.data.solar import build_solar_features
from cloud2watt.run_state import atomic_json,file_hash,run_lock
ROOT=Path('D:/code/AI_atm'); OUT=ROOT/'outputs/project-v2/satellite-quality-20261008'
CHANNELS=['VIS006','IR_016','IR_108']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def metrics(a,prev):
 a=np.asarray(a,dtype=np.float32); ok=np.isfinite(a); n=ok.sum((1,2)); safe=np.where(ok,a,0); denom=np.maximum(n,1)
 mean=safe.sum((1,2),dtype=np.float64)/denom
 std=np.sqrt(np.maximum((safe.astype('float64')**2).sum((1,2))/denom-mean**2,0))
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',RuntimeWarning)
  low=np.nanmin(np.where(ok,a,np.nan),axis=(1,2)); high=np.nanmax(np.where(ok,a,np.nan),axis=(1,2))
  qs=np.nanquantile(np.where(ok,a,np.nan)[:,::8,::8],[.01,.5,.99],axis=(1,2))
  stripe=np.maximum(np.nanmean(abs(np.diff(np.nanmean(np.where(ok,a,np.nan),axis=2),axis=1)),axis=1),np.nanmean(abs(np.diff(np.nanmean(np.where(ok,a,np.nan),axis=1),axis=1)),axis=1))/np.maximum(std,1e-6)
 hashes=[hashlib.sha256(a[c].tobytes()).hexdigest() for c in range(3)]
 rms=np.full(3,np.nan); dup=np.zeros(3,bool)
 if prev is not None:
  valid=ok&np.isfinite(prev); den=valid.sum((1,2)); rms=np.sqrt(np.where(valid,(a-prev)**2,0).sum((1,2),dtype=np.float64)/np.maximum(den,1)); rms[den==0]=np.nan
  dup=np.array([np.array_equal(a[c],prev[c],equal_nan=True) and n[c]>0 for c in range(3)])
 return [dict(finite_count=int(n[c]),pixel_count=a.shape[1]*a.shape[2],minimum=float(low[c]),maximum=float(high[c]),mean=float(mean[c]) if n[c] else np.nan,std=float(std[c]) if n[c] else np.nan,zero_fraction=float((a[c]==0).sum()/denom[c]),negative_fraction=float((a[c]<0).sum()/denom[c]),maximum_fraction=float((a[c]==high[c]).sum()/denom[c]),q01_sample=float(qs[0,c]),median_sample=float(qs[1,c]),q99_sample=float(qs[2,c]),stripe_score=float(stripe[c]),jump_rms=float(rms[c]),adjacent_duplicate=bool(dup[c]),sha256=hashes[c]) for c in range(3)]
def main():
 OUT.mkdir(exist_ok=True); start=time.monotonic()
 with run_lock(OUT):
  assert shutil.disk_usage(ROOT).free/shutil.disk_usage(ROOT).total>.2
  cfg=read(ROOT/'configs/project-v2/local-first.json'); tasks=[(part,ROOT/'data/processed/route-b-development-v1'/part,rec) for part,rec in cfg['completed_snapshot'].items()]
  extra=read(ROOT/'outputs/project-v2/spring-supplement-20261008/completion-progress.json')['completed']; tasks += [(part,ROOT/'data/processed/route-b-spring-supplement-v1'/part,rec) for part,rec in extra.items()]
  tasks.sort(key=lambda x:(x[0].split('/')[1],x[0].split('/')[2])); assert len(tasks)==143
  meta=ROOT/'outputs/project-v2/source-cache-2021/objects/.zmetadata'; m=read(meta)['metadata']; attrs=m['data/.zattrs']
  atomic_json(OUT/'preregistration.json',{'scope':'143 completed 2021 development partitions only; all core frames and all channel pixels for finite/min/max/moments/hashes; quantiles deterministic 8-pixel stride','channels':{c:{k:attrs.get(c+'_'+k) for k in ['units','calibration','standard_name','resolution']} for c in CHANNELS},'source_array':m['data/.zarray'],'metadata_sha256':file_hash(meta),'calibration':'extended fold fit input frame keys only; per channel and solar-elevation band; robust flags are diagnostic, never deletion','threshold_rule':'outside min(q0.5%, median-8*1.4826*MAD), max(q99.5%,median+8*1.4826*MAD)','network_requests':0,'final_labels_read':False,'training':False})
  previous={}; allparts=[]; geometry={}
  for i,(part,d,rec) in enumerate(tasks,1):
   assert file_hash(d/'verified.json')==rec['receipt_sha256']
   z=zarr.open_group(str(d/'satellite_frames.zarr'),mode='r'); frames=open_satellite_frames(d/'satellite_frames.zarr'); times=pd.to_datetime(np.asarray(z['time_ns']),utc=True)
   assert list(z.attrs['channel_names'])==CHANNELS
   day=pd.to_datetime(part.split('/')[1],format='%Y%m%d',utc=True); selected=np.flatnonzero((times>=day)&(times<day+pd.Timedelta(days=1)))
   sites=pd.read_parquet(d/'sites.parquet'); ids=list(z.attrs['site_ids']); groups={}
   for si,origin in enumerate(z.attrs['site_origins']):groups.setdefault(tuple(origin),[]).append(si)
   region=part.split('/')[-1]; geometry[region]={'sites':len(ids),'distinct_crops':len(groups),'groups':{str(k):[ids[x] for x in v] for k,v in groups.items()}}
   solar=build_solar_features(times[selected],sites).set_index(['ss_id','datetime_GMT']).solar_elevation_deg
   existing=pd.read_parquet(d/'satellite_quality.parquet').set_index(['ss_id','datetime_GMT'])
   rows=[]
   for ti in selected:
    timestamp=times[ti]
    for origin,indices in groups.items():
     a=np.asarray(frames[indices[0],int(ti)],dtype=np.float32); key=(region,origin); prev=previous.get(key); adjacent=prev is not None and timestamp-prev[0]==pd.Timedelta(minutes=15)
     stats=metrics(a,prev[1] if adjacent else None); previous[key]=(timestamp,a)
     for si in indices:
      site=str(ids[si]); elevation=float(solar.loc[(site,timestamp)]); old=existing.loc[(site,timestamp)]
      measured_invalid=1-np.isfinite(a).mean(); assert abs(measured_invalid-float(old.invalid_fraction))<1e-8
      for channel,stat in zip(CHANNELS,stats): rows.append(dict(partition=part,site=site,time=timestamp,season='winter' if day.month==1 else 'spring',channel=channel,elevation=elevation,adjacent=adjacent,existing_frame_valid=bool(old.is_valid),**stat))
   table=pd.DataFrame(rows); table.to_parquet(OUT/(part.replace('/','_')+'.parquet'),index=False); allparts.append(table)
   atomic_json(OUT/'status.json',{'stage':'scan','completed':i,'total':len(tasks),'last_partition':part,'elapsed_seconds':time.monotonic()-start}); print('scan',i,part,flush=True)
  f=pd.concat(allparts,ignore_index=True); assert not f.duplicated(['site','time','channel']).any()
  f['daylight']=f.elevation>5; f['solar_band']=pd.cut(f.elevation,[-91,0,5,20,40,91],labels=['night','twilight','low','medium','high']).astype(str)
  with np.load(ROOT/'outputs/project-v2/rolling-spring-20261008/extended/dataset.npz') as z:
   fit=z['role']=='fit'; train_sites=z['site'][fit]; issues=pd.to_datetime(z['time'][fit],utc=True)
  train_keys=set((str(s),int(t.value)+h*60*10**9) for s,t in zip(train_sites,issues) for h in [-45,-30,-15,0])
  f['training_frame']=[(s,int(t.value)) in train_keys for s,t in zip(f.site,f.time)]
  features=['minimum','maximum','mean','std','jump_rms','stripe_score']; thresholds=[]
  for col in features:f[col+'_flag']=False
  for (channel,band),ix in f.groupby(['channel','solar_band']).groups.items():
   subset=f.loc[ix]; fitdata=subset.loc[subset.training_frame]
   for col in features:
    values=fitdata[col].dropna().to_numpy()
    if len(values)<100:continue
    med=float(np.median(values)); mad=float(np.median(abs(values-med))); lo=min(float(np.quantile(values,.005)),med-8*1.4826*mad); hi=max(float(np.quantile(values,.995)),med+8*1.4826*mad)
    f.loc[ix,col+'_flag']=(subset[col]<lo)|(subset[col]>hi)
    thresholds.append({'channel':channel,'solar_band':band,'feature':col,'lower':lo,'upper':hi,'training_frames':len(values)})
  f['any_flag']=f[[x+'_flag' for x in features]].any(axis=1)
  f.to_parquet(OUT/'frame-channel-audit.parquet',index=False)
  atomic_json(OUT/'thresholds.json',thresholds); atomic_json(OUT/'geometry.json',geometry)
  summary=[]
  for (season,channel,daylight),g in f.groupby(['season','channel','daylight']):
   summary.append({'season':season,'channel':channel,'daylight':bool(daylight),'frame_channels':len(g),'nonfinite_pixels':int((g.pixel_count-g.finite_count).sum()),'pixel_count':int(g.pixel_count.sum()),'minimum':float(g.minimum.min()),'maximum':float(g.maximum.max()),'mean_median':float(g['mean'].median()),'mean_p05':float(g['mean'].quantile(.05)),'mean_p95':float(g['mean'].quantile(.95)),'adjacent_pairs':int(g.adjacent.sum()),'adjacent_duplicates':int(g.adjacent_duplicate.sum()),'constant_frames':int(g['std'].eq(0).sum()),'any_flag_rate':float(g.any_flag.mean()),'jump_flag_rate':float(g.jump_rms_flag.mean()),'stripe_flag_rate':float(g.stripe_score_flag.mean())})
  band=[]
  for (season,channel,solar_band),g in f.groupby(['season','channel','solar_band']):band.append({'season':season,'channel':channel,'solar_band':solar_band,'n':len(g),'mean_median':float(g['mean'].median()),'std_median':float(g['std'].median()),'jump_median':float(g.jump_rms.median())})
  suspects=f.loc[f.daylight&(f.any_flag|f.adjacent_duplicate|f['std'].eq(0))].copy(); suspects.to_parquet(OUT/'daylight-suspects.parquet',index=False)
  result={'groups':summary,'matched_solar_bands':band,'partitions':143,'unique_site_frames':len(f)//3,'channel_frames':len(f),'distinct_dates':f.time.dt.date.nunique(),'geometry':geometry,'elapsed_seconds':time.monotonic()-start,'final_labels_read':False,'network_requests':0,'limits':'quantiles spatially subsampled; no unit-based physical cuts imposed; robust tail flags are candidates not proven defects; nearby sites share pixels; no causal attribution of model error'}
  atomic_json(OUT/'summary.json',result); atomic_json(OUT/'status.json',{'stage':'complete','elapsed_seconds':result['elapsed_seconds'],'channel_frames':len(f)}); print('COMPLETE',len(f),flush=True)
if __name__=='__main__':main()
