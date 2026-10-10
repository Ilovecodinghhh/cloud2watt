"""Follow up fixed audit flags and existing rolling predictions; never retrain."""
import json
from pathlib import Path
import numpy as np,pandas as pd,zarr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cloud2watt.data.shared_tiles import open_satellite_frames
from cloud2watt.run_state import atomic_json
ROOT=Path('D:/code/AI_atm'); OUT=ROOT/'outputs/project-v2/satellite-quality-20261008'; ROLL=ROOT/'outputs/project-v2/rolling-spring-20261008'
f=pd.read_parquet(OUT/'frame-channel-audit.parquet')
def fold_flags(folder):
 with np.load(folder/'dataset.npz') as z:
  fit=z['role']=='fit'; sites=z['site'][fit]; issues=pd.to_datetime(z['time'][fit],utc=True)
 keys=set((str(s),int(t.value)+h*60*10**9) for s,t in zip(sites,issues) for h in [-45,-30,-15,0])
 training=np.array([(s,int(t.value)) in keys for s,t in zip(f.site,f.time)])
 flag=pd.Series(False,index=f.index); supported=pd.Series(False,index=f.index); thresholds=[]
 for (channel,band),ix in f.groupby(['channel','solar_band']).groups.items():
  g=f.loc[ix]; train=g.loc[training[ix]]
  if len(train)<100:continue
  supported.loc[ix]=True
  for col in ['minimum','maximum','mean','std','jump_rms','stripe_score']:
   values=train[col].dropna().to_numpy()
   if len(values)<100:continue
   median=float(np.median(values)); mad=float(np.median(abs(values-median)))
   lo=min(float(np.quantile(values,.005)),median-8*1.4826*mad); hi=max(float(np.quantile(values,.995)),median+8*1.4826*mad)
   flag.loc[ix]|=(g[col]<lo)|(g[col]>hi)
   thresholds.append({'channel':channel,'band':band,'feature':col,'lo':lo,'hi':hi,'n':len(values)})
 atomic_json(OUT/(folder.name+'-thresholds.json'),thresholds)
 temp=f[['site','time']].copy(); temp['flag']=flag; temp['supported']=supported
 return temp.groupby(['site','time']).agg(flag=('flag','any'),supported=('supported','all'))
association={}
for fold in ['early','middle','extended']:
 evidence=fold_flags(ROLL/fold); flags=evidence.flag
 with np.load(ROLL/fold/'dataset.npz') as z:
  take=z['role']=='spring_outer_development'; sites=z['site'][take]; issues=pd.to_datetime(z['time'][take],utc=True); target=z['y'][take][:,[1,2,3]]; day=z['mask'][take][:,[1,2,3]]&(z['elevation'][take][:,[1,2,3]]>5)
  flagged=np.zeros(len(sites),bool); supported=np.ones(len(sites),bool)
  for h in [-45,-30,-15,0]:
   lookup=flags.reindex(pd.MultiIndex.from_arrays([sites,issues+pd.Timedelta(minutes=h)])); assert lookup.notna().all(); flagged|=lookup.to_numpy(dtype=bool)
   supported&=evidence.supported.reindex(pd.MultiIndex.from_arrays([sites,issues+pd.Timedelta(minutes=h)])).to_numpy(dtype=bool)
 pairs=[]
 for seed in [20261008,20261009,20261010]:
  errors=[]
  for model in ['power_solar','satellite_stats']:
   with np.load(ROLL/fold/str(seed)/(model+'-predictions.npz')) as z: errors.append(abs(z['prediction'][take][:,[1,2,3]]-target))
  pairs.append(errors)
 avg=np.mean(pairs,axis=0); result={}
 for label,keep in [('all',np.ones(len(sites),bool)),('supported_no_flag',~flagged&supported),('candidate_flag',flagged),('unsupported',~supported)]:
  mask=day&keep[:,None]; den=mask.sum(0)
  result[label]={'samples':int(keep.sum()),'daylight_target_counts':den.tolist(),'power_mae':float(((avg[0]*mask).sum(0)/den).mean()) if (den>0).all() else None,'satellite_mae':float(((avg[1]*mask).sum(0)/den).mean()) if (den>0).all() else None}
 association[fold]=result
atomic_json(OUT/'error-association.json',{'folds':association,'scope':'descriptive sensitivity only; training unchanged; separate per-fold fit-derived flags; unsupported solar bands reported separately; not proof that filtering helps'})
# A deterministic visual check of tail cases, including a known rejected missing frame.
cases=[f.loc[f.daylight&f.jump_rms_flag].sort_values('jump_rms',ascending=False).iloc[0],f.loc[f.daylight&(f.season=='spring')&(f.channel=='IR_108')].sort_values('maximum',ascending=False).iloc[0],f.loc[f.daylight&(f.season=='winter')&(f.channel=='IR_016')].sort_values('minimum').iloc[0],f.loc[(f.finite_count<f.pixel_count)&(f.channel=='IR_108')].iloc[0]]
fig,axes=plt.subplots(4,3,figsize=(12,13)); provenance=[]
for row,case in enumerate(cases):
 source=ROOT/'data/processed/route-b-development-v1'/case.partition
 if not source.exists():source=ROOT/'data/processed/route-b-spring-supplement-v1'/case.partition
 z=zarr.open_group(str(source/'satellite_frames.zarr'),mode='r'); frames=open_satellite_frames(source/'satellite_frames.zarr'); times=pd.to_datetime(np.asarray(z['time_ns']),utc=True); si=list(z.attrs['site_ids']).index(case.site); ch=list(z.attrs['channel_names']).index(case.channel); ti=int(np.flatnonzero(times==case.time)[0]); first=max(0,min(ti-1,len(times)-3)); ids=list(range(first,first+3)); arrays=[np.asarray(frames[si,j])[ch].astype(float) for j in ids]; lo=min(np.nanmin(a) for a in arrays); hi=max(np.nanmax(a) for a in arrays)
 for col,(j,a) in enumerate(zip(ids,arrays)):
  ax=axes[row,col]; im=ax.imshow(a,vmin=lo,vmax=hi,cmap='viridis'); ax.set_title(str(times[j])[:16]+(' [selected]' if j==ti else '')); ax.set_xticks([]); ax.set_yticks([])
  if col==0:ax.set_ylabel(case.channel+' / site '+case.site)
 fig.colorbar(im,ax=axes[row,:].tolist(),shrink=.7,label='Stored scaled value; not physical units')
 provenance.append({'partition':case.partition,'site':case.site,'channel':case.channel,'time':str(case.time),'reason':['daylight jump','spring high maximum','winter low minimum','already rejected missing pixels'][row]})
fig.suptitle('Satellite audit: neighboring stored frames (gaps retained)',fontsize=15); fig.savefig(OUT/'suspect-review.png',dpi=140,bbox_inches='tight'); plt.close(fig)
atomic_json(OUT/'visual-review-cases.json',provenance)
nonfinite=f.loc[f.finite_count<f.pixel_count]; missing={'site_frames':len(nonfinite[['site','time']].drop_duplicates()),'timestamps':sorted(nonfinite.time.astype(str).unique().tolist()),'accepted_by_existing_qc':int(nonfinite.loc[nonfinite.existing_frame_valid,['site','time']].drop_duplicates().shape[0]),'nonfinite_pixels':int((nonfinite.pixel_count-nonfinite.finite_count).sum())}; atomic_json(OUT/'missing-pixels.json',missing)
print(json.dumps({'association':association,'missing':missing}))
