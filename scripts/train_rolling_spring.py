"""Local spring QC rebuild and preregistered forward rolling CUDA validation."""
import os
os.environ['HF_HUB_OFFLINE']='1'
import json, time, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import zarr
import train_local_gpu as t
import download_route_b as r
from cloud2watt.data.role_quality import role_quality, within_role, CONTEXT_BITS
from validate_local_mvp import blocked_ci
ROOT=t.ROOT; OUT=ROOT/'outputs/project-v2/rolling-spring-20261008'
SOURCE=ROOT/'data/processed/route-b-spring-supplement-v1'
SEEDS=[20261008,20261009,20261010]
FOLDS={'early':{'inner':['2021-04-07','2021-04-09'],'spring_outer_development':['2021-04-09','2021-04-13']},'middle':{'fit':['2021-04-07','2021-04-10'],'inner':['2021-04-10','2021-04-13'],'spring_outer_development':['2021-04-13','2021-04-17']},'extended':{'fit':['2021-04-07','2021-04-14'],'inner':['2021-04-14','2021-04-17'],'spring_outer_development':['2021-04-17','2021-04-21']}}
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def stamp(stage,**kw): t.atomic_json(OUT/'status.json',{'stage':stage,'final_labels_read':False,'network_requests':0,**kw})
def rebuild(d,p,sites):
    reg=read(d/'preregistration.json')
    issues=pd.date_range(reg['issue_start_inclusive'],reg['issue_end_exclusive'],freq='15min',inclusive='left')
    ft=pd.to_datetime(np.asarray(zarr.open_group(str(d/'satellite_frames.zarr'),mode='r')['time_ns']),utc=True)
    quality=pd.read_parquet(d/'satellite_quality.parquet').set_index(['ss_id','datetime_GMT'])
    lookup=pd.MultiIndex.from_frame(p[['ss_id','datetime_GMT']]); blocks=[]
    for si,site in enumerate(sites.ss_id.astype(str)):
        history=issues.as_unit('ns').asi8[:,None]+np.array([-45,-30,-15,0])*60*10**9
        targets=issues.as_unit('ns').asi8[:,None]+np.array([15,30,60,120,180,240])*60*10**9
        ht=pd.to_datetime(history.ravel(),utc=True); tt=pd.to_datetime(targets.ravel(),utc=True)
        hi=lookup.get_indexer(pd.MultiIndex.from_arrays([[site]*len(ht),ht])).reshape(-1,4)
        ti=lookup.get_indexer(pd.MultiIndex.from_arrays([[site]*len(tt),tt])).reshape(-1,6)
        fi=ft.get_indexer(ht).reshape(-1,4)
        sq=quality.is_valid.reindex(pd.MultiIndex.from_arrays([[site]*len(ht),ht]),fill_value=False).to_numpy().reshape(-1,4)
        keep=(hi>=0).all(1)&(fi>=0).all(1)&sq.all(1)&np.isfinite(p.normalized_power.to_numpy()[np.maximum(hi,0)]).all(1)
        mask=(ti>=0)&np.isfinite(p.normalized_power.to_numpy()[np.maximum(ti,0)])
        block=pd.DataFrame({'site_id':site,'site_index':si,'issue_time_utc':issues[keep],'satellite_frame_indices':list(fi[keep]),'power_history_row_indices':list(hi[keep]),'target_row_indices':list(ti[keep]),'target_mask':list(mask[keep]),'quality_flags':'pending_role_qc'})
        blocks.append(block)
    return pd.concat(blocks,ignore_index=True)
def prepare():
    completed=read(ROOT/'outputs/project-v2/spring-supplement-20261008/completion-progress.json')['completed']
    views=OUT/'views'; parts=[]; power=[]; snapshots={}
    for n,(part,receipt) in enumerate(sorted(completed.items()),1):
        d=SOURCE/part; assert t.file_hash(d/'verified.json')==receipt['receipt_sha256']; r.verify_shard(d)
        p=pd.read_parquet(d/'power_15min.parquet'); sites=pd.read_parquet(d/'sites.parquet')
        # All pointwise inputs, without retaining the old context-dependent selection.
        samples=rebuild(d,p,sites); v=views/part; v.mkdir(parents=True,exist_ok=True)
        p.to_parquet(v/'power_15min.parquet',index=False); sites.to_parquet(v/'sites.parquet',index=False); samples.to_parquet(v/'sample_index.parquet',index=False)
        t.atomic_json(v/'verified.json',{'source_receipt_sha256':receipt['receipt_sha256'],'files':{f:t.file_hash(v/f) for f in ['power_15min.parquet','sites.parquet','sample_index.parquet']}})
        snapshots[part]={'receipt_sha256':t.file_hash(v/'verified.json')}
        q=samples[['site_id','issue_time_utc']].copy(); q['partition']=part; q['sample_row']=np.arange(len(q)); q['role']='pending_role_qc'; parts.append(q); power.append(p)
        stamp('rebuild_candidates',completed=n,total=len(completed))
    index=pd.concat(parts,ignore_index=True); assert not index.duplicated(['site_id','issue_time_utc']).any()
    idx=OUT/'candidate-index.parquet'; index.to_parquet(idx,index=False)
    cfg=OUT/'feature-config.json'; t.atomic_json(cfg,{'completed_snapshot':snapshots,'local_audit':{'sha256':t.file_hash(idx)}})
    t.OUT=OUT; t.INDEX=idx; t.DEST=views
    data=t.prepare(cfg,satellite_root=SOURCE)
    p=pd.concat(power,ignore_index=True); p['base_flags']=p.quality_flags.astype('uint16')&np.uint16(65535^CONTEXT_BITS)
    # Overlapping shard contexts must describe the same source observations.
    cols=['normalized_power','reading_count','base_flags']
    assert (p.groupby(['ss_id','datetime_GMT'])[cols].nunique(dropna=False)<=1).all().all()
    p=p.drop_duplicates(['ss_id','datetime_GMT']).sort_values(['ss_id','datetime_GMT']).reset_index(drop=True)
    p.to_parquet(OUT/'spring-power-grid.parquet',index=False)
    return data,p

def assign(data,p,bounds):
    issues=pd.DatetimeIndex(pd.to_datetime(data['time'],utc=True)); sites=data['site']; n=len(issues)
    role=np.full(n,'excluded',dtype='<U32'); masks=np.zeros((n,6),bool); audit={}
    for name,dates in bounds.items():
        a,b=[pd.Timestamp(x,tz='UTC') for x in dates]; q=role_quality(p,a,b)
        q.to_parquet(OUT/'qc'/CURRENT/(name+'.parquet'),index=False)
        lookup=q.set_index(['ss_id','datetime_GMT']).is_valid
        def valid(offsets):
            ts=issues.as_unit('ns').asi8[:,None]+np.array(offsets)*60*10**9
            keys=pd.MultiIndex.from_arrays([np.repeat(sites,len(offsets)),pd.to_datetime(ts.ravel(),utc=True)])
            return lookup.reindex(keys,fill_value=False).to_numpy().reshape(n,len(offsets))
        hist=valid([-45,-30,-15,0]); target=valid([15,30,60,120,180,240])&data['mask']
        eligible=within_role(issues,a,b); keep=eligible&hist.all(1)&target.any(1)
        assert (role[keep]=='excluded').all(); role[keep]=name; masks[keep]=target[keep]
        audit[name]={'samples':int(keep.sum()),'dates':sorted(set(str(x)[:10] for x in data['time'][keep])),'boundary_eligible':int(eligible.sum()),'invalid_history_or_no_target':int((eligible&~keep).sum()),'qc_rows':len(q),'qc_sha256':t.file_hash(OUT/'qc'/CURRENT/(name+'.parquet'))}
    keep=role!='excluded'; result={k:v[keep].copy() for k,v in data.items()}; result['role']=role[keep]; result['mask']=masks[keep]; result['y']=np.where(result['mask'],result['y'],0)
    return result,audit

def main():
    global CURRENT
    OUT.mkdir(exist_ok=True); start=time.monotonic()
    assert 'raincalib-gpu' in t.sys.executable and torch.cuda.is_available()
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    with t.run_lock(OUT):
        assert shutil.disk_usage(ROOT).free/shutil.disk_usage(ROOT).total>.2
        prereg={'folds':FOLDS,'seeds':SEEDS,'epochs':20,'gpu':torch.cuda.get_device_name(0),'final_labels_read':False,'network_requests':0,'qc':'group-local, raw support -60min, target support +240min, all horizons guarded','selection':'pooled winter/spring inner; fit-only normalization','decision_sha256':t.file_hash(ROOT/'docs/adr/017-expanded-spring-rolling-validation.md'),'script_sha256':t.file_hash(Path(__file__)),'quality_sha256':t.file_hash(ROOT/'src/cloud2watt/data/role_quality.py'),'trainer_sha256':t.file_hash(ROOT/'scripts/train_local_gpu.py')}
        manifest=OUT/'preregistration.json'
        if manifest.exists(): assert read(manifest)==prereg
        else: t.atomic_json(manifest,prereg)
        spring,p=prepare(); old=[]
        for path in sorted((ROOT/'outputs/project-v2/local-gpu-20261008/features').glob('*.npz')):
            with np.load(path) as z: old.append({k:z[k] for k in z.files})
        assert sum(len(x['role']) for x in old)==76199
        old_hashes={path.name:t.file_hash(path) for path in sorted((ROOT/'outputs/project-v2/local-gpu-20261008/features').glob('*.npz'))}
        frozen_hashes=read(ROOT/'outputs/project-v2/spring-gpu-20261008/preregistration.json')['old_feature_hashes']; assert old_hashes==frozen_hashes
        combined_dates=[]; combined_differences=[]; combined_masks=[]; folds={}
        for CURRENT,bounds in FOLDS.items():
            folder=OUT/CURRENT; folder.mkdir(exist_ok=True); (OUT/'qc'/CURRENT).mkdir(parents=True,exist_ok=True)
            extra,audit=assign(spring,p,bounds); data={k:np.concatenate([b[k] for b in old]+[extra[k]]) for k in spring}
            keys=pd.DataFrame({'site_id':data['site'],'issue_time_utc':data['time'],'role':data['role']}); assert not keys.duplicated(['site_id','issue_time_utc']).any()
            keys.to_parquet(folder/'split-index.parquet',index=False)
            with t.atomic_path(folder/'dataset.npz') as tmp:
                with tmp.open('wb') as f: np.savez_compressed(f,**data)
            t.atomic_json(folder/'split-audit.json',audit)
            counts={str(k):int(v) for k,v in pd.Series(data['role']).value_counts().items()}
            print('FOLD',CURRENT,counts,flush=True)
            take=data['role']=='spring_outer_development'; day=data['mask'][take][:,[1,2,3]]&(data['elevation'][take][:,[1,2,3]]>5)
            differences=[]; scores=[]
            for seed in SEEDS:
                assert time.monotonic()-start<6*3600,'six-hour wallclock guard'
                assert shutil.disk_usage(ROOT).free/shutil.disk_usage(ROOT).total>.2
                t.SEED=seed; t.OUT=folder/str(seed); t.OUT.mkdir(exist_ok=True)
                identity={'parent_sha256':t.file_hash(manifest),'dataset_sha256':t.file_hash(folder/'dataset.npz'),'seed':seed,'fold':CURRENT,'old_feature_hashes':old_hashes}
                t.atomic_json(t.OUT/'preregistration.json',identity)
                stamp('training',fold=CURRENT,seed=seed,gpu=prereg['gpu'],counts=counts)
                if not (t.OUT/'status.json').exists() or read(t.OUT/'status.json').get('stage')!='complete': t.train(data)
                metrics={}; predictions={}
                for kind in ['persistence','smart_persistence','power_solar','satellite_stats']:
                    with np.load(t.OUT/(kind+'-predictions.npz')) as z:
                        for k in ['site','time','role','mask']: assert np.array_equal(z[k],data[k])
                        assert np.array_equal(z['target'],data['y']); pred=z['prediction']; assert np.isfinite(pred).all()
                    predictions[kind]=pred[take]
                    metrics[kind]=t.score_forecast(pred[take],data['y'][take],data['mask'][take],data['elevation'][take],keys.loc[take,['site_id','issue_time_utc']])
                    if kind in ['power_solar','satellite_stats']:
                        assert len(read(t.OUT/(kind+'-history.json')))==20
                        assert (t.OUT/(kind+'-best.pt')).stat().st_size>0
                t.atomic_json(t.OUT/'spring-outer-metrics.json',metrics)
                scores.append({'seed':seed,**{k:v['daylight_primary_mae'] for k,v in metrics.items()}})
                differences.append(abs(predictions['satellite_stats'][:,[1,2,3]]-data['y'][take][:,[1,2,3]])-abs(predictions['power_solar'][:,[1,2,3]]-data['y'][take][:,[1,2,3]]))
            dates=np.array([str(x)[:10] for x in data['time'][take]]); diff=np.mean(differences,axis=0)
            folds[CURRENT]={'counts':counts,'spring_roles':audit,'scores':scores,'satellite_vs_power_ci':blocked_ci(diff,day,dates)}
            combined_dates.append(dates); combined_differences.append(diff); combined_masks.append(day)
            t.atomic_json(OUT/'results.json',{'folds':folds,'final_labels_read':False})
        dates=np.concatenate(combined_dates); assert sum(len(np.unique(d)) for d in combined_dates)==len(np.unique(dates))
        result={'folds':folds,'rolling_satellite_vs_power_ci':blocked_ci(np.concatenate(combined_differences),np.concatenate(combined_masks),dates),'independent_final_test':False,'final_labels_read':False,'network_requests':0,'training_runs':18,'elapsed_seconds':time.monotonic()-start}
        t.atomic_json(OUT/'results.json',result)
        t.atomic_json(OUT/'acceptance.json',{'passed':True,'training_runs':18,'epochs_each':20,'all_prediction_keys_targets_masks_match':True,'evaluation_dates':sorted(np.unique(dates).tolist()),'final_labels_read':False,'results_sha256':t.file_hash(OUT/'results.json')})
        stamp('complete',training_runs=18,elapsed_seconds=result['elapsed_seconds'],gpu=prereg['gpu'])
        print('COMPLETE',result['rolling_satellite_vs_power_ci'],flush=True)
if __name__=='__main__': main()
