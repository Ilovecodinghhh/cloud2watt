"""Overlay observed development power on a saved forecast; never rerun inference."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cloud2watt.run_state import atomic_json,file_hash
ROOT=Path('D:/code/AI_atm')
OUT=ROOT/'outputs/project-v2/mvp-v2/example'

def main():
    f=json.loads((OUT/'forecast.json').read_text(encoding='utf-8'))
    issue=pd.Timestamp(f['issue_time_utc']); site=f['site_id']; horizons=f['horizons_minutes']
    index=pd.read_parquet(ROOT/'outputs/project-v2/local-provenance-20261007/safe-split-index.parquet')
    found=index.loc[index.site_id.astype(str).eq(site)&index.issue_time_utc.eq(issue)]
    assert len(found)==1
    row=found.iloc[0]; assert row.role in ['fit','inner','outer_development','spring_stress_development']
    cfg=json.loads((ROOT/'configs/project-v2/local-first.json').read_text(encoding='utf-8'))
    assert row.partition in cfg['completed_snapshot']
    d=ROOT/'data/processed/route-b-development-v1'/row.partition
    assert file_hash(d/'verified.json')==cfg['completed_snapshot'][row.partition]['receipt_sha256']
    power=pd.read_parquet(d/'power_15min.parquet'); sample=pd.read_parquet(d/'sample_index.parquet').iloc[int(row.sample_row)]
    assert str(sample.site_id)==site and sample.issue_time_utc==issue
    indices=np.asarray(sample.target_row_indices,dtype=int); valid=np.asarray(sample.target_mask,dtype=bool)&(indices>=0)
    truth=np.full(6,np.nan)
    for i,h in enumerate(horizons):
        if indices[i]<0: continue
        p=power.iloc[indices[i]]
        assert str(p.ss_id)==site and p.datetime_GMT==issue+pd.Timedelta(minutes=h)
        assert bool(p.is_valid)==bool(valid[i])
        if valid[i]: truth[i]=p.normalized_power
    assert np.isfinite(truth[valid]).all()
    curve=power.loc[power.ss_id.astype(str).eq(site)].set_index('datetime_GMT').reindex(pd.date_range(issue,issue+pd.Timedelta(hours=4),freq='15min'))
    actual=curve.normalized_power.where(curve.is_valid.fillna(False)).to_numpy(dtype=float)
    minutes=np.arange(0,241,15)
    scores={}
    for name,pred in f['comparisons'].items():
        if pred is None: continue
        error=np.asarray(pred)-truth
        scores[name]={'valid_targets':int(valid.sum()),'absolute_error': [float(abs(e)) if v else None for e,v in zip(error,valid)],'mae':float(np.abs(error[valid]).mean()) if valid.any() else None,'rmse':float(np.sqrt(np.mean(error[valid]**2))) if valid.any() else None}
    fig=plt.figure(figsize=(12,10),layout='constrained'); grid=fig.add_gridspec(3,1,height_ratios=[3.5,1.7,2.0]); ax=fig.add_subplot(grid[0]); sky=fig.add_subplot(grid[1]); tab=fig.add_subplot(grid[2])
    ax.plot([-45,-30,-15,0],np.asarray(f['history']['power'])*100,'ko-',label='Observed history',linewidth=2)
    for name,pred in f['comparisons'].items():
        if pred is not None: ax.plot(horizons,np.asarray(pred)*100,marker='.',label=name,alpha=.85,color={'power_only':'tab:blue','power_solar':'tab:orange','satellite_stats':'tab:green','persistence':'tab:red','smart_persistence':'tab:purple'}[name])
    ax.plot(minutes,actual*100,color='black',linewidth=2.6,linestyle='--',label='Actual future (15-min observations)',zorder=5)
    ax.scatter(np.array(horizons)[valid],truth[valid]*100,color='black',marker='D',s=35,label='Actual at scored horizons',zorder=6)
    missing=minutes[~np.isfinite(actual)]
    if len(missing): ax.text(.99,.02,'Missing actual at '+', '.join('+'+str(m)+' min' for m in missing)+'; line left broken',transform=ax.transAxes,ha='right',fontsize=8,color='dimgray')
    ax.axvline(0,color='gray',linestyle=':'); ax.set_ylabel('Power / installed capacity (%)'); ax.legend(fontsize=8,ncol=2); ax.grid(alpha=.2)
    ax.set_title(f"Site {site} | issued {issue.isoformat()}\nHistorical replay: predictions saved before this overlay; one example, not overall ranking",fontsize=11)
    sky.plot([0,*horizons],f['clear_sky_reference']['ghi'],color='orange',marker='.'); sky.set_ylabel('Clear-sky irradiance (W/m²)'); sky.set_xlabel('Minutes from issue time'); sky.grid(alpha=.2)
    names=list(scores); values=[]
    for name in names:
        s=scores[name]; values.append([name,*['missing' if e is None else f'{100*e:.2f}' for e in s['absolute_error']],f"{100*s['mae']:.2f}" if s['mae'] is not None else 'N/A'])
    tab.axis('off'); table=tab.table(cellText=values,colLabels=['Method',*[f'+{h} min' for h in horizons],'MAE'],cellLoc='center',colWidths=[.23,*([.11]*7)],loc='center'); table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1,1.6)
    best=min((name for name in names if scores[name]['mae'] is not None),key=lambda name:scores[name]['mae'],default=None)
    if best:
        for j in range(8): table[names.index(best)+1,j].set_facecolor('#dff2df')
    tab.set_title(f'Absolute errors in capacity percentage points | {int(valid.sum())}/6 valid horizons | lower is better',fontsize=10)
    fig.savefig(OUT/'explanation-with-actual.png',dpi=160); plt.close(fig)
    atomic_json(OUT/'actual-comparison.json',{'site_id':site,'issue_time':issue.isoformat(),'role':row.role,'partition':row.partition,'forecast_sha256':file_hash(OUT/'forecast.json'),'power_sha256':file_hash(d/'power_15min.parquet'),'truth':[float(x) if v else None for x,v in zip(truth,valid)],'target_mask':valid.tolist(),'horizons_minutes':horizons,'scores':scores,'best_in_this_case':best,'scoring':'all valid six forecast horizons, equal weight; no interpolated targets','final_labels_read':False})
    print(json.dumps({'truth':truth.tolist(),'mask':valid.tolist(),'scores':scores,'best':best}))
if __name__=='__main__': main()
