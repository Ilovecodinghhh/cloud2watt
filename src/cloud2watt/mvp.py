"""Explainable local replay MVP with explicit input gates and no label dependency."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from cloud2watt.data.solar import build_solar_features
from cloud2watt.run_state import file_hash
from cloud2watt.training import SOLAR_COLUMNS,SITE_COLUMNS

HORIZONS=[15,30,60,120,180,240]

class LocalForecastMVP:
    """Frozen seed, CPU inference; GPU is used for model training, not required here."""
    def __init__(self, baseline_dir, validation_dir):
        baseline_dir,validation_dir=Path(baseline_dir),Path(validation_dir)
        self.models={}; self.provenance={}
        for name,path in [('power_solar',baseline_dir/'power_solar-best.pt'),
                          ('satellite_stats',baseline_dir/'satellite_stats-best.pt'),
                          ('power_only',validation_dir/'power_only-best.pt')]:
            c=torch.load(path,map_location='cpu',weights_only=False)
            m=nn.Sequential(nn.Linear(c['input_dim'],128),nn.SiLU(),nn.Dropout(.1),
                            nn.Linear(128,128),nn.SiLU(),nn.Dropout(.1),nn.Linear(128,6))
            m.load_state_dict(c['model']); m.eval()
            self.models[name]=(m,np.asarray(c['mean']),np.asarray(c['scale']))
            self.provenance[name]={'path':str(path),'sha256':file_hash(path),'seed':20261008}

    def _predict(self,name,values):
        model,mean,scale=self.models[name]
        with torch.no_grad(): result=model(torch.tensor((values-mean)/scale,dtype=torch.float32)[None]).numpy()[0]
        if not np.isfinite(result).all(): raise ValueError('nonfinite model prediction')
        return result

    def forecast(self, *, issue_time, site, history_times, history_power,
                 history_valid=None, satellite_features=None, satellite_times=None,
                 satellite_valid=None, enable_satellite=False):
        issue=pd.Timestamp(issue_time)
        if issue.tzinfo is None: raise ValueError('issue_time must be timezone-aware')
        issue=issue.tz_convert('UTC')
        expected=pd.DatetimeIndex([issue+pd.Timedelta(minutes=x) for x in [-45,-30,-15,0]])
        result={'schema':'c2w-explainable-mvp-v1','status':'unavailable','issue_time_utc':issue.isoformat(),
                'site_id':str(site.get('ss_id','unknown')),'horizons_minutes':HORIZONS,
                'selected_model':None,'prediction':None,'quality':[],
                'scope':'historical-development replay; not independently validated for deployment',
                'satellite_experimental':True,'provenance':self.provenance,
                'explanation_type':'trained-model comparisons, not causal feature attribution'}
        try:
            times=pd.DatetimeIndex(history_times)
            power=np.asarray(history_power,dtype=np.float32)
            good=np.ones(4,dtype=bool) if history_valid is None else np.asarray(history_valid,dtype=bool)
            if times.tz is None or not times.tz_convert('UTC').equals(expected) or power.shape!=(4,) or good.shape!=(4,) or not good.all() or not np.isfinite(power).all() or (power<0).any() or (power>1.5).any():
                result['quality'].append('power_history_missing_invalid_or_misaligned'); return result
        except (ValueError,TypeError):
            result['quality'].append('power_history_missing_invalid_or_misaligned'); return result
        try:
            static=np.array([site[c] for c in SITE_COLUMNS],dtype=np.float32)
            if not np.isfinite(static).all() or static[0]<=0 or abs(static[3])>90 or abs(static[4])>180: raise ValueError()
            sites=pd.DataFrame([site]); target_times=pd.DatetimeIndex([issue+pd.Timedelta(minutes=h) for h in HORIZONS])
            solar=build_solar_features(pd.DatetimeIndex([issue,*target_times]),sites)
            sf=solar.iloc[1:][list(SOLAR_COLUMNS)].to_numpy(dtype=np.float32)
            ghi0=float(solar.iloc[0].clear_sky_ghi_wm2)
            if not np.isfinite(sf).all(): raise ValueError()
        except (KeyError,ValueError,TypeError):
            result['quality'].append('site_or_solar_features_invalid'); return result
        base=np.concatenate([power,sf.reshape(-1),static])
        p=self._predict('power_solar',base)
        p0=self._predict('power_only',np.concatenate([power,static]))
        persistence=np.repeat(power[-1],6)
        smart=persistence*np.clip(sf[:,2]/max(ghi0,20),0,5) if ghi0>20 else persistence.copy()
        sat=None; sat_ok=False
        if enable_satellite:
            try:
                features=np.asarray(satellite_features,dtype=np.float32)
                times=pd.DatetimeIndex(satellite_times)
                good=np.ones(4,dtype=bool) if satellite_valid is None else np.asarray(satellite_valid,dtype=bool)
                sat_ok=(features.shape==(105,) and np.isfinite(features).all() and times.tz is not None and times.tz_convert('UTC').equals(expected) and good.shape==(4,) and good.all())
            except (TypeError,ValueError): sat_ok=False
            if sat_ok: sat=self._predict('satellite_stats',np.concatenate([base,features]))
            else: result['quality'].append('satellite_missing_invalid_or_misaligned_fallback_to_power_solar')
        else: result['quality'].append('satellite_disabled_by_default')
        selected=sat if sat_ok else p
        result.update(status='ok',selected_model='satellite_stats' if sat_ok else 'power_solar',prediction=selected.tolist(),
            unit='power / installed_capacity',history={'times':[x.isoformat() for x in expected],'power':power.tolist()},
            forecast_times=[x.isoformat() for x in target_times],
            clear_sky_reference={'unit':'W/m² (irradiance, not PV power)','ghi':[ghi0,*sf[:,2].tolist()]},
            comparisons={'power_only':p0.tolist(),'power_solar':p.tolist(),'satellite_stats':None if sat is None else sat.tolist(),
                         'persistence':persistence.tolist(),'smart_persistence':smart.tolist()},
            explanation={'solar_model_difference':(p-p0).tolist(),'satellite_model_difference':None if sat is None else (sat-p).tolist(),
                         'caveat':'Differences include separate model fitting and parameter counts; they are not additive causal contributions.'})
        if (selected<0).any() or (selected>1.5).any(): result['quality'].append('model_output_outside_nominal_range_not_silently_clipped')
        return result


def render_explanation(result,path):
    """Standalone figure: separate axes prevent confusing irradiance and PV power."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(10,7),layout='constrained')
    if result['status']!='ok':
        axes[0].text(.05,.5,'Prediction unavailable: '+', '.join(result['quality']),wrap=True,transform=axes[0].transAxes)
        axes[1].set_visible(False)
    else:
        axes[0].plot([-45,-30,-15,0],result['history']['power'],'ko-',label='Observed recent power')
        for key,values in result['comparisons'].items():
            if values is not None: axes[0].plot(HORIZONS,values,marker='.',label=key)
        axes[0].set_ylabel('Power / installed capacity'); axes[0].legend(fontsize=8)
        axes[1].plot([0,*HORIZONS],result['clear_sky_reference']['ghi'],color='orange',marker='.')
        axes[1].set_ylabel('Clear-sky irradiance (W/m²)'); axes[1].set_xlabel('Minutes from issue time')
    axes[0].set_title(f"Site {result['site_id']} | {result['issue_time_utc']}\nSelected: {result['selected_model']} | model comparisons, not causal attribution",fontsize=10)
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(path,dpi=150); plt.close(fig)
