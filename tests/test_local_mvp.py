import numpy as np
import pandas as pd
from cloud2watt.mvp import LocalForecastMVP

def engine():
    e=object.__new__(LocalForecastMVP); e.provenance={}
    e._predict=lambda name,x: np.ones(6)*{'power_only':.1,'power_solar':.2,'satellite_stats':.3}[name]
    return e

def request():
    issue=pd.Timestamp('2021-01-10T12:00Z')
    return dict(issue_time=issue,site={'ss_id':'1','kWp':3,'tilt':30,'orientation':180,'latitude_rounded':51,'longitude_rounded':-1},history_times=[issue+pd.Timedelta(minutes=m) for m in [-45,-30,-15,0]],history_power=[.1,.2,.2,.3])

def test_default_and_explanations():
    r=engine().forecast(**request()); assert r['selected_model']=='power_solar'
    assert np.allclose(r['explanation']['solar_model_difference'],.1)
    assert r['explanation']['satellite_model_difference'] is None

def test_satellite_and_missing_fallback():
    q=request(); e=engine()
    r=e.forecast(**q,enable_satellite=True,satellite_features=np.ones(105),satellite_times=q['history_times'])
    assert r['selected_model']=='satellite_stats'
    for sat in [None,np.full(105,np.nan),np.ones(104)]:
        r=e.forecast(**q,enable_satellite=True,satellite_features=sat,satellite_times=q['history_times'])
        assert r['selected_model']=='power_solar' and r['status']=='ok'
    r=e.forecast(**q,enable_satellite=True,satellite_features=np.ones(105),satellite_times=[pd.Timestamp(t)+pd.Timedelta(minutes=15) for t in q['history_times']])
    assert r['selected_model']=='power_solar'

def test_power_failure_no_fabricated_forecast():
    for power in [[.1,np.nan,.2,.3],[.1,.2],[-1,.2,.2,.3]]:
        q=request(); q['history_power']=power; r=engine().forecast(**q)
        assert r['status']=='unavailable' and r['prediction'] is None
    q=request(); q['history_times'][-1]+=pd.Timedelta(minutes=15)
    assert engine().forecast(**q)['status']=='unavailable'
    assert engine().forecast(**request(),history_valid=[1,1,0,1])['status']=='unavailable'

def test_metadata_failure():
    q=request(); q['site']['kWp']=0
    assert engine().forecast(**q)['status']=='unavailable'
