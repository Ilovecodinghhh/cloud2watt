import numpy as np
import pandas as pd
from cloud2watt.data.role_quality import role_quality, within_role

def frame(times, values):
    return pd.DataFrame({'ss_id':'s','datetime_GMT':pd.to_datetime(times,utc=True),'normalized_power':values,'solar_elevation_deg':20.,'quality_flags':np.uint16(64),'is_valid':False})

def test_outside_role_cannot_affect_context_qc():
    times=pd.date_range('2021-04-01', periods=20, freq='15min',tz='UTC')
    p=frame(times,np.zeros(20)); a=times[10]; b=times[17]
    before=role_quality(p,a,b)
    p.loc[p.datetime_GMT.le(a)|p.datetime_GMT.ge(b),'normalized_power']=100
    pd.testing.assert_frame_equal(before,role_quality(p,a,b))
    assert before.is_valid.all()  # six zeros inside, not eight

def test_missing_time_breaks_zero_run():
    times=pd.date_range('2021-04-01T01:00Z',periods=9,freq='15min').delete(4)
    p=frame(times,np.zeros(8)); q=role_quality(p,pd.Timestamp('2021-04-01',tz='UTC'),pd.Timestamp('2021-04-02',tz='UTC'))
    assert q.is_valid.all()

def test_eight_contiguous_zero_bins_flagged():
    times=pd.date_range('2021-04-01T01:00Z',periods=8,freq='15min')
    q=role_quality(frame(times,np.zeros(8)),times[0]-pd.Timedelta(hours=1),times[-1]+pd.Timedelta(hours=1))
    assert not q.is_valid.any()

def test_support_boundary_is_strict():
    a=pd.Timestamp('2021-04-01',tz='UTC'); b=a+pd.Timedelta(days=1)
    times=[a+pd.Timedelta(minutes=45),a+pd.Timedelta(hours=1),b-pd.Timedelta(hours=4),b-pd.Timedelta(hours=4,minutes=15)]
    assert within_role(times,a,b).tolist()==[False,True,False,True]

def test_source_quality_flags_survive_context_reset():
    times=pd.date_range('2021-04-01T01:00Z',periods=2,freq='15min')
    p=frame(times,[.2,.2]); p.loc[0,'quality_flags']=66
    q=role_quality(p,times[0]-pd.Timedelta(hours=1),times[-1]+pd.Timedelta(hours=1))
    assert int(q.quality_flags.iloc[0])==2
    assert q.is_valid.tolist()==[False,True]

def test_aggregate_at_start_is_excluded():
    a=pd.Timestamp('2021-04-01',tz='UTC')
    p=frame([a,a+pd.Timedelta(minutes=15)],[.2,.2])
    q=role_quality(p,a,a+pd.Timedelta(hours=1))
    assert q.datetime_GMT.tolist()==[a+pd.Timedelta(minutes=15)]
