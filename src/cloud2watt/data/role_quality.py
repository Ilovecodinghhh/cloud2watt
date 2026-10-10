"""Recompute context QC strictly within a temporal development role."""
import numpy as np
import pandas as pd
from cloud2watt.data.quality import flag_context_quality, QualityFlag

CONTEXT_BITS = int(QualityFlag.NIGHT_NONZERO | QualityFlag.DAYTIME_STUCK_ZERO)

def role_quality(power, start, end):
    # A 15-minute aggregate labelled at start still contains pre-role readings.
    p = power.loc[(power.datetime_GMT > start) & (power.datetime_GMT < end)].copy()
    p = p.sort_values(['ss_id', 'datetime_GMT']).reset_index(drop=True)
    if p.duplicated(['ss_id', 'datetime_GMT']).any():
        raise ValueError('duplicate power keys')
    p['quality_flags'] = p.quality_flags.astype('uint16') & np.uint16(65535 ^ CONTEXT_BITS)
    p['is_valid'] = p.quality_flags.eq(0)
    segments = (p.ss_id.ne(p.ss_id.shift()) | p.datetime_GMT.diff().ne(pd.Timedelta(minutes=15))).cumsum()
    blocks = [flag_context_quality(g)[0] for _, g in p.groupby(segments, sort=False)]
    return pd.concat(blocks, ignore_index=True) if blocks else p

def within_role(issues, start, end):
    issues = pd.DatetimeIndex(issues)
    return (issues - pd.Timedelta(minutes=60) >= start) & (issues + pd.Timedelta(minutes=240) < end)
