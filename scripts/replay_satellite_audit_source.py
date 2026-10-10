"""Compare local source chunks and derived crops; no network fallback."""
from pathlib import Path
import numpy as np,pandas as pd,zarr,json
from cloud2watt.data.seviri import SeviriStore
from cloud2watt.data.shared_tiles import open_satellite_frames
from cloud2watt.run_state import atomic_json
root=Path('D:/code/AI_atm'); cache=root/'outputs/project-v2/source-cache-2021/objects'
class Local(SeviriStore):
 def _fetch_bytes(self,key):return (cache/key).read_bytes()
c=Local('offline',cache_dir=cache); results=[]; source_times=c.timestamps()
for region in ['r0','r1','r2']:
 p=root/'data/processed/route-b-spring-supplement-v1/spring/20210420'/region
 z=zarr.open_group(str(p/'satellite_frames.zarr'),mode='r'); ft=pd.to_datetime(np.asarray(z['time_ns']),utc=True); sites=pd.read_parquet(p/'sites.parquet'); site=c.map_sites(sites.iloc[:1])[0]
 channels=[c.channels().index(x) for x in ['VIS006','IR_016','IR_108']]
 for hour in [0,12,18]:
  time=pd.Timestamp('2021-04-20',tz='UTC')+pd.Timedelta(hours=hour); ti=int(np.flatnonzero(ft==time)[0])
  assert source_times[int(z['source_frame_indices'][ti])]==time
  raw=c.read_crop(time_index=int(z['source_frame_indices'][ti]),site=site,channel_indices=channels,height=128,width=128)
  stored=np.asarray(open_satellite_frames(p/'satellite_frames.zarr')[0,ti]); assert np.array_equal(raw,stored,equal_nan=True)
  results.append({'region':region,'hour':hour,'source_equals_stored':True,'ranges':[[float(v.min()),float(v.max())] for v in raw]})
atomic_json(root/'outputs/project-v2/satellite-quality-20261008/source-replay.json',{'checks':results,'network_requests':0,'scaling':'stored values equal local compressed source; physical inverse transform not established'})
print(json.dumps(results))
