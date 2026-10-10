"""Register and retrieve the pinned May source, with bounded Range responses."""
import os,sys,json,hashlib,shutil
from pathlib import Path
import importlib.util
from urllib.request import Request,urlopen
from filelock import FileLock
from huggingface_hub import hf_hub_url,get_hf_file_metadata
import psutil
from cloud2watt.run_state import atomic_json,file_hash
ROOT=Path('D:/code/AI_atm'); OUT=ROOT/'outputs/project-v2/spring-validation-20261009'
spec=importlib.util.spec_from_file_location('r',ROOT/'scripts/download_route_b.py'); r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
assert 'raincalib-gpu' in sys.executable
active=[p.info for p in psutil.process_iter(['pid','name','cmdline']) if p.info['pid']!=os.getpid() and (p.info['name'] or '').lower()=='python.exe' and any(str(x).endswith(('download_route_b.py','complete_spring_supplement.py','supplement_spring.py','download_spring_validation.py')) for x in (p.info['cmdline'] or []))]
assert not active,active
cfg=r.read_json(r.FREEZE); revision=cfg['pv_revision']; name='5_minutely/year=2021/month=05/data.parquet'
# Freeze artifacts before inspecting new validation labels.
models={}
for directory in [ROOT/'outputs/project-v2/rolling-spring-20261008/extended',ROOT/'outputs/project-v2/satellite-ablation-20261009/extended']:
 for p in directory.rglob('*'):
  if p.is_file() and (p.name.endswith('best.pt') or 'normalization' in p.name) and ('local32' in str(p) or 'satellite-ablation' not in str(p)):
   models[p.relative_to(ROOT).as_posix()]=file_hash(p)
freeze=OUT/'evaluation-freeze.json'
if freeze.exists(): assert r.read_json(freeze)['model_hashes']==models
else: atomic_json(freeze,{'model_hashes':models,'primary_comparison':'local32 minus power_solar','secondary':'original satellite_stats','baseline':'smart_persistence','metric':'daylight mean absolute error at 30,60,120 minutes; same valid targets; average seed errors','interval':'paired date-block bootstrap; serial dependence limitation reported','dates':['2021-04-21','2021-05-05'],'training':False,'scoring_in_this_task':False,'decision_sha256':file_hash(ROOT/'docs/adr/020-prospective-spring-validation-download.md')})
remote=get_hf_file_metadata(hf_hub_url('openclimatefix/uk_pv',name,repo_type='dataset',revision=revision),timeout=30)
plan={'name':name,'revision':revision,'bytes':remote.size,'sha256':remote.etag.strip('"'),'response_cap':16*1024**2,'range_bytes':8*1024**2,'retained_prior_attempts':2,'new_attempt_limit':1,'support_end':'2021-05-05T06:00:00Z'}
assert len(plan['sha256'])==64
atomic_json(OUT/'pv-plan.json',plan)
with FileLock(r.STATE/'writer.lock',timeout=0):
 receipts=r.read_json(r.STATE/'pv-receipts.json')
 if name in receipts:
  assert file_hash(Path(receipts[name]['path']))==plan['sha256']
 else:
  budget=r.Budget(cfg); attempts=budget.state.setdefault('pv_attempts',{})
  state=OUT/'pv-transfer.json'
  record=r.read_json(state) if state.exists() else None
  target=r.PREFLIGHT.parent/'pv_source/datasets--openclimatefix--uk_pv/snapshots'/revision/name
  if record is None:
   assert attempts.get(name,0)<3,'historical PV retry limit'
   attempts[name]=attempts.get(name,0)+1; budget.reserve_pv(remote.size)
   record={'status':'registered','attempt':attempts[name],'chunks':{},'actual_bytes':0,'reserved_bytes':remote.size*4};atomic_json(state,record)
  assert record['attempt']==attempts[name]==3
  target.parent.mkdir(parents=True,exist_ok=True); parts=OUT/'pv-parts';parts.mkdir(exist_ok=True)
  for start in range(0,remote.size,8*1024**2):
   end=min(start+8*1024**2,remote.size)-1; key=str(start); piece=parts/key
   if key in record['chunks']:
    assert file_hash(piece)==record['chunks'][key]['sha256'];continue
   assert key not in record.get('started',[]),'interrupted Range attempt retained; no unbounded retry'
   disk=shutil.disk_usage(ROOT);assert disk.free-remote.size*2>=disk.total*.2
   record.setdefault('started',[]).append(key);atomic_json(state,record)
   with urlopen(Request(remote.location,headers={'Range':f'bytes={start}-{end}'}),timeout=60) as response:
    assert response.status==206 and response.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/')
    payload=response.read(16*1024**2+1)
   assert len(payload)==end-start+1
   piece.write_bytes(payload);record['actual_bytes']+=len(payload);record['chunks'][key]={'bytes':len(payload),'sha256':file_hash(piece)};atomic_json(state,record)
  temp=target.with_name('data.parquet.validation-part')
  with temp.open('wb') as stream:
   for start in range(0,remote.size,8*1024**2):stream.write((parts/str(start)).read_bytes())
  assert file_hash(temp)==plan['sha256'] and temp.stat().st_size==remote.size
  temp.replace(target)
  receipts[name]={'path':str(target),'bytes':remote.size,'sha256':plan['sha256'],'role':'prospective_validation_source','labels_read_by_download_function':False};atomic_json(r.STATE/'pv-receipts.json',receipts)
  record['status']='verified';atomic_json(state,record)
 print(json.dumps({'pv':'verified','bytes':remote.size,'model_artifacts_frozen':len(models),'active_conflicting_processes':active}),flush=True)
