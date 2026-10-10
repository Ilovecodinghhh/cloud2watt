"""Bounded one-shot spring supplement; stop after registered complete-date bundles."""
import os,sys,time,json,base64,hashlib
from pathlib import Path
from urllib.request import Request,urlopen
from types import SimpleNamespace
import importlib.util
import google_crc32c
import pandas as pd
from filelock import FileLock
from cloud2watt.run_state import atomic_json,file_hash
ROOT=Path('D:/code/AI_atm'); STATE=ROOT/'outputs/project-v2/spring-supplement-20261008'; DEST=ROOT/'data/processed/route-b-spring-supplement-v1'
spec=importlib.util.spec_from_file_location('route',ROOT/'scripts/download_route_b.py'); r=importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
class OfflineStore(r.CheckedDevelopmentStore):
 def _fetch_bytes(self,key):
  path=self._cache_path(key)
  if not path.exists(): raise FileNotFoundError('Unregistered/offline cache miss: '+key)
  payload=path.read_bytes()
  if key.startswith('data/') and (len(payload)<16 or int.from_bytes(payload[12:16],'little')!=len(payload)): raise ValueError('Invalid cached block '+key)
  self.cache_hits+=1; return payload

def main():
 assert 'raincalib-gpu' in sys.executable
 os.chdir(ROOT); DEST.mkdir(exist_ok=True)
 start=time.monotonic(); deadline=start+1800
 with FileLock(r.STATE/'writer.lock',timeout=0),FileLock(STATE/'writer.lock',timeout=0):
  plan=r.read_json(STATE/'plan.json'); assert plan['status']=='sized'
  cfg=r.read_json(r.FREEZE); budget=r.Budget(cfg); objects=plan['objects']; old=r.read_json(r.STATE/'progress.json')
  tasks={t['id']:t for t in r.tasks_for(cfg)}; chosen=[]; required=set(); reserved=0
  for date in sorted({k.split('/')[1] for k in plan['partition_objects']}):
   ids=sorted(k for k in plan['partition_objects'] if k.split('/')[1]==date)
   if len(ids)!=3 or any(old.get('attempts',{}).get(k,0)>=3 for k in ids): break
   keys={x for k in ids for x in plan['partition_objects'][k]}
   extra=sum(objects[k]['bytes'] for k in keys-required if objects[k]['status']=='sized')
   # Keep one maximum response reservation for a failed transfer.
   if reserved+extra+16*1024**2>plan['new_transfer_limit']: break
   chosen.extend(ids); required|=keys; reserved+=extra
  execution={'partitions':chosen,'source_keys':sorted(required),'estimated_new_bytes':reserved,'transfer_cap':plan['new_transfer_limit'],'unselected_partitions':[k for k in plan['partition_objects'] if k not in chosen],'stop_after_batch':True,'training':False,'final_labels_locked':True}
  atomic_json(STATE/'execution.json',execution)
  if not chosen:
   atomic_json(STATE/'progress.json',{'status':'budget_blocked','execution':execution}); return
  ledger_path=STATE/'transfer.json'; ledger=r.read_json(ledger_path) if ledger_path.exists() else {'charged_bytes':0,'attempts':{},'receipts':{}}
  progress_path=STATE/'progress.json'; progress=r.read_json(progress_path) if progress_path.exists() else {'completed':{},'attempts':{}}
  progress.setdefault('completed',{}); progress.setdefault('attempts',{})
  def guard(growth=0):
   if time.monotonic()>deadline: raise TimeoutError('30-minute supplement budget reached')
   budget.check(growth+r.store_bytes(DEST))
  def save(status,**kwargs):
   progress.update(status=status,updated_utc=pd.Timestamp.now(tz='UTC').isoformat(),**kwargs); atomic_json(progress_path,progress)
  try:
   # PV is already local and verified; prohibit any additional month retrieval.
   pv=r.read_json(r.STATE/'pv-receipts.json')['5_minutely/year=2021/month=04/data.parquet']
   assert file_hash(Path(pv['path']))==pv['sha256']
   source=cfg['satellite_sources']['2021']['url']
   for key in sorted(required):
    guard(16*1024**2); path=r.CACHE/'objects'/key
    if path.exists():
     content=path.read_bytes()
     if len(content)>=16 and int.from_bytes(content[12:16],'little')==len(content):
      if objects[key]['status']=='cached': assert file_hash(path)==objects[key]['sha256']
      continue
     raise ValueError('Existing invalid block requires separate repair: '+key)
    info=objects[key]; assert info['status']=='sized' and info['bytes']<=16*1024**2 and info['generation']
    if ledger['attempts'].get(key,0)>=2: raise RuntimeError('Per-object retry limit '+key)
    integrity=r.read_json(r.CACHE/'integrity-failures.json') if (r.CACHE/'integrity-failures.json').exists() else {}
    if integrity.get(key,0)>=3: raise RuntimeError('Historical integrity limit '+key)
    while ledger['attempts'].get(key,0)<2:
     guard(16*1024**2)
     if ledger['charged_bytes']+16*1024**2>plan['new_transfer_limit']: raise ValueError('Supplement transfer cap')
     ledger['charged_bytes']+=16*1024**2; ledger['attempts'][key]=ledger['attempts'].get(key,0)+1; atomic_json(ledger_path,ledger)
     owner=r.read_json(r.CACHE/'owner.json'); owner['charged_bytes']+=16*1024**2; atomic_json(r.CACHE/'owner.json',owner)
     try:
      request=Request(source+'/'+key+'?generation='+info['generation'])
      with urlopen(request,timeout=30) as response: content=response.read(16*1024**2+1)
      assert len(content)==info['bytes'] and int.from_bytes(content[12:16],'little')==len(content)
      crc=[x[7:] for x in (info['hash'] or '').split(',') if x.startswith('crc32c=')]
      assert crc and base64.b64encode(google_crc32c.Checksum(content).digest()).decode()==crc[0]
      break
     except Exception as error:
      ledger.setdefault('failures',{})[key]=str(error); atomic_json(ledger_path,ledger)
      if ledger['attempts'][key]>=2: raise
    # Evict only owned, non-required old data blobs; keep metadata and pinned batch.
    existing=[p for p in (r.CACHE/'objects').rglob('*') if p.is_file()]; size=sum(p.stat().st_size for p in existing)
    for candidate in sorted(existing,key=lambda p:p.stat().st_mtime_ns):
     if size+len(content)<=4*1024**3: break
     rel=candidate.relative_to(r.CACHE/'objects').as_posix()
     if not rel.startswith('data/') or rel in required: continue
     assert candidate.resolve().is_relative_to((r.CACHE/'objects/data').resolve())
     size-=candidate.stat().st_size; candidate.unlink()
    assert size+len(content)<=4*1024**3
    path.parent.mkdir(parents=True,exist_ok=True); temporary=path.with_name(path.name+'.part'); temporary.write_bytes(content); temporary.replace(path)
    owner=r.read_json(r.CACHE/'owner.json'); owner['charged_bytes']-=16*1024**2-len(content); atomic_json(r.CACHE/'owner.json',owner)
    ledger['charged_bytes']-=16*1024**2-len(content); ledger['receipts'][key]={'bytes':len(content),'sha256':file_hash(path),'generation':info['generation'],'crc32c':crc[0]}; atomic_json(ledger_path,ledger)
    save('downloading',current=key,charged_bytes=ledger['charged_bytes']); print('DOWNLOADED',key,len(content),flush=True)
   client=OfflineStore(source,cache_dir=r.CACHE,max_attempts=0)
   assert client.metadata_sha256==cfg['satellite_sources']['2021']['metadata_sha256']
   builder=r.load_builder()
   for tid in chosen:
    guard(128*1024**2); output=DEST/tid
    if (output/'verified.json').exists(): r.verify_shard(output); continue
    if output.exists(): raise RuntimeError('Incomplete supplement retained for inspection: '+tid)
    prior=old.get('attempts',{}).get(tid,0)+progress['attempts'].get(tid,0)
    if prior>=3: raise RuntimeError('Historical partition attempt cap '+tid)
    progress['attempts'][tid]=progress['attempts'].get(tid,0)+1; save('building',current=tid)
    task=tasks[tid]; output.parent.mkdir(parents=True,exist_ok=True)
    builder.build(SimpleNamespace(preflight=r.PREFLIGHT,region=task['region'],window='spring',start=task['start'],end=task['end'],issue_start=task['issue_start'],issue_end=task['issue_end'],cache=r.CACHE,output=output,client=client,local_sources_only=True,data_role='spring_supplement_development',guard=lambda:guard(16*1024**2)))
    r.seal_shard(output,tid); progress['completed'][tid]={'receipt_sha256':file_hash(output/'verified.json'),'samples':r.read_json(output/'build_report.json')['samples']}; save('building',current=tid); print('VERIFIED',tid,flush=True)
   for tid in chosen: r.verify_shard(DEST/tid)
   save('batch_complete_stopped_budget_remainder' if execution['unselected_partitions'] else 'complete_stopped',elapsed_seconds=time.monotonic()-start,new_transfer_bytes=ledger['charged_bytes'],remaining_partitions=execution['unselected_partitions'],training_started=False,final_labels_read=False)
  except Exception as error:
   save('stopped_incomplete',error=repr(error),elapsed_seconds=time.monotonic()-start); raise
if __name__=='__main__': main()
