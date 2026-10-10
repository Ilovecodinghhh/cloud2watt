"""Finish the specifically authorized 13 spring days, then exit; no training."""
import os,sys,time,json,base64,shutil
from pathlib import Path
from urllib.request import Request,urlopen
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from types import SimpleNamespace
import importlib.util
import google_crc32c
import pandas as pd
import psutil
from filelock import FileLock
from cloud2watt.run_state import atomic_json,file_hash
ROOT=Path('D:/code/AI_atm'); STATE=ROOT/'outputs/project-v2/spring-supplement-20261008'; DEST=ROOT/'data/processed/route-b-spring-supplement-v1'
spec=importlib.util.spec_from_file_location('batch',ROOT/'scripts/supplement_spring.py'); batch=importlib.util.module_from_spec(spec); spec.loader.exec_module(batch); r=batch.r
CAP=3*1024**3; RESPONSE=16*1024**2

def checked_payload(payload,info):
 if len(payload)!=info['bytes'] or len(payload)>RESPONSE or len(payload)<16 or int.from_bytes(payload[12:16],'little')!=len(payload): raise ValueError('payload length/header mismatch')
 checks=[x.strip()[7:] for x in (info['hash'] or '').split(',') if x.strip().startswith('crc32c=')]
 if not checks or base64.b64encode(google_crc32c.Checksum(payload).digest()).decode()!=checks[0]: raise ValueError('CRC32C mismatch')
 return checks[0]

def main():
 assert 'raincalib-gpu' in sys.executable
 os.chdir(ROOT); started=time.monotonic(); deadline=started+10800
 with FileLock(r.STATE/'writer.lock',timeout=0),FileLock(STATE/'writer.lock',timeout=0):
  plan=r.read_json(STATE/'plan.json'); assert plan['status']=='sized'
  auth=r.read_json(STATE/'completion-authorization.json'); assert auth['transfer_cap_bytes']==CAP
  cfg=r.read_json(r.FREEZE); budget=r.Budget(cfg); owner=r.read_json(r.CACHE/'owner.json')
  ledger=r.read_json(STATE/'transfer.json'); ledger_path=STATE/'transfer.json'
  completed_before=r.read_json(STATE/'progress.json'); previous=r.read_json(STATE/'completion-progress.json') if (STATE/'completion-progress.json').exists() else {}
  progress={'completed':{**completed_before['completed'],**previous.get('completed',{})},'attempts':previous.get('attempts',{}),'pid':os.getpid(),'process_created':psutil.Process().create_time()}
  tasks={t['id']:t for t in r.tasks_for(cfg)}; selected=sorted(k for k in plan['partition_objects'] if '20210408'<=k.split('/')[1]<'20210421'); assert len(selected)==39
  required={k for tid in plan['partition_objects'] for k in plan['partition_objects'][tid]}; objects=plan['objects']
  # Reserve space once, then enforce the conservative bound on every write.
  baseline_usage=budget.check(512*1024**2); baseline_dest=r.store_bytes(DEST)
  def guard(growth=0):
   if time.monotonic()>deadline: raise TimeoutError('completion 3-hour wall limit reached')
   disk=shutil.disk_usage(ROOT)
   if disk.free-growth<disk.total*.2: raise ValueError('disk reserve')
   # Cache cannot grow over 4 GiB; allow the full extra 3 GiB conservatively.
   if baseline_usage+CAP+r.store_bytes(DEST)+growth>16*1024**3: raise ValueError('persistent budget')
   if owner['charged_bytes']+budget.state['pv_reserved_bytes']+budget.state.get('final_reserved_bytes',0)+growth>100*1024**3: raise ValueError('global source budget')
  def save(status,**fields):
   progress.update(status=status,updated_utc=pd.Timestamp.now(tz='UTC').isoformat(),elapsed_seconds=time.monotonic()-started,charged_bytes=ledger['charged_bytes'],training_started=False,final_labels_read=False,**fields); atomic_json(STATE/'completion-progress.json',progress)
  def reserve(key):
   guard(RESPONSE)
   if ledger['charged_bytes']+RESPONSE>CAP: raise ValueError('3 GiB completion cap')
   if ledger['attempts'].get(key,0)>=2: raise ValueError('retry limit '+key)
   ledger['charged_bytes']+=RESPONSE; ledger['attempts'][key]=ledger['attempts'].get(key,0)+1; atomic_json(ledger_path,ledger)
   owner['charged_bytes']+=RESPONSE; atomic_json(r.CACHE/'owner.json',owner)
  source=cfg['satellite_sources']['2021']['url']
  def fetch(key):
   info=objects[key]; assert info.get('generation') and info['bytes']<=RESPONSE
   with urlopen(Request(source+'/'+key+'?generation='+info['generation']),timeout=45) as response: payload=response.read(RESPONSE+1)
   return payload,checked_payload(payload,info)
  try:
   pv=r.read_json(r.STATE/'pv-receipts.json')['5_minutely/year=2021/month=04/data.parquet']; assert file_hash(Path(pv['path']))==pv['sha256']
   for tid,item in progress['completed'].items():
    if tid in plan['partition_objects']:
     r.verify_shard(DEST/tid); assert file_hash(DEST/tid/'verified.json')==item['receipt_sha256']
   needed=[]
   integrity=r.read_json(r.CACHE/'integrity-failures.json') if (r.CACHE/'integrity-failures.json').exists() else {}
   for key in sorted(required):
    path=r.CACHE/'objects'/key
    if path.exists():
     expected=ledger['receipts'].get(key,objects[key]).get('sha256')
     if not expected or file_hash(path)!=expected: raise ValueError('cached hash mismatch '+key)
    else:
     if key in ledger['receipts']: raise ValueError('verified required source disappeared; refuse silent redownload '+key)
     if objects[key]['status']!='sized' or integrity.get(key,0)>=3: raise ValueError('unavailable source '+key)
     needed.append(key)
   atomic_json(STATE/'completion-execution.json',{'partitions':selected,'source_keys':sorted(required),'new_objects':needed,'expected_new_bytes':sum(objects[k]['bytes'] for k in needed),'transfer_cap_including_previous_batch':CAP,'plan_sha256':file_hash(STATE/'plan.json'),'stop_after_complete':True})
   # Make room by evicting only old non-required cache blobs, never derived data.
   existing=[p for p in (r.CACHE/'objects').rglob('*') if p.is_file()]; cache_size=sum(p.stat().st_size for p in existing)
   needed_bytes=sum(objects[k]['bytes'] for k in needed)
   for candidate in sorted(existing,key=lambda p:p.stat().st_mtime_ns):
    if cache_size+needed_bytes<=4*1024**3: break
    rel=candidate.relative_to(r.CACHE/'objects').as_posix()
    if not rel.startswith('data/') or rel in required: continue
    assert candidate.resolve().is_relative_to((r.CACHE/'objects/data').resolve())
    cache_size-=candidate.stat().st_size; candidate.unlink()
   assert cache_size+needed_bytes<=4*1024**3
   save('downloading',objects_remaining=len(needed)); failures=[]
   with ThreadPoolExecutor(max_workers=8) as pool:
    pending={}; queue=list(needed)
    while queue or pending:
     while queue and len(pending)<8:
      key=queue.pop(0); reserve(key); pending[pool.submit(fetch,key)]=key
     done,_=wait(pending,return_when=FIRST_COMPLETED,timeout=5)
     for future in done:
      key=pending.pop(future)
      try: payload,crc=future.result()
      except Exception as error:
       ledger.setdefault('failures',{})[key]=repr(error); atomic_json(ledger_path,ledger)
       if ledger['attempts'][key]<2: queue.append(key)
       else: failures.append(key)
       continue
      guard(len(payload)); path=r.CACHE/'objects'/key; path.parent.mkdir(parents=True,exist_ok=True)
      temp=path.with_name(path.name+'.part'); temp.write_bytes(payload); temp.replace(path)
      ledger['charged_bytes']-=RESPONSE-len(payload); owner['charged_bytes']-=RESPONSE-len(payload)
      ledger['receipts'][key]={'bytes':len(payload),'sha256':file_hash(path),'generation':objects[key]['generation'],'crc32c':crc}
      atomic_json(ledger_path,ledger); atomic_json(r.CACHE/'owner.json',owner)
      save('downloading',objects_remaining=len(queue)+len(pending),verified_source_receipts=len(ledger['receipts'])); print('VERIFIED SOURCE',key,flush=True)
   client=batch.OfflineStore(source,cache_dir=r.CACHE,max_attempts=0); builder=r.load_builder(); old=r.read_json(r.STATE/'progress.json')
   assert client.metadata_sha256==cfg['satellite_sources']['2021']['metadata_sha256']
   for tid in selected:
    if any(k in failures for k in plan['partition_objects'][tid]): continue
    guard(128*1024**2); output=DEST/tid
    if (output/'verified.json').exists():
     r.verify_shard(output); progress['completed'][tid]={'receipt_sha256':file_hash(output/'verified.json'),'samples':r.read_json(output/'build_report.json')['samples']}; continue
    if (output/'complete.json').exists():
     r.seal_shard(output,tid); progress['completed'][tid]={'receipt_sha256':file_hash(output/'verified.json'),'samples':r.read_json(output/'build_report.json')['samples']}; continue
    prior=old.get('attempts',{}).get(tid,0)+progress['attempts'].get(tid,0)
    if prior>=3: raise ValueError('partition attempt limit '+tid)
    if output.exists():
     archived=DEST/'failed'/tid/('completion-attempt-'+str(prior))
     assert output.resolve().is_relative_to(DEST.resolve()) and archived.resolve().is_relative_to(DEST.resolve()) and not archived.exists()
     archived.parent.mkdir(parents=True,exist_ok=True); output.rename(archived)
    progress['attempts'][tid]=progress['attempts'].get(tid,0)+1; save('building',current_partition=tid)
    task=tasks[tid]; output.parent.mkdir(parents=True,exist_ok=True)
    builder.build(SimpleNamespace(preflight=r.PREFLIGHT,region=task['region'],window='spring',start=task['start'],end=task['end'],issue_start=task['issue_start'],issue_end=task['issue_end'],cache=r.CACHE,output=output,client=client,local_sources_only=True,data_role='spring_supplement_development',guard=lambda:guard(RESPONSE)))
    r.seal_shard(output,tid); progress['completed'][tid]={'receipt_sha256':file_hash(output/'verified.json'),'samples':r.read_json(output/'build_report.json')['samples']}; save('building',current_partition=tid); print('VERIFIED PARTITION',tid,flush=True)
   missing=[tid for tid in selected if tid not in progress['completed']]
   save('complete_stopped' if not missing else 'partial_failed_stopped',remaining_partitions=missing,failed_sources=failures)
  except Exception as error:
   save('stopped_incomplete',error=repr(error)); raise
if __name__=='__main__': main()
