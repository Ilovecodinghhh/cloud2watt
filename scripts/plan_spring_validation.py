"""Register exact spring objects before any new payload transfer; HEAD sizing only."""
import json,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from urllib.request import Request,urlopen
import pandas as pd
from cloud2watt.run_state import atomic_json,file_hash
ROOT=Path('D:/code/AI_atm'); OUT=ROOT/'outputs/project-v2/spring-validation-20261009'; CACHE=ROOT/'outputs/project-v2/source-cache-2021/objects'
def main():
 OUT.mkdir(exist_ok=True)
 inv=json.loads((ROOT/'outputs/project-v2/route-b-download/source-inventory.json').read_text()); freeze=json.loads((ROOT/'configs/project-v2/route-b-freeze.json').read_text())
 tasks={k:v for k,v in inv['development_by_partition'].items() if k.startswith('spring/') and '20210421'<=k.split('/')[1]<'20210505'}
 url=freeze['satellite_sources']['2021']['url']; keys=sorted({k for v in tasks.values() for k in v}); objects={}
 for key in keys:
  path=CACHE/key
  if path.exists():
   with path.open('rb') as stream: head=stream.read(16)
   valid=len(head)==16 and int.from_bytes(head[12:16],'little')==path.stat().st_size
   if valid: objects[key]={'status':'cached','bytes':path.stat().st_size,'sha256':file_hash(path)}
  if key not in objects: objects[key]={'status':'head_pending','url':url+'/'+key}
 plan={'dates':['2021-04-21','2021-05-05'],'date_end_exclusive':True,'regions':[r for r in freeze['regions'] if r['role']=='development'],'channels':['VIS006','IR_016','IR_108'],'partition_objects':tasks,'objects':objects,'new_transfer_limit':4294967296,'maximum_wall_seconds':10800,'training_authorized':False,'final_labels_locked':True,'status':'registered_before_transfer','scope':'authorized 14 new validation dates, no training','proposed_future_roles':{'prospective_validation':['2021-04-21','2021-05-05']},'role_note':'reserve for validation; no fitting, tuning, or scoring in this download task'}
 atomic_json(OUT/'plan.json',plan)
 def size(key):
  for attempt in range(2):
   try:
    with urlopen(Request(url+'/'+key,method='HEAD'),timeout=15) as response:
     return key,{'status':'sized','bytes':int(response.headers['Content-Length']),'generation':response.headers.get('x-goog-generation'),'hash':response.headers.get('x-goog-hash'),'head_attempts':attempt+1}
   except Exception as e: error=str(e)
  return key,{'status':'head_failed','error':error,'head_attempts':2}
 pending=[k for k,v in objects.items() if v['status']=='head_pending']
 with ThreadPoolExecutor(max_workers=12) as pool:
  for i,future in enumerate(as_completed([pool.submit(size,k) for k in pending]),1):
   key,value=future.result(); objects[key].update(value)
   if i%50==0: atomic_json(OUT/'plan.json',plan); print('HEAD',i,len(pending),flush=True)
 plan['status']='sized' if all(v['status'] in ['cached','sized'] for v in objects.values()) else 'head_partial'
 plan['new_payload_bytes']=sum(v.get('bytes',0) for v in objects.values() if v['status']=='sized')
 atomic_json(OUT/'plan.json',plan)
 print(json.dumps({'status':plan['status'],'objects':len(objects),'new_payload_bytes':plan['new_payload_bytes']}),flush=True)
if __name__=='__main__': main()
