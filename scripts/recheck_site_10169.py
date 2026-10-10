"""Bounded local/source metadata recheck; no final labels, no training."""
import os
os.environ['HF_HUB_DISABLE_XET']='1'
from pathlib import Path
import json
import pandas as pd
from huggingface_hub import HfApi,hf_hub_url,get_hf_file_metadata
from cloud2watt.run_state import atomic_json,file_hash
ROOT=Path('D:/code/AI_atm'); OUT=ROOT/'outputs/project-v2/site-10169-recheck'; OUT.mkdir(exist_ok=True)
name='5_minutely/year=2021/month=04/data.parquet'
receipt=json.loads((ROOT/'outputs/project-v2/route-b-download/pv-receipts.json').read_text())[name]
probe=json.loads((ROOT/'outputs/probes/uk_pv.json').read_text())['details']
atomic_json(OUT/'plan.json',{'site':'10169','dates':['2021-04-07','2021-04-21'],'objects':[name],'actions':['verify local SHA256','inspect site rows','check upstream revision and object identity'],'maximum_payload_bytes':0,'final_labels_read':False,'training':False})
path=Path(receipt['path']); assert file_hash(path)==receipt['sha256']
# Independent full file read avoids treating a pushdown-filter issue as source absence.
power=pd.read_parquet(path,columns=['ss_id','datetime_GMT','generation_Wh']); station=power.loc[power.ss_id.astype(str).eq('10169')].copy()
station.datetime_GMT=pd.to_datetime(station.datetime_GMT,utc=True)
meta=pd.read_csv(probe['metadata']['path'],dtype={'ss_id':str}); bad=pd.read_csv(probe['bad_data']['path'],dtype={'ss_id':str})
result={'site':'10169','local_sha256':receipt['sha256'],'april_total_file_rows':len(power),'april_site_rows':len(station),'requested_window_rows':int((station.datetime_GMT.ge('2021-04-07T00:00Z')&station.datetime_GMT.lt('2021-04-21T00:00Z')).sum()),'metadata':meta.loc[meta.ss_id.eq('10169')].fillna('').to_dict('records'),'source_bad_periods':bad.loc[bad.ss_id.eq('10169')].fillna('').to_dict('records'),'network_payload_downloaded_bytes':0,'final_labels_read':False,'training_started':False}
try:
 api=HfApi(); info=api.dataset_info('openclimatefix/uk_pv',files_metadata=True,timeout=30)
 result['upstream_revision']=info.sha
 remote=get_hf_file_metadata(hf_hub_url('openclimatefix/uk_pv',name,repo_type='dataset',revision=info.sha),timeout=30)
 result['upstream_object']={'etag':remote.etag,'size':remote.size,'commit':remote.commit_hash}
 result['remote_identical_to_local']=remote.etag.strip('"')==receipt['sha256'] and remote.size==receipt['bytes']
 result['related_april_paths']=[x.rfilename for x in info.siblings if 'year=2021/month=04/' in x.rfilename]
 result['status']='upstream_unchanged_source_absence' if result['remote_identical_to_local'] else 'upstream_changed_requires_bounded_payload_check'
except Exception as e:
 result['status']='upstream_check_failed'; result['error_type']=type(e).__name__
atomic_json(OUT/'result.json',result); print(json.dumps(result,ensure_ascii=False))
