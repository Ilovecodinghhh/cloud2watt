"""Independent acceptance and paired date-block summary of fixed ablations."""
import json
from pathlib import Path
import numpy as np,pandas as pd,torch
import train_local_gpu as t
from run_satellite_ablation import OUT,ROLL,FOLDS,SEEDS,load,inputs,read
from cloud2watt.satellite_ablation import VARIANTS,make_satellite,fit_transform,apply_transform

def ci(differences,mask,dates):
 unique=np.unique(dates);sums=np.array([(differences[dates==day]*mask[dates==day]).sum(0) for day in unique]);counts=np.array([mask[dates==day].sum(0) for day in unique]);rng=np.random.default_rng(813);take=rng.integers(len(unique),size=(10000,len(unique)));den=counts[take].sum(1);good=(den>0).all(1);values=(sums[take].sum(1)[good]/den[good]).mean(1)
 return {'delta_mae':float((sums.sum(0)/counts.sum(0)).mean()),'ci95':np.quantile(values,[.025,.975]).tolist(),'ci98_75_four_comparison_adjustment':np.quantile(values,[.00625,.99375]).tolist(),'dates':len(unique),'draws':len(values),'negative_means_variant_better':True,'unit':'whole UTC date, sites and seed pairs kept together','scope':'exploratory; overlapping fitted folds and weather-day dependence not eliminated'}

def main():
 assert read(OUT/'status.json')['stage'] in {'training_complete','complete'}
 torch.set_num_threads(4);assert torch.cuda.is_available() and 'raincalib-gpu' in t.sys.executable
 frame=pd.read_parquet(OUT/'frame-features.parquet').set_index(['site','time']);models=['smart_persistence','power_solar','satellite_stats',*VARIANTS];all_errors={k:[] for k in models};all_masks=[];all_dates=[];per_fold={};winter={};receipts=[]
 for fold in FOLDS:
  d=load(fold);local,elev=inputs(d,frame);fit=d['role']=='fit';outer=d['role']=='spring_outer_development';day=d['mask'][outer][:,[1,2,3]]&(d['elevation'][outer][:,[1,2,3]]>5);dates=np.array([str(x)[:10] for x in d['time'][outer]]);all_masks.append(day);all_dates.append(dates);per_fold[fold]={};winter[fold]={}
  for model in models:
   errors=[];winter_errors=[];seed_scores=[]
   for seed in SEEDS:
    if model in VARIANTS:
     folder=OUT/fold/(model+'-'+str(seed));predfile=folder/'predictions.npz';dayproc,robust,small=VARIANTS[model];sat=make_satellite(d['sat'],local,elev,daylight=dayproc,small=small);x,params=fit_transform(d['base'],sat,fit,robust=robust)
     done=read(folder/'complete.json');assert done['epochs']==20 and t.file_hash(predfile)==done['prediction_sha256'] and t.file_hash(folder/'best.pt')==done['checkpoint_sha256']
     c=torch.load(folder/'best.pt',weights_only=False)
     assert read(folder/'preregistration.json')['parent_sha256']==t.file_hash(OUT/'preregistration.json')
     assert c['params']['fit_only'] and c['params']['satellite_clip']==params['satellite_clip']
     np.testing.assert_array_equal(apply_transform(d['base'],sat,c['params']),x)
     np.testing.assert_array_equal(c['params']['center'],params['center']);np.testing.assert_array_equal(c['params']['scale'],params['scale']);assert c['params']['robust']==params['robust']
     assert c['inner_score']==min(v['inner_daylight_primary_mae'] for v in read(folder/'history.json'))
     net=t.model(132);net.load_state_dict(c['model']);net.eval();ids=np.flatnonzero(outer)[::97]
     with torch.no_grad():replay=net(torch.tensor(x[ids],device='cuda')).cpu().numpy()
     receipts.append({'fold':fold,'variant':model,'seed':seed,'checkpoint_sha256':done['checkpoint_sha256']})
    else:predfile=ROLL/fold/str(seed)/(model+'-predictions.npz')
    with np.load(predfile) as z:
     for key in ['site','time','role','mask']:np.testing.assert_array_equal(z[key],d[key])
     np.testing.assert_array_equal(z['target'],d['y']);pred=z['prediction'];assert np.isfinite(pred).all()
     if model in VARIANTS:np.testing.assert_allclose(pred[ids],replay,rtol=1e-4,atol=2e-6)
    e=abs(pred[outer][:,[1,2,3]]-d['y'][outer][:,[1,2,3]]);errors.append(e);seed_scores.append({'seed':seed,'mae':float(((e*day).sum(0)/day.sum(0)).mean())})
    wi=d['role']=='outer_development';wd=d['mask'][wi][:,[1,2,3]]&(d['elevation'][wi][:,[1,2,3]]>5);we=abs(pred[wi][:,[1,2,3]]-d['y'][wi][:,[1,2,3]]);winter_errors.append(we)
   avg=np.mean(errors,axis=0);all_errors[model].append(avg);per_fold[fold][model]={'mae':float(((avg*day).sum(0)/day.sum(0)).mean()),'seed_scores':seed_scores}
   winter[fold][model]=float(((np.mean(winter_errors,axis=0)*wd).sum(0)/wd.sum(0)).mean())
 mask=np.concatenate(all_masks);dates=np.concatenate(all_dates);errors={k:np.concatenate(v) for k,v in all_errors.items()};overall={k:float(((v*mask).sum(0)/mask.sum(0)).mean()) for k,v in errors.items()}
 comparisons={variant:ci(errors[variant]-errors['satellite_stats'],mask,dates) for variant in VARIANTS}
 result={'overall_spring_daylight_mae':overall,'per_fold':per_fold,'winter_same_model_per_fold':winter,'paired_vs_original_satellite':comparisons,'evaluation_dates':sorted(np.unique(dates).tolist()),'evaluation_samples':len(dates),'final_labels_read':False,'network_requests':0,'models_replaced':False,'training_runs':36,'epochs_each':20}
 t.atomic_json(OUT/'results.json',result);t.atomic_json(OUT/'acceptance.json',{'passed':True,'checkpoints':receipts,'fit_only_transforms':True,'inner_only_selection':True,'identical_evaluation_keys_targets_masks':True,'saved_prediction_replay':True,'results_sha256':t.file_hash(OUT/'results.json')})
 print(json.dumps({'overall':overall,'winter_extended':winter['extended'],'comparisons':comparisons}))
if __name__=='__main__':main()
