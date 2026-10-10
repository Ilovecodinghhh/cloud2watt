"""Summarize rolling development errors without changing model selection."""
import json
import numpy as np
from train_rolling_spring import OUT,ROOT,FOLDS,SEEDS
from validate_local_mvp import blocked_ci
from cloud2watt.run_state import atomic_json

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
 assert read(OUT/'independent-acceptance.json')['passed']
 models=['persistence','smart_persistence','power_solar','satellite_stats']
 errors={k:[] for k in models}; masks=[]; dates=[]; per_fold={}
 for fold in FOLDS:
  with np.load(OUT/fold/'dataset.npz') as z:
   take=z['role']=='spring_outer_development'; y=z['y'][take][:,[1,2,3]]; mask=z['mask'][take][:,[1,2,3]]&(z['elevation'][take][:,[1,2,3]]>5); dt=np.array([str(x)[:10] for x in z['time'][take]])
  masks.append(mask); dates.append(dt); per_fold[fold]={}
  for kind in models:
   all_seeds=[]
   for seed in SEEDS:
    with np.load(OUT/fold/str(seed)/(kind+'-predictions.npz')) as z: e=abs(z['prediction'][take][:,[1,2,3]]-y)
    all_seeds.append(e)
   e=np.mean(all_seeds,axis=0); errors[kind].append(e)
   per_fold[fold][kind]=float(((e*mask).sum(0)/mask.sum(0)).mean())
 mask=np.concatenate(masks); dates=np.concatenate(dates); errors={k:np.concatenate(v) for k,v in errors.items()}
 overall={k:float(((v*mask).sum(0)/mask.sum(0)).mean()) for k,v in errors.items()}
 comparisons={k:blocked_ci(errors[k]-errors['smart_persistence'],mask,dates) for k in ['power_solar','satellite_stats']}
 for v in comparisons.values():
  v.pop('negative_means_satellite_better',None); v['negative_means_model_better_than_smart_persistence']=True
 result={'overall_daylight_primary_mae':overall,'per_fold':per_fold,'vs_smart_persistence':comparisons,'daylight_dates':sorted(np.unique(dates[mask.any(1)]).tolist()),'scope':'exploratory rolling development; dates already used for development; correlated adjacent days and overlapping training folds; conditional on these seeds; not final-test significance'}
 atomic_json(OUT/'summary.json',result)
 lines=['# 春季扩展滚动验证结果','', '使用本地数据与指定 Conda RTX 5070 Ti GPU，三个窗口、三个种子、两种模型，共 18 次训练，每次 20 轮。最终测试标签未读取，旧 MVP 默认模型未替换。','', '| 验证日期 | 持续性 | 晴空修正 | 功率＋太阳 | 卫星增强 |','|---|---:|---:|---:|---:|']
 for fold,v in per_fold.items():
  bounds=FOLDS[fold]['spring_outer_development']; lines.append('| '+str(bounds)+'（右端不含） | '+' | '.join(f'{100*v[k]:.2f}%' for k in models)+' |')
 lines+=['| 合并 12 日 | '+' | '.join(f'{100*overall[k]:.2f}%' for k in models)+' |','','指标：白天未来 30/60/120 分钟 MAE，占装机容量比例；先平均每个种子的绝对误差，再在每个时效上合并样本，最后平均三个时效。不是对预测做集成后打分。','', '配对日期块 bootstrap（10,000 次）记录在 results.json 和 summary.json。三个种子不视作独立天气样本；日期已用于开发，邻近日期可能相关，滚动窗口训练集有重叠，因此置信区间仅作探索性参考。','', '边界修复：重建候选样本，组内重算上下文 QC，逐样本检查历史 −60 分钟和目标 +240 分钟；跨缺测时间间隔不拼接连续零段。QC 仍为离线组内检查，不是实时因果 QC。','', '验收：8 项边界与统计测试通过；18 个检查点均通过训练集归一化、内部验证最优轮次、保存预测复现、目标与掩码一致性及时间边界检查。','']
 report=ROOT/'docs/PROJECT_V2_ROLLING_SPRING_RESULTS.md'; report.write_text('\n'.join(lines),encoding='utf-8')
 print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
