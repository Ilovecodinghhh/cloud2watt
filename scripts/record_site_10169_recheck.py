from pathlib import Path
import pandas as pd
from cloud2watt.run_state import atomic_json
import json
root=Path('D:/code/AI_atm'); out=root/'outputs/project-v2/site-10169-recheck'
result=json.loads((out/'result.json').read_text())
receipt=json.loads((root/'outputs/project-v2/route-b-download/pv-receipts.json').read_text())['5_minutely/year=2021/month=04/data.parquet']
p=pd.read_parquet(receipt['path'],filters=[('ss_id','==',10169)])
p.datetime_GMT=pd.to_datetime(p.datetime_GMT,utc=True)
result['five_minute_dates']=p.groupby(p.datetime_GMT.dt.strftime('%Y-%m-%d')).size().to_dict()
atomic_json(out/'result.json',result)
report=f"""# 站点 10169 补充重试结果

已完成源端复核与备用分辨率下载核查，未能补回 2021-04-07 至 2021-04-20 的记录。

五分钟源端最新版本仍为 {result['upstream_revision']}，对象大小及 SHA256 与本地完全一致。独立读取整份五分钟文件后筛选，目标窗口为 0 条，排除 parquet 下推筛选造成假缺测。该站四月已有记录日期及数量：{result['five_minute_dates']}。

事前登记同源三十分钟对象，按每响应最多 8 MiB 的 Range 请求下载整个 35,553,908 字节文件并核验源端 SHA256。该站只有 4 月 29 日 47 条、4 月 30 日 48 条，目标窗口仍为 0 条。备用文件隔离保存，未混入五分钟数据或训练。项目传输账本保留了 142,215,632 字节保守预算预留（实际载荷 35,553,908 字节）。

结论是当前同源两种分辨率均无法补齐指定日期，而不是下载失败。缺测继续保留；没有替换站点、日期或生成插值目标，未训练、未读取最终标签。

机器证据：outputs/project-v2/site-10169-recheck/result.json、alternate-plan.json、alternate-transfer.json。
"""
(root/'docs/PROJECT_V2_SITE_10169_RECHECK.md').write_text(report,encoding='utf-8')
with (root/'docs/PROJECT_V2_EXECUTION.md').open('a',encoding='utf-8') as f:
 f.write('\n## 站点 10169 重试补充结束\n\n源端五分钟对象与本地 SHA256 一致；另外登记并按有界 Range 下载同源三十分钟四月文件，核验 SHA256 后仍确认 4 月 7–20 日零记录。无法补齐，保留源缺测；没有训练或更改划分。详见 PROJECT_V2_SITE_10169_RECHECK.md。\n')
print(result['five_minute_dates'])
