# ADR-006：标签时间支持、SEVIRI 投影与共享 tile

日期：2026-09-28。状态：工程决策已实施；新正式数据标签契约待定。

## 背景与证据

S2 核验固定 UK PV revision 02f17566e1b1a56dc8fd25cfabec67f065066e07 的
[数据卡](https://huggingface.co/datasets/openclimatefix/uk_pv/blob/02f17566e1b1a56dc8fd25cfabec67f065066e07/README.md)，
本地取回文件 SHA-256 为 fab51b6f5e2c75db8d0841fd50872317dea0ba11df5ecd3e0f38ed139d3e50b4。
数据卡把 5 分钟数据描述为瞬时读数，把 30 分钟数据描述为仪表积分；字段段落又给出
generation_Wh、右边界时间戳及 5 分钟值乘 12 的 W 换算。

因此，当前算术 4 × sum(三个读数) 等于 mean(12 × 三个读数)，尺度有依据，
但不能推出这三个读数是真实区间能量。S2 对 3,770 个完整 bin 重建的最大误差为 0。
另取同 revision 的单个 20.58 MB 半小时分区，在 431 个大于 1% 容量的配对日间 bin 上，
五分钟代理聚合与仪表半小时积分的归一化 MAE 为 0.007787，能量总量比 0.98249。
这是量级交叉核对，不足以确定时间支持。

SEVIRI 源 metadata 中的 geos 投影没有指定 sweep，默认是 y；
[PROJ 官方说明](https://proj.org/en/stable/operations/projections/geos.html)也明确 Meteosat 使用 y。
旧实现写成 x。五站点抽查发现新旧中心相差 2 个 x 像素，部分另差 1 个 y 像素。
详细坐标与像素尺度见 [审计记录](../audits/s2-data-audit-2026-09-28.json)。

## 决策

1. 将后续源数据构建投影修正为 sweep=y，并在新卫星存储属性中记录完整 PROJ 字符串。
   不覆盖 paired-30d-v1，也不把历史模型结论改称修正投影后的结果。
2. 保留旧数值换算供历史重现。新正式数据不能继续无条件宣称“真实 15 分钟平均功率”。
   推荐下一版明确使用“三个 5 分钟采样功率的均值代理”；若研究必须针对真实能量积分，
   应改用半小时原始数据并另定标签协议。该选择改变研究目标，须在 S3 建数前确认并升版契约。
3. 共享 tile 迁移保持历史像素坐标，逐像素检验存储等价；地理校正必须通过新的源数据构建完成。
4. observation_time 记录归档坐标时刻；available_time 在没有真实发布记录时只能是情景值。
   0/15/30 分钟延迟分别显式标记 scenario，不能当作实测 SLA。严格选择可用的历史槽，缺失不取未来帧。
5. 当前只做有界试验。标签契约未确认前停止大规模构建；预算按实际区域重叠估算，不能按站点数
   直接套用当前紧凑区域的压缩收益。

## 影响与边界

V2 主评分与恢复逻辑不变，但新源码身份变化，旧运行不能自动作为新源码的续训起点。
历史预测仍可用于回归。修正了 sweep 不等于验证了卫星图像绝对定位：站点坐标本身约公里级模糊，
归档产品也仍需海岸线/地标抽查才能作更强的定位声明。

卫星年度 attrs 的 start/end_time 是年初示例，不能用来推导每帧每像素扫描或发布时刻。
当前延迟回放验证输入可用率，没有运行正式模型精度回测；后续模型对照应在一致样本上补做。

## 待用户发送的上游询问材料

以下仅为草稿，本任务未向任何外部人员发送消息：

> We are using UK PV revision 02f17566e1b1a56dc8fd25cfabec67f065066e07.
> For the five-minute series, does generation_Wh encode an instantaneous AC power sample
> scaled by 1/12, a five-minute average, or meter-integrated energy? Should datetime_GMT
> be interpreted as the sample instant or the end of an averaging/integration interval?
> We apply the documented factor of 12 and need to distinguish a sampled-power proxy
> from a true 15-minute mean. Is per-record publication latency available?

卫星补充问题：OCF v4 年度 time 坐标代表扫描起点、终点还是取整后的标称时刻？是否有逐帧发布记录，
以及 non-HRV v4 的地理定位修订记录和再分发许可映射？
