# ADR-004：无泄漏评估与测试集锁定

- 状态：Accepted
- 日期：2026-09-04

2026-09-10：保留为 V1 历史决策。未来测试角色与选择协议由
[ADR-005](005-evaluation-v2-and-test-history.md) 和 [评估协议 V2](../EVALUATION_PROTOCOL_V2.md)
取代；完整支持区间与 embargo 原则继续保留。旧 test 已在基线阶段评估及复现。

## 决策

主评估采用连续时间 70%/15%/15% 切分。一个样本的完整物理信息区间定义为
`[issue−45min, issue+240min]`；只有完整区间落在同一 split 内才分配，train/validation 和
validation/test 之间另设 4 小时 embargo。边界对齐到 15 分钟网格，所有引用行均进行跨
split 重用检查。

另以固定 seed 42，根据容量、纬度和有效目标率分层选择 20% 站点作为空间 holdout。
Climatology 只拟合 train/development 引用到的有效功率记录。测试集默认锁定；普通命令只
生成 validation 结果，冻结配置后必须显式传入 `--unlock-test` 才能生成最终 test 报告。

统一报告 Persistence、Smart Persistence 和 Site-Time Climatology 的 masked MAE、RMSE、
nMAE、相对 Persistence skill score，以及 30 分钟上升/下降爬坡的 Precision、Recall 和
F1。爬坡事件按站点独立进行 ±15 分钟一对一匹配。

## 理由

仅按 issue time 切分仍可能让历史输入或四小时目标跨越边界。使用完整信息区间和 embargo
可以直接阻断这种泄漏。显式测试锁避免在阈值和模型选择期间反复查看 test；空间 holdout
则区分时间泛化和未见站点泛化。

## 后果

embargo 和历史质量控制会减少有效样本。30 天冬季窗口只能验证管道，不能支持全年泛化
结论。Climatology 对未见站点只能退化为训练站点全局均值，因此空间 holdout 结果应与
时间切分结果分开解释。
