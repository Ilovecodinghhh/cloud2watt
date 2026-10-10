# Power-only 模型实验报告

状态：V1 三种子 validation 实验已完成；这六组运行未输出 test 评估，旧基线 test 已曝光
最后更新：2026-09-05

2026-09-28 补注：[S1 已实现统一 V2 训练与恢复](S1_CORRECTNESS_REPORT.md)。
下文命令与实验数字为 V1 历史记录；当前入口拒绝 test 解锁，旧 checkpoint 不支持 V2 恢复。

2026-09-10 S0 补注：六组预测与 checkpoint 已核验，旧数字保留。实际 checkpoint 选择为
六时效全部有效目标平均误差，不能直接当作按日间主指标选择的 V2 模型。
后续复用范围与新独立测试规则见 [实验状态](EXPERIMENT_STATUS.md) 和
[评估协议 V2](EVALUATION_PROTOCOL_V2.md)。文中“test 未读取”限于训练/评分选样，不代表
底层 Parquet 文件从未包含或读取测试行，也不代表整个项目的测试结果从未曝光。

## 实验目的

Power-only 阶段用于建立不读取卫星图像的学习模型上限。模型只接收过去 60 分钟功率、
未来可知的太阳特征和静态站点元数据，并与 Persistence、Smart Persistence 和
Climatology 使用相同数据切分与指标协议。

## 已实现内容

- `PowerMLP` 和因果 `PowerTCN`，参数量均小于 1M；
- masked 六时效等权 MAE、梯度裁剪、早停和检查点恢复；
- 只基于 train/development 拟合的太阳及站点标准化统计量；
- validation/development checkpoint 选择，test 默认锁定；
- 运行级配置、Git/data/split/statistics hash、预测及逐时效指标；
- CPU 冒烟、输入白名单、防泄漏和检查点一致性测试。

## 运行方法

开发阶段单种子运行：

```bash
python scripts/train_power_model.py --model tcn --seed 42
```

模型和分析方案冻结后，才允许最终测试：

```bash
python scripts/train_power_model.py --model tcn --seed 42 --unlock-test
```

## 正式验证结果

使用 `paired-30d-v1`、固定无泄漏切分及 `42、123、2026` 三个种子训练 MLP 和 TCN。
表中学习模型为三种子 MAE 均值；数值越低越好。

### Validation/development

| 模型 | 15 min | 30 min | 60 min | 120 min | 180 min | 240 min | 六时效平均 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 0.005761 | 0.009615 | 0.014959 | 0.026332 | 0.035587 | 0.042258 | 0.022419 |
| Smart Persistence | **0.005419** | 0.008365 | 0.011638 | 0.017694 | 0.021755 | 0.024341 | 0.014869 |
| PowerMLP | 0.005873 | **0.007910** | **0.009809** | **0.013408** | **0.016096** | **0.018144** | **0.011874** |
| PowerTCN | 0.005786 | 0.008156 | 0.010262 | 0.014086 | 0.016743 | 0.018459 | 0.012249 |

PowerMLP 六时效平均 MAE 的跨种子标准差为 0.000177。相对 Persistence 改善 47.0%，
相对 Smart Persistence 改善 20.1%。在主要选择时效 30/60/120 分钟上，PowerMLP 平均
MAE 为 0.010376，相对 Persistence 改善 38.9%，相对 Smart Persistence 改善 17.4%。

### Validation/holdout

| 模型 | 15 min | 30 min | 60 min | 120 min | 180 min | 240 min | 六时效平均 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 0.007658 | 0.012630 | 0.020885 | 0.035765 | 0.049401 | 0.062667 | 0.031501 |
| Smart Persistence | **0.006933** | **0.010285** | **0.014598** | 0.024227 | 0.030360 | 0.034259 | 0.020110 |
| PowerMLP | 0.008191 | 0.011453 | 0.015842 | **0.022238** | **0.027459** | **0.031295** | **0.019413** |
| PowerTCN | 0.008005 | 0.011847 | 0.016523 | 0.023264 | 0.028015 | 0.031665 | 0.019886 |

PowerMLP 在留出站点的六时效平均 MAE 上相对 Persistence 改善 38.4%，相对 Smart
Persistence 改善 3.5%。它在 120–240 分钟更好，但在 15–60 分钟不如 Smart
Persistence，说明新站点的短时效预测仍应优先信任物理先验。

## 结论与下一步

- PowerMLP 是本阶段胜出模型；它在 development 的 30–240 分钟和 holdout 的
  120–240 分钟均稳定优于朴素基线。
- 小型 TCN 没有超过 MLP，不继续扩大 Power-only 时序模型。
- 15 分钟上两个学习模型都没有超过 Smart Persistence，不声称全时效胜出。
- PR #6 的“稳定超过 Persistence”决策门已经满足，可以进入卫星 DataLoader 与
  Optical Flow 阶段。
- 本轮只使用 validation；test 没有解锁或读取。最终测试要等卫星模型和分析方案冻结后执行。

当前数据只有 30 天，因此以上结果用于模型与管道决策，不据此声称跨季节泛化。
