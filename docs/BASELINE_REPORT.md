# Cloud2Watt 朴素基线报告

## 结论

本报告是 `paired-30d-v1` 上第一份冻结配置后的无泄漏基线结果。Smart Persistence 是后续
Power-only 和卫星模型必须优先超过的基线：在 test/development 上，样本加权 MAE 从
Persistence 的 0.01801 降至 0.01147，改善 36.3%；在未见站点 test/holdout 上从
0.02440 降至 0.01610，改善 34.0%。

Site-Time Climatology 在 15–30 分钟明显弱于 Persistence，但随时效增加而变强；在
test/development 的 240 分钟时效，MAE 为 0.01228，skill score 为 0.631。该现象符合
短时变化依赖最新观测、长时平均日变化更有竞争力的预期。

## 协议

- 数据：`paired-30d-v1`，manifest hash
  `4403e787227381aacbfa8a661e9320dc5bd356323ac61d12f09722f7339941ab`；
- 实现提交：`e7f1add5efc2893285feefa9084f5b2d69dae744`；
- 配置 hash：`3c3134aafe56ab6aa94985043333e6ac170851aeed63c87540fd8cbb6a30cba7`；
- split manifest hash：
  `f77fb51e54c7c36b3df5e9c7804759d670106c3b1755df81915db3fe3c1bbeec`；
- 时间切分：70%/15%/15%，4 小时 embargo，完整样本区间不得跨界；
- 空间切分：固定 seed 42，留出站点 `26892`、`26979`、`6801`、`7468`；
- validation 先独立运行并冻结配置，之后只执行一次 `--unlock-test`；
- 最终 run id：`naive-baselines-v1-9f3d90ef59fc`。

时间分配包含 38,433 个 train、7,611 个 validation、8,010 个 test 样本，另有 1,307 个
样本因 embargo 或完整区间跨界而不分配。指标只使用各目标自己的 `target_mask`。

## Test/development 按时效结果

| 时效 | Persistence MAE | Smart Persistence MAE | Smart skill | Climatology MAE | Climatology skill |
|---:|---:|---:|---:|---:|---:|
| 15 min | 0.00460 | 0.00422 | 0.083 | 0.01226 | -1.665 |
| 30 min | 0.00770 | 0.00653 | 0.151 | 0.01230 | -0.599 |
| 60 min | 0.01287 | 0.00976 | 0.241 | 0.01234 | 0.041 |
| 120 min | 0.02144 | 0.01394 | 0.350 | 0.01235 | 0.424 |
| 180 min | 0.02848 | 0.01634 | 0.426 | 0.01231 | 0.568 |
| 240 min | 0.03324 | 0.01812 | 0.455 | 0.01228 | 0.631 |

## 样本加权与站点宏平均

| 分区 | 模型 | Weighted MAE | Macro-site MAE |
|---|---|---:|---:|
| validation/development | Persistence | 0.02234 | 0.02210 |
| validation/development | Smart Persistence | 0.01483 | 0.01469 |
| validation/development | Climatology | 0.01701 | 0.01681 |
| validation/holdout | Persistence | 0.03128 | 0.03021 |
| validation/holdout | Smart Persistence | 0.02000 | 0.01926 |
| validation/holdout | Climatology | 0.04637 | 0.04522 |
| test/development | Persistence | 0.01801 | 0.01790 |
| test/development | Smart Persistence | 0.01147 | 0.01140 |
| test/development | Climatology | 0.01231 | 0.01226 |
| test/holdout | Persistence | 0.02440 | 0.02311 |
| test/holdout | Smart Persistence | 0.01610 | 0.01520 |
| test/holdout | Climatology | 0.03750 | 0.03631 |

未见站点上 Climatology 只能回退到训练站点全局均值，因此其明显退化是预期结果，不应与
同站点时间泛化混为一谈。Smart Persistence 不拟合站点参数，在空间 holdout 上保持了较好
的相对优势。

## 爬坡事件

主阈值为未来 30 分钟归一化功率变化 ±0.20，并按站点进行 ±15 分钟一对一容忍匹配；
validation 另报告 0.10、0.15 和 0.25 敏感性。0.20 阈值事件很少：test/development 有
10 个上升和 13 个下降事件，test/holdout 有 11 个上升和 8 个下降事件。Persistence 不会
产生爬坡预测；Smart Persistence 仅在 test/holdout 上识别 1 个上升事件，F1 为 0.154。
因此这些朴素基线不能可靠识别短时爬坡，后续模型需要同时报告事件数量和置信区间，避免用
少数事件的高方差 F1 作过度结论。

## 可复现性与限制

相同配置和实现提交在两个独立输出目录复跑，`split_manifest.json`、`metrics.json`、
`metrics_by_horizon.csv` 和 `predictions.parquet` 的 SHA-256 均逐字节一致。公共 CI 使用
合成数据，不访问 gated 数据、不联网读取卫星帧，也不需要 GPU。

该报告只覆盖 2020 年 12 月的 30 天冬季窗口，不能说明跨季节性能。后续 Power-only 模型
必须首先在相同 split、mask 和指标实现上超过 Smart Persistence；在达到这一条件或明确
解释失败原因前，不进入卫星 late-fusion 结论阶段。
