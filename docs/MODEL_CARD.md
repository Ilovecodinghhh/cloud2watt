# Cloud2Watt 轻量 CNN Late Fusion 模型卡（草案）

2026-09-28 更新：修正数据上的 c2w-eval-v2.1 精简矩阵已完成，结果见
[精简开发验收报告](COMPACT_DEVELOPMENT_REPORT.md)。下述状态与模型说明保留为 V1 历史记录。

版本：PR #8  
状态：S0 已核实 V1 三模式三种子 GPU 产物完成；其中 power_solar seed 42 恢复历史待核实。S1 已实现 V2 核心并完成 CPU 验收，未开展 V2 正式矩阵。
数据：`paired-30d-v1`  
测试集：本批运行未输出 test 评估；同数据版本的旧基线 test 已曝光

本卡以下结构与训练说明为 V1 历史协议。后续选择与测试规则以
[评估协议 V2](EVALUATION_PROTOCOL_V2.md) 为准；实际运行身份、结果与复用限制见
[S0 实验盘点](EXPERIMENT_STATUS.md)。V1 checkpoint 按全部有效目标的 validation loss
选择，并未实际使用下文所述的 30/60/120 选择指标。新入口已统一日间主指标，见
[S1 验收报告](S1_CORRECTNESS_REPORT.md)。下文 V1 命令仅作历史记录；当前入口拒绝 test 解锁，旧 checkpoint 不能用于 V2 恢复。

## 模型用途

该模型用于预测光伏站点未来 15、30、60、120、180 和 240 分钟的归一化功率。PR #8 的
核心问题不是追求更大网络，而是在与 Power-only 模型相同的样本、切分和损失下，验证四帧
卫星图像是否提供稳定的增量信息。

## 结构

- 三通道四帧卫星图像共享同一个 2-D CNN；
- CNN 使用 GroupNorm 和全局池化，避免小 batch 下 BatchNorm 不稳定；
- 帧 embedding 经单层 GRU 汇总；
- 历史功率、未来太阳特征和站点静态特征分别编码；
- late-fusion head 一次输出六个预测时效；
- 默认 full 模型为 5,231,430 个参数，位于冻结的 5–10M 预算内。

支持三种同预算训练模式：`power_solar`、`satellite_solar` 和 `full`。`full` 模型还必须进行
卫星全零和 batch 内确定性打乱两项推理对照。所有模式保持相同样本与 target mask。

## 数据与防泄漏

卫星输入只包含 issue time 及之前的四帧。功率和卫星归一化统计均只在
train/development 上拟合，并将统计 hash 写入 checkpoint 和 provenance。模型选择只看
validation/development，validation/holdout 仅用于空间迁移诊断；test 默认不构建评估分区。

## 训练协议

- masked MAE，六个时效等权；
- AdamW、梯度裁剪、早停和 best-checkpoint 恢复；
- CUDA 可用时启用 AMP；支持梯度累积；
- 固定种子 `42、123、2026`；
- 主要选择指标为 validation/development 的 30、60、120 分钟平均 MAE；
- 卫星价值门为相对强 Power-only 对照改善至少 5%，且多数种子方向一致。

正式矩阵命令：

```bash
python scripts/run_cnn_experiment_matrix.py
```

单次运行也可指定 `--mode` 和 `--seed`。只有在模型、分析方案冻结后，才允许显式传入
`--unlock-test`；矩阵脚本有意不提供该参数。

## 已完成验证

- 三种模型模式的 shape 和 10M 参数预算；
- 共享帧编码器只有一套卷积权重；
- 被排除的输入不会影响对应消融模型；
- 卫星置零和打乱对照可复现；
- CPU 前向、反向、梯度累积和 checkpoint round-trip；
- 最终 5.23M 参数配置在真实 Zarr 上完成 8 个训练样本、两个各 4 样本验证分区的一轮
  端到端冒烟；此前的小模型配置另完成 32/16 样本闭环；
- 冒烟运行完整覆盖 best-checkpoint、正常/置零/打乱预测和产物写出；
- 冒烟 provenance 明确记录 CPU、无 AMP、样本上限和 test 未解锁。

小样本冒烟只验证工程闭环，不能用于模型优劣或卫星价值结论。S0 已核实后续九组无样本上限、
最大 30 epoch 的 CUDA/AMP 运行产物。power_solar seed 42 的历史只记录 epoch 8–30，
需要补证或重跑才能作为严格连续训练对照。按旧全天 30/60/120 MAE，full 三种子均值
在 development 为 0.012293，弱于 PowerMLP 的 0.010376；当前不宣称达到新协议的卫星价值门。

## 已知限制

- 当前仅覆盖 30 天数据，不能支持跨季节泛化结论；
- 全局 128 × 128 裁剪可能无法同时兼顾局地云和大尺度云系；
- 卫星缺帧以 mask 处理，但未单独训练图像重建任务；
- 尚未完成日级 bootstrap、天气分组和爬坡事件审计，这些属于 PR #9。
