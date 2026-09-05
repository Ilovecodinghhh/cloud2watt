# Cloud2Watt 轻量 CNN Late Fusion 模型卡（草案）

版本：PR #8  
状态：代码与真实数据小样本闭环已验证；正式三种子 GPU 实验待执行  
数据：`paired-30d-v1`  
测试集：锁定

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

小样本冒烟只验证工程闭环，不能用于模型优劣或卫星价值结论。当前机器没有 CUDA，尚未执行
正式三种子真实数据矩阵，因此本模型卡不报告或暗示正式性能提升。

## 已知限制

- 当前仅覆盖 30 天数据，不能支持跨季节泛化结论；
- 全局 128 × 128 裁剪可能无法同时兼顾局地云和大尺度云系；
- 卫星缺帧以 mask 处理，但未单独训练图像重建任务；
- 尚未完成日级 bootstrap、天气分组和爬坡事件审计，这些属于 PR #9。
