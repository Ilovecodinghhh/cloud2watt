# Cloud2Watt 模型阶段实施计划

版本：0.1  
状态：待实施  
最后更新：2026-09-05

本文档细化 PR #5 之后的四个实施步骤。PR #1–#5 已经完成数据访问、真实配对数据、
30 天质量控制、无泄漏切分、统一指标和朴素基线；本阶段的目标是在相同评估协议下，
逐步验证历史功率、云系运动和卫星图像的增量价值。

## 1. 当前基线与推进原则

当前固定数据版本为 `paired-30d-v1`，包含 20 个站点、55,361 个有效样本。时间切分
使用 70%/15%/15% 连续区间和 4 小时 embargo，另有 20% 站点留出评估。当前最强朴素
基线是 Smart Persistence，其相对 Persistence 的加权 MAE 改善约为：

- test/development：36.3%；
- test/holdout：34.0%。

后续实验必须复用 PR #5 冻结的样本索引、切分 manifest、目标 mask 和指标实现。开发阶段
只使用 train、validation/development；test 默认锁定，在模型与分析方案冻结后只运行一次。

| 顺序 | 分支 / PR | 核心目标 | 决策门 |
|---|---|---|---|
| 6 | `codex/pr6-power-only-models` | 建立训练框架和 Power-only MLP/TCN | 稳定超过 Persistence，或解释失败原因 |
| 7 | `codex/pr7-satellite-loader-optical-flow` | 验证卫星加载性能与可解释运动特征 | 真实数据加载稳定且无未来帧泄漏 |
| 8 | `codex/pr8-cnn-late-fusion` | 同预算验证卫星图像增量价值 | 关键时效获得稳定改善 |
| 9 | `codex/pr9-satellite-value-audit` | 消融、置信区间与最终价值审计 | 形成有统计支持的下一阶段结论 |

预计工期：PR #6 和 PR #7 各 3–5 天，PR #8 为 5–7 天，PR #9 为 4–6 天。

## 2. PR #6：训练框架与 Power-only MLP/TCN

### 2.1 目标

建立所有学习模型共用的训练、检查点和实验记录框架，并用不含卫星的模型形成公平、可复现
的强基线。该 PR 不追求大规模调参，重点是确认数据、损失和训练闭环正确。

### 2.2 输入与模型

输入只包含：

- 过去 60 分钟的四个归一化功率值及有效性 mask；
- 历史和未来已知的太阳位置、晴空辐照度特征；
- 容量、倾角、朝向、纬度和经度等静态站点信息。

第一版不使用卫星，也不使用可记忆具体站点的 site embedding，以免把已知站点拟合能力误当作
可迁移能力。实现两个参数量小于 1M 的模型：

1. `PowerMLP`：展平历史和外生特征后直接输出六个时效；
2. `PowerTCN`：对历史序列使用小型因果 TCN，再与未来太阳和静态特征融合。

两者均输出 `[batch, 6]`，对应 15、30、60、120、180、240 分钟。

### 2.3 训练协议

- 使用 masked、六时效等权 MAE，与基线评估保持一致；
- 所有归一化统计量只用训练集拟合，并把统计量及其 hash 写入检查点；
- 只根据 validation/development 选择 checkpoint；holdout 仅作迁移诊断；
- 固定随机种子 `42、123、2026`，报告均值、标准差和逐种子结果；
- 支持 CPU 冒烟训练、CUDA、AMP、早停、断点恢复和梯度裁剪；
- 每次运行保存配置、Git commit、数据/切分 hash、依赖版本、预测和指标。

调参预算限定为最多 8 个候选：MLP 宽度 `64/128`，TCN 通道 `32/64`，dropout
`0/0.1`，学习率 `1e-3/3e-4`。先做小预算筛选，再对胜出配置运行三个固定种子。

### 2.4 交付物

- 统一的训练入口和 YAML 配置；
- PowerMLP、PowerTCN、masked loss 和数据 batch 适配器；
- checkpoint 保存、恢复与推理；
- 与 PR #5 相同 schema 的预测和指标文件；
- `docs/POWER_ONLY_REPORT.md`，包含逐时效和 development/holdout 对比。

### 2.5 测试

- train-only normalization 防泄漏测试；
- masked loss 与手工计算对照；
- 输入字段白名单测试，确保没有卫星、未来功率或目标派生特征；
- 模型输出 shape、全缺失标签和异常值路径测试；
- 10 个 batch 的 CPU 冒烟训练；
- checkpoint 保存/恢复后的预测一致性测试；
- 固定 seed 的小型训练确定性测试。

### 2.6 Definition of Done 与停止条件

- 一条命令可在 `paired-30d-v1` 上训练和评估；
- 三个种子均完成且无 NaN，输出可追溯；
- 最佳模型稳定超过 Persistence；
- 若未超过 Smart Persistence，报告按时效、太阳高度角和站点分析原因；
- 单次标准运行在 12 GB GPU 上尽量低于 2 小时；
- 模型选择过程未读取 test。

若连 Persistence 都无法稳定超过，暂停 PR #7–#8，优先检查标签对齐、mask、归一化、损失
和训练实现，不通过扩大网络掩盖问题。

## 3. PR #7：卫星 DataLoader 与 Optical Flow 基线

### 3.1 目标

把 PR #3–#4 的索引式卫星存储转化为稳定、高吞吐的训练 batch，并用光流建立一个低成本、
可解释的云系运动基线。该 PR 将“卫星是否可高效使用”和“运动信息是否有信号”分开验证。

### 3.2 DataLoader 契约

统一 batch 至少包含：

```text
satellite:       [B, 4, 3, 128, 128]
satellite_mask:  [B, 4]
power_history:   [B, 4]
solar_features:  [B, time, features]
site_features:   [B, features]
target_power:    [B, 6]
target_mask:     [B, 6]
metadata:        site_id, issue_time_utc, frame_indices
```

实现要求：

- 根据 frame index 懒加载并组装唯一卫星帧，不把全量卫星数据载入内存；
- 按 Zarr chunk、站点和相邻时间对样本分桶，降低随机 I/O；
- 配置 `num_workers`、prefetch、pinned memory 和 persistent workers；
- 在 Windows `spawn` 多进程模式下可运行，不共享不可序列化句柄；
- 用流式方法只在 train 帧上计算通道均值/方差并保存 hash；
- 同一四帧序列应用一致的空间 jitter/crop；
- 不使用左右翻转或旋转，因为它们会破坏地理方向和云移动方向。

### 3.3 Optical Flow 基线

- 优先在 `IR_108` 上估计连续历史帧的光流；
- 从站点中心邻域提取速度、方向、散度和幅值统计；
- 将光流特征与 PR #6 的功率、太阳和静态特征融合；
- 使用 Ridge 或小型 MLP 输出六个预测时效；
- 严格只使用 issue time 及之前的四帧，不用未来图像拟合或检查单个样本；
- 输出典型晴天、快速云变和失败样本的运动可视化。

### 3.4 性能与测试

性能目标：batch size 8 可稳定运行；热缓存吞吐不低于 50 samples/s；DataLoader 等待时间
低于单 epoch 时间的 30%；连续读取 1,000 个真实 batch 无持续内存增长。

测试包括：

- 合成平移图像的光流方向和量级测试；
- 四帧一致增强测试；
- validation/test 帧不参与通道统计的防泄漏测试；
- 夜间、缺帧、全无效帧和 NaN 路径；
- Windows 多 worker 下样本不丢失、不重复；
- 每个卫星时间均满足 `satellite_time <= issue_time`；
- DataLoader batch schema 和 dtype 测试。

### 3.5 Definition of Done

- 真实数据连续读取 1,000 个 batch，无死锁、shape 错误或内存泄漏；
- 形成包含冷/热缓存、worker 数和 batch size 的吞吐报告；
- 合成平移测试和至少一组真实光流可视化通过人工检查；
- Optical Flow 基线进入统一评估流程；
- test 仍保持锁定。

## 4. PR #8：轻量 CNN Late Fusion

### 4.1 目标

在冻结的数据、切分、训练预算和评估协议下，回答卫星图像能否为 Power-only 和 Smart
Persistence 提供稳定增量。第一版只实现轻量 late fusion，不同时引入 ConvLSTM、3D CNN
或图像外推任务。

### 4.2 模型设计

```text
四帧卫星 ── 共享 2D CNN ── temporal pooling / small GRU ─┐
历史功率 ── PowerTCN 或 MLP ─────────────────────────────┼─ Fusion MLP ─ 六时效
太阳特征 ── MLP ─────────────────────────────────────────┤
站点特征 ── MLP ─────────────────────────────────────────┘
```

- CNN 对每帧共享权重，使用 GroupNorm，避免小 batch 下 BatchNorm 不稳定；
- 参数量目标 5–10M，显存不足时优先缩减 encoder 宽度；
- 支持 AMP、batch size 4/8 和梯度累积；
- 输出、损失、早停、种子和 checkpoint 协议与 PR #6 完全相同；
- 不允许卫星分支改变样本选择或掩码，从而保证模型间逐样本配对比较。

### 4.3 必做对照

使用相同训练预算完成：

1. PR #6 最佳 Power-only；
2. satellite + solar，不含 power history；
3. power + solar + satellite 完整模型；
4. 完整模型但卫星张量置零；
5. 完整模型但在 batch 内打乱卫星与站点/时间的配对。

主要选择指标为 validation/development 上 30、60、120 分钟平均 MAE，同时报告所有时效、
相对 Smart Persistence 和 Power-only 的 skill。holdout 是次级空间泛化诊断。

### 4.4 卫星价值门

- 关键时效改善至少 5%，且多数种子方向一致：进入 PR #9 稳健性审计；
- 改善 0–5%：先做输入置零、打乱和天气分组诊断，不扩大模型；
- 指标退化：检查时间对齐、投影、通道统计、裁剪位置和缺帧处理，不进入 ConvLSTM。

5% 是项目目标门槛而非显著性结论；最终是否成立由 PR #9 的配对 bootstrap 决定。

### 4.5 测试与 Definition of Done

- 各分支张量 shape、mask 和融合输出测试；
- 单帧共享编码器权重测试；
- 卫星置零与打乱对照可复现；
- CPU 前向/反向冒烟和 GPU 小样本训练；
- checkpoint 恢复预测一致，无 NaN 或持续显存增长；
- 三个种子完成，单次标准训练尽量低于 12 小时；
- 至少一个关键时效出现可复现增益，否则提交完整负结果诊断；
- 形成 `docs/MODEL_CARD.md` 草案，test 仍未用于选择。

## 5. PR #9：消融、统计检验与卫星价值审计

### 5.1 目标

冻结候选模型后，通过逐样本配对比较、日级置信区间和条件分组，确定卫星改善是否真实、出现
在哪里，以及当前 30 天数据能够支持多强的结论。

### 5.2 优先消融矩阵

按以下顺序执行，除被消融因素外保持 split、seed、训练预算和早停一致：

1. power history；
2. power + solar；
3. satellite + solar；
4. power + solar + satellite；
5. 单帧卫星与四帧卫星；
6. `IR_108` 单通道与 `VIS006 + IR_016 + IR_108`；
7. 64 × 64 与 128 × 128 裁剪；
8. 有无静态站点信息。

### 5.3 统计与分组分析

- 以日期为 block，对同一样本上的模型误差做 1,000 次配对 bootstrap；
- 为 30、60、120、240 分钟及关键时效平均值报告 95% 置信区间；
- 同时报告按样本加权和 macro-site 指标；
- 按太阳高度角、天气波动程度、站点、日期和 development/holdout 分组；
- 对卫星相对 Power-only、Smart Persistence 的误差差值绘制分布；
- 所有分析配置和随机种子写入运行产物。

### 5.4 爬坡事件审计

- 固定 0.20 主阈值，报告 0.10、0.15、0.25 敏感性；
- 分别统计上升/下降的 Precision、Recall、F1、±15 分钟命中率和提前量；
- 报告每个 split 和阈值的事件数量，避免在极少事件上强调百分比；
- 若某 split 在 0.20 阈值下少于 100 个事件，不做强 F1 结论，转为描述性结果；
- 需要更可靠事件结论时，优先扩展跨季节数据，而不是增加模型复杂度。

### 5.5 条件式后续决策

- CNN 的改善稳定且置信区间支持：只选择 3D CNN 或 ConvLSTM 中一个进入下一 PR；
- 改善方向一致但证据不足：扩展跨季节月份并复验，不增加模型族；
- 无改善或退化：如实发布负结果，诊断裁剪尺度、通道、位置误差和标签噪声；
- 不论结果如何，本阶段都不进入 U-Net、Transformer 或 diffusion。

### 5.6 Definition of Done

- 所有核心模型完成逐样本配对的日级 bootstrap 置信区间；
- 卫星价值按时效、天气、站点和空间留出得到清晰描述；
- 模型和分析方案冻结后，只对固定 test 运行一次最终评估；
- `docs/MODEL_CARD.md`、技术报告和 README 结果摘要同步更新；
- 明确记录下一步是时空模型、扩展数据，还是以负结果结束模型升级。

## 6. 跨 PR 的实验纪律

- 所有真实数据、模型权重和大体积运行产物继续由 `.gitignore` 排除；
- Git 只提交代码、配置、轻量指标摘要、图表和可复现说明；
- 每个 PR 独立分支、独立审阅，前一 PR 合并后再从最新 `main` 创建下一分支；
- 不因一次种子结果改变阈值、切分或测试集；
- 模型比较使用相同有效样本，缺失导致的样本变化必须单独报告；
- 公共 CI 只运行合成 fixture 和 CPU 冒烟，不下载 gated 数据、不要求 GPU；
- 所有 Markdown、配置和代码使用 UTF-8 without BOM。

## 7. 总体停止条件

1. PR #6 未稳定超过 Persistence 前，不训练卫星 CNN。
2. PR #7 未通过真实 1,000-batch 稳定性与防泄漏测试前，不进行长时间 GPU 训练。
3. PR #8 未显示卫星的稳定方向性改善前，不比较更复杂时空架构。
4. 30 天样本不足以支撑跨季节或稀有爬坡结论时，先扩展数据，不把模型复杂度当作补救。

