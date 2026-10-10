# Cloud2Watt 下一阶段实施计划

2026-10-01：新的后续工作以[项目 2.0 总方案](PROJECT_V2_PLAN.md)和
[实验计划](PROJECT_V2_EXPERIMENTS.md)为准，路线 B 已启动；当前阶段与待办见 [执行记录](PROJECT_V2_EXECUTION.md)。正式三折训练尚未开始。
方案取舍与范围见 [ADR-008](adr/008-project-v2-satellite-value.md)。

历史说明（2026-09-10）：本文的“当前状态”以早期 PR #2–#5 为背景，不代表今天的进度。
后续执行使用 [S0 后路线](REVIEW_ROADMAP_2026-09.md) 和 [评估协议 V2](EVALUATION_PROTOCOL_V2.md)，
现有运行盘点见 [实验状态](EXPERIMENT_STATUS.md)。下文保留历史设计与验收记录。

本文档细化 `PROJECT_PLAN.md` 中 mini dataset 之后的三个实施步骤。每一步对应一个
边界清晰、可独立审阅的 Pull Request。

## 当前状态与推进原则

截至 PR #2，项目已经完成：

- UK PV gated dataset 和公开 SEVIRI Zarr 的访问验证；
- 真实 UK PV 20 站点、7 天功率 mini dataset；
- UTC 和区间结束时间对齐；
- 15 分钟功率聚合、容量归一化和质量标志；
- 站点中心卫星裁剪函数；
- `ForecastSample` schema、Zarr/Parquet 写入契约和合成集成测试。

当前尚未完成的关键数据门是：真实 UK PV 与真实卫星裁剪尚未形成连续 7 天的配对训练
样本。因此必须先关闭双模态数据门，再扩展数据和实现评估，暂不进入复杂神经网络。

| 顺序 | 分支 / PR | 核心目标 | 主要产出 |
|---|---|---|---|
| 3 | `feat/paired-mini-dataset` | 关闭双模态数据门 | 20 站点 × 7 天真实配对数据 |
| 4 | `feat/data-quality-30d` | 扩展数据并完成 QC | 20 站点 × 30 天、数据卡和 QC 报告 |
| 5 | `feat/evaluation-baselines` | 固化评估协议 | 无泄漏切分、指标和基线报告 |

## PR #3：真实卫星—光伏配对 mini dataset

状态：已实现，实测结果记录在 [ADR-002](adr/002-paired-mini-dataset.md)。

### 目标

从“代码契约可用”推进到“真实双模态样本可训练”，完成项目计划中的连续 7 天两模态
配对退出条件。

### 工作范围

#### 1. 自动发现时间交集

- 枚举 UK PV 月分区和 SEVIRI 年度 Zarr 的实际可用时间；
- 不沿用只有功率的 2020-12 窗口，自动选择 2021 年起至少连续 7 天的共同窗口；
- 检查候选窗口内 15 分钟卫星时间点的连续性；
- 在 manifest 中固定两个数据源的 revision、时间范围和校验和；
- 对无交集、交集不足 7 天和远程块缺失给出可操作错误信息。

#### 2. 冻结输入与目标时间协议

- 所有时间转换为 timezone-aware UTC；
- 功率和卫星输入历史均为过去 60 分钟；
- 第一版按 15 分钟取样，使用 `-45、-30、-15、0` 四个输入时间点；
- 预测时效固定为 `15、30、60、120、180、240` 分钟；
- 强制 `satellite_time <= issue_time`；
- 强制目标窗口与输入窗口不重叠；
- 首部历史不足或尾部未来标签不足的 issue time 不生成样本；
- 为时间容差、缺帧 mask 和重复时间戳建立单元测试。

#### 3. 构建真实卫星裁剪

- 将 UK PV 模糊化经纬度投影为 SEVIRI geostationary 坐标；
- 第一版固定三个通道，候选为 `VIS006、IR_016、IR_108`；
- 每个站点生成 `128 × 128` 空间裁剪；
- 批量读取远程 Zarr 块，避免逐样本重复请求；
- 缓存唯一的“站点—时间—卫星帧”，训练样本仅保存帧索引；
- 越界、缺帧、无效值和 codec 解码失败均记录质量标志；
- 至少为 5 个站点生成带位置标记的投影抽查图。

#### 4. 形成可训练样本索引

目标目录：

```text
data/processed/paired-mini-v1/
├── manifest.json
├── sites.parquet
├── power_15min.parquet
├── satellite_frames.zarr
├── sample_index.parquet
└── build_report.json
```

`sample_index.parquet` 至少包含：

```text
site_id
issue_time_utc
satellite_frame_indices
power_history_indices
target_indices
target_mask
quality_flags
```

卫星历史帧不能直接复制到每个训练样本中，否则滑动窗口会重复存储同一帧。DataLoader 在
读取时根据索引组装 `[history, channels, height, width]` 张量。

#### 5. 处理 5 分钟功率语义风险

- 在配置中增加必填项 `power_value_semantics`；
- 支持 `interval_energy` 和 `instantaneous_power` 两种明确模式；
- 未指定语义时拒绝构建，不静默使用默认值；
- 比较两种解释下的容量归一化分布、日间曲线和物理越界率；
- 根据上游说明和数据统计冻结选择，并记录到新的 ADR；
- 在最终决定前，不使用该字段产生正式模型结论。

### 测试

- UTC、DST 和右闭区间边界测试；
- 投影、最近像素和固定尺寸裁剪测试；
- 卫星输入不晚于 issue time 的防泄漏测试；
- 缺帧、越界和错误 codec 的失败路径测试；
- 合成两模态数据从源格式到一个训练 batch 的集成测试；
- 真实网络测试使用显式 opt-in，公共 CI 不下载 gated 数据。

### Definition of Done

- 20 个真实站点与连续 7 天真实卫星数据成功配对；
- 每个有效样本具有 4 个输入时刻和 6 个预测时效；
- 随机检查不少于 20 个样本，卫星时间均不晚于 issue time；
- 至少 5 个站点通过投影位置人工抽查；
- 单样本读取中位数低于 100 ms，P95 建议低于 200 ms；
- 相同配置重复构建得到相同的样本键集合和 manifest hash；
- Ruff、单元测试、合成集成测试和 `git diff --check` 全部通过。

### 主要风险与控制

- 远程读取过慢：按 Zarr chunk 合并请求并持久化本地帧缓存；
- 体积膨胀：只保存唯一卫星帧，样本保存索引；
- 站点位置约 1 km 模糊：使用 128 像素裁剪并记录该限制；
- 可见光夜间无效：保留红外通道，并通过太阳高度角标记夜间样本；
- 功率语义仍不明确：配置强制显式选择，并用 ADR 阻止静默假设。

## PR #4：30 天数据、质量控制与数据卡

### 目标

将配对 mini dataset 扩展到足以支撑基线实验的数据版本，并让每一条清洗决策都可审计。

### 工作范围

#### 1. 扩展构建规模

- 使用 PR #3 的固定数据契约构建 20 个站点、连续 30 天配对数据；
- 延续唯一卫星帧加样本索引的存储结构；
- 输出下载字节数、构建耗时、峰值内存、磁盘占用和读取延迟；
- 根据实测结果估算一年、100 个站点和 300 个站点的资源需求。

#### 2. 完整质量控制

实现并分别统计：

- 官方 `bad_data.csv` 区间；
- 缺失和重复时间戳；
- 负发电量；
- 超过容量合理范围的异常值；
- 夜间非零发电；
- 白天长时间卡零；
- 卫星关键帧缺失；
- 卫星无效值比例过高；
- 站点坐标、容量、倾角或朝向缺失。

原始派生记录不直接删除。每条记录保留 bit flags、`is_valid` 和 `target_mask`，过滤后的
训练视图由配置产生。报告必须包含规则级、站点级和日期级计数。

#### 3. 太阳特征

- 引入 `pvlib`；
- 根据 UTC、纬度和经度计算太阳高度角、方位角和晴空辐照度；
- 夜间判定只使用 issue time 当时可计算的信息；
- 对已知日期和位置增加数值合理性测试；
- 将太阳特征写入派生数据，不提前使用全数据集统计量标准化。

#### 4. 数据文档与版本

- 新建 `docs/DATA_CARD.md`；
- 记录来源、许可、revision、空间精度、时间语义和单位；
- 记录样本选择、质量规则、缺失率和已知局限；
- 新建 ADR，冻结功率语义和首版卫星通道；
- 为数据构建配置和最终 manifest 计算 SHA-256；
- 派生版本命名为 `paired-30d-v1`，不覆盖旧版本目录。

### 测试

- 每一项 QC 规则至少一个命中和一个不命中的测试；
- 重复样本键检测；
- 太阳位置昼夜边界测试；
- 构建器不可覆盖已有版本目录的测试；
- 20 站点、30 天合成压力测试；
- 真实构建作为本地或手动 CI 验证。

### Definition of Done

- 20 站点 × 30 天能够通过一条命令完整构建；
- `(site_id, issue_time_utc)` 全局唯一；
- 所有时间字段均为 timezone-aware UTC；
- 每条过滤规则都有明确的标记数量；
- 数据卡能够复现数据版本和构建配置；
- 年度 300 站点规模估算不超过约 300 GB，或已有明确降本方案；
- 构建完成后可随机读取 100 个样本且无 shape/schema 错误。

### 主要风险与控制

- QC 规则过严：保留 flags，不破坏性删除，报告规则叠加影响；
- 只覆盖单一天气：选择窗口前报告晴天、阴天和波动日比例；
- 30 天不足以支持季节结论：本阶段只用于管道和初始基线，不对全年泛化下结论；
- 数据版本漂移：记录 revision、配置 hash 和 manifest hash。

## PR #5：无泄漏切分、指标与朴素基线

### 目标

在开发 MLP、TCN 或卫星神经网络前，建立统一、可信且可重复的评估下限。

### 工作范围

#### 1. 数据切分

- 主切分采用连续时间 70%/15%/15%；
- train/validation/test 边界设置至少 4 小时 embargo；
- 任何历史输入或未来目标均不得跨越切分边界；
- 增加约 20% 站点留出方案，评估空间泛化；
- 站点划分按容量、纬度和有效数据率分层；
- 输出不可变的 `split_manifest.json` 和 hash；
- test split 默认锁定，只有最终报告命令可以读取。

#### 2. 评估指标

- 按时效实现 masked MAE、RMSE 和 nMAE；
- 实现 `1 - MAE_model / MAE_persistence` skill score；
- 输出总体、站点、预测时效、太阳高度角和日期分组结果；
- 实现未来 30 分钟归一化功率变化的爬坡标签；
- 分别评估上升与下降事件的 Precision、Recall 和 F1；
- 支持 ±15 分钟容忍窗口；
- 阈值默认 0.20，并在验证集上报告 0.10、0.15、0.25 敏感性；
- 为后续日级 block bootstrap 预留逐日指标输出。

#### 3. 朴素基线

- Persistence：所有时效等于 issue time 的最后有效功率；
- Smart Persistence：保持当前晴空指数，用未来晴空功率恢复预测；
- Seasonal/Climatology：只用训练集拟合同站点、相近日时统计量；
- 所有模型使用同一份 sample index、mask 和评估实现；
- 基线参数只根据训练集和验证集确定，不查看测试集。

#### 4. 实验可复现性

每次运行至少记录：

```text
run_id
git_commit
config_hash
data_manifest_hash
split_manifest_hash
train/validation/test ranges
site split hash
random seeds
hardware and package versions
metrics by horizon
```

输出目录建议：

```text
outputs/runs/<run_id>/
├── config.yaml
├── provenance.json
├── metrics.json
├── metrics_by_horizon.csv
├── predictions.parquet
└── plots/
```

### 测试

- 输入、目标和 embargo 不重叠测试；
- 同一物理观测不出现在切分两侧的测试；
- masked metrics 与手工计算结果对照；
- 全缺失目标和零基线误差的错误路径；
- climatology 只拟合训练数据的防泄漏测试；
- 相同 seed 和配置产生相同切分及指标的确定性测试。

### Definition of Done

- 所有自动泄漏检查通过；
- 相同配置重复运行得到一致的预测和指标；
- Persistence、Smart Persistence 和 Climatology 均生成完整结果；
- 生成 JSON、CSV 和按时效指标曲线；
- 测试集不参与阈值、统计量或模型选择；
- 第一份可复现的基线验证报告完成；
- 公共 CI 不依赖网络、gated 数据或 GPU。

### 主要风险与控制

- 30 天无法覆盖三个季节：初始报告明确标为管道验证，正式结论等待跨季节数据；
- embargo 后样本减少：构建前估算每个 split 的有效样本量；
- Smart Persistence 晴空值接近零：夜间不计算晴空指数并单独统计；
- 少数站点主导总体指标：同时报告 macro 和按样本加权结果。

## 执行顺序与停止条件

1. PR #3 未满足连续 7 天真实两模态配对前，不开始模型训练。
2. PR #4 未形成可审计的 30 天数据和数据卡前，不冻结正式评估数据。
3. PR #5 的切分和指标测试未通过前，不实现 Power-only MLP/TCN。
4. Power-only 模型未稳定超过 Persistence，或未解释无法超过的原因前，不进入卫星
   late-fusion 模型。

完成以上三个 PR 后，下一阶段依次是 Power-only MLP/TCN、卫星数据加载性能优化和
CNN late-fusion 增量价值实验。PR #6–#9 的详细任务、测试、决策门和 Definition of Done
见 [模型阶段实施计划](MODEL_PHASE_PLAN.md)。
