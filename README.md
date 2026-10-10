# Cloud2Watt

基于卫星云图的 0–4 小时光伏功率与爬坡事件临近预测。

## 项目 2.0

当前项目版本：**2.0**（成果同步：2026-10-10）。已完成30站功率与三通道卫星的质量检查、时空对齐、可恢复构建、GPU滚动训练、卫星处理消融及可解释MVP。

春季冻结模型评价覆盖14天、38,269条共同样本。日间30/60/120分钟平均MAE：晴空修正9.38%、功率＋太阳9.16%、原卫星8.85%、中心32×32卫星8.58%。小范围卫星相对功率＋太阳改善约6.4%。单日配对块95%差值区间为[-1.055,-0.137]容量百分点；连续3天块为[-1.108,0.005]，仍略跨零，不能称为稳健显著或全年泛化结论。

- [完整评价与分组结果](docs/PROJECT_V2_PROSPECTIVE_EVALUATION_20261009.md)
- [机器可读指标快照](docs/results/prospective-evaluation-20261009/results.json)
- [卫星处理消融](docs/PROJECT_V2_SATELLITE_ABLATION_RESULTS.md)
- [可解释MVP验收](docs/PROJECT_V2_MVP_ACCEPTANCE.md)
- [项目模拟面试问答](docs/PROJECT_V2_INTERVIEW_QA.md)
- [2.0总方案](docs/PROJECT_V2_PLAN.md)与[执行记录](docs/PROJECT_V2_EXECUTION.md)

MVP默认仍为功率＋太阳，卫星可选；当前为历史回放原型，尚未验证生产发布延迟、全年和新区域泛化。2022最终测试标签保持锁定。历史实验保留，不以最新结果覆盖旧结论。

仓库包含源码、测试、决策和汇总指标，不包含原始功率/卫星数据、模型权重或缓存。部分研究执行脚本固定本地D盘路径并依赖本地收据，不能仅克隆仓库就复现已训练权重；需按数据访问条件准备数据并调整路径。已有环境训练使用raincalib-gpu Conda，不需要另装或替换PyTorch。

## 项目目标

Cloud2Watt 面向能源气象与机器学习场景，构建一条可复现的端到端流程：

1. 对齐卫星云图、太阳位置、天气观测与光伏功率；
2. 使用过去若干帧云图预测未来 0–4 小时光伏功率；
3. 识别短时间内的大幅升降功率（ramp event）；
4. 与 Persistence、晴空模型和 Optical Flow 等基线公平比较；
5. 提供可复现评估、可视化动画和推理接口。

当前实施路线见 [项目 2.0 总方案](docs/PROJECT_V2_PLAN.md)。原研究范围、数据路线及十二周设计见
[历史项目设计](docs/PROJECT_PLAN.md)。

## 计划中的模型

- Persistence / Smart Persistence
- 晴空指数基线
- Optical Flow
- ConvLSTM
- 3D CNN
- 小型 U-Net
- 后期扩展：概率预测、条件生成模型

## 评估原则

- 严格按时间划分训练、验证和测试集，避免未来信息泄漏；
- 同时报告 MAE、RMSE、nMAE、skill score；
- 对爬坡事件报告 Precision、Recall、F1 和事件提前量；
- 分天气类型、太阳高度角和预测时效分析误差；
- 原始数据不提交到 Git，仓库仅提供下载与预处理脚本。

## 目录结构

```text
cloud2watt/
├── configs/             # 实验配置
├── data/                # 数据说明；原始数据不入库
├── src/cloud2watt/      # Python 包
├── tests/               # 自动化测试
└── .github/workflows/   # 持续集成
```

## 快速开始

需要 Python 3.11 或 3.12。

```bash
python -m venv .venv
source .venv/bin/activate  # Windows Git Bash
python -m pip install --upgrade pip
pip install -e ".[data,dev,ml]"
pytest
```

## 路线图

- [x] 初始化可安装的 Python 项目和 CI
- [x] 确定研究区域、卫星数据源与光伏功率数据源
- [x] 完成有凭据的 UK PV 小分区访问验证
- [x] 建立 mini dataset 的 UTC 对齐、空间裁剪和质量控制管道
- [x] 构建 20 站点、7 天真实卫星—功率配对数据
- [x] 实现 Persistence、Smart Persistence 和 Climatology 基线
- [x] 实现 Power-only MLP/TCN 与可复现训练框架
- [x] 实现 Optical Flow 基线与卫星 DataLoader
- [x] 实现轻量 CNN Late Fusion 与卫星置零/打乱对照
- [ ] 建立滚动时间验证与爬坡事件评估
- [x] 完成可离线打开的交互式预测回放与 CSV 导出
- [ ] 制作案例动画
- [ ] 提供 Docker 与推理 API
- [x] 记录项目 2.0 解决与验证方案、实验预算和决策
- [ ] 执行项目 2.0 诊断、对照实验与条件式独立验证

## 数据与合规

卫星影像和光伏功率数据可能受许可、再分发和隐私条款限制。请只提交下载脚本、元数据和小型合成样例，并在使用前核对各数据源许可证。

## 状态

项目已完成 20 站点、30 天真实配对数据、质量控制、无泄漏评估、朴素基线、
Power-only 模型、卫星 DataLoader、Optical Flow 基线，以及轻量 CNN Late Fusion 训练与
消融框架。真实加载与光流实验记录见
[`docs/SATELLITE_LOADER_REPORT.md`](docs/SATELLITE_LOADER_REPORT.md)，CNN 协议与当前验证
状态见 [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md)。
S0 已完成：[成果复用清单](docs/EXPERIMENT_STATUS.md)、[评估协议 V2](docs/EVALUATION_PROTOCOL_V2.md)
与 [决策记录](docs/adr/005-evaluation-v2-and-test-history.md) 已落地。九组 CNN GPU 产物已完成，
其中一组恢复历史待核实；当前 full 尚未超过强 PowerMLP 基线。
旧 30 天数据的基线 test 已曝光，保留为开发与回归资料。新独立测试将按 V2 另建。
S1 指标、训练与恢复正确性已完成 CPU 验收：94 项测试通过，1 项 CUDA 测试跳过；
真实小样本 Power-only 与卫星运行均通过暂停恢复和完成校验。详见
[S1 验收报告](docs/S1_CORRECTNESS_REPORT.md)。尚未开展 V2 正式模型矩阵。

S2 已交付[数据语义审计与共享 tile 基准](docs/DATA_SCALING_REPORT.md)：两天窗口存储减少
94.56%，共享加载器热缓存 662 samples/s；修正了后续构建的 Meteosat sweep 轴。
用户已选择[精简开发路线](docs/COMPACT_ROUTE_PROTOCOL.md)：保留原数据规模，以采样功率均值代理为目标，
修正卫星定位后进行固定预算对照与离线演示。原 S3 扩容路线暂缓。

精简路线已于 2026-09-28 完成：九组固定预算训练与五种方案对照已验收。
开发站点以纯功率 MLP 最好，留出站点以 Smart Persistence 最好；卫星模型未带来增益。
详见[精简开发验收报告](docs/COMPACT_DEVELOPMENT_REPORT.md)。本地离线演示位于
`outputs/compact-route-v21/demo.html`；通过 `python -m scripts.compact_route` 可生成结果与演示。
当前测试为 103 passed、1 skipped（CUDA），本轮结论仅限历史冬季开发验证。
