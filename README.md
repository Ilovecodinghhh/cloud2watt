# Cloud2Watt

基于卫星云图的 0–4 小时光伏功率与爬坡事件临近预测。

## 项目目标

Cloud2Watt 面向能源气象与机器学习场景，构建一条可复现的端到端流程：

1. 对齐卫星云图、太阳位置、天气观测与光伏功率；
2. 使用过去若干帧云图预测未来 0–4 小时光伏功率；
3. 识别短时间内的大幅升降功率（ramp event）；
4. 与 Persistence、晴空模型和 Optical Flow 等基线公平比较；
5. 提供可复现评估、可视化动画和推理接口。

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
pip install -e ".[dev]"
pytest
```

## 路线图

- [x] 初始化可安装的 Python 项目和 CI
- [ ] 确定研究区域、卫星数据源与光伏功率数据源
- [ ] 建立时空对齐和质量控制管道
- [ ] 实现 Persistence、晴空和 Optical Flow 基线
- [ ] 实现 ConvLSTM / U-Net 模型
- [ ] 建立滚动时间验证与爬坡事件评估
- [ ] 制作交互式演示和案例动画
- [ ] 提供 Docker 与推理 API

## 数据与合规

卫星影像和光伏功率数据可能受许可、再分发和隐私条款限制。请只提交下载脚本、元数据和小型合成样例，并在使用前核对各数据源许可证。

## 状态

项目处于早期设计与基础设施阶段。研究区域和正式数据源尚未确定，README 中不会把计划项描述为已完成结果。
