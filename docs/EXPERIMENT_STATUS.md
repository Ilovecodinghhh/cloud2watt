# Cloud2Watt 实验状态与复用决定（S0）

后续进展（2026-09-28）：[S1 核心实现与 CPU 验收已完成](S1_CORRECTNESS_REPORT.md)。
以下保留 S0 审计时点记录，不将历史实验改称 V2 结果。

审计日期：2026-09-10
状态：S0 完成；V2 规范已冻结，S1 尚未实施。
审计基线：11aeb43a3d18ea974d607cdb76701fa3f627ca20。

## 1. 结论

现有数据管道、基线和 CNN 代码可以继续作为实现基础。旧实验预测可用于历史分析与回归；没有一个已有学习运行可直接充当按 V2 选择出来的正式模型。

实际发现 25 个运行目录，以及 1 个光流缓存目录：16 个历史已完成、5 个源码身份待核实、1 个恢复历史待核实、3 个冒烟。未发现扫描范围内缺少最终指标的未完成运行；这不代表全机所有任务均已核查。

CNN 的三模式 × 三种子 GPU 产物已在本地完成，旧模型卡“正式 GPU 实验待执行”已过时。9 个运行中 power_solar seed 42 有恢复历史缺口，因此不能把三种子完整矩阵等同于九组连续、完全同协议的可重现实验。

## 2. 工作区与运行状态

- 起始分支 codex/resumable-cnn-training，HEAD 11aeb43；本地 main 与 origin/main 跟踪记录位于 e992e00，已包含 PR #6/#7/#8 合并。
- 315adf3 和 11aeb43 两个恢复相关提交在起始分支中，未包含在本地 main 中。以上均为本地 refs 快照，本次未 fetch 或核验实时 GitHub 状态。
- 起始工作区只有未跟踪的 REVIEW_ROADMAP_2026-09.md；本轮保留并更新状态，新建 codex/s0-evaluation-protocol 文档分支。
- 审计开始时 Windows tasklist 按 python.exe 过滤未发现匹配进程；这只说明该名称下没有进程，不排除远端/其他名称任务。后续审计自身会短暂启动 Python，但不启动训练。
- cnn-matrix.err.log 为 0 字节；out.log 最后修改为本地 2026-09-07 18:41，末尾记录 full seed 2026 产物写出。未据此单独判成功，而是与各运行文件交叉检查。
- 未停止、恢复或修改任何原实验；未读取原始 test 标签或运行新推理。只审计已有元数据、checkpoint 与神经网络 validation 预测。

## 3. 验证范围与证据强度

扫描 outputs/runs、outputs/runs-pr5-repro、outputs/runs-pr5-validation 的直接运行目录；pytest 输出作为测试 fixture 排除，feature_cache 明确为缓存。对每个运行记录 provenance、数据/切分/config/统计 hash、预算、样本上限、文件大小/mtime/SHA-256、历史 epoch 和指标分区。

- 25 个运行的数据身份均匹配当前数据 manifest；各 split 自校验与 provenance 的 split hash 匹配。
- 18 个神经网络运行（15 个正式预算、3 个冒烟）的 checkpoint ZIP CRC 校验通过，受限 weights_only 加载成功，权重全部有限；未执行模型 forward。
- 15 个正式预算神经网络运行的 validation/development 均为 6,113 行，holdout 均为 1,498 行；排序后的样本键、mask、observed hash 在各分区完全一致，无重复键，有效目标上的预测均有限。
- 共核验 27 个 best/latest checkpoint 的身份及 360 个已保存 validation MAE。排序后重算误差与原值在绝对 1e-7 加相对 1e-6 容差内一致；两项置零对照差异为单个 float32 ULP（约 1.19e-7），其余均小于 1e-7。未将重新排序的浮点归约差异当作模型结果变化。
- 记录的 Git commit 与入口/config 存在性单独核对；config hash 兼容 LF 与 CRLF 表示后比对。存在 commit 不等于记录了当时工作区 dirty diff。
- checkpoint 有限、文件 hash 一致与元数据存在不等于重现了原训练；多数学习 provenance 未完整记录依赖版本、GPU 型号或 dirty 源码状态。审计不虚构这些信息。

机器证据：[运行清单](audits/s0-run-inventory-2026-09-10.json)；[checkpoint 与 validation 核验](audits/s0-artifact-verification-2026-09-10.json)。清单中的 mechanical classification 只描述产物完成状态，以 s0_disposition 作为最终复用分类。

## 4. 逐运行清单

epoch 列为“历史记录行数 / 配置最大轮数；best checkpoint 轮次”，并非运行时长。运行路径是精确身份，不把复现副本当作额外独立实验。

| 运行目录 | seed | epoch | device | test_unlocked | 记录 commit | S0 分类 |
|---|---:|---|---|---|---|---|
| outputs/runs/cnn-late-fusion-v1-full-s123-b57a48a9ea84 | 123 | 28/30; best=23 | cuda | False | 11aeb43a | 历史已完成 |
| outputs/runs/cnn-late-fusion-v1-full-s2026-3d6c403e0ca5 | 2026 | 23/30; best=18 | cuda | False | 11aeb43a | 历史已完成 |
| outputs/runs/cnn-late-fusion-v1-full-s42-0d5636f42f60 | 42 | 1/1; best=1 | cpu | False | e42a7a5c | 冒烟 |
| outputs/runs/cnn-late-fusion-v1-full-s42-8b85af90b7e0 | 42 | 1/1; best=1 | cpu | False | e42a7a5c | 冒烟 |
| outputs/runs/cnn-late-fusion-v1-full-s42-a778e1c237a4 | 42 | 18/30; best=13 | cuda | False | 11aeb43a | 历史已完成 |
| outputs/runs/cnn-late-fusion-v1-full-s42-f9f734279294 | 42 | 1/1; best=1 | cuda | False | e992e006 | 冒烟 |
| outputs/runs/cnn-late-fusion-v1-power_solar-s123-f110ad059363 | 123 | 14/30; best=9 | cuda | False | 11aeb43a | 历史已完成 |
| outputs/runs/cnn-late-fusion-v1-power_solar-s2026-2aa9c8418e2c | 2026 | 22/30; best=17 | cuda | False | 11aeb43a | 历史已完成 |
| outputs/runs/cnn-late-fusion-v1-power_solar-s42-f00f00c0cd86 | 42 | 23/30; best=30 | cuda | False | e992e006 | 恢复历史待核实 |
| outputs/runs/cnn-late-fusion-v1-satellite_solar-s123-414992685ada | 123 | 13/30; best=8 | cuda | False | 11aeb43a | 历史已完成 |
| outputs/runs/cnn-late-fusion-v1-satellite_solar-s2026-9667ea3c5beb | 2026 | 30/30; best=28 | cuda | False | 11aeb43a | 历史已完成 |
| outputs/runs/cnn-late-fusion-v1-satellite_solar-s42-1563c30f0f2e | 42 | 9/30; best=4 | cuda | False | 11aeb43a | 历史已完成 |
| outputs/runs/feature_cache | — | —/—; best=— | — | — | — | 非实验缓存 |
| outputs/runs/naive-baselines-v1-9f3d90ef59fc | 42 | —/—; best=— | — | True | e7f1add5 | 历史已完成 |
| outputs/runs/naive-baselines-v1-e687a40fc253 | 42 | —/—; best=— | — | False | 494b2219 | 来源待核实 |
| outputs/runs/optical-flow-v1-4d925b1f1b20 | — | —/—; best=— | — | False | 6b59b62f | 来源待核实 |
| outputs/runs/optical-flow-v1-70a2e14f0fcc | — | —/—; best=— | — | False | 6b59b62f | 来源待核实 |
| outputs/runs/optical-flow-v1-9d3ffdfe2bb7 | — | —/—; best=— | — | False | 6b59b62f | 来源待核实 |
| outputs/runs/power-only-v1-mlp-s123-aa37292f6d17 | 123 | 46/50; best=39 | cpu | False | e0fba8d6 | 历史已完成 |
| outputs/runs/power-only-v1-mlp-s2026-ad108d593d4f | 2026 | 41/50; best=34 | cpu | False | e0fba8d6 | 历史已完成 |
| outputs/runs/power-only-v1-mlp-s42-d49d4f075f61 | 42 | 29/50; best=22 | cpu | False | e0fba8d6 | 历史已完成 |
| outputs/runs/power-only-v1-tcn-s123-0946dc2bbc36 | 123 | 28/50; best=21 | cpu | False | e0fba8d6 | 历史已完成 |
| outputs/runs/power-only-v1-tcn-s2026-350c071725f0 | 2026 | 41/50; best=34 | cpu | False | e0fba8d6 | 历史已完成 |
| outputs/runs/power-only-v1-tcn-s42-c151c8a525be | 42 | 21/50; best=14 | cpu | False | e0fba8d6 | 历史已完成 |
| outputs/runs-pr5-repro/naive-baselines-v1-9f3d90ef59fc | 42 | —/—; best=— | — | True | e7f1add5 | 历史已完成 |
| outputs/runs-pr5-validation/naive-baselines-v1-1d049f7cb416 | 42 | —/—; best=— | — | False | 494b2219 | 来源待核实 |

## 5. 按成果的复用决定

| 成果 | 可复用部分 | 后续约束 |
|---|---|---|
| 数据与索引管道 | schema、UTC、投影、QC、Zarr 索引和测试 | S2 确认字段语义、可用时间与共享 tile；旧数据不能代表全年 |
| 时间切分与 mask | 完整窗口、embargo 和同样本比较思想 | S3 使用新时间/区域角色；V2 增加日间评分与显式失败处理 |
| MLP/TCN 六组运行 | 旧 validation 预测、权重与历史性能诊断 | 按旧六时效有效目标均值选 checkpoint，不能直接宣称 V2 胜出；新数据/协议需重训 |
| CNN 八组常规完成运行 | 旧预测、权重、三模式实现与 GPU 闭环 | 同上；AMP、累积和 mask 语义仍需 S1 验收 |
| CNN power_solar seed 42 | 有限且可读取的终态权重/预测 | 历史仅 epoch 8–30；恢复身份跨 e992e00→315adf3，不能证明连续同协议，严格比较需补证或重跑 |
| 基线正式 test 与复现副本 | 冻结 V1 的复现证据；实现可按 V2 重新评分 | test 已曝光，新独立测试必须另建；副本不能增加统计样本数 |
| 两组早期基线 validation | 旧表和探索记录 | provenance 指向 494b221，该提交不存在运行入口；不能仅靠 commit 复建执行状态 |
| 三组 Optical Flow | 三组 metrics 与 predictions 各自逐字节相同，可复用负结果与缓存证据 | 记录 6b59b62 不含 optical-flow 入口/config；缺 dirty 源码快照，正式复现要补证或重跑 |
| 三组 CNN 冒烟 | CPU/CUDA、输入输出与 artifact 闭环 | 样本上限和 1 epoch，不能用于模型胜负；两组的记录提交尚不包含 CNN 入口 |
| DataLoader benchmark | 1,000 batches / 8,000 samples、69.80 samples/s 的既有测量 | 未记录持续 RSS，不能宣称已量化证明无内存增长；新布局要重测 |
| 当前旧 test | 既有历史事实和可审计对照 | 不作为全项目从未查看的盲测；不改写旧数值 |

五个“来源待核实”运行均已有指标，缺的是完整执行源码身份，不是缺少训练完成文件。不能从缺入口推断结果造假；只能说明记录的 commit 不足以重建运行。

## 6. 当前模型结果的准确含义

以下为已有 validation 预测重新核对的全天 MAE，学习模型为三个种子的误差平均；没有计算新日间指标，也没有执行新推理。CNN power_solar seed 42 的恢复限制同时适用于以下三种子比较。

| 模型 | development 六时效均值 | development 30/60/120 均值 | holdout 六时效均值 | holdout 30/60/120 均值 |
|---|---:|---:|---:|---:|
| PowerMLP | 0.011874 | 0.010376 | 0.019413 | 0.016511 |
| PowerTCN | 0.012249 | 0.010835 | 0.019886 | 0.017211 |
| CNN power_solar | 0.013875 | 0.012974 | 0.022161 | 0.020150 |
| CNN satellite_solar | 0.018248 | 0.017876 | 0.030877 | 0.030368 |
| CNN full | 0.013631 | 0.012293 | 0.021504 | 0.019068 |

full 对匹配 power_solar 的全天 30/60/120 平均误差约改善 5.25%（development）、5.37%（holdout），但对更强 PowerMLP 约退化 18.47%、15.49%。这说明只比较较弱匹配模型会夸大实际竞争力；不能据此宣称卫星通过下一阶段胜出条件。未经日间主指标、跨季节验证和配对区间检查，这些只能作为历史诊断。

## 7. 测试集使用记录

- 旧版本 manifest：4403e787227381aacbfa8a661e9320dc5bd356323ac61d12f09722f7339941ab。
- 旧 split：f77fb51e54c7c36b3df5e9c7804759d670106c3b1755df81915db3fe3c1bbeec。
- 旧 test 信息区间起点为 2020-12-26 16:00 UTC；样本历史也必须位于此后，因此不是所有 issue≥16:00 都进入 test。实际选择以原 split manifest/assignments 为准。
- naive-baselines-v1-9f3d90ef59fc 在 runs 与 runs-pr5-repro 均标记 test_unlocked=true，构成已曝光测试与同配置复现。旧报告已公开 test/development 与 test/holdout。
- 其他 23 个运行均记录 test_unlocked=false。该标记不是文件级访问审计；训练代码先读整张 Parquet 再按 split 选行，所以仅表述“未输出 test 评估”。
- 不读取新的测试标签即可确认该曝光事实。新协议将旧版本整体归为开发/回归资料，另建真正未查看的数据。

## 8. S1 明确待办与下一阶段胜出条件

S1 先统一日间选择指标与有限值处理，再验证不等目标数/末组梯度累积、缺帧、AMP、恢复状态与完成身份。现有 metrics.py 会过滤非有限预测，而神经网络 metrics_by_horizon 走另一实现；本次现有 validation 未发现非有限预测，但两条评分路径仍须统一失败策略。

下一阶段主指标为目标太阳高度角>5° 的 30/60/120 MAE 等权平均；必须对开发阶段选出的最强无卫星 B* 改善至少 5%，多数种子与时间折方向一致，再由配对日期块区间确认。最终时间和区域胜出分别判断，详见 [V2 协议](EVALUATION_PROTOCOL_V2.md)。

## 9. S0 完成边界

- [x] 已盘点运行、产物、源码身份、测试历史与工作区。
- [x] 已给每项成果明确复用或补证/重跑决定。
- [x] 已冻结 V2 主指标、比较对象、胜出门和独立测试角色。
- [x] 已用 ADR 记录变化并修正旧文档状态入口。
- [x] 未执行 S1 修复、训练、扩容或新测试；尚未指定的新数据日期和算力额度在 S3/S4 落实。
