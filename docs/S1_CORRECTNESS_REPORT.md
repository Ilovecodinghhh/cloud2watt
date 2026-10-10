# S1：指标、训练与恢复正确性验收

收尾日期：2026-09-28。实现与测试验收：2026-09-10。

状态：S1 核心实现及 CPU 验收完成；CUDA/AMP 验收因当前环境没有 CUDA 而待补。
本阶段没有开展正式模型矩阵，没有建立或解锁新的最终测试集。

## 1. 已完成的修正

| 范围 | 实现与验收依据 |
|---|---|
| 主指标 | 目标时刻太阳高度角严格大于 5°；分别计算 30/60/120 分钟 MAE 后等权平均。任一主要时效无有效目标则不能选为候选。checkpoint、早停和矩阵排序使用同一指标。 |
| 有效目标 | 有效标签或预测出现 NaN/Inf 即失败并报告键与失败率；无效目标不参与误差。损失先索引再相减，避免被忽略的 NaN 污染梯度。 |
| 比较身份 | 验证样本键唯一，保存排序无关的键、标签、目标 mask、日间 mask 哈希；正常、置零、打乱预测核对相同评分集合。 |
| 诊断输出 | 全天、日间、各时效 MAE/nMAE/RMSE/bias、有效目标数、日期数、太阳高度分组、站点和日期分组；区域数据缺失明确标注。 |
| 训练目标 | 仍为六时效有效目标 MAE，命名 train_masked_mae；梯度累积按每个微批的有效目标数量加权，覆盖最后不足一组的情况。 |
| 缺帧 | 无效图像清零；GRU 在缺帧位置保持隐藏状态，全缺帧返回零向量。覆盖首帧、中间、末帧缺失及全有效、全缺帧。 |
| 恢复 | epoch 边界保存优化器、GradScaler、Python/NumPy/Torch/CUDA RNG、加载器 generator、采样器 epoch、早停状态、最佳权重和完整历史。 |
| 完成判定 | 文件原子替换、进程锁、完整身份校验；全部必需产物写好后才写 complete.json。缺文件、内容变更、旧协议或身份不符均不能跳过。 |

主要实现位于 src/cloud2watt/evaluation/forecast.py、epoch.py、experiment.py 和
run_state.py；两个训练脚本共用执行路径。模型前向接口显式选择输入，原始目标太阳高度角仅供评分。

## 2. 验证结果

- Ruff 检查通过。
- 完整测试：94 passed、1 skipped，87.34 秒。跳过项为 CUDA AMP 验收。
- CPU 连续训练与 epoch 边界恢复测试覆盖 dropout、打乱采样以及 0/2 workers，核对权重、优化器状态与历史一致。
- 合成端到端测试覆盖暂停/恢复、身份不符拒绝、完成跳过、卫星消融比较集合、禁止 test 解锁，以及配置/数据/统计量/源码/环境变化。
- 原子写入中断、产物损坏、并发锁、矩阵缺种子不可胜出均有回归测试。

执行过的完整测试命令如下；重新执行时应使用新的、尚不存在的 basetemp 目录。

```bash
python -m ruff check src scripts tests
python -m pytest --basetemp=outputs/pytest-s1-full-20260910 --tb=short -ra
```

## 3. 真实数据小样本验收

数据为已有 paired-30d-v1。两组均使用 seed 42、12 个训练样本、8 个 development validation
样本和 8 个 holdout validation 样本，预算为 2 epochs；第 1 epoch 后暂停，再恢复至完成，
随后验证完成运行可安全跳过。卫星使用真实 128×128 图像及 2 workers。

| 模型 | 参数量 | 最佳 epoch | 日间主 MAE | 恢复 / 完成跳过 |
|---|---:|---:|---:|---|
| 小型 Power MLP | 822 | 2 | 0.1675360511 | 通过 / 通过 |
| 小型 full 卫星模型 | 10,918 | 2 | 0.0840400225 | 通过 / 通过 |

这些数字仅为流程验收，样本上限采用有日间支持的确定性选样，不能作为卫星增益或正式排名证据。
运行环境为 Torch 2.14.0+cpu，AMP 未启用。

机器可读记录：[原始验收](audits/s1-real-acceptance-2026-09-10.json)、
[2026-09-28 产物复核](audits/s1-closeout-verification-2026-09-28.json)。
复核确认两组 complete.json 与其全部必需文件哈希一致，当前训练源码哈希与验收源码一致：
306e628e81cf46bb5551b946e6fd35c8bf71faeea27c20c131ba010b62bc2432。

## 4. 使用与恢复约定

训练入口为 scripts/train_power_model.py 和 scripts/train_cnn_late_fusion.py，默认读取对应
V2 配置。通过 --max-train-samples、--max-validation-samples、--max-epochs 限定验收预算；
带样本上限的运行自动标记为 smoke，不进入正式候选排序。

--stop-after-epoch 用于 epoch 边界暂停，不改变总预算。恢复时使用 --resume-run 指向原目录，
并保留相同配置、预算、数据、统计量、源码和环境；也可使用 --auto-resume 查找同身份运行。
进程在 epoch 中断时，从上一个已提交 epoch 重放，不承诺 batch 内精确续接。

latest.pt 是 epoch 事务的权威状态，包含最佳权重副本和完整历史；恢复后重建 CSV 和 best.pt。
best.pt 用于已选模型推理，不能代替 latest.pt 恢复训练。
身份发生变化时必须另建运行，不允许接续旧结果。

完成目录包含 best.pt、latest.pt、metrics.json、predictions.parquet、provenance.json、
split_manifest.json、training_history.csv、config.json、config.yaml、environment.json、
statistics.json、source.json、source.zip；complete.json 最后写入并记录这些文件的 SHA-256。
源码归档包含实际未提交内容，不能仅凭 Git HEAD 判断运行身份。失败诊断保存在 failure.json。

## 5. 兼容性与验收边界

| 旧成果 | S1 后用途 |
|---|---|
| 数据管道、模型结构、旧预测 | 继续用于开发、回归和历史分析 |
| V1 checkpoint | 历史推理；不能通过 V2 训练入口恢复或当作 V2 完成运行 |
| 旧全天指标与历史排名 | 保留原值，不改称按日间主指标选出的结果 |
| 新 S1 smoke 产物 | 验证执行链路；不用于正式胜出判断 |

当前限制：

- CUDA/AMP 路径尚未在 GPU 验收；正式 GPU 长训练前必须补跑相应测试与连续/恢复对照。
- Zarr 数据身份采用文件名、大小、mtime 清单，并非逐块内容哈希；不能检测保持大小和时间戳不变的篡改。表格文件使用内容哈希。
- 缺帧策略为隐藏状态保持，尚未显式编码缺帧时间间隔。
- 单开发切分与三种子聚合不等于跨季节或跨区域稳健性；独立切分、置信区间、事件与最终测试分别由后续阶段交付。
- test 解锁参数在 S1 开发入口明确拒绝；底层表格可包含历史 test 行，不应据此宣称文件级从未读取。
- 原子替换保障进程中断场景，不承诺所有文件系统上的断电持久性。

## 6. 下一阶段审核输入

下一步为路线图 S2：确认功率单位与区间结束时间、卫星观测及可用时间、许可边界，
并测量共享 tile 的存储、读取吞吐与裁剪成本。先形成数据 ADR 与小规模基准，再进入 S3 扩数。
本次授权范围内的 S1 已交付；S2 尚未执行。
