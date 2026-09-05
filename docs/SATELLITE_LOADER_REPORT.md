# 卫星 DataLoader 与 Optical Flow 基线报告

版本：PR #7  
数据：`paired-30d-v1`  
评估状态：test 保持锁定

## 1. 结论

卫星加载链路通过了进入图像模型阶段的工程门槛：在真实 30 天数据上连续读取 1,000 个
batch（8,000 个样本），无死锁、样本重复或 shape 错误，吞吐为 69.80 samples/s，超过
计划中的 50 samples/s 门槛。

基于 `IR_108` 连续帧全局平移的 Optical Flow Ridge 能超过 Persistence，但没有超过不含
卫星的 Power + Solar Ridge，更没有超过 Smart Persistence。因此本 PR 证明卫星数据能够
稳定进入训练管道，但不能证明当前全局光流摘要具有增量预测价值。PR #8 可以按计划验证
轻量 CNN；本版光流不作为胜出模型进入后续比较。

## 2. DataLoader 实现与防泄漏约束

- 依据样本中的 frame index 懒加载 Zarr，不把卫星数组整体载入内存；
- 输出四个历史时刻、三个通道的张量 `[4, 3, 128, 128]` 及逐帧 mask；
- 按站点和 Zarr 时间 chunk 分桶，减少随机读取；
- Zarr 句柄不参与 pickle，在 Windows `spawn` worker 内按需重新打开；
- 通道均值和方差仅从 train/development 的唯一帧流式拟合；
- 四帧使用同一个空间平移增强，不做会改变地理方向的翻转或旋转；
- 每个 frame time 都必须小于或等于 issue time，违反时直接报错。

训练帧统计 hash：
`81d7b2d9d77ee06e865e875008e90cc040d77eb900f014a2324f78c96f456efa`。

## 3. 真实加载基准

运行命令：

```bash
python scripts/benchmark_satellite_loader.py
```

配置与结果：

| 项目 | 数值 |
|---|---:|
| batch size | 8 |
| workers | 2 |
| requested/read batches | 1,000 / 1,000 |
| samples | 8,000 |
| 读取时间 | 114.62 s |
| 吞吐 | 69.80 samples/s |
| train-only 统计耗时 | 23.08 s |
| test 是否读取 | 否 |

该基准覆盖连续 1,000 个真实 batch 的稳定性和吞吐；当前脚本没有采集逐批进程内存曲线，
因此不把“无持续内存增长”作为已经量化证明的结论。

## 4. Optical Flow 方法

对四个历史 `IR_108` 帧的三个相邻帧对，使用 FFT phase correlation 估计全图平移，并提取
`dy`、`dx`、速度以及方向的正弦和余弦，共 15 个特征。它们与历史功率、未来已知太阳
特征和静态站点特征拼接，再由逐时效 masked Ridge 输出六个预测。

Ridge 的 `alpha` 只在 validation/development 上从 `0.1、1、10` 中选择，最终为 `1.0`。
所有光流特征持久缓存到本地输出目录，重复运行时 train 和 validation 缓存均已验证命中。
真实帧示例可通过以下命令生成：

```bash
python scripts/plot_optical_flow_example.py
```

## 5. 真实验证结果

下表为六个时效 MAE 的简单平均；逐时效仍使用统一的有效 target mask。

| 分区 | Persistence | Smart Persistence | Power + Solar Ridge | Optical Flow Ridge |
|---|---:|---:|---:|---:|
| validation/development | 0.022419 | 0.014869 | 0.015777 | 0.017244 |
| validation/holdout | 0.031501 | 0.020110 | 0.024497 | 0.026113 |

Optical Flow Ridge 相对比较：

| 分区 | 对 Persistence | 对 Smart Persistence | 对 Power + Solar Ridge |
|---|---:|---:|---:|
| validation/development | 改善 23.1% | 退化 16.0% | 退化 9.3% |
| validation/holdout | 改善 17.1% | 退化 29.9% | 退化 6.6% |

逐时效 MAE：

| 分区 / 模型 | 15m | 30m | 60m | 120m | 180m | 240m |
|---|---:|---:|---:|---:|---:|---:|
| development / Power + Solar | 0.006721 | 0.010624 | 0.014299 | 0.019490 | 0.021446 | 0.022081 |
| development / Optical Flow | 0.007222 | 0.011517 | 0.015590 | 0.021253 | 0.023649 | 0.024233 |
| holdout / Power + Solar | 0.009383 | 0.015014 | 0.021085 | 0.030505 | 0.034559 | 0.036434 |
| holdout / Optical Flow | 0.009943 | 0.016003 | 0.022439 | 0.032415 | 0.037005 | 0.038874 |

光流在所有时效均弱于相同 Ridge 中不含光流的对照，说明负增益不是某个单独时效造成的。
合理解释包括：全图单一平移无法表示局部云形变；相位相关会受到边界、低纹理和多层云影响；
15 个全局摘要又丢失了云相对站点位置的信息。这些是下一步 CNN 验证的假设，不是已经由
本实验单独证明的因果结论。

## 6. 验证范围与复现

- 合成图像验证平移方向和量级；
- 验证四帧一致增强、train-only 统计和未来帧拒绝；
- 验证 DataLoader schema、pickle 后懒打开及多 worker 样本唯一性；
- 验证 masked Ridge、缓存复用和未拟合错误路径；
- test 分区未读取、未评估，也未用于选择 `alpha`。

主要入口：

```bash
python scripts/run_optical_flow_baseline.py
pytest
```
