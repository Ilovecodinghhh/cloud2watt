# 可解释 MVP：前四步验收

日期：2026-10-08。决策：ADR 013。范围：历史回放推理与解释，非在线生产服务。

## 交付

- src/cloud2watt/mvp.py：默认主模型、实验卫星分支、太阳特征本地计算、质量检查、回退与解释图。
- scripts/run_local_mvp.py：读取 JSON 请求并输出 forecast.json、explanation.png；无请求时选择首个冬季 outer 白天案例，显式开启卫星以展示全部对照。
- scripts/validate_local_mvp.py：三种子、日期块置信区间和卫星打乱检查。
- scripts/accept_local_mvp.py：真实 checkpoint 推理一致性和缺测行为验收。
- outputs/project-v2/mvp-v2/：登记、模型、逐轮历史、预测、validation.json、acceptance.json、示例和历史修订记录。

## 结果

指标为容量归一化功率白天 30/60/120 分钟 MAE 的平均值。差值为卫星减主模型，负数代表卫星更好。

| 角色 | 三种子平均差值 | 日期块 95% 区间 | 日期数 |
|---|---:|---|---:|
| 冬季 outer 开发 | +0.000287 | [-0.003368, +0.002984] | 6 |
| 春季压力开发 | -0.016428 | [-0.031535, -0.005688] | 5 |

冬季三个种子的主模型/卫星 MAE 分别为 0.049151/0.048310、0.048518/0.048705、0.047856/0.049369。初轮小幅收益未稳定复现。春季卫星均优于对应主模型，但所有卫星种子仍劣于晴空持续性基线 0.085775。春季区间仅是短开发窗口上的探索性证据，不能称为独立泛化显著性。

同站点同钟点打乱后，冬季三种子的平均 MAE 为 0.056347、0.057126、0.057898，均比原卫星预测差。春季为 0.101760、0.105591、0.104014，其中一个种子打乱后反而略好。这说明依赖卫星不等于稳定受益。默认保持功率＋太阳，卫星仅手动实验开启。

## 功能验收

1. 六项自动测试通过：默认模型和解释差值、卫星正常与缺测回退、功率缺测/未来时间拒绝、非法元数据、日期块常量效应与不等样本量加权。
2. 24 个真实模型案例覆盖 inner、冬季 outer、春季：本地太阳重算后的输出与训练预测一致，最大卫星 CPU/GPU 差异约 1.34e-7。
3. 缺卫星的输出与主模型逐值相同；缺功率历史不生成预测。预测接口不依赖目标标签。
4. JSON 显示预测时刻、历史时间、质量原因、所选模型、哈希和非因果说明；解释图显示历史功率、六个预测时效、训练模型对照及独立纵轴的晴空辐照（不是光伏功率）。已人工检查图片布局。
5. 训练使用 raincalib-gpu Conda 的 RTX 5070 Ti CUDA；接口默认 CPU 推理，以便轻量运行。没有新增下载，最终标签保持锁定。

## 运行

在项目根目录设置 PYTHONPATH=src 和 TORCHINDUCTOR_CACHE_DIR=D:/code/AI_atm/outputs/project-v2/torchinductor-cache，使用 C:/Users/admin/miniforge3/envs/raincalib-gpu/python.exe 运行：

```text
scripts/run_local_mvp.py
scripts/run_local_mvp.py --request outputs/project-v2/mvp-v2/example/request.json --output outputs/project-v2/mvp-v2/custom-example
scripts/accept_local_mvp.py
-m pytest tests/test_local_mvp.py tests/test_mvp_statistics.py -q
```

request.json 中 enable_satellite=false 为默认主模型，true 才尝试卫星。历史功率使用容量归一化值；卫星特征为既定四帧三通道局部统计及帧差的 105 维向量，不应输入任意像素展开值。history_valid / satellite_valid 接收四个质量布尔值；省略时只做有限值、形状、时间及范围检查，调用方需提供已质控输入。

解释是不同已训练模型的预测对照，不是因果贡献，也不保证差异相加还原输出。当前评估样本来自共同完整数据，实际缺测回退已做行为测试，但尚未证明真实缺测人群上的预测精度。全年、跨区域和独立最终泛化仍未验收。
