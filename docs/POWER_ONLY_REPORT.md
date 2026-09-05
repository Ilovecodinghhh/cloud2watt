# Power-only 模型实验报告

状态：训练框架已完成，正式多种子结果待运行  
最后更新：2026-09-05

## 实验目的

Power-only 阶段用于建立不读取卫星图像的学习模型上限。模型只接收过去 60 分钟功率、
未来可知的太阳特征和静态站点元数据，并与 Persistence、Smart Persistence 和
Climatology 使用相同数据切分与指标协议。

## 已实现内容

- `PowerMLP` 和因果 `PowerTCN`，参数量均小于 1M；
- masked 六时效等权 MAE、梯度裁剪、早停和检查点恢复；
- 只基于 train/development 拟合的太阳及站点标准化统计量；
- validation/development checkpoint 选择，test 默认锁定；
- 运行级配置、Git/data/split/statistics hash、预测及逐时效指标；
- CPU 冒烟、输入白名单、防泄漏和检查点一致性测试。

## 运行方法

开发阶段单种子运行：

```bash
python scripts/train_power_model.py --model tcn --seed 42
```

模型和分析方案冻结后，才允许最终测试：

```bash
python scripts/train_power_model.py --model tcn --seed 42 --unlock-test
```

正式比较应对胜出配置分别运行 `42、123、2026` 三个种子，并将结果补充到本报告。当前
文件不预填未经运行的模型性能，也不据 30 天数据声称跨季节泛化。
