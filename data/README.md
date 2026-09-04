# 数据目录

此目录只记录数据规范，不提交大体积或受许可限制的原始数据。

运行数据管道后可使用以下本地目录：

- `raw/`：原始卫星影像、功率与气象数据，只读保存；
- `interim/`：完成格式转换和时空对齐的数据；
- `processed/`：可直接训练的样本和数据划分。

每份数据应记录来源 URL、版本或检索时间、许可证、空间范围、时间范围、时区、单位、缺测规则与校验和。训练/验证/测试集必须按时间划分，并在任何统计量拟合前固定边界。

## Data Access Spike

安装可选的数据依赖：

```bash
pip install -e ".[data,dev]"
```

### UK PV

1. 在浏览器打开 [openclimatefix/uk_pv](https://huggingface.co/datasets/openclimatefix/uk_pv)，接受 gated dataset 条件；
2. 创建只读 Hugging Face token；
3. 只在当前终端配置 token，不写入配置文件：

```bash
export HF_TOKEN="你的只读令牌"
```

先读取 metadata、bad-data 清单，并列出与卫星数据重叠的 2020 年中最小的 5 分钟分区：

```bash
python scripts/check_uk_pv_access.py
```

确认报告中的大小合理后，才下载最小分区并检查 Parquet schema：

```bash
python scripts/check_uk_pv_access.py --download-partition
```

### SEVIRI RSS

卫星元数据和一个压缩块可匿名探测，不需要 Google 凭据：

```bash
python scripts/inspect_satellite.py
```

生成一个以目标经纬度为中心、带红色位置标记的单通道预览：

```bash
python scripts/inspect_satellite.py --preview --latitude 52.5 --longitude -1.5
```

默认经纬度只用于访问验证，不代表正式 UK PV 站点。取得 `metadata.csv` 后应传入真实选定站点的模糊化坐标。

探测报告默认写入 `outputs/probes/`，该目录已被 Git 忽略。脚本不会下载完整年度 Zarr。

### 退出码

- `0`：访问与预期 schema 均通过；
- `1`：公开卫星端点、网络或 schema 失败；
- `2`：UK PV 因缺 token、未接受条款、缺依赖或 schema 不符而阻塞。

任何日志和异常在落盘前都会清理常见 Hugging Face token 格式。不要截图或提交包含环境变量值的终端输出。

## Mini dataset

第二阶段从一个本地 UK PV 月分区构建确定性的 20 站点、7 天开发数据集。源数据保留在
Hugging Face 缓存中，派生数据写入已忽略的 `data/processed/` 目录。

```bash
python scripts/build_mini_dataset.py \
  --power "/path/to/5_minutely/year=2020/month=12/data.parquet" \
  --metadata "/path/to/metadata.csv" \
  --bad-data "/path/to/bad_data.csv" \
  --source-revision "Hugging-Face-commit-SHA" \
  --power-value-semantics interval_energy \
  --start 2020-12-01 \
  --output data/processed/mini-v1
```

输出包括：

- `manifest.json`：数据源、精确的 `(start, end]` UTC 窗口、站点选择及 QC 统计；
- `sites.parquet`：选定站点的坐标和容量元数据；
- `power_15min.parquet`：按右边界标记的 15 分钟功率、归一化值和质量标志。

站点按窗口内 5 分钟数据覆盖率选择，覆盖率相同时按 `ss_id` 排序，因此结果可复现。
仓库内的 `tests/fixtures/mini_manifest.json` 使用合成标识符验证 20 站点、7 天契约，
不重新分发 gated 数据。

数据卡目前同时把 5 分钟值描述为瞬时读数和区间能量。Mini-v1 暂按字段级换算说明，
将 `generation_Wh` 视为区间能量，并在 manifest 中明确记录此假设；正式报告模型结果前
仍需与 OCF 核实。卫星张量遵循 `ForecastSample` schema 并写入分块 Zarr，标量样本索引
写入 Parquet。站点中心裁剪在越界时明确失败，不静默补边或移动中心。

## 真实卫星—功率配对数据

在完成单源探测后，使用一个明确的共同年份构建真实配对数据。以下示例中的路径应替换为
本机 Hugging Face 缓存位置：

```bash
python scripts/build_paired_mini_dataset.py \
  --power "/path/to/5_minutely/year=2020/month=12/data.parquet" \
  --metadata "/path/to/metadata.csv" \
  --bad-data "/path/to/bad_data.csv" \
  --pv-revision "Hugging-Face-commit-SHA" \
  --power-value-semantics interval_energy \
  --issue-start auto \
  --output data/processed/paired-mini-v1
```

脚本会：

- 选择一个包含 20 个高覆盖率站点的紧凑空间区域；
- 精确匹配 15 分钟 SEVIRI 帧，不用未来帧填补缺测；
- 下载并缓存所需远程 Zarr chunk，网络中断时有限重试并可续传；
- 保存去重的卫星帧和引用它们的样本索引；
- 检查全部历史与目标时间引用；
- 输出 100 个样本的读取延迟以及 5 站点投影预览。

真实输出和远程缓存分别位于 `data/processed/` 与 `outputs/cache/`，两者均被 Git 忽略。
决策和实测结果见 [ADR-002](../docs/adr/002-paired-mini-dataset.md)。
