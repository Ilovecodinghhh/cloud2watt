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
