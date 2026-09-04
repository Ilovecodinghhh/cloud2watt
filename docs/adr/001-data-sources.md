# ADR-001：MVP 数据源与访问策略

- 状态：已接受，等待完成有凭据的 UK PV 探测
- 日期：2026-09-04

## 背景

Cloud2Watt 需要在相同地点和时间上配对光伏功率与卫星云图。完整卫星档案达到 TB 级，UK PV 数据需要用户接受访问条件，因此必须在模型开发前验证访问、许可、格式、时间语义和资源成本。

## 决策

MVP 采用英国路线：

1. 光伏标签使用 Open Climate Fix 的 `openclimatefix/uk_pv`；
2. 卫星输入使用 OCF 公开的 EUMETSAT SEVIRI RSS 2020/2021 non-HRV Zarr；
3. 所有正式下载均固定到不可变 revision 或记录远端 ETag；
4. 先读取 metadata，再选取最小功率分区和单个卫星块；
5. 不下载完整卫星年度档案，只缓存研究区域或站点周边裁剪；
6. 原始数据、访问令牌和探测输出不提交 Git。

## 已验证事实

2026-09-04 的匿名元数据和数据块探测已成功。探测确认 2021 卫星存储为 Zarr v2，主数组维度为：

```text
time × y_geostationary × x_geostationary × variable
88225 × 1392 × 3712 × 11
```

主数组块大小为 `12 × 100 × 100 × 11`，数据类型为 float16，压缩器 ID 为 `blosc2`。英国附近的测试块压缩后约 0.98 MB，解码后约 2.64 MB；`ocf-blosc2 0.0.14` 与 Zarr 3 依赖组合安装成功，并成功生成带目标位置标记的可见光预览。因此按单块访问能够把一次探测限制在数 MB 量级。`uk_pv` 凭据尚未配置，功率数据的实际文件 revision、最小分区和站点数量仍需运行脚本确认。

## 原因

- UK PV 提供容量、约 1 km 精度的位置和高频发电数据，适合站点级研究；
- OCF 的卫星数据覆盖同一区域和项目时间范围；
- OCF 已公开 PVNet 与数据采样设计，可用于核对语义；
- 该组合最接近真实业务问题，同时可以通过空间裁剪控制单机资源。

## 后果

- 用户需要在浏览器接受 Hugging Face gated dataset 条件并配置只读 token；
- 卫星数据的 `blosc2` codec 需要 `ocf-blosc2`，不能假设任意 Zarr 后端均兼容；
- 站点位置经过约 1 km 模糊，必须做空间扰动敏感性分析；
- 数据卡对 5 分钟字段存在瞬时读数与区间能量两种描述，在确认前不得擅自乘以 12 或改变单位；
- EUMETSAT 数据必须保留归属说明，原始数值数据不在本仓库再分发。

## 备用方案

若 OCF 公共 Zarr 无法稳定解码，则使用 EUMETSAT 注册账号下载有限日期的 SEVIRI 历史数据并本地转换。若站点级 UK PV 无法获得，则改用 Sheffield Solar PV Live 的 GSP 区域级标签。

## 参考

- [OCF UK PV](https://huggingface.co/datasets/openclimatefix/uk_pv)
- [OCF PVNet](https://github.com/openclimatefix/PVNet)
- [Hugging Face gated datasets](https://huggingface.co/docs/hub/datasets-gated)
- [EUMETSAT data registration and licensing](https://user.eumetsat.int/resources/user-guides/data-registration-and-licensing)
