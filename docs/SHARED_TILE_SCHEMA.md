# 共享卫星 tile schema 与迁移

schema：cloud2watt-shared-tiles-v1。实施于 S2，作为有界存储验证版本。

## 布局

| 字段/数组 | 含义 |
|---|---|
| attrs.schema / complete | schema ID；只有迁移完整写出后 complete 才为 true |
| attrs.tile_size / time_chunk | 全局网格 tile 边长与时间块长度，当前 128 / 12 |
| attrs.site_ids / site_origins | 原站点顺序、裁剪左上角全局 y/x 索引 |
| attrs.crop_shape / channel_names | 原裁剪尺寸及显式通道顺序 |
| attrs.geometry | historical_indices_preserved；迁移不修正地理定位 |
| time_ns | 有序 UTC 归档帧时间，纳秒整数 |
| source_frame_indices | 当前帧到源存储的索引映射 |
| tiles/y_x | [time, channel, tile_y, tile_x]，继承源 dtype；NaN 表示缺失像素 |
| coverage/y_x | 静态空间覆盖布尔图，区分未存储区域与观测缺失 |

tile 原点是全局像素网格的整数倍，因此不同站点共用重叠像素。跨 tile 裁剪按交集拼接，
不插值、不改变通道顺序。halo 在空间覆盖足够时读取邻 tile；超出覆盖立即拒绝，不能通过补零伪造上下文。
源域边界之外的裁剪拒绝。全缺帧保留 NaN，由现有 DataLoader 输出 false mask 和零归一化输入。

SharedFrames 提供与旧帧数组一致的 site/time 索引；每个 worker 最多缓存 16 个解码时间块。
标准 float16、3 通道、128²、12 帧配置下，解码缓存上限约 18 MiB/worker，不包括返回张量与预取队列。
现有 SatelliteForecastDataset 和训练统计函数自动识别两种布局；worker pickle 清除活动读取器。

## 迁移步骤

1. 保留旧目录；使用一个尚不存在的输出路径调用 migrate_site_frames。
2. 核对站点顺序、时间索引、裁剪边界和通道。多个站点在同一像素上的值必须完全一致（NaN 对 NaN 一致）。
3. 逐时间块写入共享空间并验证重叠；发生错误保留 incomplete 目录供诊断，不能被读取器接纳。
4. 对比像素、训练统计量、完整 DataLoader 输出及样本/目标键后再使用新布局。
5. 子窗口迁移的样本索引必须按 source_frame_indices 重映射。本次取前 192 帧，映射恰为恒等；
   非前缀窗口不能照搬旧 frame indices。迁移函数只搬卫星帧，不自动改写整套数据 manifest。
6. 新完整数据版本必须同时冻结时间与标签契约、数据来源、修正后的投影和预算。
   不允许把本次历史几何迁移目录直接改名为 S3 正式数据。

示例工具：scripts/benchmark_shared_tiles.py；它只选择两天前缀窗口，做迁移、一致性检查和加载基准，
不训练模型、不评估 test。完整源构建当前仍由既有构建器生成站点帧，再迁移；尚未实现直接从远程源写共享 tile。
因此年度构建前仍需消除全量旧布局中间文件，并落实有上限的源缓存策略。

当前迁移在每个时间块同时持有该区域的 tile，内存随空间范围增加；本次只有 4 tiles。
大范围迁移必须分区域运行，不能把全年离散站点一次性装入本实验迁移器。

## 可用时间契约

availability_contract 生成 observation_time、available_time、availability_basis。
delayed_history_indices 对每个 issue 按 [-45,-30,-15,0] 减去情景延迟选择精确槽，
同时核查 available_time ≤ issue_time；不存在的槽输出 -1。
这只是卫星延迟；功率观测发布延迟仍未知。现有训练 Dataset 保持零延迟严格对齐，
延迟情景索引不能未经显式适配直接送入旧训练入口。
