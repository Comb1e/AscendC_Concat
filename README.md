# Concat Ascend C 自定义算子

本项目实现比赛规格中的 `Concat` 算子，目标芯片为 Ascend 910B，构建工具链为 CANN 8.5。算子使用动态 `tensor_list` 输入，并与 `torch.cat` 的语义保持一致。

## 支持范围

- Rank：1 到 4。
- DType：`float32`、`float16`、`int32`、`int8`。
- Format：连续 `ND`。
- `dim`：支持正轴和负轴。
- 支持动态输入数量、不同的拼接轴长度和零长度输入分片。
- 支持片段长度和尾块不是 32 字节整数倍的场景。

PyTorch/ACLNN 验证封装必须调用本项目生成的 `aclnnConcat`，不能误用内置 `aclnnCat`。

## 目录

```text
op_host/
  concat.cpp          算子注册、Shape/DType 推导和 Host Tiling
  concat_tiling.h     Host 与 Kernel 共享的 Tiling 数据
op_kernel/
  concat.cpp          Ascend C AIV Kernel
build.sh              基于本机 CANN 模板生成 custom_*.run
CONCAT_TECHNICAL_DOCUMENT.md
                      初始正确性设计与接口分析
```

全局 Codex skill 位于：

```text
/root/.codex/skills/ascendc-operator-tuning
```

可通过 `$ascendc-operator-tuning` 调用，用于后续 Ascend C 算子开发、审查、DMA/UB 调优和 NPU 性能试验设计。
此前已补充 L2 策略应按输入/输出生命周期分别实验、性能复现需要匹配分配器和中间算子的
缓存行为、算子规格与 ACLNN 动态列表上限需要分开确认，以及“0 target tasks”通常是启动前
错误的连带结果。本轮进一步记录：对齐输出行大于 UB 时，可按输出 tile 与输入前缀区间求交
继续融合写回，但保持活跃核数仍不足以保证收益，还必须核算逐 tile 的 Scalar 映射和访问局部性；
Host 已下发 bounded 输入的精确片段长度时，Kernel 应直接复用并保留超限描述符回退；框架每轮
创建新 Tensor 不代表物理输出地址只写一次，缓存分配器可能复用存储并反转 L2 实验结果；本地
诊断结果只能作为同环境后续提交的基线，不能直接解释隐藏 Case。更新后的 skill 已通过
`skill-creator` 的 `quick_validate.py` 校验。本轮新增经验是：动态输入的 chunk 调度应统计
`coreCount * inputCount` 级别的无效描述符触达；Host 可在 tiling 容量允许时下发每核连续
work 区间，并把等字节目标吸附到既有 work-item 边界，从而不增加 DMA 命令。

## Kernel 地址模型

对于沿 `dim=d` 拼接的输入 `i`：

```text
outerSize = product(shape[0:d])
innerSize = product(shape[d+1:rank])
segmentBytes_i = shape_i[d] * innerSize * elementBytes
outputRowBytes = sum(segmentBytes_i)
```

固定 outer 行 `o` 时：

```text
sourceOffset = o * segmentBytes_i
destinationOffset = o * outputRowBytes + sum(segmentBytes_j, j < i)
```

Concat 不做数值计算，Kernel 使用 `uint8_t` 原始字节搬运统一覆盖四种 DType。非对齐尾块使用精确长度的 `DataCopyPad`，不会向后覆盖相邻输入片段。

## 本轮性能优化

优化前五个隐藏样例耗时为：

```text
18.024, 31.668, 21.568, 110.3, 1643.632 us
```

每项优化均保存为独立、累计的 Git commit，便于在 NPU 环境逐项测试和回退。

| Commit | 更改 | 主要目标 | 主要风险/验证点 |
| --- | --- | --- | --- |
| `5b55590` | 将动态 Tensor 描述符解析移出 outer 循环 | 减少大 outer 或多输入场景的 Scalar 开销 | 核对循环交换后的输出前缀地址 |
| `5a1b9ea` | 按行/按 chunk 的实际活跃核数选择调度；chunk 核只遍历自己的线性工作项 | 少量大行也能使用更多 AIV，避免所有核扫描全部 outer | 核对 work-item 到 outer/chunk 的唯一映射 |
| `0b45882` | 对一次性读取的输入设置 L2 Cache bypass | 减少大规模流式读取的 L2 污染 | 重复使用同一输入或上下游复用时可能回退 |
| `cc66c1c` | 在不降低核占用率时将 tile 从 32 KiB 提升到 64 KiB | 减少 DMA 指令和启动开销 | 大 tile 不能减少有效并行工作项 |
| `355dd77` | 每核改为连续均衡 outer 区间；在 UB 容量内合并多行 `DataCopyPad` | 减少小 inner、大 outer 场景的逐行 DMA 与循环开销 | 910B 非对齐 UB 行按 32B 计空间，blockCount 不超过 4095，stride 超限时回退单行 |
| `8d477b3` | Host 下发 `allSegmentsAligned`；全对齐场景使用普通 `DataCopy` | 避免对齐输入承担 `DataCopyPad` 开销 | 任一片段非对齐时必须完整保留 Pad 路径 |
| `75eec09` | 移除输入 L2 bypass，恢复默认缓存策略 | 适配官方脚本对同一输入连续执行 30 次的复用模式 | 一次性流式消费场景可能不受益 |
| `767c220` | 使用 `desc.GetDataPtr()` 复用 `GetDesc` 已解析的数据指针 | 每核、每输入减少一次 TensorList 指针 GM 读取 | 预期主要改善小数据或多输入的 Scalar 开销 |
| `b041908` | 对齐行在 UB 内拼接多个输入后整批连续写回 | 将每批行的 MTE3 次数从输入数降为 1 | 仅用于行调度、最多 16 输入、整行不超过 tile 的全对齐场景 |
| `b2ecea2` | 保留输入默认缓存，仅让输出绕过 L2 | 验证瞬时输出是否会挤占复用输入的缓存 | 官方测试总耗时回退 38.29%，已判负 |
| `b717f0a` | 删除输出 L2 bypass，恢复 `b041908` 的默认缓存策略 | 消除所有 Case 的一致性回退 | 保留更贴近官方生命周期的测试封装 |
| `618125d` | 对齐大行按输出 tile 在 UB 中组装后单次写回 | 合并跨输入边界的 MTE3，重点改善大搬运 Case | 仅在不减少活跃 AIV 数时启用 |
| `d6a5a57` | 删除对齐大行输出 tile 融合，恢复历史最优 Kernel | 消除逐 tile 输入扫描和区间计算 | Host/Kernel 与 `b717f0a` 完全一致 |
| `2c97dca` | 16 输入以内复用 Host 预加载片段长度，融合行每核缓存输入地址 | 减少 `GetDesc`、shape 乘积和重复 TensorList 地址读取 | 多输入通用路径保持原实现 |
| `98cc5f6` | 将 Host 片段长度预加载范围从 16 扩展到 32 | 覆盖 17～32 输入的常见动态列表边界 | Tiling 数据增大；需验证中等输入数是否受益 |
| `f3aa07d` | 大型全对齐输出行重新启用 tile 融合，但按核连续分配 tile 并递增维护输入边界 | 在大行场景减少 MTE3 写回，同时消除上一版逐 tile 重复前缀扫描 | 高风险；仅输出至少 8 个 tile 时启用，需优先检查精度和 Case5 |
| `0189f03` | 在 `local_test` 增加输入数、零长度、Rank4 首轴和 tile 尾块边界 Case | 快速暴露预加载上下限、零片段和二维地址错误 | 本地诊断不代表官方隐藏样例 |
| `375a67a` | 本地结果增加均值，并自动生成包含 commit 和所有 Case 的 Markdown 简表 | 固化同环境比较口径，避免手工转录和版本标签歧义 | HEAD 默认代表源码；测试旧安装包时需显式传实际 commit |
| `8dc7a78` | 显式回退 `f3aa07d`，保留 32 输入预加载 | 隔离第二次 mode 2 负反馈 | 32 输入扩展仍需单独上板归因 |
| `9f07f4a` | 在保持活跃核数时将普通搬运 tile 扩展到 96 KiB，并保留 64/32 KiB 回退 | 利用 910B 的 192 KiB UB，减少大流式 Case 的 DMA 和循环次数 | 两个 96 KiB Buffer 用满 UB；必须验证 Kernel 编译、阈值和大数据性能 |
| `0e38813` | 显式回退 `9f07f4a` | 恢复 64/32 KiB tile 和双 Buffer 余量 | 保留 32 输入预加载及 mode 2 回退 |
| `b702aec` | 保持 chunk 数不变，在每个输入片段内部均衡 chunk 字节数 | 避免略大于 tile 的片段产生大块与极小尾块、造成核间字节失衡 | chunk 路径每个多块片段增加均分计算 |
| `d430bc0` | 单 chunk 和零长度片段跳过均分除法 | 避免不受益输入承担新增 Scalar 成本 | 多 chunk 路径保持 `b702aec` 行为 |
| `6ad3bd3` | Host 按字节目标预计算每核连续 chunk 区间，Kernel 只遍历实际覆盖的输入 | 删除 chunk 模式下 `coreCount * inputCount` 的无效列表扫描 | 本地长列表有效，但官方总耗时回退 2.74%，已判为不适合隐藏样例 |

### 首轮 NPU 反馈与原因分析

累计版本 `8d477b3`（测试时仓库 HEAD 为文档提交 `3f17e4d`）五例均通过精度，但性能为：

| Case | 原始基线/us | `8d477b3`/us | 差值/us | 变化率 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 18.024 | 19.804 | +1.780 | +9.88% |
| 2 | 31.668 | 34.172 | +2.504 | +7.91% |
| 3 | 21.568 | 21.388 | -0.180 | -0.83% |
| 4 | 110.300 | 113.036 | +2.736 | +2.48% |
| 5 | 1643.632 | 1653.960 | +10.328 | +0.63% |
| 合计 | 1825.192 | 1842.360 | +17.168 | +0.94% |

这不是某个大数据 Case 独有的退化：除 Case3 的 `0.18 us` 小幅波动外，其余规模均变慢。
最先隔离全局作用的 L2 bypass，因为测试会对同一批输入连续调用算子，默认 L2 缓存可能保留
输入数据；强制 bypass 会让每次调用重新访问 HBM，并且 Case5 的绝对退化最大。本提交仅恢复
默认 L2 策略，其余五项优化保持不变，作为下一轮可独立归因的测试点。

### 上一轮调优的官方结果与分析

官方比赛系统对恢复默认 L2 的累计算子版本 `75eec09` 测得五例精度全部通过：

| Case | 原始基线/us | `75eec09`/us | 差值/us | 变化率 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 18.024 | 14.500 | -3.524 | -19.55% |
| 2 | 31.668 | 34.060 | +2.392 | +7.55% |
| 3 | 21.568 | 21.816 | +0.248 | +1.15% |
| 4 | 110.300 | 105.072 | -5.228 | -4.74% |
| 5 | 1643.632 | 413.720 | -1229.912 | -74.83% |
| 合计 | 1825.192 | 589.168 | -1236.024 | -67.72% |

总耗时降低到原来的 `32.28%`，约为 `3.10x` 加速。Case5 从首轮强制 L2 bypass 时的
`1653.960 us` 降到 `413.720 us`，证明该 Case 会从跨调用缓存复用中显著受益，也说明
多行 DMA、对齐快路径和调度优化此前被错误的缓存策略掩盖。Case2、Case3仍分别比原始
正确性基线慢 `2.392 us` 和 `0.248 us`。

当前满意目标为总耗时不超过 `540 us`，相对 `589.168 us` 还需减少约 `49.2 us`
（`8.35%`）。Case5 占当前总时间的 `70.22%`，只消除 Case2/Case3的回退不足以达到目标，
下一轮仍必须改善大搬运 Case，同时保持输入的默认 L2 缓存。

### 指针复用与对齐行融合轮的官方结果

官方比赛系统对累计包含 `767c220` 和 `b041908` 的算子版本测得五例精度全部通过：

| Case | `75eec09`/us | 当前版本/us | 差值/us | 变化率 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 14.500 | 13.590 | -0.910 | -6.28% |
| 2 | 34.060 | 33.8905 | -0.1695 | -0.50% |
| 3 | 21.816 | 21.990 | +0.174 | +0.80% |
| 4 | 105.072 | 106.822 | +1.750 | +1.67% |
| 5 | 413.720 | 402.688 | -11.032 | -2.67% |
| 合计 | 589.168 | 578.9805 | -10.1875 | -1.73% |

相对原始 `1825.192 us` 基线，当前总耗时减少 `1246.2115 us`（`68.28%`），约为
`3.15x` 加速。主要绝对收益来自 Case5，其单项改善 `11.032 us`，大于总收益是因为
Case3/Case4 合计回退 `1.924 us`。Case1 也有 `6.28%` 的相对改善。

当前结果是两个 Kernel 改动的累计测量，单凭官方总表不能把每个 Case 的收益精确拆分到
某一个 commit。不过 Case2 的测试形状和拆分已由参考目录确认：9 个 FP16 末轴片段均不是
32B 对齐，因此不会进入 `b041908` 的对齐行融合路径。Case2 仍改善 `0.1695 us`，与
`767c220` 减少动态列表指针读取的方向一致。Case5 的 `11.032 us` 改善与融合写回减少
MTE3 指令的预期一致，但隐藏形状未知，当前只能作为强推断；需要 `row_aligned` 本地
profiling 或隔离测试 `767c220` 才能完成严格归因。

Case4 回退 `1.750 us`，是本轮最明确的负项。由于 Case3 的 `0.174 us` 与 Case2 的
`0.1695 us` 接近小 Case 测量波动量级，下一轮不为这两个亚微秒变化增加复杂分支；继续
以 Case5 为主，同时观察 Case4 是否持续回退。Case5 目前占总耗时 `69.55%`。距离
`540 us` 满意目标还差 `38.9805 us`（当前总耗时的 `6.73%`）；若仅由 Case5 提供收益，
Case5 还需降低约 `9.68%`。

### 已验证优化的设计

`767c220` 根据 CANN 8.5 的 `ListTensorDesc::GetDesc` 实现，改用描述符自身的
`desc.GetDataPtr()`。`GetDesc` 已将数据地址写入描述符，原代码随后调用
`inputs.GetDataPtr(inputIdx)` 会再次从动态列表中读取同一地址。该改动不改变地址映射、
DMA 或调度，只减少每核、每输入一次元数据 GM 访问。

`b041908` 在此基础上增加全对齐行融合路径。满足以下全部条件时，Kernel 将一批行的多个
输入片段通过 MTE2 放到各自在 UB 中的最终位置，再对连续 UB 区间只执行一次 MTE3 写回：

- 使用 row schedule；
- 输入数不超过 16，Host 已下发每个 `segmentBytes`；
- 所有片段均为 32B 整数倍；
- `outputRowBytes <= tileBytes`。

新路径中第 `r` 行第 `i` 个输入的 UB 地址仍是
`r * outputRowBytes + sum(segmentBytes_j, j < i)`。单批最多 64 KiB，普通 `DataCopy`
的 block length、block count 和 stride 均不会越过当前 910B 路径的字段范围。非对齐、
超过 16 个输入、超大行及 chunk schedule 完整回退到已有实现。该优化与 CANN 8.5
内置 Concat 的“多个输入组装到同一 UB 后合并写回”策略一致。当前累计版本已在 910B
官方系统通过精度并获得净收益，但仍需隔离提交才能严格区分它与指针复用各自的贡献。

### 输出 L2 bypass 实验结果与回退分析

官方比赛系统对仅增加输出 `CACHE_MODE_DISABLE` 的 `b2ecea2` 测得五例精度全部通过，但性能为：

| Case | `b041908` 累计版本/us | `b2ecea2`/us | 差值/us | 变化率 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 13.590 | 13.6005 | +0.0105 | +0.08% |
| 2 | 33.8905 | 43.511 | +9.6205 | +28.39% |
| 3 | 21.990 | 25.431 | +3.441 | +15.65% |
| 4 | 106.822 | 136.0125 | +29.1905 | +27.33% |
| 5 | 402.688 | 582.102 | +179.414 | +44.55% |
| 合计 | 578.9805 | 800.657 | +221.6765 | +38.29% |

除 Case1 基本持平外，其余四项均明显变慢，且绝对回退随 Case 耗时增大；Case5 一项就贡献
`80.94%` 的总回退。该结果否定了“绕过瞬时输出可为复用输入保留更多 L2”这一假设，说明默认
输出缓存对当前 910B/CANN 8.5 搬运路径非常重要。可能原因包括禁用缓存直接降低 MTE3 有效
写带宽，以及 PyTorch NPU 缓存分配器让逻辑上每轮新建的输出复用相同物理存储，因而输出并非
真正的一次性地址。当前只有端到端耗时，无法在没有 `L2Cache.csv` 和 MTE3 计数器时严格区分
这两种机制。

由于五个 Case 均无可靠收益，本轮按预先设定的停止条件完整撤销输出 L2 hint，恢复
`b041908` 累计版本的输入、输出默认缓存策略。后续不再把缓存策略与其他 Kernel 改动叠加，
`578.9805 us` 继续作为下一项独立优化的比较基线。

### 对齐大行输出 tile 融合的官方结果与回退

官方比赛系统对包含 `618125d` 的累计版本测得五例精度全部通过，但性能为：

| Case | 历史最优 `b041908` 累计版本/us | `618125d`/us | 差值/us | 变化率 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 13.590 | 13.880 | +0.290 | +2.13% |
| 2 | 33.8905 | 35.148 | +1.2575 | +3.71% |
| 3 | 21.990 | 21.920 | -0.070 | -0.32% |
| 4 | 106.822 | 108.512 | +1.690 | +1.58% |
| 5 | 402.688 | 414.000 | +11.312 | +2.81% |
| 合计 | 578.9805 | 593.460 | +14.4795 | +2.50% |

Case5 一项贡献 `78.12%` 的总回退；Case3 的 `0.07 us` 改善属于小幅波动，不能抵消其余四项
的一致退步。虽然 mode 2 只在新旧方案活跃 AIV 数相同时启用，但保持核占用只是必要条件，
不是充分条件。新路径将每个输出 tile 作为工作项，每次都需要执行 work-item 除法、扫描输入
前缀、区间求交和多处分支。当 tile 只覆盖一个或少数输入时，减少的 MTE3 命令不足以抵消新增
Scalar 控制开销；按输出 tile 交错访问不同输入还可能弱化旧 chunk 路径按输入连续搬运的局部性。

已知 Case2 为 9 个非 32B 对齐片段，不满足 mode 2 条件，但本轮仍回退 `1.2575 us`。这部分
不能归因于新分支的实际执行，可能来自比赛测量波动或较大 Kernel 代码对指令布局的影响；在
缺少逐 Case tiling key 和 `aiv_scalar_time` 时不能严格区分。由于官方总结果已明确为负，提交
`d6a5a57` 完整删除 mode 2，Host/Kernel 与 `b717f0a` 恢复点无差异，`578.9805 us` 继续作为
下一项实验的官方比较基线。

本轮资料核对也支持停止扩大逐 tile 控制逻辑。昇腾官方搬运优化建议尽量使用较大数据块，并用
`blockCount/blockLen/srcStride/dstStride` 一次表达规则搬运，避免用循环拆成小搬运；官方
`ListTensorDesc` 文档则明确区分只获取地址的 `GetDataPtr` 与解析完整 shape/地址的 `GetDesc`。
本机 CANN 8.5 内置 Concat 会在 Tiling 中预加载一部分拼接维长度，超出预加载范围才在 Kernel
解析描述符。相关资料：

- [Ascend C 搬运优化](https://www.hiascend.com/zh/developer/techArticles/20240906-1)
- [CANN 8.5 ListTensorDesc API](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0009.html)
- [Ascend C API 使用优化](https://www.hiascend.com/developer/techArticles/20241107-1)

### `2c97dca` 后的官方结果与原因分析

官方测试对删除 mode 2、并复用前 16 个输入元数据的累计版本测得：

| Case | `618125d`/us | `2c97dca`/us | 差值/us | 变化率 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 13.880 | 11.140 | -2.740 | -19.74% |
| 2 | 35.148 | 32.4705 | -2.6775 | -7.62% |
| 3 | 21.920 | 22.390 | +0.470 | +2.14% |
| 4 | 108.512 | 103.653 | -4.859 | -4.48% |
| 5 | 414.000 | 403.9585 | -10.0415 | -2.43% |
| 合计 | 593.460 | 573.612 | -19.848 | -3.35% |

相对历史最好累计版本 `578.9805 us`，当前再减少 `5.3685 us`（`0.93%`）；距离用户提出的
`540 us` 满意线还差 `33.612 us`，距离已知约 `206 us` 的领先结果仍有 `2.78x` 差距。
Case5 仍占总耗时 `70.43%`，所以下一步不能只围绕小 Case 的 Scalar 微优化。

本轮收益与 `2c97dca` 的边界访问优化一致：Case1/2/4 均明显下降，说明隐藏样例中至少有
多个输入数不超过预加载范围、且动态描述符解析占据可见比例。Case3 上升 `0.47 us`，绝对值
接近小算子测量波动，不能据此否定该方向；Case5 下降 `10.0415 us`，但它同时受缓存、DMA
和调度影响，不能把全部收益归因于元数据。相比 `618125d` 的回退，主要修复来自删除逐
输出 tile 的重复输入扫描；这也说明上一轮 mode 2 的控制流成本确实覆盖了其 MTE3 合并收益。

用户提供的 `localtest.md` 同环境记录也支持保留预加载方向：

| 本地 Case | `618125d` median/us | `2c97dca` median/us | 变化率 |
| --- | ---: | ---: | ---: |
| `ref` | 13.530 | 10.620 | -21.51% |
| `row_unaligned` | 20.011 | 17.601 | -12.04% |
| `row_aligned` | 36.571 | 36.971 | +1.09% |
| `chunk_aligned` | 16.660 | 15.650 | -6.06% |
| `fused_tiles` | 11.040 | 10.450 | -5.34% |
| `many_inputs` | 89.431 | 88.872 | -0.63% |

这组本地输入不是官方五个隐藏样例，不能直接换算排名；它的价值是说明 `many_inputs=256`
在通用描述符回退路径上基本不变，而 16 输入以内的 `ref`、行和 chunk Case 有明显改善。

### 下一轮实验：32 输入预加载与大行 tile 融合

提交 `98cc5f6` 将 `ConcatTilingData.segmentBytes` 从 16 扩展到 32 项，`inputCount <= 32`
时复用 Host 精确计算的片段长度和 `GetDataPtr`，并使 17～32 个全对齐、单行不超过 UB 的
输入可进入已有整行融合写回。超过 32 输入仍使用 `GetDesc`，因此 `max_inputs=256` 继续是
通用路径对照。该改动只增加 128 字节 Tiling 数据，但是否抵消加载和缓存收益必须上板确认。

提交 `f3aa07d` 是本轮更高风险的方向。它重新启用 `scheduleMode=2`，但设置为仅当输出行至少
包含 8 个 64 KiB tile、输入数不超过 32 且片段全部 32B 对齐时才使用。与已回退的
`618125d` 不同，新 Kernel：

- 将 tile 工作项按核分配为连续区间，而不是 `blockIdx + blockCount` 交错访问；
- 每核在同一 output row 内递增维护 `inputIdx/inputStart`，避免每个 tile 从输入 0 扫描；
- 当 tile 只覆盖某个输入的前半段时保留该输入作为下一 tile 起点，源偏移继续递增；
- 仍在 UB 内按输入交集搬入，并对每个完整 tile 只执行一次 MTE3。

这是一项有意扩大风险的实验，目标是接近 CANN 内置 Concat 的二维输出分块模型。静态随机
10000 组片段区间检查已验证交集覆盖连续、无遗漏和重复；但当前环境没有 CANN Kernel 编译
和 NPU，不能替代上板精度。若官方结果回退，优先新建回退提交恢复 `98cc5f6`，再按需要
单独测试 32 输入预加载；不要重写历史提交。

本轮 `local_test` 增加以下边界输入，均不依赖 `test-ref/`：`single_input`（1 输入）、
`zero_segments`（零长度前缀/中间片段）、`preload_16`、`preload_17`（预加载边界）、
`max_inputs`（256 输入）、`rank4_axis0`（Rank4 首轴和零片段）以及 `tile_tail`
（跨 64 KiB tile 的非整行尾块）。当前只完成 Python 语法、分片约束和主机区间静态检查，
请在 NPU 环境优先运行：

```bash
bash local_test/run.sh preload_16
bash local_test/run.sh preload_17
bash local_test/run.sh fused_tiles
bash local_test/run.sh tile_tail
bash local_test/run.sh max_inputs
```

如需一次跑完所有新增边界，使用 `bash local_test/run.sh all`；当前 `run.sh` 会按 `CASES`
列表逐个执行并检查自定义 `aclnnConcat` 符号。

### `f3aa07d` 后的官方结果与原因分析

官方对包含 32 输入预加载和受限 mode 2 的累计版本测得：

| Case | 上轮最好/us | 当前/us | 差值/us | 变化率 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 11.140 | 11.2205 | +0.0805 | +0.72% |
| 2 | 32.4705 | 33.691 | +1.2205 | +3.76% |
| 3 | 22.390 | 20.480 | -1.910 | -8.53% |
| 4 | 103.653 | 105.072 | +1.419 | +1.37% |
| 5 | 403.9585 | 404.288 | +0.3295 | +0.08% |
| 合计 | 573.612 | 574.7515 | +1.1395 | +0.20% |

总差值只有 `1.1395 us`，小于各 Case 正负变化的幅度，且 Case3 单项改善 `1.91 us`，不能排除
比赛环境噪声或隐藏随机输入变化。但累计方案没有形成可重复的净收益：Case1/2/4/5 均回退，
Case5 仍占总耗时 `70.34%`，距离 `540 us` 还差 `34.7515 us`。32 输入预加载与 mode 2 在同一
累计版本中上板，官方五项数据无法进一步拆分两者；下一轮应保留前者、单独删除后者，避免继续
叠加不可归因改动。

`localtest.md` 新记录与 `2c97dca` 的本地中位数对比如下：

| 本地 Case | `2c97dca`/us | 新记录/us | 差值/us | 变化率 |
| --- | ---: | ---: | ---: | ---: |
| `ref` | 10.620 | 10.580 | -0.040 | -0.38% |
| `row_unaligned` | 17.601 | 18.041 | +0.440 | +2.50% |
| `row_aligned` | 36.971 | 37.101 | +0.130 | +0.35% |
| `chunk_aligned` | 15.650 | 18.310 | +2.660 | +17.00% |
| `fused_tiles` | 10.450 | 12.320 | +1.870 | +17.89% |
| `many_inputs` | 88.872 | 86.812 | -2.060 | -2.32% |

`chunk_aligned` 和 `fused_tiles` 的退步方向与 mode 2 风险一致，但当前段落标签为 `0189f03`；
该提交只增加测试 Case，且早于 `98cc5f6/f3aa07d`，所以标签不能证明实际安装的算子二进制包含
mode 2。若当时确实运行当前累计二进制，这两个约 `17%` 的回退是删除 mode 2 的强证据；若
运行的就是 `0189f03`，则 Kernel 与 `2c97dca` 相同，差值只能视为本地测量波动。为避免后续
继续出现这种归因歧义，本地脚本下一步会自动把代码 HEAD、逐 Case 正确性和统计值写入汇总文档。

综合已有两轮官方 mode 2 负反馈，当前判断仍是：输出 tile 融合减少 MTE3 次数，但额外区间映射、
输入切换和对多个 GM 源的交错读取未能被摊薄。下一实验不再修改输出区间映射，而是在保持活跃
AIV 数的前提下扩大普通 row/chunk 路径单次 UB/DMA 搬运块，直接针对 Case5 的大规模流式搬运。

### 下一轮实验：自适应 96 KiB 搬运块

提交 `8dc7a78` 已用新的 Git revert 提交删除 `f3aa07d` 的全部 mode 2 Host/Kernel 代码，
`98cc5f6` 的 32 输入元数据预加载保持不变。提交 `9f07f4a` 随后把普通 row/chunk 路径可选
最大 tile 从 64 KiB 提升为 96 KiB。Ascend 910B 的 UB 为 192 KiB，当前绑定队列使用两个
Buffer，因此 `2 x 96 KiB` 恰好覆盖全部 UB；Kernel 的地址公式、输入遍历顺序、DMA 类型和
双 Buffer 数量均未改变。

Host 同时计算 32/64/96 KiB 下的工作项数，并选择不损失活跃 AIV 的最大块：outer 行本身已
填满核或 96 KiB chunk 数能填满核时选择 96 KiB；否则若 64 KiB 能填满核则选择 64 KiB；
再否则保留 32 KiB。这样避免在占用率临界形状上为了减少 DMA 次数而丢失并行度。新本地 Case
`tile_medium_occupancy` 使用单个 3 MiB 片段，在 40/48 核上只能选择 64 KiB；
`tile_large_occupancy` 使用单个 4.6875 MiB 片段，可选择 96 KiB。两者用于检查阈值和单大段
流式吞吐，不与官方隐藏 Case 直接对应。

该实验风险高于原 64 KiB：队列没有 UB 余量，虽然本机 CANN 8.5 的 910B 内置算子也将
192 KiB 作为可用 UB 上限，但当前环境未完成 Kernel 编译。若构建阶段报告 UB 分配失败，或
`tile_large_occupancy/chunk_aligned/Case5` 明显回退，应新建 revert 提交撤销 `9f07f4a`；
此时 `8dc7a78` 是仅保留 32 输入预加载的隔离点。

### `9f07f4a` 后的官方结果与原因分析

官方对删除 mode 2、保留 32 输入预加载并启用 96 KiB tile 的累计版本测得：

| Case | 上轮/us | 当前/us | 差值/us | 变化率 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 11.2205 | 11.360 | +0.1395 | +1.24% |
| 2 | 33.691 | 35.128 | +1.437 | +4.27% |
| 3 | 20.480 | 20.792 | +0.312 | +1.52% |
| 4 | 105.072 | 106.708 | +1.636 | +1.56% |
| 5 | 404.288 | 411.268 | +6.980 | +1.73% |
| 合计 | 574.7515 | 585.256 | +10.5045 | +1.83% |

五项全部回退，不能再按随机输入波动保留 96 KiB 实验。当前结果相对历史最好 `573.612 us`
已慢 `11.644 us`（`2.03%`），距离 `540 us` 满意线扩大到 `45.256 us`；Case5 仍占总耗时
`70.27%`。96 KiB 虽保持了 Host 计算的活跃核数，但 `2 x 96 KiB` 用满 192 KiB UB，减少
了双 Buffer 轮转频率，并改变所有 outer 已填满核场景的队列大小和 DMA 包长。保持核数再次被
证明只是必要条件：更大的单包没有抵消流水重叠、尾块比例和硬件最佳搬运粒度的损失。

`localtest.md` 中自动汇总的代码标签为 `7695c49`，15 项均通过 bit-exact 检查。与此前
`[0189f03]` 的 13 个共同 Case 比较，中位数合计从 `289.716 us` 变为 `290.164 us`，只增加
`0.448 us`（`0.15%`）；其中 `chunk_aligned` 从 `18.310` 降至 `17.230 us`，`fused_tiles`
从 `12.320` 降至 `11.280 us`，但 `row_aligned` 从 `37.101` 升至 `38.270 us`。这说明本地
Case 能验证精度、阈值和相对分支行为，却没有复现官方 Case5 对 96 KiB 的负反馈。后续仍应
分别记录本地和官方结果，不以本地合计预测隐藏总分。

下一步先用显式 revert 删除 `9f07f4a`，恢复已验证较合适的 64/32 KiB 策略。新的性能实验不再
调整 UB 容量，而改进 chunk 调度中的字节负载均衡：当前每段按固定 64 KiB 切分，略大于 tile
的片段会形成“一个完整大块加一个极小尾块”；工作项数量看似填满 AIV，实际各核拥有的字节数
可能严重不均。计划保持每段 chunk 数不变，将段内 chunk 长度重新均分并对齐到 32B，以不改变
核占用和 DMA 次数的方式缩小长尾核负载。

### 下一轮实验：均衡段内 chunk 字节数

提交 `0e38813` 已完整撤销 96 KiB 实验，当前重新使用经过官方反馈的 64/32 KiB tile。提交
`b702aec` 只修改 `ProcessByChunks` 的段内边界。对片段字节数 `S` 和原 tile `T`：

```text
chunkCount = ceil(S / T)
balancedChunkBytes = align_up(ceil(S / chunkCount), 32)
chunkStart = chunkIndex * balancedChunkBytes
chunkBytes = min(balancedChunkBytes, S - chunkStart)
```

`chunkCount`、`inputWorkItems`、全局 work-item 编号和 blockDim 均不改变，因此不会减少活跃核或
增加 DMA 命令。所有段内区间仍从 0 连续覆盖到 `S`，源地址和目标地址只把原固定 tile 偏移替换
为均衡偏移。对齐片段的首地址、均衡块长和尾块仍全部 32B 对齐，继续使用普通 `DataCopy`；
非对齐片段继续完整使用 `DataCopyPad`。

极端例子是 40 个 `65568B` 对齐片段。旧方案将每段切为 `65536+32B`，80 个工作项分给 40 核后，
每核总字节范围可达到 `64～131072B`；新方案切为 `32800+32768B`，范围收敛到
`65536～65600B`。`chunk_imbalanced_aligned` 固定覆盖该场景；
`chunk_imbalanced_unaligned` 使用 40 个 `65537B` 片段，覆盖 `65536+1B` 尾块和 Pad 路径。
随机 10000 组片段/tile 静态检查已确认新边界连续、无遗漏和重叠。

新增均分需要多块片段每核、每输入执行一次整数除法。提交 `d430bc0` 将计算限制为
`chunkCount > 1`，单 chunk、零长度输入和全部 row 调度不增加该成本。主要未知是隐藏 Case5
是否确实包含大量略大于 tile 的片段；若片段本来已是整 tile 或远大于 tile，收益会较小。

### `d430bc0` 后的官方结果与原因分析

官方对恢复 64/32 KiB tile、保留 32 输入预加载并启用段内 chunk 字节均衡的累计版本测得：

| Case | 上一轮 `9f07f4a`/us | 当前版本/us | 差值/us | 变化率 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 11.360 | 11.580 | +0.220 | +1.94% |
| 2 | 35.128 | 35.332 | +0.204 | +0.58% |
| 3 | 20.792 | 21.192 | +0.400 | +1.92% |
| 4 | 106.708 | 107.220 | +0.512 | +0.48% |
| 5 | 411.268 | 409.956 | -1.312 | -0.32% |
| 合计 | 585.256 | 585.280 | +0.024 | +0.004% |

总耗时仅相差 `0.024 us`，远小于各小 Case 的常见波动，不能认为 chunk 均衡带来了可测的净收益；
Case5 虽下降 `1.312 us`，也被 Case1～4 合计 `1.336 us` 的回退抵消。相对不含 96 KiB 和
mode 2 的上一轮 `574.7515 us`，当前仍慢 `10.5285 us`（`1.83%`）；相对历史最好
`573.612 us` 慢 `11.668 us`（`2.03%`），距离 `540 us` 目标还有 `45.280 us`。

这组结果否定了“略大于 tile 的片段尾块负载不均是当前主要瓶颈”这一假设。若隐藏 Case5
确实大量命中该形状，静态模型中从 `64～131072B` 收敛到 `65536～65600B` 的每核负载应产生
比 `0.32%` 更清晰的收益；实际几乎持平，说明 Case5 更可能受动态列表扫描、DMA 命令数、缓存
或总体带宽限制。当前实现仍有更大的结构性成本：chunk 模式下每个 AIV 都遍历全部输入并计算
本核首个 work item，即使该输入没有任何 chunk 分配给该核。下一轮不再继续微调 tile 或尾块，
改为由 Host 下发每核连续工作区间，减少 Kernel 的 `coreCount * inputCount` 元数据访问。

`localtest.md` 中标记为 `39282b7` 的最新本地汇总也没有显示稳定收益。与上一份 `7695c49`
汇总的 13 个共同 Case 相比，中位数合计从 `290.164 us` 变为 `290.636 us`，增加
`0.472 us`（`0.16%`）；均值合计从 `289.694 us` 变为 `288.296 us`，反而减少
`1.398 us`（`0.48%`）。`chunk_aligned` 中位数增加 `0.230 us`，`max_inputs` 增加
`1.861 us`，而 `row_aligned` 减少 `1.549 us`。中位数与均值方向相反，且改善/回退没有按
chunk 路径一致分布，因此本地数据也只能把上一轮判定为测量波动范围内的中性结果。官方和
本地 Case 仍是两套独立口径，不能把本地合计换算为隐藏总分。

### 下一轮实验：Host 预分区连续 chunk 工作

提交 `6ad3bd3` 改写 chunk 模式的工作分配，但不改变 row 模式、tile 大小、chunk 数、DMA 命令
总数或每个 chunk 的源/目标区间。旧 Kernel 对每个输入计算当前核对应的模余起点，即使该输入
没有 work item 属于本核，也必须执行 `LoadInput`、chunk 计算和分支；输入列表较长时，总元数据
成本近似 `blockDim * inputCount`。新 Host 为每个 AIV 下发：

```text
chunkStartInput    本核第一个输入
chunkStartWork     第一个输入内的 work-item 偏移
chunkWorkCount     本核连续消费的 work-item 数量
chunkOutputOffset  第一个输入在输出行中的字节前缀
```

Kernel 从该起点向后推进，完成 `chunkWorkCount` 后立即退出，不再从输入 0 扫描到末尾。Host
先以 `totalBytes * core / blockDim` 计算每核字节目标，再把边界吸附到现有 chunk work-item
边界；相邻边界至少相差一个 work item，因此每核非空、全部边界严格递增，且 DMA 工作集合与
上一版本完全相同。与纯粹按 work-item 数量均分相比，该方法还能避免大小悬殊的单 chunk 输入
集中到少数核。

910B 最多使用 40 个 AIV；新增 40 项数组后 `ConcatTilingData` 约为 `1416B`，低于当前注册
接口默认的 `2048B` tiling 上限，并在保存前显式检查实际 capacity。该设计也参考了本机
CANN 8.5 内置 Concat：内置实现同样由 Host 下发每核 `endTensorIdx/endTensorOffset`，而不是
让所有核完整扫描 TensorList。

静态区间模型随机验证了 `19965` 组实际进入 chunk 调度的输入，其中 `498` 组展开到每个
work item；每项均唯一归属一个核，零长度输入能正确跳过，源和目标区间连续且无遗漏。对本地
`max_inputs`（2 行、256 个 1B 输入），预估输入元数据触达从 `40*256=10240` 次降到约
`272` 次；新增 `chunk_many_inputs`（1 行、256 个 64 KiB 输入）降到 `256` 次。新增
`chunk_single_large` 使用同样 16 MiB 总量但只有一个输入，触达次数仍为 `40`，用于判断收益
来自列表扫描消除还是单纯测量波动。

这是高风险结构性实验。主要风险是更大的 tiling 读取抵消小 Case 收益，以及连续 work 区间
改变各核同时访问的输入顺序后影响缓存/带宽。若 `chunk_many_inputs` 和 `max_inputs` 明显改善、
而 `chunk_single_large` 与 `chunk_aligned` 基本持平，才说明假设成立；若单输入大块也显著
回退，应直接 revert `6ad3bd3`，不再把其他优化叠加到该版本。

### `6ad3bd3` 后的官方结果与原因分析

官方对 Host 连续 chunk 预分区的累计版本测得五项精度全部通过，但性能为：

| Case | 上一轮/us | 当前/us | 差值/us | 变化率 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 11.580 | 14.064 | +2.484 | +21.45% |
| 2 | 35.332 | 43.756 | +8.424 | +23.84% |
| 3 | 21.192 | 20.752 | -0.440 | -2.08% |
| 4 | 107.220 | 108.980 | +1.760 | +1.64% |
| 5 | 409.956 | 413.780 | +3.824 | +0.93% |
| 合计 | 585.280 | 601.332 | +16.052 | +2.74% |

Case1/2 的 `2.484/8.424 us` 回退远大于前几轮小 Case 的普通波动，不能继续把本轮判为中性。
当前总耗时比历史最好 `573.612 us` 慢 `27.720 us`（`4.83%`），比满意线 `540 us` 高
`61.332 us`。唯一改善的 Case3 下降 `0.440 us`，不足以抵消其余四项回退。

本地 `localtest.md` 中标记为 `bb86f5e` 的 17 项结果则证明了预分区机制本身确实生效。与
`39282b7` 的 15 个共同 Case 相比，中位数合计从 `344.087 us` 降到 `264.434 us`，减少
`79.653 us`（`23.15%`）；均值合计从 `341.434 us` 降到 `266.804 us`，减少
`74.630 us`（`21.86%`）。其中 `max_inputs` 从 `64.772` 降到 `27.490 us`，两个刻意构造的
长片段 Case `chunk_imbalanced_aligned/unaligned` 分别从 `26.600/26.851` 降到
`9.460/9.350 us`，`many_inputs` 也从 `86.672` 降到 `82.342 us`。新增的同字节数对照中，
`chunk_many_inputs=15.131 us`，`chunk_single_large=16.870 us`，长列表已不再比单输入更慢。

本地与官方方向相反并不矛盾。本地改善主要集中在 256 输入、40 核都会扫描完整列表的专门
诊断形状；官方隐藏 Case 并不保证具有该分布。新版本还把固定 Tiling 从约 `296B` 扩大到
约 `1416B`，并增加 chunk 起点恢复、连续区间和按字节边界逻辑。即使 row 地址公式没有改变，
更大的 Kernel 参数与代码布局也可能增加固定开销；chunk 隐藏形状还可能因连续输入分组改变
并发访存和核间尾部负载。缺少逐 Case tiling key 和流水线计数时不能把 Case2 回退严格归因于
其中一项，但官方总表已经足以否定把这套大数组调度作为通用默认路径。

下一轮回到 `2c97dca` 的最优原则：只在小型 Tiling 中预计算高复用元数据，并尽量用一次
`blockCount/stride` DMA 表达规则搬运。将显式撤销 Host 预分区和未显示净收益的 chunk 均衡，
同时把预加载上限恢复为官方最优版本的 16。新的高风险方向不再微调 chunk，而是针对缺失的
“非对齐小片段整行融合”：先把多输入行搬入 UB 的对齐 staging 区，通过 910B 支持的 Gather
压紧为连续输出，再用一次 MTE3 写回。该路径只覆盖 2B/4B 元素、输入数不超过 16 且整行较小
的 row 调度，INT8 和不满足 UB 预算的形状继续使用旧路径。

这一选择来自本机 CANN 8.5 内置 Concat 的专用路径划分与昇腾官方搬运建议。内置实现对
全对齐、非对齐同形状和非对齐异形状分别处理；官方文档建议用
`blockCount/blockLen/srcStride/dstStride` 合并规则搬运。更新架构的
`PaddingMode::Compact` 不适用于 910B，因此本项目使用 910B 已提供的 Gather，而不直接照搬
新架构 MicroAPI：

- [Ascend C 高效使用搬运 API](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/devguide/opdevg/ascendcbestP/atlas_ascendc_best_practices_10_0015.html)
- [DataCopyPad 参数与非对齐 dummy 规则](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/apiref/ascendcopapi/atlasascendc_api_07_0265.html)
- [Compact 模式的产品限制](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_00017.html)

### 下一轮实验：复用 Host 预加载的片段字节数

提交 `2c97dca` 使用当前 Host 已写入 Tiling 的前 16 个输入 `segmentBytes`。此前普通
row/chunk Kernel 仍对每个输入调用 `GetDesc`，将 shape 从 GM 复制到核侧缓冲，再循环计算 `innerSize` 和
`segmentBytes`。现在 `inputCount <= 16` 时直接读取 Tiling 中的 `segmentBytes`，并只用
`GetDataPtr` 获取动态输入地址；超过 16 输入时保留原描述符路径。该实验不改变 work-item、
tile、DMA 参数、缓存策略或源/目标地址公式，重点降低小数据、多输入和高核数场景的 Scalar
元数据成本。

已有的对齐整行融合路径原本在每个 UB batch 内重新遍历 TensorList 获取输入地址。由于该路径
本来就限制为最多 16 输入，当前版本在每核开始时读取一次非空输入地址，后续 batch 直接复用；
这样不改变每批 MTE2/MTE3 次数，但避免 `batchCount * inputCount` 次地址解析。指针数组形式已在
此前通过官方精度测试的 `618125d` 中使用过，当前只是将其用于已验证有收益的整行融合路径。

官方 API 文档证明 `GetDataPtr` 可以独立取得动态列表数据地址；本机内置 Concat 的预加载长度
策略说明 Host 预计算后避免重复 shape 解析是已有实现模式。预计本地 `ref`（9 输入）最能观察
小数据元数据收益；`many_inputs` 有 256 输入，会继续走通用描述符路径，应该保持不变。

静态复算固定随机拆分后，`ref/row_unaligned/row_aligned/chunk_aligned/fused_tiles` 的输入数依次为
`9/11/10/16/16`，均进入新路径；`many_inputs=256` 是通用路径对照。对 Rank 1-4、每个合法轴、
四种元素字节宽度随机生成的 10000 组 shape，Host `segmentBytes` 与旧 Kernel shape 乘积公式完全
一致。该检查只证明元数据公式等价，不替代 CANN Kernel 编译、NPU 精度或性能测试。

此前 mode 2 的地址模型本身仍然正确：

第 `o` 行输出 tile `[t, t + n)` 与输入 `i` 的输出区间 `[p_i, p_i + s_i)` 的交集为
`[max(t, p_i), min(t + n, p_i + s_i))`。交集对应的源偏移为
`o * s_i + overlapStart - p_i`，UB 偏移为 `overlapStart - t`，因此输出地址模型不变；所有
片段边界、tile 起点及尾 tile 长度均为 32B 整数倍，可以继续使用普通 `DataCopy`。
失败原因是性能模型遗漏了 Scalar 和访问顺序成本，而不是精度映射错误。

### 调优示例的取舍

本轮使用了示例中适合纯搬运算子的原则：多核负载均衡、大包搬运、双 Buffer 队列、L2 策略和对齐/UB 规划。

没有强行应用以下策略：

- Vector/RegBase：Concat 没有算术计算链。
- UB Bank Conflict 优化：当前主路径是 MTE 搬运，不是 Vector 访存瓶颈。
- Cube/Vector 融合或 L0C 到 UB 直通：Concat 是 AIV-only 算子，且部分通路仅新架构支持。
- 额外手写事件流水：CANN 内置纯拷贝采用的也是双 Buffer 队列下的 `CopyIn -> EnQue -> DeQue -> CopyOut` 结构。

## 构建

在 CANN 8.5 / Ascend 910B 环境执行：

```bash
bash build.sh
```

成功后应在 `build_out/` 中得到 `custom_*.run`。构建脚本会重新生成 `.build/`，因此不要在该目录保存手工修改。

本次开发环境没有 NPU。曾启动本地构建并完成 Host 编译、进入 Kernel 生成阶段，但按要求提前停止，因此不能把当前版本声明为完整编译通过或 NPU 验证通过。

## 本地快速测试

`local_test/` 根据 `test-ref/` 的比赛调用链编写，仍通过 `EXEC_NPU_CMD(aclnnConcat, ...)`
调用本项目自定义算子。与官方封装一致，每轮重新分配输出；每个 Case 连续执行 30 次，
丢弃前 10 次后报告后 20 次 `Task Duration(us)` 中位数。与参考脚本相比，它移除了无关
`aclnnMul`，并在
`op_summary*.csv` 中只收集名称包含 `Concat` 的任务。

诊断 Case 分别覆盖：

| 名称 | 主要用途 |
| --- | --- |
| `ref` | 复现参考目录中的 `[128, 256]`、FP16、末轴随机拆分 |
| `row_unaligned` | 多 outer 行、非 32B 对齐片段和多行 `DataCopyPad` |
| `row_aligned` | 多 outer 行、32B 对齐片段及 UB 内融合写回路径 |
| `chunk_aligned` | `outerSize=1` 的大对齐片段、多核 chunk 调度 |
| `fused_tiles` | 16 个 32 KiB 对齐片段组成 512 KiB 大行，直接覆盖 mode 2 跨输入 tile 融合 |
| `many_inputs` | ACLNN 上限 256 个小片段，放大动态 TensorList 元数据开销 |
| `single_input` | 单输入退化路径，检查无拼接边界时的地址和调度 |
| `zero_segments` | 零长度前缀和中间输入，检查跳过空片段后的输出偏移 |
| `preload_16` / `preload_17` | 16/32 元数据预加载边界及全对齐整行融合 |
| `max_inputs` | ACLNN 256 输入上限的描述符回退路径 |
| `rank4_axis0` | Rank4 首轴拼接、负/正轴地址模型和零长度片段 |
| `tile_tail` | 输出行跨越 64 KiB tile 且存在非整 tile 尾块 |
| `chunk_imbalanced_aligned` | 40 个 `65568B` 对齐片段，放大固定 tile 的核间字节失衡 |
| `chunk_imbalanced_unaligned` | 40 个 `65537B` 非对齐片段，验证均衡尾块的 Pad 路径 |
| `chunk_many_inputs` | 单行 256 个 64 KiB 输入，放大 chunk 模式的全列表元数据扫描 |
| `chunk_single_large` | 与上一项同为 16 MiB，但只有一个输入，隔离连续分区自身开销 |

测试工具所需的 ACLNN/PyTorch NPU helper 已复制到
`local_test/common/pytorch_npu_helper.hpp`，运行时不依赖 `test-ref/`，整个
`local_test/` 目录可以独立迁移。先重新构建、安装当前算子包，再首次构建测试扩展并运行全部 Case：

```bash
bash build.sh
# 安装本次 build_out/custom_*.run 后执行：
bash local_test/run.sh all --build
```

后续只修改并重新安装算子包时，无需重建测试 wheel：

```bash
bash local_test/run.sh all
# 也可只跑一个分支，例如：
bash local_test/run.sh ref
```

`get_time.py` 对预热后的 20 个样本同时输出中位数、算术均值、最小值和最大值。每个成功 Case
会保存到 `local_test/results/<commit>/<case>.txt`；运行单 Case 或 `all` 后，脚本自动生成
`local_test/results/<commit>/summary.md`，并复制为便于查看的
`local_test/results/latest.md`。简表包含逐 Case 的 `median/mean/min/max`，以及所有已完成
Case 的 median 合计和 mean 合计。该目录是机器生成产物，已加入 `.gitignore`。

默认 commit 来自运行时源码 HEAD。若当前安装的自定义算子包由另一个 commit 构建，应显式
标注真实代码版本，避免再次出现源码标签与二进制不一致：

```bash
CONCAT_CODE_COMMIT=<installed-op-commit> bash local_test/run.sh all
```

### 测试扩展兼容性修复

首次在用户的 PyTorch/Python 3.9 环境构建测试扩展时，本地 helper 的
`EXEC_NPU_CMD` 宏使用了未限定命名空间的 `kByte`，而该版本 PyTorch 只公开
`c10::kByte`，导致 C++ 编译失败。现已在 `local_test` 自带的 helper 中改为
`c10::kByte`。此更改只修复测试扩展的源码兼容性，不修改算子实现或执行语义。

构建日志中的 `ninja` 缺失提示只表示回退到较慢的 distutils 后端；
`torch_npu` 头文件产生的 unused warning 也不是本次失败原因。只要最终不出现
`error:` 且 wheel 能成功生成和安装，这些 warning 可忽略。

测试扩展成功导入后，曾在执行阶段出现
`aclnnConcat or aclnnConcatGetWorkspaceSize not in libopapi.so`。helper 实际会先查找
`libcust_opapi.so`，旧错误文本只显示系统 `libopapi.so`，无法区分算子包未安装、安装到
其他自定义 OPP 根目录或加载到同名旧库。`run.sh` 现在会在 profiler 启动前完成以下检查：

- 扫描 `ASCEND_OPP_PATH` 和冒号分隔的 `ASCEND_CUSTOM_OPP_PATH`；
- 使用 `nm` 确认同一份 `libcust_opapi.so` 同时导出 `aclnnConcat` 和
  `aclnnConcatGetWorkspaceSize`；
- 通过 `CONCAT_OPAPI_LIB` 将已验证库的绝对路径传给 C++ helper，并把所在目录加入
  `LD_LIBRARY_PATH`。

若扫描失败，先执行 `bash build.sh` 并安装 `build_out/custom_*.run`，再检查上述 OPP
环境变量。也可以显式指定已安装库后运行：

```bash
CONCAT_OPAPI_LIB=/absolute/path/to/libcust_opapi.so bash local_test/run.sh all --build
```

脚本缺少 `nm` 时会在采集前退出。ACLNN 调用失败后 profiler 中出现“0 Concat tasks”是
连带现象，不是另一个 Kernel 性能问题。

`many_inputs` 最初使用 `max_step=32` 拆分长度 10000 的末轴，在固定随机种子下生成了
613 个输入，超过当前 ACLNN 动态输入列表的 256 上限。调用因此在参数校验阶段失败，Kernel
和 Tiling 均未执行，随后 profiler 报告“0 Concat tasks”也是连带现象。修复后保持
`[64, 10000]`、INT8 和末轴拼接不变，将 `max_step` 调整为 64，并让拆分器根据剩余槽位
动态限制随机下界，保证任何 case 都不会生成超过 256 个输入。当前固定种子下
`many_inputs` 恰好生成 256 个输入，其中包含 5 个零长度分片，仍覆盖列表上限、非对齐片段
和零长度输入。

本次只修改 Python case 生成逻辑，不需要重建测试扩展，可直接重跑：

```bash
bash local_test/run.sh many_inputs
```

本次失败前 `chunk_aligned` 已完成本地测试，结果为 `median=17.920 us`、
`min=16.160 us`、`max=19.480 us`。这是当前本地诊断 case 的单次结果，没有旧版本的同环境
对照，不能直接推导官方隐藏 Case 的收益。

新增 `fused_tiles` 固定使用 `[8, 524288]` INT8 输入和 16 个 32 KiB 片段。64 KiB tile
每次恰好跨越两个输入边界；旧 chunk 路径每行产生 16 个输出 DMA，新 mode 2 每行产生 8 个，
且两者在 910B 上都能激活 40 个 AIV。只修改测试 case 后不需要重建测试扩展，但需要先重新
构建并安装当前算子包，然后可单独运行：

```bash
bash local_test/run.sh fused_tiles
```

用户在正确 NPU 环境对 `618125d` 运行六个本地诊断 Case，并记录在仓库根目录
`localtest.md`。这组数据只作为该本地输入集和环境的后续对照，不与官方隐藏 Case 的结果直接
比较：

| 本地 Case | median/us | min/us | max/us |
| --- | ---: | ---: | ---: |
| `ref` | 13.530 | 11.900 | 15.200 |
| `row_unaligned` | 20.011 | 17.221 | 21.660 |
| `row_aligned` | 36.571 | 35.641 | 37.741 |
| `chunk_aligned` | 16.660 | 14.101 | 18.080 |
| `fused_tiles` | 11.040 | 9.900 | 12.760 |
| `many_inputs` | 89.431 | 79.101 | 101.942 |

`fused_tiles=11.040 us` 说明 mode 2 在刻意构造的跨边界大行上可以快速执行，但没有同环境旧
版本结果，不能单独证明它比旧 chunk 路径更快；官方总表已经表明该策略不适合当前隐藏样例。
`2c97dca` 应在同一环境重跑全部六例，以 `ref` 观察 16 输入以内的 shape 解析消除效果，以
`many_inputs` 作为超过预加载上限的通用路径对照，并记录撤销 mode 2 后 `fused_tiles` 的变化。

每个 Case 成功时输出两行便于直接回传：

```text
CASE_RESULT name=<case> correctness=pass
PERF_RESULT name=<case> samples=20 median_us=<time> mean_us=<time> min_us=<time> max_us=<time>
```

此环境没有 NPU，因此上述测试脚本只完成源代码和 Shell 静态检查，未在本机实际执行。

## NPU 验证

先测试当前累计版本：

```bash
git log --oneline -12
# 当前待测版本由 Host 下发每核连续 chunk 工作区间，Kernel 只解析覆盖的输入

bash build.sh
# 按比赛环境原有流程安装 custom_*.run，运行精度与五个性能样例
```

性能测试应先预热，再在相同环境重复运行并取中位数。请同时记录精度结果、SoC/CANN 版本和 commit：

```text
commit: <git rev-parse --short HEAD 的输出>
code_baseline: <git rev-parse --short HEAD 的输出>
soc/cann: <version>
correctness: pass|fail
times_us: <case1>, <case2>, <case3>, <case4>, <case5>
```

本地测试优先回传 `max_inputs`、`chunk_many_inputs`、`chunk_single_large`、`chunk_aligned` 和
`ref`。前三项直接隔离动态列表扫描收益与单输入开销，后两项检查普通 chunk 和 row 回归；
测试扩展已构建时无需因 Python Case 改动执行 `--build`。当前版本应与刚测得的
`585.280 us`、不含 chunk 均衡的 `574.7515 us` 和历史最好 `573.612 us` 同时比较。若连续
预分区回退，先新建 revert 提交撤销 `6ad3bd3`；若还要隔离 chunk 均衡，再依次撤销
`d430bc0` 和 `b702aec`。`0e38813` 是恢复 64/32 KiB、保留 32 输入预加载且不含 mode 2 的
隔离点。

更早的累计版本顺序为：

```text
355dd77 -> cc66c1c -> 0b45882 -> 5a1b9ea -> 5b55590
```

这些都是累计版本。相邻版本的差异对应表格中的一项优化，可据此判断 L2 bypass、tile 大小、多行 DMA 或对齐快路径是否适合隐藏样例。

## Profiling

可使用 `msOpProf` 采集 NPU 数据：

```bash
msopprof <测试可执行程序>
```

优先关注：

- `Task Duration`：最终判定指标。
- `aiv_scalar_time`：动态描述符和索引计算开销。
- `aiv_mte2_time` / `aiv_mte3_time`：GM/UB 搬运瓶颈。
- `Memory.csv`、`L2Cache.csv`：带宽和 L2 行为。

流水线指标可能重叠，不能简单相加。双 Buffer 生效后，单项 MTE 时间增加并不必然表示端到端性能回退。

## 正确性重点

NPU 测试至少覆盖：

- 四种 DType 和 Rank 1 到 4；
- 首轴、中间轴、末轴和负轴；
- 单输入、多输入、不同拼接长度和零长度分片；
- 非 32B 对齐片段；
- `outerSize` 小于、等于和大于 AIV 核数；
- 片段跨越 32 KiB/64 KiB tile 的尾块；
- 多行 DMA 的行数、UB 容量和 stride 边界。

纯搬运路径应按位完全一致，不应使用浮点容差掩盖字节错误。
