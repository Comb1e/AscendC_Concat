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
本轮已补充两条可复用经验：L2 策略应按输入/输出的生命周期分别实验；性能复现除了 shape，
还必须匹配输入复用、逐轮输出分配、分配器复用及中间算子造成的缓存行为。更新后的 skill
还记录了算子规格与 ACLNN 等调用前端的动态列表上限需要分开确认，以及“0 target tasks”
通常是启动前错误的连带结果。更新后的 skill 已通过 `skill-creator` 的
`quick_validate.py` 校验。

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
| `b2ecea2` | 保留输入默认缓存，仅让输出绕过 L2 | 避免 30 轮新输出挤占复用输入的缓存 | 可能降低 MTE3 写带宽，并损失下游输出复用 |

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

### 下一轮实验：仅输出绕过 L2

官方参考封装在 30 轮循环中复用同一组输入，但每轮重新分配输出；前 29 个中间输出不会被
后续算子读取，只有最后一轮结果被返回。当前实验保留所有输入的默认 L2 策略，仅对输出
`GlobalTensor` 设置 `CACHE_MODE_DISABLE`。目标是避免连续写入的新输出占用 L2，从而为复用的
输入保留更多缓存容量，主要观察占总耗时 `69.55%` 的 Case5。

这与已证明回退的 `0b45882` 不同：旧实验绕过的是输入缓存，使 Case5 从 `413.720 us`
恶化到 `1653.960 us`；本实验不改变任何输入读策略。输出绕过不会改变写入地址和数值，
但可能降低单次 MTE3 写带宽，而且在普通算子图中会使下游消费者失去输出的 L2 热数据，
因此只作为适配比赛测量封装的独立可回退实验。若 Case5 没有明显收益或 Case4 继续回退，
应撤销本提交，不与更多 Kernel 改动叠加。

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

五个诊断 Case 分别覆盖：

| 名称 | 主要用途 |
| --- | --- |
| `ref` | 复现参考目录中的 `[128, 256]`、FP16、末轴随机拆分 |
| `row_unaligned` | 多 outer 行、非 32B 对齐片段和多行 `DataCopyPad` |
| `row_aligned` | 多 outer 行、32B 对齐片段及 UB 内融合写回路径 |
| `chunk_aligned` | `outerSize=1` 的大对齐片段、多核 chunk 调度 |
| `many_inputs` | ACLNN 上限 256 个小片段，放大动态 TensorList 元数据开销 |

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

每个 Case 成功时输出两行便于直接回传：

```text
CASE_RESULT name=<case> correctness=pass
PERF_RESULT name=<case> samples=20 median_us=<time> min_us=<time> max_us=<time>
```

此环境没有 NPU，因此上述测试脚本只完成源代码和 Shell 静态检查，未在本机实际执行。

## NPU 验证

先测试当前累计版本：

```bash
git log --oneline -12
# 本轮待测算子代码基线应为 b2ecea2

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

本地测试优先回传 `row_aligned`、`many_inputs` 和 `ref`；其中 `row_aligned` 直接覆盖
`b041908` 新路径，`many_inputs` 观察动态列表元数据开销，`ref` 对应已知 Case2 的
`[128, 256]` FP16 非对齐随机分片。本轮修改了本地 C++ 测试封装，需要重新执行一次
`bash local_test/run.sh all --build`。官方系统下一次提交应记录算子代码基线 `b2ecea2`。
若性能回退，先测试 `b041908` 隔离输出 L2 策略；若还需隔离前一轮，再测试 `767c220`
排除 UB 融合路径，最后用 `75eec09` 复现 `589.168 us` 结果。

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
