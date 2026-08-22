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
下一轮仍必须改善大搬运 Case，同时保持输入和输出的默认 L2 缓存。

### 下一轮实验

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
内置 Concat 的“多个输入组装到同一 UB 后合并写回”策略一致，但性能仍需在 910B 上实测。

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
调用本项目自定义算子。每个 Case 连续执行 30 次，丢弃前 10 次后报告后 20 次
`Task Duration(us)` 中位数。与参考脚本相比，它移除了无关 `aclnnMul`，并在
`op_summary*.csv` 中只收集名称包含 `Concat` 的任务。

五个诊断 Case 分别覆盖：

| 名称 | 主要用途 |
| --- | --- |
| `ref` | 复现参考目录中的 `[128, 256]`、FP16、末轴随机拆分 |
| `row_unaligned` | 多 outer 行、非 32B 对齐片段和多行 `DataCopyPad` |
| `row_aligned` | 多 outer 行、32B 对齐片段及 UB 内融合写回路径 |
| `chunk_aligned` | `outerSize=1` 的大对齐片段、多核 chunk 调度 |
| `many_inputs` | 约数百个小片段，放大动态 TensorList 元数据开销 |

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
# 算子代码基线应包含 b041908、767c220 和 75eec09

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
`[128, 256]` FP16 非对齐随机分片。官方系统下一次提交应记录算子代码基线 `b041908`。
若精度或性能回退，可先测试 `767c220` 隔离 UB 融合路径，再测试 `75eec09` 复现本轮
`589.168 us` 结果。

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
