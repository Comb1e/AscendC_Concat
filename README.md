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
| `fd2f8ab` | 将动态 Tensor 描述符解析移出 outer 循环 | 减少大 outer 或多输入场景的 Scalar 开销 | 核对循环交换后的输出前缀地址 |
| `61555ba` | 按行/按 chunk 的实际活跃核数选择调度；chunk 核只遍历自己的线性工作项 | 少量大行也能使用更多 AIV，避免所有核扫描全部 outer | 核对 work-item 到 outer/chunk 的唯一映射 |
| `92e6de8` | 对一次性读取的输入设置 L2 Cache bypass | 减少大规模流式读取的 L2 污染 | 重复使用同一输入或上下游复用时可能回退 |
| `128fea9` | 在不降低核占用率时将 tile 从 32 KiB 提升到 64 KiB | 减少 DMA 指令和启动开销 | 大 tile 不能减少有效并行工作项 |
| `b30737a` | 每核改为连续均衡 outer 区间；在 UB 容量内合并多行 `DataCopyPad` | 减少小 inner、大 outer 场景的逐行 DMA 与循环开销 | 910B 非对齐 UB 行按 32B 计空间，blockCount 不超过 4095，stride 超限时回退单行 |
| `faff960` | Host 下发 `allSegmentsAligned`；全对齐场景使用普通 `DataCopy` | 避免对齐输入承担 `DataCopyPad` 开销 | 任一片段非对齐时必须完整保留 Pad 路径 |

当前推荐测试版本为 `faff960`。

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

## NPU 验证

先测试当前累计版本：

```bash
git log --oneline -8
# 历史中应包含 faff960；当前 HEAD 还会包含后续文档提交

bash build.sh
# 按比赛环境原有流程安装 custom_*.run，运行精度与五个性能样例
```

性能测试应先预热，再在相同环境重复运行并取中位数。请同时记录精度结果、SoC/CANN 版本和 commit：

```text
commit: <git rev-parse --short HEAD 的输出>
code_baseline: faff960
soc/cann: <version>
correctness: pass|fail
times_us: <case1>, <case2>, <case3>, <case4>, <case5>
```

若当前版本精度失败，先测试 `b30737a`，以隔离对齐快路径。若精度正确但性能回退，建议依次测试：

```text
b30737a -> 128fea9 -> 92e6de8 -> 61555ba -> fd2f8ab
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
