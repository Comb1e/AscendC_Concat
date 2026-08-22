# Concat 自定义算子技术文档

## 1. 文档目的与当前状态

本文记录本项目在分析 `concat.xlsx`、现有 PyTorch 调用样例、CANN 8.5.0
官方工程模板、动态输入接口和 Concat 参考实现过程中得到的、与代码实现和后续验收有关的信息。

当前实现位于 `concat/`：

```text
concat/
├── op_host/
│   ├── concat.cpp          # 算子注册、Shape/DType 推导、Tiling
│   └── concat_tiling.h     # Host 与 Kernel 共享的 Tiling 数据定义
├── op_kernel/
│   └── concat.cpp          # Ascend C Kernel
├── build.sh                # 基于本机 CANN 8.5 模板生成 .run
└── README.md
```

本机没有可用于验收的 NPU，因此目前只能确认源码结构、构建脚本语法及接口设计；
没有完成 Kernel 编译结果确认、NPU 精度测试或性能测试。`concat/.build/` 中即使存在
中间产物，也只能视为可再生构建缓存，不能视为 NPU 测试通过的证据。

## 2. 赛题规格

规格来源为根目录 `concat.xlsx`。

| 项目 | 规格 |
| --- | --- |
| 算子名 | `Concat` |
| 参考语义 | `torch.cat` |
| 输入名 | `inputs` |
| 输入类型 | 动态 `tensor_list` |
| 输入 Rank | 1 到 4 |
| 输入 Shape | `[(N4, N3, N2, N)...]`，具体 Rank 可为 1、2、3、4 |
| 数据类型 | `float32`、`float16`、`int32`、`int8` |
| 数据格式 | `ND` |
| 属性 | `dim`，类型为 `int` |
| 输出名 | `output` |
| 输出类型 | `tensor` |
| 输出 Shape | 沿 `dim` 拼接，其余维度不变 |
| 目标芯片 | Ascend 910B |

维度范围：

```text
N  ∈ [1, 10000]
N2 ∈ [1, 10000]
N3 ∈ [1, 1000]
N4 ∈ [1, 1000]
```

题目明确指出 `N` 到 `N4` 都可能不是 32 的整数倍。因此，不能只使用要求
32 字节对齐的普通搬运路径，必须正确处理头尾非对齐数据。

表格未给出动态输入个数的上限。测试样例通过随机拆分产生输入列表，其中允许某个
输入在拼接维度上的长度为 0，所以实现也需要允许零长度分片。

### 2.1 数学语义

设输入个数为 `K`，每个输入为 `X_i`，Rank 为 `R`，归一化后的拼接轴为 `d`。
合法输入必须满足：

```text
0 <= d < R
shape(X_i)[a] == shape(X_0)[a], 对所有 a != d
dtype(X_i) == dtype(X_0)
```

输出 Shape 为：

```text
shape(Y)[a] = shape(X_0)[a],                         a != d
shape(Y)[d] = sum(shape(X_i)[d] for i in [0, K))
```

负轴按以下规则归一化：

```text
d = dim,          dim >= 0
d = dim + R,      dim < 0
```

合法范围为 `[-R, R - 1]`。

## 3. 总体实现架构

完整调用链如下：

```text
PyTorch 测试代码
    │
    ▼
PyTorch C++ 扩展 / ACLNN 接口 aclnnConcat
    │
    ▼
自定义 OPP 中的 Concat 注册信息
    │
    ├── InferShape / InferDataType
    ├── Host Tiling
    └── Ascend C Kernel: concat
            │
            ▼
        输出 ND Tensor
```

Host 侧负责合法性检查、输出推导、调度模式选择和核数设置；Kernel 侧只处理已经
确定的动态输入列表，并按字节把每个输入的连续片段搬到输出对应位置。Concat 不做
数值计算，因此以 `uint8_t` 进行无损原始字节搬运即可统一支持四种数据类型。

## 4. Host 侧实现

实现文件：`concat/op_host/concat.cpp`。

### 4.1 算子注册

注册类名和算子名均为 `Concat`：

```cpp
this->Input("inputs").ParamType(DYNAMIC);
this->Output("output").ParamType(REQUIRED);
this->Attr("dim").Int();
this->AICore().AddConfig("ascend910b");
```

输入和输出均注册以下 DType/Format 组合：

```text
DT_FLOAT   + FORMAT_ND
DT_FLOAT16 + FORMAT_ND
DT_INT32   + FORMAT_ND
DT_INT8    + FORMAT_ND
```

动态输入不是普通 `Input(0)、Input(1)...`。Host 侧通过：

```cpp
context->GetIrInputInstanceInfo(0)->GetInstanceNum()
context->GetDynamicInputShape(0, i)
```

分别获得第 0 个 IR 动态输入的实例个数和第 `i` 个实际输入 Shape。

### 4.2 Shape 推导

`InferShape` 执行以下步骤：

1. 检查动态输入信息、属性、首个输入 Shape 和输出 Shape 指针。
2. 检查输入列表非空。
3. 对负 `dim` 归一化，并检查范围。
4. 复制首个输入 Shape 作为输出基础 Shape。
5. 检查所有输入 Rank 相同。
6. 检查所有非拼接维度相同。
7. 累加拼接维度，并检查 `int64_t` 加法溢出。
8. 写回输出拼接维度。

`InferDataType` 把首个动态输入的 DType 设置为输出 DType。比赛调用方必须保证所有
输入 DType 相同；根目录 PyTorch 封装也包含此项检查。

### 4.3 Tiling 的逻辑展开

将一个 Rank 为 `R` 的输入按拼接轴 `d` 分解：

```text
outerSize = product(shape[0:d])
innerSize = product(shape[d+1:R])
```

空乘积为 1。例如：

- Rank 2、`dim=0`：`outerSize=1`，`innerSize=shape[1]`。
- Rank 2、`dim=1`：`outerSize=shape[0]`，`innerSize=1`。
- Rank 4、`dim=2`：`outerSize=shape[0]*shape[1]`，`innerSize=shape[3]`。

对于输入 `i`，每个 outer 行需要搬运的连续字节数为：

```text
segmentBytes_i = shape_i[d] * innerSize * elementBytes
```

输出每个 outer 行的字节数为：

```text
outputRowBytes = sum(segmentBytes_i)
```

输入 `i` 的第 `outer` 行源地址偏移为：

```text
sourceOffset = outer * segmentBytes_i
```

该行在输出中的起始地址为：

```text
outer * outputRowBytes + sum(segmentBytes_j, j < i)
```

这正是连续 ND 张量在任意轴进行 Concat 时的线性内存映射。

### 4.4 Tiling 数据

`concat/op_host/concat_tiling.h` 定义了以下字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `outerSize` | `uint64_t` | 拼接轴之前所有维度的乘积 |
| `outputRowBytes` | `uint64_t` | 一个 outer 行的输出字节数 |
| `inputCount` | `uint32_t` | 动态输入个数 |
| `concatDim` | `uint32_t` | 归一化后的拼接轴 |
| `elementBytes` | `uint32_t` | 单个元素字节数 |
| `scheduleMode` | `uint32_t` | 0 为按行调度，1 为按块调度 |
| `tileBytes` | `uint32_t` | 单次 UB 搬运块大小，当前为 32 KiB |

当前固定 `tileBytes = 32 * 1024`。Kernel 使用两个队列 Buffer，因此队列占用约
`2 * 32 KiB = 64 KiB` UB，不计框架和其他少量开销。

### 4.5 核数和双调度策略

最大 AIV 核数通过 `PlatformAscendC::GetCoreNumAiv()` 获取。平台信息缺失或返回 0
时使用 40 作为保底值，但正常编译/运行环境应使用平台返回值。

当前使用固定阈值 `outerSize >= 8` 选择调度模式：

| 条件 | 模式 | 工作项数量 | 目的 |
| --- | --- | --- | --- |
| `outerSize >= 8` | 按 outer 行调度 | `outerSize` | 减少大量小输入时重复解析动态 Tensor 描述符 |
| `outerSize < 8` | 按 32 KiB 块调度 | `outerSize * chunksPerOuter` | outer 行太少时仍可让多个 AIV 核并行搬运大段数据 |

其中：

```text
chunksPerOuter = sum(ceil(segmentBytes_i / tileBytes))
blockDim = min(AIV core count, workItems)，且至少为 1
```

这个阈值是工程启发式选择，不是题目规定的最优值。获得真实 910B 性能数据后，可
重点测试 `outerSize` 为 1、2、4、8、16 且 segment 大小跨越 32 KiB 的场景，再调整
阈值或改为基于总字节数、输入数量的多条件策略。

## 5. Kernel 侧实现

实现文件：`concat/op_kernel/concat.cpp`。

### 5.1 动态输入解析

Kernel 入口签名为：

```cpp
extern "C" __global__ __aicore__ void concat(
    GM_ADDR inputs, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
```

动态输入通过 CANN 的列表描述接口解析：

```cpp
ListTensorDesc inputList(reinterpret_cast<__gm__ void*>(inputs));
TensorDesc<uint8_t> desc;
desc.SetShapeAddr(shapeBuffer);
inputList.GetDesc(desc, inputIdx);
inputList.GetDataPtr<uint8_t>(inputIdx);
```

`shapeBuffer` 当前容量为 8 个维度。比赛规格 Rank 最大为 4，因此容量足够。该实现
不应在没有扩大 Buffer 和增加 Host 检查的情况下用于 Rank 大于 8 的输入。

### 5.2 为什么统一按字节搬运

Concat 只改变各输入片段在输出内存中的排列，不进行算术运算。将 GM Tensor 绑定为
`GlobalTensor<uint8_t>` 后按原始字节复制，可以同时覆盖：

```text
float32: 4 bytes
float16: 2 bytes
int32:   4 bytes
int8:    1 byte
```

`elementBytes` 仅用于把元素数量转换成片段字节数。搬运不会改变浮点位模式、整数符号
或精度，因此这是严格无损复制。

### 5.3 非 32 字节对齐

`CopyBytes` 使用 `DataCopyPad` 完成 GM 到 UB、UB 到 GM 的搬运：

```cpp
DataCopyPad(local, source[sourceOffset], copyParams, padParams);
DataCopyPad(destination[destinationOffset], local, copyParams);
```

每次实际字节数为：

```text
bytes = min(tileBytes, segmentBytes - copied)
```

因此最后一个 Tile 可以小于 32 KiB，也可以不是 32 字节整数倍。没有把尾部向上取整
后写回 GM，避免覆盖输出中紧随其后的输入片段。

### 5.4 按行调度

每个核处理：

```text
outer = blockIdx, blockIdx + blockCount, blockIdx + 2*blockCount, ...
```

对于自己拥有的 outer 行，该核按输入顺序解析描述符并顺序追加各 `segmentBytes`。
不同核拥有不同 outer 行，所以输出地址不重叠，不需要核间同步。

优点是每个输入描述符在每个 outer 行只解析一次，适合 outer 行较多、单行较小或输入
列表较长的情况。限制是一个很大的单行只由一个核处理，所以 outer 行很少时并行度不足。

### 5.5 按块调度

对于 outer 行少而片段大的情况，每个片段按 32 KiB 划分为多个 Chunk。使用
`globalChunkBase` 把所有 outer 行、所有输入的 Chunk 看成一个连续工作项序列。

核 `blockIdx` 处理满足以下条件的 Chunk：

```text
(globalChunkBase + chunk) % blockCount == blockIdx
```

代码中的 `firstChunk` 是上述同余方程在当前片段内的第一个非负解，之后每次加
`blockCount`。每个 Chunk 只有一个核负责，因此不存在重复写入或并发覆盖。

### 5.6 零长度输入

当某个输入在拼接轴上的长度为 0：

```text
segmentBytes = 0
chunkCount = 0
```

按行模式的 `while (copied < segmentBytes)` 不执行；按块模式的 Chunk 循环也不执行；
输出偏移增加 0。该输入被正确保留在动态输入列表语义中，但不会触发非法搬运。

## 6. 正确性依据

可以把每个输入重解释为三维逻辑布局：

```text
[outerSize, concatAxisSize_i, innerSize]
```

对固定 `outer`，中间两维在内存中形成长度为
`concatAxisSize_i * innerSize` 的连续片段。Concat 的定义就是按输入顺序将这些片段
写入输出相同 outer 行。当前 Kernel 正好执行这一操作。

两种调度只改变“哪个核复制哪个连续片段”，不改变源地址、目标地址或输入顺序：

- 按行模式划分 outer 行，核间目标区间互斥。
- 按块模式划分 Tile，核间目标 Tile 互斥。
- 每个 Tile 内部是逐字节等值复制。
- 非对齐尾块使用精确字节长度。

因此，在输入为连续 ND Tensor、Shape/DType 合法且运行时动态列表描述正确的前提下，
输出与 `torch.cat(inputs, dim)` 的位模式一致。

## 7. 构建与 `.run` 打包

### 7.1 环境要求

当前脚本针对：

```text
CANN: 8.5.0
SoC:  ascend910b
```

构建机需要安装完整 CANN 开发包，且包含：

```text
tools/op_project_templates/ascendc/customize
```

`concat/build.sh` 按以下优先级定位 CANN：

```text
1. ASCEND_CANN_PACKAGE_PATH
2. ASCEND_HOME_PATH
3. /usr/local/Ascend/cann-8.5.0
```

推荐显式设置实际路径，避免环境中 `latest` 指向错误版本：

```bash
export ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-8.5.0
bash concat/build.sh
```

### 7.2 构建脚本做了什么

脚本执行顺序为：

1. 删除并重建 `concat/.build/`。
2. 复制 CANN 8.5 官方 `customize` 完整工程模板。
3. 将本项目 `op_host/`、`op_kernel/` 复制进模板。
4. 把计算单元占位符替换为 `ascend910b`。
5. 关闭模板测试用例构建。
6. 写入实际 CANN 路径。
7. 调用官方模板的 `build.sh`。
8. 将生成的 `custom_*.run` 复制到 `concat/build_out/`。
9. 如果没有找到 `.run`，构建脚本返回失败。

预期最终产物类似：

```text
concat/build_out/custom_opp_<platform>.run
```

`.build/` 是中间工程，`build_out/` 是最终输出目录。重新构建会清理这两个目录中的
对应内容，因此不要在其中保存手工修改。

### 7.3 三个旧赛季参考文件的结论

曾检查过 `CMakePresets-ref.json`、`build-ref.sh`、`CMakeLists-ref.txt`：

- `CMakeLists-ref.txt` 与本机 CANN 8.5 官方模板一致。
- `CMakePresets-ref.json` 只是预先写入 `ascend910b`，并硬编码旧环境路径
  `/home/ma-user/Ascend/ascend-toolkit/latest`。
- `build-ref.sh` 基本是官方模板脚本，仅额外兼容旧环境变量
  `BASE_LIBS_PATH`、`ASCEND_AICPU_PATH`。

当前不需要把这三个文件复制进 `concat/`。它们本身也不足以构建，因为还依赖
`cmake/`、`scripts/`、`framework/` 和子目录 CMake 文件。当前 `concat/build.sh`
直接从实际 CANN 安装中复制完整模板，版本一致性和可移植性更好。

如果比赛环境要求完全离线、源码目录内不得依赖 CANN 的模板路径，则应随项目携带
整套 CANN 8.5 `customize` 模板，而不是只携带这三个顶层文件。是否允许这样提交应以
当前赛季页面规则为准。

## 8. ACLNN 与 PyTorch 调用

### 8.1 自动生成的自定义接口

根据当前注册信息，CANN 工程会生成：

```cpp
aclnnStatus aclnnConcatGetWorkspaceSize(
    const aclTensorList *inputs,
    int64_t dim,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnConcat(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

安装 `.run` 后，自定义接口通常位于 vendor `customize` 的 `libcust_opapi.so` 中。
根目录 `common/pytorch_npu_helper.hpp` 的查找顺序是：

```text
1. libcust_opapi.so
2. libopapi.so
```

所以自定义接口名必须使用 `aclnnConcat`，才能解析到当前项目生成的自定义 ACLNN API。

### 8.2 `.run` 和 wheel 是两种不同产物

根目录 `run.sh` 调用 `setup.py` 生成的是：

```text
dist/custom_ops-*.whl
```

该 wheel 只是 PyTorch C++ 调用封装，不包含 Ascend C 自定义 OPP 安装包，也不会生成
比赛要求的 `.run`。

`concat/build.sh` 生成的是：

```text
concat/build_out/custom_opp_*.run
```

它才是自定义算子 Host、Kernel 和 ACLNN 接口的安装包。典型验收流程应为：

```text
构建并安装 custom_opp_*.run
        ↓
确认 ASCEND_OPP_PATH/vendors/customize 下存在自定义库和 Kernel
        ↓
重新构建并安装 PyTorch wheel
        ↓
PyTorch wrapper 调用 aclnnConcat
        ↓
执行精度与性能测试
```

安装 `.run` 时应使用包自身的 `--help` 查看当前 CANN 版本支持的参数，并按赛事环境的
OPP 安装规则执行。`run.sh` 已把
`$ASCEND_OPP_PATH/vendors/customize/op_api/lib/` 加入 `LD_LIBRARY_PATH`。

### 8.3 当前根目录调用代码存在的关键问题

当前 `extension/custom_op.cpp` 中：

```cpp
EXEC_NPU_CMD(aclnnCat, input_list, dim, result);
```

`aclnnCat` 是系统内置 Cat 接口，不是本项目生成的 `aclnnConcat`。保持此代码时，测试
即使通过，也只能证明系统内置算子正确，不能证明提交的自定义 Kernel 被执行。

正确验收自定义算子前，至少需要改为：

```cpp
EXEC_NPU_CMD(aclnnConcat, input_list, dim, result);
```

另一个接口不一致是：当前 PyBind 导出名为 `concat`：

```cpp
m.def("concat", &concat_impl_npu, "torch.cat");
```

但 `test_op.py` 调用：

```python
custom_ops_lib.custom_op(inputs_npu, dim, input_x.shape)
```

重新构建当前源码后，Python 模块很可能没有 `custom_op` 属性。应统一调用名，或者同时
导出 `custom_op` 和 `concat` 两个别名。

### 8.4 旧 wheel 会造成假通过

`run.sh` 在 `dist/` 非空时直接安装已有 wheel，不会检查 C++ 源码是否比 wheel 新。
因此修改 `extension/custom_op.cpp` 后执行 `bash run.sh 1`，仍可能装回旧 wheel。

正式验收前应删除或移走 `dist/` 中旧 wheel，并重新执行：

```bash
python3 setup.py build bdist_wheel
pip3 install dist/custom_ops*.whl --force-reinstall
```

还应在日志或 Profiling CSV 中确认实际算子名为自定义 `Concat/aclnnConcat`，而不是
内置 `Cat/aclnnCat`。

### 8.5 当前测试编号不一致

当前 `test_op.py` 只定义了 `case2`，而 `run.sh` 把参数原样传给测试脚本。
因此按当前文件内容：

```text
bash run.sh 2  -> 选择 case2，但不会触发 run.sh 中参数为 1 的 wheel 安装分支
bash run.sh 1  -> 触发 wheel 安装，但 test_op.py 会寻找不存在的 case1
```

正式测试前需要修正该流程，例如补充 `case1`、统一使用 `case1`，或将“是否重装 wheel”
与“测试 Case 编号”拆成两个参数。

### 8.6 性能采样逻辑

PyTorch C++ 封装当前连续发起 30 次调用。`get_time.py` 收集 Profiling CSV 中的任务
耗时，并计算 `time_use_list[10:30]` 的中位数，即跳过前 10 次预热，统计后 20 次。

需要注意：脚本目前除 `aclnnMul` 外会收集 CSV 中所有算子任务。如果一次测试包含其他
NPU 任务，时间列表可能混入无关项。更可靠的做法是按自定义算子名称精确过滤。

## 9. 建议测试矩阵

由于当前环境没有 NPU，以下测试必须在 CANN 8.5/Ascend 910B 环境完成。

### 9.1 基础功能

| 类别 | 建议用例 |
| --- | --- |
| Rank | 1、2、3、4 全覆盖 |
| 拼接轴 | 每个合法正轴，以及对应负轴 |
| DType | `float32`、`float16`、`int32`、`int8` |
| 输入个数 | 1、2、多个、较长列表 |
| 零长度分片 | 首个、中间、最后一个输入长度为 0 |
| 非对齐 | 片段为 1、2、3、15、31、33、63、65 字节附近 |
| Tile 边界 | 片段小于、等于、大于 32 KiB，以及 32 KiB 非整数倍 |
| Shape | 拼接首轴、中间轴、末轴 |

### 9.2 调度分支

必须明确覆盖：

```text
outerSize = 1、2、4、7       -> scheduleMode = 1，按块
outerSize = 8、9、40、>40    -> scheduleMode = 0，按行
```

对 `outerSize < 8` 的用例，应至少让单个片段大于 32 KiB，才能真正验证多 Chunk 和
多核分配。对按行模式，应覆盖 outer 行数不是核数整数倍的情况。

### 9.3 边界和错误输入

Host/PyTorch 层应检查：

- 输入列表为空。
- Rank 为 0。
- `dim < -rank` 或 `dim >= rank`。
- 输入 Rank 不一致。
- 非拼接维度不一致。
- DType 不一致。
- CPU Tensor 和 NPU Tensor 混用。
- 输出 Shape 与推导结果不一致。

题目未要求错误输入行为时，错误用例主要用于确认能明确失败而不是越界访问。

### 9.4 精度判定

因为实现是纯字节复制，理论上应逐元素完全相等。建议除沿用现有浮点容差比较外，再做：

```python
expected = torch.cat(inputs_cpu, dim=dim)
actual = output_npu.cpu()
assert torch.equal(actual, expected)
```

对于浮点输入，`torch.equal` 仍可验证位级复制；若测试数据包含 NaN，需要按比赛判定规则
额外处理 NaN，因为普通相等比较会认为 NaN 不等于自身。

### 9.5 确认执行的是自定义算子

每次正式验收都应同时满足：

1. `.run` 是本次源码重新构建的。
2. `.run` 已安装到当前使用的 CANN/OPP 环境。
3. wheel 是本次 wrapper 源码重新构建的。
4. wrapper 调用 `aclnnConcat`。
5. Profiling 中出现自定义 Concat，而不是内置 Cat。
6. 修改 Kernel 后性能或日志能反映本次修改，排除缓存产物。

## 10. 已知假设、限制与风险

### 10.1 连续内存假设

Kernel 按连续 ND 布局计算线性偏移，没有读取或处理任意 Stride。因此输入必须是连续
ND Tensor，或运行时必须已将逻辑 Tensor 转换成连续存储。当前测试通过把 CPU 分片
逐个 `.npu()`，通常会得到独立 NPU Tensor，但仍应在目标环境确认。

如果赛事隐藏用例直接传入非连续 NPU View，则需要在 ACLNN/框架层做连续化，或扩展
Kernel 读取 Stride；当前实现不覆盖任意 Strided Tensor。

### 10.2 Rank 上限依赖题目规格

Host 注册允许动态 Rank，Kernel 的 Shape Buffer 容量为 8，但正式支持范围按题目限定
为 Rank 1 到 4。不要把“Buffer 可容纳 8”解释为已经支持 Rank 5 到 8。

### 10.3 同 DType 假设

输出 DType 取首个输入，Kernel 对所有输入使用同一个 `elementBytes`。若运行时允许混合
DType 输入，地址计算会错误。PyTorch wrapper 已检查 DType 相同，目标 ACLNN 路径也应
保证这一前置条件。必要时可在 Host Tiling 中对每个动态输入描述符增加 DType 一致性检查。

### 10.4 大 Shape 与计数溢出

`outerSize`、`outputRowBytes` 和 Kernel 地址均使用 `uint64_t`；题目单 Tensor 维度乘积在
`uint64_t` 范围内。输入个数没有书面上限，理论上极端数量的输入可能使累加溢出，但这种
输出远超 910B 可分配内存。若追求防御性，可为 Host 的乘法和字节累加增加显式溢出检查。

动态输入个数最终写入 `uint32_t inputCount`。实际比赛输入不可能接近 `2^32` 个 Tensor；
若做通用库，需要先限制或检查输入数量。

### 10.5 调度阈值尚未实测

`outerSize >= 8` 的切换点和 32 KiB Tile 是合理的初始工程值，但没有在真实 NPU 上调优。
性能竞争中应通过 Profiling 比较：

- 动态描述符解析开销；
- GM 搬运带宽；
- 小 Tile 调度开销；
- 单行大数据时的核利用率；
- 输入列表很长时的循环开销。

### 10.6 构建缓存和环境路径

`concat/.build` 可能包含不同环境或不同源码版本生成的文件。排障时应以全新执行
`concat/build.sh` 的结果为准。CANN `latest` 软链接可能变化，最好固定到 CANN 8.5.0
的实际路径。

## 11. 可追溯参考资料

### 11.1 项目内资料

- `concat.xlsx`：本题接口、Shape、DType、Format、范围和非对齐要求。
- `test_op.py`：随机拆分、零长度分片和精度验证方式。
- `extension/custom_op.cpp`：PyTorch 封装、输入检查和 30 次性能调用。
- `common/pytorch_npu_helper.hpp`：ACL Tensor/TensorList 转换、动态加载自定义
  `libcust_opapi.so`、Workspace 和 Stream 调用流程。
- `run.sh`、`get_time.py`：wheel 安装、msprof 启动和耗时采样方式。
- CANN 8.5.0 官方模板：
  `/usr/local/Ascend/cann-8.5.0/tools/op_project_templates/ascendc/customize`。
- CANN 8.5.0 内置 Concat Ascend C 参考入口：
  `/usr/local/Ascend/cann-8.5.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/ascendc/concat/concat.cpp`。
- 动态 TensorList 接口头文件：
  `/usr/local/Ascend/cann-8.5.0/x86_64-linux/asc/include/basic_api/kernel_operator_list_tensor_intf.h`。

### 11.2 官方在线资料

- CANN 8.5 Ascend C 动态输入：
  <https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00005.html>
- CANN 8.5 自定义算子工程编译和打包：
  <https://www.hiascend.com/document/detail/zh/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00035.html>
- CANN 8.5 自定义算子包部署：
  <https://www.hiascend.com/document/detail/zh/canncommercial/850/devaids/optool/atlasopdev_16_0024.html>
- Ascend C `DataCopyPad`：
  <https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/apiref/opdevgapi/atlasascendc_api_07_0258.html>
- PyTorch 2.5 `torch.cat`：
  <https://docs.pytorch.org/docs/2.5/generated/torch.cat.html>
- Ascend Samples C++ Extension 调用样例：
  <https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/1_add_frameworklaunch/CppExtensionInvocation>

## 12. 目标环境验收清单

按以下顺序执行可以最大限度排除旧产物和错误调用链：

```text
[ ] 确认 CANN 版本为 8.5.0，芯片为 Ascend 910B
[ ] 确认 ASCEND_CANN_PACKAGE_PATH/ASCEND_HOME_PATH 指向本次环境
[ ] 执行 bash concat/build.sh 并得到新的 custom_opp_*.run
[ ] 安装新的 .run，确认 customize vendor 路径和 libcust_opapi.so
[ ] 将 wrapper 调用由 aclnnCat 改为 aclnnConcat
[ ] 统一 PyBind 导出名与 test_op.py 调用名
[ ] 清理或移走旧 wheel，重新构建并强制安装
[ ] 修正 case1/case2 参数冲突
[ ] 运行四种 DType、Rank 1-4、正负 dim、零长度和非对齐用例
[ ] 分别覆盖按行和按块调度
[ ] 用 torch.cat 做 Golden，验证 Shape、DType 和全部元素
[ ] 在 Profiling 中确认执行的是自定义 Concat
[ ] 记录精度、性能、算子名、CANN 版本和最终 .run 校验值
[ ] 按当前赛季页面要求整理提交目录和压缩包，避免提交 .build 缓存
```

只有上述调用链确认完成后，`run.sh` 的通过结果才能作为自定义 Concat 提交代码通过的
有效证据。
