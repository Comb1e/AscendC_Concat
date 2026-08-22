#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

using namespace AscendC;

namespace {
constexpr uint32_t kBufferCount = 2;
constexpr uint32_t kMaxRank = 8;

__aicore__ inline uint64_t MinU64(uint64_t lhs, uint64_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

template <typename Queue>
__aicore__ inline void CopyBytes(GlobalTensor<uint8_t>& destination, uint64_t destinationOffset,
                                 GlobalTensor<uint8_t>& source, uint64_t sourceOffset,
                                 uint32_t bytes, Queue& queue)
{
    DataCopyExtParams copyParams{1, bytes, 0, 0, 0};
    DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
    LocalTensor<uint8_t> local = queue.template AllocTensor<uint8_t>();
    DataCopyPad(local, source[sourceOffset], copyParams, padParams);
    queue.template EnQue<QuePosition::VECIN, QuePosition::VECOUT, uint8_t>(local);
    local = queue.template DeQue<QuePosition::VECIN, QuePosition::VECOUT, uint8_t>();
    DataCopyPad(destination[destinationOffset], local, copyParams);
    queue.FreeTensor(local);
}

template <typename Queue>
__aicore__ inline void ProcessByRows(ListTensorDesc& inputs, GlobalTensor<uint8_t>& output,
                                    const ConcatTilingData& tilingData, Queue& queue)
{
    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t blockCount = GetBlockNum();
    uint64_t outputInputOffset = 0;
    for (uint32_t inputIdx = 0; inputIdx < tilingData.inputCount; ++inputIdx) {
        uint64_t shapeBuffer[kMaxRank];
        TensorDesc<uint8_t> desc;
        desc.SetShapeAddr(shapeBuffer);
        inputs.GetDesc(desc, inputIdx);

        uint64_t innerSize = 1;
        for (uint32_t axis = tilingData.concatDim + 1; axis < desc.GetDim(); ++axis) {
            innerSize *= desc.GetShape(axis);
        }
        const uint64_t segmentBytes = desc.GetShape(tilingData.concatDim) * innerSize * tilingData.elementBytes;
        GlobalTensor<uint8_t> source;
        source.SetGlobalBuffer(inputs.GetDataPtr<uint8_t>(inputIdx));
        source.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

        for (uint64_t outer = blockIdx; outer < tilingData.outerSize; outer += blockCount) {
            const uint64_t outputOffset = outer * tilingData.outputRowBytes + outputInputOffset;
            uint64_t copied = 0;
            while (copied < segmentBytes) {
                const uint32_t bytes = static_cast<uint32_t>(
                    MinU64(tilingData.tileBytes, segmentBytes - copied));
                CopyBytes(output, outputOffset + copied, source, outer * segmentBytes + copied, bytes, queue);
                copied += bytes;
            }
        }
        outputInputOffset += segmentBytes;
    }
}

template <typename Queue>
__aicore__ inline void ProcessByChunks(ListTensorDesc& inputs, GlobalTensor<uint8_t>& output,
                                      const ConcatTilingData& tilingData, Queue& queue)
{
    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t blockCount = GetBlockNum();
    uint64_t globalChunkBase = 0;
    uint64_t outputInputOffset = 0;

    for (uint32_t inputIdx = 0; inputIdx < tilingData.inputCount; ++inputIdx) {
        uint64_t shapeBuffer[kMaxRank];
        TensorDesc<uint8_t> desc;
        desc.SetShapeAddr(shapeBuffer);
        inputs.GetDesc(desc, inputIdx);

        uint64_t innerSize = 1;
        for (uint32_t axis = tilingData.concatDim + 1; axis < desc.GetDim(); ++axis) {
            innerSize *= desc.GetShape(axis);
        }
        const uint64_t segmentBytes = desc.GetShape(tilingData.concatDim) * innerSize * tilingData.elementBytes;
        const uint64_t chunkCount = (segmentBytes + tilingData.tileBytes - 1) / tilingData.tileBytes;
        const uint64_t inputWorkItems = tilingData.outerSize * chunkCount;
        const uint64_t firstWorkItem =
            (blockIdx + blockCount - globalChunkBase % blockCount) % blockCount;

        GlobalTensor<uint8_t> source;
        source.SetGlobalBuffer(inputs.GetDataPtr<uint8_t>(inputIdx));
        source.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        for (uint64_t workItem = firstWorkItem; workItem < inputWorkItems; workItem += blockCount) {
            const uint64_t outer = workItem / chunkCount;
            const uint64_t chunk = workItem - outer * chunkCount;
            const uint64_t copied = chunk * tilingData.tileBytes;
            const uint32_t bytes = static_cast<uint32_t>(
                MinU64(tilingData.tileBytes, segmentBytes - copied));
            const uint64_t outputOffset = outer * tilingData.outputRowBytes + outputInputOffset;
            CopyBytes(output, outputOffset + copied, source, outer * segmentBytes + copied, bytes, queue);
        }
        globalChunkBase += inputWorkItems;
        outputInputOffset += segmentBytes;
    }
}
}  // namespace

extern "C" __global__ __aicore__ void concat(
    GM_ADDR inputs, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);

    ListTensorDesc inputList(reinterpret_cast<__gm__ void*>(inputs));
    GlobalTensor<uint8_t> outputTensor;
    outputTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(output));

    TPipe pipe;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, kBufferCount> queue;
    pipe.InitBuffer(queue, kBufferCount, tilingData.tileBytes);

    if (tilingData.scheduleMode == 0) {
        ProcessByRows(inputList, outputTensor, tilingData, queue);
    } else {
        ProcessByChunks(inputList, outputTensor, tilingData, queue);
    }
}
