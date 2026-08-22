#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

using namespace AscendC;

namespace {
constexpr uint32_t kBufferCount = 2;
constexpr uint32_t kMaxRank = 8;
constexpr uint32_t kPreloadedSegmentCount = 16;
constexpr uint32_t kDataBlockBytes = 32;
constexpr uint64_t kMaxCopyStride = 0xFFFFFFFFULL;
constexpr uint64_t kMaxCopyRows = 4095ULL;
constexpr uint64_t kMaxAlignedCopyStride = 65535ULL * kDataBlockBytes;

__aicore__ inline uint64_t MinU64(uint64_t lhs, uint64_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline uint64_t AlignUpU64(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

__aicore__ inline uint64_t LoadInput(ListTensorDesc& inputs, uint32_t inputIdx,
                                    const ConcatTilingData& tilingData,
                                    GlobalTensor<uint8_t>& source)
{
    if (tilingData.inputCount <= kPreloadedSegmentCount) {
        const uint64_t segmentBytes = tilingData.segmentBytes[inputIdx];
        if (segmentBytes != 0) {
            source.SetGlobalBuffer(inputs.GetDataPtr<uint8_t>(inputIdx));
        }
        return segmentBytes;
    }

    uint64_t shapeBuffer[kMaxRank];
    TensorDesc<uint8_t> desc;
    desc.SetShapeAddr(shapeBuffer);
    inputs.GetDesc(desc, inputIdx);

    uint64_t innerSize = 1;
    for (uint32_t axis = tilingData.concatDim + 1; axis < desc.GetDim(); ++axis) {
        innerSize *= desc.GetShape(axis);
    }
    const uint64_t segmentBytes =
        desc.GetShape(tilingData.concatDim) * innerSize * tilingData.elementBytes;
    if (segmentBytes != 0) {
        source.SetGlobalBuffer(desc.GetDataPtr());
    }
    return segmentBytes;
}

template <typename Queue>
__aicore__ inline void CopyStridedBytes(GlobalTensor<uint8_t>& destination, uint64_t destinationOffset,
                                        GlobalTensor<uint8_t>& source, uint64_t sourceOffset,
                                        uint16_t rows, uint32_t bytes, uint32_t sourceStride,
                                        uint32_t destinationStride, bool aligned, Queue& queue)
{
    LocalTensor<uint8_t> local = queue.template AllocTensor<uint8_t>();
    if (aligned) {
        DataCopyParams copyInParams{rows, static_cast<uint16_t>(bytes / kDataBlockBytes),
                                    static_cast<uint16_t>(sourceStride / kDataBlockBytes), 0};
        DataCopy(local, source[sourceOffset], copyInParams);
    } else {
        DataCopyExtParams copyInParams{rows, bytes, sourceStride, 0, 0};
        DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
        DataCopyPad(local, source[sourceOffset], copyInParams, padParams);
    }
    queue.template EnQue<QuePosition::VECIN, QuePosition::VECOUT, uint8_t>(local);
    local = queue.template DeQue<QuePosition::VECIN, QuePosition::VECOUT, uint8_t>();
    if (aligned) {
        DataCopyParams copyOutParams{rows, static_cast<uint16_t>(bytes / kDataBlockBytes), 0,
                                     static_cast<uint16_t>(destinationStride / kDataBlockBytes)};
        DataCopy(destination[destinationOffset], local, copyOutParams);
    } else {
        DataCopyExtParams copyOutParams{rows, bytes, 0, destinationStride, 0};
        DataCopyPad(destination[destinationOffset], local, copyOutParams);
    }
    queue.FreeTensor(local);
}

template <typename Queue>
__aicore__ inline void CopyBytes(GlobalTensor<uint8_t>& destination, uint64_t destinationOffset,
                                 GlobalTensor<uint8_t>& source, uint64_t sourceOffset,
                                 uint32_t bytes, bool aligned, Queue& queue)
{
    CopyStridedBytes(destination, destinationOffset, source, sourceOffset, 1, bytes, 0, 0, aligned, queue);
}

template <typename Queue>
__aicore__ inline void ProcessByRows(ListTensorDesc& inputs, GlobalTensor<uint8_t>& output,
                                    const ConcatTilingData& tilingData, Queue& queue)
{
    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t blockCount = GetBlockNum();
    const uint64_t rowsPerCore = tilingData.outerSize / blockCount;
    const uint64_t extraRows = tilingData.outerSize % blockCount;
    const uint64_t coreRows = rowsPerCore + (blockIdx < extraRows ? 1 : 0);
    const uint64_t firstOuter = blockIdx * rowsPerCore + MinU64(blockIdx, extraRows);
    uint64_t outputInputOffset = 0;
    for (uint32_t inputIdx = 0; inputIdx < tilingData.inputCount; ++inputIdx) {
        GlobalTensor<uint8_t> source;
        const uint64_t segmentBytes = LoadInput(inputs, inputIdx, tilingData, source);

        uint64_t copied = 0;
        while (copied < segmentBytes) {
            const uint32_t bytes = static_cast<uint32_t>(
                MinU64(tilingData.tileBytes, segmentBytes - copied));
            const uint64_t sourceStride = segmentBytes - bytes;
            const uint64_t destinationStride = tilingData.outputRowBytes - bytes;
            const uint64_t alignedBytes = (bytes + kDataBlockBytes - 1) / kDataBlockBytes * kDataBlockBytes;
            uint64_t maxBatchRows = tilingData.tileBytes / alignedBytes;
            if (sourceStride > kMaxCopyStride || destinationStride > kMaxCopyStride) {
                maxBatchRows = 1;
            }
            maxBatchRows = MinU64(maxBatchRows, kMaxCopyRows);

            uint64_t row = 0;
            while (row < coreRows) {
                const uint16_t batchRows = static_cast<uint16_t>(MinU64(maxBatchRows, coreRows - row));
                const uint64_t outer = firstOuter + row;
                const uint64_t sourceOffset = outer * segmentBytes + copied;
                const uint64_t outputOffset = outer * tilingData.outputRowBytes + outputInputOffset + copied;
                const uint32_t copySourceStride = batchRows > 1 ? static_cast<uint32_t>(sourceStride) : 0;
                const uint32_t copyDestinationStride =
                    batchRows > 1 ? static_cast<uint32_t>(destinationStride) : 0;
                const bool aligned = tilingData.allSegmentsAligned != 0 &&
                                     (batchRows == 1 || (sourceStride <= kMaxAlignedCopyStride &&
                                                        destinationStride <= kMaxAlignedCopyStride));
                CopyStridedBytes(output, outputOffset, source, sourceOffset, batchRows, bytes,
                                 copySourceStride, copyDestinationStride, aligned, queue);
                row += batchRows;
            }
            copied += bytes;
        }
        outputInputOffset += segmentBytes;
    }
}

template <typename Queue>
__aicore__ inline void ProcessFusedAlignedRows(ListTensorDesc& inputs, GlobalTensor<uint8_t>& output,
                                              const ConcatTilingData& tilingData, Queue& queue)
{
    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t blockCount = GetBlockNum();
    const uint64_t rowsPerCore = tilingData.outerSize / blockCount;
    const uint64_t extraRows = tilingData.outerSize % blockCount;
    const uint64_t coreRows = rowsPerCore + (blockIdx < extraRows ? 1 : 0);
    const uint64_t firstOuter = blockIdx * rowsPerCore + MinU64(blockIdx, extraRows);
    const uint64_t rowsPerBatch = tilingData.tileBytes / tilingData.outputRowBytes;

    __gm__ uint8_t* sourcePointers[kPreloadedSegmentCount];
    for (uint32_t inputIdx = 0; inputIdx < tilingData.inputCount; ++inputIdx) {
        if (tilingData.segmentBytes[inputIdx] != 0) {
            sourcePointers[inputIdx] = inputs.GetDataPtr<uint8_t>(inputIdx);
        }
    }

    uint64_t row = 0;
    while (row < coreRows) {
        const uint16_t batchRows = static_cast<uint16_t>(MinU64(rowsPerBatch, coreRows - row));
        LocalTensor<uint8_t> local = queue.template AllocTensor<uint8_t>();
        uint64_t outputInputOffset = 0;
        for (uint32_t inputIdx = 0; inputIdx < tilingData.inputCount; ++inputIdx) {
            const uint64_t segmentBytes = tilingData.segmentBytes[inputIdx];
            if (segmentBytes != 0) {
                GlobalTensor<uint8_t> source;
                source.SetGlobalBuffer(sourcePointers[inputIdx]);
                DataCopyParams copyInParams{
                    batchRows, static_cast<uint16_t>(segmentBytes / kDataBlockBytes), 0,
                    static_cast<uint16_t>((tilingData.outputRowBytes - segmentBytes) / kDataBlockBytes)};
                DataCopy(local[outputInputOffset], source[(firstOuter + row) * segmentBytes], copyInParams);
            }
            outputInputOffset += segmentBytes;
        }

        queue.template EnQue<QuePosition::VECIN, QuePosition::VECOUT, uint8_t>(local);
        local = queue.template DeQue<QuePosition::VECIN, QuePosition::VECOUT, uint8_t>();
        const uint32_t batchBytes = static_cast<uint32_t>(batchRows * tilingData.outputRowBytes);
        DataCopyParams copyOutParams{
            1, static_cast<uint16_t>(batchBytes / kDataBlockBytes), 0, 0};
        DataCopy(output[(firstOuter + row) * tilingData.outputRowBytes], local, copyOutParams);
        queue.FreeTensor(local);
        row += batchRows;
    }
}

__aicore__ inline void ProcessFusedUnalignedRows(ListTensorDesc& inputs,
                                                 GlobalTensor<uint8_t>& output,
                                                 const ConcatTilingData& tilingData,
                                                 TPipe& pipe)
{
    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t blockCount = GetBlockNum();
    const uint64_t rowsPerCore = tilingData.outerSize / blockCount;
    const uint64_t extraRows = tilingData.outerSize % blockCount;
    const uint64_t coreRows = rowsPerCore + (blockIdx < extraRows ? 1 : 0);
    const uint64_t firstOuter = blockIdx * rowsPerCore + MinU64(blockIdx, extraRows);
    const uint32_t outputElements =
        static_cast<uint32_t>(tilingData.outputRowBytes / tilingData.elementBytes);
    const uint32_t offsetBufferBytes = static_cast<uint32_t>(
        AlignUpU64(static_cast<uint64_t>(outputElements) * sizeof(uint32_t), kDataBlockBytes));
    const uint32_t stagingBufferBytes = tilingData.compactBatchRows * tilingData.stagingRowBytes;
    const uint32_t outputBufferBytes = tilingData.compactBatchRows * tilingData.alignedOutputRowBytes;

    TQue<QuePosition::VECIN, 1> stagingQueue;
    TQue<QuePosition::VECOUT, 1> outputQueue;
    TBuf<QuePosition::VECCALC> offsetBuffer;
    pipe.InitBuffer(stagingQueue, 1, stagingBufferBytes);
    pipe.InitBuffer(outputQueue, 1, outputBufferBytes);
    pipe.InitBuffer(offsetBuffer, offsetBufferBytes);

    LocalTensor<int32_t> offsetsInt = offsetBuffer.Get<int32_t>();
    uint32_t outputElementOffset = 0;
    uint32_t stagingInputOffset = 0;
    for (uint32_t inputIdx = 0; inputIdx < tilingData.inputCount; ++inputIdx) {
        const uint32_t segmentBytes = static_cast<uint32_t>(tilingData.segmentBytes[inputIdx]);
        const uint32_t segmentElements = segmentBytes / tilingData.elementBytes;
        if (segmentElements != 0) {
            ArithProgression(offsetsInt[outputElementOffset], static_cast<int32_t>(stagingInputOffset),
                             static_cast<int32_t>(tilingData.elementBytes),
                             static_cast<int32_t>(segmentElements));
        }
        outputElementOffset += segmentElements;
        stagingInputOffset += static_cast<uint32_t>(AlignUpU64(segmentBytes, kDataBlockBytes));
    }
    PipeBarrier<PIPE_V>();
    LocalTensor<uint32_t> offsets = offsetsInt.ReinterpretCast<uint32_t>();

    __gm__ uint8_t* sourcePointers[kPreloadedSegmentCount];
    for (uint32_t inputIdx = 0; inputIdx < tilingData.inputCount; ++inputIdx) {
        if (tilingData.segmentBytes[inputIdx] != 0) {
            sourcePointers[inputIdx] = inputs.GetDataPtr<uint8_t>(inputIdx);
        }
    }

    uint64_t row = 0;
    while (row < coreRows) {
        const uint16_t batchRows = static_cast<uint16_t>(
            MinU64(tilingData.compactBatchRows, coreRows - row));
        LocalTensor<uint8_t> staging = stagingQueue.AllocTensor<uint8_t>();
        uint32_t stagingOffset = 0;
        for (uint32_t inputIdx = 0; inputIdx < tilingData.inputCount; ++inputIdx) {
            const uint32_t segmentBytes = static_cast<uint32_t>(tilingData.segmentBytes[inputIdx]);
            if (segmentBytes != 0) {
                GlobalTensor<uint8_t> source;
                source.SetGlobalBuffer(sourcePointers[inputIdx]);
                const uint32_t alignedSegmentBytes = static_cast<uint32_t>(
                    AlignUpU64(segmentBytes, kDataBlockBytes));
                DataCopyExtParams copyInParams{
                    batchRows, segmentBytes, 0,
                    (tilingData.stagingRowBytes - alignedSegmentBytes) / kDataBlockBytes, 0};
                DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
                DataCopyPad(staging[stagingOffset],
                            source[(firstOuter + row) * segmentBytes], copyInParams, padParams);
                stagingOffset += alignedSegmentBytes;
            }
        }
        stagingQueue.EnQue<uint8_t>(staging);
        staging = stagingQueue.DeQue<uint8_t>();

        LocalTensor<uint8_t> compact = outputQueue.AllocTensor<uint8_t>();
        if (tilingData.elementBytes == sizeof(uint16_t)) {
            LocalTensor<uint16_t> stagingElements = staging.ReinterpretCast<uint16_t>();
            LocalTensor<uint16_t> outputElementsLocal = compact.ReinterpretCast<uint16_t>();
            for (uint32_t batchRow = 0; batchRow < batchRows; ++batchRow) {
                Gather(outputElementsLocal[
                           batchRow * tilingData.alignedOutputRowBytes / sizeof(uint16_t)],
                       stagingElements, offsets, batchRow * tilingData.stagingRowBytes,
                       outputElements);
            }
        } else {
            LocalTensor<uint32_t> stagingElements = staging.ReinterpretCast<uint32_t>();
            LocalTensor<uint32_t> outputElementsLocal = compact.ReinterpretCast<uint32_t>();
            for (uint32_t batchRow = 0; batchRow < batchRows; ++batchRow) {
                Gather(outputElementsLocal[
                           batchRow * tilingData.alignedOutputRowBytes / sizeof(uint32_t)],
                       stagingElements, offsets, batchRow * tilingData.stagingRowBytes,
                       outputElements);
            }
        }
        outputQueue.EnQue<uint8_t>(compact);
        stagingQueue.FreeTensor(staging);

        compact = outputQueue.DeQue<uint8_t>();
        DataCopyExtParams copyOutParams{
            batchRows, static_cast<uint32_t>(tilingData.outputRowBytes), 0, 0, 0};
        DataCopyPad(output[(firstOuter + row) * tilingData.outputRowBytes], compact,
                    copyOutParams);
        outputQueue.FreeTensor(compact);
        row += batchRows;
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
        GlobalTensor<uint8_t> source;
        const uint64_t segmentBytes = LoadInput(inputs, inputIdx, tilingData, source);
        const uint64_t chunkCount = (segmentBytes + tilingData.tileBytes - 1) / tilingData.tileBytes;
        const uint64_t inputWorkItems = tilingData.outerSize * chunkCount;
        const uint64_t firstWorkItem =
            (blockIdx + blockCount - globalChunkBase % blockCount) % blockCount;

        for (uint64_t workItem = firstWorkItem; workItem < inputWorkItems; workItem += blockCount) {
            const uint64_t outer = workItem / chunkCount;
            const uint64_t chunk = workItem - outer * chunkCount;
            const uint64_t copied = chunk * tilingData.tileBytes;
            const uint32_t bytes = static_cast<uint32_t>(
                MinU64(tilingData.tileBytes, segmentBytes - copied));
            const uint64_t outputOffset = outer * tilingData.outputRowBytes + outputInputOffset;
            CopyBytes(output, outputOffset + copied, source, outer * segmentBytes + copied, bytes,
                      tilingData.allSegmentsAligned != 0, queue);
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
    if (tilingData.scheduleMode == 2) {
        ProcessFusedUnalignedRows(inputList, outputTensor, tilingData, pipe);
        return;
    }

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, kBufferCount> queue;
    pipe.InitBuffer(queue, kBufferCount, tilingData.tileBytes);

    const bool fuseAlignedRows = tilingData.scheduleMode == 0 && tilingData.allSegmentsAligned != 0 &&
                                 tilingData.inputCount <= kPreloadedSegmentCount &&
                                 tilingData.outputRowBytes != 0 &&
                                 tilingData.outputRowBytes <= tilingData.tileBytes;
    if (fuseAlignedRows) {
        ProcessFusedAlignedRows(inputList, outputTensor, tilingData, queue);
    } else if (tilingData.scheduleMode == 0) {
        ProcessByRows(inputList, outputTensor, tilingData, queue);
    } else {
        ProcessByChunks(inputList, outputTensor, tilingData, queue);
    }
}
