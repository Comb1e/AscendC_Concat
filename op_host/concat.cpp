#include "concat_tiling.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr uint32_t kSmallTileBytes = 32U * 1024U;
constexpr uint32_t kLargeTileBytes = 64U * 1024U;
constexpr uint32_t kFallbackVectorCores = 40U;
constexpr uint32_t kDataBlockBytes = 32U;
constexpr uint32_t kCompactUbBytes = 160U * 1024U;
constexpr uint32_t kMaxCompactRowBytes = 8U * 1024U;
constexpr uint32_t kMaxCopyRows = 4095U;

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1U) / alignment * alignment;
}

bool NormalizeDim(int64_t rawDim, size_t rank, uint32_t& dim)
{
    if (rank == 0 || rawDim < -static_cast<int64_t>(rank) || rawDim >= static_cast<int64_t>(rank)) {
        return false;
    }
    dim = static_cast<uint32_t>(rawDim < 0 ? rawDim + static_cast<int64_t>(rank) : rawDim);
    return true;
}

uint64_t Product(const gert::Shape& shape, size_t begin, size_t end)
{
    uint64_t result = 1;
    for (size_t i = begin; i < end; ++i) {
        result *= static_cast<uint64_t>(shape.GetDim(i));
    }
    return result;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const auto* instanceInfo = context->GetIrInputInstanceInfo(0);
    const auto* attrs = context->GetAttrs();
    const auto* firstStorageShape = context->GetDynamicInputShape(0, 0);
    const auto* firstDesc = context->GetInputDesc(0);
    if (instanceInfo == nullptr || attrs == nullptr || firstStorageShape == nullptr || firstDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const size_t inputCount = instanceInfo->GetInstanceNum();
    const auto& firstShape = firstStorageShape->GetStorageShape();
    const size_t rank = firstShape.GetDimNum();
    const auto* dimAttr = attrs->GetAttrPointer<int64_t>(0);
    uint32_t concatDim = 0;
    if (inputCount == 0 || dimAttr == nullptr || !NormalizeDim(*dimAttr, rank, concatDim)) {
        return ge::GRAPH_FAILED;
    }

    const int32_t dtypeBytes = ge::GetSizeByDataType(firstDesc->GetDataType());
    if (dtypeBytes <= 0) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t outerSize = Product(firstShape, 0, concatDim);
    const uint64_t innerSize = Product(firstShape, concatDim + 1, rank);
    uint64_t outputRowBytes = 0;
    uint64_t smallChunksPerOuter = 0;
    uint64_t largeChunksPerOuter = 0;
    uint64_t stagingRowBytes = 0;
    uint64_t preloadedSegmentBytes[optiling::kPreloadedSegmentCount] = {};
    bool allSegmentsAligned = true;
    for (size_t i = 0; i < inputCount; ++i) {
        const auto* storageShape = context->GetDynamicInputShape(0, i);
        if (storageShape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const auto& shape = storageShape->GetStorageShape();
        if (shape.GetDimNum() != rank) {
            return ge::GRAPH_FAILED;
        }
        for (size_t axis = 0; axis < rank; ++axis) {
            if (axis != concatDim && shape.GetDim(axis) != firstShape.GetDim(axis)) {
                return ge::GRAPH_FAILED;
            }
        }

        const uint64_t segmentBytes = static_cast<uint64_t>(shape.GetDim(concatDim)) * innerSize *
                                      static_cast<uint64_t>(dtypeBytes);
        if (i < optiling::kPreloadedSegmentCount) {
            preloadedSegmentBytes[i] = segmentBytes;
        }
        if (inputCount <= optiling::kPreloadedSegmentCount && segmentBytes != 0) {
            stagingRowBytes += AlignUp(segmentBytes, kDataBlockBytes);
        }
        outputRowBytes += segmentBytes;
        smallChunksPerOuter += (segmentBytes + kSmallTileBytes - 1) / kSmallTileBytes;
        largeChunksPerOuter += (segmentBytes + kLargeTileBytes - 1) / kLargeTileBytes;
        allSegmentsAligned = allSegmentsAligned && segmentBytes % 32U == 0;
    }

    uint32_t maxCoreCount = kFallbackVectorCores;
    if (context->GetPlatformInfo() != nullptr) {
        const platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
        maxCoreCount = platform.GetCoreNumAiv();
        if (maxCoreCount == 0) {
            maxCoreCount = kFallbackVectorCores;
        }
    }

    const uint64_t largeChunkWorkItems = outerSize * largeChunksPerOuter;
    const bool largeTileKeepsCoreOccupancy = outerSize >= maxCoreCount || largeChunkWorkItems >= maxCoreCount;
    const uint32_t tileBytes = largeTileKeepsCoreOccupancy ? kLargeTileBytes : kSmallTileBytes;
    const uint64_t chunksPerOuter = largeTileKeepsCoreOccupancy ? largeChunksPerOuter : smallChunksPerOuter;
    const uint64_t rowWorkItems = outerSize;
    const uint64_t chunkWorkItems = outerSize * chunksPerOuter;
    const uint64_t rowCoreCount = std::min<uint64_t>(maxCoreCount, rowWorkItems);
    const uint64_t chunkCoreCount = std::min<uint64_t>(maxCoreCount, chunkWorkItems);
    // Keep the lower-overhead row path unless chunking activates more AIV cores.
    const bool rowSchedule = chunkCoreCount <= rowCoreCount;
    const uint64_t workItems = rowSchedule ? rowWorkItems : chunkWorkItems;
    const uint32_t blockDim = static_cast<uint32_t>(
        std::max<uint64_t>(1, std::min<uint64_t>(maxCoreCount, workItems)));

    uint32_t scheduleMode = rowSchedule ? 0U : 1U;
    uint32_t alignedOutputRowBytes = 0;
    uint32_t compactBatchRows = 0;
    const bool compactRowsEligible =
        rowSchedule && !allSegmentsAligned && inputCount <= optiling::kPreloadedSegmentCount &&
        (dtypeBytes == 2 || dtypeBytes == 4) && outputRowBytes != 0 &&
        outputRowBytes <= kMaxCompactRowBytes;
    if (compactRowsEligible) {
        const uint64_t alignedOutputBytes = AlignUp(outputRowBytes, kDataBlockBytes);
        const uint64_t outputElements = outputRowBytes / static_cast<uint64_t>(dtypeBytes);
        const uint64_t offsetBufferBytes = AlignUp(outputElements * sizeof(uint32_t), kDataBlockBytes);
        const uint64_t bytesPerBatchRow = stagingRowBytes + alignedOutputBytes;
        if (offsetBufferBytes < kCompactUbBytes && bytesPerBatchRow != 0) {
            const uint64_t ubBatchRows = (kCompactUbBytes - offsetBufferBytes) / bytesPerBatchRow;
            const uint64_t maxCoreRows = (outerSize + blockDim - 1U) / blockDim;
            const uint64_t batchRows = std::min<uint64_t>(
                kMaxCopyRows, std::min<uint64_t>(ubBatchRows, maxCoreRows));
            if (batchRows != 0) {
                scheduleMode = 2U;
                alignedOutputRowBytes = static_cast<uint32_t>(alignedOutputBytes);
                compactBatchRows = static_cast<uint32_t>(batchRows);
            }
        }
    }

    ConcatTilingData tiling;
    tiling.set_outerSize(outerSize);
    tiling.set_outputRowBytes(outputRowBytes);
    tiling.set_inputCount(static_cast<uint32_t>(inputCount));
    tiling.set_concatDim(concatDim);
    tiling.set_elementBytes(static_cast<uint32_t>(dtypeBytes));
    tiling.set_scheduleMode(scheduleMode);
    tiling.set_tileBytes(tileBytes);
    tiling.set_allSegmentsAligned(allSegmentsAligned ? 1U : 0U);
    tiling.set_stagingRowBytes(static_cast<uint32_t>(stagingRowBytes));
    tiling.set_alignedOutputRowBytes(alignedOutputRowBytes);
    tiling.set_compactBatchRows(compactBatchRows);
    tiling.set_segmentBytes(preloadedSegmentBytes);

    context->SetBlockDim(blockDim);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const auto* instanceInfo = context->GetIrInputInstanceInfo(0);
    const auto* attrs = context->GetAttrs();
    const auto* firstShape = context->GetDynamicInputShape(0, 0);
    auto* outputShape = context->GetOutputShape(0);
    if (instanceInfo == nullptr || attrs == nullptr || firstShape == nullptr || outputShape == nullptr) {
        return GRAPH_FAILED;
    }

    const size_t inputCount = instanceInfo->GetInstanceNum();
    const size_t rank = firstShape->GetDimNum();
    const auto* dimAttr = attrs->GetAttrPointer<int64_t>(0);
    uint32_t concatDim = 0;
    if (inputCount == 0 || dimAttr == nullptr || !NormalizeDim(*dimAttr, rank, concatDim)) {
        return GRAPH_FAILED;
    }

    *outputShape = *firstShape;
    int64_t concatSize = 0;
    for (size_t i = 0; i < inputCount; ++i) {
        const auto* shape = context->GetDynamicInputShape(0, i);
        if (shape == nullptr || shape->GetDimNum() != rank) {
            return GRAPH_FAILED;
        }
        for (size_t axis = 0; axis < rank; ++axis) {
            if (axis != concatDim && shape->GetDim(axis) != firstShape->GetDim(axis)) {
                return GRAPH_FAILED;
            }
        }
        if (shape->GetDim(concatDim) > std::numeric_limits<int64_t>::max() - concatSize) {
            return GRAPH_FAILED;
        }
        concatSize += shape->GetDim(concatDim);
    }
    outputShape->SetDim(concatDim, concatSize);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetDynamicInputDataType(0, 0));
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Concat : public OpDef {
public:
    explicit Concat(const char* name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> dtypes = {
            ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8};
        const std::initializer_list<ge::Format> formats = {
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

        this->Input("inputs")
            .ParamType(DYNAMIC)
            .DataType(dtypes)
            .Format(formats)
            .UnknownShapeFormat(formats);
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType(dtypes)
            .Format(formats)
            .UnknownShapeFormat(formats);
        this->Attr("dim").Int();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Concat);
}  // namespace ops
