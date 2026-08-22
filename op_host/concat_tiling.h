#ifndef CONCAT_TILING_H
#define CONCAT_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
constexpr uint32_t kPreloadedSegmentCount = 16U;

BEGIN_TILING_DATA_DEF(ConcatTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, outerSize);
    TILING_DATA_FIELD_DEF(uint64_t, outputRowBytes);
    TILING_DATA_FIELD_DEF(uint32_t, inputCount);
    TILING_DATA_FIELD_DEF(uint32_t, concatDim);
    TILING_DATA_FIELD_DEF(uint32_t, elementBytes);
    TILING_DATA_FIELD_DEF(uint32_t, scheduleMode);
    TILING_DATA_FIELD_DEF(uint32_t, tileBytes);
    TILING_DATA_FIELD_DEF(uint32_t, allSegmentsAligned);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, 16, segmentBytes);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Concat, ConcatTilingData)
}  // namespace optiling

#endif  // CONCAT_TILING_H
