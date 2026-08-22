#ifndef CONCAT_TILING_H
#define CONCAT_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
// Keep a bounded prefix in tiling data so common 17-32 input lists avoid
// repeating descriptor shape parsing in the Kernel. Larger lists fall back.
constexpr uint32_t kPreloadedSegmentCount = 32U;

BEGIN_TILING_DATA_DEF(ConcatTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, outerSize);
    TILING_DATA_FIELD_DEF(uint64_t, outputRowBytes);
    TILING_DATA_FIELD_DEF(uint32_t, inputCount);
    TILING_DATA_FIELD_DEF(uint32_t, concatDim);
    TILING_DATA_FIELD_DEF(uint32_t, elementBytes);
    TILING_DATA_FIELD_DEF(uint32_t, scheduleMode);
    TILING_DATA_FIELD_DEF(uint32_t, tileBytes);
    TILING_DATA_FIELD_DEF(uint32_t, allSegmentsAligned);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, 32, segmentBytes);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Concat, ConcatTilingData)
}  // namespace optiling

#endif  // CONCAT_TILING_H
