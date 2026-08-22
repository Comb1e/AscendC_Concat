#ifndef CONCAT_TILING_H
#define CONCAT_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConcatTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, outerSize);
    TILING_DATA_FIELD_DEF(uint64_t, outputRowBytes);
    TILING_DATA_FIELD_DEF(uint32_t, inputCount);
    TILING_DATA_FIELD_DEF(uint32_t, concatDim);
    TILING_DATA_FIELD_DEF(uint32_t, elementBytes);
    TILING_DATA_FIELD_DEF(uint32_t, scheduleMode);
    TILING_DATA_FIELD_DEF(uint32_t, tileBytes);
    TILING_DATA_FIELD_DEF(uint32_t, allSegmentsAligned);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Concat, ConcatTilingData)
}  // namespace optiling

#endif  // CONCAT_TILING_H
