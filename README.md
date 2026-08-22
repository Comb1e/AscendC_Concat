# Concat Ascend C operator

Contents:

- op_host: dynamic-input registration, shape inference, and tiling
- op_kernel: Ascend C implementation for Ascend 910B
- build.sh: creates build_out/custom_opp_*.run with the installed CANN 8.5 template

Supported inputs match concat.xlsx: rank 1-4 ND tensors, float32, float16,
int32, and int8, with positive or negative dim. Zero-length input slices and
non-32-byte-aligned shapes are supported.

Build on the CANN 8.5 / Ascend 910B environment with:

    build.sh

The PyTorch invocation wrapper must call aclnnConcat, not the built-in
aclnnCat, when validating this custom operator.
