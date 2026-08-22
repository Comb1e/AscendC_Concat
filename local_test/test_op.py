import random
import sys

import torch
import torch_npu

import concat_profile_lib


torch.npu.config.allow_internal_format = False

ACLNN_MAX_TENSOR_LIST_SIZE = 256

CASES = {
    "ref": {
        "shape": (128, 256),
        "dtype": torch.float16,
        "dim": -1,
        "max_step": 64,
        "split_alignment": 1,
    },
    "row_unaligned": {
        "shape": (1024, 257),
        "dtype": torch.float16,
        "dim": -1,
        "max_step": 64,
        "split_alignment": 1,
    },
    "row_aligned": {
        "shape": (4096, 1024),
        "dtype": torch.float32,
        "dim": -1,
        "max_step": 256,
        "split_alignment": 8,
    },
    "chunk_aligned": {
        "shape": (4096, 1024),
        "dtype": torch.float32,
        "dim": 0,
        "max_step": 512,
        "split_alignment": 1,
    },
    "fused_tiles": {
        "shape": (8, 524288),
        "dtype": torch.int8,
        "dim": -1,
        "splits": [32768] * 16,
    },
    "many_inputs": {
        "shape": (64, 10000),
        "dtype": torch.int8,
        "dim": -1,
        "max_step": 64,
        "split_alignment": 1,
    },
    "single_input": {
        "shape": (256, 4096),
        "dtype": torch.float16,
        "dim": -1,
        "splits": [4096],
    },
    "zero_segments": {
        "shape": (8, 256),
        "dtype": torch.int32,
        "dim": -1,
        "splits": [0, 64, 0, 192],
    },
    "preload_16": {
        "shape": (32, 512),
        "dtype": torch.float32,
        "dim": -1,
        "splits": [32] * 16,
    },
    "preload_17": {
        "shape": (32, 544),
        "dtype": torch.float32,
        "dim": -1,
        "splits": [32] * 17,
    },
    "max_inputs": {
        "shape": (2, 256),
        "dtype": torch.int8,
        "dim": -1,
        "splits": [1] * ACLNN_MAX_TENSOR_LIST_SIZE,
    },
    "rank4_axis0": {
        "shape": (8, 4, 8, 16),
        "dtype": torch.float16,
        "dim": 0,
        "splits": [0, 1, 2, 5],
    },
    "tile_tail": {
        "shape": (2, 70000),
        "dtype": torch.int8,
        "dim": -1,
        "splits": [65536, 4464],
    },
}


def generate_splits(total: int, max_step: int, alignment: int) -> list[int]:
    if total <= 0 or max_step <= 0 or alignment <= 0:
        raise ValueError("total, max_step and alignment must be positive")
    if total % alignment != 0:
        raise ValueError("aligned test dimension must be divisible by split_alignment")

    rng = random.Random(111)
    remaining_units = total // alignment
    max_units = max(1, max_step // alignment)
    if remaining_units > ACLNN_MAX_TENSOR_LIST_SIZE * max_units:
        raise ValueError(
            f"cannot split {total} elements into at most {ACLNN_MAX_TENSOR_LIST_SIZE} "
            f"parts with max_step={max_step} and alignment={alignment}"
        )

    split_units = []
    while remaining_units > 0:
        remaining_slots = ACLNN_MAX_TENSOR_LIST_SIZE - len(split_units)
        min_pick = max(0, remaining_units - max_units * (remaining_slots - 1))
        picked = rng.randint(min_pick, min(remaining_units, max_units))
        split_units.append(picked)
        remaining_units -= picked
    return [units * alignment for units in split_units]


def make_input(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    torch.manual_seed(2026)
    if dtype in (torch.int8, torch.int32):
        return torch.randint(-100, 101, shape, dtype=dtype)
    return (torch.rand(shape, dtype=torch.float32) * 1000.0 - 500.0).to(dtype)


def run_case(case_name: str) -> None:
    if case_name not in CASES:
        raise KeyError(f"unknown case {case_name!r}; choose from {', '.join(CASES)}")

    case = CASES[case_name]
    source = make_input(case["shape"], case["dtype"])
    normalized_dim = case["dim"] % source.dim()
    if "splits" in case:
        splits = case["splits"]
        if sum(splits) != source.shape[normalized_dim] or any(split < 0 for split in splits):
            raise AssertionError("fixed splits must be nonnegative and cover the concat dimension")
    else:
        splits = generate_splits(
            source.shape[normalized_dim], case["max_step"], case["split_alignment"]
        )
    if len(splits) > ACLNN_MAX_TENSOR_LIST_SIZE:
        raise AssertionError(
            f"generated {len(splits)} inputs, ACLNN limit is {ACLNN_MAX_TENSOR_LIST_SIZE}"
        )
    cpu_inputs = list(torch.split(source, splits, dim=case["dim"]))
    npu_inputs = [tensor.npu() for tensor in cpu_inputs]

    print(
        f"CASE_CONFIG name={case_name} shape={tuple(source.shape)} dtype={source.dtype} "
        f"dim={case['dim']} inputs={len(splits)}"
    )
    print(f"SPLITS first16={splits[:16]} zero_count={splits.count(0)}")

    output_npu = concat_profile_lib.run(npu_inputs, case["dim"], list(source.shape))
    output = output_npu.cpu()
    if not torch.equal(output, source):
        mismatches = torch.count_nonzero(output != source).item()
        raise AssertionError(f"{case_name} failed bit-exact comparison: {mismatches} mismatches")
    print(f"CASE_RESULT name={case_name} correctness=pass")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: python3 {sys.argv[0]} <{'|'.join(CASES)}>")
    run_case(sys.argv[1])
