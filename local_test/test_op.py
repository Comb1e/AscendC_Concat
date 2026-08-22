import random
import sys

import torch
import torch_npu

import concat_profile_lib


torch.npu.config.allow_internal_format = False

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
    "many_inputs": {
        "shape": (64, 10000),
        "dtype": torch.int8,
        "dim": -1,
        "max_step": 32,
        "split_alignment": 1,
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
    split_units = []
    while remaining_units > 0:
        picked = rng.randint(0, min(remaining_units, max_units))
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
    splits = generate_splits(
        source.shape[normalized_dim], case["max_step"], case["split_alignment"]
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
