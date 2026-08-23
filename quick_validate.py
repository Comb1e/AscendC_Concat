#!/usr/bin/env python3
"""Fast host-side checks for concat experiments and local test definitions.

This script does not execute an NPU kernel. When the staged Gather experiment is
enabled it checks its source invariants and UB/GM address model. The model remains
available as a regression reference after that experiment is reverted. It also
parses the local test scripts. Pass --build to run the repository's offline build.
"""

from __future__ import annotations

import argparse
import ast
import math
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATA_BLOCK_BYTES = 32
COMPACT_UB_BYTES = 160 * 1024
MAX_COMPACT_ROW_BYTES = 8 * 1024
MAX_COPY_ROWS = 4095
MAX_INPUTS = 16
DEFAULT_AIV_BLOCKS = 40
DTYPE_BYTES = {"float16": 2, "float32": 4, "int8": 1, "int32": 4}


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def align_up(value: int, alignment: int = DATA_BLOCK_BYTES) -> int:
    return (value + alignment - 1) // alignment * alignment


def generate_splits(total: int, max_step: int, alignment: int) -> list[int]:
    rng = random.Random(111)
    remaining_units = total // alignment
    max_units = max(1, max_step // alignment)
    result: list[int] = []
    while remaining_units > 0:
        remaining_slots = 256 - len(result)
        minimum = max(0, remaining_units - max_units * (remaining_slots - 1))
        picked = rng.randint(minimum, min(remaining_units, max_units))
        result.append(picked * alignment)
        remaining_units -= picked
    return result


def evaluate_static(node: ast.AST, names: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    if isinstance(node, ast.List):
        return [evaluate_static(item, names) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(evaluate_static(item, names) for item in node.elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -evaluate_static(node.operand, names)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mult)):
        left = evaluate_static(node.left, names)
        right = evaluate_static(node.right, names)
        return left + right if isinstance(node.op, ast.Add) else left * right
    raise ValidationError(f"unsupported static test expression at line {getattr(node, 'lineno', '?')}")


def load_local_cases() -> dict[str, dict[str, Any]]:
    path = ROOT / "local_test" / "test_op.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {"ACLNN_MAX_TENSOR_LIST_SIZE": 256}
    cases_node: ast.Dict | None = None
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if isinstance(target, ast.Name) and target.id == "ACLNN_MAX_TENSOR_LIST_SIZE":
            names[target.id] = evaluate_static(statement.value, names)
        if isinstance(target, ast.Name) and target.id == "CASES":
            require(isinstance(statement.value, ast.Dict), "CASES must be a dictionary")
            cases_node = statement.value
            break
    require(cases_node is not None, "CASES was not found in local_test/test_op.py")

    cases: dict[str, dict[str, Any]] = {}
    for key_node, value_node in zip(cases_node.keys, cases_node.values):
        require(key_node is not None and isinstance(value_node, ast.Dict), "invalid CASES entry")
        name = evaluate_static(key_node, names)
        fields: dict[str, Any] = {}
        for field_node, item_node in zip(value_node.keys, value_node.values):
            require(field_node is not None, f"invalid field in case {name}")
            field = evaluate_static(field_node, names)
            if field == "dtype":
                require(isinstance(item_node, ast.Attribute), f"case {name} has unsupported dtype")
                fields[field] = item_node.attr
            else:
                fields[field] = evaluate_static(item_node, names)
        if "splits" not in fields:
            dim = fields["dim"] % len(fields["shape"])
            fields["splits"] = generate_splits(
                fields["shape"][dim], fields["max_step"], fields["split_alignment"]
            )
        cases[name] = fields
    return cases


@dataclass(frozen=True)
class CompactShape:
    name: str
    segment_bytes: tuple[int, ...]
    element_bytes: int
    outer_size: int
    block_dim: int


def to_compact_shape(name: str, case: dict[str, Any]) -> CompactShape:
    shape = case["shape"]
    dim = case["dim"] % len(shape)
    inner = math.prod(shape[dim + 1 :])
    outer = math.prod(shape[:dim])
    element_bytes = DTYPE_BYTES[case["dtype"]]
    splits = case["splits"]
    require(sum(splits) == shape[dim], f"{name}: splits do not cover concat dimension")
    return CompactShape(
        name=name,
        segment_bytes=tuple(split * inner * element_bytes for split in splits),
        element_bytes=element_bytes,
        outer_size=outer,
        block_dim=max(1, min(DEFAULT_AIV_BLOCKS, outer)),
    )


def compact_eligible(case: CompactShape) -> bool:
    output_row_bytes = sum(case.segment_bytes)
    all_aligned = all(segment % DATA_BLOCK_BYTES == 0 for segment in case.segment_bytes)
    return (
        not all_aligned
        and len(case.segment_bytes) <= MAX_INPUTS
        and case.element_bytes in (2, 4)
        and 0 < output_row_bytes <= MAX_COMPACT_ROW_BYTES
    )


def validate_compact_addresses(case: CompactShape) -> tuple[int, int, int]:
    require(compact_eligible(case), f"{case.name}: expected compact path is not eligible")
    output_row_bytes = sum(case.segment_bytes)
    output_elements = output_row_bytes // case.element_bytes
    staging_row_bytes = sum(align_up(segment) for segment in case.segment_bytes if segment)
    aligned_output_row_bytes = align_up(output_row_bytes)
    offset_buffer_bytes = align_up(output_elements * 4) + DATA_BLOCK_BYTES
    bytes_per_batch_row = staging_row_bytes + aligned_output_row_bytes
    max_core_rows = (case.outer_size + case.block_dim - 1) // case.block_dim
    ub_batch_rows = (COMPACT_UB_BYTES - offset_buffer_bytes) // bytes_per_batch_row
    batch_rows = min(MAX_COPY_ROWS, ub_batch_rows, max_core_rows)
    require(batch_rows > 0, f"{case.name}: compact batch is empty")
    require(
        offset_buffer_bytes + batch_rows * bytes_per_batch_row <= COMPACT_UB_BYTES,
        f"{case.name}: UB budget overflow",
    )

    staging_prefix = 0
    output_index = 0
    offsets: list[int] = []
    staging_allocation = batch_rows * staging_row_bytes
    for segment_bytes in case.segment_bytes:
        require(segment_bytes % case.element_bytes == 0, f"{case.name}: partial element")
        if segment_bytes == 0:
            continue
        aligned_segment_bytes = align_up(segment_bytes)
        gap_bytes = staging_row_bytes - aligned_segment_bytes
        require(gap_bytes % DATA_BLOCK_BYTES == 0, f"{case.name}: UB gap is not aligned")
        destination_stride_blocks = gap_bytes // DATA_BLOCK_BYTES
        ub_step_bytes = aligned_segment_bytes + destination_stride_blocks * DATA_BLOCK_BYTES
        require(ub_step_bytes == staging_row_bytes, f"{case.name}: encoded UB row step mismatch")
        last_ub_end = (
            staging_prefix
            + (batch_rows - 1) * ub_step_bytes
            + aligned_segment_bytes
        )
        require(last_ub_end <= staging_allocation, f"{case.name}: staged MTE write exceeds UB")

        segment_elements = segment_bytes // case.element_bytes
        offsets.extend(
            staging_prefix + element * case.element_bytes
            for element in range(segment_elements)
        )
        output_index += segment_elements
        staging_prefix += aligned_segment_bytes

    require(staging_prefix == staging_row_bytes, f"{case.name}: staging prefixes do not cover row")
    require(output_index == output_elements, f"{case.name}: Gather offset count mismatch")
    require(all(0 <= offset < staging_row_bytes for offset in offsets), f"{case.name}: Gather offset OOB")
    for row in range(batch_rows):
        require(
            row * staging_row_bytes + max(offsets) + case.element_bytes
            <= staging_allocation,
            f"{case.name}: Gather source exceeds staging allocation",
        )

    output_allocation = batch_rows * aligned_output_row_bytes
    last_output_source_end = (batch_rows - 1) * aligned_output_row_bytes + output_row_bytes
    require(last_output_source_end <= output_allocation, f"{case.name}: MTE3 source exceeds UB")
    return staging_row_bytes, batch_rows, offset_buffer_bytes + batch_rows * bytes_per_batch_row


def validate_kernel_source() -> tuple[bool, int, int]:
    source = (ROOT / "op_kernel" / "concat.cpp").read_text(encoding="utf-8")
    buffer_count_match = re.search(r"kBufferCount\s*=\s*(\d+)\s*;", source)
    require(buffer_count_match is not None, "kernel buffer count was not found")
    buffer_count = int(buffer_count_match.group(1))
    queue_depth_match = re.search(r"kQueueDepth\s*=\s*(\d+)\s*;", source)
    queue_depth = int(queue_depth_match.group(1)) if queue_depth_match else buffer_count
    require(buffer_count == 2, f"expected Double Buffer count 2, got {buffer_count}")
    require(queue_depth in (1, 2), f"unsupported queue depth {queue_depth}")

    gather_enabled = "ProcessFusedUnalignedRows" in source
    if not gather_enabled:
        require("tilingData.scheduleMode == 2" not in source, "partial staged Gather dispatch remains")
        return False, queue_depth, buffer_count

    stride_formula = re.compile(
        r"destinationStrideBlocks\s*=\s*"
        r"\(tilingData\.stagingRowBytes\s*-\s*alignedSegmentBytes\)\s*/\s*kDataBlockBytes\s*;"
    )
    copy_params = re.compile(
        r"DataCopyExtParams\s+copyInParams\s*\{\s*batchRows\s*,\s*segmentBytes\s*,\s*0\s*,"
        r"\s*destinationStrideBlocks\s*,\s*0\s*\}"
    )
    require(stride_formula.search(source) is not None, "kernel UB dstStride is not in datablocks")
    require(copy_params.search(source) is not None, "staging copy does not use destinationStrideBlocks")
    return True, queue_depth, buffer_count


def validate_known_cases() -> None:
    cases = load_local_cases()
    expected = {
        "ref": True,
        "row_unaligned": True,
        "row_unaligned_fp32": True,
        "row_unaligned_int8": False,
        "compact_limit": True,
        "compact_over_limit": False,
        "compact_zero_segments": True,
        "compact_offset_aligned": True,
        "compact_output_tail": True,
        "compact_batch_one": True,
        "row_aligned": False,
    }
    for name, should_compact in expected.items():
        require(name in cases, f"missing local case {name}")
        shape = to_compact_shape(name, cases[name])
        actual = compact_eligible(shape)
        require(actual == should_compact, f"{name}: compact eligibility changed to {actual}")
        if actual:
            staging_row, batch_rows, live_ub = validate_compact_addresses(shape)
            print(
                f"KNOWN_CASE name={name} staging_row={staging_row} "
                f"batch_rows={batch_rows} live_ub={live_ub} status=pass"
            )


def validate_random_cases(count: int, seed: int) -> None:
    rng = random.Random(seed)
    validated = 0
    while validated < count:
        element_bytes = rng.choice((2, 4))
        input_count = rng.randint(1, MAX_INPUTS)
        splits = [rng.randint(0, 256) for _ in range(input_count)]
        if sum(splits) == 0:
            continue
        segment_bytes = tuple(split * element_bytes for split in splits)
        output_row_bytes = sum(segment_bytes)
        if output_row_bytes > MAX_COMPACT_ROW_BYTES:
            continue
        if all(segment % DATA_BLOCK_BYTES == 0 for segment in segment_bytes):
            continue
        outer_size = rng.randint(1, 4096)
        case = CompactShape(
            name=f"random_{validated}",
            segment_bytes=segment_bytes,
            element_bytes=element_bytes,
            outer_size=outer_size,
            block_dim=min(DEFAULT_AIV_BLOCKS, outer_size),
        )
        validate_compact_addresses(case)
        validated += 1
    print(f"RANDOM_MODEL cases={validated} seed={seed} status=pass")


def validate_test_syntax() -> None:
    python_files = sorted((ROOT / "local_test").glob("*.py"))
    shell_files = sorted((ROOT / "local_test").glob("*.sh"))
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for path in shell_files:
        subprocess.run(["bash", "-n", str(path)], check=True)
    print(f"TEST_SYNTAX python={len(python_files)} shell={len(shell_files)} status=pass")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-cases", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--build", action="store_true", help="also run bash build.sh")
    args = parser.parse_args()
    require(args.random_cases >= 0, "--random-cases must be nonnegative")

    gather_enabled, queue_depth, buffer_count = validate_kernel_source()
    print(
        f"KERNEL_PIPELINE queue_depth={queue_depth} buffer_count={buffer_count} "
        "double_buffer=enabled status=pass"
    )
    if gather_enabled:
        print("KERNEL_SOURCE staged_gather=enabled staged_ub_stride=datablocks status=pass")
    else:
        print("KERNEL_SOURCE staged_gather=disabled status=pass")
    validate_known_cases()
    validate_random_cases(args.random_cases, args.seed)
    validate_test_syntax()
    if args.build:
        subprocess.run(["bash", "build.sh"], cwd=ROOT, check=True)
        print("OFFLINE_BUILD status=pass")
    else:
        print("OFFLINE_BUILD status=skipped (use --build)")
    print("NPU_EXECUTION status=not-run")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as error:
        print(f"VALIDATION_FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
