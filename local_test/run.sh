#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CASE_NAME=${1:-all}
BUILD_MODE=${2:-}
PROFILE_ROOT="$SCRIPT_DIR/profiles"
CASES=(ref row_unaligned row_aligned chunk_aligned many_inputs)

if [ -z "${ASCEND_OPP_PATH:-}" ]; then
    echo "ASCEND_OPP_PATH is not set" >&2
    exit 1
fi
export LD_LIBRARY_PATH="$ASCEND_OPP_PATH/vendors/customize/op_api/lib:${LD_LIBRARY_PATH:-}"

if [ "$BUILD_MODE" = "--build" ]; then
    (
        cd "$SCRIPT_DIR"
        python3 setup.py build bdist_wheel
        pip3 install dist/concat_profile*.whl --force-reinstall
    )
fi

profile_case()
{
    local current_case=$1
    local profile_dir="$PROFILE_ROOT/$current_case"
    rm -rf "$profile_dir"
    mkdir -p "$profile_dir"

    set +e
    (
        cd "$profile_dir"
        timeout 180 msprof --application="python3 $SCRIPT_DIR/test_op.py $current_case"
    )
    local profile_status=$?
    set -e

    if [ "$profile_status" -eq 124 ]; then
        echo "$current_case timed out" >&2
        return 1
    fi
    if [ "$profile_status" -ne 0 ]; then
        echo "$current_case profiling failed with status $profile_status" >&2
        return "$profile_status"
    fi
    python3 "$SCRIPT_DIR/get_time.py" "$profile_dir" "$current_case"
}

if [ "$CASE_NAME" = "all" ]; then
    for current_case in "${CASES[@]}"; do
        profile_case "$current_case"
    done
else
    valid=0
    for current_case in "${CASES[@]}"; do
        if [ "$CASE_NAME" = "$current_case" ]; then
            valid=1
            break
        fi
    done
    if [ "$valid" -ne 1 ]; then
        echo "unknown case: $CASE_NAME" >&2
        echo "valid cases: all ${CASES[*]}" >&2
        exit 2
    fi
    profile_case "$CASE_NAME"
fi
