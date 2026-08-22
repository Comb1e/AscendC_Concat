#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CASE_NAME=${1:-all}
BUILD_MODE=${2:-}
PROFILE_ROOT="$SCRIPT_DIR/profiles"
CASES=(ref row_unaligned row_aligned chunk_aligned fused_tiles many_inputs single_input zero_segments preload_16 preload_17 max_inputs rank4_axis0 tile_tail)

if [ -z "${ASCEND_OPP_PATH:-}" ]; then
    echo "ASCEND_OPP_PATH is not set" >&2
    exit 1
fi

has_concat_symbols()
{
    nm -D --defined-only "$1" 2>/dev/null | awk '
        $NF ~ /^aclnnConcat(@@.*)?$/ { concat = 1 }
        $NF ~ /^aclnnConcatGetWorkspaceSize(@@.*)?$/ { workspace = 1 }
        END { exit !(concat && workspace) }
    '
}

find_concat_opapi()
{
    local search_root
    local candidate
    local -a search_roots=("$ASCEND_OPP_PATH")
    if [ -n "${ASCEND_CUSTOM_OPP_PATH:-}" ]; then
        local -a custom_roots
        local old_ifs=$IFS
        IFS=:
        read -ra custom_roots <<< "$ASCEND_CUSTOM_OPP_PATH"
        IFS=$old_ifs
        search_roots+=("${custom_roots[@]}")
    fi

    for search_root in "${search_roots[@]}"; do
        if [ ! -d "$search_root" ]; then
            continue
        fi
        while IFS= read -r candidate; do
            if has_concat_symbols "$candidate"; then
                echo "$candidate"
                return 0
            fi
        done < <(find "$search_root" -maxdepth 6 -name libcust_opapi.so -print 2>/dev/null)
    done
    return 1
}

if ! command -v nm >/dev/null 2>&1; then
    echo "nm is required to validate the installed custom op-api library" >&2
    exit 1
fi

if [ -n "${CONCAT_OPAPI_LIB:-}" ]; then
    if [ ! -f "$CONCAT_OPAPI_LIB" ] || ! has_concat_symbols "$CONCAT_OPAPI_LIB"; then
        echo "CONCAT_OPAPI_LIB does not export aclnnConcat and aclnnConcatGetWorkspaceSize: $CONCAT_OPAPI_LIB" >&2
        exit 1
    fi
else
    if ! CONCAT_OPAPI_LIB=$(find_concat_opapi); then
        echo "No installed libcust_opapi.so exporting aclnnConcat was found." >&2
        echo "Build and install build_out/custom_*.run, then check ASCEND_OPP_PATH or ASCEND_CUSTOM_OPP_PATH." >&2
        exit 1
    fi
fi
export CONCAT_OPAPI_LIB
export LD_LIBRARY_PATH="$(dirname "$CONCAT_OPAPI_LIB"):${LD_LIBRARY_PATH:-}"
echo "Using custom op-api library: $CONCAT_OPAPI_LIB"

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
