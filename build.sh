#!/usr/bin/env bash
set -euo pipefail

operator_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cann_path="${ASCEND_CANN_PACKAGE_PATH:-${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-8.5.0}}"
template_dir="${cann_path}/tools/op_project_templates/ascendc/customize"
work_dir="${operator_root}/.build"
out_dir="${operator_root}/build_out"

if [[ ! -d "${template_dir}" ]]; then
    echo "CANN Ascend C project template not found: ${template_dir}" >&2
    exit 1
fi

rm -rf "${work_dir}"
cp -aL "${template_dir}" "${work_dir}"
cp -a "${operator_root}/op_host/." "${work_dir}/op_host/"
cp -a "${operator_root}/op_kernel/." "${work_dir}/op_kernel/"

sed -i 's/__ASCNED_COMPUTE_UNIT__/ascend910b/' "${work_dir}/CMakePresets.json"
sed -i '/"ENABLE_TEST"/{n;n;s/"True"/"False"/;}' "${work_dir}/CMakePresets.json"
sed -i "s#/usr/local/Ascend/latest#${cann_path}#" "${work_dir}/CMakePresets.json"

(
    cd "${work_dir}"
    bash build.sh
)

rm -rf "${out_dir}"
mkdir -p "${out_dir}"
find "${work_dir}/build_out" -maxdepth 1 -type f -name 'custom_*.run' -exec cp {} "${out_dir}/" \;

if ! find "${out_dir}" -maxdepth 1 -type f -name 'custom_*.run' | grep -q .; then
    echo "Build completed without producing a custom_*.run package" >&2
    exit 1
fi

echo "Operator package copied to ${out_dir}"
