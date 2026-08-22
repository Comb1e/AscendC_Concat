#include <torch/extension.h>

#include <vector>

#include "../common/pytorch_npu_helper.hpp"

namespace {
constexpr int64_t kProfileRounds = 30;

at::Tensor ConcatProfile(const std::vector<at::Tensor>& inputs, int64_t dim,
                         const at::IntArrayRef& outputShape)
{
    TORCH_CHECK(!inputs.empty(), "concat profile requires at least one input");
    at::Tensor result;
    const at::TensorList inputList(inputs);
    for (int64_t round = 0; round < kProfileRounds; ++round) {
        result = at::empty(outputShape, inputs[0].options());
        EXEC_NPU_CMD(aclnnConcat, inputList, dim, result);
    }
    return result;
}
}  // namespace

TORCH_LIBRARY(concat_profile, m)
{
    m.def("run(Tensor[] inputs, int dim, int[] output_shape) -> Tensor");
}

TORCH_LIBRARY_IMPL(concat_profile, PrivateUse1, m)
{
    m.impl("run", &ConcatProfile);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("run", &ConcatProfile, "Profile custom aclnnConcat");
}
