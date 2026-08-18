/**
*
* Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include <torch/extension.h>
#include <torch/csrc/autograd/custom_function.h>
#include "../common/pytorch_npu_helper.hpp"
using tensor_list = std::vector<at::Tensor>;


at::Tensor my_op_impl_npu(const tensor_list inputs, int64_t dim,
                          const at::IntArrayRef& output_shape) {
    TORCH_CHECK(!inputs.empty(), "concat expects at least one input tensor");

    at::Tensor result = at::empty(output_shape, inputs.front().options());
    at::TensorList input_list(inputs);
    constexpr size_t kBenchmarkRounds = 30;
    for (size_t i = 0; i < kBenchmarkRounds; ++i) {
        EXEC_NPU_CMD(aclnnCat, input_list, dim, result);
    }
    return result;
}



// 修改my_op的输入输出
TORCH_LIBRARY(myops, m) {
		m.def("my_op(Tensor[] inputs, int dim, int[] output_shape) -> Tensor");
}

// 不修改
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
		m.impl("my_op", &my_op_impl_npu);
}

// 不修改
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
		m.def("custom_op", &my_op_impl_npu, "torch.cat");
}

