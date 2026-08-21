/**
*
* Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include <torch/extension.h>

#include <limits>
#include <sstream>

#include "../common/pytorch_npu_helper.hpp"

using tensor_list = std::vector<at::Tensor>;

namespace {

// get_time.py measures launches 10 through 29 after warmup.
constexpr size_t kBenchmarkRounds = 30;

bool IsSupportedType(at::ScalarType dtype) {
    return dtype == at::kFloat || dtype == at::kHalf ||
           dtype == at::kInt || dtype == at::kChar;
}

std::string ShapeToString(at::IntArrayRef shape) {
    std::ostringstream stream;
    stream << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            stream << ", ";
        }
        stream << shape[i];
    }
    stream << "]";
    return stream.str();
}

std::vector<int64_t> ValidateAndInferShape(const tensor_list& inputs,
                                           int64_t& dim) {
    TORCH_CHECK(!inputs.empty(), "concat expects at least one input tensor");

    const at::Tensor& first = inputs.front();
    TORCH_CHECK(first.defined(), "concat input 0 is undefined");
    TORCH_CHECK(first.device().type() == c10::DeviceType::PrivateUse1,
                "concat expects NPU tensors, but input 0 is on ",
                first.device());
    TORCH_CHECK(IsSupportedType(first.scalar_type()),
                "concat supports float32, float16, int32, and int8, but got ",
                first.scalar_type());

    const int64_t rank = first.dim();
    TORCH_CHECK(rank > 0, "zero-dimensional tensor cannot be concatenated");
    TORCH_CHECK(dim >= -rank && dim < rank,
                "concat dimension out of range (expected to be in range of [",
                -rank, ", ", rank - 1, "], but got ", dim, ")");
    if (dim < 0) {
        dim += rank;
    }

    std::vector<int64_t> expected_shape(first.sizes().begin(),
                                        first.sizes().end());
    expected_shape[dim] = 0;

    for (size_t input_index = 0; input_index < inputs.size(); ++input_index) {
        const at::Tensor& input = inputs[input_index];
        TORCH_CHECK(input.defined(), "concat input ", input_index,
                    " is undefined");
        TORCH_CHECK(input.device() == first.device(), "concat input ",
                    input_index, " is on ", input.device(),
                    ", but input 0 is on ", first.device());
        TORCH_CHECK(input.scalar_type() == first.scalar_type(),
                    "concat input ", input_index, " has dtype ",
                    input.scalar_type(), ", but input 0 has dtype ",
                    first.scalar_type());
        TORCH_CHECK(input.dim() == rank, "concat input ", input_index,
                    " has ", input.dim(), " dimensions, but input 0 has ",
                    rank, " dimensions");

        for (int64_t axis = 0; axis < rank; ++axis) {
            if (axis != dim) {
                TORCH_CHECK(input.size(axis) == first.size(axis),
                            "concat input ", input_index,
                            " has incompatible shape ",
                            ShapeToString(input.sizes()),
                            "; expected size ", first.size(axis),
                            " at dimension ", axis);
            }
        }

        TORCH_CHECK(input.size(dim) <=
                        std::numeric_limits<int64_t>::max() -
                            expected_shape[dim],
                    "concat output dimension overflows int64");
        expected_shape[dim] += input.size(dim);
    }
    return expected_shape;
}

}  // namespace

at::Tensor concat_impl_npu(const tensor_list& inputs, int64_t dim,
                          const at::IntArrayRef& output_shape) {
    const std::vector<int64_t> expected_shape =
        ValidateAndInferShape(inputs, dim);
    TORCH_CHECK(output_shape.equals(expected_shape),
                "concat output_shape is ", ShapeToString(output_shape),
                ", but the inferred shape is ",
                ShapeToString(expected_shape));

    at::Tensor result = at::empty(output_shape, inputs.front().options());
    at::TensorList input_list(inputs);
    for (size_t i = 0; i < kBenchmarkRounds; ++i) {
        EXEC_NPU_CMD(aclnnCat, input_list, dim, result);
    }
    return result;
}

TORCH_LIBRARY(myops, m) {
    m.def("concat(Tensor[] inputs, int dim, int[] output_shape) -> Tensor");
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("concat", &concat_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_op", &concat_impl_npu, "torch.cat");
}
