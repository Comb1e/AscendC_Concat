import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops_lib
torch.npu.config.allow_internal_format = False
import numpy as np
import sys  
import random

case_data = {
    'case1': {
        'input': np.random.uniform(-500, 500, [128,256]).astype(np.float16),
        'dim': -1,
        'max_step': 64
    }
}

def verify_result(real_result, golden):
    # 根据数据类型设置误差阈值（与原逻辑一致）
    if golden.dtype == torch.float32:
        rtol = 1e-4  # fp32相对误差阈值
        atol = 1e-4  # fp32绝对误差阈值
    else:
        rtol = 1e-3  # fp16相对误差阈值
        atol = 1e-3  # fp16绝对误差阈值

    minimum = 10e-10
    golden = torch.where(golden == 0, minimum, golden)
    real_result = torch.where(real_result == 0, minimum, real_result)

    abs_diff = torch.abs(real_result - golden)
    rel_diff = abs_diff / torch.max(torch.abs(real_result), torch.abs(golden))
    is_close = (abs_diff <= atol) | (rel_diff <= rtol)
    # 补充NaN判断（与equal_nan=True保持一致）
    both_nan = torch.isnan(real_result) & torch.isnan(golden)
    is_close = is_close | both_nan
    err_num = torch.sum(~is_close).item()  # 取反统计不满足的数量

    # 与原逻辑相同的误差数量判断
    if real_result.numel() * rtol < err_num:
        print(f"[ERROR] result error")
        return False
    print("test pass")
    return True

def generate_random_split_sizes(total_len: int, max_step: int) -> list[int]:
    """
    随机生成分割长度列表 split_sizes
    约束：
        1. sum(split_sizes) == total_len
        2. 每个元素 ∈ [0, max_step]
        3. 自动生成任意份数分片，随机切分
    """
    if total_len < 0:
        raise ValueError("total_len 不能为负数")
    if max_step < 0:
        raise ValueError("max_step 不能为负数")
    random.seed(111)
    remain = total_len
    split_sizes = []

    while True:
        # 本轮可取上限：剩余长度和max_step取更小值
        upper = min(remain, max_step)
        pick = random.randint(0, upper)
        split_sizes.append(pick)
        remain -= pick

        if remain == 0:
            break

    return split_sizes
    
def unpack_tensor_by_dim(
    tensor: torch.Tensor,
    split_sizes: list[int],
    dim: int = 0
) -> list[torch.Tensor]:
    """
    在指定维度dim拆分张量为多份子张量，支持分片长度=0
    :param tensor: 输入原始张量
    :param split_sizes: 各分片在dim上的长度列表，允许元素=0
                        sum(split_sizes) 必须等于 tensor.size(dim)
    :param dim: 要拆分的维度，支持负数索引
    :return: 子张量list，顺序和split_sizes一一对应，size=0的张量正常保留
    """
    dim = dim % tensor.dim()  # 负维度转正
    # torch.split原生支持size=0
    chunks = torch.split(tensor, split_sizes, dim=dim)
    return list(chunks)
    
class TestCustomOP(TestCase):
    def test_custom_op_case(self,num):
        print(num)
        caseName='case'+str(num) 

        input_x = torch.from_numpy(case_data[caseName]["input"])
        print(input_x.shape)
        dim = case_data[caseName]["dim"]
        step = case_data[caseName]["max_step"]
        total_len = input_x.shape[dim]

        split_sizes = generate_random_split_sizes(total_len, step)
        inputs = list(torch.split(input_x, split_sizes, dim=dim))
        inputs_npu = [ele.npu() for ele in inputs]
        print(split_sizes)
        print([ele.shape for ele in inputs])
        # 修改输入
        output_npu = custom_ops_lib.custom_op(inputs_npu, dim, input_x.shape)
        if output_npu is None:
            print(f"{caseName} execution timed out!")
        else:
            output_cpu = output_npu.cpu()

            if verify_result(output_cpu, input_x):
                print(f"{caseName} verify result pass!")
            else:
                print(f"{caseName} verify result failed!")

if __name__ == "__main__":
    TestCustomOP().test_custom_op_case(sys.argv[1])
    
