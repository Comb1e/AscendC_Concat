import os

import torch_npu
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch_npu.utils.cpp_extension import NpuExtension


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTORCH_NPU_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))

extension = NpuExtension(
    name="concat_profile_lib",
    sources=[os.path.join(SCRIPT_DIR, "extension", "custom_op.cpp")],
    extra_compile_args=[
        "-I" + os.path.join(PYTORCH_NPU_PATH, "include", "third_party", "acl", "inc"),
    ],
)

setup(
    name="concat_profile",
    version="1.0",
    ext_modules=[extension],
    cmdclass={"build_ext": BuildExtension},
)
