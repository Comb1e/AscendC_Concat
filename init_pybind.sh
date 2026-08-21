#!/bin/bash

# pybind依赖安装脚本，基于notebook 910B云环境，python3.9，cann8.0.0.alpha003
# 910B云环境可直接执行脚本，基础赛道根据自身的设备参考安装。

pip3 install pip==21.0.1
pip3 install pybind11==2.13.1 numpy==1.24 expecttest
pip3 install tensorflow==2.18.0

# 下载torch 2.5.1
wget https://download.pytorch.org/whl/cpu/torch-2.5.1-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip3 install torch-2.5.1-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
# 下载torch_npu 2.5.1
wget https://gitcode.com/Ascend/pytorch/releases/download/v7.1.0-pytorch2.5.1/torch_npu-2.5.1.post1-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip3 install torch_npu-2.5.1.post1-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip3 install numpy==1.24.0

python3 -m pip install protobuf==3.20.0 -i https://mirrors.huaweicloud.com/repository/pypi/simple
