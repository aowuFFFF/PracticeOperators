#!/bin/bash


######################     构建trust    ########################

# # 克隆 Thrust 和 CUB
# git clone --recursive https://github.com/NVIDIA/thrust.git

# # 进入目录
# cd thrust 

# # # 创建构建目录
# # rm -rf build
# mkdir build
# cd build 

# # # 配置构建
# cmake -DTHRUST_INCLUDE_CUB_CMAKE=ON ..

# # # 编译项目（这里使用16个并行任务）
# make -j16

# # # 运行测试和示例
# ctest


# 99% tests passed, 2 tests failed out of 377

# Total Test time (real) = 5062.73 sec

# The following tests FAILED:
#         313 - cub.cpp14.test.device_radix_sort.cdp_0.bytes_1.pairs_1 (Subprocess killed)
#         317 - cub.cpp14.test.device_radix_sort.cdp_0.bytes_2.pairs_1 (Subprocess killed)
# Errors while running CTest
# Output from these tests are in: /home/ubuntu/yujie/PracticeOperators/thrust/build/Testing/Temporary/LastTest.log
# Use "--rerun-failed --output-on-failure" to re-run the failed cases verbosely.


######################     构建 cub    ########################
cd /home/ubuntu/yujie/PracticeOperators/thrust/dependencies/cub
mkdir build
cd build
cmake ..
make -j16

