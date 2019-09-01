cmake_minimum_required (VERSION 3.10)

project(DynamicGPUHashTableIEETests CXX CUDA)

add_executable(TestSlabList TestSlabList.cu)

add_test(TestSlabList TestSlabList)