cmake_minimum_required (VERSION 3.10)

project(DynamicGPUHashTableIEE CXX CUDA)

set(default_build_type "Debug")

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

message("BUILD TYPE: ${CMAKE_BUILD_TYPE}")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    string(APPEND CMAKE_CUDA_FLAGS " -G")
    message("CUDA FLAGS: ${CMAKE_CUDA_FLAGS}")
endif(CMAKE_BUILD_TYPE STREQUAL "Debug")

if(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    string(APPEND CMAKE_CUDA_FLAGS " -G")
    message("CUDA FLAGS: ${CMAKE_CUDA_FLAGS}")
endif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")

if(NOT DEFINED CUDA_ARCH)
    message("WARNING CUDA_ARCH IS BEING SET TO DEFAULT")
    set(CUDA_ARCH 75)
endif(NOT DEFINED CUDA_ARCH)

string(APPEND CMAKE_CUDA_FLAGS " -Xptxas=-v -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} -Wreorder")

add_compile_options("$<$<CONFIG:DEBUG>:-DDEBUG>")

ENABLE_TESTING()
ADD_SUBDIRECTORY(tests)


add_library(Operations Operations.cu)