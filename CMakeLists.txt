cmake_minimum_required(VERSION 3.2)
project(as3_bench)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)



if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message("release mode include cuda")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin=/opt/rh/devtoolset-10/root/usr/bin/gcc -gencode=arch=compute_75,code=[sm_75,compute_75]")
    add_executable(cuda src/cuda.cu)
    set_target_properties(cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(
       cuda 
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES OFF
        )
endif()