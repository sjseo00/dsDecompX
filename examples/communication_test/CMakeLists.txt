cmake_minimum_required(VERSION 3.18)
project(communication_test LANGUAGES CXX CUDA)

# dsDecompX built with CUDA and installed. The CPU run uses the same build (gpu=0).
find_package(dsDecompX 1.0 REQUIRED CONFIG)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)   # pass -DCMAKE_CUDA_ARCHITECTURES=90 for GH200
endif()

add_executable(communication_test main.cu)
target_link_libraries(communication_test PRIVATE dsdecompx::dsdecompx MPI::MPI_CXX CUDA::cudart)
