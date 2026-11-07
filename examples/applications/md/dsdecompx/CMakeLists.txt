cmake_minimum_required(VERSION 3.18)
project(md_dsdecompx LANGUAGES CXX CUDA)

# dsDecompX built with CUDA and installed. The CPU run uses the same build (gpu=0).
find_package(dsDecompX 1.0 REQUIRED CONFIG)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)   # pass -DCMAKE_CUDA_ARCHITECTURES=90 for GH200
endif()

set(MAIN_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/../main.cpp)
add_executable(md_dsdecompx ${MAIN_SOURCE})
set_source_files_properties(${MAIN_SOURCE} PROPERTIES LANGUAGE CUDA)
target_compile_definitions(md_dsdecompx PRIVATE USE_DSDECOMPX)
target_link_libraries(md_dsdecompx PRIVATE dsdecompx::dsdecompx MPI::MPI_CXX CUDA::cudart)
