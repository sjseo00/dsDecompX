cmake_minimum_required(VERSION 3.18)
project(sph_cabana LANGUAGES CXX)

# Kokkos built with CUDA gives the GPU program, Kokkos built with OpenMP gives the CPU program.
# For the GPU use nvcc_wrapper of the Kokkos source as the C++ compiler (-DCMAKE_CXX_COMPILER=.../nvcc_wrapper).
set(CMAKE_CXX_STANDARD 17)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(Cabana REQUIRED)

add_executable(sph_cabana ${CMAKE_CURRENT_SOURCE_DIR}/../main.cpp)
target_compile_definitions(sph_cabana PRIVATE USE_CABANA)
target_link_libraries(sph_cabana PRIVATE Cabana::cabanacore MPI::MPI_CXX)
