cmake_minimum_required(VERSION 3.18)
project(dem_amrex LANGUAGES C CXX)       # the AMReX configuration also looks for the C component of MPI

# AMReX_GPU_BACKEND=CUDA gives the GPU program, NONE gives the CPU program.
set(CMAKE_CXX_STANDARD 17)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(AMReX REQUIRED)

set(MAIN_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/../main.cpp)
add_executable(dem_amrex ${MAIN_SOURCE})
target_compile_definitions(dem_amrex PRIVATE USE_AMREX)
target_link_libraries(dem_amrex PRIVATE AMReX::amrex MPI::MPI_CXX)
if(AMReX_GPU_BACKEND STREQUAL "CUDA")
    # Marking the .cpp file as CUDA source makes nvcc compile it. The CUDA code of libamrex needs device linking.
    enable_language(CUDA)
    set_source_files_properties(${MAIN_SOURCE} PROPERTIES LANGUAGE CUDA)
    set_target_properties(dem_amrex PROPERTIES LINKER_LANGUAGE CUDA CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
