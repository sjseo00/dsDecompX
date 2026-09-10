cmake_minimum_required(VERSION 3.18)

project(dsDecompX VERSION 0.3.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(BUILD_TESTS "Build dsDecompX tests" OFF)
option(BUILD_EXAMPLES "Build dsDecompX examples" OFF)
option(BUILD_BENCHMARKS "Build the dsDecompX benchmark" OFF)
option(RUN_GPU_TESTS "Register GPU/MPI tests with CTest" OFF)
option(ENABLE_PERF_COUNTERS "Compile performance counters" ON)
option(ENABLE_CUDA "Build the CUDA execution layer" OFF)
option(ENABLE_NCCL "Build the NCCL data-plane backend" OFF)
option(ENABLE_NVSHMEM "Build the NVSHMEM data-plane backend" OFF)
option(AUTO_DETECT_GPU_BACKENDS
    "Automatically include installed NCCL/NVSHMEM backends in CUDA builds" ON)

if(ENABLE_NCCL OR ENABLE_NVSHMEM)
    set(ENABLE_CUDA ON CACHE BOOL
        "Build the CUDA execution layer" FORCE)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)

if(ENABLE_CUDA AND DEFINED CMAKE_CUDA_HOST_COMPILER)
    message(STATUS "dsDecompX: CUDA host compiler ${CMAKE_CUDA_HOST_COMPILER}")
endif()

# Enable CUDA before creating the target so CMake initializes the target's
# CUDA_ARCHITECTURES property. Enabling it only after add_library() leaves the
# property empty with CMake 3.18-3.22 and causes generation to fail.
if(ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(AUTO_DETECT_GPU_BACKENDS)
        if(NOT ENABLE_NVSHMEM)
            find_package(NVSHMEM QUIET)
            if(NVSHMEM_FOUND)
                set(ENABLE_NVSHMEM ON)
                message(STATUS "dsDecompX: automatically enabled NVSHMEM backend")
            endif()
        endif()
        if(NOT ENABLE_NCCL)
            find_package(NCCL QUIET)
            if(NCCL_FOUND)
                set(ENABLE_NCCL ON)
                message(STATUS "dsDecompX: automatically enabled NCCL backend")
            endif()
        endif()
    endif()
endif()

add_library(dsdecompx
    src/backend_selection.cpp
    src/decomp.cpp
    src/runtime.cpp
    src/topology.cpp
    src/grid.cpp
    src/halo.cpp
    src/migration.cpp
    src/buffer.cpp
    src/backends/mpi_backend.cpp)

add_library(dsdecompx::dsdecompx ALIAS dsdecompx)

target_compile_features(dsdecompx PUBLIC cxx_std_17)
target_compile_definitions(dsdecompx PUBLIC
    ENABLE_PERF_COUNTERS=$<BOOL:${ENABLE_PERF_COUNTERS}>
    HAS_CUDA=$<BOOL:${ENABLE_CUDA}>
    HAS_NCCL=$<BOOL:${ENABLE_NCCL}>
    HAS_NVSHMEM=$<BOOL:${ENABLE_NVSHMEM}>)

target_include_directories(dsdecompx
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(dsdecompx PUBLIC MPI::MPI_CXX)

if(ENABLE_CUDA)
    target_sources(dsdecompx PRIVATE
        src/cuda/cuda_buffer.cpp
        src/cuda/cuda_event.cpp
        src/cuda/execution.cpp
        src/cuda/pack.cu
        src/cuda/unpack.cu
        src/cuda/classify.cu
        src/cuda/migration_layout.cu
        src/cuda/compact.cu
        src/cuda/prefix_scan.cu)
    target_link_libraries(dsdecompx PUBLIC CUDA::cudart)
    target_compile_options(dsdecompx PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    set_target_properties(dsdecompx PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
        # Produce a self-contained installed static library. Consumers should
        # not need to device-link dsDecompX objects together with their own CUDA
        # sources (or expose host-only MPI/OpenMP dependencies to nvlink).
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

if(ENABLE_NCCL)
    find_package(NCCL REQUIRED)
    target_sources(dsdecompx PRIVATE src/backends/nccl/nccl_backend.cpp)
    target_link_libraries(dsdecompx PUBLIC NCCL::NCCL)
endif()

if(ENABLE_NVSHMEM)
    find_package(NVSHMEM REQUIRED)
    target_sources(dsdecompx PRIVATE
        src/backends/nvshmem/nvshmem_backend.cpp)
    target_link_libraries(dsdecompx PUBLIC NVSHMEM::NVSHMEM)
endif()

set_target_properties(dsdecompx PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME dsdecompx
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    EXPORT_NAME dsdecompx)

if(BUILD_TESTS)
    include(CTest)
    add_subdirectory(tests)
endif()

if(BUILD_EXAMPLES OR BUILD_BENCHMARKS)
    add_subdirectory(examples)
endif()

install(TARGETS dsdecompx
    EXPORT dsDecompXTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(FILES
    include/dsdecompx.hpp
    include/config.hpp
    include/domain.hpp
    include/field.hpp
    include/grid.hpp
    include/perf.hpp
    include/performance.hpp
    include/topology.hpp
    include/backend_selection.hpp
    include/version.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(FILES LICENSE README.md README.ko.md CHANGELOG.md CONTRIBUTING.md VERSION
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/dsDecompX)
install(DIRECTORY docs/
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/dsDecompX/docs)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/dsDecompXConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

configure_package_config_file(
    cmake/dsDecompXConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/dsDecompXConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dsDecompX)

install(FILES cmake/FindNCCL.cmake cmake/FindNVSHMEM.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dsDecompX)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/dsDecompXConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/dsDecompXConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dsDecompX)

install(EXPORT dsDecompXTargets
    FILE dsDecompXTargets.cmake
    NAMESPACE dsdecompx::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dsDecompX)

export(EXPORT dsDecompXTargets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/dsDecompXTargets.cmake
    NAMESPACE dsdecompx::)
