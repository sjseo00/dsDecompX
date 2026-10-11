# FindNCCL.cmake -- locate NVIDIA NCCL and define NCCL::NCCL.
#
# Search order (first hit wins):
#   -DNCCL_ROOT=<prefix> or -DNCCL_INCLUDE_DIR=... -DNCCL_LIBRARY=...
#   env NCCL_ROOT / NCCL_HOME / NCCL_DIR / NCCL_PATH
#   NVIDIA HPC SDK:  $NVHPC_ROOT/comm_libs/nccl
#   CUDA toolkit:    -DCUDAToolkit_ROOT, env CUDA_HOME / CUDA_PATH
#   system paths (CMAKE_PREFIX_PATH, /usr, /usr/local)
# Both libnccl.so and a runtime-only libnccl.so.2 are accepted.

set(_dsdx_nccl_hints "${NCCL_ROOT}")
foreach(_v NCCL_ROOT NCCL_HOME NCCL_DIR NCCL_PATH CUDA_HOME CUDA_PATH CUDAToolkit_ROOT)
    if(DEFINED ENV{${_v}} AND NOT "$ENV{${_v}}" STREQUAL "")
        list(APPEND _dsdx_nccl_hints "$ENV{${_v}}")
    endif()
endforeach()
if(DEFINED ENV{NVHPC_ROOT} AND NOT "$ENV{NVHPC_ROOT}" STREQUAL "")
    list(APPEND _dsdx_nccl_hints "$ENV{NVHPC_ROOT}/comm_libs/nccl")
endif()
if(CUDAToolkit_ROOT)
    list(APPEND _dsdx_nccl_hints "${CUDAToolkit_ROOT}")
endif()

# Location recorded by the dsDecompX package config (see dsDecompXConfig.cmake.in).
if(NOT NCCL_INCLUDE_DIR AND DSDECOMPX_NCCL_INCLUDE_HINT
   AND EXISTS "${DSDECOMPX_NCCL_INCLUDE_HINT}/nccl.h")
    set(NCCL_INCLUDE_DIR "${DSDECOMPX_NCCL_INCLUDE_HINT}" CACHE PATH "NCCL include directory")
endif()
if(NOT NCCL_LIBRARY AND DSDECOMPX_NCCL_LIBRARY_HINT
   AND EXISTS "${DSDECOMPX_NCCL_LIBRARY_HINT}")
    set(NCCL_LIBRARY "${DSDECOMPX_NCCL_LIBRARY_HINT}" CACHE FILEPATH "NCCL library")
endif()

find_path(NCCL_INCLUDE_DIR
    NAMES nccl.h
    HINTS ${_dsdx_nccl_hints}
    PATH_SUFFIXES include)
find_library(NCCL_LIBRARY
    NAMES nccl libnccl.so.2
    HINTS ${_dsdx_nccl_hints}
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL
    REQUIRED_VARS NCCL_INCLUDE_DIR NCCL_LIBRARY
    FAIL_MESSAGE
        "NCCL not found. Point CMake at it with -DNCCL_ROOT=<prefix containing include/nccl.h and lib/libnccl.so> (or env NCCL_ROOT), or build without it: -DENABLE_NCCL=OFF -DAUTO_DETECT_GPU_BACKENDS=OFF.")

if(NCCL_FOUND AND NOT TARGET NCCL::NCCL)
    add_library(NCCL::NCCL UNKNOWN IMPORTED)
    set_target_properties(NCCL::NCCL PROPERTIES
        IMPORTED_LOCATION "${NCCL_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}")
endif()

mark_as_advanced(NCCL_INCLUDE_DIR NCCL_LIBRARY)
