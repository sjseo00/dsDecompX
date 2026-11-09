# FindNVSHMEM.cmake -- locate NVIDIA NVSHMEM and define NVSHMEM::NVSHMEM.
#
# Search order (first hit wins):
#   -DNVSHMEM_ROOT=<prefix> or -DNVSHMEM_INCLUDE_DIR=... -DNVSHMEM_HOST_LIBRARY=...
#   env NVSHMEM_ROOT / NVSHMEM_HOME / NVSHMEM_DIR / NVSHMEM_PATH
#   NVIDIA HPC SDK:  $NVHPC_ROOT/comm_libs/nvshmem
#   CUDA toolkit:    -DCUDAToolkit_ROOT, env CUDA_HOME / CUDA_PATH
#   system paths (distro packages: include/nvshmem_12, lib/nvshmem/12)
# dsDecompX calls host/on-stream APIs from plain C++ and has no device-side
# NVSHMEM code. Both libnvshmem_host.so and a runtime-only libnvshmem_host.so.3
# are accepted. libnvshmem_device.a is linked too when it exists: the inline
# nvshmemx_init_attr()/nvshmem_finalize() in the headers call nvshmemi_init_thread()
# and nvshmemi_finalize(), which newer NVSHMEM exports only from that archive.

set(_dsdx_nvshmem_hints "${NVSHMEM_ROOT}")
foreach(_v NVSHMEM_ROOT NVSHMEM_HOME NVSHMEM_DIR NVSHMEM_PATH CUDA_HOME CUDA_PATH CUDAToolkit_ROOT)
    if(DEFINED ENV{${_v}} AND NOT "$ENV{${_v}}" STREQUAL "")
        list(APPEND _dsdx_nvshmem_hints "$ENV{${_v}}")
    endif()
endforeach()
if(DEFINED ENV{NVHPC_ROOT} AND NOT "$ENV{NVHPC_ROOT}" STREQUAL "")
    list(APPEND _dsdx_nvshmem_hints "$ENV{NVHPC_ROOT}/comm_libs/nvshmem")
endif()
if(CUDAToolkit_ROOT)
    list(APPEND _dsdx_nvshmem_hints "${CUDAToolkit_ROOT}")
endif()

# Location recorded by the dsDecompX package config (see dsDecompXConfig.cmake.in).
if(NOT NVSHMEM_INCLUDE_DIR AND DSDECOMPX_NVSHMEM_INCLUDE_HINT
   AND EXISTS "${DSDECOMPX_NVSHMEM_INCLUDE_HINT}/nvshmem.h")
    set(NVSHMEM_INCLUDE_DIR "${DSDECOMPX_NVSHMEM_INCLUDE_HINT}" CACHE PATH
        "NVSHMEM include directory")
endif()
if(NOT NVSHMEM_HOST_LIBRARY AND DSDECOMPX_NVSHMEM_LIBRARY_HINT
   AND EXISTS "${DSDECOMPX_NVSHMEM_LIBRARY_HINT}")
    set(NVSHMEM_HOST_LIBRARY "${DSDECOMPX_NVSHMEM_LIBRARY_HINT}" CACHE FILEPATH
        "NVSHMEM host library")
endif()

find_path(NVSHMEM_INCLUDE_DIR
    NAMES nvshmem.h
    HINTS ${_dsdx_nvshmem_hints}
    PATH_SUFFIXES include include/nvshmem include/nvshmem_12 include/nvshmem_13)
if(NVSHMEM_INCLUDE_DIR AND NOT EXISTS "${NVSHMEM_INCLUDE_DIR}/nvshmemx.h")
    set(NVSHMEM_INCLUDE_DIR "NVSHMEM_INCLUDE_DIR-NOTFOUND")
endif()
find_library(NVSHMEM_HOST_LIBRARY
    NAMES nvshmem_host libnvshmem_host.so.3
    HINTS ${_dsdx_nvshmem_hints}
    PATH_SUFFIXES lib lib64 lib/nvshmem/12 lib/nvshmem/13
                  lib/x86_64-linux-gnu lib/x86_64-linux-gnu/nvshmem/12
                  lib/aarch64-linux-gnu lib/aarch64-linux-gnu/nvshmem/12)

get_filename_component(_dsdx_nvshmem_libdir "${NVSHMEM_HOST_LIBRARY}" DIRECTORY)
find_library(NVSHMEM_DEVICE_LIBRARY
    NAMES libnvshmem_device.a
    HINTS "${_dsdx_nvshmem_libdir}" ${_dsdx_nvshmem_hints}
    PATH_SUFFIXES lib lib64 lib/nvshmem/12 lib/nvshmem/13)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVSHMEM
    REQUIRED_VARS NVSHMEM_INCLUDE_DIR NVSHMEM_HOST_LIBRARY
    FAIL_MESSAGE
        "NVSHMEM not found. Point CMake at it with -DNVSHMEM_ROOT=<prefix containing include/nvshmem.h, include/nvshmemx.h and lib/libnvshmem_host.so> (or env NVSHMEM_ROOT), or build without it: -DENABLE_NVSHMEM=OFF -DAUTO_DETECT_GPU_BACKENDS=OFF.")

if(NVSHMEM_FOUND AND NOT TARGET NVSHMEM::NVSHMEM)
    add_library(NVSHMEM::NVSHMEM UNKNOWN IMPORTED)
    set_target_properties(NVSHMEM::NVSHMEM PROPERTIES
        IMPORTED_LOCATION "${NVSHMEM_HOST_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${NVSHMEM_INCLUDE_DIR}")
    if(NVSHMEM_DEVICE_LIBRARY)
        set_target_properties(NVSHMEM::NVSHMEM PROPERTIES
            INTERFACE_LINK_LIBRARIES "${NVSHMEM_DEVICE_LIBRARY}")
    endif()
endif()

mark_as_advanced(NVSHMEM_INCLUDE_DIR NVSHMEM_HOST_LIBRARY NVSHMEM_DEVICE_LIBRARY)
