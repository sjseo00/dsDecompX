cmake_minimum_required(VERSION 3.18)
project(dsdecompx_examples LANGUAGES CXX)

# Small examples of dsDecompX and the sanity check of an installation, built against an installed dsDecompX.
#
#   cmake -S examples -B build -DCMAKE_PREFIX_PATH=<dsDecompX install>
#   cmake --build build -j
#   ctest --test-dir build --output-on-failure
#
# The CUDA part of the sanity check needs a CUDA build of dsDecompX and GPUs:
#   -DENABLE_CUDA=ON -DRUN_GPU_TESTS=ON   (add -DENABLE_NCCL=ON or -DENABLE_NVSHMEM=ON to test those transports)
#
# The folders applications/, communication_test/, and load_balancing/ are separate projects with their own CMakeLists.txt.

option(ENABLE_CUDA "Build the CUDA programs of the sanity check (dsDecompX must be built with CUDA)" OFF)
option(RUN_GPU_TESTS "Register the CUDA sanity check with CTest (needs GPUs)" OFF)
option(ENABLE_NCCL "Also check the NCCL transport" OFF)
option(ENABLE_NVSHMEM "Also check the NVSHMEM transport" OFF)

if(ENABLE_CUDA)
    enable_language(CUDA)
endif()

find_package(dsDecompX 1.0 REQUIRED CONFIG)
find_package(MPI REQUIRED COMPONENTS CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()

# The checks use assert(), which a Release build (-DNDEBUG) would remove, so keep assertions on in every build type.
if(MSVC)
    add_compile_options(/UNDEBUG)
else()
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX,CUDA>:-UNDEBUG>")
endif()

foreach(example domain_decomposition halo_exchange particle_migration getting_started registered_field)
    add_executable(dsdecompx_${example} ${example}/main.cpp)
    target_link_libraries(dsdecompx_${example} PRIVATE dsdecompx::dsdecompx MPI::MPI_CXX)
    foreach(ranks 1 2 4)
        add_test(NAME dsdecompx.example.${example}.${ranks}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:dsdecompx_${example}>
                    ${MPIEXEC_POSTFLAGS})
        set_tests_properties(dsdecompx.example.${example}.${ranks} PROPERTIES PROCESSORS ${ranks})
    endforeach()
endforeach()

add_subdirectory(sanity_check)
