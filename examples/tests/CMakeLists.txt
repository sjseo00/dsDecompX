add_executable(dsdecompx_grid_unit unit/grid_planner.cpp)
target_link_libraries(dsdecompx_grid_unit PRIVATE dsdecompx::dsdecompx)
add_test(NAME dsdecompx.grid.unit COMMAND dsdecompx_grid_unit)

add_executable(dsdecompx_backend_selection_unit unit/backend_selection.cpp)
target_link_libraries(dsdecompx_backend_selection_unit PRIVATE dsdecompx::dsdecompx)
add_test(NAME dsdecompx.backend-selection.unit
    COMMAND dsdecompx_backend_selection_unit)

add_executable(dsdecompx_mpi_regression mpi/regression.cpp)
target_link_libraries(dsdecompx_mpi_regression PRIVATE dsdecompx::dsdecompx)
foreach(ranks 1 2 4)
    add_test(NAME dsdecompx.mpi.${ranks}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:dsdecompx_mpi_regression>
                ${MPIEXEC_POSTFLAGS})
    set_tests_properties(dsdecompx.mpi.${ranks} PROPERTIES PROCESSORS ${ranks})
endforeach()

if(ENABLE_CUDA)
    add_executable(dsdecompx_cuda_backend_smoke cuda/backend_smoke.cu)
    target_link_libraries(dsdecompx_cuda_backend_smoke PRIVATE dsdecompx::dsdecompx)
    set_target_properties(dsdecompx_cuda_backend_smoke PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    add_executable(dsdecompx_cuda_halo_reuse cuda/halo_reuse.cu)
    target_link_libraries(dsdecompx_cuda_halo_reuse PRIVATE dsdecompx::dsdecompx)
    set_target_properties(dsdecompx_cuda_halo_reuse PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    add_executable(dsdecompx_cuda_forward_exchange cuda/forward_exchange.cu)
    target_link_libraries(dsdecompx_cuda_forward_exchange PRIVATE dsdecompx::dsdecompx)
    set_target_properties(dsdecompx_cuda_forward_exchange PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    if(RUN_GPU_TESTS)
        set(_gpu_backends mpi)
        if(ENABLE_NCCL)
            list(APPEND _gpu_backends nccl)
        endif()
        if(ENABLE_NVSHMEM)
            list(APPEND _gpu_backends nvshmem)
        endif()
        foreach(backend IN LISTS _gpu_backends)
            add_test(NAME dsdecompx.cuda.${backend}
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                        ${MPIEXEC_PREFLAGS}
                        $<TARGET_FILE:dsdecompx_cuda_backend_smoke> ${backend}
                        ${MPIEXEC_POSTFLAGS})
            set_tests_properties(dsdecompx.cuda.${backend} PROPERTIES
                PROCESSORS 2 LABELS "cuda;backend-equivalence")
            add_test(NAME dsdecompx.cuda.halo-reuse.${backend}
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                        ${MPIEXEC_PREFLAGS}
                        $<TARGET_FILE:dsdecompx_cuda_halo_reuse> ${backend}
                        ${MPIEXEC_POSTFLAGS})
            set_tests_properties(dsdecompx.cuda.halo-reuse.${backend} PROPERTIES
                PROCESSORS 2 LABELS "cuda;halo-reuse")
            add_test(NAME dsdecompx.cuda.forward-exchange.${backend}
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                        ${MPIEXEC_PREFLAGS}
                        $<TARGET_FILE:dsdecompx_cuda_forward_exchange> ${backend}
                        ${MPIEXEC_POSTFLAGS})
            set_tests_properties(dsdecompx.cuda.forward-exchange.${backend} PROPERTIES
                PROCESSORS 2 LABELS "cuda;forward-exchange")
        endforeach()
    endif()
endif()
