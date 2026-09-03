# MPI backend

MPI is the reference backend and supports host and CUDA views. It duplicates the
input communicator, creates a Cartesian communicator, exchanges counts, and
uses nonblocking receives/sends followed by `MPI_Waitall`. Device payloads are
passed directly when CUDA-aware MPI is detected. Otherwise persistent pinned
staging is available only when `advanced.allow_host_staging` is true.
If a vendor MPI cannot advertise its capability, set
`DSDECOMP_CUDA_AWARE_MPI=1` only after verifying device-pointer support; set it
to `0` to force staging.

Overlap depends on MPI progress support. Select `mpi` explicitly when comparing
logical results with NCCL/NVSHMEM.
