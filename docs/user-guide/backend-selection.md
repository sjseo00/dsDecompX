# Backend selection

Normal applications select only `Config::execution`. A CUDA-enabled build
defaults to `Cuda + Auto`; a CPU-only build defaults to `Host + Auto`. CUDA
tries NVSHMEM, NCCL, then MPI, while Host uses MPI.

Expert diagnostics and benchmarks may force a backend with
`Config::advanced.selection` or `BACKEND=auto|mpi|nccl|nvshmem`.
Explicit selection never falls back. Set
`Config::advanced.read_environment=false` to ignore the environment.
`backend_compiled()` reports build availability, while successful construction
confirms runtime availability. NCCL/NVSHMEM require CUDA particle memory.
