# Communication test

Times the operations of dsDecompX without any force evaluation, with a fixed number of particles per rank and an increasing number of ranks.
This is the communication test of the dsDecompX paper. The CPU and the GPU use the same program.

Every step moves the particles randomly, migrates them and rebuilds the halo every `rebuild_every` steps (otherwise it refreshes the halo),
and adds the forces of the ghost particles to their owners with `reverse_halo`. The time is the largest value over the ranks.
The program prints one `RESULT` line with the step time and the time of each operation (all in ms, except `owner_rank_us`).

## Build

dsDecompX must be built with CUDA and installed. The CPU run uses the same build (`gpu=0`).

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=<dsDecompX install> -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build -j
```

## Run

```sh
srun -n 16 ./communication_test gpu=1 particles_per_rank=1000000      # GPU, one rank per GPU
srun -n 576 ./communication_test gpu=0 particles_per_rank=10000       # CPU
```

`key=value` arguments change the defaults: `gpu`, `backend` (0 auto, 1 MPI, 2 NCCL, 3 NVSHMEM), `particles_per_rank`, `particles_total`, `halo`,
`move`, `rebuild_every`, `warmup`, `steps`, `field_components`. On the GPU use a GPU-aware MPI.
