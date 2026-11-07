# Dynamic load balancing

Compares fixed subdomains (`GridPolicy::Balanced`) with moving subdomain boundaries (`GridPolicy::Dynamic`) for a clustered particle distribution.
This is the dynamic load-balancing test of the dsDecompX paper. The CPU and the GPU use the same program.

70% of the particles form a Gaussian cluster that moves slowly along the diagonal of the box, and the other 30% are uniformly distributed.
Every particle performs the same amount of work, so the rank that holds the most particles sets the step time.
The program prints a `STEP` line every `print_every` steps and a final `RESULT` line, averaged over the second half of the run.

## Build

dsDecompX must be built with CUDA and installed. The CPU run uses the same build (`gpu=0`).

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=<dsDecompX install> -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build -j
```

## Run

```sh
srun -n 16 ./load_balancing gpu=1 particles_total=64000000 work=100000 dynamic=0    # fixed subdomains
srun -n 16 ./load_balancing gpu=1 particles_total=64000000 work=100000 dynamic=1    # dynamic boundaries
srun -n 144 ./load_balancing gpu=0 particles_total=2880000 work=500 dynamic=1       # CPU
```

`key=value` arguments change the defaults: `gpu`, `particles_total`, `dynamic`, `blob_fraction`, `blob_sigma`, `work`, `halo`, `rebuild_every`,
`interval`, `threshold`, `relaxation`, `steps`, `print_every`. On the GPU use a GPU-aware MPI.
