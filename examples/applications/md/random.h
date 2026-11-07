// random.h -- hash-based random numbers.
// The same input gives the same value on every rank, on the CPU and on the GPU.
#pragma once

#ifdef __CUDACC__
#define CPU_AND_GPU __host__ __device__
#else
#define CPU_AND_GPU
#endif

// Random number in [0, 1) from a 64-bit key.
CPU_AND_GPU inline double random01(unsigned long long n)
{
    n ^= n >> 33;
    n *= 0xff51afd7ed558ccdULL;
    n ^= n >> 33;
    n *= 0xc4ceb9fe1a85ec53ULL;
    n ^= n >> 33;
    return (n >> 11) * (1.0 / 9007199254740992.0);
}
