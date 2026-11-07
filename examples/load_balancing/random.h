// random.h -- Random numbers from the particle number (id).
// The same id and step give the same value on every rank, on the CPU and on the GPU.
#pragma once

#ifdef __CUDACC__
#define CPU_AND_GPU __host__ __device__
#else
#define CPU_AND_GPU
#endif

// random number in [0, 1)
CPU_AND_GPU inline double random01(unsigned long long n)
{
    n ^= n >> 33;
    n *= 0xff51afd7ed558ccdULL;
    n ^= n >> 33;
    n *= 0xc4ceb9fe1a85ec53ULL;
    n ^= n >> 33;
    return (n >> 11) * (1.0 / 9007199254740992.0);
}

// initial position of a particle in [0, 1); axis = 0, 1, 2 is x, y, z
CPU_AND_GPU inline double random_start(long long id, int axis)
{
    return random01(3ULL * id + axis);
}

// move of a particle in this step in [-1, 1]; axis = 0, 1, 2 is x, y, z
CPU_AND_GPU inline double random_move(long long id, long long step, int axis)
{
    return 2.0 * random01(((unsigned long long)(step + 1) << 40) + 3ULL * id + axis) - 1.0;
}
