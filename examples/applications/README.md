# Application examples

Three small applications that run the same problem with dsDecompX, Cabana, and AMReX. Each folder is complete and independent. It contains
one program for each library with the same physics, neighbor list, and force; only the particle storage and the communication differ.

| Folder | Application | Communication in a step |
|---|---|---|
| [`md`](md/) | Lennard-Jones molecular dynamics | halo refresh of the positions |
| [`dem`](dem/) | Discrete element method (soft spheres with damping) | halo refresh of the whole particle (the damping needs the velocity of the neighbors) |
| [`sph`](sph/) | Smoothed particle hydrodynamics (weakly compressible) | halo refresh of the positions, then a second transfer of the whole particle after the density |

The programs print one `RESULT` line with the step time, the time of each section, and the energies. See the `README.md` of each folder for building and running.
These are simplified models that carry the communication pattern of the applications. They are not production codes.
