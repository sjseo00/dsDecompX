# Contributing

Build with warnings and tests enabled:

```sh
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Changes to topology, packing order, tags, migration ordering, or synchronization
must include an MPI regression test and, where applicable, run the CUDA smoke
case once per compiled backend. Keep application-specific adapters outside
this repository subtree. Do not add DeepScaleMD headers, types, namespaces,
compile definitions, or global variables to dsDecomp.

Performance changes must report the MPI implementation, compiler, rank count,
node/GPU mapping, problem size, and whether counters were enabled.
