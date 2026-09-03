# MD integration guide

The application initializes MPI, creates one `Decomp`, then creates persistent
halo and migration plans. It exposes owned particle arrays through non-owning
`ConstParticleView` objects. dsDecomp owns only packed communication/results.

After integration, call migration if particles may have crossed local bounds;
replace application-owned storage with the complete migration result. Invalidate
the halo plan. Before force calculation, begin halo exchange, calculate interior
forces, wait, then append/unpack ghost records and calculate boundary forces.
Ghost slot management and neighbor-list index updates stay in the application.

DeepScaleMD currently uses the host compatibility adapter in `md_system.cpp` to
preserve its established numerical/storage behavior. The adapter constructs
`Decomp`, copies topology bounds into application fields, presents host views,
and copies result records into its SoA. Backend discovery and communication are
not performed in DeepScaleMD.

For a GPU-resident application, set execution to CUDA and supply an explicit
communication stream. Do not reuse application input/output or library result
buffers before operation completion. Establish a compute-stream dependency only
before boundary work consumes unpacked ghosts.
