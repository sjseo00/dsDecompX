# NCCL design

MPI duplicates the bootstrap communicator, broadcasts the NCCL unique ID, and
maps Cartesian ranks to NCCL ranks once. For each exchange, nonzero receives and
sends are deterministically sorted by peer/tag and submitted in a single NCCL
group on the caller stream. Self edges use device-to-device copies. The group is
enqueue completion, not host payload completion; a CUDA event orders consumers.
