# NCCL backend

NCCL requires MPI, CUDA, and NCCL at build/runtime. MPI broadcasts one NCCL
unique ID and remains the topology/count control plane. One NCCL communicator is
created at initialization. Each payload phase sorts peer operations and batches
sparse `ncclRecv`/`ncclSend` calls inside one group on the communication stream.
No host payload staging or global all-to-all collective is used.

NCCL accepts GPU views only. Initialization or peer errors include backend,
rank, operation, and action context.
