# CUDA execution

Set `Config::execution=Cuda`, pass `CudaDevice` or `CudaManaged` particle views,
and always supply a communication stream. Halo performs boundary classification,
scan, stable pack, data-plane exchange, then optional unpack. Migration performs
owner classification, stable destination layout, exchange, and compaction.

Buffers grow geometrically and are reused. There is no device-wide synchronize.
A small count copy is synchronized to the host because variable-size sparse
transfers must be posted with known sizes. Payloads remain device-resident unless
the explicit MPI staging fallback is active.
