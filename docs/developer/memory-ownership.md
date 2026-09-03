# Memory ownership

Application SoA pointers are borrowed and never freed by dsDecomp. Host result
vectors and device result buffers are library-owned and remain valid until the
same plan storage is reused. CUDA `DeviceBuffer`/`PinnedBuffer` distinguish
logical size from capacity and grow geometrically. Outstanding operations hold
exclusive use of a `Decomp` object. NVSHMEM allocations are separately tracked
because allocation/free are symmetric collective operations.
