# NVSHMEM synchronization

`putmem_on_stream` initiates one-sided payload writes. `quiet_on_stream` orders
the following notification after remote completion of prior puts. A remote
signal publishes the operation epoch. The receiver waits for each expected
local signal on the communication stream before exposing its buffer. Stream
order then makes payload bytes visible to unpack/consumer work.

Local completion alone is insufficient for reuse; remote completion and the
signal establish the protocol. Epochs prevent an old notification satisfying a
new operation. No global barrier appears in the timestep path.
