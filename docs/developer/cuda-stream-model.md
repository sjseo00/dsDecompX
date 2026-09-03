# CUDA stream model

The caller supplies the communication stream. Pack kernels precede payload
operations on that stream. Backend completion precedes unpack; when a separate
consumer stream is specified, a recorded event establishes the dependency.
The application may run interior work independently and must wait only before
boundary work consumes ghosts. Buffer-reuse events prevent a later operation
from overwriting storage still consumed by an earlier stream.
