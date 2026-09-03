# Performance

Enable counters with `set_performance_counters_enabled(true, detail)`. `get_stats`
is rank-local and noncollective; `get_global_stats` performs explicit MPI
reductions and must be called by every rank. Counters retain count/sum/min/max for
timings plus calls, bytes, reallocations, capacity, and optional neighbor bytes.

GPU event timing and CPU wall time are different domains. Instrumentation never
adds a device-wide synchronize; the required count-metadata wait is part of the
algorithm. Version 0.3 aggregate phase timers are host/control-plane wall time;
they are not presented as kernel time. Measure actual overlap with Nsight Systems and compare identical
initial states against the MPI backend.
