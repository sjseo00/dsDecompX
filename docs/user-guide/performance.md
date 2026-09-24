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

## Dynamic load balancing

Set `config.grid_policy = GridPolicy::Dynamic` when particle density or force
cost is spatially uneven. Automatic balancing uses owned-particle count and is
evaluated every `config.dynamic.interval` migration calls. Increasing the
interval reduces collective/control overhead; increasing
`imbalance_threshold` avoids reacting to small fluctuations; lowering
`relaxation` makes boundary movement more stable but slower to converge.

For heterogeneous force costs, call `decomp.rebalance()` with a measured metric
such as force-kernel seconds or neighbor-pair count. Do not use total step time
when it contains a synchronization wait: fast ranks then appear artificially
busy because they wait for the slow rank. Rebalancing invalidates halo and
neighbor ownership, so applications must rebuild their neighbor list after a
reported geometry change.
