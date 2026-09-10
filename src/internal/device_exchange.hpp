#pragma once

#include "decomposition.hpp"

#include <cstddef>
#include <vector>

namespace dsdecompx::detail {

struct DeviceTransfer {
    int peer_rank = -1;
    const void* send_buffer = nullptr;
    void* receive_buffer = nullptr;
    // NVSHMEM uses a symmetric address in the remote PE. Other backends ignore it.
    void* remote_receive_buffer = nullptr;
    std::size_t send_bytes = 0;
    std::size_t receive_bytes = 0;
    int send_tag = 0;
    int receive_tag = 0;
    std::uint64_t* local_signal = nullptr;
    std::uint64_t* remote_signal = nullptr;
    std::uint64_t signal_value = 0;
};

struct DeviceExchange {
    const std::vector<DeviceTransfer>* transfers = nullptr;
    ExecutionStream communication_stream = nullptr;
    ExecutionStream consumer_stream = nullptr;
    const char* operation = nullptr;
};

} // namespace dsdecompx::detail
