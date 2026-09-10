#pragma once

#include <cuda_runtime_api.h>

namespace dsdecompx::cuda {

class Event {
public:
    explicit Event(unsigned flags = cudaEventDisableTiming);
    ~Event();
    Event(Event&& other) noexcept;
    Event& operator=(Event&& other) noexcept;
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    void record(cudaStream_t stream);
    void wait(cudaStream_t stream) const;
    bool ready() const;
    void synchronize_host() const;
    cudaEvent_t native() const noexcept { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

} // namespace dsdecompx::cuda
