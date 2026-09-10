#include "cuda_event.hpp"
#include "cuda_error.hpp"

#include <utility>

namespace dsdecompx::cuda {

Event::Event(unsigned flags) {
    detail::check(cudaEventCreateWithFlags(&event_, flags), "cudaEventCreateWithFlags");
}

Event::~Event() {
    if (event_) cudaEventDestroy(event_);
}

Event::Event(Event&& other) noexcept
    : event_(std::exchange(other.event_, nullptr)) {}

Event& Event::operator=(Event&& other) noexcept {
    if (this == &other) return *this;
    if (event_) cudaEventDestroy(event_);
    event_ = std::exchange(other.event_, nullptr);
    return *this;
}

void Event::record(cudaStream_t stream) {
    detail::check(cudaEventRecord(event_, stream), "cudaEventRecord");
}

void Event::wait(cudaStream_t stream) const {
    detail::check(cudaStreamWaitEvent(stream, event_, 0), "cudaStreamWaitEvent");
}

bool Event::ready() const {
    const cudaError_t status = cudaEventQuery(event_);
    if (status == cudaSuccess) return true;
    if (status == cudaErrorNotReady) return false;
    detail::check(status, "cudaEventQuery");
    return false;
}

void Event::synchronize_host() const {
    detail::check(cudaEventSynchronize(event_), "cudaEventSynchronize");
}

} // namespace dsdecompx::cuda
