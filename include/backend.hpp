#pragma once

#include "backend_selection.hpp"

#include <memory>

namespace dsdecomp {

namespace detail {
class Backend;
}

class Backend {
public:
    Backend() noexcept;
    ~Backend();

    Backend(Backend&&) noexcept;
    Backend& operator=(Backend&&) noexcept;

    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    explicit Backend(std::unique_ptr<detail::Backend> implementation);

    BackendKind kind() const noexcept;
    explicit operator bool() const noexcept;

private:
    friend class Decomposition;
    std::unique_ptr<detail::Backend> implementation_;
};

} // namespace dsdecomp
