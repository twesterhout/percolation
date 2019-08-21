#include "detail/particle.hpp"

struct dummy_system_t {
    template <class T> struct unique_ptr {
        void* data[2];

        unique_ptr()                      = delete;
        unique_ptr(unique_ptr const&)     = delete;
        unique_ptr(unique_ptr&&) noexcept = default;
        auto operator=(unique_ptr const&) -> unique_ptr& = delete;
        auto operator=(unique_ptr&&) noexcept -> unique_ptr& = default;

        friend auto operator==(unique_ptr const& x, std::nullptr_t) noexcept
            -> bool
        {
            return x.data[0] == nullptr;
        }

        friend auto operator!=(unique_ptr const& x, std::nullptr_t) noexcept
            -> bool
        {
            return !(x == nullptr);
        }

        auto operator*() const noexcept -> T&
        {
            return *reinterpret_cast<T*>(data[1]);
        }
    };
};

template union tcm::particle_t<dummy_system_t>;

auto main() -> int {}
