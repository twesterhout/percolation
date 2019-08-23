#pragma once

#include "config.h"
#include <boost/pool/pool.hpp>
#include <memory>

TCM_NAMESPACE_BEGIN

struct PoolDeleter;

template <class T> using pool_unique_ptr = std::unique_ptr<T, PoolDeleter>;

template <class T> auto thread_local_pool() noexcept -> boost::pool<>&;

struct PoolDeleter {
    template <class T> auto operator()(T* const ptr) const noexcept -> void
    {
        TCM_ASSERT(ptr != nullptr, "Trying to delete a nullptr");
        auto& pool = thread_local_pool<T>();
        ptr->~T();
        pool.free(ptr);
    }
};

template <class T, class... Args>
auto pool_make_unique(Args&&... args) -> pool_unique_ptr<T>
{
    auto& pool = thread_local_pool<T>();
    // Useful for the case T's constructor throws
    auto deleter  = [&pool](auto* p) { pool.free(p); };
    auto temp_ptr = std::unique_ptr<void, decltype(deleter)>{
        pool.malloc(), std::move(deleter)};
    ::new (temp_ptr.get()) T{std::forward<Args>(args)...};
    return pool_unique_ptr<T>{static_cast<T*>(temp_ptr.release())};
}

TCM_NAMESPACE_END
