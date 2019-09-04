// Copyright (c) 2018-2019, Tom Westerhout
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "config.h"
#include "memory.hpp"

#include <gsl/gsl-lite.hpp>
#include <cstdint>

TCM_NAMESPACE_BEGIN

class geometric_cluster_base_t;

#if 0
struct from_parent_index_t {};
constexpr from_parent_index_t from_parent_index{};
#endif

struct particle_base_t {
    using unique_ptr = pool_unique_ptr<geometric_cluster_base_t>;

  private:
    union {
        intptr_t   _raw;
        unique_ptr _cluster;

        static_assert(sizeof(unique_ptr) == sizeof(void*));
        static_assert(alignof(unique_ptr) == alignof(void*));
    };

    /// If root, destroys the geometric cluster and does nothing otherwise.
    constexpr auto destroy() noexcept -> void;

  public:
    /// Returns whether the site is empty.
    ///
    /// \noexcept
    [[nodiscard]] constexpr auto is_empty() const noexcept -> bool
    {
        return _raw == 0;
    }

    /// Returns whether the site is the root of a geometric cluster.
    ///
    /// \noexcept
    [[nodiscard]] constexpr auto is_root() const noexcept -> bool
    {
        return _raw > 0;
    }

    /// Returns whether the site has a parent.
    [[nodiscard]] constexpr auto is_child() const noexcept -> bool
    {
        return _raw < 0;
    }

    /// Swaps two particles.
    friend constexpr auto swap(particle_base_t& left,
                               particle_base_t& right) noexcept -> void
    {
        auto const temp = left._raw;
        left._raw       = right._raw;
        right._raw      = temp;
    }

    /// Default constructor.
    ///
    /// Creates an empty site.
    constexpr particle_base_t() noexcept : _raw{0} {}

    /// **Deleted** copy constructor.
    constexpr particle_base_t(particle_base_t const&) = delete;
    /// **Deleted** copy assignment operator.
    constexpr auto operator =(particle_base_t const&)
        -> particle_base_t& = delete;
    /// Move constructor.
    ///
    /// \noexcept
    constexpr particle_base_t(particle_base_t&& other) noexcept
        : particle_base_t{}
    {
        swap(*this, other);
    }
    /// Move assignment operator.
    ///
    /// \noexcept
    /*constexpr*/ auto operator=(particle_base_t&& other) noexcept
        -> particle_base_t&
    {
        particle_base_t temp{std::move(other)};
        swap(*this, temp);
        return *this;
    }

    /// Constructs a new particle given its parent.
    explicit particle_base_t(gsl::not_null<particle_base_t*> parent)
        TCM_NOEXCEPT : _raw{-reinterpret_cast<intptr_t>(parent.get())}
    {
        TCM_ASSERT(is_child(), "post-condition violated");
    }

    /// Constructs a new particle given a geometric cluster.
    ///
    /// \pre \p cluster is not `nullptr`.
    /// \noexcept
    explicit particle_base_t(unique_ptr cluster) TCM_NOEXCEPT
        : _cluster{std::move(cluster)}
    {
        TCM_ASSERT(is_root(), "post-condition violated");
    }

    ~particle_base_t() noexcept { destroy(); }

    [[nodiscard]] auto parent() const TCM_NOEXCEPT -> particle_base_t const&
    {
        TCM_ASSERT(is_child(), "only children have parents");
        return *reinterpret_cast<particle_base_t const*>(-_raw);
    }

    [[nodiscard]] auto parent() TCM_NOEXCEPT -> particle_base_t&
    {
        TCM_ASSERT(is_child(), "only children have parents");
        return *reinterpret_cast<particle_base_t*>(-_raw);
    }

    /// Changes the parent of this site.
    auto parent(particle_base_t& new_parent) TCM_NOEXCEPT -> void
    {
        TCM_ASSERT(is_child(), "only children have parents.");
        _raw = -reinterpret_cast<intptr_t>(std::addressof(new_parent));
    }

    /// Returns the cluster the particle owns.
    ///
    /// \pre is_empty() returns `false` and is_root() returns `true`.
    /// \noexcept
    auto cluster() const TCM_NOEXCEPT -> geometric_cluster_base_t const&
    {
        TCM_ASSERT(is_root(), "only cluster root nodes store the info.");
        return *_cluster;
    }

    /// \overload
    auto cluster() TCM_NOEXCEPT -> geometric_cluster_base_t&
    {
        TCM_ASSERT(is_root(), "only cluster root nodes store the info.");
        return *_cluster;
    }
};

inline auto find_root(particle_base_t& particle) -> particle_base_t&;

#if 0
/// A lightweight variant<index, owner<geometric_cluster>> built on
/// top of intptr_t.
///
// NOTE: This class is just a great collection of undefined behaviour...
// Assumptions which ensure it works correctly are really shaky :)
//                                                          Tom
template <class System> union particle_t { // {{{
    using cluster_type = geometric_cluster_t<System>;
    using unique_ptr   = typename System::template unique_ptr<cluster_type>;

    struct child_data_t {
        size_t parent_index;
        void*  dummy;
    };

    static_assert(sizeof(child_data_t) == sizeof(unique_ptr));

  private:
    child_data_t _child;
    unique_ptr   _cluster;

    static constexpr std::size_t empty_value =
        std::numeric_limits<std::size_t>::max();

    static auto roundtrip_is_okay(void* p) noexcept -> bool
    {
        TCM_ASSERT(p != nullptr,
                   "Roundtrip checks are only relevant for valid pointers.");
        auto const x = -reinterpret_cast<std::intptr_t>(p);
        return reinterpret_cast<void*>(-x) == p;
    }

  public:
    /// Swaps two particles.
    friend constexpr auto swap(particle_t& left, particle_t& right) noexcept
        -> void
    {
        auto const temp = left._child;
        left._child     = right._child;
        right._child    = temp;
    }

    /// Returns whether the site is empty.
    ///
    /// \noexcept
    constexpr auto is_empty() const noexcept
    {
        return _child.parent_index == empty_value && _child.dummy == nullptr;
    }

    /// Returns whether the site is the root of a geometric cluster.
    ///
    /// \noexcept
    constexpr auto is_root() const noexcept { return _child.dummy != nullptr; }

    /// Returns whether the site has a parent.
    constexpr auto is_child() const noexcept { return _child.dummy == nullptr; }

  private:
    /// If root, destroys the geometric cluster and does nothing otherwise.
    constexpr auto destroy() noexcept -> void
    {
        if (is_root()) { _cluster.~unique_ptr(); }
    }

  public:
    /// Default constructor.
    ///
    /// Creates an empty site.
    constexpr particle_t() noexcept : _child{empty_value, nullptr} {}

    /// **Deleted** copy constructor.
    constexpr particle_t(particle_t const&) = delete;
    /// **Deleted** copy assignment operator.
    constexpr particle_t& operator=(particle_t const&) = delete;
    /// Move constructor.
    ///
    /// \noexcept
    constexpr particle_t(particle_t&& other) noexcept : particle_t{}
    {
        swap(*this, other);
    }
    /// Move assignment operator.
    ///
    /// \noexcept
    constexpr particle_t& operator=(particle_t&& other) noexcept
    {
        particle_t temp{std::move(other)};
        swap(*this, temp);
        return *this;
    }

    /// Constructs a new particle given its parent index.
    ///
    /// \param parent_index Index of the parent site.
    /// \noexcept
    explicit constexpr particle_t(from_parent_index_t /*unused*/,
                                  size_t const parent_index) noexcept
        : _child{parent_index, nullptr}
    {
        TCM_ASSERT(parent_index != empty_value, "Bug! This index is reserved");
        TCM_ASSERT(is_child(), "Postcondition violated");
    }

    /// Constructs a new particle given a geometric cluster.
    ///
    /// \pre \p cluster is not `nullptr`.
    /// \noexcept
    explicit particle_t(unique_ptr cluster) noexcept
        : _cluster{std::move(cluster)}
    {
        TCM_ASSERT(_cluster != nullptr, "`cluster` should not be NULL");
        TCM_ASSERT(is_root(), "Postcondition violated");
    }

    ~particle_t() noexcept { destroy(); }

    /// Returns the index of the parent site.
    ///
    /// \pre is_empty() returns `false` and is_child() returns `true`.
    /// \noexcept
    constexpr auto parent_index() const noexcept -> size_t
    {
        TCM_ASSERT(!is_empty(), "Non-existent particle has no parent");
        TCM_ASSERT(is_child(), "Only children have parents");
        return _child.parent_index;
    }

    /// Changes the parent of this site.
    ///
    /// \pre is_empty() returns `false` and is_child() returns `true`.
    /// \noexcept
    constexpr auto parent_index(size_t const new_parent) noexcept -> void
    {
        TCM_ASSERT(!is_empty(), "Non-existent particle can't have a parent");
        TCM_ASSERT(is_child(), "Only children have parents.");
        TCM_ASSERT(new_parent != empty_value, "Bug! This index is reserved");
        _child.parent_index = new_parent;
    }

    /// Returns the cluster the particle owns.
    ///
    /// \pre is_empty() returns `false` and is_root() returns `true`.
    /// \noexcept
    auto cluster() const noexcept -> cluster_type const&
    {
        TCM_ASSERT(!is_empty(), "Non-existent particle owns no clusters");
        TCM_ASSERT(is_root(), "Only cluster root nodes store the info.");
        return *_cluster;
    }

    /// \overload
    auto cluster() noexcept -> cluster_type&
    {
        TCM_ASSERT(!is_empty(), "Non-existent particle owns no clusters");
        TCM_ASSERT(is_root(), "Only cluster root nodes store the info.");
        return *_cluster;
    }
}; // }}}
#endif

TCM_NAMESPACE_END
