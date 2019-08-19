// Copyright (c) 2018, Tom Westerhout
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
#include <boost/align/is_aligned.hpp>
#include <gsl/gsl-lite.hpp>
#include <immintrin.h>
#include <utility>

TCM_NAMESPACE_BEGIN

namespace detail {
TCM_FORCEINLINE auto find_fast_impl(intptr_t const* xs,
                                    intptr_t const  y) TCM_NOEXCEPT -> size_t
{
    static_assert(sizeof(intptr_t) == 8);
    TCM_ASSERT(boost::alignment::is_aligned(xs, 16),
               "Array is not aligned properly.");
    auto const p  = reinterpret_cast<__m128i const*>(xs);
    auto       x1 = _mm_load_si128(p);
    auto       x2 = _mm_load_si128(p + 1);
    auto       x3 = _mm_load_si128(p + 2);
    auto       x4 = _mm_load_si128(p + 3);
    x1            = _mm_cmpeq_epi64(x1, _mm_set1_epi64x(y));
    x2            = _mm_cmpeq_epi64(x2, _mm_set1_epi64x(y));
    x3            = _mm_cmpeq_epi64(x3, _mm_set1_epi64x(y));
    x4            = _mm_cmpeq_epi64(x4, _mm_set1_epi64x(y));
    auto const i1 = _mm_movemask_epi8(x1);
    auto const i2 = _mm_movemask_epi8(x2);
    auto const i3 = _mm_movemask_epi8(x3);
    auto const i4 = _mm_movemask_epi8(x4);
    return (static_cast<size_t>(i4) << 48) | (static_cast<size_t>(i3) << 32)
           | (static_cast<size_t>(i2) << 16) | (static_cast<size_t>(i1) << 0);
}

inline auto find_fast(intptr_t const (&xs)[8], intptr_t const y) noexcept
    -> unsigned
{
    auto const i = find_fast_impl(xs, y);
    return (i != 0) ? static_cast<unsigned>(__builtin_ctzl(i) >> 3) : 8;
}
} // namespace detail

template <class MagneticCluster, size_t N> struct neighbour_stats_t {
    using magnetic_cluster_type = MagneticCluster;
    using value_type =
        std::tuple<magnetic_cluster_type*, gsl::span<size_t const>>;

  private:
    /// Magnetic clusters to which we are connected. The first `_size` pointers
    /// should be valid and the rest are `nullptr`s.
    alignas(64) std::array<magnetic_cluster_type*, N> _clusters;
    /// Number of connections with magnetic clusters. So `_counts[i]` is the
    /// number of connections we have with cluster `*_clusters[i]`. Counts are
    /// always positive.
    alignas(64) std::array<size_t, N> _counts;
    /// Sites to which we are connected in each magnetic cluster. So `_sites[i]`
    /// is an array of `_counts[i]` sites from `*_clusters[i]` to which we are
    /// connected.
    alignas(64) std::array<std::array<size_t, N>, N> _sites;
    int      _boundaries; ///< Boundaries bitmask.
    unsigned _size;       ///< Total number of magnetic clusters.

    struct const_iterator_t;
    friend struct const_iterator_t;

    template <class T, size_t Size, size_t Alignment = 64>
    static auto reset_array(std::array<T, Size>& xs, T value) noexcept -> void
    {
#if 0
        auto* p =
            static_cast<T*>(__builtin_assume_aligned(xs.data(), Alignment));
#    pragma omp simd
        for (auto i = size_t{0}; i < Size; ++i) {
            p[i] = value;
        }
#else
        using std::begin, std::end;
        TCM_ASSERT(reinterpret_cast<std::uintptr_t>(xs.data()) % Alignment == 0,
                   "Bug! Array is not aligned.");
        std::fill(begin(xs), end(xs), value);
#endif
    }

  public:
    neighbour_stats_t() TCM_NOEXCEPT { reset(); }

    neighbour_stats_t(neighbour_stats_t const&) = delete;
    neighbour_stats_t(neighbour_stats_t&&)      = delete;
    neighbour_stats_t& operator=(neighbour_stats_t const&) = delete;
    neighbour_stats_t& operator=(neighbour_stats_t&&) = delete;

    /// Returns the number of magnetic clusters to which we are connected.
    constexpr auto size() const noexcept -> size_t { return _size; }
    /// Returns the boundaries bitmask
    constexpr auto boundaries() const noexcept -> int { return _boundaries; }

    auto begin() const TCM_NOEXCEPT -> const_iterator_t { return {*this, 0ul}; }
    auto end() const TCM_NOEXCEPT -> const_iterator_t { return {*this, _size}; }

    /// Resets the statistics so that a new site can be analysed.
    auto reset() TCM_NOEXCEPT
    {
        reset_array(_clusters, static_cast<magnetic_cluster_type*>(nullptr));
        reset_array(_counts, size_t{0});
        for (auto& x : _sites) {
            reset_array(x, std::numeric_limits<size_t>::max());
        }
        _boundaries = 0;
        _size       = 0;
    }

    /// Adds neighbour `i`
    auto insert(size_t const                 i,
                magnetic_cluster_type* const cluster) TCM_NOEXCEPT -> void
    {
        TCM_ASSERT(cluster != nullptr, "`cluster` should not be nullptr");
        if (auto const index = find(cluster); index != N) {
            TCM_ASSERT(index < _size, "Hehe!");
            TCM_ASSERT(_clusters[index] == cluster, "Hehe!");
            _sites[index][_counts[index]] = i;
            ++_counts[index];
        }
        else {
            _clusters[_size] = cluster;
            _sites[_size][0] = i;
            ++_counts[_size];
            ++_size;
        }
    }

    /// Adds boundary `boundary`
    constexpr auto insert(int const boundary) noexcept -> void
    {
        _boundaries |= boundary;
    }

    auto sort() TCM_NOEXCEPT -> void
    {
        // We sort in reverse
        auto const less = [this](size_t const i, size_t const j) {
            return _counts[i] > _counts[j];
        };
        auto const swap = [this](size_t const i, size_t const j) {
            std::swap(_clusters[i], _clusters[j]);
            std::swap(_counts[i], _counts[j]);
            for (auto n = size_t{0}; n < N; ++n) {
                std::swap(_sites[i * N + n], _sites[j * N + n]);
            }
        };

        // Simple bubble sort
        auto begin = size_t{0};
        auto end   = size_t{_size};
        do {
            auto new_end = begin;
            for (auto i = begin; i != end; ++i) {
                if (less(i + 1, i)) {
                    swap(i, i + 1);
                    new_end = i;
                }
            }
            end = new_end;
        } while (end - begin > 1);

#if 0
        for (auto i = size_t{0}; i < N; ++i) {
            std::fprintf(stderr, "%p, ", _clusters[i]);
        }
        std::fprintf(stderr, "\n");
#endif
    }

  private:
    struct const_iterator_t {
        struct value_type {
            magnetic_cluster_type*  cluster;
            gsl::span<size_t const> neighbours;
        };
        // using value_type =
        //     std::tuple<magnetic_cluster_type*, gsl::span<size_t const>>;
        using reference         = value_type const&;
        using pointer           = value_type const*;
        using difference_type   = std::ptrdiff_t;
        using iterator_category = std::input_iterator_tag;

      private:
        constexpr auto fetch() TCM_NOEXCEPT -> void
        {
            TCM_ASSERT(_obj != nullptr, "Bug! Can't fetch from a nullptr");
            TCM_ASSERT(_i < _obj->_size, "Index out of bounds");
            _value.cluster    = _obj->_clusters[_i];
            _value.neighbours = _obj->sites(_i);
        }

      public:
        constexpr const_iterator_t() noexcept : _obj{nullptr}, _i{0}, _value{}
        {}

        constexpr const_iterator_t(neighbour_stats_t const& obj,
                                   size_t const             i) TCM_NOEXCEPT
            : _obj{&obj}
            , _i{i}
        {
            TCM_ASSERT(i <= obj._size, "");
            if (_i != _obj->_size) { fetch(); }
        }

        constexpr const_iterator_t(const const_iterator_t&) noexcept = default;
        constexpr const_iterator_t(const_iterator_t&&) noexcept      = default;
        constexpr const_iterator_t&
        operator=(const_iterator_t const&) noexcept = default;
        constexpr const_iterator_t&
        operator=(const_iterator_t&&) noexcept = default;

        constexpr auto operator*() const TCM_NOEXCEPT -> reference
        {
            TCM_ASSERT(_obj != nullptr, "Iterator not dereferenceable");
            TCM_ASSERT(_i < _obj->_size, "Iterator not dereferenceable");
            return _value;
        }

        constexpr auto operator-> () const TCM_NOEXCEPT -> pointer
        {
            TCM_ASSERT(_obj != nullptr, "Iterator not dereferenceable");
            TCM_ASSERT(_i < _obj->_size, "Iterator not dereferenceable");
            return &_value;
        }

        constexpr auto operator++() TCM_NOEXCEPT -> const_iterator_t&
        {
            TCM_ASSERT(_obj != nullptr, "Iterator not incrementable");
            TCM_ASSERT(_i < _obj->_size, "Iterator not incrementable");
            ++_i;
            if (_i != _obj->_size) { fetch(); }
            return *this;
        }

        constexpr auto operator++(int) TCM_NOEXCEPT -> const_iterator_t
        {
            const_iterator_t old{*this};
            ++(*this);
            return old;
        }

        friend auto operator==(const_iterator_t const& x,
                               const_iterator_t const& y) TCM_NOEXCEPT -> bool
        {
            TCM_ASSERT(x._obj == y._obj,
                       "Can't compare iterators pointing to different objects");
            return x._i == y._i;
        }

        friend auto operator!=(const_iterator_t const& x,
                               const_iterator_t const& y) TCM_NOEXCEPT -> bool
        {
            return !(x == y);
        }

      private:
        value_type               _value;
        neighbour_stats_t const* _obj;
        size_t                   _i;
    };

    /// Returns all the sites in cluster `_clusters[i]` to which we are
    /// connected.
    constexpr auto sites(size_t const i) const TCM_NOEXCEPT
        -> gsl::span<size_t const>
    {
        TCM_ASSERT(i < _size, "Index out of bounds");
        return {_sites[i].data(), _counts[i]};
    }

    auto find(magnetic_cluster_type* const cluster) noexcept -> size_t
    {
#if 0
        auto  index = N;
        auto* p     = static_cast<magnetic_cluster_type**>(
            __builtin_assume_aligned(_clusters.data(), 64));
#    pragma omp simd
        for (auto i = size_t{0}; i < N; ++i) {
            index = (p[i] == cluster) ? i : index;
        }
        return index;
#else
        using std::begin, std::end;
        return static_cast<size_t>(
            std::find(begin(_clusters), end(_clusters), cluster)
            - begin(_clusters));
#endif
    }
};

TCM_NAMESPACE_END
