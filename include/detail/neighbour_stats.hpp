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
// This would be so much better with AVX2...
template <size_t N>
TCM_NOINLINE auto find_fast_impl(int64_t const* xs,
                                 int64_t const  y) TCM_NOEXCEPT -> unsigned
{
    constexpr auto vector_size = 2U;
    static_assert(N % vector_size == 0);
    TCM_ASSERT(boost::alignment::is_aligned(xs, 32UL),
               "array is not aligned properly.");

    auto v = _mm_set1_epi64x(y);
    auto p = reinterpret_cast<__m128i const*>(xs);
    if constexpr (N == 8) {
        auto const x1 = _mm_load_si128(p);
        auto const x2 = _mm_load_si128(p + 1);
        auto const x3 = _mm_load_si128(p + 2);
        auto const x4 = _mm_load_si128(p + 3);

        auto mask1 =
            static_cast<unsigned>(_mm_movemask_epi8(_mm_cmpeq_epi64(x1, v)));
        auto mask2 =
            static_cast<unsigned>(_mm_movemask_epi8(_mm_cmpeq_epi64(x2, v)));
        auto mask3 =
            static_cast<unsigned>(_mm_movemask_epi8(_mm_cmpeq_epi64(x3, v)));
        auto mask4 =
            static_cast<unsigned>(_mm_movemask_epi8(_mm_cmpeq_epi64(x4, v)));

        mask1 |= (mask2 << 16U);
        mask3 |= (mask4 << 16U);
        auto const mask = static_cast<uint64_t>(mask1)
                          | (static_cast<uint64_t>(mask3) << 32U);
        return mask == 0 ? N
                         : (static_cast<unsigned>(__builtin_ctzl(mask)) >> 3);
    }
    else {
        for (auto i = 0U; i < N / vector_size; ++i, ++p) {
            auto x1    = _mm_load_si128(p);
            auto mask1 = static_cast<unsigned>(
                _mm_movemask_epi8(_mm_cmpeq_epi64(x1, v)));
            if (mask1 != 0) {
                return vector_size * i
                       + (static_cast<unsigned>(__builtin_ctz(mask1)) >> 3);
            }
        }
        return N;
    }
}

template <class T, size_t N>
TCM_FORCEINLINE auto find_fast(std::array<T*, N> const& xs, T* const y) noexcept
    -> unsigned
{
    return find_fast_impl<N>(reinterpret_cast<intptr_t const*>(xs.data()),
                             reinterpret_cast<intptr_t>(y));
}
} // namespace detail

template <class Cluster, size_t N> struct alignas(64) neighbour_stats_t {
    using cluster_type = Cluster;

    template <size_t Alignment>
    static constexpr auto round_up(size_t const value) noexcept -> size_t
    {
        static_assert(Alignment != 0 && (Alignment & (Alignment - 1)) == 0,
                      "Invalid alignment");
        return (value + (Alignment - 1U)) & ~(Alignment - 1U);
    }

    static constexpr auto capacity = round_up<32U / sizeof(void*)>(N);

  private:
    alignas(64) std::array<cluster_type*, capacity> _clusters;
    alignas(32) std::array<size_t, capacity> _sites;
    bool     _still_searching;
    unsigned _size;

    struct sentinel_t {};
    struct const_iterator_t;
    friend struct const_iterator_t;

    template <int64_t value, class T>
    static auto reset_array(std::array<T, capacity>& xs) noexcept -> void
    {
        constexpr auto alignment   = 32UL; // Alignment for AVX;
        constexpr auto vector_size = 4;
        TCM_ASSERT(reinterpret_cast<uintptr_t>(xs.data()) % alignment == 0,
                   "array is not aligned.");
        static_assert(capacity % vector_size == 0);
        auto const a = _mm256_set1_epi64x(value);
        auto*      p = reinterpret_cast<__m256i*>(xs.data());
        for (auto i = 0U; i < capacity / vector_size; ++i, ++p) {
            _mm256_store_si256(p, a);
        }
        TCM_ASSERT(
            std::all_of(xs.begin(), xs.end(),
                        [](auto x) {
                            if constexpr (std::is_integral<T>::value) {
                                return x == static_cast<T>(value);
                            }
                            else if constexpr (std::is_pointer<T>::value) {
                                return reinterpret_cast<int64_t>(x) == value;
                            }
                            else {
                                static_assert(!std::is_same<T, T>::value);
                            }
                        }),
            "reset_array is broken");
    }

  public:
    neighbour_stats_t() TCM_NOEXCEPT { reset(); }

    neighbour_stats_t(neighbour_stats_t const&) = delete;
    neighbour_stats_t(neighbour_stats_t&&)      = delete;
    neighbour_stats_t& operator=(neighbour_stats_t const&) = delete;
    neighbour_stats_t& operator=(neighbour_stats_t&&) = delete;

    /// Returns the number of magnetic clusters to which we are connected.
    [[nodiscard]] constexpr auto size() const noexcept -> unsigned
    {
        return _size;
    }

    [[nodiscard]] constexpr auto begin() const TCM_NOEXCEPT -> const_iterator_t
    {
        return const_iterator_t{*this};
    }

    [[nodiscard]] constexpr auto end() const TCM_NOEXCEPT -> sentinel_t
    {
        return {};
    }

    /// Resets the statistics so that a new site can be analysed.
    auto reset() TCM_NOEXCEPT
    {
        reset_array<0>(_clusters);
        reset_array<~0>(_sites);
        _still_searching = true;
        _size            = 0;
    }

    /// Adds neighbour `i`
    auto insert(size_t const i, cluster_type* const cluster) TCM_NOEXCEPT
        -> void
    {
        TCM_ASSERT(cluster != nullptr, "`cluster` should not be nullptr");

        if (_still_searching) {
            if (auto const index = find(cluster); index != N) {
                if (index != 0) {
                    using std::swap;
                    swap(_clusters[0], _clusters[index]);
                    swap(_sites[0], _sites[index]);
                }
                _still_searching = false;
            }
        }
        _clusters[_size] = cluster;
        _sites[_size]    = i;
        ++_size;
    }

  private:
    struct const_iterator_t {
        struct value_type {
            cluster_type*           cluster;
            gsl::span<size_t const> neighbours;
        };
        using reference         = value_type const&;
        using pointer           = value_type const*;
        using difference_type   = std::ptrdiff_t;
        using iterator_category = std::input_iterator_tag;

      private:
        constexpr auto fetch() TCM_NOEXCEPT -> void
        {
            TCM_ASSERT(_obj != nullptr, "Bug! Can't fetch from a nullptr");
            // We find the index of the first cluster we haven't visited yet.
            auto i = static_cast<unsigned>(
                std::find(std::begin(_todo), std::end(_todo), true)
                - std::begin(_todo));
            TCM_ASSERT(i != _todo.size(), "No more clusters to visit");
            _value.cluster = _obj->_clusters[i];
            // Now we iterate over all _obj->_clusters and find all occurences
            // of _value.cluster. The first one is obvious
            auto count      = 0U;
            _sites[count++] = _obj->_sites[i];
            _todo[i]        = false;
            for (++i; i < _obj->_size; ++i) {
                if (_obj->_clusters[i] == _value.cluster) {
                    _sites[count++] = _obj->_sites[i];
                    _todo[i]        = false;
                }
            }
            _value.neighbours = gsl::span<size_t const>{_sites.data(), count};
        }

      public:
        constexpr const_iterator_t() noexcept
            : _obj{nullptr}, _value{nullptr, {}}, _sites{}, _todo{}
        {}

        explicit constexpr const_iterator_t(neighbour_stats_t const& obj)
            TCM_NOEXCEPT
            : _obj{&obj}
            , _value{nullptr, {}}
            , _sites{}
            , _todo{}
        {
            auto i = 0U;
            for (; i < _obj->_size; ++i) {
                _todo[i] = true;
            }
            for (; i < N; ++i) {
                _todo[i] = false;
            }
            if (_obj->_size > 0) { fetch(); }
        }

        constexpr const_iterator_t(const const_iterator_t&) noexcept = default;
        constexpr const_iterator_t(const_iterator_t&&) noexcept      = default;
        constexpr auto operator  =(const_iterator_t const&) noexcept
            -> const_iterator_t& = default;
        constexpr auto operator  =(const_iterator_t&&) noexcept
            -> const_iterator_t& = default;

        constexpr auto operator*() const TCM_NOEXCEPT -> reference
        {
            TCM_ASSERT(_obj != nullptr, "Iterator not dereferenceable");
            TCM_ASSERT(_value.cluster != nullptr,
                       "Iterator not dereferenceable");
            return _value;
        }

        constexpr auto operator-> () const TCM_NOEXCEPT -> pointer
        {
            TCM_ASSERT(_obj != nullptr, "Iterator not dereferenceable");
            TCM_ASSERT(_value.cluster != nullptr,
                       "Iterator not dereferenceable");
            return &_value;
        }

        constexpr auto operator++() TCM_NOEXCEPT -> const_iterator_t&
        {
            TCM_ASSERT(_obj != nullptr, "Iterator not incrementable");
            TCM_ASSERT(_value.cluster != nullptr, "Iterator not incrementable");
            using std::begin, std::end;
            if (std::any_of(begin(_todo), end(_todo),
                            [](auto b) { return b; })) {
                fetch();
            }
            else {
                _value = value_type{nullptr, {}};
            }
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
            using std::begin, std::end;
            return x._value == y._value
                   && std::equal(begin(x._todo), end(x._todo), begin(y._todo));
        }

        friend auto operator!=(const_iterator_t const& x,
                               const_iterator_t const& y) TCM_NOEXCEPT -> bool
        {
            return !(x == y);
        }

        friend auto operator==(const_iterator_t const& x,
                               sentinel_t /*unused*/) TCM_NOEXCEPT -> bool
        {
            TCM_ASSERT(x._obj != nullptr,
                       "Iterator does not belong to a container");
            return x._value.cluster == nullptr;
        }

        friend auto operator!=(const_iterator_t const& x,
                               sentinel_t              y) TCM_NOEXCEPT -> bool
        {
            return !(x == y);
        }

      private:
        neighbour_stats_t const* _obj;
        value_type               _value;
        std::array<size_t, N>    _sites;
        std::array<bool, N>      _todo;
    };

    auto find(cluster_type* const cluster) noexcept -> unsigned
    {
        using std::begin, std::end;
        return static_cast<unsigned>(
            std::find(begin(_clusters), end(_clusters), cluster)
            - begin(_clusters));
    }
};

TCM_NAMESPACE_END
