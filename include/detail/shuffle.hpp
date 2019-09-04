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

#include "detail/utility.hpp"
#include <gsl/gsl-lite.hpp>
#include <random>
#include <type_traits>

TCM_NAMESPACE_BEGIN

/// Shuffles integers in the range [0, N)
template <class T, class Generator> class shuffler_t {
    static_assert(std::is_integral_v<T>,
                  "`tcm::shuffler_t<T, Generator>` currently only supports "
                  "integral `T`s.");
    static_assert(
        std::is_same_v<T, std::decay_t<T>>,
        "`tcm::shuffle_t<T, Generator>` requires `T` to be an unqualified "
        "value type.");

    class Iterator {
        using value_type        = T;
        using difference_type   = std::ptrdiff_t;
        using reference         = T;
        using pointer           = T const*;
        using iterator_category = std::input_iterator_tag;

        friend class shuffler_t;

      private:
        auto generate_one()
        {
            TCM_ASSERT(_shuffler != nullptr && _shuffler->_buffer.size() > _i,
                       "There are no more elements to generate.");
            using Dist   = std::uniform_int_distribution<size_t>;
            using Params = Dist::param_type;
            using std::swap;
            Dist       uid;
            auto const shift = uid(
                _shuffler->_gen, Params{0, _shuffler->_buffer.size() - 1 - _i});
            if (shift != 0) {
                swap(_shuffler->_buffer[_i], _shuffler->_buffer[_i + shift]);
            }
        }

        explicit Iterator(shuffler_t& obj) : _shuffler{&obj}, _i{0}
        {
            if (_shuffler->_buffer.size() > 1) { generate_one(); }
        }

        constexpr Iterator(shuffler_t& obj, size_t const i) noexcept
            : _shuffler{&obj}, _i{i}
        {
            TCM_ASSERT(i == _shuffler->_buffer.size(), "Index out of bounds");
        }

      public:
        constexpr Iterator() noexcept : _shuffler{nullptr}, _i{0} {}
        constexpr Iterator(Iterator const&) noexcept = default;
        constexpr Iterator(Iterator&&) noexcept      = default;
        constexpr Iterator& operator=(Iterator const&) noexcept = default;
        constexpr Iterator& operator=(Iterator&&) noexcept = default;

        constexpr auto operator*() const noexcept -> reference
        {
            TCM_ASSERT(_shuffler != nullptr && _i < _shuffler->_buffer.size(),
                       "Iterator is not dereferenceable.");
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
            return _shuffler->_buffer[_i];
        }

        constexpr auto operator-> () const noexcept -> pointer
        {
            TCM_ASSERT(_shuffler != nullptr && _i < _shuffler->_buffer.size(),
                       "Iterator is not dereferenceable.");
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
            return _shuffler->_buffer.data() + _i;
        }

        auto operator++() -> Iterator&
        {
            TCM_ASSERT(_shuffler != nullptr && _i < _shuffler->_buffer.size(),
                       "Iterator is not incrementable.");
            ++_i;
            if (_shuffler->_buffer.size() > _i) { generate_one(); }
            return *this;
        }

        auto operator++(int) -> Iterator
        {
            auto temp = *this;
            ++(*this);
            return temp;
        }

        constexpr auto operator==(Iterator const& other) const noexcept -> bool
        {
            TCM_ASSERT(_shuffler == other._shuffler,
                       "Can't compare iterators into different shuffler_t "
                       "objects.");
            return _i == other._i;
        }

        constexpr auto operator!=(Iterator const& other) const noexcept -> bool
        {
            TCM_ASSERT(_shuffler == other._shuffler,
                       "Can't compare iterators into different shuffler_t "
                       "objects.");
            return _i != other._i;
        }

      private:
        shuffler_t* _shuffler;
        size_t      _i;
    };

  public:
    using value_type      = T;
    using difference_type = std::ptrdiff_t;

    /// Given a pointer `data` to a buffer of `size` elements, creates a
    /// shuffler that is able to iterate through the elements in random order.
    /// `gen` is used for generation of random numbers. It must satisfy the
    /// UniformRandomBitGenerator concept.
    constexpr shuffler_t(gsl::span<value_type> buffer, Generator& gen) noexcept
        : _buffer{buffer}, _gen{gen}
    {}

    constexpr shuffler_t(shuffler_t const&) noexcept = delete;
    constexpr shuffler_t(shuffler_t&&) noexcept      = default;
    constexpr shuffler_t& operator=(shuffler_t const&) noexcept = delete;
    constexpr shuffler_t& operator=(shuffler_t&&) noexcept = default;

    constexpr auto begin() noexcept { return Iterator{*this}; }
    constexpr auto end() noexcept { return Iterator{*this, _buffer.size()}; }

  private:
    gsl::span<T> _buffer;
    Generator&   _gen;
};

template <class T, class Generator>
shuffler_t(gsl::span<T>, Generator&)->shuffler_t<T, Generator>;

template <class Generator>
auto make_stateful_shuffler(uint32_t const number_sites, Generator& generator)
{
    using buffer_t =
        decltype(make_buffer_of<uint32_t>(std::declval<uint32_t>()));
    using shuffler_t = shuffler_t<uint32_t, Generator>;

    class stateful_shuffler_t : public shuffler_t {
        buffer_t _buffer;

      public:
        stateful_shuffler_t(buffer_t&& buffer, uint32_t const size,
                            Generator& generator)
            : shuffler_t{gsl::span<uint32_t>{buffer.get(), size}, generator}
            , _buffer{std::move(buffer)}
        {
            std::iota(_buffer.get(), _buffer.get() + size, 0U);
        }
    };
    return stateful_shuffler_t{make_buffer_of<uint32_t>(number_sites),
                               number_sites, generator};
}

TCM_NAMESPACE_END
