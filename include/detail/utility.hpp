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
// #include "trace.hpp"
#include <immintrin.h>
// #include <cassert>
#include <cstdint>
#include <memory>
#include <random>

TCM_NAMESPACE_BEGIN

using std::int32_t;
using std::size_t;

struct free_deleter_t {
    template <class T> auto operator()(T* p) const noexcept -> void
    {
        TCM_ASSERT(p != nullptr, "Trying to delete a nullptr.");
        std::free(p);
    }
};

using FreeDeleter = free_deleter_t;

template <class T, size_t Alignment = 64>
auto make_buffer_of(size_t const n) -> std::unique_ptr<T[], FreeDeleter>
{
    static_assert(Alignment > 0 && ((Alignment - 1) & Alignment) == 0,
                  "Invalid alignment.");
    if (n > std::numeric_limits<size_t>::max() / sizeof(T)) {
        throw std::overflow_error{"integer overflow in make_buffer_of"};
    }
    auto* p =
        reinterpret_cast<T*>(std::aligned_alloc(Alignment, sizeof(T) * n));
    if (p == nullptr) { throw std::bad_alloc{}; }
    return {p, free_deleter_t{}};
}

template <class T> using tcm_unique_ptr = std::unique_ptr<T, FreeDeleter>;

union V3 {
#if defined(TCM_CLANG)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#endif
    __m128i raw;
    struct {
        std::int32_t x;
        std::int32_t y;
        std::int32_t z;
        std::int32_t _padding;
    };
#if defined(TCM_CLANG)
#    pragma clang diagnostic pop
#endif

    constexpr V3() noexcept : x{0}, y{0}, z{0}, _padding{0} {}

    V3(__m128i raw_) noexcept : raw{raw_}
    {
        TCM_ASSERT(_padding == 0, "`_padding` must be zero-initialised");
    }

    constexpr V3(std::int32_t const x_, std::int32_t const y_,
                 std::int32_t const z_) noexcept
        : x{x_}, y{y_}, z{z_}, _padding{0}
    {}

    V3(V3 const& other) noexcept : raw{other.raw}
    {
        TCM_ASSERT(other._padding == 0, "`_padding` must be zero-initialised");
    }

    V3(V3&& other) noexcept : raw{other.raw}
    {
        TCM_ASSERT(other._padding == 0, "`_padding` must be zero-initialised");
    }

    V3& operator=(V3 const& other) noexcept
    {
        TCM_ASSERT(other._padding == 0, "`_padding` must be zero-initialised");
        raw = other.raw;
        return *this;
    }

    V3& operator=(V3&& other) noexcept
    {
        TCM_ASSERT(other._padding == 0, "`_padding` must be zero-initialised");
        raw = other.raw;
        return *this;
    }
};

#if 0
inline auto to_chars(char* first, char* last, V3 const& v) noexcept
    -> std::to_chars_result
{
    constexpr char shortest_vector[] = "(0,0,0)";
    if (last - first
        <= static_cast<std::ptrdiff_t>(std::size(shortest_vector))) {
        return {last, std::errc::value_too_large};
    }

    std::to_chars_result result{first, std::errc{}};
    *(result.ptr++) = '(';
    result          = std::to_chars(result.ptr, last, v.x);
    if (result.ptr == last) return {last, std::errc::value_too_large};
    *(result.ptr++) = ',';
    result          = std::to_chars(result.ptr, last, v.y);
    if (result.ptr == last) return {last, std::errc::value_too_large};
    *(result.ptr++) = ',';
    result          = std::to_chars(result.ptr, last, v.z);
    if (result.ptr == last) return {last, std::errc::value_too_large};
    *(result.ptr++) = ')';
}

inline auto operator<<(std::ostream& out, Position const& x) -> std::ostream&
{
    if (x._padding == 0) {
        assert(x._padding == 0);
        return out << "(" << x.x << ", " << x.y << ", " << x.z << ")";
    } else {
        out << "(" << x.x << ", " << x.y << ", " << x.z << ", " << x._padding << ")";
        assert(x._padding == 0);
        return out;
    }
}
#endif

inline auto operator==(V3 const& a, V3 const& b) noexcept -> bool
{
    TCM_ASSERT(a._padding == 0 && b._padding == 0, "`_padding` must be zero");
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

inline auto operator!=(V3 const& a, V3 const& b) noexcept -> bool
{
    return !(a == b);
}

inline auto operator+(V3 const& x, V3 const& y) noexcept -> V3
{
    return {_mm_add_epi32(x.raw, y.raw)};
}

inline auto operator+=(V3& x, V3 const& y) noexcept -> V3&
{
    x.raw = _mm_add_epi32(x.raw, y.raw);
    return x;
}

inline auto operator-(V3 const& x) noexcept -> V3
{
    return {_mm_sub_epi32(_mm_set1_epi32(0), x.raw)};
}

inline auto operator-(V3 const& x, V3 const& y) noexcept -> V3
{
    return {_mm_sub_epi32(x.raw, y.raw)};
}

inline auto operator-=(V3& x, V3 const y) noexcept -> V3&
{
    x.raw = _mm_sub_epi32(x.raw, y.raw);
    return x;
}

enum class chirality_t : int {
    Left  = -1,
    Right = 1,
};

/*
template <class Generator>
auto random_chirality(Generator& gen) -> chirality_t
{
    std::uniform_int_distribution<std::uint8_t> dist;
    auto const chirality = 2 * static_cast<int>(dist(gen) % 2) - 1;
    assert(
        (chirality == -1 || chirality == 1) && "Bug! Postcondition violated.");
    return static_cast<chirality_t>(chirality);
}
*/

namespace detail {
template <class T> constexpr inline auto pi = static_cast<T>(M_PI);

template <class T>
constexpr inline auto
    two_pi = static_cast<T>(6.2831853071795864769252867665586666312879240);

// constexpr inline float two_pi = 6.2831855f;

#if 0
constexpr auto mod12(int const x) noexcept -> int
{
    TCM_ASSERT(x + 12 >= 0, "`x` must be >= -12");
    // TODO(twesterhout): Add a __builtin_assume here
    return static_cast<unsigned>(x + 12) % 12;
}

inline auto mod_2_pi(double const x) noexcept -> double
{
    TCM_ASSERT(x + two_pi >= 0, "`x` must be >= -2 * PI");
    return std::fmod(x + two_pi, two_pi);
}
#endif
} // namespace detail

struct angle_t {
  private:
    float _raw;

  public:
    constexpr explicit angle_t(float const angle = 0) noexcept : _raw{angle}
    {
        // if (angle < 0 || angle >= detail::two_pi) {
        //     std::fprintf(stderr, "angle_t{%f}\n", static_cast<double>(angle));
        // }
        TCM_ASSERT(std::isnan(angle)
                       || (0 <= angle && angle < detail::two_pi<float>),
                   "Angle out of domain");
    }

    constexpr angle_t(angle_t const&) noexcept = default;
    constexpr angle_t(angle_t&&) noexcept      = default;
    constexpr angle_t& operator=(angle_t const&) noexcept = default;
    constexpr angle_t& operator=(angle_t&&) noexcept = default;

    explicit constexpr operator float() const noexcept { return _raw; }

    friend constexpr auto operator==(angle_t const& x,
                                     angle_t const& y) noexcept -> bool
    {
        return x._raw == y._raw;
    }

    friend constexpr auto operator!=(angle_t const& x,
                                     angle_t const& y) noexcept -> bool
    {
        return x._raw != y._raw;
    }

    friend auto operator+(angle_t const& x, angle_t const& y) noexcept
        -> angle_t
    {
        constexpr auto two_pi = detail::two_pi<float>;
        auto const     result = x._raw + y._raw;
        return angle_t{result - (result >= two_pi) * two_pi};
    }

    friend auto operator+=(angle_t& x, angle_t const& y) noexcept -> angle_t&
    {
        x = x + y;
        return x;
    }

    friend auto operator-(angle_t const& x, angle_t const& y) noexcept
        -> angle_t
    {
        constexpr auto two_pi  = detail::two_pi<float>;
        constexpr auto epsilon = -4.7683716E-7f;
        auto const     result  = x._raw - y._raw;
        if (result < epsilon) { return angle_t{result + two_pi}; }
        else if (result < 0) {
            return angle_t{0};
        }
        else {
            return angle_t{result};
        }
    }

    friend auto operator-=(angle_t& x, angle_t const& y) noexcept -> angle_t&
    {
        x = x - y;
        return x;
    }
};

template <class Generator> auto random_angle(Generator& g) -> angle_t
{
    using Dist   = std::uniform_real_distribution<float>;
    using Params = Dist::param_type;
    Dist urd;
    return angle_t{urd(g, Params{0.0f, detail::two_pi<float>})};
}

#if 0
constexpr auto sin(angle_t const x) noexcept -> double
{
    constexpr auto   sqrt_3_over_2 = 0.8660254037844386467637231707529361834714;
    constexpr double sin_table[12] = {
        0.0,            // sin(0 * pi / 6)
        0.5,            // sin(1 * pi / 6)
        sqrt_3_over_2,  // sin(2 * pi / 6)
        1.0,            // sin(3 * pi / 6)
        sqrt_3_over_2,  // sin(4 * pi / 6)
        0.5,            // sin(5 * pi / 6)
        0.0,            // sin(6 * pi / 6)
        -0.5,           // sin(7 * pi / 6)
        -sqrt_3_over_2, // sin(8 * pi / 6)
        -1.0,           // sin(9 * pi / 6)
        -sqrt_3_over_2, // sin(10 * pi / 6)
        -0.5,           // sin(11 * pi / 6)
    };
    return sin_table[static_cast<int>(x)];
}

constexpr auto cos(angle_t const x) noexcept -> double
{
    constexpr auto   sqrt_3_over_2 = 0.8660254037844386467637231707529361834714;
    constexpr double cos_table[12] = {
        1.0,            // cos(0 * pi / 6)
        sqrt_3_over_2,  // cos(1 * pi / 6)
        0.5,            // cos(2 * pi / 6)
        0.0,            // cos(3 * pi / 6)
        -0.5,           // cos(4 * pi / 6)
        -sqrt_3_over_2, // cos(5 * pi / 6)
        -1.0,           // cos(6 * pi / 6)
        -sqrt_3_over_2, // cos(7 * pi / 6)
        -0.5,           // cos(8 * pi / 6)
        0.0,            // cos(9 * pi / 6)
        0.5,            // cos(10 * pi / 6)
        sqrt_3_over_2,  // cos(11 * pi / 6)
    };
    return cos_table[static_cast<int>(x)];
}

namespace detail {
template <>
struct debug_print_fn<angle_t> {
    auto operator()(angle_t const x)
    {
        _debug_print("%i", static_cast<int>(x));
    }
};
} // namespace detail
#endif

TCM_NAMESPACE_END
