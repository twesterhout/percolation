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

#include "lattice.h"
#include <gsl/gsl-lite.hpp>
#include <cmath>
#include <cstdint>
#include <cstdlib> // std::div
#include <type_traits>

TCM_NAMESPACE_BEGIN

/// Types of interactions two neighbouring sites can have.
enum class interaction_t {
    Ferromagnetic     = -1,
    Antiferromagnetic = 1,
};

/// Different sublattices.
enum class sublattice_t {
    A = 0,
    B = 1,
    C = 2,
};

/// Utility function that updates \p has_wrapped given additional boundaries \p
/// boundaries.
///
/// \noexcept
constexpr auto update_has_wrapped(bool (&has_wrapped)[3],
                                  int const boundaries) noexcept -> void
{
    constexpr auto x_mask = TCM_BOUNDARY_X_LOW | TCM_BOUNDARY_X_HIGH;
    constexpr auto y_mask = TCM_BOUNDARY_Y_LOW | TCM_BOUNDARY_Y_HIGH;
    constexpr auto z_mask = TCM_BOUNDARY_Z_LOW | TCM_BOUNDARY_Z_HIGH;
    if ((boundaries & x_mask) == x_mask) { has_wrapped[0] = true; }
    if ((boundaries & y_mask) == y_mask) { has_wrapped[1] = true; }
    if ((boundaries & z_mask) == z_mask) { has_wrapped[2] = true; }
}

namespace detail {
constexpr auto update_has_wrapped(std::array<bool, 3>& has_wrapped,
                                  int const boundaries) noexcept -> void
{
    constexpr auto x_mask = TCM_BOUNDARY_X_LOW | TCM_BOUNDARY_X_HIGH;
    constexpr auto y_mask = TCM_BOUNDARY_Y_LOW | TCM_BOUNDARY_Y_HIGH;
    constexpr auto z_mask = TCM_BOUNDARY_Z_LOW | TCM_BOUNDARY_Z_HIGH;
    if ((boundaries & x_mask) == x_mask) { has_wrapped[0] = true; }
    if ((boundaries & y_mask) == y_mask) { has_wrapped[1] = true; }
    if ((boundaries & z_mask) == z_mask) { has_wrapped[2] = true; }
}
} // namespace detail

/// Returns the number of sites in the \p lattice.
///
/// \noexcept
template <class Lattice>
constexpr auto size(Lattice const& lattice) noexcept -> size_t
{
    TCM_ASSERT(lattice.size >= 0, "Lattice is in an invalid state");
    return static_cast<size_t>(lattice.size);
}

/// Given a lattice and a site index, returns the sublattice to which the site
/// belongs.
constexpr auto sublattice(tcm_square_lattice_t const& /*lattice*/,
                          size_t /*site index*/) noexcept -> sublattice_t
{
    return sublattice_t::A;
}

/// \overload
constexpr auto sublattice(tcm_cubic_lattice_t const& /*lattice*/,
                          size_t /*site index*/) noexcept -> sublattice_t
{
    return sublattice_t::A;
}

/// \overload
inline auto sublattice(tcm_triangular_lattice_t const& lattice,
                       size_t const site) noexcept -> sublattice_t
{
    TCM_ASSERT(site < size(lattice), "Index out of bounds");
    auto const i      = static_cast<std::int64_t>(site);
    auto const [y, x] = std::div(i, lattice.length);
    auto const n      = (y % 2) - x;
    if (n >= 0) return static_cast<sublattice_t>(n);
    return static_cast<sublattice_t>((n + 3 * (-n / 3 + 1)) % 3);
}

namespace detail {
template <class T> struct array_size;

template <class T, size_t N>
struct array_size<T[N]> : std::integral_constant<size_t, N> {};

template <class T, size_t N> struct array_size<T (&)[N]> : array_size<T[N]> {};
} // namespace detail

/// Returns the maximal number of neighbours a site on a lattice of type \p
/// Lattice can have.
template <class Lattice> constexpr auto max_neighbours() noexcept -> size_t
{
    using neighbours_type = std::remove_reference_t<decltype(
        std::declval<Lattice>().neighbours[std::declval<int>()])>;
    return detail::array_size<neighbours_type>::value;
}

/// Returns a non-owning view of the neighbours of \p site on the \p lattice.
template <class Lattice>
constexpr auto neighbours(Lattice const& lattice, size_t const site)
    -> gsl::span<int64_t const>
{
    TCM_ASSERT(site < size(lattice), "Index out of bounds");
    return {lattice.neighbours[site]};
}

/// Given a lattice and two neighbouring sites belonging to it, returns the type
/// of interaction between the sites.
constexpr auto interaction(tcm_square_lattice_t const& /*lattice*/,
                           size_t /*i*/, size_t /*j*/) noexcept -> interaction_t
{
    return interaction_t::Ferromagnetic;
}

constexpr auto interaction(tcm_cubic_lattice_t const& /*lattice*/, size_t /*i*/,
                           size_t /*j*/) noexcept -> interaction_t
{
    return interaction_t::Ferromagnetic;
}

constexpr auto interaction(tcm_triangular_lattice_t const& /*lattice*/,
                           size_t /*i*/, size_t /*j*/) noexcept -> interaction_t
{
    return interaction_t::Antiferromagnetic;
}

template <class Lattice>
constexpr auto coupling(Lattice const& lattice, size_t const i,
                        size_t const j) noexcept -> float
{
    return static_cast<float>(interaction(lattice, i, j));
}

/// Applies a function to each elementary upward triangle.
///
/// Suppose that we have the following `lattice`
///
///   8   9  10  11
///     4   5   6   7
///   0   1   2   3
///
/// Then an invocation of `for_each_plaquette(lattice, func)` is equivalent to
///
///     func(0, 1, 4);
///     func(1, 2, 5);
///     func(2, 3, 6);
///     func(4, 5, 9);
///     func(5, 6, 10);
///     func(6, 7, 11);
///
template <class Function>
constexpr auto for_each_plaquette(
    tcm_triangular_lattice_t const& lattice,
    Function func) noexcept(std::is_nothrow_invocable_v<Function, size_t,
                                                        size_t, size_t>)
{
    if (lattice.length_y < 2 || lattice.length < 2) { return; }
    auto i = size_t{0};
    for (int64_t y = 0; y < lattice.length_y - 1; ++y) {
        for (int64_t x = 0; x < lattice.length - 1; ++x, ++i) {
            func(i, i + 1, i + static_cast<size_t>(lattice.length));
        }
        ++i;
    }
    TCM_ASSERT(i == lattice.size, "");
}

template <class Lattice, class Predicate>
constexpr auto root_mean_square_chirality(Lattice const& lattice,
                                          float const* __restrict__ S_x,
                                          float const* __restrict__ S_y,
                                          Predicate exists) -> float
{
    // constexpr auto scale = 0.38490018f;
    constexpr auto scale = 0.3849001794597505;
    struct Accumulator {
        float const*     S_x;
        float const*     S_y;
        Predicate const& exists;
        double           chirality;
        size_t           count;

        constexpr auto operator()(size_t const i, size_t const j,
                                  size_t const k) -> void
        {
            ++count;
            if (exists(i) && exists(j) && exists(k)) {
                auto const k_p = edge(i, j) + edge(j, k) + edge(k, i);
                chirality += k_p * k_p;
            }
        }

        constexpr auto edge(size_t const i, size_t const j) const noexcept
        {
            return static_cast<double>(S_x[i] * S_y[j] - S_y[i] * S_x[j]);
        }
    } accumulator{S_x, S_y, exists, 0.0, 0};

    for_each_plaquette(lattice, std::ref(accumulator));
    return static_cast<float>(scale * std::sqrt(accumulator.chirality)
                              / static_cast<double>(accumulator.count));
}

TCM_NAMESPACE_END
