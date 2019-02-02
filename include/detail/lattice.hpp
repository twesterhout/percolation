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
#include <cstddef> // size_t
#include <cstdint> // std::int64_t
#include <cstdlib> // std::div
#include <type_traits>

TCM_NAMESPACE_BEGIN

enum class interaction_t {
    Ferromagnetic,
    Antiferromagnetic,
};

enum class sublattice_t {
    A = 0,
    B = 1,
    C = 2,
};

inline auto update_has_wrapped(bool (&has_wrapped)[3],
                               int const boundaries) noexcept -> void
{
    constexpr auto x_mask = TCM_BOUNDARY_X_LOW | TCM_BOUNDARY_X_HIGH;
    constexpr auto y_mask = TCM_BOUNDARY_Y_LOW | TCM_BOUNDARY_Y_HIGH;
    constexpr auto z_mask = TCM_BOUNDARY_Z_LOW | TCM_BOUNDARY_Z_HIGH;
    if ((boundaries & x_mask) == x_mask) { has_wrapped[0] = true; }
    if ((boundaries & y_mask) == y_mask) { has_wrapped[1] = true; }
    if ((boundaries & z_mask) == z_mask) { has_wrapped[2] = true; }
}

template <class Lattice>
inline constexpr auto size(Lattice const& lattice) noexcept -> size_t
{
    TCM_ASSERT(lattice.size >= 0, "Lattice is in an invalid state");
    return static_cast<size_t>(lattice.size);
}

inline constexpr auto sublattice(tcm_square_lattice_t const& /*lattice*/,
                                 size_t /*site index*/) noexcept -> sublattice_t
{
    return sublattice_t::A;
}

inline constexpr auto sublattice(tcm_cubic_lattice_t const& /*lattice*/,
                                 size_t /*site index*/) noexcept -> sublattice_t
{
    return sublattice_t::A;
}

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

template <class T, std::size_t N>
struct array_size<T[N]> : std::integral_constant<std::size_t, N> {};

template <class T, std::size_t N>
struct array_size<T (&)[N]> : array_size<T[N]> {};
} // namespace detail

template <class Lattice> constexpr auto max_neighbours() noexcept -> size_t
{
    using neighbours_type = std::remove_reference_t<decltype(
        std::declval<Lattice>().neighbours[std::declval<int>()])>;
    return detail::array_size<neighbours_type>::value;
}

template <class Lattice>
inline constexpr auto neighbours(Lattice const& lattice, size_t const site)
{
    TCM_ASSERT(site < size(lattice), "Index out of bounds");
    return gsl::span<std::int64_t>{lattice.neighbours[site]};
}

inline constexpr auto interaction(tcm_square_lattice_t const& /*lattice*/,
                                  size_t /*i*/, size_t /*j*/) noexcept
    -> interaction_t
{
    return interaction_t::Ferromagnetic;
}

inline constexpr auto interaction(tcm_cubic_lattice_t const& /*lattice*/,
                                  size_t /*i*/, size_t /*j*/) noexcept
    -> interaction_t
{
    return interaction_t::Ferromagnetic;
}

inline constexpr auto interaction(tcm_triangular_lattice_t const& /*lattice*/,
                                  size_t /*i*/, size_t /*j*/) noexcept
    -> interaction_t
{
    return interaction_t::Antiferromagnetic;
}

TCM_NAMESPACE_END
