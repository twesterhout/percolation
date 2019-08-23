// Copyright (c) 2019, Tom Westerhout
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

#include "geometric_cluster.hpp"
#include "magnetic_cluster.hpp"
#include "particle.hpp"

#include <boost/align/aligned_allocator.hpp>

#include <array>

TCM_NAMESPACE_BEGIN

class system_base_t {

    template <class T>
    using buffer_type =
        std::vector<T, boost::alignment::aligned_allocator<T, 64>>;

    buffer_type<particle_base_t>          _particles;
    buffer_type<magnetic_cluster_base_t*> _clusters;
    buffer_type<angle_t>                  _angles;
    buffer_type<float>                    _S_x;
    buffer_type<float>                    _S_y;

    uint32_t            _number_sites;
    uint32_t            _number_clusters;
    uint32_t            _max_cluster_size;
    std::array<bool, 3> _has_wrapped[3];

  public:
    system_base_t(system_base_t const&) = delete;
    system_base_t(system_base_t&&)      = delete;
    auto operator=(system_base_t const&) -> system_base_t& = delete;
    auto operator=(system_base_t &&) -> system_base_t& = delete;

    // {{{ Callbacks
    /// This is a callback that should be invoked when the size of a geometric
    /// cluster changes.
    ///
    /// Helps keep track of the #_max_cluster_size property.
    constexpr auto on_size_changed(geometric_cluster_base_t const&) noexcept
        -> system_base_t&;
    constexpr auto on_size_changed(magnetic_cluster_base_t const&) noexcept
        -> system_base_t&;

    /// This is a callback that should be invoked when the "boundaries" of a
    /// geometric cluster change.
    ///
    /// Helps keep track of the #_has_wrapped property.
    constexpr auto
    on_boundaries_changed(geometric_cluster_base_t const&) noexcept
        -> system_base_t&;

    /// This is a callback that should be invoked when a new geometric cluster
    /// is created.
    ///
    /// Helps keep track of the #_number_clusters property.
    constexpr auto on_cluster_created(geometric_cluster_base_t const&) noexcept
        -> system_base_t&;
    constexpr auto on_cluster_created(magnetic_cluster_base_t const&,
                                      uint32_t site, angle_t phase) noexcept
        -> system_base_t&;

    /// This is a callback that should be invoked when a geometric cluster is
    /// destroyed (i.e. merged into another).
    ///
    /// Helps keep track of the #_number_clusters property
    constexpr auto
    on_cluster_destroyed(geometric_cluster_base_t const&) noexcept
        -> system_base_t&;

    /// This is a callback that should be called when \p small is merged into \p
    /// big.
    ///
    /// Helps keep track of the #_particles.
    constexpr auto on_cluster_merged(geometric_cluster_base_t& big,
                                     geometric_cluster_base_t& small) noexcept
        -> system_base_t&;
    // }}}

    // constexpr auto optimizing() const noexcept -> bool { return _optimizing; }

    // TCM_NOINLINE auto optimizing(bool do_opt) -> void;

    constexpr auto max_number_sites() const noexcept -> size_t;
    constexpr auto number_sites() const noexcept -> size_t;
    constexpr auto number_clusters() const noexcept -> size_t;
    constexpr auto max_cluster_size() const noexcept -> size_t;
    constexpr auto has_wrapped() const & noexcept -> gsl::span<bool const>;

    constexpr auto is_empty(uint32_t site) const TCM_NOEXCEPT -> bool;
    constexpr auto get_angle(uint32_t site) const TCM_NOEXCEPT -> angle_t;
    constexpr auto S_x(uint32_t site) const TCM_NOEXCEPT -> float;
    constexpr auto S_y(uint32_t site) const TCM_NOEXCEPT -> float;

    auto set_angle(uint32_t site, angle_t angle) TCM_NOEXCEPT -> system_base_t&;
    template <class RAIter>
    auto set_angle(RAIter first, RAIter last, angle_t angle) TCM_NOEXCEPT
        -> system_base_t&;

    constexpr auto rotate(uint32_t i, angle_t angle) TCM_NOEXCEPT -> void;
    template <class RAIter>
    auto rotate(RAIter first, RAIter last, angle_t angle) TCM_NOEXCEPT -> void;

    // magnetic clusters {{{
    constexpr auto get_magnetic_cluster(uint32_t site) TCM_NOEXCEPT
        -> magnetic_cluster_base_t&;
    constexpr auto get_magnetic_cluster(uint32_t site) const TCM_NOEXCEPT
        -> magnetic_cluster_base_t const&;

    constexpr auto
    set_magnetic_cluster(uint32_t                 site,
                         magnetic_cluster_base_t& cluster) TCM_NOEXCEPT
        -> system_base_t&;

    auto get_geometric_cluster(magnetic_cluster_base_t const& cluster)
        -> geometric_cluster_base_t&;
};

TCM_NAMESPACE_END
