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

  protected:
    template <class T>
    using buffer_type =
        std::vector<T, boost::alignment::aligned_allocator<T, 64>>;

    buffer_type<particle_base_t>          _particles;
    buffer_type<magnetic_cluster_base_t*> _clusters;
    buffer_type<angle_t>                  _angles;

    uint32_t            _number_sites;
    uint32_t            _number_clusters;
    uint32_t            _max_cluster_size;
    std::array<bool, 3> _has_wrapped;

  public:
    TCM_NOINLINE inline system_base_t();

    system_base_t(system_base_t const&)     = delete;
    system_base_t(system_base_t&&) noexcept = default;
    auto operator=(system_base_t const&) -> system_base_t& = delete;
    auto operator=(system_base_t&&) noexcept -> system_base_t& = default;

    TCM_NOINLINE inline auto reset(size_t max_number_sites) -> void;

    // {{{ Callbacks
    /// This is a callback that should be invoked when the size of a cluster
    /// changes.
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
    constexpr auto on_cluster_created(magnetic_cluster_base_t&, uint32_t site,
                                      angle_t phase) noexcept -> system_base_t&;

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

    [[nodiscard]] /*constexpr*/ inline auto max_number_sites() const noexcept
        -> size_t;
    [[nodiscard]] constexpr auto number_sites() const noexcept -> uint32_t;
    [[nodiscard]] constexpr auto number_clusters() const noexcept -> uint32_t;
    [[nodiscard]] constexpr auto max_cluster_size() const noexcept -> uint32_t;
    [[nodiscard]] constexpr auto has_wrapped() const
        & noexcept -> gsl::span<bool const>;
    [[nodiscard]] constexpr auto is_empty(uint32_t site) const TCM_NOEXCEPT
        -> bool;
    [[nodiscard]] constexpr auto get_angle(uint32_t site) const TCM_NOEXCEPT
        -> angle_t;

    auto set_angle(uint32_t site, angle_t angle) TCM_NOEXCEPT -> system_base_t&;
#if 0
    template <class RAIter>
    auto set_angle(RAIter first, RAIter last, angle_t angle) TCM_NOEXCEPT
        -> system_base_t&;
#endif

    constexpr auto rotate(uint32_t site, angle_t angle) TCM_NOEXCEPT -> void;
#if 0
    template <class RAIter>
    auto rotate(RAIter first, RAIter last, angle_t angle) TCM_NOEXCEPT -> void;
#endif

    // magnetic clusters {{{
    [[nodiscard]] constexpr auto
    get_magnetic_cluster(uint32_t site) TCM_NOEXCEPT
        -> magnetic_cluster_base_t&;
    [[nodiscard]] constexpr auto
    get_magnetic_cluster(uint32_t site) const TCM_NOEXCEPT
        -> magnetic_cluster_base_t const&;

    constexpr auto
    set_magnetic_cluster(uint32_t                 site,
                         magnetic_cluster_base_t& cluster) TCM_NOEXCEPT
        -> system_base_t&;

    [[nodiscard]] auto
    get_geometric_cluster(magnetic_cluster_base_t const& cluster)
        -> geometric_cluster_base_t&;

    TCM_NOINLINE inline auto connect(uint32_t left, uint32_t right) -> void;

    TCM_NOINLINE inline auto merge(magnetic_cluster_base_t& left,
                                   magnetic_cluster_base_t& right) -> void;
};

system_base_t::system_base_t()
    : _particles{}
    , _clusters{}
    , _angles{}
    , _number_sites{0}
    , _number_clusters{0}
    , _max_cluster_size{0}
    , _has_wrapped{{false, false, false}}
{}

auto system_base_t::reset(size_t max_number_sites) -> void
{
    using std::begin, std::end;
    if (this->max_number_sites() != max_number_sites) {
        _particles.resize(max_number_sites);
        _clusters.resize(max_number_sites);
        _angles.resize(max_number_sites);
    }
    std::for_each(begin(_particles), end(_particles),
                  [](auto& x) { x = particle_base_t{}; });
    std::fill(begin(_clusters), end(_clusters), nullptr);
    std::fill(begin(_angles), end(_angles),
              angle_t{std::numeric_limits<float>::quiet_NaN()});
    _number_sites     = 0U;
    _number_clusters  = 0U;
    _max_cluster_size = 0U;
    std::fill(begin(_has_wrapped), end(_has_wrapped), false);
}

auto system_base_t::connect(uint32_t left, uint32_t right) -> void
{
    TCM_ASSERT(!_particles[left].is_empty() && !_particles[right].is_empty(),
               "only non-empty sites can be connected.");
    TCM_ASSERT(_clusters[left] != nullptr && _clusters[right] != nullptr, "");
    if (_clusters[left] == _clusters[right]) { return; }
    auto& left_root  = find_root(_particles[left]).cluster();
    auto& right_root = find_root(_particles[right]).cluster();
    // i and j belong to the same geometric cluster cluster
    if (std::addressof(left_root) == std::addressof(right_root)) {
        left_root.form_cycle(*_clusters[left], *_clusters[right]);
        return;
    }
    // i and j belong to different geometric clusters
    if (left_root.size() >= right_root.size()) {
        left_root.merge({left, right}, right_root);
    }
    else {
        right_root.merge({right, left}, left_root);
    }
}

auto system_base_t::merge(magnetic_cluster_base_t& left,
                          magnetic_cluster_base_t& right) -> void
{
    if (std::addressof(left) == std::addressof(right)) { return; }
    auto& left_root  = get_geometric_cluster(left);
    auto& right_root = get_geometric_cluster(right);
    if (std::addressof(left_root) == std::addressof(right_root)) {
        left_root.form_cycle(left, right);
        return;
    }
    if (left_root.size() >= right_root.size()) {
        left_root.merge({left, right}, right_root);
    }
    else {
        right_root.merge({right, left}, left_root);
    }
}

TCM_NAMESPACE_END
