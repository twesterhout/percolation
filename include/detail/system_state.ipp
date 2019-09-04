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

#include "system_state.hpp"

#include "lattice.hpp"

TCM_NAMESPACE_BEGIN

// Callbacks {{{
constexpr auto
system_base_t::on_size_changed(geometric_cluster_base_t const& cluster) noexcept
    -> system_base_t&
{
    auto const size = cluster.size();
    if (size > _max_cluster_size) { _max_cluster_size = size; }
    return *this;
}

constexpr auto system_base_t::on_size_changed(
    magnetic_cluster_base_t const& /*unused*/) noexcept -> system_base_t&
{
    return *this;
}

constexpr auto system_base_t::on_boundaries_changed(
    geometric_cluster_base_t const& cluster) noexcept -> system_base_t&
{
    detail::update_has_wrapped(_has_wrapped, cluster.boundaries());
    return *this;
}

constexpr auto system_base_t::on_cluster_created(
    geometric_cluster_base_t const& /*unused*/) noexcept -> system_base_t&
{
    ++_number_clusters;
    return *this;
}

constexpr auto
system_base_t::on_cluster_created(magnetic_cluster_base_t& cluster,
                                  uint32_t site, angle_t /*phase*/) noexcept
    -> system_base_t&
{
    TCM_ASSERT(_clusters[site] == nullptr, "cluster already exists");
    _clusters[site] = std::addressof(cluster);
    return *this;
}

constexpr auto system_base_t::on_cluster_destroyed(
    geometric_cluster_base_t const& /*cluster*/) TCM_NOEXCEPT -> system_base_t&
{
    TCM_ASSERT(_number_clusters > 0, "there are no clusters to destroy");
    --_number_clusters;
    return *this;
}

constexpr auto
system_base_t::on_cluster_merged(geometric_cluster_base_t& big,
                                 geometric_cluster_base_t& small) noexcept
    -> system_base_t&
{
    TCM_ASSERT(std::addressof(big) != std::addressof(small),
               "can't merge a cluster with itself");
    auto const big_root   = big.root_index();
    auto const small_root = small.root_index();
    _particles[small_root] =
        particle_base_t{std::addressof(_particles[big_root])};
    return *this;
}
// }}}

/*constexpr*/ auto system_base_t::max_number_sites() const noexcept -> size_t
{
    return _particles.size();
}

constexpr auto system_base_t::number_sites() const noexcept -> uint32_t
{
    return _number_sites;
}

constexpr auto system_base_t::number_clusters() const noexcept -> uint32_t
{
    return _number_clusters;
}

constexpr auto system_base_t::max_cluster_size() const noexcept -> uint32_t
{
    return _max_cluster_size;
}

constexpr auto system_base_t::has_wrapped() const
    & noexcept -> gsl::span<bool const>
{
    return _has_wrapped;
}

constexpr auto system_base_t::is_empty(uint32_t const site) const TCM_NOEXCEPT
    -> bool
{
    TCM_ASSERT(site < max_number_sites(), "index out of bounds");
    return _particles[site].is_empty();
}

constexpr auto system_base_t::get_angle(uint32_t const site) const TCM_NOEXCEPT
    -> angle_t
{
    TCM_ASSERT(!is_empty(site), "index out of bounds");
    return _angles[site];
}

auto system_base_t::set_angle(uint32_t const site,
                              angle_t const  angle) TCM_NOEXCEPT
    -> system_base_t&
{
    TCM_ASSERT(site < max_number_sites(), "index out of bounds");
    _angles[site] = angle;
    return *this;
}

#if 0
template <class RAIter>
auto system_base_t::set_angle(RAIter first, RAIter last,
                              angle_t angle) TCM_NOEXCEPT -> system_base_t&
{
    std::for_each(first, last,
                  [angle, this](auto const site) { set_angle(site, angle); });
    return *this;
}
#endif

constexpr auto system_base_t::rotate(uint32_t const site,
                                     angle_t const  angle) TCM_NOEXCEPT -> void
{
    TCM_ASSERT(!is_empty(site), "site is empty");
    set_angle(site, get_angle(site) + angle);
}

#if 0
template <class RAIter>
auto rotate(RAIter first, RAIter last, angle_t angle) TCM_NOEXCEPT -> void;
#endif

constexpr auto
system_base_t::get_magnetic_cluster(uint32_t const site) TCM_NOEXCEPT
    -> magnetic_cluster_base_t&
{
    TCM_ASSERT(!is_empty(site), "site is empty");
    TCM_ASSERT(_clusters[site] != nullptr, "");
    return *_clusters[site];
}

constexpr auto
system_base_t::get_magnetic_cluster(uint32_t site) const TCM_NOEXCEPT
    -> magnetic_cluster_base_t const&
{
    TCM_ASSERT(!is_empty(site), "site is empty");
    TCM_ASSERT(_clusters[site] != nullptr, "");
    return *_clusters[site];
}

constexpr auto system_base_t::set_magnetic_cluster(
    uint32_t site, magnetic_cluster_base_t& cluster) TCM_NOEXCEPT
    -> system_base_t&
{
    TCM_ASSERT(site < max_number_sites(), "index out of bounds");
    // TCM_ASSERT(!_particles[i].is_empty(), "Site is empty");
    _clusters[site] = std::addressof(cluster);
    return *this;
}

auto system_base_t::get_geometric_cluster(
    magnetic_cluster_base_t const& cluster) -> geometric_cluster_base_t&
{
    TCM_ASSERT(cluster.number_sites() > 0, "magnetic cluster can't be empty");
    return find_root(_particles[cluster.sites()[0]]).cluster();
}

TCM_NAMESPACE_END
