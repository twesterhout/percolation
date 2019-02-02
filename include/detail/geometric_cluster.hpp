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

#include "lattice.hpp"
#include "magnetic_cluster.hpp"

TCM_NAMESPACE_BEGIN

template <class System> class geometric_cluster_t { // {{{

  public:
    using system_type           = System;
    using magnetic_cluster_type = magnetic_cluster_t<system_type>;
    using magnetic_unique_ptr   = typename magnetic_cluster_type::unique_ptr;

  private:
    size_t              _root_index;   ///< Index of the root site
    size_t              _size;         ///< Number of sites in the cluster
    int                 _boundaries;   ///< Boundaries bitmask
    magnetic_unique_ptr _root;         ///< Root magnetic cluster
    system_type&        _system_state; ///< The system

  public:
    /// Creates a one-site geometric cluster consisting of a single magnetic
    /// cluster.
    ///
    /// This function automatically calls `on_*` functions of #tcm_system_t
    /// notifying it that a new cluster has been created.
    ///
    /// \param site       Index of the site.
    /// \param boundaries Which boundaries `site` touches. It it a bit-mask
    ///                   built by xor'ing values of type #tcm_bounrary_t.
    /// \param phase      Orientation of the spin. This value should probably be
    ///                   chosen randomly.
    /// \param system     Reference to the global system state.
    ///
    /// \note May throw if memory allocation fails.
    inline geometric_cluster_t(size_t site, int boundaries, angle_t phase,
                               system_type& system);

    geometric_cluster_t(geometric_cluster_t const&)     = delete;
    geometric_cluster_t(geometric_cluster_t&&) noexcept = delete;
    geometric_cluster_t& operator=(geometric_cluster_t const&) = delete;
    geometric_cluster_t& operator=(geometric_cluster_t&&) noexcept = delete;

    ~geometric_cluster_t() noexcept
    {
        _system_state.on_cluster_destroyed(*this);
    }

    constexpr auto size() const noexcept { return _size; }
    constexpr auto boundaries() const noexcept { return _boundaries; }
    constexpr auto root_index() const noexcept { return _root_index; }

  private:
    /// Inverts the tree such that `cluster` becomes the new root. If `cluster`
    /// is already root, nothing's done.
    ///
    /// \param cluster The new root. It must belong to `*this`.
    auto invert(magnetic_cluster_type& cluster) -> void;

  public:
    /// Merges `other` into `*this`. `edge` is the new edge connecting `*this`
    /// to `other` (i.e. `edge.first` must belong to `*this` and `edge.second`
    /// must belong to `other`).
    ///
    /// \note `other` is destroyed in this function! Trying to use the reference
    /// to it is undefined behaviour.
    inline auto merge(std::pair<size_t, size_t> edge,
                      geometric_cluster_t&      other) -> void;

    inline auto merge(no_optimize_t, std::pair<size_t, size_t> edge,
                      geometric_cluster_t& other) -> void;

    /// Connects `left` and `right` forming a cycle. All magnetic clusters
    /// belonging to this cycle are merged into one.
    ///
    /// `left` and `right` must belong to `*this`.
    auto connect(magnetic_cluster_type& left, magnetic_cluster_type& right)
        -> void;

    auto connect(no_optimize_t, magnetic_cluster_type& left,
                 magnetic_cluster_type& right) -> void;

    /// First, merges `other` into `*this` and then connects `*edge.first` and
    /// `*edge.second`.
    inline auto
    merge_and_connect(std::pair<gsl::not_null<magnetic_cluster_type*>,
                                gsl::not_null<magnetic_cluster_type*>>
                                           edge,
                      geometric_cluster_t& other) -> void;

    inline auto
    merge_and_connect(no_optimize_t,
                      std::pair<gsl::not_null<magnetic_cluster_type*>,
                                gsl::not_null<magnetic_cluster_type*>>
                                           edge,
                      geometric_cluster_t& other) -> void;

    auto optimize_full() -> void;

    auto rotate(angle_t const delta_angle) { _root->rotate(delta_angle); }
}; // }}}

// {{{ IMPLEMENTATION geometric_cluster_t
template <class System>
TCM_FORCEINLINE geometric_cluster_t<System>::geometric_cluster_t(
    size_t const site, int const boundaries, angle_t const phase,
    system_type& system_state)
    : _root_index{site}
    , _size{1}
    , _boundaries{boundaries}
    , _root{system_state.make_magnetic_cluster(site, phase, system_state)}
    , _system_state{system_state}
{
    _system_state.on_cluster_created(*this)
        .on_size_changed(*this)
        .on_boundaries_changed(*this);
}

// WARNING: !!! This invalidates the _root_index !!!
template <class System>
auto geometric_cluster_t<System>::invert(magnetic_cluster_type& new_root)
    -> void
{
    if (new_root.is_root()) { return; }

    // Constructs the full paths to the root.
    auto const find_path_to_root = [this](auto node) {
        std::stack<size_t, std::vector<size_t>> path;
        while (!node->is_root()) {
            path.push(node->index_in_parent());
            node = node->parent();
        }
        TCM_ASSERT(node->is_root(), "");
        TCM_ASSERT(node == _root.get(), "");
        return path;
    };
    auto path = find_path_to_root(
        gsl::not_null<magnetic_cluster_type*>{std::addressof(new_root)});
    while (!path.empty()) {
        _root = move_root_down<System>(std::move(_root), path.top());
        path.pop();
    }

    TCM_ASSERT(new_root.is_root(), "Post-condition failure");
    TCM_ASSERT(_root.get() == &new_root, "Post-condition failure!");
}

template <class System>
TCM_FORCEINLINE auto
geometric_cluster_t<System>::merge(std::pair<size_t, size_t> edge,
                                   geometric_cluster_t&      other) -> void
{
    auto const [this_site, other_site] = edge;
    auto& this_cluster  = _system_state.get_magnetic_cluster(this_site);
    auto& other_cluster = _system_state.get_magnetic_cluster(other_site);

    // Prepare the `other` cluster
    TCM_ASSERT(&_system_state.get_geometric_cluster(other_cluster) == &other,
               "");
    other.invert(other_cluster);
    auto const delta_angle = [system =
                                  std::cref(_system_state)](auto i, auto j) {
        auto const angle_i = system.get().get_angle(i);
        auto const angle_j = system.get().get_angle(j);
        switch (::TCM_NAMESPACE::interaction(system.get().lattice(), i, j)) {
        case interaction_t::Ferromagnetic: return angle_i - angle_j;
        case interaction_t::Antiferromagnetic:
            return angle_i - angle_j + angle_t{detail::pi<float>};
        } // end switch
    }(this_site, other_site);
    other.rotate(delta_angle);

    // Attach it to `*this`.
    this_cluster.attach({edge, std::move(other._root)});
    _size += other._size;
    // std::fprintf(stderr, "Merging {%zu, %zu}: boundaries was %i, became %i\n",
    //              edge.first, edge.second, _boundaries,
    //              _boundaries | other._boundaries);
    _boundaries |= other._boundaries;

    // Update global statistics
    _system_state.on_size_changed(*this)
        .on_boundaries_changed(*this)
        .on_cluster_merged(*this, other);
}

template <class System>
TCM_FORCEINLINE auto
geometric_cluster_t<System>::merge(no_optimize_t /*unused*/,
                                   std::pair<size_t, size_t> edge,
                                   geometric_cluster_t&      other) -> void
{
    auto const [this_site, other_site] = edge;
    auto& this_cluster  = _system_state.get_magnetic_cluster(this_site);
    auto& other_cluster = _system_state.get_magnetic_cluster(other_site);

    TCM_ASSERT(&_system_state.get_geometric_cluster(other_cluster) == &other,
               "");
    // Prepare the `other` cluster
    other.invert(other_cluster);

    // Attach it to `*this`.
    this_cluster.attach(no_optimize, {edge, std::move(other._root)});
    _size += other._size;
    // std::fprintf(
    //     stderr,
    //     "Merging (non-optimising) {%zu, %zu}: boundaries was %i, became %i\n",
    //     edge.first, edge.second, _boundaries, _boundaries | other._boundaries);
    _boundaries |= other._boundaries;

    // Update global statistics
    _system_state.on_size_changed(*this)
        .on_boundaries_changed(*this)
        .on_cluster_merged(*this, other);
}

template <class System>
TCM_FORCEINLINE auto geometric_cluster_t<System>::merge_and_connect(
    std::pair<gsl::not_null<magnetic_cluster_type*>,
              gsl::not_null<magnetic_cluster_type*>>
                         edge,
    geometric_cluster_t& other) -> void
{
    TCM_ASSERT(this != std::addressof(other),
               "Geometric clusters are already connected");

    auto& this_cluster  = *edge.first;
    auto& other_cluster = *edge.second;
    TCM_ASSERT(&_system_state.get_geometric_cluster(this_cluster) == this, "");
    TCM_ASSERT(&_system_state.get_geometric_cluster(other_cluster) == &other,
               "");
    other.invert(other_cluster);
    this_cluster.merge(std::move(other._root));
    _size += other._size;
    // std::fprintf(stderr,
    //              "Merging and connecting: boundaries was %i, became %i\n",
    //              _boundaries, _boundaries | other._boundaries);
    _boundaries |= other._boundaries;

    // Update global statistics
    _system_state.on_size_changed(*this)
        .on_boundaries_changed(*this)
        .on_cluster_merged(*this, other);
}

template <class System>
TCM_FORCEINLINE auto geometric_cluster_t<System>::merge_and_connect(
    no_optimize_t,
    std::pair<gsl::not_null<magnetic_cluster_type*>,
              gsl::not_null<magnetic_cluster_type*>>
                         edge,
    geometric_cluster_t& other) -> void
{
    TCM_ASSERT(this != std::addressof(other),
               "Geometric clusters are already connected");

    auto& this_cluster  = *edge.first;
    auto& other_cluster = *edge.second;
    TCM_ASSERT(&_system_state.get_geometric_cluster(this_cluster) == this, "");
    TCM_ASSERT(&_system_state.get_geometric_cluster(other_cluster) == &other,
               "");
    other.invert(other_cluster);
    this_cluster.merge(no_optimize, std::move(other._root));
    _size += other._size;
    // std::fprintf(stderr,
    //              "Merging and connecting (non-optimising): boundaries was %i, "
    //              "became %i\n",
    //              _boundaries, _boundaries | other._boundaries);
    _boundaries |= other._boundaries;

    // Update global statistics
    _system_state.on_size_changed(*this)
        .on_boundaries_changed(*this)
        .on_cluster_merged(*this, other);
}

namespace detail {
template <class System>
TCM_FORCEINLINE auto path_to_root(magnetic_cluster_t<System>& node)
    -> std::vector<gsl::not_null<magnetic_cluster_t<System>*>>
{
    using magnetic_cluster_type = magnetic_cluster_t<System>;
    auto p = gsl::not_null<magnetic_cluster_type*>{std::addressof(node)};
    std::vector<gsl::not_null<magnetic_cluster_type*>> path;
    while (!p->is_root()) {
        path.emplace_back(p);
        p = p->parent();
    }
    path.emplace_back(p);
    return path;
}

template <class System>
auto find_lca_impl(magnetic_cluster_t<System>& left,
                   magnetic_cluster_t<System>& right)
    -> std::tuple<gsl::not_null<magnetic_cluster_t<System>*>,
                  std::vector<gsl::not_null<magnetic_cluster_t<System>*>>,
                  std::vector<gsl::not_null<magnetic_cluster_t<System>*>>>
{
    TCM_ASSERT(std::addressof(left) != std::addressof(right),
               "`left` and `right` must be different");
    auto path_left  = path_to_root(left);
    auto path_right = path_to_root(right);
    TCM_ASSERT(path_left.front() == std::addressof(left),
               "Path to root must start with the node itself");
    TCM_ASSERT(path_right.front() == std::addressof(right),
               "Path to root must start with the node itself");
    TCM_ASSERT(path_left.back() == path_right.back(),
               "`left` and `right` must belong to the same cluster.");

    // Throws away the common parts
    magnetic_cluster_t<System>* common = nullptr;
    while (!path_left.empty() && !path_right.empty()
           && path_left.back() == path_right.back()) {
        common = path_left.back();
        path_left.pop_back();
        path_right.pop_back();
    }

    // Some sanity checks which compile to noop when optimisations are enabled
    TCM_ASSERT(common != nullptr, "Paths must have at least one common node");
    if (path_left.empty()) { // `left` is an ancestor of `right`
        TCM_ASSERT(common == std::addressof(left), "");
        TCM_ASSERT(path_right.size() > 0, "");
    }
    else if (path_right.empty()) { // `right` is an ancestor of `left`
        TCM_ASSERT(common == std::addressof(right), "");
        TCM_ASSERT(path_left.size() > 0, "");
    }
    else {
        TCM_ASSERT(path_left.size() > 0, "");
        TCM_ASSERT(path_right.size() > 0, "");
    }

    return {common, std::move(path_left), std::move(path_right)};
}
} // namespace detail

template <class System>
inline auto find_lca(magnetic_cluster_t<System>& left,
                     magnetic_cluster_t<System>& right)
    -> std::tuple<magnetic_cluster_t<System>*,
                  std::vector<typename magnetic_cluster_t<System>::unique_ptr>,
                  std::vector<typename magnetic_cluster_t<System>::unique_ptr>>
{
    auto [common, path_left, path_right] = detail::find_lca_impl(left, right);

    auto const to_owning_path = [common = common](auto& path) {
        using unique_ptr = typename magnetic_cluster_t<System>::unique_ptr;
        if (path.size() == 0) { return std::vector<unique_ptr>{}; }
        std::vector<unique_ptr> owning_path;
        owning_path.reserve(path.size());
        for (auto i = std::size_t{0}; i < path.size() - 1; ++i) {
            owning_path.emplace_back(
                path[i + 1]->detach(path[i]->index_in_parent()).data);
        }
        owning_path.emplace_back(
            common->detach(path.back()->index_in_parent()).data);
        TCM_ASSERT(owning_path.size() == path.size(),
                   "Post-condition violated");
        return owning_path;
    };

    auto owning_path_left  = to_owning_path(path_left);
    auto owning_path_right = to_owning_path(path_right);

    return {common, std::move(owning_path_left), std::move(owning_path_right)};
}

template <class System>
auto geometric_cluster_t<System>::connect(magnetic_cluster_type& left,
                                          magnetic_cluster_type& right) -> void
{
    auto [common, path_left, path_right] = find_lca(left, right);

    while (!path_left.empty()) {
        common->merge(no_optimize, std::move(path_left.back()));
        path_left.pop_back();
    }
    while (!path_right.empty()) {
        common->merge(no_optimize, std::move(path_right.back()));
        path_right.pop_back();
    }

    common->optimize_full();
}

template <class System>
auto geometric_cluster_t<System>::connect(no_optimize_t,
                                          magnetic_cluster_type& left,
                                          magnetic_cluster_type& right) -> void
{
    auto [common, path_left, path_right] = find_lca(left, right);

    while (!path_left.empty()) {
        common->merge(no_optimize, std::move(path_left.back()));
        path_left.pop_back();
    }
    while (!path_right.empty()) {
        common->merge(no_optimize, std::move(path_right.back()));
        path_right.pop_back();
    }
}

template <class System>
auto geometric_cluster_t<System>::optimize_full() -> void
{
    _root->dfs([](auto& x) { x.optimize_full(); });
}

// }}}

TCM_NAMESPACE_END
