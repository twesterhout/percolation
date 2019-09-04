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

// #include "lattice.hpp"
#include "magnetic_cluster.hpp"

TCM_NAMESPACE_BEGIN

class system_base_t;

class geometric_cluster_base_t { // {{{

  public:
    using node_type = magnetic_cluster_base_t::unique_ptr;

  private:
    uint32_t       _root_index;   ///< Index of the root site
    uint32_t       _size;         ///< Number of sites in the cluster
    int            _boundaries;   ///< Boundaries bitmask
    node_type      _root;         ///< Root magnetic cluster
    system_base_t& _system_state; ///< The system

  public:
    /// Creates a one-site geometric cluster.
    ///
    /// This function automatically calls `on_*` functions of #tcm_system_t
    /// notifying it that a new cluster has been created.
    ///
    /// \param site       Index of the site.
    /// \param boundaries Which boundaries `site` touches. It is a bit-mask
    ///                   built by xor'ing values of type #tcm_bounrary_t.
    /// \param phase      Orientation of the spin. This value is typically
    ///                   chosen randomly.
    /// \param system     Reference to the global system state.
    ///
    /// \throws std::bad_alloc If memory allocation fails.
    inline geometric_cluster_base_t(uint32_t site, int boundaries,
                                    angle_t phase, system_base_t& system);

    /// **Deleted** copy constructor.
    geometric_cluster_base_t(geometric_cluster_base_t const&) = delete;
    /// **Deleted** move constructor.
    geometric_cluster_base_t(geometric_cluster_base_t&&) = delete;
    /// **Deleted** copy assignment operator.
    auto operator                    =(geometric_cluster_base_t const&)
        -> geometric_cluster_base_t& = delete;
    /// **Deleted** move assignment operator.
    auto operator                    =(geometric_cluster_base_t &&)
        -> geometric_cluster_base_t& = delete;

    /// Destructor which tells the #_system_state that the number of geometric
    /// clusters should be decreased.
    inline ~geometric_cluster_base_t() noexcept;

    /// Returns the number of sites in this cluster.
    ///
    /// \noexcept
    [[nodiscard]] constexpr auto size() const noexcept -> uint32_t;

    /// Returns the boundaries bit-mask.
    ///
    /// \noexcept
    [[nodiscard]] constexpr auto boundaries() const noexcept -> int;

    /// Returns the index of the root site.
    ///
    /// \noexcept
    [[nodiscard]] constexpr auto root_index() const noexcept -> uint32_t;

  private:
    /// Inverts the tree such that \p cluster becomes the new root. If \p
    /// cluster is already root, nothing's done.
    ///
    /// \param new_root The new root. It must belong to `*this`.
    auto invert(magnetic_cluster_base_t& new_root) -> void;

  public:
    /// Merges \p other into `*this`.
    ///
    /// \param edge The new edge connecting `*this` to \p other (i.e.
    ///             `edge.first` must belong to `*this` and `edge.second` must
    ///             belong to \p other).
    /// \param other The cluster to merge into `*this`.
    ///
    /// \note **\p other is destroyed in this function!** Trying to use the
    /// reference to it is undefined behaviour.
    inline auto merge(std::pair<uint32_t, uint32_t> edge,
                      geometric_cluster_base_t&     other) -> void;

    inline auto
    merge(std::pair<magnetic_cluster_base_t&, magnetic_cluster_base_t&> edge,
          geometric_cluster_base_t& other) -> void;

    /// Connects \p left and \p right forming a cycle in the graph. All magnetic
    /// clusters belonging to this cycle are merged into one.
    ///
    /// \p left and \p right must belong to `*this`.
    inline auto form_cycle(magnetic_cluster_base_t& left,
                           magnetic_cluster_base_t& right) -> void;

    // auto rotate(angle_t const delta_angle) { _root->rotate(delta_angle); }
}; // }}}

#if 0
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
    /// Creates a one-site geometric cluster.
    ///
    /// This function automatically calls `on_*` functions of #tcm_system_t
    /// notifying it that a new cluster has been created.
    ///
    /// \param site       Index of the site.
    /// \param boundaries Which boundaries `site` touches. It is a bit-mask
    ///                   built by xor'ing values of type #tcm_bounrary_t.
    /// \param phase      Orientation of the spin. This value is typically
    ///                   chosen randomly.
    /// \param system     Reference to the global system state.
    ///
    /// \throws std::bad_alloc If memory allocation fails.
    inline geometric_cluster_t(size_t site, int boundaries, angle_t phase,
                               system_type& system);

    /// **Deleted** copy constructor.
    geometric_cluster_t(geometric_cluster_t const&) = delete;
    /// **Deleted** move constructor.
    geometric_cluster_t(geometric_cluster_t&&) noexcept = delete;
    /// **Deleted** copy assignment operator.
    geometric_cluster_t& operator=(geometric_cluster_t const&) = delete;
    /// **Deleted** move assignment operator.
    geometric_cluster_t& operator=(geometric_cluster_t&&) noexcept = delete;

    /// Destructor which tells the #_system_state that the number of geometric
    /// clusters should be decreased.
    ~geometric_cluster_t() noexcept
    {
        _system_state.on_cluster_destroyed(*this);
    }

    /// Returns the number of sites in this cluster.
    ///
    /// \noexcept
    constexpr auto size() const noexcept { return _size; }

    /// Returns the boundaries bit-mask.
    ///
    /// \noexcept
    constexpr auto boundaries() const noexcept { return _boundaries; }

    /// Returns the index of the root site.
    ///
    /// \noexcept
    constexpr auto root_index() const noexcept { return _root_index; }

  private:
    /// Inverts the tree such that \p cluster becomes the new root. If \p
    /// cluster is already root, nothing's done.
    ///
    /// \param cluster The new root. It must belong to `*this`.
    auto invert(magnetic_cluster_type& cluster) -> void;

  public:
    /// Merges \p other into `*this`.
    ///
    /// \param edge The new edge connecting `*this` to \p other (i.e.
    ///             `edge.first` must belong to `*this` and `edge.second` must
    ///             belong to \p other).
    /// \param other The cluster to merge into `*this`.
    ///
    /// \note **\p other is destroyed in this function!** Trying to use the
    /// reference to it is undefined behaviour.
    inline auto merge(std::pair<size_t, size_t> edge,
                      geometric_cluster_t&      other) -> void;

    // inline auto merge(no_optimize_t, std::pair<size_t, size_t> edge,
    //                   geometric_cluster_t& other) -> void;

    /// Connects \p left and \p right forming a cycle in the graph. All magnetic
    /// clusters belonging to this cycle are merged into one.
    ///
    /// \p left and \p right must belong to `*this`.
    auto connect(magnetic_cluster_type& left, magnetic_cluster_type& right)
        -> void;

    // auto connect(no_optimize_t, magnetic_cluster_type& left,
    //              magnetic_cluster_type& right) -> void;

    /// First, merges \p other into `*this` and then connects `*edge.first` and
    /// `*edge.second`.
    ///
    /// This function can be emulated by calling merge() followed by connect().
    /// However, here we know that `*edge.first` belongs to `*this` and
    /// `*edge.second` belongs to \p other which eliminates the costly
    /// find_lca() call in connect().
    inline auto
    merge_and_connect(std::pair<gsl::not_null<magnetic_cluster_type*>,
                                gsl::not_null<magnetic_cluster_type*>>
                                           edge,
                      geometric_cluster_t& other) -> void;

    // inline auto
    // merge_and_connect(no_optimize_t,
    //                   std::pair<gsl::not_null<magnetic_cluster_type*>,
    //                             gsl::not_null<magnetic_cluster_type*>>
    //                                        edge,
    //                   geometric_cluster_t& other) -> void;

    // auto optimize_full() -> void;

    auto rotate(angle_t const delta_angle) { _root->rotate(delta_angle); }
}; // }}}
#endif

#if 0
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
    auto const find_path_to_root =
        [this](gsl::not_null<magnetic_cluster_type const*> node) {
            using vector_type = boost::container::small_vector<size_t, 64>;
            std::stack<size_t, vector_type> path;
            while (!node->is_root()) {
                path.push(node->index_in_parent());
                node = node->parent();
            }
            TCM_ASSERT(node->is_root(), "");
            TCM_ASSERT(node == _root.get(), "");
            return path;
        };
    auto path = find_path_to_root({&new_root});
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
        // NOTE: This call destroys `other`
        .on_cluster_merged(*this, other);
}

#    if 0
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
#    endif

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

#    if 0
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
#    endif

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
        common->merge(std::move(path_left.back()));
        path_left.pop_back();
    }
    while (!path_right.empty()) {
        common->merge(std::move(path_right.back()));
        path_right.pop_back();
    }
}

#    if 0
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
#    endif

#    if 0
template <class System>
auto geometric_cluster_t<System>::optimize_full() -> void
{
    _root->dfs([](auto& x) { x.optimize_full(); });
}
#    endif

// }}}
#endif

TCM_NAMESPACE_END
