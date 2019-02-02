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

#include "detail/lattice.hpp"
#include "detail/thermalisation.hpp"
#include "detail/utility.hpp"
#include <boost/container/small_vector.hpp>
// #include <boost/pool/pool_alloc.hpp>
#include <gsl/gsl-lite.hpp>
#include <sleef.h>
#include <memory>
#include <optional>
#include <stack>
#include <utility>
#include <vector>

#if !defined(__AVX__)
#    error "WTF?"
#endif

TCM_NAMESPACE_BEGIN

using size_t = std::size_t;

/// A tag indicating that a function should not optimise the energy.
struct no_optimize_t {};
inline constexpr no_optimize_t no_optimize{};

/// A magnetic cluster, i.e. an irreducible component in our graph.
template <class System> class magnetic_cluster_t { // {{{
  public:
    using system_type = System;
    using unique_ptr =
        typename System::template unique_ptr<magnetic_cluster_t<System>>;

  private:
    /// Information about the connection between this cluster and a child
    /// cluster.
    struct child_conn_info_t {
        /// The edge which connects this cluster to the child.
        std::pair<size_t, size_t> edge;
        /// The child cluster itself. Yes, parents own their children.
        unique_ptr data;
    };

    /// Information about the connection between this cluster and its parent.
    struct parent_conn_info_t {
        /// Index in the `_children` array of the parent.
        size_t index;
        /// The edge which connects the parent to this cluster.
        std::pair<size_t, size_t> edge;
        /// The parent owns this cluster so it's safe to store a non-owning pointer.
        gsl::not_null<magnetic_cluster_t*> data;
    };

    std::optional<parent_conn_info_t> _parent;
    // std::vector<child_conn_info_t>    _children;
    boost::container::small_vector<child_conn_info_t, 1> _children;
    /// A list of indices of _all_ the sites that belong to this cluster.
    std::vector<size_t> _sites;
    /// A reference to the global system state
    system_type& _system_state;
    bool         _is_optimised;

    static_assert(sizeof(boost::container::small_vector<child_conn_info_t, 1>)
                  == 56);

  public:
    inline magnetic_cluster_t(size_t /*site index*/, angle_t /*spin direction*/,
                              system_type& /*global system state*/);

    magnetic_cluster_t(magnetic_cluster_t const&) = delete;
    magnetic_cluster_t(magnetic_cluster_t&&)      = delete;
    magnetic_cluster_t& operator=(magnetic_cluster_t const&) = delete;
    magnetic_cluster_t& operator=(magnetic_cluster_t&&) = delete;

    inline auto attach(child_conn_info_t /*connection*/) -> void;
    inline auto attach(no_optimize_t, child_conn_info_t /*connection*/) -> void;

    inline auto detach(size_t /*child index*/) -> child_conn_info_t;

    inline auto insert(no_optimize_t, size_t /*site index*/) -> void;
    inline auto insert(size_t /*site index*/) -> void;

    inline auto merge(unique_ptr cluster) -> void;
    auto        merge(no_optimize_t, unique_ptr cluster) -> void;

    inline auto rotate(angle_t /*angle by which to rotate*/) -> void;

    /// Optimizes the energy of the magnetic cluster assuming that the cluster
    /// was optimized and then a new site `site` was added.
    auto optimize_one(size_t site) -> void;

    /// Optimizes the energy of the magnetic cluster.
    TCM_NOINLINE auto optimize_full() -> void;

    constexpr auto is_root() const noexcept { return !_parent.has_value(); }

    constexpr auto index_in_parent() const TCM_NOEXCEPT
    {
        TCM_ASSERT(!is_root(), "Root nodes are orphans");
        TCM_ASSERT(_parent->index < _parent->data->_children.size(),
                   "Index out of bounds");
        TCM_ASSERT(_parent->data->_children[_parent->index].data.get() == this,
                   "Bug! Invalid index in parent");
        return _parent->index;
    }

    constexpr auto parent() const TCM_NOEXCEPT
    {
        TCM_ASSERT(!is_root(), "Root nodes are orphans");
        return _parent->data;
    }

    constexpr auto sites() const noexcept -> gsl::span<size_t const>
    {
        return {_sites};
    }

    constexpr auto is_optimized() const noexcept -> bool
    {
        return _is_optimised;
    }

    auto number_sites() const noexcept -> size_t { return _sites.size(); }
    auto number_children() const noexcept -> size_t { return _children.size(); }

    template <class S>
    friend inline auto
        move_root_down(typename magnetic_cluster_t<S>::unique_ptr /*root*/,
                       size_t /*child index*/) ->
        typename magnetic_cluster_t<S>::unique_ptr;

    /// Runs a Depth-First Search
    template <class Function> TCM_FORCEINLINE auto dfs(Function&& fn)
    {
        using container_type =
            boost::container::small_vector<magnetic_cluster_t*, 10>;
        using stack_type = std::stack<magnetic_cluster_t*, container_type>;
        static_assert(sizeof(stack_type) == sizeof(container_type),
                      "std::stack is wasting memory");

        stack_type todo;
        todo.push(this);
        while (!todo.empty()) {
            auto& x = *todo.top();
            todo.pop();
            for (auto& child : x._children) {
                todo.push(child.data.get());
            }
            fn(x);
        }
    }

  private:
    auto align_with_parent() -> void
    {
        TCM_ASSERT(_parent.has_value(), "Can't align an orphan");
        auto const delta_angle = [this]() {
            auto const [i, j] = _parent->edge;
            switch (
                ::TCM_NAMESPACE::interaction(_system_state.lattice(), i, j)) {
            case interaction_t::Ferromagnetic:
                return _system_state.get_angle(i) - _system_state.get_angle(j);
            case interaction_t::Antiferromagnetic:
                return _system_state.get_angle(i) - _system_state.get_angle(j)
                       + angle_t{detail::pi<float>};
            } // end switch
        }();
        rotate(delta_angle);
    }

    auto thermalise(sa_pars_t const& parameters) -> gsl::span<float>;

    /*
    auto check_index_in_parent() const -> void
    {
        if (!is_root()) {
            TCM_ASSERT(_parent->data->_children.at(_parent->index).data.get()
                           == this,
                       "");
        }
        for (auto i = size_t{0}; i < _children.size(); ++i) {
            TCM_ASSERT(_children[i].data->index_in_parent() == i, "");
        }
    }
    */
}; // }}}

// {{{ IMPLEMENTATION: magnetic_cluster_t
template <class System>
TCM_FORCEINLINE magnetic_cluster_t<System>::magnetic_cluster_t(
    size_t const site, angle_t const phase, System& system_state)
    : _parent{std::nullopt}
    , _children{}
    // TODO(twesterhout): Should we reserve some memory in advance?
    , _sites({site})
    , _system_state{system_state}
    , _is_optimised{true}
{
    _system_state.set_angle(site, phase);
    _system_state.set_magnetic_cluster(site, *this);
}

template <class System>
TCM_FORCEINLINE auto magnetic_cluster_t<System>::insert(no_optimize_t,
                                                        size_t const site)
    -> void
{
    _sites.push_back(site);
    _system_state.set_magnetic_cluster(site, *this);
    _is_optimised = false;
}

template <class System>
TCM_FORCEINLINE auto magnetic_cluster_t<System>::insert(size_t const site)
    -> void
{
    TCM_ASSERT(_is_optimised, "It is assumed that the cluster is optimized");
    insert(no_optimize, site);
    optimize_one(site);
    _is_optimised = true;
}

template <class System>
TCM_FORCEINLINE auto magnetic_cluster_t<System>::attach(child_conn_info_t child)
    -> void
{
    TCM_ASSERT(_is_optimised, "It is assumed that the cluster is optimized");
    attach(no_optimize, std::move(child));
    _children.back().data->align_with_parent();
    _is_optimised = true;
}

template <class System>
TCM_FORCEINLINE auto
magnetic_cluster_t<System>::attach(no_optimize_t /*unused*/,
                                   child_conn_info_t child) -> void
{
    using std::begin, std::end;
    TCM_ASSERT(child.data != nullptr, "Can't attach a non-existent cluster");
    TCM_ASSERT(!child.data->_parent.has_value(), "Cluster must be an orphan");
    TCM_ASSERT(std::addressof(child.data->_system_state)
                   == std::addressof(_system_state),
               "Cluster must belong to the same system");

    TCM_ASSERT(std::count(begin(child.data->_sites), end(child.data->_sites),
                          child.edge.second)
                   == 1,
               "Invalid child_conn_info_t");
    TCM_ASSERT(std::count(begin(_sites), end(_sites), child.edge.first) == 1,
               "Invalid child_conn_info_t");

    // Attach the child
    child.data->_parent = {_children.size(), child.edge,
                           gsl::not_null<magnetic_cluster_t*>{this}};
    _children.emplace_back(std::move(child));
    _is_optimised = false;
}

template <class System>
auto magnetic_cluster_t<System>::merge(no_optimize_t /*unused*/,
                                       unique_ptr cluster) -> void
{
    using std::begin, std::end;
    TCM_ASSERT(cluster != nullptr, "Can't merge with a non-existant cluster");
    TCM_ASSERT(!cluster->_parent.has_value(), "Cluster must be an orphan");
    TCM_ASSERT(std::addressof(cluster->_system_state)
                   == std::addressof(_system_state),
               "Cluster must belong to the same system");

    // Steal children from `cluster`. For each child we have to update its
    // `parent_conn_info_t.data` to point to this. After that, the child can be
    // safely moved into `_children`.
    _children.reserve(_children.size() + cluster->_children.size());
    for (auto& conn : cluster->_children) {
        conn.data->_parent->index = _children.size();
        conn.data->_parent->data  = this;
        _children.emplace_back(std::move(conn));
    }
    // cluster->_children.clear();

    // Now we steal `cluster->_sites`. Each site now belongs to a different
    // magnetic cluster. This info has to be propagated to the `_system_state`.
    for (auto const site : cluster->_sites) {
        _system_state.set_magnetic_cluster(site, *this);
    }
    _sites.reserve(_sites.size() + cluster->_sites.size());
    _sites.insert(end(_sites), begin(cluster->_sites), end(cluster->_sites));
    _is_optimised = false;
}

template <class System>
TCM_FORCEINLINE auto magnetic_cluster_t<System>::merge(unique_ptr cluster)
    -> void
{
    merge(no_optimize, std::move(cluster));
    optimize_full();
}

template <class System>
auto magnetic_cluster_t<System>::rotate(angle_t const angle) -> void
{
    dfs([angle, this](auto& x) {
        using std::begin, std::end;
        _system_state.rotate(begin(x._sites), end(x._sites), angle);
    });
}

template <class System>
TCM_FORCEINLINE auto
magnetic_cluster_t<System>::detach(size_t const child_index)
    -> child_conn_info_t
{
    using std::swap;
    TCM_ASSERT(child_index < _children.size(), "Index out of bounds");

    if (child_index != _children.size() - 1) {
        _children.back().data->_parent->index = child_index;
        swap(_children[child_index], _children.back());
    }
    auto conn = std::move(_children.back());
    _children.pop_back();
    conn.data->_parent = std::nullopt;
    return conn;
}

template <class System>
auto magnetic_cluster_t<System>::thermalise(sa_pars_t const& parameters)
    -> gsl::span<float>
{
    auto& sa_buffers = _system_state.sa_buffers();
    sa_buffers.resize(_sites.size());

    auto& energy_buffers = _system_state.energy_buffers();
    auto energy_fn = energy_buffers.energy_fn(sites(), _system_state.lattice());

    return optimise(std::move(energy_fn), parameters, sa_buffers,
                    _system_state.rng_stream(),
                    [this](auto initial) {
                        using std::begin, std::end;
                        std::transform(begin(_sites), end(_sites),
                                       begin(initial), [this](auto const i) {
                                           return static_cast<float>(
                                               _system_state.get_angle(i));
                                       });
                    })
        .buffer;
}

template <class System>
auto magnetic_cluster_t<System>::optimize_one(size_t const /*site*/) -> void
{
    using std::begin, std::end;
    sa_pars_t params{/*q_V = */ 2.62f, /*q_A = */ -1.0f, /*t_0 = */ 10.0f,
                     /*n = */ 1000u};

    auto const angles = thermalise(params);

    if (_parent.has_value()) {
        auto const delta_angle = [this, &angles]() {
            auto const [i, j] = _parent->edge;
            auto const local_j =
                _system_state.energy_buffers().global_to_local(j);
            switch (interaction(_system_state.lattice(), i, j)) {
            case interaction_t::Ferromagnetic:
                return _system_state.get_angle(i) - angle_t{angles[local_j]};
            case interaction_t::Antiferromagnetic:
                return _system_state.get_angle(i) - angle_t{angles[local_j]}
                       + angle_t{static_cast<float>(M_PI)};
            } // end switch
        }();

        std::transform(begin(angles), end(angles), begin(angles),
                       [delta_angle](auto const x) {
                           return static_cast<float>(angle_t{x} + delta_angle);
                       });
    }

    for (auto i = size_t{0}; i < _sites.size(); ++i) {
        _system_state.set_angle(_sites[i], angle_t{angles[i]});
    }
    for (auto& child_conn : _children) {
        child_conn.data->align_with_parent();
    }
    _is_optimised = true;
}

template <class System>
TCM_NOINLINE auto magnetic_cluster_t<System>::optimize_full() -> void
{
    using std::begin, std::end;
    if (_parent.has_value()) {
        auto const fixed_index = _parent->edge.second;
        auto const fixed_angle = _system_state.get_angle(fixed_index);
        TCM_ASSERT(std::find(begin(_sites), end(_sites), fixed_index)
                       != end(_sites),
                   "`fixed_index` must belong to `*this`.");
        _system_state.set_angle(begin(_sites), end(_sites), fixed_angle);
        /*
        for (auto& conn : _children) {
            TCM_ASSERT(conn.data->is_optimized(),
                       "It makes little sense to align a child which is not "
                       "optimised.");
            conn.data->align_with_parent();
        }
        */
    }
    else {
        auto const fixed_index = _sites.front();
        auto const fixed_angle = _system_state.get_angle(fixed_index);
        _system_state.set_angle(begin(_sites), end(_sites), fixed_angle);
    }
    _is_optimised = true;
}

template <class System>
TCM_FORCEINLINE auto
move_root_down(typename magnetic_cluster_t<System>::unique_ptr root,
               size_t const                                    child_index) ->
    typename magnetic_cluster_t<System>::unique_ptr
{
    // root->check_index_in_parent();
    using child_conn_info_t =
        typename magnetic_cluster_t<System>::child_conn_info_t;
    TCM_ASSERT(root->is_root(), "`root` must be the of the tree");
    TCM_ASSERT(child_index < root->_children.size(), "Index out of bounds");

    // Remove `child_index` from `root`'s list of children
    auto [edge, new_root] = root->detach(child_index);
    // Construct the new connection with the direction inverted.
    auto conn = child_conn_info_t{{edge.second, edge.first}, std::move(root)};
    // Attach it to the new root
    auto const old_state = new_root->_is_optimised;
    new_root->attach(no_optimize, std::move(conn));
    new_root->_is_optimised = old_state;
    TCM_ASSERT(new_root->is_root(), "Post-condition violated");
    // new_root->check_index_in_parent();
    return std::move(new_root);
}

TCM_NAMESPACE_END
