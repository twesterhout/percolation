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

// vim: foldenable foldmethod=marker

#include "magnetic_cluster.hpp"
#include "system_state.hpp"
#include <stack>

TCM_NAMESPACE_BEGIN

magnetic_cluster_base_t::magnetic_cluster_base_t(uint32_t const site,
                                                 angle_t const  phase,
                                                 system_base_t& system_state)
    : _parent{std::nullopt}
    , _children{}
    // TODO(twesterhout): Should we reserve some memory in advance?
    , _sites({site})
    , _system_state{system_state}
{
    _system_state.on_cluster_created(*this, site, phase);
}

auto magnetic_cluster_base_t::attach(child_conn_t child) -> void
{
    using std::begin, std::end;
    TCM_ASSERT(child.data != nullptr, "can't attach a non-existent cluster");
    TCM_ASSERT(!child.data->_parent.has_value(), "cluster must be an orphan");
    TCM_ASSERT(&child.data->_system_state == &_system_state,
               "clusters must belong to the same system");
    TCM_ASSERT(std::count(begin(child.data->_sites), end(child.data->_sites),
                          child.edge.second)
                   == 1,
               "invalid child_conn_t");
    TCM_ASSERT(std::count(begin(_sites), end(_sites), child.edge.first) == 1,
               "invalid child_conn_t");

    // Attach the child
    child.data->_parent = parent_conn_t{
        /*index=*/static_cast<uint32_t>(_children.size()), /*edge=*/child.edge,
        /*data=*/gsl::not_null<magnetic_cluster_base_t*>{this}};
    _children.emplace_back(std::move(child));
}

auto magnetic_cluster_base_t::detach(size_t const child_index) -> child_conn_t
{
    using std::swap;
    TCM_ASSERT(child_index < _children.size(), "index out of bounds");

    if (child_index != _children.size() - 1) {
        _children.back().data->_parent->index =
            static_cast<uint32_t>(child_index);
        swap(_children[child_index], _children.back());
    }
    auto conn = std::move(_children.back());
    _children.pop_back();
    conn.data->_parent = std::nullopt;
    return conn;
}

auto magnetic_cluster_base_t::insert(uint32_t const site, angle_t const phase)
    -> void
{
    _sites.push_back(site);
    _system_state.set_magnetic_cluster(site, *this)
        .set_angle(site, phase)
        .on_size_changed(*this);
}

TCM_NOINLINE auto magnetic_cluster_base_t::merge(unique_ptr cluster) -> void
{
    using std::begin, std::end;
    TCM_ASSERT(cluster != nullptr, "can't merge with a non-existant cluster");
    TCM_ASSERT(!cluster->_parent.has_value(), "cluster must be an orphan");
    TCM_ASSERT(&cluster->_system_state == &_system_state,
               "clusters must belong to the same system");

    // Steal children from `cluster`. For each child we have to update its
    // `parent_conn_t.data` to point to `this`. Afterwards, that child can be
    // safely moved into `this->_children`.
    _children.reserve(_children.size() + cluster->_children.size());
    for (auto& conn : cluster->_children) {
        conn.data->_parent->index = static_cast<uint32_t>(_children.size());
        // conn.data->_parent->edge is unchanged
        conn.data->_parent->data = this;
        _children.emplace_back(std::move(conn));
    }

    // Now we steal `cluster->_sites`. Each site now belongs to a different
    // magnetic cluster. This info has to be propagated to the `_system_state`.
    for (auto const site : cluster->_sites) {
        _system_state.set_magnetic_cluster(site, *this);
    }
    _sites.reserve(_sites.size() + cluster->_sites.size());
    _sites.insert(end(_sites), begin(cluster->_sites), end(cluster->_sites));
    _system_state.on_size_changed(*this);
}

auto magnetic_cluster_base_t::rotate(angle_t const angle) -> void
{
    dfs([angle, this](auto& x) {
        using std::begin, std::end;
        std::for_each(begin(x._sites), end(x._sites),
                      [angle, this](auto const site) {
                          _system_state.rotate(site, angle);
                      });
    });
}

constexpr auto magnetic_cluster_base_t::is_root() const noexcept -> bool
{
    return !_parent.has_value();
}

constexpr auto magnetic_cluster_base_t::index_in_parent() const TCM_NOEXCEPT
    -> uint32_t
{
    TCM_ASSERT(!is_root(), "root nodes are orphans");
    TCM_ASSERT(_parent->index < _parent->data->_children.size(),
               "index out of bounds");
    TCM_ASSERT(_parent->data->_children[_parent->index].data.get() == this,
               "invalid index in parent");
    return _parent->index;
}

constexpr auto magnetic_cluster_base_t::parent() const TCM_NOEXCEPT
    -> gsl::not_null<magnetic_cluster_base_t*>
{
    TCM_ASSERT(!is_root(), "root nodes are orphans");
    return _parent->data;
}

auto magnetic_cluster_base_t::sites() const noexcept
    -> gsl::span<uint32_t const>
{
    return _sites;
}

constexpr auto magnetic_cluster_base_t::system() const noexcept
    -> system_base_t const&
{
    return _system_state;
}

constexpr auto magnetic_cluster_base_t::system() noexcept -> system_base_t&
{
    return _system_state;
}

/// Returns the number of sites in this cluster.
///
/// \noexcept
auto magnetic_cluster_base_t::number_sites() const noexcept -> size_t
{
    return _sites.size();
}

/// Returns the number of child nodes.
///
/// \noexcept
auto magnetic_cluster_base_t::number_children() const noexcept -> size_t
{
    return _children.size();
}

template <class Function>
auto magnetic_cluster_base_t::dfs(Function&& fn) -> void
{
    using container_type =
        boost::container::small_vector<magnetic_cluster_base_t*, 29>;
    using stack_type = std::stack<magnetic_cluster_base_t*, container_type>;
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

auto move_root_down(magnetic_cluster_base_t::unique_ptr root,
                    size_t const                        child_index)
    -> magnetic_cluster_base_t::unique_ptr
{
    using child_conn_t = magnetic_cluster_base_t::child_conn_t;
    TCM_ASSERT(root->is_root(), "`root` must be the of the tree");
    TCM_ASSERT(child_index < root->_children.size(), "index out of bounds");

    // Remove `child_index` from `root`'s list of children
    auto [edge, new_root] = root->detach(child_index);
    // Construct the new connection with the direction inverted.
    auto conn = child_conn_t{/*edge=*/{edge.second, edge.first},
                             /*data=*/std::move(root)};
    // Attach it to the new root
    new_root->attach(std::move(conn));
    TCM_ASSERT(root == nullptr, "post-condition violated");
    TCM_ASSERT(new_root->is_root(), "post-condition violated");
    return std::move(new_root);
}

template <>
auto thread_local_pool<magnetic_cluster_base_t>() noexcept -> boost::pool<>&
{
#if defined(TCM_CLANG)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wexit-time-destructors"
#endif
    thread_local boost::pool<> pool{sizeof(magnetic_cluster_base_t)};
#if defined(TCM_CLANG)
#    pragma clang diagnostic pop
#endif
    return pool;
}

TCM_NAMESPACE_END
