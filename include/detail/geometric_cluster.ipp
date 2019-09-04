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

#include "geometric_cluster.hpp"
#include "system_state.hpp"
#include <stack>

TCM_NAMESPACE_BEGIN

TCM_FORCEINLINE geometric_cluster_base_t::geometric_cluster_base_t(
    uint32_t const site, int const boundaries, angle_t const phase,
    system_base_t& system_state)
    : _root_index{site}
    , _size{1}
    , _boundaries{boundaries}
    , _root{pool_make_unique<magnetic_cluster_base_t>(site, phase,
                                                      system_state)}
    , _system_state{system_state}
{
    _system_state.on_cluster_created(*this)
        .on_size_changed(*this)
        .on_boundaries_changed(*this);
}

geometric_cluster_base_t::~geometric_cluster_base_t() noexcept
{
    _system_state.on_cluster_destroyed(*this);
}

constexpr auto geometric_cluster_base_t::size() const noexcept -> uint32_t
{
    return _size;
}

constexpr auto geometric_cluster_base_t::boundaries() const noexcept -> int
{
    return _boundaries;
}

constexpr auto geometric_cluster_base_t::root_index() const noexcept -> uint32_t
{
    return _root_index;
}

// WARNING: !!! This invalidates the _root_index !!!
auto geometric_cluster_base_t::invert(magnetic_cluster_base_t& new_root) -> void
{
    if (new_root.is_root()) { return; }

    // Constructs the full paths to the root.
    auto const find_path_to_root =
        [this](gsl::not_null<magnetic_cluster_base_t const*> node) {
            using vector_type = boost::container::small_vector<uint32_t, 58>;
            std::stack<uint32_t, vector_type> path;
            static_assert(sizeof(decltype(path)) == 256);
            while (!node->is_root()) {
                path.push(node->index_in_parent());
                node = node->parent();
            }
            TCM_ASSERT(node->is_root(), "");
            TCM_ASSERT(node == _root.get(), "");
            return path;
        };
    auto path = find_path_to_root(std::addressof(new_root));
    while (!path.empty()) {
        _root = move_root_down(std::move(_root), path.top());
        path.pop();
    }

    TCM_ASSERT(new_root.is_root(), "Post-condition failure");
    TCM_ASSERT(_root.get() == &new_root, "Post-condition failure!");
}

TCM_FORCEINLINE auto
geometric_cluster_base_t::merge(std::pair<uint32_t, uint32_t> edge,
                                geometric_cluster_base_t&     other) -> void
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

auto geometric_cluster_base_t::merge(
    std::pair<magnetic_cluster_base_t&, magnetic_cluster_base_t&> edge,
    geometric_cluster_base_t&                                     other) -> void
{
    auto& [this_cluster, other_cluster] = edge;
    // Prepare the `other` cluster
    other.invert(other_cluster);
    // Attach it to `*this`.
    this_cluster.merge(std::move(other._root));
    _size += other._size;
    _boundaries |= other._boundaries;
    // Update global statistics
    _system_state.on_size_changed(*this)
        .on_boundaries_changed(*this)
        // NOTE: This call destroys `other`
        .on_cluster_merged(*this, other);
}

namespace detail {
TCM_FORCEINLINE auto path_to_root(magnetic_cluster_base_t& node)
{
    using pointer_type = gsl::not_null<magnetic_cluster_base_t*>;
    std::vector<pointer_type> path;

    auto p = pointer_type{&node};
    while (!p->is_root()) {
        path.push_back(p);
        p = p->parent();
    }
    path.emplace_back(p);
    return path;
}

inline auto find_lca_impl(magnetic_cluster_base_t& left,
                          magnetic_cluster_base_t& right)
    -> std::tuple<gsl::not_null<magnetic_cluster_base_t*>,
                  std::vector<gsl::not_null<magnetic_cluster_base_t*>>,
                  std::vector<gsl::not_null<magnetic_cluster_base_t*>>>
{
    TCM_ASSERT(std::addressof(left) != std::addressof(right),
               "`left` and `right` must be different");
    auto path_left  = path_to_root(left);
    auto path_right = path_to_root(right);
    TCM_ASSERT(path_left.back() == path_right.back(),
               "`left` and `right` must belong to the same cluster.");

    // Throws away the common parts
    auto common = path_left.back();
    path_left.pop_back();
    path_right.pop_back();
    while (!path_left.empty() && !path_right.empty()
           && path_left.back() == path_right.back()) {
        common = path_left.back();
        path_left.pop_back();
        path_right.pop_back();
    }

    // Some sanity checks which compile to noop when optimisations are enabled
    if (path_left.empty()) { // `left` is an ancestor of `right`
        TCM_ASSERT(common == std::addressof(left), "");
        TCM_ASSERT(!path_right.empty(), "");
    }
    else if (path_right.empty()) { // `right` is an ancestor of `left`
        TCM_ASSERT(common == std::addressof(right), "");
        TCM_ASSERT(!path_left.empty(), "");
    }
    return std::make_tuple(common, std::move(path_left), std::move(path_right));
}
} // namespace detail

inline auto find_lca(magnetic_cluster_base_t& left,
                     magnetic_cluster_base_t& right)
    -> std::tuple<magnetic_cluster_base_t*,
                  std::vector<magnetic_cluster_base_t::unique_ptr>,
                  std::vector<magnetic_cluster_base_t::unique_ptr>>
{
    auto [common, path_left, path_right] = detail::find_lca_impl(left, right);

    auto const to_owning_path = [common = common](auto& path) {
        using unique_ptr = typename magnetic_cluster_base_t::unique_ptr;
        if (path.empty()) { return std::vector<unique_ptr>{}; }
        std::vector<unique_ptr> owning_path;
        owning_path.reserve(path.size());
        for (auto i = std::size_t{0}; i < path.size() - 1; ++i) {
            owning_path.emplace_back(
                path[i + 1]->detach(path[i]->index_in_parent()).data);
        }
        owning_path.emplace_back(
            common->detach(path.back()->index_in_parent()).data);
        return owning_path;
    };

    return std::make_tuple(common, to_owning_path(path_left),
                           to_owning_path(path_right));
}

auto geometric_cluster_base_t::form_cycle(magnetic_cluster_base_t& left,
                                          magnetic_cluster_base_t& right)
    -> void
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

template <>
auto thread_local_pool<geometric_cluster_base_t>() noexcept -> boost::pool<>&
{
#if defined(TCM_CLANG)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wexit-time-destructors"
#endif
    thread_local boost::pool<> pool{sizeof(geometric_cluster_base_t)};
#if defined(TCM_CLANG)
#    pragma clang diagnostic pop
#endif
    return pool;
}

TCM_NAMESPACE_END
