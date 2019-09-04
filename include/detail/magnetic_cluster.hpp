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

// vim: foldenable foldmethod=marker

#pragma once

#include "detail/lattice.hpp"
#include "detail/memory.hpp"
#include "detail/utility.hpp"
#include <boost/container/small_vector.hpp>
#include <boost/pool/pool.hpp>
#include <gsl/gsl-lite.hpp>
#include <sleef.h>
#include <memory>
#include <optional>
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

#if 0
/// Let the direction of spin on site `site` be `x`. Denote the spins on the
/// nearest neighbours of `x` by `{αᵢ}`. Furthermore, let `cᵢ = +1` if
/// interaction between sites is antiferromagnetic and `cᵢ = -1` if interaction
/// is ferromagnetic.
///
/// Then our goal here is to find `argmin{E(x) | x∈[0, 2π)}` where
///
///     E(x) = ∑cᵢcos(x - αᵢ) = ∑cᵢ(sin(x)sin(αᵢ) + cos(x)cos(αᵢ))
///          = [∑cᵢsin(αᵢ)] * sin(x) + [∑cᵢcos(αᵢ)] * cos(x) .
///            ^^^^^^^^^^^^            ^^^^^^^^^^^^
///                := A                    := B
///
/// Since `E` is continuously differentiable, let's just compute its derivative
///
///     dE/dx = Acos(x) - Bsin(x) .
///
/// Now,
///
///     dE/dx = 0  <=>  Acos(x) = Bsin(x)  <=>  tan(x) = A/B, if B != 0
///                                             Acos(x) = 0,  if B == 0
///
/// * B != 0: We have a choice between
///
///        x₁ = tan⁻¹(A/B) + 2π | mod 2π,   and
///        x₂ = x₁ + π          | mod 2π
///
///   We simply compute `E` for both values and choose the one that results in
///   lower energy.
///
/// * `B == 0 && A != 0`: We have a choice between
///
///        x₁ = π/2,    and
///        x₂ = 3π/2
///
///   Again, we compute `E` for both `x`'s and determine the best.
///
/// * `B == 0 && A == 0`: `E` is constant, so we can choose arbitrary `x`. For
/// simplicity, we pick `x = 0`.
///
template <class System>
auto minimise_local_energy(size_t const site, System const& system) -> angle_t
{
    // Determines the coefficients cᵢ
    struct get_c_t {
        using Lattice = typename System::lattice_type;
        Lattice const& lattice;

        auto operator()(size_t const i, size_t const j) const TCM_NOEXCEPT
            -> float
        {
            switch (::TCM_NAMESPACE::interaction(lattice, i, j)) {
            case interaction_t::Ferromagnetic: return -1.0f;
            case interaction_t::Antiferromagnetic: return 1.0f;
            } // end switch
        }
    } get_c{system.lattice()};

    auto A = 0.0f;
    auto B = 0.0f;
    for (int64_t const _i : system.lattice().neighbours[site]) {
        if (_i >= 0 && !system.is_empty(static_cast<size_t>(_i))) {
            auto const i = static_cast<size_t>(_i);
            auto const c = get_c(site, i);
            // Projection of spin on the y-axis is the sine
            A += c * system.S_y(i);
            // Projection of spin on the x-axis is the cosine
            B += c * system.S_x(i);
        }
    }

    auto const E = [A, B](angle_t const x) {
        auto const v     = Sleef_sincosf_u10(static_cast<float>(x));
        auto const sin_x = v.x;
        auto const cos_x = v.y;
        return A * sin_x + B * cos_x;
    };

    if (B != 0.0f) {
        auto x1 = [A, B]() {
            constexpr auto epsilon = -4.7683716E-7f;
            constexpr auto two_pi  = detail::two_pi<float>;
            auto           result  = Sleef_atanf_u10(A / B);
            if (result < epsilon) { return angle_t{result + two_pi}; }
            else if (result < 0.0f) {
                return angle_t{0.0f};
            }
            else {
                return angle_t{result};
            }
        }();
        auto const x2 = x1 + angle_t{detail::pi<float>};
        return (E(x1) <= E(x2)) ? x1 : x2;
    }
    else if (A != 0.0f) {
        auto const x1 = angle_t{1.5707964f}; // pi / 2
        auto const x2 = angle_t{4.712389f};  // 3 * pi / 2
        return (E(x1) <= E(x2)) ? x1 : x2;
    }
    else {
        return angle_t{0.0f};
    }
}
#endif

class system_base_t;

struct magnetic_cluster_base_t {
  public:
    using unique_ptr = pool_unique_ptr<magnetic_cluster_base_t>;

    /// Information about the connection between this cluster and a child
    /// cluster.
    struct child_conn_t {
        /// The edge which connects this cluster to the child. `edge.first`
        /// belongs to this cluster and `edge.second` belongs to the child.
        std::pair<uint32_t, uint32_t> edge;
        /// The child cluster itself.
        ///
        /// \note Yes, parents own their children. No shared ownership.
        unique_ptr data;
    };

    /// Information about the connection between this cluster and its parent.
    struct parent_conn_t {
        /// Index in the #_children array of the parent.
        uint32_t index;
        /// The edge which connects the parent to this cluster. `edge.first`
        /// belongs to the parent and `edge.second` belongs to this cluster.
        std::pair<uint32_t, uint32_t> edge;
        /// Pointer to the parent node.
        ///
        /// \note The parent owns this cluster so it's safe to store a
        /// non-owning pointer here.
        gsl::not_null<magnetic_cluster_base_t*> data;
    };

  private:
    /// Connection to the parent node.
    std::optional<parent_conn_t> _parent;
    /// Connection to children nodes.
    boost::container::small_vector<child_conn_t, 1> _children;
    /// A list of indices of all the sites that belong to this cluster.
    std::vector<uint32_t> _sites;
    /// A reference to the global system state
    system_base_t& _system_state;

  public:
    /// Constructs a new magnetic cluster consisting of just one site
    ///
    /// \param index Index of the site.
    /// \param angle Direction of the spin at site \p index.
    /// \param system A reference to the system state.
    TCM_FORCEINLINE magnetic_cluster_base_t(uint32_t index, angle_t angle,
                                            system_base_t& system);

    /// **Deleted** copy constructor.
    magnetic_cluster_base_t(magnetic_cluster_base_t const&) = delete;
    /// **Deleted** move constructor.
    magnetic_cluster_base_t(magnetic_cluster_base_t&&) = delete;
    /// **Deleted** copy assignment operator.
    auto operator                   =(magnetic_cluster_base_t const&)
        -> magnetic_cluster_base_t& = delete;
    /// **Deleted** move assignment operator.
    auto operator                   =(magnetic_cluster_base_t &&)
        -> magnetic_cluster_base_t& = delete;

    /// Attaches a child node.
    ///
    /// \param connection Specifies the child node and the edge connecting it to
    /// `*this`.
    ///
    /// \throws std::bad_alloc if memory allocation fails.
    TCM_FORCEINLINE auto attach(child_conn_t connection) -> void;

    /// Detaches the `i`'th child.
    ///
    /// \param i Index of the child to detach. \p i must be less than
    /// number_children().
    /// \return The child connection.
    TCM_FORCEINLINE auto detach(size_t i) -> child_conn_t;

    /// Adds site `i` to the cluster.
    ///
    /// \param i     Index of the site.
    /// \param angle Direction of the spin.
    TCM_FORCEINLINE auto insert(uint32_t i, angle_t angle) -> void;

    /// Merges \p cluster into `*this`.
    ///
    /// The merging is done in two steps:
    ///   -# stealing all the *children* from \p cluster, and
    ///   -# stealing all the *sites* from \p cluster.
    TCM_FORCEINLINE auto merge(unique_ptr cluster) -> void;

    /// Rotates the subtree (i.e. `*this` and all the children recursively) by
    /// \p angle.
    ///
    /// Rotation amounts to a simle depth-first search over the subtree.
    TCM_FORCEINLINE auto rotate(angle_t angle) -> void;

    /// Returns whether the cluster is the root of the geometric cluster.
    ///
    /// \noexcept
    constexpr auto is_root() const noexcept -> bool;

    /// Returns the index of `*this` in the array of children of its parent.
    ///
    /// \pre `*this` is not the root node, i.e. is_root() returns `false`.
    /// \noexcept
    constexpr auto index_in_parent() const TCM_NOEXCEPT -> uint32_t;

    /// Returns a non-owning pointer to the parent node.
    ///
    /// \pre `*this` has a parent, i.e. is_root() returns `false`.
    /// \noexcept
    constexpr auto parent() const TCM_NOEXCEPT
        -> gsl::not_null<magnetic_cluster_base_t*>;

    /// Returns a non-owning view of the sites in the cluster.
    ///
    /// \noexcept
    inline auto sites() const noexcept -> gsl::span<uint32_t const>;

    constexpr auto system() const noexcept -> system_base_t const&;
    constexpr auto system() noexcept -> system_base_t&;

    /// Returns the number of sites in this cluster.
    ///
    /// \noexcept
    inline auto number_sites() const noexcept -> size_t;

    /// Returns the number of child nodes.
    ///
    /// \noexcept
    inline auto number_children() const noexcept -> size_t;

    friend TCM_FORCEINLINE auto
    move_root_down(magnetic_cluster_base_t::unique_ptr root, size_t child_index)
        -> magnetic_cluster_base_t::unique_ptr;

    /// Runs Depth-First Search on the subtree applying \p fn to each node.
    ///
    /// \param fn Function to apply to each node. \p fn should be callable with
    /// a reference to magnetic_cluster_t. Return value is ignored.
    template <class Function> TCM_FORCEINLINE auto dfs(Function&& fn) -> void;

    template <class Lattice>
    inline auto align_with_parent(Lattice const& lattice) -> void;
};

#if 0

/// A magnetic cluster, i.e. an irreducible component in our graph.
template <class System> class magnetic_cluster_t { // {{{
  public:
    using system_type = System;
#    if !defined(DOXYGEN_IS_IN_THE_HOUSE)
    using unique_ptr =
        typename System::template unique_ptr<magnetic_cluster_t<System>>;
#    else
    using unique_ptr = std::unique_ptr<magnetic_cluster_t<System>>;
#    endif

  private:
    /// Information about the connection between this cluster and a child
    /// cluster.
    struct child_conn_info_t {
        /// The edge which connects this cluster to the child. `edge.first`
        /// belongs to this cluster and `edge.second` belongs to the child.
        std::pair<size_t, size_t> edge;
        /// The child cluster itself.
        ///
        /// \note Yes, parents own their children. No shared ownership.
        unique_ptr data;
    };

    /// Information about the connection between this cluster and its parent.
    struct parent_conn_info_t {
        /// Index in the #_children array of the parent.
        size_t index;
        /// The edge which connects the parent to this cluster. `edge.first`
        /// belongs to the parent and `edge.second` belongs to this cluster.
        std::pair<size_t, size_t> edge;
        /// Pointer to the parent node.
        ///
        /// \note The parent owns this cluster so it's safe to store a
        /// non-owning pointer here.
        gsl::not_null<magnetic_cluster_t*> data;
    };

    /// Connection to the parent node.
    std::optional<parent_conn_info_t> _parent;
    /// Connection to children nodes.
    boost::container::small_vector<child_conn_info_t, 1> _children;
    /// A list of indices of all the sites that belong to this cluster.
    std::vector<size_t> _sites;
    /// A reference to the global system state
    system_type& _system_state;

    // static_assert(sizeof(boost::container::small_vector<child_conn_info_t, 1>)
    //               == 56);

  public:
    /// Constructs a new magnetic cluster consisting of just one site
    ///
    /// \param index Index of the site.
    /// \param angle Direction of the spin at site \p index.
    /// \param system A reference to the system state.
    inline magnetic_cluster_t(size_t index, angle_t angle, system_type& system);

    /// **Deleted** copy constructor.
    magnetic_cluster_t(magnetic_cluster_t const&) = delete;
    /// **Deleted** move constructor.
    magnetic_cluster_t(magnetic_cluster_t&&) = delete;
    /// **Deleted** copy assignment operator.
    magnetic_cluster_t& operator=(magnetic_cluster_t const&) = delete;
    /// **Deleted** move assignment operator.
    magnetic_cluster_t& operator=(magnetic_cluster_t&&) = delete;

    /// Attaches a child node.
    ///
    /// \param connection Specifies the child node and the edge connecting it to
    /// `*this`.
    ///
    /// \throws std::bad_alloc if memory allocation fails.
    inline auto attach(child_conn_info_t connection) -> void;
    // inline auto attach(no_optimize_t, child_conn_info_t /*connection*/) -> void;

    /// Detaches the `i`'th child.
    ///
    /// \param i Index of the child to detach. \p i must be less than
    /// number_children().
    /// \return The child connection.
    inline auto detach(size_t i) -> child_conn_info_t;

    /// Adds site `i` to the cluster.
    ///
    /// \param i Index of the site.
    inline auto insert(size_t i) -> void;
    // inline auto insert(no_optimize_t, size_t /*site index*/) -> void;

    /// Merges \p cluster into `*this`.
    ///
    /// The merging is done in two steps:
    ///   -# stealing all the *children* from \p cluster, and
    ///   -# stealing all the *sites* from \p cluster.
    inline auto merge(unique_ptr cluster) -> void;
    // auto        merge(no_optimize_t, unique_ptr cluster) -> void;

    /// Rotates the subtree (i.e. `*this` and all the children recursively) by
    /// \p angle.
    ///
    /// Rotation amounts to a simle depth-first search over the subtree.
    inline auto rotate(angle_t angle) -> void;

    /// Optimizes the energy of the magnetic cluster assuming that the cluster
    /// was optimized and then a new site `site` was added.
    // auto optimize_one(size_t site) -> void;

    /// Optimizes the energy of the magnetic cluster.
    // TCM_NOINLINE auto optimize_full() -> void;

    /// Returns whether the cluster is the root of the geometric cluster.
    ///
    /// \noexcept
    constexpr auto is_root() const noexcept { return !_parent.has_value(); }

    /// Returns the index of `*this` in the array of children of its parent.
    ///
    /// \pre `*this` is not the root node, i.e. is_root() returns `false`.
    /// \noexcept
    constexpr auto index_in_parent() const TCM_NOEXCEPT
    {
        TCM_ASSERT(!is_root(), "Root nodes are orphans");
        TCM_ASSERT(_parent->index < _parent->data->_children.size(),
                   "Index out of bounds");
        TCM_ASSERT(_parent->data->_children[_parent->index].data.get() == this,
                   "Bug! Invalid index in parent");
        return _parent->index;
    }

    /// Returns a non-owning pointer to the parent node.
    ///
    /// \pre `*this` has a parent, i.e. is_root() returns `false`.
    /// \noexcept
    constexpr auto parent() const TCM_NOEXCEPT
    {
        TCM_ASSERT(!is_root(), "Root nodes are orphans");
        return _parent->data;
    }

    /// Returns a non-owning view of the sites in the cluster.
    ///
    /// \noexcept
    constexpr auto sites() const noexcept -> gsl::span<size_t const>
    {
        return {_sites};
    }

    constexpr auto system() const noexcept -> system_type const&
    {
        return _system_state;
    }

    constexpr auto system() noexcept -> system_type& { return _system_state; }

    /// Returns the number of sites in this cluster.
    ///
    /// \noexcept
    auto number_sites() const noexcept -> size_t { return _sites.size(); }

    /// Returns the number of sites in this cluster.
    ///
    /// \noexcept
    auto size() const noexcept -> size_t { return number_sites(); }

    /// Returns the number of child nodes.
    ///
    /// \noexcept
    auto number_children() const noexcept -> size_t { return _children.size(); }

    template <class S>
    friend inline auto
        move_root_down(typename magnetic_cluster_t<S>::unique_ptr /*root*/,
                       size_t /*child index*/) ->
        typename magnetic_cluster_t<S>::unique_ptr;

    /// Runs Depth-First Search on the subtree applying \p fn to each node.
    ///
    /// \param fn Function to apply to each node. \p fn should be callable with
    /// a reference to magnetic_cluster_t. Return value is ignored.
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
    /// Aligns `*this` with its parent.
    ///
    /// Let `(i, j)` be the edge connecting `*this` to its parent. Site `i`
    /// belongs to the parent and site `j` belongs to `*this`. If the
    /// interaction between sites `i` and `j` is of ferromagnetic character, we
    /// rotate `*this` (and its children) such that sites `i` and `j` become
    /// aligned.  Otherwise (i.e. if the interaction is of antiferromagnetic
    /// character) we rotate `*this` such that the angle between `i` and `j`
    /// becomes *π*.
    ///
    /// \throws std::bad_alloc If dfs() fails to allocate memory.
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
{
    _system_state.set_angle(site, phase);
    _system_state.set_magnetic_cluster(site, *this);
}

template <class System>
TCM_FORCEINLINE auto magnetic_cluster_t<System>::insert(size_t const site)
    -> void
{
    _sites.push_back(site);
    _system_state.set_magnetic_cluster(site, *this);
    _system_state.set_angle(site, minimise_local_energy(site, _system_state));
}

#    if 0
template <class System>
TCM_FORCEINLINE auto magnetic_cluster_t<System>::insert(size_t const site)
    -> void
{
    TCM_ASSERT(_is_optimised, "It is assumed that the cluster is optimized");
    insert(no_optimize, site);
    optimize_one(site);
    _is_optimised = true;
}
#    endif

#    if 0
template <class System>
TCM_FORCEINLINE auto magnetic_cluster_t<System>::attach(child_conn_info_t child)
    -> void
{
    // TCM_ASSERT(_is_optimised, "It is assumed that the cluster is optimized");
    attach(no_optimize, std::move(child));
    // _children.back().data->align_with_parent();
    // _is_optimised = true;
}
#    endif

template <class System>
TCM_FORCEINLINE auto magnetic_cluster_t<System>::attach(child_conn_info_t child)
    -> void
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
    _children.back().data->align_with_parent();
}

template <class System>
auto magnetic_cluster_t<System>::merge(unique_ptr cluster) -> void
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
}

#    if 0
template <class System>
TCM_FORCEINLINE auto magnetic_cluster_t<System>::merge(unique_ptr cluster)
    -> void
{
    merge(no_optimize, std::move(cluster));
    optimize_full();
}
#    endif

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

#    if 0
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
#    endif

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
    // auto const old_state = new_root->_is_optimised;
    new_root->attach(std::move(conn));
    // new_root->_is_optimised = old_state;
    TCM_ASSERT(new_root->is_root(), "Post-condition violated");
    // new_root->check_index_in_parent();
    return std::move(new_root);
}
#endif

TCM_NAMESPACE_END
