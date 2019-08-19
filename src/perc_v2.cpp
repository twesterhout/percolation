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

#include "detail/geometric_cluster.hpp"
#include "detail/lattice.hpp"
#include "detail/magnetic_cluster.hpp"
#include "detail/neighbour_stats.hpp"
#include "detail/particle.hpp"
#include "detail/random.hpp"
#include "detail/shuffle.hpp"
#include "detail/thermalisation.hpp"
#include "detail/utility.hpp"
#include "perc_v2.h"
#include <boost/align/aligned_allocator.hpp>
#include <boost/container/static_vector.hpp>
#include <boost/pool/pool.hpp>
#include <gsl/gsl-lite.hpp>
#include <sys/user.h>
#include <cmath>
#include <optional>
#include <stack>
#include <utility>
#include <vector>

TCM_NAMESPACE_BEGIN

using std::size_t;

template <class Lattice> class system_state_t { // {{{

  public:
    using pool_type              = boost::pool<>;
    using magnetic_cluster_type  = magnetic_cluster_t<system_state_t>;
    using geometric_cluster_type = geometric_cluster_t<system_state_t>;
    using particle_type          = particle_t<system_state_t>;
    using lattice_type           = Lattice;

    struct pool_deleter {
        template <class T> auto operator()(T* const p) noexcept -> void
        {
            TCM_ASSERT(p != nullptr, "Trying to delete a nullptr");
            p->~T();
            pool->free(p);
        }
        gsl::not_null<pool_type*> pool;
    };

    template <class T> using unique_ptr = std::unique_ptr<T, pool_deleter>;

    static_assert(sizeof(pool_deleter) == sizeof(void*), "");
    static_assert(sizeof(unique_ptr<geometric_cluster_type>)
                      == 2 * sizeof(void*),
                  "");

    static_assert(sizeof(particle_type)
                  == sizeof(unique_ptr<geometric_cluster_type>));

    template <class T>
    using buffer_type =
        std::vector<T, boost::alignment::aligned_allocator<T, PAGE_SIZE>>;

    using neighbour_stats_t =
        ::TCM_NAMESPACE::neighbour_stats_t<magnetic_cluster_type,
                                           max_neighbours<Lattice>()>;

  private:
    pool_type _magnetic_pool;  ///< Memory pool for magnetic clusters
    pool_type _geometric_pool; ///< Memory pool for geometric clusters
    buffer_type<particle_type> _particles; ///< All particles in the system
    buffer_type<magnetic_cluster_type*>
        _clusters; ///< All magnetic clusters in the system. `_clusters[i]`
                   ///< points to the magnetic cluster which contains site `i`.
                   ///< `_clusters[i] == nullptr` if `i` is empty.
    buffer_type<angle_t> _angles; ///< Orientation of spins as an angle
                                  ///< (in rads) between the spin and
                                  ///< the X-axis.
    buffer_type<float> _S_x;      ///< Projections of spins to X-axis.
    buffer_type<float> _S_y;      ///< Projections of spins to Y-axis.
    Lattice const&     _lattice;
    random_generator_t&
        _generator; ///< General purpose random number generator.

    thermaliser_t _thermaliser;
    // sa_buffers_t     _sa_buffers;     ///< Buffers for Simulated Annealing
    // energy_buffers_t _energy_buffers; ///< Buffers for the Hamiltonians.

    size_t _number_sites;     ///< Total number of sites in the system.
    size_t _number_clusters;  ///< Number of geometric clusters.
    size_t _max_cluster_size; ///< Size of the largest geometric cluster.
    bool   _has_wrapped[3];
    bool   _optimizing;
    std::unique_ptr<neighbour_stats_t>
        _neighbour_stats; ///< Buffers for analysing neighbours

  public: // geometric_cluster relies on this
    /// Allocates and constructs a new magnetic cluster.
    ///
    /// \throws std::bad_alloc if memory allocation fails
    /// \throws whatever #magnetic_cluster_type constructor may throw.
    template <class... Args>
    auto make_magnetic_cluster(Args&&... args)
        -> unique_ptr<magnetic_cluster_type>
    {
        auto* p = _magnetic_pool.malloc();
        if (p == nullptr) { throw std::bad_alloc{}; }
        try {
            new (p) magnetic_cluster_type{std::forward<Args>(args)...};
            return unique_ptr<magnetic_cluster_type>{
                reinterpret_cast<magnetic_cluster_type*>(p),
                pool_deleter{&_magnetic_pool}};
        }
        catch (...) {
            _magnetic_pool.free(p);
            throw;
        }
    }

  private:
    /// Allocates and constructs a new geometric cluster.
    ///
    /// \throws std::bad_alloc if memory allocation fails
    /// \throws whatever #geometric_cluster_type constructor may throw.
    template <class... Args>
    auto make_geometric_cluster(Args&&... args)
        -> unique_ptr<geometric_cluster_type>
    {
        auto* p = _geometric_pool.malloc();
        if (p == nullptr) { throw std::bad_alloc{}; }
        try {
            new (p) geometric_cluster_type{std::forward<Args>(args)...};
            return unique_ptr<geometric_cluster_type>{
                reinterpret_cast<geometric_cluster_type*>(p),
                pool_deleter{&_geometric_pool}};
        }
        catch (...) {
            _geometric_pool.free(p);
            throw;
        }
    }

  public:
    TCM_NOINLINE system_state_t(Lattice const&, random_generator_t&);

    system_state_t(system_state_t const&) = delete;
    system_state_t(system_state_t&&)      = delete;
    system_state_t& operator=(system_state_t const&) = delete;
    system_state_t& operator=(system_state_t&&) = delete;

    // {{{ Callbacks
    /// This is a callback that should be invoked when the size of a geometric
    /// cluster changes.
    ///
    /// Helps keep track of the #_max_cluster_size property.
    constexpr auto on_size_changed(geometric_cluster_type const&) noexcept
        -> system_state_t&;

    /// This is a callback that should be invoked when the "boundaries" of a
    /// geometric cluster change.
    ///
    /// Helps keep track of the #_has_wrapped property.
    constexpr auto on_boundaries_changed(geometric_cluster_type const&) noexcept
        -> system_state_t&;

    /// This is a callback that should be invoked when a new geometric cluster
    /// is created.
    ///
    /// Helps keep track of the #_number_clusters property.
    constexpr auto on_cluster_created(geometric_cluster_type const&) noexcept
        -> system_state_t&;

    /// This is a callback that should be invoked when a geometric cluster is
    /// destroyed (i.e. merged into another).
    ///
    /// Helps keep track of the #_number_clusters property
    constexpr auto on_cluster_destroyed(geometric_cluster_type const&) noexcept
        -> system_state_t&;

    /// This is a callback that should be called when \p small is merged into \p
    /// big.
    ///
    /// Helps keep track of the #_particles.
    constexpr auto on_cluster_merged(geometric_cluster_type& big,
                                     geometric_cluster_type& small) noexcept
        -> system_state_t&;
    // }}}

    // constexpr auto optimizing() const noexcept -> bool { return _optimizing; }

    // TCM_NOINLINE auto optimizing(bool do_opt) -> void;

    constexpr auto max_number_sites() const noexcept -> size_t;
    constexpr auto number_sites() const noexcept -> size_t;
    constexpr auto number_clusters() const noexcept -> size_t;
    constexpr auto max_cluster_size() const noexcept -> size_t;
    constexpr auto has_wrapped() const & noexcept -> gsl::span<bool const>;
    constexpr auto lattice() const noexcept -> Lattice const&;
    // constexpr auto sa_buffers() & noexcept -> sa_buffers_t&;
    // constexpr auto energy_buffers() & noexcept -> energy_buffers_t&;
    auto rng_stream() const -> VSLStreamStatePtr;

    constexpr auto is_empty(size_t const i) const TCM_NOEXCEPT -> bool
    {
        TCM_ASSERT(0 <= i && i < max_number_sites(), "Index out of bounds");
        return _particles[i].is_empty();
    }

    // [Spins] {{{
    constexpr auto get_angle(size_t const i) const TCM_NOEXCEPT -> angle_t
    {
        TCM_ASSERT(0 <= i && i < max_number_sites(), "Index out of bounds");
        TCM_ASSERT(!_particles[i].is_empty(), "Site is empty");
        return _angles[i];
    }

    constexpr auto S_x(size_t const i) const TCM_NOEXCEPT -> float
    {
        TCM_ASSERT(0 <= i && i < max_number_sites(), "Index out of bounds");
        TCM_ASSERT(!_particles[i].is_empty(), "Site is empty");
        return _S_x[i];
    }

    constexpr auto S_y(size_t const i) const TCM_NOEXCEPT -> float
    {
        TCM_ASSERT(0 <= i && i < max_number_sites(), "Index out of bounds");
        TCM_ASSERT(!_particles[i].is_empty(), "Site is empty");
        return _S_y[i];
    }

    auto set_angle(size_t i, angle_t new_angle) TCM_NOEXCEPT -> void;

    template <class RAIter>
    auto set_angle(RAIter first, RAIter last, angle_t new_angle) noexcept
        -> void;

    constexpr auto rotate(size_t const i, angle_t const angle) TCM_NOEXCEPT
        -> void
    {
        TCM_ASSERT(0 <= i && i < max_number_sites(), "Index out of bounds");
        TCM_ASSERT(!_particles[i].is_empty(), "Site is empty");
        set_angle(i, get_angle(i) + angle);
    }

    template <class RAIter>
    auto rotate(RAIter first, RAIter last, angle_t const angle) noexcept
        -> void;
    // [Spins] }}}

    // magnetic clusters {{{
    constexpr auto get_magnetic_cluster(size_t const i) TCM_NOEXCEPT
        -> magnetic_cluster_type&
    {
        TCM_ASSERT(0 <= i && i < max_number_sites(), "Index out of bounds");
        TCM_ASSERT(!_particles[i].is_empty(), "Site is empty");
        return *_clusters[i];
    }

    constexpr auto get_magnetic_cluster(size_t const i) const TCM_NOEXCEPT
        -> magnetic_cluster_type const&
    {
        TCM_ASSERT(0 <= i && i < max_number_sites(), "Index out of bounds");
        TCM_ASSERT(!_particles[i].is_empty(), "Site is empty");
        return *_clusters[i];
    }

    constexpr auto set_magnetic_cluster(
        size_t const i, magnetic_cluster_type& new_cluster) TCM_NOEXCEPT -> void
    {
        TCM_ASSERT(0 <= i && i < max_number_sites(), "Index out of bounds");
        // TCM_ASSERT(!_particles[i].is_empty(), "Site is empty");
        _clusters[i] = std::addressof(new_cluster);
    }
    // magnetic clusters }}}

    auto get_geometric_cluster(magnetic_cluster_type const& cluster)
        -> geometric_cluster_type&
    {
        TCM_ASSERT(cluster.sites().size() > 0,
                   "Magnetic cluster can't be empty");
        auto const site = cluster.sites()[0];
        auto const root = find_root_index(site);
        return _particles[root].cluster();
    }

#if 0
    constexpr auto get_interaction_type(size_t const i, size_t const j) const
        noexcept -> interaction_t
    {
        auto const are_connected = [this](auto const i,
                                          auto const j) noexcept->bool
        {
            using std::begin, std::end;
            auto const nns = neighbours(_lattice, i);
            return std::count(begin(nns), end(nns),
                              static_cast<std::int64_t>(j))
                   == 1;
        };
        TCM_ASSERT(0 <= i && i < max_number_sites(), "Index out of bounds");
        TCM_ASSERT(0 <= j && j < max_number_sites(), "Index out of bounds");
        TCM_ASSERT(!_particles[i].is_empty(), "Site is empty");
        TCM_ASSERT(!_particles[j].is_empty(), "Site is empty");
        TCM_ASSERT(are_connected(i, j),
                   "There's no interaction between sites i and j");
        if (sublattice(_lattice, i) == sublattice(_lattice, j)) {
            return interaction_t::Ferromagnetic;
        }
        return interaction_t::Antiferromagnetic;
    }
#endif

#if 0
    // XY Hamiltonian
    auto energy() const -> double
    {
        std::vector<double> thetas;
        for (auto i = size_t{0}; i < static_cast<size_t>(_lattice.size); ++i) {
            if (!_particles[i].is_empty()) {
                for (auto const j : _lattice.neighbours[i]) {
                    if (j >= 0 && !_particles[j].is_empty() && i < j) {
                        thetas.push_back(get_angle(i) - get_angle(j));
                    }
                }
            }
        }

        return std::accumulate(std::begin(thetas), std::end(thetas), 0.0,
                               [](auto const acc, auto const theta) {
                                   return acc + std::cos(theta);
                               });
    }
#endif

    TCM_NOINLINE auto magnetisation() const -> std::array<float, 2>;

    auto stats() const
    {
        using std::begin, std::end;
        std::vector<magnetic_cluster_type const*> clusters;
        clusters.reserve(_clusters.size());
        std::copy_if(begin(_clusters), end(_clusters),
                     std::back_inserter(clusters),
                     [](auto* x) { return x != nullptr; });
        std::sort(begin(clusters), end(clusters));
        clusters.erase(std::unique(begin(clusters), end(clusters)),
                       end(clusters));
        // std::fprintf(stderr, "clusters.size(): %zu\n", clusters.size());

        auto max_number_sites     = 0.0;
        auto mean_number_sites    = 0.0;
        auto max_number_children  = 0.0;
        auto mean_number_children = 0.0;

        if (!clusters.empty()) {
            for (auto const* const x : clusters) {
                auto const number_sites =
                    static_cast<double>(x->number_sites());
                auto const number_children =
                    static_cast<double>(x->number_children());
                mean_number_sites += number_sites;
                mean_number_children += number_children;
                if (number_sites > max_number_sites) {
                    max_number_sites = number_sites;
                }
                if (number_children > max_number_children) {
                    max_number_children = number_children;
                }
            }
            mean_number_sites /= clusters.size();
            mean_number_children /= clusters.size();
        }

        return std::make_tuple(max_number_sites, mean_number_sites,
                               max_number_children, mean_number_children);
    }

  private:
    auto create_new_cluster(size_t const i, int const boundaries) -> void
    {
        TCM_ASSERT(0 <= i && i < max_number_sites(), "Index out of bounds");
        TCM_ASSERT(_particles[i].is_empty(), "Site already exists");
        TCM_ASSERT(_clusters[i] == nullptr,
                   "Non-existent site can't belong to a cluster!");

        auto cluster = make_geometric_cluster(i, boundaries,
                                              random_angle(_generator), *this);
        TCM_ASSERT(cluster != nullptr, "Bug!");
        _particles[i] = particle_t<system_state_t>{std::move(cluster)};
    }

    /// Analyses the neighbours of `i` to determine the optimal merging
    /// strategy.
    auto analyse_site(size_t i) TCM_NOEXCEPT -> neighbour_stats_t const&;

    TCM_NOINLINE auto find_root_index(size_t i) -> size_t;

    TCM_NOINLINE auto connect(size_t i, size_t j) -> void;

    TCM_NOINLINE auto connect_and_merge(magnetic_cluster_type&,
                                        magnetic_cluster_type&) -> void;

  public:
    TCM_NOINLINE auto operator()(size_t const i,
                                 std::false_type /*is periodic?*/);

    TCM_NOINLINE auto thermalise() -> void;
}; // }}}

template <class Lattice>
TCM_NOINLINE
system_state_t<Lattice>::system_state_t(Lattice const&      lattice,
                                        random_generator_t& generator)
    : _magnetic_pool{sizeof(magnetic_cluster_type),
                     PAGE_SIZE / sizeof(magnetic_cluster_type)}
    , _geometric_pool{sizeof(geometric_cluster_type),
                      PAGE_SIZE / sizeof(magnetic_cluster_type)}
    , _particles{}
    , _clusters{}
    , _angles{}
    , _S_x{}
    , _S_y{}
    , _lattice{lattice}
    , _generator{generator}
    // The biggest cluster we could ever have to thermalise is the whole system.
    , _thermaliser{size(lattice)} // , _sa_buffers{size(lattice)}
    // , _energy_buffers{size(lattice)}
    , _number_sites{0}
    , _number_clusters{0}
    , _max_cluster_size{0}
    , _has_wrapped{false, false, false}
    , _optimizing{false}
    , _neighbour_stats{std::make_unique<neighbour_stats_t>()}
{
    _particles.resize(size(lattice));
    _clusters.resize(size(lattice));
    _angles.resize(size(lattice));
    _S_x.resize(size(lattice), 0.0f);
    _S_y.resize(size(lattice), 0.0f);
}

// [Callbacks] {{{
template <class Lattice>
constexpr auto system_state_t<Lattice>::on_size_changed(
    geometric_cluster_type const& cluster) noexcept -> system_state_t&
{
    auto const size = cluster.size();
    if (size > _max_cluster_size) { _max_cluster_size = size; }
    return *this;
}

template <class Lattice>
constexpr auto system_state_t<Lattice>::on_boundaries_changed(
    geometric_cluster_type const& cluster) noexcept -> system_state_t&
{
    update_has_wrapped(_has_wrapped, cluster.boundaries());
    return *this;
}

template <class Lattice>
constexpr auto system_state_t<Lattice>::on_cluster_created(
    geometric_cluster_type const&) noexcept -> system_state_t&
{
    ++_number_clusters;
    return *this;
}

template <class Lattice>
constexpr auto system_state_t<Lattice>::on_cluster_destroyed(
    geometric_cluster_type const&) noexcept -> system_state_t&
{
    TCM_ASSERT(_number_clusters > 0, "There are no clusters to destroy");
    --_number_clusters;
    return *this;
}

template <class Lattice>
constexpr auto system_state_t<Lattice>::on_cluster_merged(
    geometric_cluster_type& big, geometric_cluster_type& small) noexcept
    -> system_state_t&
{
    TCM_ASSERT(&big != &small, "Can't merge the same clusters");
    auto const big_root    = big.root_index();
    auto const small_root  = small.root_index();
    _particles[small_root] = particle_type{from_parent_index, big_root};
    return *this;
}
// [Callbacks] }}}

#if 0
template <class Lattice>
TCM_NOINLINE auto system_state_t<Lattice>::optimizing(bool const do_opt) -> void
{
#    if 0
    if (!_optimizing && do_opt) {
        for (auto& x : _particles) {
            if (x.is_root()) { x.cluster().optimize_full(); }
        }
    }
#    endif
    _optimizing = do_opt;
}
#endif

// {{{ Statistics
template <class Lattice>
constexpr auto system_state_t<Lattice>::max_number_sites() const noexcept
    -> size_t
{
    return size(_lattice);
}

template <class Lattice>
constexpr auto system_state_t<Lattice>::number_sites() const noexcept -> size_t
{
    return _number_sites;
}

template <class Lattice>
constexpr auto system_state_t<Lattice>::number_clusters() const noexcept
    -> size_t
{
    return _number_clusters;
}

template <class Lattice>
constexpr auto system_state_t<Lattice>::max_cluster_size() const noexcept
    -> size_t
{
    return _max_cluster_size;
}

template <class Lattice>
    constexpr auto system_state_t<Lattice>::has_wrapped() const
    & noexcept -> gsl::span<bool const>
{
    return {_has_wrapped};
}

template <class Lattice>
constexpr auto system_state_t<Lattice>::lattice() const noexcept
    -> Lattice const&
{
    return _lattice;
}

#if 0
template <class Lattice>
    constexpr auto system_state_t<Lattice>::sa_buffers()
    & noexcept -> sa_buffers_t&
{
    return _sa_buffers;
}

template <class Lattice>
    constexpr auto system_state_t<Lattice>::energy_buffers()
    & noexcept -> energy_buffers_t&
{
    return _energy_buffers;
}
#endif

template <class Lattice>
auto system_state_t<Lattice>::rng_stream() const -> VSLStreamStatePtr
{
    return random_stream();
}

// [Spins] {{{
template <class Lattice>
auto system_state_t<Lattice>::set_angle(size_t const  i,
                                        angle_t const new_angle) TCM_NOEXCEPT
    -> void
{
    TCM_ASSERT(0 <= i && i < max_number_sites(), "Index out of bounds");
    // TCM_ASSERT(!_particles[i].is_empty(), "Site is empty");
    _angles[i] = new_angle;
    _S_x[i]    = std::cos(static_cast<float>(new_angle));
    _S_y[i]    = std::sin(static_cast<float>(new_angle));
}

template <class Lattice>
template <class RAIter>
auto system_state_t<Lattice>::set_angle(RAIter first, RAIter last,
                                        angle_t const new_angle) TCM_NOEXCEPT
    -> void
{
    auto const S_x = std::cos(static_cast<float>(new_angle));
    auto const S_y = std::sin(static_cast<float>(new_angle));
    std::for_each(first, last, [new_angle, S_x, S_y, this](auto const i) {
        TCM_ASSERT(0 <= i && i < max_number_sites(), "Index out of bounds");
        _angles[i] = new_angle;
        _S_x[i]    = S_x;
        _S_y[i]    = S_y;
    });
}

template <class Lattice>
template <class RAIter>
auto system_state_t<Lattice>::rotate(RAIter first, RAIter last,
                                     angle_t const angle) noexcept -> void
{
    std::for_each(first, last,
                  [angle, this](auto const i) { rotate(i, angle); });
}
// [Spins] }}}

// [Magnetisation] {{{
namespace detail {
inline auto magnetisation(size_t const n, float const* __restrict__ S_x,
                          float const* __restrict__ S_y) TCM_NOEXCEPT
    -> std::array<float, 2>
{
    constexpr auto alignment = 64ul;
    TCM_ASSERT(reinterpret_cast<std::uintptr_t>(S_x) % alignment == 0,
               "S_x is not aligned properly");
    TCM_ASSERT(reinterpret_cast<std::uintptr_t>(S_y) % alignment == 0,
               "S_x is not aligned properly");
    S_x = reinterpret_cast<float const*>(
        __builtin_assume_aligned(S_x, alignment));
    S_y = reinterpret_cast<float const*>(
        __builtin_assume_aligned(S_y, alignment));

    auto m_x = 0.0f;
    auto m_y = 0.0f;

#pragma omp simd
    for (auto i = size_t{0}; i < n; ++i) {
        m_x += S_x[i];
        m_y += S_y[i];
    }
    return {m_x, m_y};
}
} // namespace detail

template <class Lattice>
auto system_state_t<Lattice>::magnetisation() const -> std::array<float, 2>
{
    return detail::magnetisation(max_number_sites(), _S_x.data(), _S_y.data());
}
// [Magnetisation] }}}

// [Adding sites] {{{
template <class Lattice>
auto system_state_t<Lattice>::analyse_site(size_t const i) TCM_NOEXCEPT
    -> neighbour_stats_t const&
{
    TCM_ASSERT(0 <= i && i < max_number_sites(), "Index out of bounds");
    auto& result = *_neighbour_stats;
    result.reset();

    for (auto const _j : ::TCM_NAMESPACE::neighbours(_lattice, i)) {
        // std::fprintf(stderr, "Neighbour %li\n", _j);
        if (_j < 0) { result.insert(static_cast<int>(-_j)); }
        else {
            auto const j = static_cast<size_t>(_j);
            if (!_particles[j].is_empty()) {
                result.insert(j, std::addressof(get_magnetic_cluster(j)));
            }
        }
    }

    result.sort();
    return result;
}

template <class Lattice>
auto system_state_t<Lattice>::find_root_index(size_t i) -> size_t
{
    TCM_ASSERT(0 <= i && i < max_number_sites(), "Index out of bounds");
    TCM_ASSERT(!_particles[i].is_empty(),
               "An empty site does not belong to a cluster.");
    // A stack with enough storage for 100 elements.
    static thread_local auto path = []() {
        using workspace_type = std::vector<size_t>;
        workspace_type workspace;
        workspace.reserve(100);
        return std::stack<size_t, workspace_type>{std::move(workspace)};
    }();
    TCM_ASSERT(path.empty(), "Bug! The path stack should initially be empty.");

    // Moving up the tree.
    while (!_particles[i].is_root()) {
        path.push(i);
        i = _particles[i].parent_index();
    }
    auto const root = i;

    // Path compression
    while (!path.empty()) {
        _particles[path.top()].parent_index(root);
        path.pop();
    }
    return root;
}

template <class Lattice>
auto system_state_t<Lattice>::connect_and_merge(magnetic_cluster_type& m1,
                                                magnetic_cluster_type& m2)
    -> void
{
    auto& g1 = get_geometric_cluster(m1);
    auto& g2 = get_geometric_cluster(m1);
    if (&g1 == &g2) { g1.connect(m1, m2); }
    else if (g1.size() >= g2.size()) {
        g1.merge_and_connect({{&m1}, {&m2}}, g2);
    }
    else {
        g2.merge_and_connect({{&m2}, {&m1}}, g1);
    }
}

template <class Lattice>
auto system_state_t<Lattice>::connect(size_t i, size_t j) -> void
{
    TCM_ASSERT(!_particles[i].is_empty() && !_particles[j].is_empty(),
               "Only non-empty sites can be connected.");
    TCM_ASSERT(_clusters[i] != nullptr && _clusters[j] != nullptr, "");
    auto root_big   = find_root_index(i);
    auto root_small = find_root_index(j);
    // i and j belong to the same geometric cluster cluster
    if (root_big == root_small) {
        // We only need to connect i and j if they don't already belong to the
        // same magnetic cluster.
        if (_clusters[i] != _clusters[j]) {
            auto& geometric = _particles[root_big].cluster();
            geometric.connect(*_clusters[i], *_clusters[j]);
        }
        return;
    }
    // i and j belong to different geometric clusters
    if (_particles[root_big].cluster().size()
        < _particles[root_small].cluster().size()) {
        std::swap(root_big, root_small);
        std::swap(i, j);
    }
    // now `i` is part of the bigger  cluster with root `root_big`
    //     `j` is part of the smaller cluster with root `root_small`
    auto& big   = _particles[root_big].cluster();
    auto& small = _particles[root_small].cluster();
    big.merge({i, j}, small);
}

template <class Lattice>
TCM_NOINLINE auto system_state_t<Lattice>::
                  operator()(size_t const i, std::false_type /*is periodic?*/)
{
    TCM_ASSERT(0 <= i && i < max_number_sites(), "Index out of bounds");
    TCM_ASSERT(_particles[i].is_empty(), "Site already exists");

    ++_number_sites;
    auto const& neighbour_stats = analyse_site(i);
    // std::fprintf(stderr, "Adding %zu: boundaries = %i, size = %zu\n", i,
    //              neighbour_stats.boundaries(), neighbour_stats.size());

    if (neighbour_stats.size() == 0) {
        // `i` has no occupied neighbours, so we don't need to
        // merge or optimise anything.
        create_new_cluster(i, neighbour_stats.boundaries());
        return;
    }

    auto it = neighbour_stats.begin();
    if (it->cluster->size() > 1) {
        // There is at least one magnetic cluster with which `i` has more than
        // one connection. Thus `i` can be directly added to this cluster.
        _particles[i] = particle_type{from_parent_index, it->neighbours[0]};
        // TODO(twesterhout): Remove this, because the angle is optimised on
        // insertion anyway.
        set_angle(i, random_angle(_generator));
        it->cluster->insert(i);
        ++it;
        for (; it != neighbour_stats.end() && it->cluster->size() > 1; ++it) {
            // std::fprintf(stderr, "Connect and merging...\n");
            TCM_ASSERT(&get_magnetic_cluster(i) != it->cluster, "Noooo!");
            // We know
            //   * that `i` has more that one connection with `*it->cluster`, and
            //   * that `i` already belongs to another magnetic cluster.
            // Thus these two magnetic clusters should be merged (it doesn't
            // matter whether they already belong to the same geometric cluster
            // or not).
            connect_and_merge(get_magnetic_cluster(i), *it->cluster);
        }
    }
    else {
        // The is no magnetic cluster to which `i` can be added directly, so we
        // have to create a free-standing magnetic cluster first.
        create_new_cluster(i, neighbour_stats.boundaries());
    }
    for (; it != neighbour_stats.end(); ++it) {
        // std::fprintf(stderr, "Connecting...\n");
        TCM_ASSERT(it->cluster->size() == 1, "");
        connect(it->neighbours[0], i);
    }

#if 0
    size_t neighbours[max_neighbours<Lattice>()] = {};
    for (auto const j : tcm::neighbours(_lattice, i)) {
        if (j >= 0) { // Values >=0 represent neighbours
            TCM_ASSERT(number_neighbours < max_neighbours<Lattice>(),
                       "Index out of bounds.");
            neighbours[number_neighbours] = static_cast<size_t>(j);
            ++number_neighbours;
        }
        else { // Values <0 represent boundaries
            boundaries |= (-j);
        }
    }

    create_new_cluster(i, boundaries);

    // Adds edges between i and all neighbouring sites.
    for (auto neighbour :
         gsl::span<size_t const>{neighbours, number_neighbours}) {
        if (!_particles[neighbour].is_empty()) { connect(neighbour, i); }
    }
#endif
}
// [Adding sites] }}}

template <class Lattice>
auto percolate(Lattice const& lattice, size_t const n_min, size_t const n_max,
               tcm_percolation_results_t const* results,
               tcm_percolation_stats_t const*   stats) -> int
{
    TCM_ASSERT(results != nullptr, "`results` must not be NULL");
    if (n_min > n_max || n_max > ::TCM_NAMESPACE::size(lattice)) {
        return EDOM;
    }

    auto const size  = ::TCM_NAMESPACE::size(lattice);
    auto&      gen   = random_generator();
    auto const sites = enumerate_sites(size);
    shuffler_t shuffler{gsl::span<int32_t>{sites.get(), size}, gen};

    system_state_t<Lattice> state{lattice, gen};

    auto const save = [&results, &stats, &state](auto const i) -> void {
        if (results->number_clusters != nullptr) {
            results->number_clusters[i] = state.number_clusters();
        }
        if (results->max_cluster_size != nullptr) {
            results->max_cluster_size[i] = state.max_cluster_size();
        }
        auto const count = static_cast<int>(state.has_wrapped()[0])
                           + static_cast<int>(state.has_wrapped()[1])
                           + static_cast<int>(state.has_wrapped()[2]);
        if (results->has_wrapped_one != nullptr) {
            // std::fprintf(stderr, "count[%zu] = %i\n", i, count);
            results->has_wrapped_one[i] = count == 1;
        }
        if (results->has_wrapped_two != nullptr) {
            results->has_wrapped_two[i] = count == 2;
        }
        if (results->magnetisation != nullptr) {
            auto const m = state.magnetisation();
            results->magnetisation[i] =
                (i == 0) ? 0.0
                         : std::sqrt(m[0] * m[0] + m[1] * m[1])
                               / static_cast<float>(state.number_sites());
        }
        if (stats != nullptr) {
            auto const [max_magnetic_cluster_size, mean_magnetic_cluster_size,
                        max_number_children, mean_number_children] =
                state.stats();
            stats->max_magnetic_cluster_size[i]  = max_magnetic_cluster_size;
            stats->mean_magnetic_cluster_size[i] = mean_magnetic_cluster_size;
            stats->max_number_children[i]        = max_number_children;
            stats->mean_number_children[i]       = mean_number_children;
        }
    };

    // state.optimizing(false);
    auto       i     = size_t{0};
    auto       first = shuffler.begin();
    auto const last  = shuffler.end();
    for (; i < n_min && first != last; ++first, ++i) {
        state(static_cast<size_t>(*first), std::false_type{});
    }

    // state.optimizing(true);
    save(i);
    for (; i < n_max && first != last; ++first, ++i) {
        state(static_cast<size_t>(*first), std::false_type{});
        save(i);
    }
    return 0;
}

// clang-format off
template <class Function, class... Args>
TCM_FORCEINLINE
decltype(auto) should_not_throw(Function&& func, Args&&... args) noexcept
// clang-format on
{
    try {
        return std::forward<Function>(func)(std::forward<Args>(args)...);
    }
    catch (std::exception const& e) {
        std::fprintf(stderr,
                     "[percolation] An unrecoverable error occured: an "
                     "exception was "
                     "thrown in a noexcept context.\n"
                     "              Description: %s\n"
                     "              Calling terminate now.\n",
                     e.what());
        std::fflush(stderr);
        std::terminate();
    }
    catch (...) {
        std::fprintf(
            stderr,
            "[percolation] An unrecoverable error occured: an *unexpected* "
            "exception was thrown in a noexcept context.\n"
            "              Description: Not available.\n"
            "              Calling terminate now.\n");
        std::fflush(stderr);
        std::terminate();
    }
}

TCM_NAMESPACE_END

extern "C" TCM_EXPORT int
tcm_percolate_square(size_t const n_min, size_t const n_max,
                     tcm_square_lattice_t const       lattice,
                     tcm_percolation_results_t const* results,
                     tcm_percolation_stats_t const*   stats)
{
    if (results == nullptr) return EINVAL;
    return tcm::should_not_throw([&lattice, results, stats, n_min, n_max]() {
        if (lattice.periodic) { return EINVAL; }
        else {
            return tcm::percolate(lattice, n_min, n_max, results, stats);
        }
    });
}
