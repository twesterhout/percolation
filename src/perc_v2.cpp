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

template <class MagneticCluster, size_t N> struct neighbour_stats_t {
    using magnetic_cluster_type = MagneticCluster;
    using value_type =
        std::tuple<magnetic_cluster_type*, gsl::span<size_t const>>;

  private:
    alignas(64) std::array<
        magnetic_cluster_type*,
        N> _clusters; ///< Magnetic clusters to which we are connected.
    alignas(64) std::array<
        size_t, N> _counts; ///< Number of connections with magnetic clusters.
    alignas(64) std::array<size_t,
                           N * N> _sites; ///< Sites to which we are connected
                                          ///< in each magnetic cluster.
    int      _boundaries;                 ///< Boundaries bitmask.
    unsigned _size; ///< Total number of magnetic clusters.

    struct const_iterator_t;
    friend struct const_iterator_t;

  public:
    neighbour_stats_t() TCM_NOEXCEPT { reset(); }

    neighbour_stats_t(neighbour_stats_t const&) = delete;
    neighbour_stats_t(neighbour_stats_t&&)      = delete;
    neighbour_stats_t& operator=(neighbour_stats_t const&) = delete;
    neighbour_stats_t& operator=(neighbour_stats_t&&) = delete;

    /// Returns the number of magnetic clusters to which we are connected.
    constexpr auto size() const noexcept -> size_t { return _size; }
    /// Returns the boundaries bitmask
    constexpr auto boundaries() const noexcept -> int { return _boundaries; }

    auto begin() const TCM_NOEXCEPT -> const_iterator_t { return {*this, 0ul}; }
    auto end() const TCM_NOEXCEPT -> const_iterator_t { return {*this, _size}; }

    /// Resets the statistics so that a new site can be analysed.
    auto reset() TCM_NOEXCEPT
    {
        reset_array(_clusters, static_cast<magnetic_cluster_type*>(nullptr));
        reset_array(_counts, size_t{0});
        reset_array(_sites, std::numeric_limits<size_t>::max());
        _boundaries = 0;
        _size       = 0;
    }

    /// Adds neighbour `i`
    auto insert(size_t const                 i,
                magnetic_cluster_type* const cluster) TCM_NOEXCEPT -> void
    {
        TCM_ASSERT(cluster != nullptr, "`cluster` should not be nullptr");
        if (auto const index = find(cluster); index != N) {
            TCM_ASSERT(index < _size, "Hehe!");
            TCM_ASSERT(_clusters[index] == cluster, "Hehe!");
            _sites[index * N + _counts[index]] = i;
            ++_counts[index];
        }
        else {
            _clusters[_size]  = cluster;
            _sites[_size * N] = i;
            ++_counts[_size];
            ++_size;
        }
    }

    /// Adds boundary `boundary`
    constexpr auto insert(int const boundary) noexcept -> void
    {
        _boundaries |= boundary;
    }

    auto sort() TCM_NOEXCEPT -> void
    {
        // We sort in reverse
        auto const less = [this](size_t const i, size_t const j) {
            return _counts[i] > _counts[j];
        };
        auto const swap = [this](size_t const i, size_t const j) {
            std::swap(_clusters[i], _clusters[j]);
            std::swap(_counts[i], _counts[j]);
            for (auto n = size_t{0}; n < N; ++n) {
                std::swap(_sites[i * N + n], _sites[j * N + n]);
            }
        };

        // Simple bubble sort
        auto begin = size_t{0};
        auto end   = size_t{_size};
        do {
            auto new_end = begin;
            for (auto i = begin; i != end; ++i) {
                if (less(i + 1, i)) {
                    swap(i, i + 1);
                    new_end = i;
                }
            }
            end = new_end;
        } while (end - begin > 1);

#if 0
        for (auto i = size_t{0}; i < N; ++i) {
            std::fprintf(stderr, "%p, ", _clusters[i]);
        }
        std::fprintf(stderr, "\n");
#endif
    }

  private:
    struct const_iterator_t {
        using value_type =
            std::tuple<magnetic_cluster_type*, gsl::span<size_t const>>;
        using reference         = value_type const&;
        using pointer           = value_type const*;
        using difference_type   = std::ptrdiff_t;
        using iterator_category = std::input_iterator_tag;

      private:
        auto fetch() TCM_NOEXCEPT -> void
        {
            TCM_ASSERT(_obj != nullptr, "Bug! Can't fetch from a nullptr");
            TCM_ASSERT(_i < _obj->_size, "Index out of bounds");
            auto& [cluster, sites] = _value;
            cluster                = _obj->_clusters[_i];
            sites                  = _obj->sites(_i);
        }

      public:
        constexpr const_iterator_t() noexcept : _obj{nullptr}, _i{0}, _value{}
        {}

        const_iterator_t(neighbour_stats_t const& obj, size_t const i)
            : _obj{std::addressof(obj)}, _i{i}
        {
            TCM_ASSERT(i <= obj._size, "");
            if (_i != _obj->_size) { fetch(); }
        }

        constexpr const_iterator_t(const const_iterator_t&) noexcept =
            delete; // TODO(twesterhout): Fix this!
        constexpr const_iterator_t(const_iterator_t&&) noexcept = default;
        constexpr const_iterator_t&
        operator=(const_iterator_t const&) noexcept =
            delete; // TODO(twesterhout): Fix this!
        constexpr const_iterator_t&
        operator=(const_iterator_t&&) noexcept = default;

        constexpr auto operator*() const TCM_NOEXCEPT -> reference
        {
            TCM_ASSERT(_obj != nullptr, "Iterator not dereferenceable");
            TCM_ASSERT(_i < _obj->_size, "Iterator not dereferenceable");
            return _value;
        }

        constexpr auto operator-> () const TCM_NOEXCEPT -> pointer
        {
            TCM_ASSERT(_obj != nullptr, "Iterator not dereferenceable");
            TCM_ASSERT(_i < _obj->_size, "Iterator not dereferenceable");
            return std::addressof(_value);
        }

        auto operator++() TCM_NOEXCEPT -> const_iterator_t&
        {
            TCM_ASSERT(_obj != nullptr, "Iterator not incrementable");
            TCM_ASSERT(_i < _obj->_size, "Iterator not incrementable");
            ++_i;
            if (_i != _obj->_size) { fetch(); }
            return *this;
        }

        auto operator++(int) TCM_NOEXCEPT -> const_iterator_t
        {
            const_iterator_t old{*this};
            ++(*this);
            return old;
        }

        friend auto operator==(const_iterator_t const& x,
                               const_iterator_t const& y) TCM_NOEXCEPT -> bool
        {
            TCM_ASSERT(x._obj == y._obj,
                       "Can't compare iterators pointing to different objects");
            return x._i == y._i;
        }

        friend auto operator!=(const_iterator_t const& x,
                               const_iterator_t const& y) TCM_NOEXCEPT -> bool
        {
            return !(x == y);
        }

      private:
        value_type               _value;
        neighbour_stats_t const* _obj;
        size_t                   _i;
    };

    /// Returns all the sites in cluster `_clusters[i]` to which we are
    /// connected.
    auto sites(size_t const i) const TCM_NOEXCEPT -> gsl::span<size_t const>
    {
        TCM_ASSERT(i < _size, "Index out of bounds");
        return {_sites.data() + i * N, _counts[i]};
    }

    template <class T, size_t Size, size_t Alignment = 64>
    static auto reset_array(std::array<T, Size>& xs, T value) noexcept
    {
#if 0
        auto* p =
            static_cast<T*>(__builtin_assume_aligned(xs.data(), Alignment));
#    pragma omp simd
        for (auto i = size_t{0}; i < Size; ++i) {
            p[i] = value;
        }
#else
        using std::begin, std::end;
        TCM_ASSERT(reinterpret_cast<std::uintptr_t>(xs.data()) % Alignment == 0,
                   "Bug! Array is not aligned.");
        std::fill(begin(xs), end(xs), value);
#endif
    }

    auto find(magnetic_cluster_type* const cluster) noexcept -> size_t
    {
#if 0
        auto  index = N;
        auto* p     = static_cast<magnetic_cluster_type**>(
            __builtin_assume_aligned(_clusters.data(), 64));
#    pragma omp simd
        for (auto i = size_t{0}; i < N; ++i) {
            index = (p[i] == cluster) ? i : index;
        }
        return index;
#else
        using std::begin, std::end;
        return static_cast<size_t>(
            std::find(begin(_clusters), end(_clusters), cluster)
            - begin(_clusters));
#endif
    }
};

template <class Lattice> class system_state_t { // {{{

  public:
    using pool_type              = boost::pool<>;
    using magnetic_cluster_type  = magnetic_cluster_t<system_state_t>;
    using geometric_cluster_type = geometric_cluster_t<system_state_t>;
    using particle_type          = particle_t<system_state_t>;

    struct pool_deleter {
        template <class T> auto operator()(T* const p) -> void
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

    sa_buffers_t     _sa_buffers;     ///< Buffers for Simulated Annealing
    energy_buffers_t _energy_buffers; ///< Buffers for the Hamiltonians.
    std::unique_ptr<VSLStreamStatePtr, vsl_stream_deleter_t>
        _rng_stream; ///< Intel MKL's random number generator
                     ///< stream used for thermalisation.

    size_t _number_sites;     ///< Total number of sites in the system.
    size_t _number_clusters;  ///< Number of geometric clusters.
    size_t _max_cluster_size; ///< Size of the largest geometric cluster.
    bool   _has_wrapped[3];
    bool   _optimizing;
    std::unique_ptr<neighbour_stats_t>
        _neighbour_stats; ///< Buffers for analysing neighbours

  public: // geometric_cluster relies on this
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
    constexpr auto on_size_changed(geometric_cluster_type const&) noexcept
        -> system_state_t&;

    /// This is a callback that should be invoked when the "bounraries" of a
    /// geometric cluster change.
    constexpr auto on_boundaries_changed(geometric_cluster_type const&) noexcept
        -> system_state_t&;

    /// This is a callback that should be invoked when a new geometric cluster
    /// is created.
    constexpr auto on_cluster_created(geometric_cluster_type const&) noexcept
        -> system_state_t&;

    /// This is a callback that should be invoked when a geometric cluster is
    /// destroyed (i.e. merged into another).
    constexpr auto on_cluster_destroyed(geometric_cluster_type const&) noexcept
        -> system_state_t&;

    constexpr auto on_cluster_merged(geometric_cluster_type& big,
                                     geometric_cluster_type& small) noexcept
        -> system_state_t&;
    // }}}

    constexpr auto optimizing() const noexcept -> bool { return _optimizing; }

    TCM_NOINLINE auto optimizing(bool do_opt) -> void;

    constexpr auto max_number_sites() const noexcept -> size_t;
    constexpr auto number_sites() const noexcept -> size_t;
    constexpr auto number_clusters() const noexcept -> size_t;
    constexpr auto max_cluster_size() const noexcept -> size_t;
    constexpr auto has_wrapped() const & noexcept -> gsl::span<bool const>;
    constexpr auto lattice() const noexcept -> Lattice const&;
    constexpr auto sa_buffers() & noexcept -> sa_buffers_t&;
    constexpr auto energy_buffers() & noexcept -> energy_buffers_t&;
    constexpr auto rng_stream() & noexcept -> VSLStreamStatePtr;

    // [Spins] {{{
    constexpr auto get_angle(size_t const i) const TCM_NOEXCEPT -> angle_t
    {
        TCM_ASSERT(0 <= i && i < max_number_sites(), "Index out of bounds");
        TCM_ASSERT(!_particles[i].is_empty(), "Site is empty");
        return _angles[i];
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
    , _sa_buffers{size(lattice)}
    , _energy_buffers{size(lattice)}
    , _rng_stream{make_rng_stream(VSL_BRNG_ARS5)}
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

template <class Lattice>
TCM_NOINLINE auto system_state_t<Lattice>::optimizing(bool const do_opt) -> void
{
    if (!_optimizing && do_opt) {
        for (auto& x : _particles) {
            if (x.is_root()) { x.cluster().optimize_full(); }
        }
    }
    _optimizing = do_opt;
}

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

template <class Lattice>
    constexpr auto system_state_t<Lattice>::rng_stream()
    & noexcept -> VSLStreamStatePtr
{
    return *_rng_stream;
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
auto system_state_t<Lattice>::connect_and_merge(magnetic_cluster_type& x,
                                                magnetic_cluster_type& y)
    -> void
{
    auto& x_geom = get_geometric_cluster(x);
    auto& y_geom = get_geometric_cluster(y);
    if (std::addressof(x_geom) == std::addressof(y_geom)) {
        x_geom.connect(no_optimize, x, y);
    }
    else if (x_geom.size() >= y_geom.size()) {
        x_geom.merge_and_connect(
            no_optimize, {{std::addressof(x)}, {std::addressof(y)}}, y_geom);
    }
    else {
        y_geom.merge_and_connect(
            no_optimize, {{std::addressof(y)}, {std::addressof(x)}}, x_geom);
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
        if (_clusters[i] != _clusters[j]) {
            if (_optimizing) {
                _particles[root_big].cluster().connect(*_clusters[i],
                                                       *_clusters[j]);
            }
            else {
                _particles[root_big].cluster().connect(
                    no_optimize, *_clusters[i], *_clusters[j]);
            }
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

    if (_optimizing) {
        _particles[root_big].cluster().merge({i, j},
                                             _particles[root_small].cluster());
    }
    else {
        _particles[root_big].cluster().merge(no_optimize, {i, j},
                                             _particles[root_small].cluster());
    }
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
        // std::fprintf(stderr, "Doing nothing...\n");
        // `i` has no occupied neighbours, so we don't need to
        // merge or optimise anything.
        create_new_cluster(i, neighbour_stats.boundaries());
        return;
    }

    if (auto it = neighbour_stats.begin(); std::get<1>(*it).size() > 1) {
        // There is at least one magnetic cluster with which `i` has more than
        // one connection. Thus `i` can be directly added to this cluster.
        _particles[i] = particle_type{from_parent_index, std::get<1>(*it)[0]};
        set_angle(i, random_angle(_generator));
        std::get<0>(*it)->insert(no_optimize, i);
        ++it;

        bool have_connected = false;
        for (; it != neighbour_stats.end() && std::get<1>(*it).size() > 1;
             ++it) {
            // std::fprintf(stderr, "Connect and merging...\n");
            TCM_ASSERT(&get_magnetic_cluster(i) != std::get<0>(*it), "Noooo!");
            connect_and_merge(get_magnetic_cluster(i), *std::get<0>(*it));
            have_connected = true;
        }
        if (!have_connected) { get_magnetic_cluster(i).optimize_one(i); }
        else {
            get_magnetic_cluster(i).optimize_full();
        }

        for (; it != neighbour_stats.end(); ++it) {
            TCM_ASSERT(std::get<1>(*it).size() == 1, "");
            // std::fprintf(stderr, "Connecting...\n");
            connect(std::get<1>(*it)[0], i);
        }
    }
    else {
        create_new_cluster(i, neighbour_stats.boundaries());
        for (; it != neighbour_stats.end(); ++it) {
            TCM_ASSERT(std::get<1>(*it).size() == 1, "");
            // std::fprintf(stderr, "Connecting...\n");
            connect(std::get<1>(*it)[0], i);
        }
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

auto enumerate_sites(std::int32_t const n) noexcept
    -> std::unique_ptr<std::int32_t[], FreeDeleter>;

TCM_NOINLINE
auto enumerate_sites(std::int32_t const n) noexcept
    -> std::unique_ptr<std::int32_t[], FreeDeleter>
{
    TCM_ASSERT(n > 0, "Negative number of sites");
    auto const bytes_count = static_cast<std::size_t>(n) * sizeof(std::int32_t);
    auto* const p =
        reinterpret_cast<std::int32_t*>(std::aligned_alloc(64ul, bytes_count));
    if (p == nullptr) { return nullptr; }
    for (std::int32_t i = 0; i < n; ++i) {
        p[i] = i;
    }
    return {p, FreeDeleter{}};
}

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
    shuffler_t shuffler{sites.get(), static_cast<std::ptrdiff_t>(size),
                        gen}; // TODO(twesterhout): Fix this to use gsl::span

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

    state.optimizing(false);
    auto       i     = size_t{0};
    auto       first = shuffler.begin();
    auto const last  = shuffler.end();
    for (; i < n_min && first != last; ++first, ++i) {
        state(static_cast<size_t>(*first), std::false_type{});
    }

    state.optimizing(true);
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
