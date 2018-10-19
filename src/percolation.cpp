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

#include "percolation.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stack>
#include <vector>

#include <immintrin.h>

using index_type = std::int64_t;

union Position {
    struct {
        std::int32_t x;
        std::int32_t y;
        std::int32_t z;
        std::int32_t _padding;
    };
    __m128i raw;

    constexpr Position() noexcept : x{0}, y{0}, z{0}, _padding{0} {}

    constexpr Position(
        index_type const x_, index_type const y_, index_type const z_) noexcept
        : x{static_cast<std::int32_t>(x_)}
        , y{static_cast<std::int32_t>(y_)}
        , z{static_cast<std::int32_t>(z_)}
        , _padding{0}
    {
    }

    auto abs() const noexcept -> Position
    {
        return {std::abs(x), std::abs(y), std::abs(z)};
    }

    auto sum() const noexcept -> std::int32_t { return x + y + z; }
};

inline auto operator<<(std::ostream& out, Position const& x) -> std::ostream&
{
    if (x._padding == 0) {
        assert(x._padding == 0);
        return out << "(" << x.x << ", " << x.y << ", " << x.z << ")";
    } else {
        out << "(" << x.x << ", " << x.y << ", " << x.z << ", " << x._padding << ")";
        assert(x._padding == 0);
        return out;
    }
}

inline auto operator+(Position const x, Position const y) noexcept -> Position
{
    Position dst;
    dst.raw = _mm_add_epi32(x.raw, y.raw);
    return dst;
}

inline auto operator+=(Position& x, Position const y) noexcept -> Position&
{
    x.raw = _mm_add_epi32(x.raw, y.raw);
    return x;
}

inline auto operator==(Position const a, Position const b) noexcept -> bool
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

inline auto operator!=(Position const a, Position const b) noexcept -> bool
{
    return !(a == b);
}

inline auto operator-(Position const x) noexcept -> Position
{
    Position dst;
    dst.raw = _mm_sub_epi32(_mm_set1_epi32(0), x.raw);
    return dst;
}

inline auto operator-(Position const x, Position const y) noexcept -> Position
{
    Position dst;
    dst.raw = _mm_sub_epi32(x.raw, y.raw);
    return dst;
}

inline auto operator-=(Position& x, Position const y) noexcept -> Position&
{
    x.raw = _mm_sub_epi32(x.raw, y.raw);
    return x;
}

template <class T, class Generator>
class Shuffler {
    class Iterator {
        using value_type        = T;
        using difference_type   = std::ptrdiff_t;
        using reference         = T;
        using pointer           = T const*;
        using iterator_category = std::input_iterator_tag;

        friend class Shuffler;

      private:
        auto generate_one()
        {
            assert(_shuffler != nullptr && _shuffler->_size > _i
                   && "There are no more elements to generate.");
            using Dist   = std::uniform_int_distribution<difference_type>;
            using Params = Dist::param_type;
            using std::swap;
            Dist           uid;
            std::ptrdiff_t shift =
                uid(_shuffler->_gen, Params{0, _shuffler->_size - 1 - _i});
            if (shift != 0) {
                swap(_shuffler->_data[_i], _shuffler->_data[_i + shift]);
            }
        }

        Iterator(Shuffler& obj) noexcept : _shuffler{&obj}, _i{0}
        {
            if (_shuffler->_size > 1) { generate_one(); }
        }

        Iterator(Shuffler& obj, difference_type i) noexcept
            : _shuffler{&obj}, _i{i}
        {
            assert(i == _shuffler->_size);
        }

      public:
        constexpr Iterator() noexcept : _shuffler{nullptr} {}
        constexpr Iterator(Iterator const&) noexcept = default;
        constexpr Iterator(Iterator&&) noexcept      = default;
        constexpr Iterator& operator=(Iterator const&) noexcept = default;
        constexpr Iterator& operator=(Iterator&&) noexcept = default;

        constexpr auto operator*() const noexcept -> T
        {
            assert(_shuffler != nullptr && _shuffler->_size > _i
                   && "Iterator is not dereferenceable.");
            return _shuffler->_data[_i];
        }

        constexpr auto operator-> () const noexcept -> T const*
        {
            assert(_shuffler != nullptr && _shuffler->_size > _i
                   && "Iterator is not dereferenceable.");
            return _shuffler->_data + _i;
        }

        auto operator++() -> Iterator&
        {
            assert(_shuffler != nullptr && _shuffler->_size > _i
                   && "Iterator is not incrementable.");
            ++_i;
            if (_shuffler->_size > _i) { generate_one(); }
            return *this;
        }

        auto operator++(int) -> Iterator
        {
            auto temp = *this;
            ++(*this);
            return temp;
        }

        auto operator==(Iterator const& other) const noexcept -> bool
        {
            assert(_shuffler == other._shuffler);
            return _i == other._i;
        }

        auto operator!=(Iterator const& other) const noexcept -> bool
        {
            assert(_shuffler == other._shuffler);
            return _i != other._i;
        }

      private:
        Shuffler*       _shuffler;
        difference_type _i;
    };

  public:
    constexpr Shuffler(
        T* data, std::ptrdiff_t const size, Generator& gen) noexcept
        : _data{data}, _size{size}, _gen{gen}
    {
    }

    constexpr Shuffler(Shuffler const&) noexcept = default;
    constexpr Shuffler(Shuffler&&) noexcept      = default;
    constexpr Shuffler& operator=(Shuffler const&) noexcept = delete;
    constexpr Shuffler& operator=(Shuffler&&) noexcept = delete;

    auto begin() noexcept -> Iterator { return {*this}; }
    auto end() noexcept -> Iterator { return {*this, _size}; }

  private:
    // TODO(twesterhout): This should be replaced by span<T>, but I don't want
    // to add this extra dependency just for this...
    T*             _data;
    std::ptrdiff_t _size;
    Generator&     _gen;
};

// clang-format off
#define SET_NEIGHBOURS(nn, x, y, z, length,                                    \
                       x0, x1, x2, x3, x4, x5,                                 \
                       y0, y1, y2, y3, y4, y5,                                 \
                       z0, z1, z2, z3, z4, z5)                                 \
    do {                                                                       \
        __m256i xs         = _mm256_set_epi64x(x3, x2, x1, x0);                \
        __m256i ys         = _mm256_set_epi64x(y3, y2, y1, y0);                \
        __m256i zs         = _mm256_set_epi64x(z3, z2, z1, z0);                \
        __m256i neighbours = (length * length) * zs + length * ys + xs;        \
        _mm256_storeu_pd(reinterpret_cast<double*>(std::data(nn)),             \
            reinterpret_cast<__m256d>(neighbours));                            \
        xs         = _mm256_set_epi64x(0, 0, x5, x4);                          \
        ys         = _mm256_set_epi64x(0, 0, y5, y4);                          \
        zs         = _mm256_set_epi64x(0, 0, z5, z4);                          \
        neighbours = (length * length) * zs + length * ys + xs;                \
        _mm256_storeu_pd(reinterpret_cast<double*>(std::data(nn) + 4),         \
            reinterpret_cast<__m256d>(neighbours));                            \
    } while(false)
// clang-format on

inline auto safe_set_neighbours(std::int64_t (&nn)[8], std::int64_t const x,
    std::int64_t const y, std::int64_t const z,
    std::int64_t const length) noexcept
{
    auto const find_neighbours = [length](auto const i, auto* is) noexcept
    {
        if (i == 0) {
            is[0] = length - 1;
            is[1] = i + 1;
        }
        else if (i == length - 1) {
            is[0] = i - 1;
            is[1] = 0;
        }
        else {
            is[0] = i - 1;
            is[1] = i + 1;
        }
    };
    std::int64_t nn_x[2];
    std::int64_t nn_y[2];
    std::int64_t nn_z[2];
    find_neighbours(x, nn_x);
    find_neighbours(y, nn_y);
    find_neighbours(z, nn_z);
    // clang-format off
    SET_NEIGHBOURS(nn, x, y, x, length,
                   nn_x[0], nn_x[1], x, x, x, x,
                   y, y, nn_y[0], nn_y[1], y, y,
                   z, z, z, z, nn_z[0], nn_z[1]);
    // clang-format on
}

#undef SET_NEIGHBOURS

inline auto number_sites(cubic_lattice_t const& lattice) noexcept -> std::size_t
{
    auto const n = static_cast<std::size_t>(lattice.length);
    return n * n * n;
}

extern "C" int cubic_lattice_init(cubic_lattice_t* lattice)
{
    constexpr std::int64_t max_length = 2642245;
    if (lattice == nullptr) { return EINVAL; }
    if (lattice->length <= 2 || lattice->length > max_length) { return EDOM; }
    auto const n        = number_sites(*lattice);
    lattice->neighbours = reinterpret_cast<std::int64_t(*)[8]>(
        std::aligned_alloc(64ul, sizeof(std::int64_t[8]) * n));
    if (lattice->neighbours == nullptr) { return ENOMEM; }
    index_type i = 0;
    for (index_type z = 0; z < lattice->length; ++z) {
        for (index_type y = 0; y < lattice->length; ++y) {
            for (index_type x = 0; x < lattice->length; ++x, ++i) {
                safe_set_neighbours(
                    lattice->neighbours[i], x, y, z, lattice->length);
            }
        }
    }
    return 0;
}

extern "C" void cubic_lattice_deinit(cubic_lattice_t* lattice)
{
    assert(lattice != nullptr);
    if (lattice->neighbours != nullptr) {
        std::free(lattice->neighbours);
        lattice->neighbours = nullptr;
    }
}

/// \brief Given a site position, returns its index.
inline constexpr auto position_to_index(
    cubic_lattice_t const& lattice, Position const p) noexcept -> std::int64_t
{
    return lattice.length * lattice.length * p.z + lattice.length * p.y + p.x;
}

/// \brief Given a size index, returns its position.
inline auto index_to_position(
    cubic_lattice_t const& lattice, index_type const i) noexcept -> Position
{
    auto const [z, rem] = std::div(i, lattice.length * lattice.length);
    auto const [y, x]   = std::div(rem, lattice.length);
    return {x, y, z};
}

/// \brief Given positions of two sites, returns the shortest vector from the first
/// to the second.
inline auto path_from_to(cubic_lattice_t const& lattice, std::int64_t const a,
    std::int64_t const b) noexcept -> Position
{
    // clang-format off
    auto path = index_to_position(lattice, b) - index_to_position(lattice, a);
    auto const max         = _mm_set1_epi32(lattice.length / 2);
    auto const zero        = reinterpret_cast<__m128>(_mm_set1_epi32(0));
    auto const length      = reinterpret_cast<__m128>(_mm_set1_epi32(lattice.length));
    auto const to_sub_mask = reinterpret_cast<__m128>(_mm_cmpgt_epi32(path.raw, max));
    auto const to_add_mask = reinterpret_cast<__m128>(
        _mm_cmplt_epi32(path.raw, _mm_sub_epi32(_mm_set1_epi32(0), max)));
    auto term = reinterpret_cast<__m128i>(_mm_blendv_ps(zero, length, to_add_mask));
    path.raw = _mm_add_epi32(path.raw, term);
    term = reinterpret_cast<__m128i>(_mm_blendv_ps(zero, length, to_sub_mask));
    path.raw = _mm_sub_epi32(path.raw, term);
    return path;
    // clang-format on
}

struct FreeDeleter {
    template <class T>
    auto operator()(T* p) const noexcept
    {
        std::free(p);
    }
};

struct State {
    using index_type = std::int64_t;
    static constexpr index_type empty_value =
        std::numeric_limits<index_type>::max();

    State(cubic_lattice_t const&);
    State(State const&)     = delete;
    State(State&&) noexcept = default;
    State& operator=(State const&) = delete;
    State& operator=(State&&) noexcept = default;

    constexpr auto number_sites() const noexcept { return _number_sites; }
    constexpr auto number_clusters() const noexcept { return _number_clusters; }
    constexpr auto max_cluster_size() const noexcept { return _max_cluster_size; }
    constexpr auto is_percolating() const noexcept { return _is_percolating; }

    /// \brief Returns whether the site at index `i` is occupied.
    auto is_empty(index_type const i) const noexcept -> bool
    {
        assert(0 <= i && i < _size && "Index out of bounds.");
        return _storage[static_cast<std::size_t>(i)] == empty_value;
    }

    /// \brief Returns whether the site at index `i` is the root of a cluster.
    auto is_root(index_type const i) const noexcept -> bool
    {
        assert(0 <= i && i < _size && "Index out of bounds.");
        return _storage[static_cast<std::size_t>(i)] < 0;
    }

    /// \brief Returns the parent of the site `i`.
    auto parent(index_type const i) const noexcept -> index_type
    {
        assert(!is_root(i) && "Root element has no parent.");
        return _storage[static_cast<std::size_t>(i)];
    }

    auto set_parent(index_type const i, index_type const j) const noexcept
        -> void
    {
        _storage[static_cast<std::size_t>(i)] = j;
    }

    /// \brief Returns the size of the cluster `i`. Site `i` must be the root.
    auto size(index_type const i) const noexcept -> index_type
    {
        assert(is_root(i) && "Only the root element stores the cluster size.");
        return -_storage[static_cast<std::size_t>(i)];
    }

    /// \brief Returns the root of the cluster that site `i` belongs to.
    auto find_root(index_type const i) const -> index_type;

    /// \brief Adds an edge between sites i and j. If they belong to different
    /// clusters, the clusters are merged.
    auto merge(index_type i, index_type j) -> void;

    /// \brief Adds a new site to the system.
    auto operator()(index_type const i)
    {
        assert(is_empty(i) && "Site already exists.");
        // Create a new cluster of size 1.
        ++_number_sites;
        ++_number_clusters;
        _storage[static_cast<std::size_t>(i)] = -1;

        // Update statistics
        if (size(i) > _max_cluster_size) { _max_cluster_size = size(i); }

        // Adds edges between i and all neighbouring sites.
        for (auto const j : _lattice.neighbours[i]) {
            if (!is_empty(j)) { merge(j, i); }
        }
    }

  private:
    std::unique_ptr<index_type[], FreeDeleter> _storage;
    std::unique_ptr<Position[], FreeDeleter>   _displacements;
    cubic_lattice_t const&                     _lattice;
    std::int64_t                               _size;
    std::int64_t                               _number_sites;
    std::int64_t                               _number_clusters;
    std::int64_t                               _max_cluster_size;
    bool                                       _is_percolating;
};

State::State(cubic_lattice_t const& lattice)
    : _storage{}
    , _displacements{}
    , _lattice{lattice}
    , _size{static_cast<index_type>(::number_sites(lattice))}
    , _number_sites{0}
    , _number_clusters{0}
    , _max_cluster_size{0}
    , _is_percolating{false}
{
    auto const n = static_cast<std::size_t>(_size);
    auto       storage = std::unique_ptr<index_type[], FreeDeleter>{
        reinterpret_cast<index_type*>(
            std::aligned_alloc(64ul, sizeof(index_type) * n))};
    auto displacements =
        std::unique_ptr<Position[], FreeDeleter>{reinterpret_cast<Position*>(
            std::aligned_alloc(64ul, sizeof(Position) * n))};
    if (storage == nullptr || displacements == nullptr) {
        throw std::bad_alloc{};
    }
    _storage       = std::move(storage);
    _displacements = std::move(displacements);
    for (std::size_t i = 0; i < n; ++i) {
        _storage[i] = empty_value;
    }
    for (std::size_t i = 0; i < n; ++i) {
        _displacements[i] = Position{};
    }
}

/// \brief Returns the root of the cluster that site `i` belongs to.
auto State::find_root(index_type const i) const -> index_type
{
    assert(!is_empty(i) && "An empty site does not belong to a cluster.");
    // A stack with enough storage for 100 elements.
    static auto path = []() {
        std::vector<index_type> workspace;
        workspace.reserve(100);
        return std::stack<index_type, std::vector<index_type>>{
            std::move(workspace)};
    }();
    assert(path.empty() && "BUG!");

    // Moving up the tree.
    auto root = i;
    while (!is_root(root)) {
        path.push(root);
        root = parent(root);
    }

    // Path compression
    while (!path.empty()) {
        auto const j = path.top();
        path.pop();
        _displacements[static_cast<std::size_t>(j)] +=
            _displacements[static_cast<std::size_t>(parent(j))];
        set_parent(j, root);
    }
    return root;
}

auto State::merge(index_type i, index_type j) -> void
{
    auto root_big   = find_root(i);
    auto root_small = find_root(j);

    // Same cluster
    if (root_big == root_small) {
        // A fancy trick to determine whether the cluster wraps around in at
        // least one direction.
        auto const distance = (_displacements[static_cast<std::size_t>(i)]
                               - _displacements[static_cast<std::size_t>(j)])
                                  .abs()
                                  .sum();
        if (distance != 1) { _is_percolating = true; }
        return;
    }

    // Different clusters
    if (size(root_big) < size(root_small)) {
        std::swap(root_big, root_small);
        std::swap(i, j);
    }

    // Increases the size of the bigger cluster.
    _storage[static_cast<std::size_t>(root_big)] +=
        _storage[static_cast<std::size_t>(root_small)];
    // Updates the displacement of the root of the smaller cluster. Path from
    // root_small to root_big is constructed in the following way:
    // root_small -> j -> i -> root_big.
    _displacements[static_cast<std::size_t>(root_small)] =
        -_displacements[static_cast<std::size_t>(j)]
        + path_from_to(_lattice, j, i)
        + _displacements[static_cast<std::size_t>(i)];
    // Update the parent of the root_small
    set_parent(root_small, root_big);

    // Update statistics
    if (size(root_big) > _max_cluster_size) {
        _max_cluster_size = size(root_big);
    }
    --_number_clusters;
}

// auto seed_engine();

__attribute__((noinline)) auto seed_engine()
{
    using Gen               = std::mt19937;
    constexpr std::size_t N = (Gen::word_size + 31) / 32 * Gen::state_size;
    std::uint32_t         random_data[N];
    std::random_device    source;
    std::generate(
        std::begin(random_data), std::end(random_data), std::ref(source));
    std::seed_seq seeds(std::begin(random_data), std::end(random_data));
    return Gen{seeds};
}

inline auto enumerate_sites(std::size_t const n) noexcept
    -> std::unique_ptr<std::int64_t[], FreeDeleter>
{
    auto* const p = reinterpret_cast<std::int64_t*>(
        std::aligned_alloc(64ul, sizeof(std::int64_t) * n));
    if (p == nullptr) { return nullptr; }
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
        p[i] = i;
    }
    return std::unique_ptr<std::int64_t[], FreeDeleter>{p};
}

extern "C" int percolate(cubic_lattice_t const* lattice, result_t* result)
{
    assert(lattice != nullptr && result != nullptr);
    auto const size  = number_sites(*lattice);
    auto const sites = enumerate_sites(size);
    if (sites == nullptr) { return ENOMEM; }
    try {
        // May throw
        auto     gen = seed_engine();
        Shuffler shuffler{sites.get(), static_cast<std::int64_t>(size), gen};
        // May throw bad_alloc
        State state{*lattice};

        for (auto const site : shuffler) {
            state(site);
            if (state.is_percolating()) { break; }
        }
        assert(state.is_percolating());
        result->number_sites     = state.number_sites();
        result->number_clusters  = state.number_clusters();
        result->max_cluster_size = state.max_cluster_size();
        result->is_percolating   = state.is_percolating();
        return 0;
    }
    catch (std::bad_alloc&) {
        return ENOMEM;
    }
    catch (std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        std::terminate();
    }
    catch (...) {
        std::fprintf(stderr, "Error: An unknown error occured :(\n");
        std::terminate();
    }
}



