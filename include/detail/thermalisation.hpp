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

#pragma once

#include "detail/lattice.hpp"
#include "detail/magnetic_cluster.hpp"
#include "detail/simulated_annealing.hpp"
#include "detail/utility.hpp"
#include <boost/align/aligned_allocator.hpp>
#include <gsl/gsl-lite.hpp>
#include <sleef.h>
#include <sys/user.h>
#include <vector>

#if !defined(__AVX__)
#    error "Thermalisation relies on AVX"
#endif

TCM_NAMESPACE_BEGIN

using size_t = std::size_t;

namespace detail {
/// Wrapping to [0, 2pi).
struct to_0_2pi_fn {
    static constexpr float one_over_two_pi = 0.15915494f;
    static constexpr float two_pi          = 6.2831855f;

    auto operator()(float const x) const noexcept -> float
    {
        return x - two_pi * std::floor(one_over_two_pi * x);
    }

    auto operator()(__m256 const x) const noexcept -> __m256
    {
        return x
               - _mm256_set1_ps(two_pi)
                     * Sleef_floorf8_avx(_mm256_set1_ps(one_over_two_pi) * x);
    }
};

#if 0
/// Performs the actual computation of energy
///
/// NOTE: Uses AVX intrinsics.
inline auto energy_kernel(
    gsl::span<float const> const                     spin,
    gsl::span<std::pair<size_t, size_t> const> const edges) TCM_NOEXCEPT
    -> float
{ // {{{
    auto const load_angles =
        [s = spin.data()](auto const* const d, auto const n)
            TCM_NOEXCEPT -> __m256 {
        switch (n) {
        case 0: return _mm256_set1_ps(0);
        case 1:
            return _mm256_set_ps(0, 0, 0, 0, 0, 0, 0, //
                                 s[d[0].first] - s[d[0].second]);
        case 2:
            return _mm256_set_ps(0, 0, 0, 0, 0, 0,               //
                                 s[d[1].first] - s[d[1].second], //
                                 s[d[0].first] - s[d[0].second]);
        case 3:
            return _mm256_set_ps(0, 0, 0, 0, 0,                  //
                                 s[d[2].first] - s[d[2].second], //
                                 s[d[1].first] - s[d[1].second], //
                                 s[d[0].first] - s[d[0].second]);
        case 4:
            return _mm256_set_ps(0, 0, 0, 0,                     //
                                 s[d[3].first] - s[d[3].second], //
                                 s[d[2].first] - s[d[2].second], //
                                 s[d[1].first] - s[d[1].second], //
                                 s[d[0].first] - s[d[0].second]);
        case 5:
            return _mm256_set_ps(0, 0, 0,                        //
                                 s[d[4].first] - s[d[4].second], //
                                 s[d[3].first] - s[d[3].second], //
                                 s[d[2].first] - s[d[2].second], //
                                 s[d[1].first] - s[d[1].second], //
                                 s[d[0].first] - s[d[0].second]);
        case 6:
            return _mm256_set_ps(0, 0,                           //
                                 s[d[5].first] - s[d[5].second], //
                                 s[d[4].first] - s[d[4].second], //
                                 s[d[3].first] - s[d[3].second], //
                                 s[d[2].first] - s[d[2].second], //
                                 s[d[1].first] - s[d[1].second], //
                                 s[d[0].first] - s[d[0].second]);
        case 7:
            return _mm256_set_ps(0,                              //
                                 s[d[6].first] - s[d[6].second], //
                                 s[d[5].first] - s[d[5].second], //
                                 s[d[4].first] - s[d[4].second], //
                                 s[d[3].first] - s[d[3].second], //
                                 s[d[2].first] - s[d[2].second], //
                                 s[d[1].first] - s[d[1].second], //
                                 s[d[0].first] - s[d[0].second]);

        case 8:
            return _mm256_set_ps(s[d[7].first] - s[d[7].second], //
                                 s[d[6].first] - s[d[6].second], //
                                 s[d[5].first] - s[d[5].second], //
                                 s[d[4].first] - s[d[4].second], //
                                 s[d[3].first] - s[d[3].second], //
                                 s[d[2].first] - s[d[2].second], //
                                 s[d[1].first] - s[d[1].second], //
                                 s[d[0].first] - s[d[0].second]);
        default: TCM_ASSERT(false, "Bug! This should never have happened.");
        } // end switch
    };

    constexpr auto vector_size = 8;
    auto const     chunks      = edges.size() / vector_size;
    auto const     rest        = edges.size() % vector_size;

    auto        energy = _mm256_set1_ps(0);
    auto const* data   = edges.data();

    for (auto i = std::size_t{0}; i < chunks; ++i, data += vector_size) {
        energy += Sleef_cosf8_u35avx(load_angles(data, vector_size));
    }
    if (rest != 0) { energy += Sleef_cosf8_u35avx(load_angles(data, rest)); }

    return energy[0] + energy[1] + energy[2] + energy[3] + energy[4] + energy[5]
           + energy[6] + energy[7] - static_cast<float>(rest);
} // }}}
#endif

inline auto
energy_kernel(gsl::span<float const> const                         spin,
              gsl::span<std::pair<unsigned, unsigned> const> const edges,
              gsl::span<float const> const coeffs) TCM_NOEXCEPT -> float
{ // {{{
    auto const load_chunk = [s = spin, d = edges](auto const n)
                                TCM_NOEXCEPT -> __m256 {
        TCM_ASSERT(8 * (n + 1) <= d.size(), "Index out of bounds");
        auto const i = 8 * n;
        auto const x = _mm256_set_ps(s[d[i + 7].first], s[d[i + 6].first],
                                     s[d[i + 5].first], s[d[i + 4].first],
                                     s[d[i + 3].first], s[d[i + 2].first],
                                     s[d[i + 1].first], s[d[i + 0].first]);
        auto const y = _mm256_set_ps(s[d[i + 7].second], s[d[i + 6].second],
                                     s[d[i + 5].second], s[d[i + 4].second],
                                     s[d[i + 3].second], s[d[i + 2].second],
                                     s[d[i + 1].second], s[d[i + 0].second]);
        return _mm256_sub_ps(x, y);
    };

    auto const load_part = [s = spin, d = edges](auto const n, auto const rest)
                               TCM_NOEXCEPT -> __m256 {
        TCM_ASSERT(0 < rest && rest < 8, "Precondition violated");
        TCM_ASSERT(8 * n + rest <= d.size(), "Index out of bounds");
        auto const i = 8 * n;
        switch (rest) {
        case 1:
            return _mm256_set_ps(0, 0, 0, 0, 0, 0, 0, //
                                 s[d[i + 0].first] - s[d[i + 0].second]);
        case 2:
            return _mm256_set_ps(0, 0, 0, 0, 0, 0,                       //
                                 s[d[i + 1].first] - s[d[i + 1].second], //
                                 s[d[i + 0].first] - s[d[i + 0].second]);
        case 3:
            return _mm256_set_ps(0, 0, 0, 0, 0,                          //
                                 s[d[i + 2].first] - s[d[i + 2].second], //
                                 s[d[i + 1].first] - s[d[i + 1].second], //
                                 s[d[i + 0].first] - s[d[i + 0].second]);
        case 4:
            return _mm256_set_ps(0, 0, 0, 0,                             //
                                 s[d[i + 3].first] - s[d[i + 3].second], //
                                 s[d[i + 2].first] - s[d[i + 2].second], //
                                 s[d[i + 1].first] - s[d[i + 1].second], //
                                 s[d[i + 0].first] - s[d[i + 0].second]);
        case 5:
            return _mm256_set_ps(0, 0, 0,                                //
                                 s[d[i + 4].first] - s[d[i + 4].second], //
                                 s[d[i + 3].first] - s[d[i + 3].second], //
                                 s[d[i + 2].first] - s[d[i + 2].second], //
                                 s[d[i + 1].first] - s[d[i + 1].second], //
                                 s[d[i + 0].first] - s[d[i + 0].second]);
        case 6:
            return _mm256_set_ps(0, 0,                                   //
                                 s[d[i + 5].first] - s[d[i + 5].second], //
                                 s[d[i + 4].first] - s[d[i + 4].second], //
                                 s[d[i + 3].first] - s[d[i + 3].second], //
                                 s[d[i + 2].first] - s[d[i + 2].second], //
                                 s[d[i + 1].first] - s[d[i + 1].second], //
                                 s[d[i + 0].first] - s[d[i + 0].second]);
        case 7:
            return _mm256_set_ps(0,                                      //
                                 s[d[i + 6].first] - s[d[i + 6].second], //
                                 s[d[i + 5].first] - s[d[i + 5].second], //
                                 s[d[i + 4].first] - s[d[i + 4].second], //
                                 s[d[i + 3].first] - s[d[i + 3].second], //
                                 s[d[i + 2].first] - s[d[i + 2].second], //
                                 s[d[i + 1].first] - s[d[i + 1].second], //
                                 s[d[i + 0].first] - s[d[i + 0].second]);
        default: TCM_ASSERT(false, "Bug! This should never have happened.");
        } // end switch
    };

    constexpr auto vector_size = 8;
    auto const     chunks      = edges.size() / vector_size;
    auto const     rest        = edges.size() % vector_size;

    auto energy = _mm256_set1_ps(0.0f);
    for (auto i = size_t{0}; i < chunks; ++i) {
        energy += _mm256_mul_ps(_mm256_load_ps(coeffs.data() + i * vector_size),
                                Sleef_cosf8_u35avx(load_chunk(i)));
    }
    auto energy_rest = 0.0f;
    if (rest != 0) {
        auto const begin = chunks * vector_size;
        auto const end   = begin + rest;
        for (auto i = begin; i < end; ++i) {
            energy_rest +=
                coeffs[i]
                * Sleef_cosf_u35(spin[edges[i].first] - spin[edges[i].second]);
        }
    }
    return energy[0] + energy[1] + energy[2] + energy[3] + energy[4] + energy[5]
           + energy[6] + energy[7] + energy_rest;
} // }}}
} // namespace detail

#if 0
struct energy_buffers_t {

    struct energy_fn_type {
        gsl::span<std::pair<unsigned, unsigned> const> edges;
        gsl::span<float const>                         coeffs;
        unsigned                                       number_sites;

        auto operator()(gsl::span<float const> const spin) TCM_NOEXCEPT -> float
        {
            TCM_ASSERT(spin.size() == number_sites, "");
            return detail::energy_kernel(spin, edges, coeffs);
        }
    };

  private:
    template <class T>
    using buffer_type =
        std::vector<T, boost::alignment::aligned_allocator<T, PAGE_SIZE>>;

    static constexpr auto empty = std::numeric_limits<unsigned>::max();

    buffer_type<unsigned> _g_to_l; ///< Mapping of global to local indices.
    buffer_type<std::pair<unsigned, unsigned>>
                       _edges;  ///< Edges (given in terms of local indices)
    buffer_type<float> _coeffs; ///< Couplings (+1 for antiferromagnetic
                                ///< connections and -1 for ferromagnetic ones)

  public:
    explicit energy_buffers_t(size_t const size)
        : _g_to_l(size)
        , _edges(size)  // NOTE: This is just an initial estimate
        , _coeffs(size) // And this as well
    {}

    energy_buffers_t(const energy_buffers_t&)     = delete;
    energy_buffers_t(energy_buffers_t&&) noexcept = default;
    energy_buffers_t& operator=(energy_buffers_t const&) = delete;
    energy_buffers_t& operator=(energy_buffers_t&&) noexcept = default;

    template <class System>
    auto energy_fn(System const& system) -> energy_fn_type;

  private:
    template <class System>
    auto set_initial_state(sa_buffers_t const& sa_buffers) -> energy_fn_type;

    auto global_to_local(size_t const i) const TCM_NOEXCEPT -> size_t
    {
        TCM_ASSERT(i < _g_to_l.size(), "Index out of bounds");
        TCM_ASSERT(_g_to_l[i] != empty, "Site does not belong to the cluster");
        return _g_to_l[i];
    }
};
#endif

#if 0
// {{{ energy_buffers_t::energy_fn IMPLEMENTATION
template <class Lattice>
TCM_NOINLINE auto energy_buffers_t::energy_fn(gsl::span<size_t const> sites,
                                              Lattice const&          lattice)
    -> energy_fn_type
{
    using std::begin, std::end;
    if (::TCM_NAMESPACE::size(lattice) != _g_to_l.size()) {
        _g_to_l.resize(::TCM_NAMESPACE::size(lattice));
    }

    std::fill(begin(_g_to_l), end(_g_to_l), empty);
    for (auto local_i = size_t{0}; local_i < sites.size(); ++local_i) {
        auto const global_i = sites[local_i];
        _g_to_l[global_i]   = local_i;
    }

    _fm_edges.clear();
    _afm_edges.clear();
    for (auto local_i = size_t{0}; local_i < sites.size(); ++local_i) {
        auto const global_i = sites[local_i];
        for (std::int64_t const _global_j :
             ::TCM_NAMESPACE::neighbours(lattice, global_i)) {
            // We want signed comparison here, because j may denote a
            // boundary and thus be negative. This condition ensures that
            // every edge is counted only once.
            if (static_cast<std::int64_t>(global_i) < _global_j) {
                // Here we know that _global_j > 0
                TCM_ASSERT(_global_j > 0, "");
                auto const global_j = static_cast<size_t>(_global_j);
                auto const local_j  = _g_to_l[global_j];
                if (local_j != empty) {
                    switch (::TCM_NAMESPACE::interaction(lattice, global_i,
                                                         global_j)) {
                    case interaction_t::Ferromagnetic:
                        _fm_edges.emplace_back(local_i, local_j);
                        break;
                    case interaction_t::Antiferromagnetic:
                        _afm_edges.emplace_back(local_i, local_j);
                        break;
                    } // end switch
                }
            }
        }
    }

    return {{_fm_edges}, {_afm_edges}};
}
#endif

#if 0
template <class System>
TCM_NOINLINE auto energy_buffers_t::energy_fn(System const& system)
    -> energy_fn_type
{
    using std::begin, std::end;

    auto const& lattice = system.lattice();
    if (::TCM_NAMESPACE::size(system.lattice()) != _g_to_l.size()) {
        _g_to_l.resize(::TCM_NAMESPACE::size(lattice));
    }
    std::fill(begin(_g_to_l), end(_g_to_l), empty);
    _edges.clear();
    _coeffs.clear();

    auto number_sites = unsigned{0};
    for (auto i = size_t{0}; i < ::TCM_NAMESPACE::size(lattice); ++i) {
        if (!system.is_empty(i)) {
            for (auto const _j : ::TCM_NAMESPACE::neighbours(lattice, i)) {
                if (_j >= 0) {
                    auto const j = static_cast<size_t>(_j);
                    if (!system.is_empty(j)
                        && system.get_magnetic_cluster(i)
                               == system.get_magnetic_cluster(j)) {
                        if (_g_to_l[i] == empty) {
                            _g_to_l[i] = number_sites++;
                        }
                        if (j < i) {
                            TCM_ASSERT(
                                _g_to_l[j] != empty && _g_to_l[i] != empty, "");
                            _coeffs.emplace_back(
                                ::TCM_NAMESPACE::coupling(lattice, j, i));
                            _edges.emplace_back(_g_to_l[j], _g_to_l[i]);
                        }
                    }
                }
            }
        }
    }

    return {{_edges}, {_coeffs}, number_sites};
}
// }}}
#endif

struct thermaliser_t {

    struct energy_fn_type {
        gsl::span<std::pair<unsigned, unsigned> const> edges;
        gsl::span<float const>                         coeffs;
        size_t                                         number_sites;

        auto operator()(gsl::span<float const> const spin) TCM_NOEXCEPT -> float
        {
            TCM_ASSERT(spin.size() == number_sites, "");
            return detail::energy_kernel(spin, edges, coeffs);
        }
    };

  private:
    template <class T>
    using buffer_type =
        std::vector<T, boost::alignment::aligned_allocator<T, PAGE_SIZE>>;

    static constexpr auto empty = std::numeric_limits<unsigned>::max();

    sa_buffers_t          _sa_buffers; ///< Simulated Annealing buffers.
    buffer_type<unsigned> _l_to_g; ///< Mapping from local to global indices.
    buffer_type<unsigned> _g_to_l; ///< Mapping from global to local indices.
    buffer_type<std::pair<unsigned, unsigned>>
                       _edges;  ///< Edges (given in terms of local indices)
    buffer_type<float> _coeffs; ///< Couplings (+1 for antiferromagnetic
                                ///< connections and -1 for ferromagnetic ones)

  public:
    explicit thermaliser_t(size_t const size)
        : _sa_buffers{size}
        , _l_to_g(size)
        , _g_to_l(size)
        , _edges(size)  // NOTE: This is just an initial estimate
        , _coeffs(size) // And this as well
    {}

    thermaliser_t(const thermaliser_t&)     = delete;
    thermaliser_t(thermaliser_t&&) noexcept = default;
    thermaliser_t& operator=(thermaliser_t const&) = delete;
    thermaliser_t& operator=(thermaliser_t&&) noexcept = default;

    template <class System>
    TCM_NOINLINE auto thermalise(magnetic_cluster_t<System>& cluster,
                                 sa_pars_t const& parameters) -> void;

    template <class System>
    TCM_NOINLINE auto thermalise(System& system, sa_pars_t const& parameters)
        -> void;

  private:
    template <class System>
    TCM_NOINLINE auto reset(System const& system) -> void;

    template <class System>
    TCM_NOINLINE auto reset(magnetic_cluster_t<System> const& cluster) -> void;

    template <class System>
    TCM_NOINLINE auto store(gsl::span<float const>      best,
                            magnetic_cluster_t<System>& cluster) -> void;

    template <class System>
    TCM_NOINLINE auto store(gsl::span<float const> best, System& system)
        -> void;
};

template <class System>
auto thermaliser_t::thermalise(magnetic_cluster_t<System>& cluster,
                               sa_pars_t const&            parameters) -> void
{
    reset(cluster);
    auto&      system    = cluster.system();
    auto       energy_fn = energy_fn_type{{_edges}, {_coeffs}, _l_to_g.size()};
    auto       wrap_fn   = detail::to_0_2pi_fn{};
    sa_chain_t chain{_sa_buffers, parameters, energy_fn, wrap_fn,
                     system.rng_stream()};
    for (auto i = 0u; i < parameters.n; ++i) {
        chain();
    }
    store(chain.best(), system);
}

template <class System>
auto thermaliser_t::thermalise(System& system, sa_pars_t const& parameters)
    -> void
{
    reset(system);
    auto       energy_fn = energy_fn_type{{_edges}, {_coeffs}, _l_to_g.size()};
    auto       wrap_fn   = detail::to_0_2pi_fn{};
    sa_chain_t chain{_sa_buffers, parameters, energy_fn, wrap_fn,
                     system.rng_stream()};
    for (auto i = 0u; i < parameters.n; ++i) {
        chain();
    }
    store(chain.best(), system);
}

template <class System>
auto thermaliser_t::reset(magnetic_cluster_t<System> const& cluster) -> void
{
    auto const& system  = cluster.system();
    auto const& lattice = system.lattice();
    if (::TCM_NAMESPACE::size(lattice) != _g_to_l.size()) {
        _g_to_l.resize(::TCM_NAMESPACE::size(lattice));
    }
    std::fill(begin(_g_to_l), end(_g_to_l), empty);
    _l_to_g.clear();
    _edges.clear();
    _coeffs.clear();

    for (auto const i : cluster.sites()) {
        if (!system.is_empty(i)) {
            for (auto const _j : ::TCM_NAMESPACE::neighbours(lattice, i)) {
                if (_j >= 0) {
                    auto const j = static_cast<unsigned>(_j);
                    if (!system.is_empty(j)
                        && (&system.get_magnetic_cluster(j) == &cluster)) {
                        if (_g_to_l[i] == empty) {
                            _g_to_l[i] = static_cast<unsigned>(_l_to_g.size());
                            _l_to_g.push_back(i);
                        }
                        if (j < i) {
                            TCM_ASSERT(
                                _g_to_l[j] != empty && _g_to_l[i] != empty, "");
                            _coeffs.push_back(
                                ::TCM_NAMESPACE::coupling(lattice, j, i));
                            _edges.emplace_back(_g_to_l[j], _g_to_l[i]);
                        }
                    }
                }
            }
        }
    }

    _sa_buffers.resize(_l_to_g.size());
    auto initial = _sa_buffers.current();
    for (auto const i : _l_to_g) {
        initial[i] = system.get_angle(i);
    }
}

template <class System> auto thermaliser_t::reset(System const& system) -> void
{
    auto const& lattice = system.lattice();
    if (::TCM_NAMESPACE::size(lattice) != _g_to_l.size()) {
        _g_to_l.resize(::TCM_NAMESPACE::size(lattice));
    }
    std::fill(begin(_g_to_l), end(_g_to_l), empty);
    _l_to_g.clear();
    _edges.clear();
    _coeffs.clear();

    for (auto i = 0u; i < ::TCM_NAMESPACE::size(lattice); ++i) {
        if (!system.is_empty(i)) {
            for (auto const _j : ::TCM_NAMESPACE::neighbours(lattice, i)) {
                if (_j >= 0) {
                    auto const j = static_cast<unsigned>(_j);
                    if (!system.is_empty(j)
                        && system.get_magnetic_cluster(i)
                               == system.get_magnetic_cluster(j)) {
                        if (_g_to_l[i] == empty) {
                            _g_to_l[i] = static_cast<unsigned>(_l_to_g.size());
                            _l_to_g.push_back(i);
                        }
                        if (j < i) {
                            TCM_ASSERT(
                                _g_to_l[j] != empty && _g_to_l[i] != empty, "");
                            _coeffs.push_back(
                                ::TCM_NAMESPACE::coupling(lattice, j, i));
                            _edges.emplace_back(_g_to_l[j], _g_to_l[i]);
                        }
                    }
                }
            }
        }
    }

    _sa_buffers.resize(_l_to_g.size());
    auto initial = _sa_buffers.current();
    for (auto const i : _l_to_g) {
        initial[i] = system.get_angle(i);
    }
}

template <class System>
auto thermaliser_t::store(gsl::span<float const>      best,
                          magnetic_cluster_t<System>& cluster) -> void
{
    auto const& system = cluster.system();
    for (auto i = 0u; i < _l_to_g.size(); ++i) {
        system.set_angle(_l_to_g[i], angle_t{best[i]});
    }
}

template <class System>
auto thermaliser_t::store(gsl::span<float const> best, System& system) -> void
{
    for (auto i = 0u; i < _l_to_g.size(); ++i) {
        system.set_angle(_l_to_g[i], angle_t{best[i]});
    }
}

#if 0
template <class EnergyFn, class InitFn>
inline auto optimise(EnergyFn&& energy, sa_pars_t const& parameters,
                     sa_buffers_t& buffers, VSLStreamStatePtr stream,
                     InitFn&& init_fn)
{
    std::forward<InitFn>(init_fn)(buffers.current());
    sa_chain_t chain{buffers, parameters, std::forward<EnergyFn>(energy),
                     detail::to_0_2pi_fn{}, stream};
    for (auto i = 0u; i < parameters.n; ++i) {
        chain();
    }
    return chain.best();
}
#endif

TCM_NAMESPACE_END
