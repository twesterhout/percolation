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
#include "detail/utility.hpp"
#include "gensa.hpp"
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
} // namespace detail

struct energy_buffers_t {

    struct energy_fn_type {
        gsl::span<std::pair<size_t, size_t> const> fm_edges;
        gsl::span<std::pair<size_t, size_t> const> afm_edges;

        auto operator()(gsl::span<float const> const spin) TCM_NOEXCEPT -> float
        {
            return detail::energy_kernel(spin, afm_edges)
                   - detail::energy_kernel(spin, fm_edges);
        }
    };

  private:
    template <class T>
    using buffer_type =
        std::vector<T, boost::alignment::aligned_allocator<T, PAGE_SIZE>>;

    static constexpr auto empty = std::numeric_limits<size_t>::max();

    buffer_type<size_t> _g_to_l; ///< Mapping of global to local indices.
                                 ///<
    buffer_type<std::pair<size_t, size_t>>
        _fm_edges; ///< Edges (specified in local indices) between
                   ///< sites with ferromagnetic interaction
    buffer_type<std::pair<size_t, size_t>>
        _afm_edges; ///< Edges (specified in local indices) between
                    ///< sites with antiferromagnetic interaction

  public:
    explicit energy_buffers_t(size_t const size)
        : _g_to_l(size)
        , _fm_edges(size)  // NOTE: This is just an initial estimate
        , _afm_edges(size) // And this as well
    {}

    energy_buffers_t(const energy_buffers_t&)     = delete;
    energy_buffers_t(energy_buffers_t&&) noexcept = default;
    energy_buffers_t& operator=(energy_buffers_t const&) = delete;
    energy_buffers_t& operator=(energy_buffers_t&&) noexcept = default;

    template <class Lattice>
    auto energy_fn(gsl::span<size_t const> sites, Lattice const& lattice)
        -> energy_fn_type;

    auto global_to_local(size_t const i) const TCM_NOEXCEPT -> size_t
    {
        TCM_ASSERT(i < _g_to_l.size(), "Index out of bounds");
        TCM_ASSERT(_g_to_l[i] != empty, "Site does not belong to the cluster");
        return _g_to_l[i];
    }
};

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
// }}}

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

TCM_NAMESPACE_END
