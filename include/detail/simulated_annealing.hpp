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

#include "detail/random.hpp"
#include "config.h"
#include <boost/align/aligned_allocator.hpp>
#include <gsl/gsl-lite.hpp>
#include <mkl_cblas.h>
#include <mkl_vsl.h>
#include <cmath>
#include <functional>
#include <iostream>
#include <type_traits>

TCM_NAMESPACE_BEGIN

namespace detail {
/// Checks that the last `vslRng*` operation succeeded. Throws an exception
/// otherwise.
inline auto check_vsl_status_after_generation(int const status) -> void
{
    if (status == VSL_STATUS_OK) { return; }
    char const* const msg = [](auto code) {
        switch (code) {
        case VSL_ERROR_NULL_PTR: return "`stream` is a NULL pointer";
        case VSL_RNG_ERROR_BAD_STREAM:
            return "`stream` is not a valid random stream";
        case VSL_RNG_ERROR_BAD_UPDATE:
            return "callback function for an abstract BRNG returns an "
                   "invalid number of updated entries in a buffer, "
                   "that is, < 0 or > nmax";
        case VSL_RNG_ERROR_NO_NUMBERS:
            return "callback function for an abstract BRNG returns 0 "
                   "as the number of updated entries in a buffer.";
        case VSL_RNG_ERROR_QRNG_PERIOD_ELAPSED:
            return "period of the generator has been exceeded";
        case VSL_RNG_ERROR_NONDETERM_NRETRIES_EXCEEDED:
            return "number of retries to generate a random number by "
                   "using non-deterministic random number generator "
                   "exceeds threshold";
        case VSL_RNG_ERROR_ARS5_NOT_SUPPORTED:
            return "ARS-5 random number generator is not supported on "
                   "the CPU running the application";
        default: return "unknown error code: a bug in Intel MKL?";
        } // end switch
    }(status);
    throw std::runtime_error{msg};
}

/// Uses Intel MKL to generate numbers distributed according to
/// `Gamma(shape, scale)`.
///
/// https://en.wikipedia.org/wiki/Gamma_distribution.
template <class RealT> class buffered_gamma_distribution_t { // {{{
    static_assert(std::is_same_v<RealT, float> || std::is_same_v<RealT, double>,
                  "tcm::detail::buffered_gamma_distribution_t currently only "
                  "supports `float`s and `double`s.");

  public:
    /// Generation method.
    ///
    /// Possible values are:
    ///
    ///   * `VSL_RNG_METHOD_GAMMA_GNORM`
    ///   * `VSL_RNG_METHOD_GAMMA_GNORM_ACCURATE`
    ///
    /// See https://software.intel.com/en-us/mkl-developer-reference-c-vrnggamma
    /// for more info.
    static constexpr MKL_INT method = VSL_RNG_METHOD_GAMMA_GNORM_ACCURATE;

    static constexpr size_t default_buffer_size = 1024;

    /// Buffer type.
    ///
    /// Performance of Intel MKL's random number generation increases
    /// considerably when multiple variates a generated at once (see
    /// https://software.intel.com/en-us/mkl-vsperfdata-uniform).
    using buffer_type =
        std::vector<RealT, boost::alignment::aligned_allocator<RealT, 64u>>;

    using stream_type = VSLStreamStatePtr;

    /// Distribution parameters.
    class param_type {
        RealT _shape; ///< Shape parameter
        RealT _scale; ///< Scale parameter

      public:
        constexpr explicit param_type(RealT const shape,
                                      RealT const scale) TCM_NOEXCEPT
            : _shape{shape}
            , _scale{scale}
        {
            TCM_ASSERT(shape > 0, "`shape` must be positive");
            TCM_ASSERT(scale > 0, "`scale` must be positive");
        }

        constexpr param_type(param_type const&) noexcept = default;
        constexpr param_type(param_type&&) noexcept      = default;
        constexpr param_type& operator=(param_type const&) noexcept = default;
        constexpr param_type& operator=(param_type&&) noexcept = default;

        constexpr auto shape() const noexcept { return _shape; }
        constexpr auto scale() const noexcept { return _scale; }
    };

    static_assert(std::is_trivially_copyable_v<param_type>);

  private:
    stream_type _stream; ///< Intel MKL random number generator stream
                         ///<
    buffer_type _buffer; ///< Output buffer where the generated
                         ///< numbers are written to
    size_t _i; ///< Current position in `_buffer`. `_i == _buffer.size()`
               ///< indicates that the buffer needs to be refilled.
    param_type _params; ///< Shape and scale parameters
                        ///<

  public:
    /// Constructs a new Gamma distribution and allocates memory for internal
    /// buffers.
    buffered_gamma_distribution_t(
        stream_type const stream, RealT const alpha, RealT const beta,
        size_t const buffer_size = default_buffer_size)
        : _stream{stream}
        , _buffer(buffer_size)
        , _i{buffer_size}
        , _params{alpha, beta}
    {
        TCM_ASSERT(stream != nullptr, "`stream` should not be null");
        TCM_ASSERT(buffer_size > 0, "buffer should not be empty");
    }

    /// Deleted copy constructor to avoid unintentionally copying potentially
    /// big buffers.
    buffered_gamma_distribution_t(buffered_gamma_distribution_t const&) =
        delete;

    /// Move constructor.
    buffered_gamma_distribution_t(buffered_gamma_distribution_t&&) noexcept =
        default;

    /// Deleted copy assignment operator to avoid unintentionally copying
    /// potentially big buffers.
    buffered_gamma_distribution_t&
    operator=(buffered_gamma_distribution_t const&) = delete;

    /// Move assignment operator.
    buffered_gamma_distribution_t&
    operator=(buffered_gamma_distribution_t&&) noexcept = default;

    constexpr auto param() const noexcept -> param_type { return _params; }

    constexpr auto param(param_type const& params) noexcept -> void
    {
        _params = params;
        _i      = _buffer.size();
    }

#if 0 // This is probably too low-level and should not be exposed.
    constexpr auto buffer() noexcept -> gsl::span<RealT> { return {_buffer}; }

    constexpr auto buffer() const noexcept -> gsl::span<RealT const>
    {
        return {_buffer};
    }
#endif

    constexpr auto stream() const noexcept -> VSLStreamStatePtr
    {
        return _stream;
    }

    auto resize(size_t const buffer_size) -> void
    {
        TCM_ASSERT(buffer_size > 0, "buffer should not be empty");
        // TODO(twesterhout): We could implement an optimisation here which
        // would copy the unused part of the buffer. This could potentially
        // avoid generating some random numbers.
        _buffer.resize(buffer_size);
        _i = buffer_size;
    }

  private:
    static auto vsl_generate(VSLStreamStatePtr const stream, MKL_INT const n,
                             float* const r, float const alpha, float const a,
                             float const beta) -> int
    {
        return vsRngGamma(method, stream, n, r, alpha, a, beta);
    }

    static auto vsl_generate(VSLStreamStatePtr const stream, MKL_INT const n,
                             double* const r, double const alpha,
                             double const a, double const beta) -> int
    {
        return vdRngGamma(method, stream, n, r, alpha, a, beta);
    }

    static auto fill(gsl::span<RealT> out, VSLStreamStatePtr const stream,
                     param_type const& params) -> void
    {
        // NOTE: Intel MKL calls the scale parameter `beta`. Do not confuse
        // `beta` with the _inverse_ scale parameter.
        auto const status =
            vsl_generate(stream, static_cast<MKL_INT>(out.size()), out.data(),
                         /*alpha = */ params.shape(),
                         /*displacement = */ RealT{0},
                         /*beta = */ params.scale());
        check_vsl_status_after_generation(status);
    }

  public:
    auto operator()(gsl::span<RealT> out, param_type const& params) const
        -> void
    {
        fill(out, _stream, params);
    }

    auto operator()(gsl::span<RealT> out) const -> void
    {
        (*this)(out, _params);
    }

    auto operator()() -> RealT
    {
        TCM_ASSERT(_i <= _buffer.size(), "Pre-condition violated");
        if (_i == _buffer.size()) {
            fill(_buffer, _stream, _params);
            _i = 0;
        }
        return _buffer[_i++];
    }
};
// }}}

/// Uses Intel MKL to generate numbers distributed according to N(mu, sigma^2).
///
/// See https://en.wikipedia.org/wiki/Normal_distribution.
template <class RealT> class buffered_gauss_distribution_t { // {{{
    static_assert(std::is_same_v<RealT, float> || std::is_same_v<RealT, double>,
                  "tcm::detail::buffered_gauss_distribution_t currently only "
                  "supports `float`s and `double`s.");

  public:
    /// Generation method.
    ///
    /// Possible values are:
    ///
    ///   * `VSL_RNG_METHOD_GAUSSIAN_BOXMULLER`
    ///   * `VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2`
    ///   * `VSL_RNG_METHOD_GAUSSIAN_ICDF`
    ///
    /// See https://software.intel.com/en-us/mkl-developer-reference-c-vrnggaussian
    /// for more info.
    static constexpr MKL_INT method = VSL_RNG_METHOD_GAUSSIAN_BOXMULLER;

    static constexpr size_t default_buffer_size = 1024;

    /// Buffer type.
    ///
    /// Performance of Intel MKL's random number generation increases
    /// considerably when multiple variates a generated at once (see
    /// https://software.intel.com/en-us/mkl-vsperfdata-uniform).
    using buffer_type =
        std::vector<RealT, boost::alignment::aligned_allocator<RealT, 64u>>;

    using stream_type = VSLStreamStatePtr;

    class param_type {
        RealT _mu;    ///< Mean
        RealT _sigma; ///< Standard deviation

      public:
        constexpr explicit param_type(RealT const mu,
                                      RealT const sigma) TCM_NOEXCEPT
            : _mu{mu}
            , _sigma{sigma}
        {
            TCM_ASSERT(sigma > 0, "`sigma` must be positive");
        }

        constexpr param_type(param_type const&) noexcept = default;
        constexpr param_type(param_type&&) noexcept      = default;
        constexpr param_type& operator=(param_type const&) noexcept = default;
        constexpr param_type& operator=(param_type&&) noexcept = default;

        constexpr auto mu() const noexcept { return _mu; }
        constexpr auto sigma() const noexcept { return _sigma; }
    };

  private:
    stream_type _stream; ///< Intel MKL random number generator stream.
                         ///<
    buffer_type _buffer; ///< Output buffer where the generated numbers
                         ///< are written.
    size_t _i; ///< Current position in `_buffer`. `_i == _buffer.size()`
               ///< means that the buffer needs to be refilled.
    param_type _params; ///< Distribution parameters.
                        ///<

  public:
    buffered_gauss_distribution_t(
        stream_type const stream, RealT const mu, RealT const sigma,
        size_t const buffer_size = default_buffer_size) TCM_NOEXCEPT
        : _stream{stream}
        , _buffer(buffer_size)
        , _i{buffer_size}
        , _params{mu, sigma}
    {
        TCM_ASSERT(stream != nullptr, "`stream` should not be null");
        TCM_ASSERT(buffer_size > 0, "buffer should not be empty");
    }

    buffered_gauss_distribution_t(buffered_gauss_distribution_t const&) =
        delete;

    buffered_gauss_distribution_t(buffered_gauss_distribution_t&&) noexcept =
        default;

    buffered_gauss_distribution_t&
    operator=(buffered_gauss_distribution_t const&) = delete;

    buffered_gauss_distribution_t&
    operator=(buffered_gauss_distribution_t&&) noexcept = default;

    constexpr auto param() const noexcept -> param_type { return _params; }
    constexpr auto param(param_type const params) noexcept -> void
    {
        _params = params;
        _i      = _buffer.size();
    }

#if 0
    constexpr auto buffer() noexcept -> gsl::span<RealT> { return {_buffer}; }
    constexpr auto buffer() const noexcept -> gsl::span<RealT const>
    {
        return {_buffer};
    }
#endif

    auto resize(size_t const buffer_size) -> void
    {
        TCM_ASSERT(buffer_size > 0, "buffer should not be empty");
        // TODO(twesterhout): We could implement an optimisation here which
        // would copy the unused part of the buffer. This could potentially
        // avoid generating some random numbers.
        _buffer.resize(buffer_size);
        _i = buffer_size;
    }

    constexpr auto stream() const noexcept -> VSLStreamStatePtr
    {
        return _stream;
    }

  private:
    static auto vsl_generate(VSLStreamStatePtr const stream, MKL_INT const n,
                             float* const r, float const a, float const sigma)
        -> int
    {
        return vsRngGaussian(method, stream, n, r, a, sigma);
    }

    static auto vsl_generate(VSLStreamStatePtr const stream, MKL_INT const n,
                             double* const r, double const a,
                             double const sigma) -> int
    {
        return vdRngGaussian(method, stream, n, r, a, sigma);
    }

    static auto fill(gsl::span<RealT> out, VSLStreamStatePtr stream,
                     param_type const& params) -> void
    {
        auto const status =
            vsl_generate(stream, static_cast<MKL_INT>(out.size()), out.data(),
                         /*a = */ params.mu(),
                         /*sigma = */ params.sigma());
        check_vsl_status_after_generation(status);
    }

  public:
    auto operator()(gsl::span<RealT> out, param_type const& params) const
        -> void
    {
        fill(out, _stream, params);
    }

    auto operator()(gsl::span<RealT> out) const -> void
    {
        (*this)(out, _params);
    }

    auto operator()() -> RealT
    {
        TCM_ASSERT(_i <= _buffer.size(), "Pre-condition violated");
        if (_i == _buffer.size()) {
            fill(_buffer, _stream, _params);
            _i = 0;
        }
        return _buffer[_i++];
    }
};
// }}}

template <class RealT> class buffered_tsallis_distribution_t { // {{{
    static_assert(std::is_same_v<RealT, float> || std::is_same_v<RealT, double>,
                  "Currently, only `float`s and `double`s are supported.");

  private:
    /// Calculation of \f$p\f$ given \f$q_V\f$ (see `Tsallis_RNG` function in
    /// [@Schanze2006] for an explanation).
    ///
    /// [@Schanze2006]: Thomas Schanze, "An exact D-dimensional Tsallis random
    ///                 number generator for generalized simulated annealing", 2006.
    static constexpr auto get_p(RealT const q_V) noexcept -> RealT
    {
        using R = RealT;
        return (R{3} - q_V) / (R{2} * (q_V - R{1}));
    }

    /// Calculation of \f$s\f$ given \f$q_V\f$ and \f$t_V\f$ (see `Tsallis_RNG`
    /// function in [@Schanze2006] for an explanation).
    ///
    /// [@Schanze2006]: Thomas Schanze, "An exact D-dimensional Tsallis random
    ///                 number generator for generalized simulated annealing", 2006.
    static /*constexpr*/ auto get_s(RealT const q_V, RealT const t_V) noexcept
        -> RealT
    {
        using R = RealT;
        return std::sqrt(R{2} * (q_V - R{1}))
               / std::pow(t_V, R{1} / (R{3} - q_V));
    }

  public:
    static constexpr size_t default_buffer_size = 1024;

    /// Buffer type.
    ///
    /// Performance of Intel MKL's random number generation increases
    /// considerably when multiple variates a generated at once (see
    /// https://software.intel.com/en-us/mkl-vsperfdata-uniform).
    using buffer_type =
        std::vector<RealT, boost::alignment::aligned_allocator<RealT, 64u>>;

    class param_type {
        RealT _q_V; ///< Visiting distribution shape parameter
        RealT _t_V; ///< Visiting temperature
        RealT _s;   ///< Variable `s` from `Tsallis_RNG` in [@Schanze2006].

      public:
        explicit param_type(RealT const q_V, RealT const t_V) TCM_NOEXCEPT
            : _q_V{q_V}
            , _t_V{t_V}
        {
            TCM_ASSERT(1 < q_V && q_V < 3, "`q_V` must be in (1, 3)");
            TCM_ASSERT(t_V > 0, "`t_V` must be positive");
            _s = get_s(q_V, t_V);
        }

        constexpr param_type(param_type const&) noexcept = default;
        constexpr param_type(param_type&&) noexcept      = default;
        constexpr param_type& operator=(param_type const&) noexcept = default;
        constexpr param_type& operator=(param_type&&) noexcept = default;

        constexpr auto q_V() const noexcept { return _q_V; }
        constexpr auto t_V() const noexcept { return _t_V; }
        constexpr auto s() const noexcept { return _s; }
    };

  private:
    buffered_gamma_distribution_t<RealT> _gamma_dist;
    buffered_gauss_distribution_t<RealT> _gauss_dist;
    param_type                           _params;

  public:
    buffered_tsallis_distribution_t(
        VSLStreamStatePtr const stream, RealT const q_V, RealT const t_V,
        size_t const buffer_size = default_buffer_size)
        : _gamma_dist{stream, get_p(q_V), RealT{1}, buffer_size}
        , _gauss_dist{stream, RealT{0}, RealT{1}, buffer_size}
        , _params{q_V, t_V}
    {}

    constexpr auto param() const noexcept -> param_type { return _params; }
    constexpr auto param(param_type const params) noexcept -> void
    {
        _gamma_dist.param(
            typename buffered_gamma_distribution_t<RealT>::param_type{
                get_p(params.q_V()), RealT{1}});
        // _gauss_dist doesn't depend on params
        _params = params;
    }

    constexpr auto stream() const noexcept -> VSLStreamStatePtr
    {
        return _gamma_dist.stream();
    }

  private:
    auto fill(gsl::span<RealT> out) -> void
    {
        using R      = RealT;
        using Params = typename buffered_gauss_distribution_t<R>::param_type;

        // Updates internal buffer and is the reason `fill` is not const.
        auto const u = _gamma_dist();
        auto const y = _params.s() * std::sqrt(u);

        // TODO(twesterhout): Profile what is more efficient: using different
        // sigma or calling cblas_?scal.
        _gauss_dist(out, Params{R{0}, R{1} / y});

        // NOLINTNEXTLINE(readability-braces-around-statements)
        // if constexpr (std::is_same_v<R, float>) {
        //     cblas_sscal(static_cast<MKL_INT>(out.size()), R{1} / y, out.data(),
        //                 1);
        // }
        // else {
        //     cblas_dscal(static_cast<MKL_INT>(out.size()), R{1} / y, out.data(),
        //                 1);
        // }
    }

  public:
    // Generates an N-dimensional sample
    auto operator()(gsl::span<RealT> out) { fill(out); }

    // Generates a 1-dimensional sample
    auto operator()() -> RealT
    {
        auto const u = _gamma_dist();
        auto const y = _params.s() * std::sqrt(u);
        auto const x = _gauss_dist();
        return x / y;
    }
};
// }}}

template class buffered_gamma_distribution_t<float>;
template class buffered_gamma_distribution_t<double>;

template class buffered_gauss_distribution_t<float>;
template class buffered_gauss_distribution_t<double>;

template class buffered_tsallis_distribution_t<float>;
template class buffered_tsallis_distribution_t<double>;

} // namespace detail

template <class EnergyFn, class WrapFn> class sa_chain_t;

struct sa_pars_t {
    float    q_V;
    float    q_A;
    float    t_0;
    unsigned dimension;
    unsigned num_iterations;
};

class sa_buffers_t { // {{{
  public:
    template <class EnergyFn, class WrapFn> friend class sa_chain_t;

    using buffer_type =
        std::vector<float, boost::alignment::aligned_allocator<float, 64u>>;

    struct state_type {
        float       value;
        buffer_type buffer;

      private:
        /// Rounds x up to the closest multiple of 64.
        static constexpr auto round_up_to_64(size_t const x) noexcept -> size_t
        {
            constexpr size_t N = 64;
            return (x + N - 1) & N;
        }

      public:
        /// Allocates a buffer of the specified size.
        explicit state_type(size_t const size)
            : value{std::numeric_limits<float>::infinity()}, buffer{}
        {
            resize(size);
        }

        state_type(state_type const&)     = default;
        state_type(state_type&&) noexcept = default;
        state_type& operator=(state_type const&) = default;
        state_type& operator=(state_type&&) noexcept = default;

        /// Ensures buffer is big enough to hold `size` elements.
        ///
        /// We use a small trick here. We round `size` up to a multiple of 64 so
        /// that we can safely use SIMD instructions without worrying about
        /// writing past the end of the buffer.
        auto resize(size_t const size) -> void
        {
            buffer.reserve(round_up_to_64(size));
            buffer.resize(size);
        }
    };

  private:
    using distribution_type = detail::buffered_tsallis_distribution_t<float>;

    state_type        _current;
    state_type        _proposed;
    state_type        _best;
    distribution_type _tsallis;

  public:
    sa_buffers_t(sa_pars_t const&  params,
                 VSLStreamStatePtr stream = random_stream());

    sa_buffers_t(sa_buffers_t const&)     = delete;
    sa_buffers_t(sa_buffers_t&&) noexcept = default;
    sa_buffers_t& operator=(sa_buffers_t const&) = delete;
    sa_buffers_t& operator=(sa_buffers_t&&) = default;

    auto reset(sa_pars_t const& params) -> void;
    auto reset(distribution_type::param_type const& params) -> void;

    template <class Initialise> auto guess(Initialise&& init) -> void;
};

inline sa_buffers_t::sa_buffers_t(sa_pars_t const&  params,
                                  VSLStreamStatePtr stream)
    : _current{params.dimension}
    , _proposed{params.dimension}
    , _best{params.dimension}
    , _tsallis{stream, params.q_V, params.t_0}
{}

inline auto sa_buffers_t::reset(sa_pars_t const& params) -> void
{
    _current.resize(params.dimension);
    _proposed.resize(params.dimension);
    _best.resize(params.dimension);
    _tsallis.param(distribution_type::param_type{params.q_V, params.t_0});
}

inline auto sa_buffers_t::reset(distribution_type::param_type const& params)
    -> void
{
    _tsallis.param(params);
}

template <class Initialise> auto sa_buffers_t::guess(Initialise&& init) -> void
{
    std::forward<Initialise>(init)(gsl::span<float>{_current.buffer});
}
// }}}

template <class EnergyFn, class WrapFn> class sa_chain_t { // {{{
  public:
    using energy_fn_type    = EnergyFn;
    using wrap_fn_type      = WrapFn;
    using tsallis_dist_type = detail::buffered_tsallis_distribution_t<float>;

  private:
    energy_fn_type _energy_fn; ///< Energy function `f`.
                               ///<
    wrap_fn_type _wrap_fn;     ///< Wrapping function `wrap` which ensures that
                               ///< `wrap(x + dx)` is in the domain of `f`.
    sa_buffers_t&    _buffers;
    sa_pars_t const& _params;
    size_t           _i; ///< Current iteration.
                         ///<

  public:
    sa_chain_t(sa_buffers_t& buffers, sa_pars_t const& params,
               energy_fn_type energy_fn, wrap_fn_type wrap_fn);

    sa_chain_t(sa_chain_t const&) = delete;
    sa_chain_t(sa_chain_t&&)      = delete;
    sa_chain_t& operator=(sa_chain_t const&) = delete;
    sa_chain_t& operator=(sa_chain_t&&) = delete;

    inline auto    operator()() -> void;
    constexpr auto restart() noexcept -> void;
    constexpr auto current() const noexcept -> sa_buffers_t::state_type const&;
    constexpr auto best() const noexcept -> sa_buffers_t::state_type const&;

  private:
    constexpr auto t_0() const noexcept { return _params.t_0; }
    constexpr auto q_V() const noexcept { return _params.q_V; }
    constexpr auto q_A() const noexcept { return _params.q_A; }

    /// Returns the dimension of the parameter space.
    auto dim() const noexcept { return _buffers._current.buffer.size(); }

    /// Calculates the visiting temperature `t_V` for the given iteration.
    inline auto temperature(size_t i) const noexcept -> float;

    /// Sets the best state equal to current.
    inline auto update_best() noexcept -> void;

    /// Accepts the move, i.e. `proposed` becomes `current`
    inline auto accept_full() noexcept -> void;
    inline auto accept_one(size_t i, float const x) noexcept -> void;

    /// Rejects the move.
    inline auto reject_full() noexcept -> void;
    inline auto reject_one(size_t i, float const x) noexcept -> void;

    /// Returns a uniform random number
    ///
    /// TODO(twesterhout): This function is terribly inefficient.
    inline auto uniform() -> float;

    template <class Accept, class Reject>
    inline auto accept_or_reject(float const dE, float const t_A,
                                 Accept&& accept, Reject&& reject) -> void;

    inline auto generate_full() -> void;
    inline auto generate_one(size_t const i) -> std::tuple<float, float>;
};
// }}}

// {{{ sa_chain_t IMPLEMENTATION
template <class E, class W>
inline sa_chain_t<E, W>::sa_chain_t(sa_buffers_t&    buffers,
                                    sa_pars_t const& params,
                                    energy_fn_type   energy_fn,
                                    wrap_fn_type     wrap_fn)
    : _energy_fn{std::move(energy_fn)}
    , _wrap_fn{std::move(wrap_fn)}
    , _buffers{buffers}
    , _params{params}
    , _i{0}
{
    _buffers._current.value = _energy_fn(_buffers._current.buffer);
    update_best();
}

template <class E, class W>
inline auto sa_chain_t<E, W>::temperature(std::size_t const i) const noexcept
    -> float
{
    auto const num = t_0() * (std::pow(2.0f, q_V() - 1.0f) - 1.0f);
    auto const den = std::pow(static_cast<float>(2 + i), q_V() - 1.0f) - 1.0f;
    return num / den;
}

template <class E, class W>
inline auto sa_chain_t<E, W>::update_best() noexcept -> void
{
    using std::begin, std::end;
    std::copy(begin(_buffers._current.buffer), end(_buffers._current.buffer),
              begin(_buffers._best.buffer));
    _buffers._best.value = _buffers._current.value;
}

template <class E, class W>
TCM_NOINLINE auto sa_chain_t<E, W>::uniform() -> float
{
    float      r;
    auto const status =
        vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, _buffers._tsallis.stream(), 1,
                     &r, 0.0f, 1.0f);
    detail::check_vsl_status_after_generation(status);
    return r;
}

template <class E, class W>
template <class Accept, class Reject>
inline auto sa_chain_t<E, W>::accept_or_reject(float const dE, float const t_A,
                                               Accept&& accept, Reject&& reject)
    -> void
{
#if 0
    auto const accept = [this]() {
        using std::swap;
        swap(_buffers._current, _buffers._proposed);
        if (_buffers._current.value < _buffers._best.value) { update_best(); }
    };
    auto const reject = []() {};
#endif

    // Always accept moves that reduce the energy
    if (dE < 0.0f) {
        std::forward<Accept>(accept)();
        return;
    }

    // Eq. (5)
    //
    // pqv_temp = (q_A - 1.0) * (e - self.energy_state.current_energy) / ( self.temperature_step + 1.)
    //
    // auto const factor = 1.0f + (q_A() - 1.0f) * dE / t_A;
    auto const factor = 1.0f + (q_A() - 1.0f) * dE / t_A;
    auto const P_qA =
        factor <= 0.0f ? 0.0f : std::pow(factor, 1.0f / (1.0f - q_A()));
    if (uniform() <= P_qA) {
        // std::fprintf(stderr, "Ping!\n");
        std::forward<Accept>(accept)();
    }
    else {
        // std::fprintf(stderr, "T_A = %f, factor = %f, r = %f, P_qA = %f\n",
        //              static_cast<double>(t_A), static_cast<double>(factor),
        //              static_cast<double>(r), static_cast<double>(P_qA));
        std::forward<Reject>(reject)();
    }
}

template <class E, class W>
inline auto sa_chain_t<E, W>::generate_full() -> void
{
    using std::begin, std::end;
    _buffers._tsallis(_buffers._proposed.buffer);
    std::transform(
        begin(_buffers._current.buffer), end(_buffers._current.buffer),
        begin(_buffers._proposed.buffer), begin(_buffers._proposed.buffer),
        [this](auto const x, auto const dx) -> float {
            return _wrap_fn(x + dx);
        });
    _buffers._proposed.value = _energy_fn(_buffers._proposed.buffer);
}

template <class E, class W>
inline auto sa_chain_t<E, W>::generate_one(std::size_t const i)
    -> std::tuple<float, float>
{
    auto const x = _wrap_fn(_buffers._current.buffer[i] + _buffers._tsallis());
    auto       temp_buffer = _buffers._current.buffer;
    temp_buffer[i]         = x;
    auto const value       = _energy_fn(temp_buffer);
    // auto const value =
    //     _energy_fn(i, x, _buffers._current.value, _buffers._current.buffer);
    return std::make_tuple(x, value);
}

template <class E, class W>
constexpr auto sa_chain_t<E, W>::current() const noexcept
    -> sa_buffers_t::state_type const&
{
    return _buffers._current;
}

template <class E, class W>
constexpr auto sa_chain_t<E, W>::best() const noexcept
    -> sa_buffers_t::state_type const&
{
    return _buffers._best;
}

template <class E, class W>
constexpr auto sa_chain_t<E, W>::restart() noexcept -> void
{
    _i = 0;
}

template <class E, class W>
TCM_NOINLINE auto sa_chain_t<E, W>::operator()() -> void
{
    // (iv) Calculate new temperature...
    auto const t_V = temperature(_i);
    auto const t_A = t_V / static_cast<float>(_i + 1);
    _buffers._tsallis.param(typename tsallis_dist_type::param_type{q_V(), t_V});

    // Markov chain at constant temperature
    for (auto j = size_t{0}; j < dim(); ++j) {
        auto const accept = [this]() {
            using std::swap;
            swap(_buffers._current, _buffers._proposed);
            if (_buffers._current.value < _buffers._best.value) {
                update_best();
            }
            std::cerr << _buffers._current.value << '\n';
        };
        auto const reject = []() {};
        generate_full();
        accept_or_reject(_buffers._proposed.value - _buffers._current.value,
                         t_A, accept, reject);
    }
    for (auto j = size_t{0}; j < dim(); ++j) {
        auto const [x, value] = generate_one(j);
        auto const accept     = [this, j, x = x, value = value]() {
            _buffers._current.buffer[j] = x;
            _buffers._current.value     = value;
            if (_buffers._current.value < _buffers._best.value) {
                update_best();
            }
            std::cerr << _buffers._current.value << '\n';
        };
        auto const reject = []() {};
        accept_or_reject(value - _buffers._current.value, t_A, accept, reject);
    }

    // NOTE: Don't forget this!
    ++_i;

#if 0
        for (auto j = std::size_t{0}; j < 2 * dim(); ++j) {
            // (ii) Then randomly generate x...
            if (j < dim()) { generate_full(); }
            else {
                generate_one(j - dim());
            }
            // (iii)
            auto const proposed_energy = std::get<0>(_proposed);
            auto const current_energy  = std::get<0>(_current);
            accept_or_reject(proposed_energy - current_energy, t_A);
            auto const proposed_energy = std::get<0>(_proposed);
            auto const current_energy  = std::get<0>(_current);
            if (proposed_energy < current_energy) { accept(); }
            else {
                real_type r;
                int       status = vsRngUniform(
                    VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, _stream, 1, &r,
                    /*a = */ real_type{0}, /*b = */ real_type{1});
                if (status != VSL_STATUS_OK)
                    throw std::runtime_error{"vsRngUniform failed."};

                // Eq. (5)
                auto const factor = real_type{1}
                                    + (_params.q_A - 1)
                                          * (proposed_energy - current_energy)
                                          / t_A;
                auto const P_qA =
                    factor <= 0
                        ? real_type{0}
                        : std::pow(factor,
                                   real_type{1} / (real_type{1} - _params.q_A));
                if (r <= P_qA) {
                    std::fprintf(stderr, "Ping!\n");
                    accept();
                }
                else {
                    std::fprintf(
                        stderr, "T_V = %f, factor = %f, r = %f, P_qA = %f\n",
                        static_cast<double>(t_V),
                        static_cast<double>(factor), static_cast<double>(r),
                        static_cast<double>(P_qA));
                    reject();
                }
            }
        }
#endif
}
// }}}

inline auto simulated_annealing_buffers(sa_pars_t const& params)
    -> sa_buffers_t&
{
    constexpr auto default_params = sa_pars_t{/*q_V=*/2.62f,
                                              /*q_A=*/-5.0f,
                                              /*t_0=*/5230.0f,
                                              /*dimension=*/1000000,
                                              /*num_iterations=*/1000};

    thread_local sa_buffers_t buffers{default_params, random_stream()};
    buffers.reset(params);
    return buffers;
}

template <class EnergyFn, class WrapFn, class Initialise, class Finalise>
inline auto minimise_using_simulated_annealing(EnergyFn energy, WrapFn wrap,
                                               sa_pars_t const& params,
                                               Initialise&&     input,
                                               Finalise&&       output) -> void
{
    auto& buffers = simulated_annealing_buffers(params);
    buffers.guess(std::forward<Initialise>(input));

    sa_chain_t chain{buffers, params, std::move(energy), std::move(wrap)};
    for (auto i = 0u; i < params.num_iterations; ++i) {
        chain();
    }
    auto const& best = chain.best();
    std::forward<Finalise>(output)(gsl::span<float const>{best.buffer});
}

TCM_NAMESPACE_END
