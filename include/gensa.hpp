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

#include "config.h"
#include <gsl/gsl-lite.hpp>
#include <mkl_cblas.h>
#include <mkl_vsl.h>
#include <cmath>
#include <functional>
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
                   "exceeds "
                   "threshold";
        case VSL_RNG_ERROR_ARS5_NOT_SUPPORTED:
            return "ARS-5 random number generator is not supported on "
                   "the CPU running the application";
        default: return "unknown error code: a bug in Intel MKL?";
        } // end switch
    }(status);
    throw std::runtime_error{msg};
}

/// Uses Intel MKL to generate numbers distributed according to
/// Gamma(alpha, beta).
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
    static constexpr MKL_INT method = VSL_RNG_METHOD_GAMMA_GNORM;

    class param_type {
        RealT _alpha; ///< Shape parameter
        RealT _beta;  ///< Rate parameter (inverse scale parameter)

      public:
        constexpr explicit param_type(RealT const alpha,
                                      RealT const beta) TCM_NOEXCEPT
            : _alpha{alpha}
            , _beta{beta}
        {
            TCM_ASSERT(alpha > 0, "alpha must be positive");
            TCM_ASSERT(beta > 0, "beta must be positive");
        }

        constexpr param_type(param_type const&) noexcept = default;
        constexpr param_type(param_type&&) noexcept      = default;
        constexpr param_type& operator=(param_type const&) noexcept = default;
        constexpr param_type& operator=(param_type&&) noexcept = default;

        constexpr auto alpha() const noexcept { return _alpha; }
        constexpr auto beta() const noexcept { return _beta; }
    };

  private:
    gsl::span<RealT> _buffer;  ///< Output buffer where the generated
                               ///< numbers are written to
    VSLStreamStatePtr _stream; ///< Intel MKL random number generator stream
                               ///<
    std::size_t _i; ///< Current position in `_buffer`. `_i == _buffer.size()`
                    ///< indicates that the buffer needs to be refilled.
    param_type _params; ///< alpha and beta

  public:
    constexpr buffered_gamma_distribution_t(VSLStreamStatePtr const stream,
                                            gsl::span<RealT> const  workspace,
                                            RealT const             alpha,
                                            RealT const beta) TCM_NOEXCEPT
        : _buffer{workspace}
        , _stream{stream}
        , _i{workspace.size()}
        , _params{alpha, beta}
    {
        TCM_ASSERT(!workspace.empty(), "Workspace must not be empty");
        TCM_ASSERT(workspace.size() <= static_cast<std::size_t>(
                       std::numeric_limits<MKL_INT>::max()),
                   "Integer overflow");
    }

    buffered_gamma_distribution_t(buffered_gamma_distribution_t const&) =
        delete;

    constexpr buffered_gamma_distribution_t(
        buffered_gamma_distribution_t&&) noexcept = default;

    buffered_gamma_distribution_t&
    operator=(buffered_gamma_distribution_t const&) = delete;

    constexpr buffered_gamma_distribution_t&
    operator=(buffered_gamma_distribution_t&&) noexcept = default;

    constexpr auto param() const noexcept -> param_type { return _params; }
    constexpr auto param(param_type const params) noexcept -> void
    {
        _params = params;
        _i      = _buffer.size();
    }

    constexpr auto buffer() const noexcept -> gsl::span<RealT>
    {
        return _buffer;
    }

    constexpr auto stream() const noexcept -> VSLStreamStatePtr
    {
        return _stream;
    }

    /// Replaces the buffer with a new one.
    ///
    /// TODO(twesterhout): Currently, this clears the cache, but it may be
    /// suboptimal.
    constexpr auto buffer(gsl::span<RealT> workspace) TCM_NOEXCEPT -> void
    {
        TCM_ASSERT(!workspace.empty(), "Workspace must not be empty");
        TCM_ASSERT(workspace.size() <= static_cast<std::size_t>(
                       std::numeric_limits<MKL_INT>::max()),
                   "Integer overflow");
        _buffer = workspace;
        _i      = workspace.size();
    }

  private:
    auto fill(gsl::span<RealT> out) const -> void
    {
        int status;
        // NOLINTNEXTLINE(readability-braces-around-statements)
        if constexpr (std::is_same_v<RealT, float>) {
            status = vsRngGamma(method, _stream,
                                static_cast<MKL_INT>(out.size()), out.data(),
                                /*alpha = */ _params.alpha(),
                                /*displacement = */ RealT{0},
                                /*beta = */ _params.beta());
        }
        else {
            status = vdRngGamma(method, _stream,
                                static_cast<MKL_INT>(out.size()), out.data(),
                                /*alpha = */ _params.alpha(),
                                /*displacement = */ RealT{0},
                                /*beta = */ _params.beta());
        }
        check_vsl_status_after_generation(status);
    }

  public:
    auto operator()(gsl::span<RealT> out) const -> void { fill(out); }

    auto operator()() -> RealT
    {
        TCM_ASSERT(_i <= _buffer.size(), "Pre-condition violated");
        if (_i == _buffer.size()) {
            fill(_buffer);
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
    static constexpr MKL_INT method = VSL_RNG_METHOD_GAUSSIAN_ICDF;

  public:
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
    gsl::span<RealT> _buffer;  ///< Output buffer where the generated numbers
                               ///< are written.
    VSLStreamStatePtr _stream; ///< Intel MKL random number generator stream.
                               ///<
    std::size_t _i; ///< Current position in `_buffer`. `_i == _buffer.size()`
                    ///< means that the buffer needs to be refilled.
    param_type _params; ///< Distribution parameters.

  public:
    constexpr buffered_gauss_distribution_t(VSLStreamStatePtr const stream,
                                            gsl::span<RealT> const  workspace,
                                            RealT const             mu,
                                            RealT const sigma) TCM_NOEXCEPT
        : _buffer{workspace}
        , _stream{stream}
        , _i{workspace.size()}
        , _params{mu, sigma}
    {
        TCM_ASSERT(!workspace.empty(), "Workspace must not be empty");
        TCM_ASSERT(workspace.size() <= static_cast<std::size_t>(
                       std::numeric_limits<MKL_INT>::max()),
                   "Integer overflow");
    }

    buffered_gauss_distribution_t(buffered_gauss_distribution_t const&) =
        delete;

    constexpr buffered_gauss_distribution_t(
        buffered_gauss_distribution_t&&) noexcept = default;

    buffered_gauss_distribution_t&
    operator=(buffered_gauss_distribution_t const&) = delete;

    constexpr buffered_gauss_distribution_t&
    operator=(buffered_gauss_distribution_t&&) noexcept = default;

    constexpr auto param() const noexcept -> param_type { return _params; }
    constexpr auto param(param_type const params) noexcept -> void
    {
        _params = params;
        _i      = _buffer.size();
    }

    constexpr auto buffer() const noexcept -> gsl::span<RealT>
    {
        return _buffer;
    }

    constexpr auto buffer(gsl::span<RealT> workspace) noexcept
    {
        TCM_ASSERT(!workspace.empty(), "Workspace must not be empty");
        TCM_ASSERT(workspace.size() <= static_cast<std::size_t>(
                       std::numeric_limits<MKL_INT>::max()),
                   "Integer overflow");
        _buffer = workspace;
        _i      = workspace.size();
    }

    constexpr auto stream() const noexcept -> VSLStreamStatePtr
    {
        return _stream;
    }

  private:
    auto fill(gsl::span<RealT> out) const -> void
    {
        int status;
        // NOLINTNEXTLINE(readability-braces-around-statements)
        if constexpr (std::is_same_v<RealT, float>) {
            status = vsRngGaussian(method, _stream,
                                   static_cast<MKL_INT>(out.size()), out.data(),
                                   /*a = */ _params.mu(),
                                   /*sigma = */ _params.sigma());
        }
        else {
            status = vdRngGaussian(method, _stream,
                                   static_cast<MKL_INT>(out.size()), out.data(),
                                   /*a = */ _params.mu(),
                                   /*sigma = */ _params.sigma());
        }
        check_vsl_status_after_generation(status);
    }

  public:
    auto operator()(gsl::span<RealT> out) const -> void { fill(out); }

    auto operator()() -> RealT
    {
        TCM_ASSERT(_i <= _buffer.size(), "Pre-condition violated");
        if (_i == _buffer.size()) {
            fill(_buffer);
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
    static constexpr auto get_p(RealT const q_V) noexcept -> RealT
    {
        using R = RealT;
        return (R{3} - q_V) / (R{2} * (q_V - R{1}));
    }

    static /*constexpr*/ auto get_s(RealT const q_V, RealT const t_V) noexcept
        -> RealT
    {
        using R = RealT;
        return std::sqrt(R{2} * (q_V - R{1}))
               / std::pow(t_V, R{1} / (R{3} - q_V));
    }

  public:
    class param_type {
        RealT _q_V; ///< Visiting distribution shape parameter
        RealT _t_V; ///< Visiting temperature
        RealT _s;   ///< Variable s from `Tsallis_RNG` function

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
    constexpr buffered_tsallis_distribution_t(VSLStreamStatePtr      stream,
                                              gsl::span<RealT> const out,
                                              gsl::span<RealT> const workspace,
                                              RealT const            q_V,
                                              RealT const t_V) TCM_NOEXCEPT
        : _gamma_dist{stream, workspace, get_p(q_V), RealT{1}}
        , _gauss_dist{stream, out, RealT{0}, RealT{1}}
        , _params{q_V, t_V}
    {
        TCM_ASSERT(!out.empty(), "Output buffer must not be empty");
        TCM_ASSERT(!workspace.empty(), "Workspace must not be empty");
        TCM_ASSERT(workspace.size() <= static_cast<std::size_t>(
                       std::numeric_limits<MKL_INT>::max())
                       && out.size() <= static_cast<std::size_t>(
                              std::numeric_limits<MKL_INT>::max()),
                   "Integer overflow");
        TCM_ASSERT(out.size() == workspace.size(), "???");
    }

    constexpr auto param() const noexcept -> param_type { return _params; }
    constexpr auto param(param_type const params) noexcept -> void
    {
        _gamma_dist.param(
            typename buffered_gamma_distribution_t<RealT>::param_type{
                get_p(params.q_V()), RealT{1}});
        _params = params;
    }

    constexpr auto stream() const noexcept -> VSLStreamStatePtr
    {
        return _gamma_dist.stream();
    }

  private:
    auto fill(gsl::span<RealT> out) -> void
    {
        using R = RealT;

        auto const u =
            _gamma_dist(); // Updates internal buffer and is the reason `fill` is not const.
        auto const y = _params.s() * std::sqrt(u);

        _gauss_dist(out);

        // NOLINTNEXTLINE(readability-braces-around-statements)
        if constexpr (std::is_same_v<R, float>) {
            cblas_sscal(static_cast<MKL_INT>(out.size()), R{1} / y, out.data(),
                        1);
        }
        else {
            cblas_dscal(static_cast<MKL_INT>(out.size()), R{1} / y, out.data(),
                        1);
        }
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

template class detail::buffered_gamma_distribution_t<float>;
template class detail::buffered_gamma_distribution_t<double>;

template class detail::buffered_gauss_distribution_t<float>;
template class detail::buffered_gauss_distribution_t<double>;

template class detail::buffered_tsallis_distribution_t<float>;
template class detail::buffered_tsallis_distribution_t<double>;

} // namespace detail

struct sa_pars_t {
    float    q_V;
    float    q_A;
    float    t_0;
    unsigned n;
};

class sa_buffers_t { // {{{
  public:
    template <class T> using buffer_type = std::vector<T>;

  private:
    buffer_type<float> _current;
    buffer_type<float> _proposed;
    buffer_type<float> _best;
    buffer_type<float> _gaussian;
    buffer_type<float> _gamma;
    std::size_t        _size;

  public:
    auto resize(std::size_t const size) -> void;

    explicit sa_buffers_t(std::size_t const size)
        : _current{}, _proposed{}, _best{}, _gaussian{}, _gamma{}, _size{0}
    {
        resize(size);
    }

    sa_buffers_t(sa_buffers_t const&)     = delete;
    sa_buffers_t(sa_buffers_t&&) noexcept = default;
    sa_buffers_t& operator=(sa_buffers_t const&) = delete;
    sa_buffers_t& operator=(sa_buffers_t&&) = default;

    auto current() noexcept -> gsl::span<float>
    {
        return {_current.data(), _size};
    }

    auto proposed() noexcept -> gsl::span<float>
    {
        return {_proposed.data(), _size};
    }

    auto best() noexcept -> gsl::span<float>
    {
        return {_proposed.data(), _size};
    }

    auto gaussian() noexcept -> gsl::span<float>
    {
        return {_gaussian.data(), _size};
    }

    auto gamma() noexcept -> gsl::span<float> { return {_gamma.data(), _size}; }
};
// }}}

// {{{ sa_buffers_t IMPLEMENTATION
inline auto sa_buffers_t::resize(std::size_t const size) -> void
{
    TCM_ASSERT(size > 0, "Empty buffers are not supported");
    if (size > _current.size()) {
        _current.resize(size);
        _proposed.resize(size);
        _best.resize(size);
        _gaussian.resize(size);
        _gamma.resize(size);
    }
    _size = size;
}
// }}}

template <class EnergyFn, class WrapFn> class sa_chain_t { // {{{
  public:
    using energy_fn_type    = EnergyFn;
    using wrap_fn_type      = WrapFn;
    using tsallis_dist_type = detail::buffered_tsallis_distribution_t<float>;

    struct state_type {
        float            value;
        gsl::span<float> buffer;
    };

  private:
    state_type _current;  ///< Current position `x` and energy `E = f(x)`
                          ///<
    state_type _proposed; ///< Proposed position `x + dx` and
                          ///< energy `f(wrap(x + dx))`.
    state_type _best;     ///< Best position end energy encountered
                          ///< so far.
    tsallis_dist_type _tsallis_dist; ///< Tsallis distribution.
                                     ///<
    sa_pars_t   _pars;
    std::size_t _i;            ///< Current iteration.
                               ///<
    energy_fn_type _energy_fn; ///< Energy function `f`.
                               ///<
    wrap_fn_type _wrap_fn;     ///< Wrapping function `wrap` which ensures that
                               ///< `wrap(x + dx)` is in the domain of `f`.

  public:
    sa_chain_t(sa_buffers_t& buffers, sa_pars_t const& parameters,
               energy_fn_type energy_fn, wrap_fn_type wrap_fn,
               VSLStreamStatePtr stream)
        : _current{{}, buffers.current()}
        , _proposed{{}, buffers.proposed()}
        , _best{{}, buffers.best()}
        , _tsallis_dist{stream, buffers.gaussian(), buffers.gamma(),
                        parameters.q_V, parameters.t_0}
        , _pars{parameters}
        , _i{0}
        , _energy_fn{std::move(energy_fn)}
        , _wrap_fn{std::move(wrap_fn)}
    {
        _current.value = _energy_fn(_current.buffer);
        update_best();
    }

    sa_chain_t(sa_chain_t const&) = delete;
    sa_chain_t(sa_chain_t&&)      = delete;
    sa_chain_t& operator=(sa_chain_t const&) = delete;
    sa_chain_t& operator=(sa_chain_t&&) = delete;

    inline auto operator()() -> void;

    constexpr auto current() const noexcept -> state_type;
    constexpr auto best() const noexcept -> state_type;

  private:
    constexpr auto t_0() const noexcept { return _pars.t_0; }
    constexpr auto q_V() const noexcept { return _pars.q_V; }
    constexpr auto q_A() const noexcept { return _pars.q_A; }

    /// Returns the dimension of the parameter space.
    auto dim() const noexcept { return _current.buffer.size(); }

    /// Calculates the visiting temperature `t_V` for the given iteration.
    inline auto temperature(std::size_t const i) const noexcept -> float;

    /// Sets the best state equal to current.
    inline auto update_best() noexcept -> void;

    /// Accepts the move, i.e. `proposed` becomes `current`
    inline auto accept() noexcept -> void;

    /// Rejects the move.
    inline auto reject() noexcept -> void;

    /// Returns a uniform random number
    ///
    /// TODO(twesterhout): This function is terribly inefficient.
    inline auto uniform() -> float;

    inline auto accept_or_reject(float const dE, float const t_A) -> void;

    inline auto generate_full() -> void;

    inline auto generate_one(std::size_t const i) -> void;
};
// }}}

// {{{ sa_chain_t IMPLEMENTATION
template <class E, class W>
inline auto sa_chain_t<E, W>::temperature(std::size_t const i) const noexcept
    -> float
{
    auto const num = t_0() * (std::pow(2.0f, q_V() - 1.0f) - 1.0f);
    auto const den = std::pow(static_cast<float>(1 + i), q_V() - 1.0f) - 1.0f;
    return num / den;
}

template <class E, class W>
inline auto sa_chain_t<E, W>::update_best() noexcept -> void
{
    using std::begin, std::end;
    std::copy(begin(_current.buffer), end(_current.buffer),
              begin(_best.buffer));
    _best.value = _current.value;
}

template <class E, class W>
inline auto sa_chain_t<E, W>::accept() noexcept -> void
{
    using std::begin, std::end, std::swap;
    swap(_current, _proposed);
    if (_current.value < _best.value) { update_best(); }
}

template <class E, class W>
inline auto sa_chain_t<E, W>::reject() noexcept -> void
{}

template <class E, class W>
TCM_NOINLINE auto sa_chain_t<E, W>::uniform() -> float
{
    MKL_INT status;
    float   r;
    status = vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, _tsallis_dist.stream(), 1,
                          &r, 0.0f, 1.0f);
    detail::check_vsl_status_after_generation(status);
    return r;
}

template <class E, class W>
inline auto sa_chain_t<E, W>::accept_or_reject(float const dE, float const t_A)
    -> void
{
    // Always accept moves that reduce the energy
    if (dE < 0.0f) {
        accept();
        return;
    }

    // Eq. (5)
    auto const factor = 1.0f + (q_A() - 1.0f) * dE / t_A;
    auto const P_qA =
        factor <= 0.0f ? 0.0f : std::pow(factor, 1.0f / (1.0f - q_A()));
    if (uniform() <= P_qA) {
        // std::fprintf(stderr, "Ping!\n");
        accept();
    }
    else {
        // std::fprintf(stderr, "T_A = %f, factor = %f, r = %f, P_qA = %f\n",
        //              static_cast<double>(t_A), static_cast<double>(factor),
        //              static_cast<double>(r), static_cast<double>(P_qA));
        reject();
    }
}

template <class E, class W>
inline auto sa_chain_t<E, W>::generate_full() -> void
{
    using std::begin, std::end;

    _tsallis_dist(_proposed.buffer);
    std::transform(begin(_current.buffer), end(_current.buffer),
                   begin(_proposed.buffer), begin(_proposed.buffer),
                   [this](auto const x, auto const dx) -> float {
                       return _wrap_fn(x + dx);
                   });
    _proposed.value = _energy_fn(gsl::span<float const>{_proposed.buffer});
}

template <class E, class W>
inline auto sa_chain_t<E, W>::generate_one(std::size_t const i) -> void
{
    using std::begin, std::end;

    auto const dx = _tsallis_dist();
    std::copy(begin(_current.buffer), end(_current.buffer),
              begin(_proposed.buffer));
    _proposed.buffer[i] = _wrap_fn(_proposed.buffer[i] + dx);
    _proposed.value     = _energy_fn(gsl::span<float const>{_proposed.buffer});
}

template <class E, class W>
constexpr auto sa_chain_t<E, W>::current() const noexcept -> state_type
{
    return _current;
}

template <class E, class W>
constexpr auto sa_chain_t<E, W>::best() const noexcept -> state_type
{
    return _best;
}

template <class E, class W>
TCM_NOINLINE auto sa_chain_t<E, W>::operator()() -> void
{
    // (iv) Calculate new temperature...
    auto const t_V = temperature(_i);
    auto const t_A = t_V / static_cast<float>(_i + 1);
    _tsallis_dist.param(typename tsallis_dist_type::param_type{q_V(), t_V});

    // Markov chain at constant temperature
    for (auto j = std::size_t{0}; j < dim(); ++j) {
        generate_full();
        accept_or_reject(_proposed.value - _current.value, t_A);
    }
    for (auto j = std::size_t{0}; j < dim(); ++j) {
        generate_one(j);
        accept_or_reject(_proposed.value - _current.value, t_A);
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

TCM_NAMESPACE_END
