#include "convolution.h"
#include <array>
#include <cmath>
#include <limits>

TCM_NAMESPACE_BEGIN

template <std::size_t Count>
auto convolution(std::int64_t const             n_max,
                 tcm_convolution_state_t const& state) noexcept -> void
{
    // floor(sqrt(2^52 - 1)), because a double has 52 bits matissa, and we
    // need to store number_sites^2 exactly.
    constexpr std::int64_t max_number_sites = 67108863;
    constexpr auto         eps = std::numeric_limits<double>::epsilon();
    static_assert(Count > 0);
    TCM_ASSERT(state.functions != nullptr && state.outputs != nullptr,
               "Inputs and outputs must not be NULL");
    TCM_ASSERT(1 <= state.number_sites
                   && state.number_sites <= max_number_sites,
               "Invalid number of sites");
    TCM_ASSERT(0 <= n_max && n_max <= state.number_sites,
               "`n_max` out of bounds");
    TCM_ASSERT(state.number_functions == Count, "Bug!");

    // Initialise the results
    std::array<double, Count> results;
    for (auto& r : results) {
        r = 0.0;
    }

    auto const add_to_result =
        [&results, fns = state.functions ](auto const n, auto const B) noexcept
    {
        for (std::size_t i = 0; i < Count; ++i) {
            results[i] += B * fns[i][n];
        }
        for (auto const r : results) {
            TCM_ASSERT(!std::isnan(r), "Results must not be NaNs");
        }
    };
    std::int64_t const N = state.number_sites;

    // Postulate that B(N, n_max, p) = 1
    auto B     = 1.0;
    auto B_sum = B;
    add_to_result(n_max, B);
    // We use Eq. (10): for n > n_max
    //
    // B(N, n, p) = B(N, n - 1, p) * (N - n + 1) / n * p / (1 - p)
    //            = B(N, n - 1, p) * (N - n + 1) / n * n_max / (N - n_max)
    //            = B(N, n - 1, p) * ((N - n + 1)*n_max) / (n*(N - n_max))
    for (std::int64_t n = n_max + 1; n <= N && B >= eps; ++n) {
        B *= static_cast<double>((N - n + 1) * n_max)
             / static_cast<double>(n * (N - n_max));
        TCM_ASSERT(!std::isnan(B), "B must not be NaN");
        B_sum += B;
        add_to_result(n, B);
    }
    // We use Eq. (10): for n < n_max
    //
    // B(N, n, p) = B(N, n + 1, p) * (n + 1) / (N - n) * (1 - p) / p
    //            = B(N, n + 1, p) * (n + 1) / (N - n) * (N - n_max) / n_max
    //            = B(N, n + 1, p) * ((n + 1)*(N - n_max)) / ((N - n)*n_max)
    B = 1.0;
    for (std::int64_t n = n_max - 1; n >= 0 && B >= eps; --n) {
        B *= static_cast<double>((n + 1) * (N - n_max))
             / static_cast<double>((N - n) * n_max);
        TCM_ASSERT(!std::isnan(B), "B must not be NaN");
        B_sum += B;
        add_to_result(n, B);
    }

    // Normalise the distribution B
    for (auto& r : results) {
        r /= B_sum;
    }
    // Write the results to outputs
    state.outputs[0][n_max] =
        static_cast<double>(n_max) / static_cast<double>(N);
    for (std::size_t i = 0; i < Count; ++i) {
        state.outputs[i + 1][n_max] = results[i];
    }
}

TCM_NAMESPACE_END

extern "C" TCM_EXPORT int tcm_convolution(int64_t const                  n_min,
                                          int64_t const                  n_max,
                                          tcm_convolution_state_t const* state)
{
    using tcm::convolution;

    if (!(state != nullptr && state->functions != nullptr
          && state->outputs != nullptr && state->number_sites >= 1
          && state->number_functions >= 1)) {
        return EINVAL;
    }
    if (n_min < 0 || n_max > state->number_sites || n_min > n_max) {
        return EDOM;
    }
    switch (state->number_functions) {
    case 1:
        for (auto n = n_min; n <= n_max; ++n) {
            convolution<1>(n, *state);
        }
        return 0;
    case 2:
        for (auto n = n_min; n <= n_max; ++n) {
            convolution<2>(n, *state);
        }
        return 0;
    case 3:
        for (auto n = n_min; n <= n_max; ++n) {
            convolution<3>(n, *state);
        }
        return 0;
    case 4:
        for (auto n = n_min; n <= n_max; ++n) {
            convolution<4>(n, *state);
        }
        return 0;
    case 5:
        for (auto n = n_min; n <= n_max; ++n) {
            convolution<5>(n, *state);
        }
        return 0;
    default: return EDOM;
    };
}




