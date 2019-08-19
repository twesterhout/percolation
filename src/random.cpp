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

#include "detail/random.hpp"
#include <algorithm>
#include <functional>

TCM_NAMESPACE_BEGIN

namespace {
inline auto seed_engine()
{
    using std::begin, std::end;
    constexpr std::size_t N = (random_generator_t::word_size + 31) / 32
                              * random_generator_t::state_size;
    std::uint32_t      random_data[N];
    std::random_device source;
    std::generate(begin(random_data), end(random_data), std::ref(source));
    std::seed_seq seeds(begin(random_data), end(random_data));
    return random_generator_t{seeds};
    // return Gen{};
}
} // namespace

auto random_generator() noexcept -> random_generator_t&
{
    static thread_local auto generator = seed_engine();
    return generator;
}

/// Functor for destroying `VSLStreamStatePtr`s.
struct vsl_stream_deleter_t {
    auto operator()(VSLStreamStatePtr p) const noexcept
    {
        TCM_ASSERT(p != nullptr, "Trying to delete a nullptr");
        // TODO(twesterhout): Yeah, I know, ignoring the error code is not a
        // great idea, but we can't do anything about the errors anyway.
        vslDeleteStream(&p);
    }
};

/// Function to guard against the case when VSLStreamStatePtr is not a pointer.
///
/// In that case, rather than failing with a weird error message, we trigger a
/// static assertion failure with an explanation of the problem.
template <class _Dummy1 = void, class = void,
          class = typename std::enable_if<
              std::is_same<_Dummy1, _Dummy1>::value
              && !std::is_pointer<VSLStreamStatePtr>::value>::type>
auto make_rng_stream(MKL_INT /*unused*/, MKL_UINT /*unused*/) -> void
{
    static_assert(std::is_pointer<VSLStreamStatePtr>::value,
                  "It is assumed that `VSLStreamStatePtr` is a pointer (most "
                  "likely a `void *`).");
}

template <class _Dummy1 = void,
          class         = typename std::enable_if<
              std::is_same<_Dummy1, _Dummy1>::value
              && std::is_pointer<VSLStreamStatePtr>::value>::type>
auto make_rng_stream(MKL_INT const method, MKL_UINT const seed)
    -> std::unique_ptr<std::remove_pointer<VSLStreamStatePtr>::type,
                       vsl_stream_deleter_t>
{
    VSLStreamStatePtr stream;
    auto const        status = vslNewStream(&stream, method, seed);
    if (status == VSL_STATUS_OK) {
        return std::unique_ptr<std::remove_pointer<VSLStreamStatePtr>::type,
                               vsl_stream_deleter_t>{stream};
    }

    char const* const msg = [](auto code) {
        switch (code) {
        case VSL_RNG_ERROR_INVALID_BRNG_INDEX: return "BRNG index is invalid";
        case VSL_ERROR_MEM_FAILURE:
            return "System cannot allocate memory for stream";
        case VSL_RNG_ERROR_NONDETERM_NOT_SUPPORTED:
            return "Non-deterministic random number generator is not supported";
        case VSL_RNG_ERROR_ARS5_NOT_SUPPORTED:
            return "ARS-5 random number generator is not supported on "
                   "the CPU running the application";
        default: return "unknown error code: a bug in Intel MKL?";
        } // end switch
    }(status);
    throw std::runtime_error{msg};
}

TCM_EXPORT auto random_stream() -> VSLStreamStatePtr
{
    auto const get_seed = []() {
        std::random_device rd;
        return rd();
    };
    thread_local auto vsl_stream = make_rng_stream(VSL_BRNG_ARS5, get_seed());
    return vsl_stream.get();
}

auto enumerate_sites(size_t const n)
    -> std::unique_ptr<int32_t[], free_deleter_t>
{
    auto sites = make_buffer_of<int32_t>(n);
    for (auto i = 0; i < n; ++i) {
        sites[static_cast<size_t>(i)] = i;
    }
    return sites;
}

TCM_NAMESPACE_END
