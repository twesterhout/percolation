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

TCM_NOINLINE
auto random_generator() noexcept -> random_generator_t&
{
    static thread_local auto generator = seed_engine();
    return generator;
}

auto make_rng_stream(MKL_INT const method, MKL_UINT const seed)
    -> std::unique_ptr<VSLStreamStatePtr, vsl_stream_deleter_t>
{
    VSLStreamStatePtr stream;
    auto const        status = vslNewStream(&stream, method, seed);
    if (status == VSL_STATUS_OK) {
        try {
            return std::unique_ptr<VSLStreamStatePtr, vsl_stream_deleter_t>{
                new VSLStreamStatePtr{stream}};
        }
        catch (std::bad_alloc& /*unused*/) {
            vsl_stream_deleter_t{}(&stream);
            throw;
        }
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

auto make_rng_stream(MKL_INT const method)
    -> std::unique_ptr<VSLStreamStatePtr, vsl_stream_deleter_t>
{
    std::random_device urandom;
    return make_rng_stream(method, urandom());
}

TCM_NAMESPACE_END
