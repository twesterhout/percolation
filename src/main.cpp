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

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::fprintf(stderr, "Expected 1 argument: <L>\n");
        return EXIT_FAILURE;
    }
    auto const      length = std::atol(argv[1]);
    cubic_lattice_t lattice{nullptr, length};
    if (cubic_lattice_init(&lattice) != 0) {
        std::fprintf(stderr, "Could not initialise the lattice\n");
        return EXIT_FAILURE;
    }
    result_t result;
    auto const status = percolate(&lattice, &result);
    if (status != 0) {
        std::fprintf(stderr, "Error: %s", std::strerror(status));
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "%li\t%li\t%li\n", result.number_sites,
        result.number_clusters, result.max_cluster_size);
    cubic_lattice_deinit(&lattice);
    return EXIT_SUCCESS;
}

