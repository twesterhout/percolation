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

#include "config.hpp"

#include <immintrin.h>

#if defined(__cplusplus)
#include <cstdint>
extern "C" {
#else
#include <stdint.h>
#include <stdbool.h>
#endif

#if defined(__cplusplus)
static_assert(
    sizeof(void*) == sizeof(std::intptr_t), "What kind of system is this?");
#endif


#if 0
typedef enum tcm_lattice_kind {
    TCM_SQUARE = 0,
    TCM_CUBIC = 1
} tcm_lattice_kind_t;

#if defined(__cplusplus)
static_assert(sizeof(tcm_lattice_kind_t) == 4,
    "It is assumed that tcm_lattice_kind_t occupies exactly 32 bits.");
#endif

typedef struct tcm_lattice {
    tcm_lattice_kind_t _type;
    char               _data[28];
} tcm_lattice_t;

#if defined(__cplusplus)
static_assert(sizeof(tcm_lattice_t) == 32,
    "It is assumed that tcm_lattice_t occupies exactly 256 bits.");
#endif
#endif


struct _tcm_square_lattice {
    int32_t (*neighbours)[4];
    int32_t length;
    int32_t size;
    bool    periodic;
};
#if defined(__cplusplus)
using tcm_square_lattice_t = _tcm_square_lattice;
#else
typedef struct _tcm_square_lattice tcm_square_lattice_t;
#endif

struct _tcm_cubic_lattice {
    int32_t (*neighbours)[6];
    int32_t length;
    int32_t size;
    bool    periodic;
};
#if defined(__cplusplus)
using tcm_cubic_lattice_t = _tcm_cubic_lattice;
#else
typedef struct _tcm_cubic_lattice  tcm_cubic_lattice_t;
#endif

struct _tcm_triangular_lattice {
    int32_t (*neighbours)[6];
    int32_t length;
    int32_t length_y;
    int32_t size;
};
#if defined(__cplusplus)
using tcm_triangular_lattice_t = _tcm_triangular_lattice;
#else
typedef struct _tcm_triangular_lattice tcm_triangular_lattice_t;
#endif

struct _tcm_triangular_stacked_lattice {
    int32_t (*neighbours)[8];
    int32_t length;
    int32_t length_y;
    int32_t size;
};
#if defined(__cplusplus)
using tcm_triangular_stacked_lattice_t = _tcm_triangular_stacked_lattice;
#else
typedef struct _tcm_triangular_stacked_lattice tcm_triangular_stacked_lattice_t;
#endif

typedef struct tcm_convolution_state {
    int32_t              number_sites;
    int32_t              number_functions;
    double const* const* functions;
    double* const*       outputs;
} tcm_convolution_state_t;

typedef enum tcm_boundary {
    TCM_BOUNDARY_X_LOW  = 1,
    TCM_BOUNDARY_X_HIGH = 2,
    TCM_BOUNDARY_Y_LOW  = 4,
    TCM_BOUNDARY_Y_HIGH = 8,
    TCM_BOUNDARY_Z_LOW  = 16,
    TCM_BOUNDARY_Z_HIGH = 32,
    _TCM_BOUNDARY_MAX   = 64
} tcm_boundary_t;

typedef struct tcm_geometric_stats {
    int64_t number_sites;
    int64_t number_clusters;
    int64_t max_cluster_size;
    bool    has_wrapped[3];
} tcm_geometric_stats_t;


#if defined(TCM_CLANG)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#endif
typedef union tcm_V2d {
    __m128 raw;
    struct {
        double x;
        double y;
    };
} tcm_V2d_t;
#if defined(TCM_CLANG)
#pragma clang diagnostic pop
#endif

#if defined(__cplusplus)
struct tcm_magnetic_stats {
    tcm_V2d_t magnetisation[3];
    int64_t   chirality;
    int64_t   helicity;
};
using tcm_magnetic_stats_t = tcm_magnetic_stats;
#endif

typedef struct tcm_percolation_results {
    double* number_clusters;
    double* max_cluster_size;
    double* has_wrapped_one;
    double* has_wrapped_two;
    double* chirality;
    double* helicity;
    double* magnetisation;
} tcm_percolation_results_t;

#if 0
typedef struct result {
    int64_t number_sites;
    int64_t number_clusters;
    int64_t max_cluster_size;
    int64_t has_wrapped;
} result_t;
#endif

/**
 * Initialises a new square lattice with side length `length`. If `precompute`
 * is `true` adjacency list is calculated and cached such that accessing nearest
 * neighbours is a matter a simple array lookup.
 *
 * NOTE: if precompute is true, a new array of O(N) (16*N to be precise) bytes
 * is allocated.
 */
#if 0
TCM_EXPORT int tcm_square_lattice_init(
    int32_t length, bool precompute, tcm_lattice_t* out);
TCM_EXPORT void tcm_square_lattice_deinit(tcm_lattice_t*);

TCM_EXPORT int  tcm_cubic_lattice_init(int32_t, bool, tcm_lattice_t*);
TCM_EXPORT void tcm_cubic_lattice_deinit(tcm_lattice_t*);

TCM_EXPORT void tcm_lattice_print(tcm_lattice_t const*);
TCM_EXPORT int32_t tcm_lattice_size(tcm_lattice_t const*);

TCM_EXPORT int tcm_percolate(tcm_lattice_t const* lattice,
    double* number_clusters, double* max_cluster_size, double* has_wrapped_x,
    double* has_wrapped_y, double* has_wrapped_z);
#endif

TCM_EXPORT int tcm_compute_neighbours_square(tcm_square_lattice_t);
TCM_EXPORT int tcm_compute_neighbours_cubic(tcm_cubic_lattice_t);
TCM_EXPORT int tcm_compute_neighbours_triangular(tcm_triangular_lattice_t);
TCM_EXPORT int tcm_compute_neighbours_triangular_stacked(tcm_triangular_stacked_lattice_t);

int tcm_percolate_square(
    tcm_square_lattice_t, tcm_percolation_results_t const*);
int tcm_percolate_cubic(tcm_cubic_lattice_t, tcm_percolation_results_t const*);
TCM_EXPORT int tcm_percolate_triangular(
    tcm_triangular_lattice_t, tcm_percolation_results_t const*);
int tcm_percolate_triangular_stacked(
    tcm_triangular_stacked_lattice_t, tcm_percolation_results_t const*);

TCM_EXPORT int tcm_convolution(
    int32_t n_min, int32_t n_max, tcm_convolution_state_t const* state);

/*
TCM_EXPORT int cubic_lattice_init(cubic_lattice_t*);

TCM_EXPORT void cubic_lattice_deinit(cubic_lattice_t*);

TCM_EXPORT int percolate(cubic_lattice_t const*, result_t*);


TCM_EXPORT int tcm_calculate_binomials(int64_t, int64_t, double*);

TCM_EXPORT int tcm_convolution(int64_t const N, int64_t const n_max,
    double const* functions[], double out[], int64_t out_size);
*/

#if defined(__cplusplus)
} // extern "C"
#endif
