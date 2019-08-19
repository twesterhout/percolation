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

// vim: foldenable foldmethod=marker
#pragma once

#include "config.h"
#if defined(__cplusplus)
#    include <cstdint>
#else
#    include <stdbool.h>
#    include <stdint.h>
#endif

/// Two-dimensional square lattice.
struct _tcm_square_lattice {
    int64_t (*neighbours)[4]; /// Adjacency list of the graph
    int64_t length;           ///< Side length
    int64_t size;             ///< Total number of spins
    bool    periodic;         ///< Whether boundary conditions are periodic
};

/// Three-dimensional cubic lattice
struct _tcm_cubic_lattice {
    int64_t (*neighbours)[6]; ///< Adjacency list of the graph
    int64_t length;           ///< Side length
    int64_t size;             ///< Total number of spins
    bool    periodic;         ///< Whether boundary conditions are periodic
};

/// Two-dimensional triangular lattice
struct _tcm_triangular_lattice {
    int64_t (*neighbours)[6]; ///< Adjacency list of the graph
    int64_t length;           ///< Width of the sample
    int64_t length_y;         ///< Height of the sample
    int64_t size;             ///< Total number of spins
};

/// Three dimensional lattice which is build by vertically stacking layers with
/// triangular lattice.
struct _tcm_triangular_stacked_lattice {
    int64_t (*neighbours)[8];
    int64_t length;
    int64_t length_y;
    int64_t size;
};

#if defined(__cplusplus)
using tcm_square_lattice_t             = _tcm_square_lattice;
using tcm_cubic_lattice_t              = _tcm_cubic_lattice;
using tcm_triangular_lattice_t         = _tcm_triangular_lattice;
using tcm_triangular_stacked_lattice_t = _tcm_triangular_stacked_lattice;
#else
typedef struct _tcm_square_lattice             tcm_square_lattice_t;
typedef struct _tcm_cubic_lattice              tcm_cubic_lattice_t;
typedef struct _tcm_triangular_lattice         tcm_triangular_lattice_t;
typedef struct _tcm_triangular_stacked_lattice tcm_triangular_stacked_lattice_t;
#endif

typedef enum tcm_boundary {
    TCM_BOUNDARY_X_LOW  = 1,
    TCM_BOUNDARY_X_HIGH = 2,
    TCM_BOUNDARY_Y_LOW  = 4,
    TCM_BOUNDARY_Y_HIGH = 8,
    TCM_BOUNDARY_Z_LOW  = 16,
    TCM_BOUNDARY_Z_HIGH = 32,
    _TCM_BOUNDARY_MAX   = 64
} tcm_boundary_t;

#if defined(__cplusplus)
extern "C" {
#endif

TCM_EXPORT int tcm_compute_neighbours_square(tcm_square_lattice_t);
TCM_EXPORT int tcm_compute_neighbours_cubic(tcm_cubic_lattice_t);
TCM_EXPORT int tcm_compute_neighbours_triangular(tcm_triangular_lattice_t);
TCM_EXPORT int
    tcm_compute_neighbours_triangular_stacked(tcm_triangular_stacked_lattice_t);

#if defined(__cplusplus)
} // extern "C"
#endif

// {{{ position_to_index
static inline int64_t
tcm_triangular_position_to_index(tcm_triangular_lattice_t const* lattice,
                                 int64_t const x, int64_t const y)
{
    int boundaries = 0;

    int64_t const X = x + (y - (y < 0 ? 1 : 0)) / 2;
    if (X < 0) { boundaries |= TCM_BOUNDARY_X_LOW; }
    else if (X >= lattice->length) {
        boundaries |= TCM_BOUNDARY_X_HIGH;
    }

    int64_t const Y = y;
    if (Y < 0) { boundaries |= TCM_BOUNDARY_Y_LOW; }
    else if (Y >= lattice->length_y) {
        boundaries |= TCM_BOUNDARY_Y_HIGH;
    }

    return boundaries == 0 ? lattice->length * Y + X : -boundaries;
}

static inline int64_t tcm_triangular_stacked_position_to_index(
    tcm_triangular_stacked_lattice_t const* lattice, int64_t const x,
    int64_t const y, int64_t const z)
{
    int boundaries = 0;

    int64_t const X = x + (y - (y < 0 ? 1 : 0)) / 2;
    if (X < 0) { boundaries |= TCM_BOUNDARY_X_LOW; }
    else if (X >= lattice->length) {
        boundaries |= TCM_BOUNDARY_X_HIGH;
    }

    int64_t const Y = y;
    if (Y < 0) { boundaries |= TCM_BOUNDARY_Y_LOW; }
    else if (Y >= lattice->length_y) {
        boundaries |= TCM_BOUNDARY_Y_HIGH;
    }

    int64_t const Z = z;
    if (Z < 0) { boundaries |= TCM_BOUNDARY_Z_LOW; }
    else if (Z >= lattice->length) {
        boundaries |= TCM_BOUNDARY_Z_HIGH;
    }

    return boundaries == 0 ? lattice->length * lattice->length_y * Z
                                 + lattice->length * Y + X
                           : -boundaries;
}

static inline int64_t
tcm_square_position_to_index(tcm_square_lattice_t const* lattice,
                             int64_t const x, int64_t const y)
{
    return lattice->length * y + x;
}

static inline int64_t
tcm_cubic_position_to_index(tcm_cubic_lattice_t const* lattice, int64_t const x,
                            int64_t const y, int64_t const z)
{
    return lattice->length * lattice->length * z + lattice->length * y + x;
}
// }}}
