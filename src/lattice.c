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

#include "lattice.h"
#include <errno.h>
#include <math.h>
#include <tgmath.h>
#include <stddef.h>

static inline void
tcm_square_periodic_neighbours_for(tcm_square_lattice_t const* lattice,
                                   int64_t const i, int64_t const x,
                                   int64_t const y, int64_t nn[4])
{
    TCM_ASSERT(lattice != NULL, "Lattice should not be NULL");
    TCM_ASSERT(lattice->size == lattice->length * lattice->length,
               "Lattice is in an invalid state.");
    TCM_ASSERT(0 <= i && i < lattice->size, "Index out of bounds.");
    TCM_ASSERT(lattice->periodic,
               "Only periodic boundary conditions supported.");
    int64_t const L  = lattice->length;
    int64_t const L2 = L * L;
    if (x == 0) {
        nn[0] = i + 1;
        nn[1] = i + L - 1;
    }
    else if (x == L - 1) {
        nn[0] = i - L + 1;
        nn[1] = i - 1;
    }
    else {
        nn[0] = i + 1;
        nn[1] = i - 1;
    }
    if (y == 0) {
        nn[2] = i + L;
        nn[3] = i + L2 - L;
    }
    else if (y == L - 1) {
        nn[2] = i - L2 + L;
        nn[3] = i - L;
    }
    else {
        nn[2] = i + L;
        nn[3] = i - L;
    }
}

static inline void
tcm_square_open_neighbours_for(tcm_square_lattice_t const* lattice,
                               int64_t const i, int64_t const x,
                               int64_t const y, int64_t nn[4])
{
    TCM_ASSERT(lattice != NULL, "Lattice should not be NULL");
    TCM_ASSERT(lattice->size == lattice->length * lattice->length,
               "Lattice is in an invalid state.");
    TCM_ASSERT(0 <= i && i < lattice->size, "Index out of bounds.");
    TCM_ASSERT(!lattice->periodic, "Only open boundary conditions supported.");
    int64_t const L = lattice->length;
    if (x == 0) {
        nn[0] = i + 1;
        nn[1] = -TCM_BOUNDARY_X_LOW;
    }
    else if (x == L - 1) {
        nn[0] = -TCM_BOUNDARY_X_HIGH;
        nn[1] = i - 1;
    }
    else {
        nn[0] = i + 1;
        nn[1] = i - 1;
    }
    if (y == 0) {
        nn[2] = i + L;
        nn[3] = -TCM_BOUNDARY_Y_LOW;
    }
    else if (y == L - 1) {
        nn[2] = -TCM_BOUNDARY_Y_HIGH;
        nn[3] = i - L;
    }
    else {
        nn[2] = i + L;
        nn[3] = i - L;
    }
}

static inline void tcm_cubic_periodic_neighbours_for(
    tcm_cubic_lattice_t const* lattice, int64_t const i, int64_t const x,
    int64_t const y, int64_t const z, int64_t nn[6])
{
    TCM_ASSERT(lattice != NULL, "Lattice should not be NULL");
    TCM_ASSERT(lattice->size
                   == lattice->length * lattice->length * lattice->length,
               "Lattice is in an invalid state.");
    TCM_ASSERT(0 <= i && i < lattice->size, "Index out of bounds.");
    TCM_ASSERT(lattice->periodic,
               "Only periodic boundary conditions supported.");
    int64_t const L  = lattice->length;
    int64_t const L2 = L * L;
    int64_t const L3 = L * L * L;
    // X
    if (x == 0) {
        nn[0] = i + 1;
        nn[1] = i + L - 1;
    }
    else if (x == L - 1) {
        nn[0] = i - L + 1;
        nn[1] = i - 1;
    }
    else {
        nn[0] = i + 1;
        nn[1] = i - 1;
    }
    // Y
    if (y == 0) {
        nn[2] = i + L;
        nn[3] = i + L2 - L;
    }
    else if (y == L - 1) {
        nn[2] = i - L2 + L;
        nn[3] = i - L;
    }
    else {
        nn[2] = i + L;
        nn[3] = i - L;
    }
    // Z
    if (z == 0) {
        nn[4] = i + L2;
        nn[5] = i + L3 - L2;
    }
    else if (z == L - 1) {
        nn[4] = i - L3 + L2;
        nn[5] = i - L2;
    }
    else {
        nn[4] = i + L2;
        nn[5] = i - L2;
    }
}

static inline void
tcm_cubic_open_neighbours_for(tcm_cubic_lattice_t const* lattice,
                              int64_t const i, int64_t const x, int64_t const y,
                              int64_t const z, int64_t nn[6])
{
    TCM_ASSERT(lattice != NULL, "Lattice should not be NULL");
    TCM_ASSERT(lattice->size
                   == lattice->length * lattice->length * lattice->length,
               "Lattice is in an invalid state.");
    TCM_ASSERT(0 <= i && i < lattice->size, "Index out of bounds.");
    TCM_ASSERT(!lattice->periodic, "Only open boundary conditions supported.");
    int64_t const L  = lattice->length;
    int64_t const L2 = L * L;
    // X
    if (x == 0) {
        nn[0] = i + 1;
        nn[1] = -TCM_BOUNDARY_X_LOW;
    }
    else if (x == L - 1) {
        nn[0] = -TCM_BOUNDARY_X_HIGH;
        nn[1] = i - 1;
    }
    else {
        nn[0] = i + 1;
        nn[1] = i - 1;
    }
    // Y
    if (y == 0) {
        nn[2] = i + L;
        nn[3] = -TCM_BOUNDARY_Y_LOW;
    }
    else if (y == L - 1) {
        nn[2] = -TCM_BOUNDARY_Y_HIGH;
        nn[3] = i - L;
    }
    else {
        nn[2] = i + L;
        nn[3] = i - L;
    }
    // Z
    if (z == 0) {
        nn[4] = i + L2;
        nn[5] = -TCM_BOUNDARY_Z_LOW;
    }
    else if (z == L - 1) {
        nn[4] = -TCM_BOUNDARY_Z_HIGH;
        nn[5] = i - L2;
    }
    else {
        nn[4] = i + L2;
        nn[5] = i - L2;
    }
}

TCM_EXPORT int tcm_compute_neighbours_square(tcm_square_lattice_t const lattice)
{
    static int64_t const max_length = 46340; // floor(sqrt(2^31))
    if (lattice.length > max_length) { return EDOM; }
    if (lattice.size != lattice.length * lattice.length) { return EINVAL; }
    if (lattice.periodic) {
        if (lattice.length < 3) { return EDOM; }
        int64_t i = 0;
        for (int64_t y = 0; y < lattice.length; ++y) {
            for (int64_t x = 0; x < lattice.length; ++x, ++i) {
                tcm_square_periodic_neighbours_for(&lattice, i, x, y,
                                                   lattice.neighbours[i]);
            }
        }
    }
    else {
        if (lattice.length < 2) { return EDOM; }
        int64_t i = 0;
        for (int64_t y = 0; y < lattice.length; ++y) {
            for (int64_t x = 0; x < lattice.length; ++x, ++i) {
                tcm_square_open_neighbours_for(&lattice, i, x, y,
                                               lattice.neighbours[i]);
            }
        }
    }
    return 0;
}

TCM_EXPORT int tcm_compute_neighbours_cubic(tcm_cubic_lattice_t const lattice)
{
    static int64_t const max_length = 1290; // floor((2^31)^(1/3))
    if (lattice.length < 3 || lattice.length > max_length) { return EDOM; }
    if (lattice.size != lattice.length * lattice.length * lattice.length) {
        return EINVAL;
    }
    if (lattice.periodic) {
        int64_t i = 0;
        for (int64_t z = 0; z < lattice.length; ++z) {
            for (int64_t y = 0; y < lattice.length; ++y) {
                for (int64_t x = 0; x < lattice.length; ++x, ++i) {
                    tcm_cubic_periodic_neighbours_for(&lattice, i, x, y, z,
                                                      lattice.neighbours[i]);
                }
            }
        }
    }
    else {
        int64_t i = 0;
        for (int64_t z = 0; z < lattice.length; ++z) {
            for (int64_t y = 0; y < lattice.length; ++y) {
                for (int64_t x = 0; x < lattice.length; ++x, ++i) {
                    tcm_cubic_open_neighbours_for(&lattice, i, x, y, z,
                                                  lattice.neighbours[i]);
                }
            }
        }
    }
    return 0;
}

TCM_EXPORT int
tcm_compute_neighbours_triangular(tcm_triangular_lattice_t const lattice)
{
    static int64_t const max_length = 43125; // floor(sqrt(2^31 * sqrt(3)/2))
    if (lattice.length < 2 || lattice.length > max_length) { return EDOM; }
#if defined(TCM_CLANG)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wbad-function-cast"
#endif
    int64_t const expected_length_y =
        (int64_t)round(lattice.length * 2.0L / sqrt(3.0L));
#if defined(TCM_CLANG)
#    pragma clang diagnostic pop
#endif
    if (lattice.length_y != expected_length_y
        || lattice.size != lattice.length * lattice.length_y) {
        return EINVAL;
    }
    int64_t i = 0;
    for (int64_t y = 0; y < lattice.length_y; ++y) {
        for (int64_t _x = 0; _x < lattice.length; ++_x, ++i) {
            int64_t const  x  = _x - y / 2;
            int64_t* const nn = lattice.neighbours[i];
            // clang-format off
            nn[0] = tcm_triangular_position_to_index(&lattice, x - 1, y    );
            nn[1] = tcm_triangular_position_to_index(&lattice, x - 1, y + 1);
            nn[2] = tcm_triangular_position_to_index(&lattice, x,     y + 1);
            nn[3] = tcm_triangular_position_to_index(&lattice, x + 1, y    );
            nn[4] = tcm_triangular_position_to_index(&lattice, x + 1, y - 1);
            nn[5] = tcm_triangular_position_to_index(&lattice, x,     y - 1);
            // clang-format on
        }
    }
    return 0;
}

TCM_EXPORT int tcm_compute_neighbours_triangular_stacked(
    tcm_triangular_stacked_lattice_t const lattice)
{
    static int64_t const max_length = 1229; // floor((2^31 * sqrt(3)/2)^(1/3))
    if (lattice.length < 2 || lattice.length > max_length) { return EDOM; }
#if defined(TCM_CLANG)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wbad-function-cast"
#endif
    int64_t const expected_length_y =
        (int64_t)round(lattice.length * 2.0L / sqrt(3.0L));
#if defined(TCM_CLANG)
#    pragma clang diagnostic pop
#endif
    if (lattice.length_y != expected_length_y
        || lattice.size != lattice.length * lattice.length_y * lattice.length) {
        return EINVAL;
    }
    int64_t i = 0;
    for (int64_t z = 0; z < lattice.length; ++z) {
        for (int64_t y = 0; y < lattice.length_y; ++y) {
            for (int64_t _x = 0; _x < lattice.length; ++_x, ++i) {
                int64_t const  x  = _x - y / 2;
                int64_t* const nn = lattice.neighbours[i];
                // clang-format off
                nn[0] = tcm_triangular_stacked_position_to_index(&lattice, x - 1, y,     z    );
                nn[1] = tcm_triangular_stacked_position_to_index(&lattice, x - 1, y + 1, z    );
                nn[2] = tcm_triangular_stacked_position_to_index(&lattice, x,     y + 1, z    );
                nn[3] = tcm_triangular_stacked_position_to_index(&lattice, x + 1, y,     z    );
                nn[4] = tcm_triangular_stacked_position_to_index(&lattice, x + 1, y - 1, z    );
                nn[5] = tcm_triangular_stacked_position_to_index(&lattice, x,     y - 1, z    );
                nn[6] = tcm_triangular_stacked_position_to_index(&lattice, x,     y,     z - 1);
                nn[7] = tcm_triangular_stacked_position_to_index(&lattice, x,     y,     z + 1);
                // clang-format on
            }
        }
    }
    return 0;
}

