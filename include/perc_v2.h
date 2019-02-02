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
#include "convolution.h"
#include "lattice.h"
#include "tsallis_distribution.h"
#if defined(__cplusplus)
#    include <cstddef>
#else
#    include <stddef.h>
#endif

struct _tcm_percolation_results {
    double* number_clusters;
    double* max_cluster_size;
    double* has_wrapped_one;
    double* has_wrapped_two;
    double* chirality;
    double* helicity;
    double* magnetisation;
};

struct _tcm_percolation_stats {
    double* max_magnetic_cluster_size;
    double* mean_magnetic_cluster_size;
    double* max_number_children;
    double* mean_number_children;
};

#if defined(__cplusplus)
using tcm_percolation_results_t = _tcm_percolation_results;
using tcm_percolation_stats_t   = _tcm_percolation_stats;
#else
typedef struct _tcm_percolation_results tcm_percolation_results_t;
typedef struct _tcm_percolation_stats   tcm_percolation_stats_t;
#endif

#if defined(__cplusplus)
extern "C" {
#endif

TCM_EXPORT
int tcm_percolate_square(size_t, size_t, tcm_square_lattice_t,
                         tcm_percolation_results_t const*,
                         tcm_percolation_stats_t const*);

TCM_EXPORT int tcm_percolate_cubic(tcm_cubic_lattice_t,
                                   tcm_percolation_results_t const*);

TCM_EXPORT int tcm_percolate_triangular(tcm_triangular_lattice_t,
                                        tcm_percolation_results_t const*);
TCM_EXPORT int
tcm_percolate_triangular_stacked(tcm_triangular_stacked_lattice_t,
                                 tcm_percolation_results_t const*);

#if defined(__cplusplus)
} // extern "C"
#endif
