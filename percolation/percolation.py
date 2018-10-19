#!/usr/bin/env python3
#
# Copyright Tom Westerhout (c) 2018
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#
#     * Neither the name of Tom Westerhout nor the names of other
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from ctypes import *
import itertools
import multiprocessing
import os
import re
import sys

import numpy as np
import psutil

_percolation = cdll.LoadLibrary(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 'libpercolation.so'))

class Lattice(Structure):
    _fields_ = [('_neighbours', POINTER(c_long)),
                ('_length', c_long)]

    def __init__(self, length):
        super().__init__(None, length)
        if length <= 2:
            raise ValueError('L must be at least 3, but got {}'.format(length))
        status = _percolation.cubic_lattice_init(byref(self))
        if status != 0:
            raise RuntimeError(
                'Failed to initialise the Lattice: {}'.format(os.strerror(status)))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _percolation.cubic_lattice_deinit(byref(self))

class Result(Structure):
    _fields_ = [('number_sites', c_long),
                ('number_clusters', c_long),
                ('max_cluster_size', c_long),
                ('is_percolating', c_bool)]

    def __str__(self):
        return 'Result(number_sites={}, number_clusters={}, ' \
               'max_cluster_size={}, is_percolating={})'\
                    .format(self.number_sites,
                            self.number_clusters,
                            self.max_cluster_size,
                            self.is_percolating)

    def to_array(self):
        return np.array([self.number_sites, self.number_clusters,
                         self.max_cluster_size], dtype=np.float64)

def percolate(length, batch_size=1):
    result = Result()
    status = 0
    pcs = np.empty((batch_size,), dtype=float)
    with Lattice(length) as lattice:
        for i in range(batch_size):
            status = _percolation.percolate(byref(lattice), byref(result))
            if status != 0:
                raise RuntimeError('Failed to percolate: {}'\
                                   .format(os.strerror(status)))
            pcs[i] = float(result.number_sites)
    pcs /= float(length**3)
    return pcs.mean(), pcs.var()

def sample(number_samples, lengths, out_file):
    for length in lengths:

        cpu_count = psutil.cpu_count(logical=False)
        number_batches = 10 * cpu_count
        if number_samples % number_batches != 0:
            raise ValueError('<number_samples> must be dividible by {}'\
                             .format(number_batches))
        batch_size = number_samples // number_batches
        assert batch_size > 0 and number_samples % number_batches == 0

        with multiprocessing.Pool(cpu_count) as p:
            results = p.starmap(percolate,
                [(length, batch_size) for _ in range(number_batches)])
        pcs = np.array(results, dtype=float)

        def save_to(f):
            f.write('{}\t{:.20e}\t{:.20e}\t{:.20e}\t{:.20e}\n'.format(
                length, pcs[:,0].mean(), pcs[:,0].var(),
                        pcs[:,1].mean(), pcs[:,1].var()))
        if out_file is not None:
            with open(out_file, 'a') as f:
                save_to(f)
        else:
            save_to(sys.stdout)
