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

import click
import percolation

class IntList(click.ParamType):
    name = 'IntList'

    def convert(self, value, param, ctx):
        try:
            def natural(x):
                x = int(x)
                if x < 3:
                    self.fail('L >= 3, but got {}'.format(x), param, ctx)
                return x
            return [natural(x) for x in value.split(',')]
        except ValueError:
            self.fail('{} is not a valid list of integers'.format(value),
                      param, ctx)

INT_LIST = IntList()

@click.command()
@click.argument('number_samples', type=int, metavar='<number_samples>')
@click.option('-o', '--out', 'out_file',
    type=click.Path(writable=True, resolve_path=True, path_type=str),
    help='File where to write the simulation results to. If not given, the '
         'results will be written to standard output.')
@click.option('--lengths', type=INT_LIST, required=True,
    help='Comma-separated list of integers. These are different L\'s for which '
         'the simulation will be run. NOTE: the list should not contain spaces.')
@click.option('--append', type=bool, default=True, show_default=True,
    help='Whether to append the results to the output file. If False, the '
         'output file will be truncated first.')
def main(number_samples, lengths, out_file, append):
    """
    Runs the percolation simulation for a square lattice.
    """
    if out_file is not None and not append:
        with open(out_file, 'w'):
            pass
    percolation.sample(number_samples, lengths, out_file)

if __name__ == '__main__':
    main()

