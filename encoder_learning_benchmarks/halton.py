#  Encoder Learning Benchmark
#  Copyright (C) 2020 Andreas Stöckel
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import math

# See the following paper for the choice of the permutation:
#
# Chi, H., Mascagni, M., & Warnock, T. (2005).
# On the optimal Halton sequence. Mathematics and Computers in Simulation,
# 70(1), 9–21. https://doi.org/10.1016/j.matcom.2005.03.004

def chi_et_al_permutation(i, pi, bj):
    w = [
        1, 2, 2, 5, 3, 7, 3, 10, 18, 11, 17, 5, 17, 26, 40, 14, 40, 44, 12, 31,
        45, 70, 8, 38, 82, 8, 12, 38, 47, 70, 29, 57, 97, 110, 32, 48, 84, 124,
        155, 26, 69, 83, 157, 171, 8, 32, 112, 205, 15, 31
    ]
    return (w[i] * bj) % pi

def primes():
    lst, i = [], 1
    while True:
        i += 1
        if not any(i % j == 0 for j in lst):
            lst.append(i)
            yield i

def van_der_corput(i, pi, n):
    # Represent n in the basis pi
    N = math.ceil(math.log(n + 1) / math.log(pi))
    bs = [((n // (pi**j)) % pi) for j in range(N)]

    # Permute the elements of bs
    bs = [chi_et_al_permutation(i, pi, bs[j]) for j in range(N)]

    # Compute the result
    return sum(bs[j] / (pi**(j + 1)) for j in range(N))

def halton(N, dims):
    return np.array([
        [van_der_corput(i, pi, n) for n in range(N)]
        for i, pi in zip(range(dims), primes())
    ]).T

