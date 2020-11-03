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

from ..common import *

import scipy.linalg


class Multiplication(Dataset):
    def do_init(self, skewed=False, log_sigma_interval=(-1.5, 0.5)):
        # Compute a skew matrix if "skewed" is true
        if skewed:
            phi = self.rng.uniform(-np.pi, np.pi)
            Q = np.array(
                ((np.cos(phi), np.sin(phi)), (-np.sin(phi), np.cos(phi))))
            scales = np.power(10.0, self.rng.uniform(*log_sigma_interval, 2))
            self.sigma = Q @ np.diag(scales) @ Q.T
        else:
            self.sigma = np.eye(2)

        return DatasetDescriptor(n_dim_in=2, n_dim_out=1)

    def do_sample(self, n_smpls, mode=None):
        phis = self.rng.uniform(-np.pi, np.pi, n_smpls)
        rs = self.rng.uniform(0, 1, n_smpls)

        xs = np.array((rs * np.cos(phis), rs * np.sin(phis))).T
        ys = xs[:, 0] * xs[:, 1]
        return xs @ self.sigma.T, ys


manifest = DatasetManifest(name="multiplication",
                           ctor=Multiplication,
                           params={"skewed": (False, True)})

