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
from ..hilbertcurve import HilbertCurve

import math
import scipy.linalg


class Multiplication(Dataset):
    def do_init(self,
                skewed=False,
                log_sigma_interval=(-1.5, 0.5),
                biased=False):
        # Copy the "biased" flag. If "biased" is true, generate hilbert curve
        # through (polar) 2D space that we will use for sampling.
        self.biased = biased

        # Pre-compute a Hilbert curve when doing biased sampling
        if self.biased:
            # Generate some hilbert curve points in polar space
            self.hilbert_curve_pnts, self.hilbert_curve_scale = \
                generate_hilbert_curve_points(2, 12, ((
                (-np.pi, 0.0),
                ( np.pi, 1.0)
            )))

            # Current position along the Hilbert curve
            self.hilbert_curve_offs = 0

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

    def _do_sample_unbiased(self, n_smpls):
        phis = self.rng.uniform(-np.pi, np.pi, n_smpls)
        rs = self.rng.uniform(0, 1, n_smpls)
        return phis, rs

    def _do_sample_biased(self, n_smpls):
        # Create the output array and a write pointer pointing at the current
        # output entry
        xs = np.zeros((n_smpls, 2))
        ptr, n_pnts = 0, self.hilbert_curve_pnts.shape[0]
        offs = self.hilbert_curve_offs

        # Fill the output array
        while ptr < n_smpls:
            # Compute the number of samples to copy, then perform the copy
            n_cpy = min(n_pnts - offs, n_smpls - ptr)
            xs[ptr:(ptr + n_cpy)] = self.hilbert_curve_pnts[offs:(offs +
                                                                  n_cpy)]

            # Add some uniform noise to the Hilbert curve points
            noise = self.rng.uniform(-1, 1, (n_cpy, 2))
            noise *= self.hilbert_curve_scale[None, :]  # Correct scale
            xs[ptr:(ptr + n_cpy)] += noise

            # Advance the target pointer and the source offset
            ptr, offs = ptr + n_cpy, (offs + n_cpy) % n_pnts

        # Remember the current offset
        self.hilbert_curve_offs = offs

        return xs.T

    def do_sample(self, n_smpls, mode=None):
        if (not self.biased) or (mode != "training"):
            phis, rs = self._do_sample_unbiased(n_smpls)
        else:
            phis, rs = self._do_sample_biased(n_smpls)

        xs = np.array((rs * np.cos(phis), rs * np.sin(phis))).T
        ys = xs[:, 0] * xs[:, 1]
        return xs @ self.sigma.T, ys


def select_multiplication_biased(task_descr):
    """
    This function selects the "biased" parameter based on the given trial
    descriptor. In particular, if the trial descriptor indicates that learning
    is supposed to be "online", then we'll switch to a "biased" dataset.
    """
    return task_descr.sequential


manifest = DatasetManifest(name="multiplication",
                           ctor=Multiplication,
                           params={
                               "skewed": (False, True),
                               "biased": select_multiplication_biased,
                           })

