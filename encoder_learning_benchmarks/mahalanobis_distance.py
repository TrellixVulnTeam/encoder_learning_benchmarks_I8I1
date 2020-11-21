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

from .positive_definite_matrix import PositiveDefiniteMatrix


class MahalanobisDistance:
    """
    Computes the squared MahalanobisDistance (x - y)^T Σ^-1 (x - y) and the
    corresponding partial derivatives for an unconstrained parameter set.
    """
    def __init__(self, k, linlog=False):
        # Copy the given parameter and construct the PositiveDefiniteMatrix
        # instance used internally.
        self._k = k
        self._pdm = PositiveDefiniteMatrix(k, linlog)

        # Fetch the column in the Cholesky matrix corresponding to each
        # parameter
        self._iks, self._jks = self._pdm._idcs

    def _condition_input(self, xs, ys, thetas):
        # Convert all parameters to numpy arrays
        xs = np.asarray(xs, dtype=np.float)
        ys = np.asarray(ys, dtype=np.float)
        thetas = np.asarray(thetas, dtype=np.float)

        # Make sure that all parameters are two-dimensional
        if xs.ndim != 2:
            xs = xs.reshape(1, -1)
        if ys.ndim != 2:
            ys = ys.reshape(1, -1)
        if thetas.ndim != 2:
            thetas = thetas.reshape(1, -1)

        # The dimensionality of each x and mu yst match the number of
        # dimensions this instance is defined over
        assert xs.shape[1] == ys.shape[1] == self.n_dim

        # The number of parameters describing the precision matrix yst match
        assert thetas.shape[1] == self.n_dim_theta

        # The number of individual ys and thetas yst be the same
        assert ys.shape[0] == thetas.shape[0]

        # Reshape xs and ys to facilitate the computation of the pairwise vector
        # differences through automatic broadcasting
        Ns = xs.shape[0]
        Nd = ys.shape[0]
        xs = xs.reshape(Ns, 1, self._k)
        ys = ys.reshape(1, Nd, self._k)

        # Return the matrices and all the dimensions and counts a caller could
        # possibly need
        return xs, ys, thetas, Ns, Nd, self.n_dim, self.n_dim_theta

    def __call__(self, xs, ys, thetas):
        # Make sure the input matrices have the right shape
        xs, ys, thetas, _, _, _, _ = self._condition_input(xs, ys, thetas)

        # Compute (x - y)^T Σ^-1 (x - y) while keeping the outermost dimensions
        # intact
        Δ, ΣInvs = xs - ys, self._pdm(thetas)
        return np.einsum('...i,...ij,...j->...', Δ, ΣInvs, Δ)

    def jacobian(self, xs, ys, thetas):
        # Make sure the input matrices have the right shape
        xs, ys, thetas, Ns, Nd, k, q = self._condition_input(xs, ys, thetas)

        # Compute the derivative with respect to xs or ys
        Δ, ΣInvs = xs - ys, self._pdm(thetas)
        dxs = 2.0 * np.einsum('...i,...ij', Δ, ΣInvs)
        dys = -dxs

        # Compute the derivative with respect to theta
        L = self._pdm.cholesky(thetas)
        thetap = np.zeros((Nd, q))
        self._pdm._linearise_params_derivative(thetap, thetas)
        LΔ = np.einsum('ijk,lik->lij', L[:, self._iks], Δ)
        dthetas = 2.0 * thetap * Δ[:, :, self._jks] * LΔ

        return dxs, dys, dthetas

    @property
    def pdm(self):
        return self._pdm

    @property
    def n_dim(self):
        return self._k

    @property
    def n_dim_theta(self):
        return self._pdm.n_params

