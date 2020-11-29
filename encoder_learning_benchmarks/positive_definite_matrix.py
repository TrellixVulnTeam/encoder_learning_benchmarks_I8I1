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
import scipy.linalg.lapack


class PositiveDefiniteMatrix:
    """
    Uses the log-Cholesky representation of a positive definite matrix.
    This is useful when representing covariance or precision matrices, such as
    those used in the RBF basis functions.

    See Pinheiro and Bates (1996), Unconstrained parametrizations for
    variance-covariance matrices, Statistics and Computing 6, 289-296.
    """
    def __init__(self, k, linlog=False):
        # Copy the given arguments
        self._k = k
        self._linlog = linlog

        # Compute the total number of parameters. All positive semidefinite
        # matrices can be represented using a riagonal matrix
        self._n_params = (k * (k + 1)) // 2

        # Compute indices pointing at the upper tridiagonal matrix
        def mkidcs(dim, first_off_diagonal=0):
            for i in range(first_off_diagonal, k):
                for j in range(k - i):
                    yield (j, i + j)[dim]

        self._idcs = (tuple(mkidcs(0)), tuple(mkidcs(1)))
        self._idcs_T = (tuple(mkidcs(1)), tuple(mkidcs(0)))

        # We'll also use the upper triagonal matrix indices and the lower
        # triagonal matrix indices
        self._idcs_upper = (tuple(mkidcs(0, 1)), tuple(mkidcs(1, 1)))
        self._idcs_lower = (tuple(mkidcs(1, 1)), tuple(mkidcs(0, 1)))

    def _condition_input(self, params):
        # Make sure the incoming array is 2d, with the number of samples in the
        # first dimension
        params = np.asarray(params)
        if params.ndim != 2:
            params = params.reshape(1, -1)
        assert params.shape[1] == self._n_params
        return params.shape[0], params, self._k, self._n_params

    def __call__(self, params):
        L = self.cholesky(params)
        return np.einsum('...ji,...jk->...ik', L, L)

    def _linearise_params(self, p_tar, p_src):
        k = self._k
        if self._linlog:
            pos = p_src[..., :k] > 0.0
            p_tar[..., :k][~pos] = np.exp(p_src[..., :k][~pos])
            p_tar[..., :k][pos] = p_src[..., :k][pos] + 1.0
            p_tar[..., k:] = p_src[..., k:]
        else:
            p_tar[..., :k] = np.exp(p_src[..., :k])
            p_tar[..., k:] = p_src[..., k:]

    def _encode_params(self, p_tar, p_src):
        k = self._k
        if self._linlog:
            gto = p_src[..., :k] > 1.0
            p_tar[..., :k][~gto] = np.log(p_src[..., :k][~gto])
            p_tar[..., :k][gto] = p_src[..., :k][gto] - 1.0
            p_tar[..., k:] = p_src[..., k:]
        else:
            p_tar[..., :k] = np.log(p_src[..., :k])
            p_tar[..., k:] = p_src[..., k:]

    def _linearise_params_derivative(self, p_tar, p_src):
        k = self._k
        if self._linlog:
            pos = p_src[..., :k] > 0.0
            p_tar[..., :k][~pos] = np.exp(p_src[..., :k][~pos])
            p_tar[..., :k][pos] = 1.0
            p_tar[..., k:] = 1.0
        else:
            p_tar[..., :k] = np.exp(p_src[..., :k])
            p_tar[..., k:] = 1.0

    def cholesky(self, params):
        # Condition the input parameters
        N, params, k, q = self._condition_input(params)

        # Create the temporary parameter vector and the result matrix
        p, L = np.zeros(q), np.zeros((N, k, k))
        for i in range(N):
            # Fill the parameter vector, the diagonal elements are stored as a
            # logarithm; this ensures that the entries are strictly positive.
            self._linearise_params(p, params[i])

            # Fill the upper tridiagonal
            L[i][self._idcs] = p

        return L

    def inverse(self, params):
        # Condition the input parameters
        N, params, k, q = self._condition_input(params)

        # Create temporary parameter vector and upper triagonal matrix
        p, L = np.zeros(q), np.zeros((k, k))

        # Compute the resulting matrices
        res = np.zeros((N, k, k))
        for i in range(N):
            # Fill the parameter vector, the diagonal elements are stored as a
            # logarithm; this ensures that the entries are strictly positive.
            self._linearise_params(p, params[i])

            # Fill the upper tridiagonal
            L[self._idcs] = p

            # Compute the inverse; we already have the output matrix in upper
            # triagonal form, so we can use a special function to compute the
            # inverse
            res[i], _ = scipy.linalg.lapack.dpotri(L)
            res[i][self._idcs_lower] = res[i][self._idcs_upper]

        return res

    def jacobian(self, params):
        # Condition the input parameters
        N, params, k, q = self._condition_input(params)

        # Create a temporary parameter vector and upper triagonal matrix
        p, dp, L = np.zeros(q), np.zeros(q), np.zeros((k, k))

        # Compute the resulting matrices
        res = np.zeros((N, k, k, q))
        for i in range(N):
            # Fill the parameter vector and the corresponding derivatives
            self._linearise_params(p, params[i])
            self._linearise_params_derivative(dp, params[i])

            # Fill the upper triangle
            L[self._idcs] = p

            # Compute the individual derivatives. We have
            #  d                  d                     d
            # -- L(x)^T L(x) = ( -- L(x)^T ) L(x) + (( -- L(x)^T ) L(x))^T
            # dx                 dx                    dx
            # Since the individual derivatives of d/dx L(x) only have one
            # non-zero entry, this just corresponds to scaling one row of
            # L and adding it to one row/column of the result matrix.
            for j in range(q):
                I0, I1 = self._idcs[0][j], self._idcs[1][j]
                res[i, I1, :, j] += dp[j] * L[I0, :]
                res[i, :, I1, j] += dp[j] * L[I0, :]

        return res

    def inverse_jacobian(self, params):
        # Condition the input parameters
        N, params, k, q = self._condition_input(params)

        # We'll need the inverses and the Jacobians
        JInv = self.inverse(params)
        dJ = self.jacobian(params)

        # See Helmut Lütkepohl (1996), Handbook of Matrices, Chp 10.9, p. 208
        res = np.zeros((N, k, k, q))
        for i in range(N):
            for j in range(q):
                res[i, :, :, j] = -JInv[i] @ dJ[i, :, :, j] @ JInv[i]
        return res

    def cov_from_givens(self, sigmas, angles):
        """
        Constructs a covariance matrix from the given standard deviations and
        rotation angles.
        """
        # Make sure all incoming matrices are ndarrays
        sigmas = np.asarray(sigmas, dtype=np.float)
        angles = np.asarray(angles, dtype=np.float)

        # We need a positive definite matrix!
        assert np.all(sigmas > 0.0)

        # Some handy aliases
        k, n_sigmas = self._k, self._k
        n_angles = ((self._k - 1) * self._k) // 2
        n_params = self.n_params

        # Condition the input parameters
        sigmas = np.asarray(sigmas, dtype=np.float)
        angles = np.asarray(angles, dtype=np.float)
        if sigmas.ndim != 2:
            sigmas = sigmas.reshape(-1, n_sigmas)
        if angles.ndim != 2:
            angles = angles.reshape(-1, n_angles)
        assert sigmas.shape[0] == angles.shape[0]

        # Make sure that the given matrices have the right shape
        N = sigmas.shape[0]
        assert sigmas.shape == (N, n_sigmas)
        assert angles.shape == (N, n_angles)

        def R(i, j, alpha):
            # Construct the Givens matrix Rij
            s, c = np.sin(alpha), np.cos(alpha)
            res = np.eye(k)
            res[i, i], res[j, i], res[i, j], res[j, j] = c, -s, s, c
            return res

        # Iterate over all samples
        res = np.zeros((N, k, k))
        for smpl in range(N):
            V = np.eye(k)
            idx = 0
            # Iterate over all Givens matrices and multiply them
            for i in range(k):
                for j in range(i + 1, k):
                    V = V @ R(i, j, angles[smpl, idx])
                    idx = idx + 1

            # Construct the diagonal matrix of Eigenvalues
            Λ = np.diag(np.square(sigmas[smpl]))

            # Construct the final covariance matrix
            res[smpl] = V @ Λ @ V.T
        return res

    def params_from_givens(self, sigmas, angles):
        """
        Constructs the parameters theta from the given standard deviations
        and rotation angles.
        """
        # Compute the covariance matrices
        covs = self.cov_from_givens(sigmas, angles)

        # Iterate over all samples
        res = np.zeros((N, n_params))
        for smpl in range(N):
            # Compute the Cholesky decomposition of the covariance matrix
            L = np.linalg.cholesky(covs[smpl])

            # Encode the parameter vector according to the encoding given
            # in the constructor.
            self._encode_params(res[smpl], L[self._idcs_T])

        return res

    @property
    def order(self):
        return self._k

    @property
    def n_params(self):
        return self._n_params

