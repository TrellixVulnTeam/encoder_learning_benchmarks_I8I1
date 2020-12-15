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
from ..positive_definite_matrix import PositiveDefiniteMatrix


class PCA(EncoderLearningRule):
    def do_init(self, lambda_=10.0, scale=6.0, dt=1e-3):
        """
        lambda_: Time constant of the sliding window for covariance estimation
                 in seconds.
        dt:      Conversion factor between samples and time-steps.
        """

        # Combine lambda and dt into a single parameter used for computing the
        # update.
        self._beta = np.exp(dt / lambda_)
        self._scale = scale

        # Initial network data. This dictionary will be populated in the first
        # iteration of do_step.
        self._initial_net = None

        # Current timestep
        self._k = 0

        # Positive definite matrix instance used to convert between compressed
        # covariance matrix parameters theta and covariance matrices.
        self._pdm = PositiveDefiniteMatrix(self.n_dim_in)

        # Initial covariance estimate
        self._mu = np.zeros(self.n_dim_in)
        self._cov = np.zeros((self.n_dim_in, self.n_dim_in))
        #self._cov = np.eye(self.n_dim_in)


    def do_step(self, _, xs, __, ___, ____, net):
        # If this is the first timestep, copy the current RBF centres and
        # covariances
        if self._k == 0:
            self._initial_net = {
                "mus": np.copy(net.mus),
                "thetas": np.copy(net.thetas),
            }

        # Fetch the number of samples. Then iterate over each sample to update
        # the covariance estimate.
        N, beta = xs.shape[0], self._beta
        mu_next = np.zeros(self.n_dim_in)
        cov_next = np.zeros((self.n_dim_in, self.n_dim_in))
        for i in range(N):
            # Update the timestep and compute the current windowing factor
            self._k += 1
            alpha = (beta - 1.0) / (beta - np.power(beta, 1.0 - self._k))

            # Compute the updated mean and covariance estimates
            mu_next[...] = (1.0 - alpha) * self._mu + alpha * xs[i]
            cov_next[...] = (1.0 - alpha) * self._cov + alpha * np.outer(
                xs[i] - mu_next, xs[i] - self._mu)

            # Make the updated mean and covariance estimate the current mean and
            # covariance update
            self._mu[...] = mu_next
            self._cov[...] = cov_next

        # Compute the eigenvectors and eigenvalues of the covariance matrix.
        # We're adding a small identity matrix to ensure numerical stability and
        # that all eigenvalues are strictly positive.
        Λ, V = np.linalg.eigh(self._cov + 1e-6 * np.eye(self.n_dim_in))
        P = np.diag(np.sqrt(self._scale * Λ)) @ V

        # Compute the new RBF centres and covariance matrices by projecting them
        # onto the eigen-space of the covariance matrix.
        mus0, thetas0 = self._initial_net["mus"], self._initial_net["thetas"]
        covs0 = self._pdm.inverse(thetas0)
        mus_next = mus0 @ P + self._mu[None, :]
        covs_next = np.einsum('ki,...kj,jl', P, covs0, P)
        thetas_next = self._pdm.params_from_cov(np.array([np.linalg.inv(C) for C in covs_next]))

        return {
            "mus": net.mus - mus_next,
            "thetas": net.thetas - thetas_next,
        }

manifest = EncoderLearningRuleManifest(name="pca",
                                       ctor=PCA,
                                       supported_network_classes={"RBF"},
                                       is_supervised=False)

