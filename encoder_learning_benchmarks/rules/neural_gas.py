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


class NeuralGas(EncoderLearningRule):
    def do_init(self,
                lambda_=1.0,
                batch=True,
                cov_scale=16.0,
                tau=500e-3,
                dt=1e-3,
                A_tar=1.0):
        """
        lambda_:   Neighbourhood coefficient
        batch:     If True, use Batch Neural Gas in the update step. If False,
                   the standard online algorithm is used.
        cov_scale: Factor by which the covariance matrices are supposed to
                   be scaled.
        tau:       Filter for the target neuron activity (online only).
        dt:        Time passing for each sample (used for the low-pass filter).
        A_tar:     Target RBF activity (online only).
        """

        # Copy the given parameters
        self._lambda = lambda_
        self._batch = batch
        self._pdm = PositiveDefiniteMatrix(self.n_dim_in)
        self._cov_scale = cov_scale
        self._tau = tau
        self._dt = dt
        self._A_tar = A_tar / self.n_dim_hidden

        # Create the internal state matrices
        self._cov_est = None
        self._As_flt = None

    def _do_step_common(self, As, xs, net):
        # Some useful aliases
        N, n, d = xs.shape[0], self.n_dim_hidden, self.n_dim_in
        mus, thetas = net.mus, net.thetas

        # Compute the distance from each prototype to each sample
        Δ = xs[:, None] - mus[None]
        D = np.einsum('...i,...i->...', Δ, Δ)

        # Compute the rank matrix
        def rank(arr):
            res = np.zeros(n)
            res[np.argsort(arr)] = np.arange(n)
            return res

        K = np.apply_along_axis(rank, axis=1, arr=D)

        # Compute the neighbourhood structure h_lambda(K)
        H = np.exp(-K / self._lambda)

        return N, n, d, mus, thetas, Δ, D, K, H

    def _do_step_batch(self, As, xs, net):
        # This is a slightly adapted version of the batch neural gas algorithm
        # described in
        #
        #   Cottrell, M., Hammer, B., Hasenfuß, A., & Villmann, T. (2006).
        #   Batch and median neural gas. Neural Networks, 19(6), 762–771.
        #   https://doi.org/10.1016/j.neunet.2006.05.018
        #
        # In addition to updating the means, the code here uses the same scheme
        # to update the RBF covariance according to the local neighbourhood.

        # Execute the common steps
        N, n, d, mus, thetas, Δ, D, K, H = self._do_step_common(As, xs, net)

        # Compute the normalisation factor (this is the main difference to the
        # online update step, where the normalisation factor is 1 / N).
        Hnorm = 1.0 / np.sum(H, axis=0)

        # Compute the update for the centres
        dmus = np.einsum('ji,jik,i->...ik', H, Δ, Hnorm)

        # Compute the optimal covariance matrices for these samples.
        Δ_zm = Δ - np.mean(Δ, axis=0)[None]  # Zero-mean deltas for each neuron
        covs = self._cov_scale * \
            np.einsum('ji,jik,jil,i->ikl', H, Δ_zm, Δ_zm, Hnorm)

        # Turn the covariance matrices into the log-Cholesky parametrisation,
        # use this to compute the updates
        thetas_next = np.zeros((n, (d * (d + 1)) // 2))
        for i in range(n):
            Ls = np.linalg.cholesky(np.linalg.inv(covs[i]))
            self._pdm._encode_params(thetas_next[i], Ls[self._pdm._idcs_T])
        dthetas = thetas_next - thetas

        # Return the RBF parameter updates
        return {
            "mus": -dmus,
            "thetas": -dthetas,
        }

    def _do_step_online(self, As, xs, net):
        # Execute the common steps
        N, n, d, mus, thetas, Δ, D, K, H = self._do_step_common(As, xs, net)

        # Create the internal state if it doesn't exist yet
        if self._As_flt is None:
            self._As_flt = np.ones(n) * self._A_tar
        if self._cov_est is None:
            self._cov_est = self._pdm(thetas)
            for i in range(n):
                self._cov_est[i] = np.linalg.inv(self._cov_est[i])

        # Compute the average update for the center, as well as the covariances
        dmus = np.einsum('ji,jik->ik', H, Δ) / N
        dcovs = np.einsum('ji,jik,jil->ikl', H, Δ, Δ) / N

        # Filter the activities using a low-pass filter
        self._As_flt += N * self._dt * (np.mean(As, axis=0) -
                                    self._As_flt / self._tau)

        # Accumulate the covariance updates; scale the covariances such that
        # a certain activitiy is reached
        scale = 1.0 + (self._A_tar - self._As_flt)
        self._cov_est += dcovs
        self._cov_est *= scale[:, None, None]

        # Turn the covariance matrices into the log-Cholesky parametrisation,
        # use this to compute the updates
        thetas_next = np.zeros((n, (d * (d + 1)) // 2))
        for i in range(n):
            try:
                Ls = np.linalg.cholesky(np.linalg.inv(self._cov_est[i]))
                self._pdm._encode_params(thetas_next[i], Ls[self._pdm._idcs_T])
            except np.linalg.LinAlgError:
                thetas_next[i] = thetas[i]
        dthetas = thetas_next - thetas

        # Compute and return the mean over all gradients
        return {
            "mus": -dmus,
            "thetas": -dthetas,
        }

    def do_step(self, As, xs, _, __, ___, net):
        if (not self._batch) or (xs.shape[0] <= 1):
            return self._do_step_online(As, xs, net)
        else:
            return self._do_step_batch(As, xs, net)


def neural_gas_validate_task(t):
    # Batch learning is only properly supported if the batch size is larger than
    # one. Otherwise the above implementation will fall back to the online
    # version anyways.
    if t.encoder_learner_params["batch"] and (t.batch_size <= 1):
        return False
    return True


manifest = EncoderLearningRuleManifest(name="neural_gas",
                                       ctor=NeuralGas,
                                       supported_network_classes={
                                           "RBF",
                                       },
                                       params={"batch": (True, False)},
                                       validate_task=neural_gas_validate_task,
                                       is_supervised=False)

