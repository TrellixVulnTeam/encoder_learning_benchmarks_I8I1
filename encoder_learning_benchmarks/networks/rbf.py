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
from ..halton import halton
from ..mahalanobis_distance import MahalanobisDistance
from ..positive_definite_matrix import PositiveDefiniteMatrix


class RBF(Network):
    def do_init(
        self,
        radius=1.0,
        initialisation="halton",
        learn_covariance=True,
    ):
        assert initialisation in {"halton", "uniform"}

        # Some handy aliases
        rng = self.rng
        n, d, r = self.n_dim_hidden, self.n_dim_in, radius

        # Create the Mahalanobis distance instance; fetch the number of
        # parameters for theta
        self._dist = MahalanobisDistance(d)
        self._n_dim_theta = self._dist.n_dim_theta

        # Copy the given arguments
        self.radius = r
        self.initialisation = initialisation
        self.learn_covariance = learn_covariance

        # Generate the centres mu and the parameters describing the covariance
        # matrix
        if (initialisation == "halton") and (d <= 50):
            self.mus = r * (2.0 * halton(n, d) - 1.0)
        else:
            self.mus = r * np.random.uniform(-1, 1, (n, d))

        # Randomly select the thetas. This is easiest in the computationally
        # more complex Givens representation, since we can just randomly select
        # angles and standard deviations.
        sigmas_inv_log10 = np.random.uniform(-1.0, 0.0, (n, d))
        sigmas_inv = np.power(n, 1.0 / d) / r * np.power(10.0, sigmas_inv_log10)
        angles = np.random.uniform(0.0, np.pi, (n, ((d - 1) * d) // 2))
        self.thetas = self._dist.pdm.params_from_givens(sigmas_inv, angles)

        # Create and return a map containing all the trainable parameters
        res = {}
        res["mus"] = self.mus
        if self.learn_covariance:
            res["thetas"] = self.thetas
        return res

    def do_activities(self, xs):
        return np.exp(-self._dist(xs, self.mus, self.thetas))

    def do_jacobian(self, xs):
        # Compute the activities
        A = self.do_activities(xs)

        # Compute the derivatives of the distance measure
        _, dmus, dthetas = self._dist.jacobian(xs, self.mus, self.thetas)

        # Compute the final derivative by applying the chain rule for exp(-d)
        res = {}
        res["mus"] = -dmus * A[:, :, None]
        if self.learn_covariance:
            res["thetas"] = -dthetas * A[:, :, None]
        return res

    @property
    def n_dim_theta(self):
        return self._n_dim_theta


manifest = NetworkManifest(name="rbf",
                           ctor=RBF,
                           params={
                               "initialisation": ("halton",),
                               "learn_covariance": (True,),
#                               "initialisation": ("halton", "uniform"),
#                               "learn_covariance": (True, False),
                           })
