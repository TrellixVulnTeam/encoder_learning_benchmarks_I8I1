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

class Softmax(Network):
    def do_init(
        self,
        beta=10.0,
    ):
        # This is a pass-through network. The number of output dimensions must
        # be equal to the number of hidden dimensions
        assert self.n_dim_hidden == self.n_dim_in

        # Make sure that beta is larger than zero
        assert beta > 0.0
        self.beta = beta

        # We have no parameters that should be optimized
        return {}

    def do_activities(self, xs):
        # Shift all samples for numerical stability
        xs = self.beta * (xs - np.max(xs, axis=1)[:, None])

        # Compute the exp(xs), as well as the sum of the exponentials
        exs = np.exp(xs)
        sigmas = np.sum(exs, axis=1)

        # Compute the softmax by dividing 
        return exs / sigmas[:, None]

    def do_jacobian(self, xs):
        # Shift all samples for numerical stability
        xs = self.beta * (xs - np.max(xs, axis=1)[:, None])

        # Compute exp(xs), as well as the sum of the exponentials
        exs = np.exp(xs)
        sigmas = np.sum(exs, axis=1)

        # Compute the outer product of exs scaled by the square of the
        # exponential sum
        s = self.beta / np.square(sigmas)
        res = -(s[:, None] * exs)[:, :, None] * (exs[:, None, :])

        # Add beta / sigma * exs to the diagonals of the N individual matrices
        I = np.arange(self.n_dim_in)
        res[:, I, I] += exs * (self.beta / sigmas)[:, None]
#        res = np.zeros((xs.shape[0], self.n_dim_hidden, self.n_dim_in))
#        res[:, I, I] = 1.0

        return res

manifest = NetworkManifest(name="softmax",
                           ctor=Softmax,
                           passthrough=True)
