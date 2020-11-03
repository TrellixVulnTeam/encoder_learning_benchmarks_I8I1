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


class FeedbackAlignment(EncoderLearningRule):
    def do_init(self):
        # Generate the random feedback-alignment matrix
        self.F = np.random.normal(0, 1, (self.n_dim_out, self.n_dim_hidden))
        self.F /= np.linalg.norm(self.F, axis=0)

    def do_step(self, As, xs, errs, _, net):
        # Compute the network Jacobian
        jacobian = net.jacobian(xs)

        # Feedback alignment step. Project the error backwards using a random
        # feedback alignment matrix.
        local_err = errs @ self.F

        # Compute the parameter update
        dparams = {}
        for key, J in jacobian.items():
            # Scale the back-propagated error by the gradient dA
            dparams[key] = np.mean((J.T * local_err.T).T, axis=0)

        return dparams

manifest = EncoderLearningRuleManifest(name="feedback_alignment", ctor=FeedbackAlignment)

