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
from ..networks.perceptron import Perceptron


class Voja(EncoderLearningRule):
    def do_init(self, kappa=1e-4, learn_gains=False):
        self.kappa = kappa
        self.learn_gains = learn_gains

    def do_step(self, As, xs, _, __, net):
        # Initialize the result array
        dparams = {}

        # Fetch the encoders and gains
        E, α = net.params["encoders"], net.params["gains"]

        # Compute the encoder delta (line commented out is the unoptimized
        # version)
#        dE = np.mean(As[:, :, None] * (xs[:, None, :] - E[None, :, :]), axis=0)
        n_smpls = xs.shape[0]
        dE = (As.T @ xs) / n_smpls - np.mean(As, axis=0)[:, None] * E

        dparams["encoders"] = -self.kappa * dE

        if self.learn_gains:
            dα = np.linalg.norm(α[:, None] * E + self.kappa * dE, axis=1) - α
            dparams["gains"] = -dα

        return dparams


manifest = EncoderLearningRuleManifest(name="voja",
                                       ctor=Voja,
                                       supported_network_classes={
                                           Perceptron,
                                       },
                                       is_supervised=False)

