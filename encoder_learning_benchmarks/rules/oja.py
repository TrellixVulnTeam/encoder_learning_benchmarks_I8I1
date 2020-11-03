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


class Oja(EncoderLearningRule):
    def do_init(self, kappa=1.0):
        self.kappa = kappa

    def do_step(self, As, xs, _, __, net):
        E = net.params["encoders"]
        return {
            "encoders":
            -self.kappa * np.mean(As[:, :, None] * (xs[:, None, :] - E[None, :, :]), axis=0)
        }


#manifest = EncoderLearningRuleManifest(name="oja",
#                                       ctor=Oja,
#                                       supported_network_classes={
#                                           Perceptron,
#                                       },
#                                       is_supervised=False)

