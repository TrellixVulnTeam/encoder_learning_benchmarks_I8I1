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


class Backprop(EncoderLearningRule):
    def do_init(self):
        pass

    def do_step(self, As, xs, errs, D, Jpt, net):
        # Compute the network Jacobian
        jacobian = net.jacobian(xs)

        # Back-propagation step. Compute the influence of each unit on the
        # error.
        if Jpt is None:
            local_err = errs @ D
        else:
            local_err = np.einsum('...ij,jk,...kk->...k', errs, D, Jpt)

        # Compute the parameter update
        dparams = {}
        for key, J in jacobian.items():
            # Scale the back-propagated error by the gradient dA
            dparams[key] = np.mean((J.T * local_err.T).T, axis=0)

        return dparams

manifest = EncoderLearningRuleManifest(name="backprop", ctor=Backprop)

