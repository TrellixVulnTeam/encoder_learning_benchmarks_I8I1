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

class SGD(Optimizer):
    def do_init(self, eta=1e-4):
        assert eta > 0.0
        self.eta = eta

    def do_step(self, p, dp):
        for key in dp.keys():
            p[key][...] = p[key] - self.eta * dp[key]

def sgd_select_eta(trial_descr):
    """
    A specific learning rate can be selected here, depending on the trial
    descriptor. We need to do this because the SGD rule is very sensitive to
    factors such as the dataset size.
    """
    return 1e-4

manifest = OptimizerManifest(name="sgd", ctor=SGD, params={
    "eta": sgd_select_eta
})
