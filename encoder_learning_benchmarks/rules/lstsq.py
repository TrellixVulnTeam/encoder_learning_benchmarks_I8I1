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


class LSTSQ(DecoderLearningRule):
    def do_init(self, rcond=1e-2):
        self.rcond = rcond
        return DecoderLearningRuleDescriptor(returns_gradient=False)

    def do_step(self, As, ys, _, D):
        return np.linalg.lstsq(As, ys, rcond=self.rcond)[0].T


manifest = DecoderLearningRuleManifest(name="lstsq", ctor=LSTSQ)

