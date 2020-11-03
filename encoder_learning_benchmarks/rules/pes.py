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


class PES(DecoderLearningRule):
    def do_init(self, normalisation_mode=None):
        # Make sure that the normalisation mode has a supported value
        assert normalisation_mode in {None, "activity", "n_neurons"}
        self.normalisation_mode = normalisation_mode

        # Return the default decoder learning rule descriptor
        return DecoderLearningRuleDescriptor()

    def do_step(self, As, _, errs, __):
        # Fetch some information about the given dataset
        n_samples = As.shape[0]
        n_neurons = As.shape[1]

        # Compute the normalisation factor
        norm = 1.0 / n_samples
        if self.normalisation_mode == "n_neurons":
            # Divide by the number of neurons. This is what Nengo does
            # internally
            norm /= n_neurons
        elif self.normalisation_mode == "activity":
            # Compute average activity sum per sample. This is similar to
            # dividing by the number of neurons under the assumption that the
            # activity sum scales with the number of neurons. This is not true
            # for some of the sparsification techniques we are exploring here.
            norm /= max(1e-6, (np.sum(np.abs(As)) / n_samples))

        # Compute the update according to the PES learning rule
        return (errs.T @ As) * norm


manifest = DecoderLearningRuleManifest(
    name="pes",
    ctor=PES,
)

