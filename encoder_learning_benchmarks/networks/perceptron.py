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


class ReLU:
    @staticmethod
    def inverse(a):
        return a

    @staticmethod
    def activity(x):
        return np.maximum(0, x)

    @staticmethod
    def derivative(x):
        return 1.0 * (x > 0.0)


class LIF:
    slope = 2.0 / 3.0

    @staticmethod
    def inverse(a):
        valid = a > 0
        return 1.0 / (1.0 - np.exp(LIF.slope - (1.0 / (valid * a + 1e-6))))

    @staticmethod
    def activity(x):
        valid = x > (1.0 + 1e-6)
        return valid / (LIF.slope - np.log(1.0 - valid * (1.0 / x)))

    @staticmethod
    def derivative(x):
        a = LIF.activity(x)
        valid = x > (1.0 + 1e-6)
        return (a * a) / ((~valid) + valid * (x * x * (1.0 - 1.0 / x)))


class Perceptron(Network):
    def do_init(
        self,
        initialisation="nef",
        neuron_type=ReLU,
        radius=1.0,
    ):
        assert initialisation in {"normal", "uniform", "nef"}

        # Some handy aliases
        rng = self.rng
        n, d = self.n_dim_hidden, self.n_dim_in

        # Copy the given arguments
        self.initialisation = initialisation
        self.neuron_type = neuron_type
        self.radius = radius

        # Create the initial matrices
        self.gains = np.zeros((n, ))
        self.biases = np.zeros((n, ))
        self.encoders = np.zeros((n, d))

        # Initialize the parameter matrices according to the chosen
        # initialisation method
        if (initialisation == "normal") or (initialisation == "uniform"):
            # Select the encoders and biases randomly according to the chosen
            # random function
            if initialisation == "normal":
                rnd = lambda shape: rng.normal(0, 1, shape)
            else:
                rnd = lambda shape: rng.uniform(-1, 1, shape)
            self.encoders = rnd((n, d))
            self.biases = rnd(n)

            # Set the gains to the length of the encoding vector
            self.gains = np.linalg.norm(self.encoders, axis=1)

            # Normalise the encoders to unit length
            self.encoders /= self.gains[:, None]
        elif initialisation == "nef":
            # Initialize the matrices depending on the initialisation method
            α, β = 0.5, 0.5 * (d + 1)
            intercepts = rng.choice([-1, 1], n) * np.sqrt(rng.beta(α, β, n))
            max_rates = rng.uniform(0.5, 1.0, n)

            # Compute the current causing the maximum rate/the intercept
            J_0 = neuron_type.inverse(0)
            J_max_rates = neuron_type.inverse(max_rates)

            # Compute the gain and bias
            self.gains = (J_0 - J_max_rates) / (radius * (intercepts - 1.0))
            self.biases = J_max_rates - radius * self.gains

            # Randomly select unit-length encoders
            self.encoders = rng.normal(0, 1, (n, d))
            self.encoders /= np.linalg.norm(self.encoders, axis=1)[:, None]

        # Create and return a map containing all the trainable parameters
        return {
            "gains": self.gains,
            "biases": self.biases,
            "encoders": self.encoders,
        }

    def J(self, xs):
        return (xs @ self.encoders.T) * self.gains + self.biases

    def normalize_params(self):
        self.encoders /= np.linalg.norm(self.encoders, axis=1)[:, None]

    def do_activities(self, xs):
        return self.neuron_type.activity(self.J(xs))

    def do_jacobian(self, xs):
        # Compute the derivative of the neural non-linearity for the given xs
        E, J, gains = self.encoders, self.J(xs), self.gains
        a, da = self.neuron_type.activity(J), self.neuron_type.derivative(J)

        # Compute and return the individual parameter gradients
        dgains = da * (xs @ E.T)
        dbiases = da
        dencoders = ((gains[None, :] * da)[:, :, None] *
                     (xs[:, None, :] - (E @ xs.T).T[:, :, None] * E))
        return {"gains": dgains, "biases": dbiases, "encoders": dencoders}


manifest = NetworkManifest(name="perceptron",
                           ctor=Perceptron,
                           params={
#                               "initialisation": ("normal", "nef",),
                               "initialisation": ("nef",),
                               "neuron_type": {
                                    "relu": ReLU
                               },
                           })

