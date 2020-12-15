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

import numpy as np


def plot_gaussian(ax,
                  mu,
                  cov=None,
                  theta=None,
                  inverse_cov=False,
                  scale=1.0,
                  plot_cross=True,
                  **kwargs):
    import matplotlib.patches
    from .positive_definite_matrix import PositiveDefiniteMatrix

    assert (cov is None) != (theta is None)
    if cov is None:
        cov = PositiveDefiniteMatrix(len(mu), linlog=False)(theta)[0]
    Λ, V = np.linalg.eigh(cov)
    if inverse_cov:
        Λ = 1.0 / Λ
    σ0 = np.sqrt(Λ[0]) * scale
    σ1 = np.sqrt(Λ[1]) * scale
    angle_rad = np.arctan2(V[1, 0], V[0, 0])
    angle = angle_rad / np.pi * 180.0

    ellipse = matplotlib.patches.Ellipse(np.copy(mu),
                                         2.0 * σ0,
                                         2.0 * σ1,
                                         angle,
                                         fill=False,
                                         **kwargs)
    ax.add_patch(ellipse)
    if "linewidth" in kwargs:
        kwargs["linewidth"] *= 0.5
    else:
        kwargs["linewidth"] = 0.75
    if plot_cross:
        ax.plot(
            [mu[0] - σ0 * np.cos(angle_rad), mu[0] + σ0 * np.cos(angle_rad)],
            [mu[1] - σ0 * np.sin(angle_rad), mu[1] + σ0 * np.sin(angle_rad)],
            **kwargs)
        ax.plot(
            [mu[0] + σ1 * np.sin(angle_rad), mu[0] - σ1 * np.sin(angle_rad)],
            [mu[1] - σ1 * np.cos(angle_rad), mu[1] + σ1 * np.cos(angle_rad)],
            **kwargs)


def _visualise_rbf_network(ax, net, activities, cov_scale=0.25):
    # Draw each unit as an ellipse. Make the line thickness and color
    # proportional to the activity of the unit
    for i in range(net.n_dim_hidden):
        plot_gaussian(ax,
                      mu=net.mus[i],
                      theta=net.thetas[i],
                      inverse_cov=True,
                      scale=cov_scale,
                      linewidth=0.5 + 2.0 * activities[i],
                      color=np.clip(np.ones(3) * (0.5 - 2.0 * activities[i]), 0, 1))


def _visualise_perceptron(ax, net, activities):
    pass


def visualise_network(ax, net, activities=None, cov_scale=0.25):
    # Make sure the given activities have the right shape
    if activities is None:
        activities = 0.25 * np.ones(net.n_dim_hidden)
    assert (activities.ndim == 1) and (activities.size == net.n_dim_hidden)

    if net.__class__.__name__ == "RBF":
        _visualise_rbf_network(ax, net, activities, cov_scale=cov_scale)
    elif net.__clas__.__name__ == "Perceptron":
        _visualise_perceptron(ax, net, activities)

