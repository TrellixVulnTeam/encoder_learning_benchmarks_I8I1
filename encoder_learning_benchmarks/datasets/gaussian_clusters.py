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
from ..hilbertcurve import HilbertCurve

import math
import scipy.linalg
import scipy.stats


class GaussianClusters(ClassificationDataset):
    """
    The CurveManifold class creates a random 1D curves within 2D space. The
    generated observation is moving along the curve according to a low-pass
    filtered Gaussian noise signal.
    """
    def do_init(self,
                n_dim=2,
                n_classes=5,
                mu_interval=(-10, 10),
                log_sigma_interval=(-1.5, 0.5),
                biased=False):
        # Copy the given arguments
        assert n_dim > 0
        assert n_classes > 0
        self.n_dim = n_dim
        self.n_classes = n_classes
        self.biased = biased

        # Select some random weights for the individual clusters
        self.ps = self.rng.uniform(0, 1, n_classes)
        self.ps = self.ps / np.sum(self.ps)

        # For each class create a mean and a random covariance matrix
        mu_interval = np.asarray(mu_interval) * n_classes / np.power(
            2.0, n_dim)
        self.mus = self.rng.uniform(*mu_interval, (n_classes, n_dim))
        self.sigmas = np.zeros((n_classes, n_dim, n_dim))
        Qs = np.zeros((n_classes, n_dim, n_dim))
        scales = np.zeros((n_classes, n_dim))
        for i in range(n_classes):
            A = self.rng.randn(n_dim, n_dim)
            Qs[i], _ = scipy.linalg.qr(A)
            scales[i] = np.power(10.0,
                                 self.rng.uniform(*log_sigma_interval, n_dim))
            self.sigmas[i] = Qs[i] @ np.square(np.diag(scales[i])) @ Qs[i].T

        # Pre-compute a Hilbert curve when doing biased sampling
        if self.biased:
            # Determine the order of the hilbert curve. We would like about
            # 100,000 points to sample from, since this is the epoch size.
            # The number of points on the Hilbert curve is given as n = 2**(p*N), 
            # where p is the order and N is the number of dimensions.
            n_pnts_tar = 131072  # (2**17)
            p = int(math.ceil(math.log2(n_pnts_tar) / n_dim))
            self.hilbert_curve_order = p
            n_pnts = 2**(n_dim * p)

            # Create the hilbert curve object and sample all points
            hc = HilbertCurve(p, n_dim)
            self.hilbert_curve_pnts = np.zeros((n_pnts, n_dim))
            for i in range(n_pnts):
                self.hilbert_curve_pnts[i] = hc.coordinates_from_distance(i)

            # Compute the axis-aligned bounding-box (AABB) to which we should scale
            # the hilbert curve
            p_min = 1e-4
            ext = scipy.stats.norm.ppf(1.0 - p_min)
            mins = np.ones(n_dim) * np.inf
            maxs = -np.ones(n_dim) * np.inf
            for i in range(n_classes):
                for j in range(n_dim):
                    for sgn in [-1, 1]:
                        vec = sgn * ext * scales[i] * Qs[i] @ np.eye(n_dim)[j]
                        mins = np.minimum(mins, vec + self.mus[i])
                        maxs = np.maximum(maxs, vec + self.mus[i])
            self.aabb = np.array((mins, maxs))

            # Scale the hilbert curve to fit into the AABB
            scale = ((self.aabb[1] - self.aabb[0]))
            offs = self.aabb[0]
            self.hilbert_curve_pnts = self.hilbert_curve_pnts / ((2**p) - 1)
            self.hilbert_curve_pnts *= scale[None, :]
            self.hilbert_curve_pnts += offs[None, :]

            # Remember the scaling factors
            self.hilbert_curve_scale = np.ones(self.n_dim) / ((2**p) - 1)
            self.hilbert_curve_scale *= scale

            # Current position along the Hilbert curve
            self.hilbert_curve_offs = 0

        # Rember some internal values for debugging
        self.Qs = Qs
        self.scales = scales

        return DatasetDescriptor(n_dim_in=n_dim, n_dim_out=n_classes, is_time_continous=self.biased)

    def pdf(self, pnts):
        # Make sure the given array has the right shape
        pnts = np.asarray(pnts)
        if pnts.ndim != 2:
            pnts = pnts.reshape(-1, self.n_dim)
        n_smpls = pnts.shape[0]
        assert pnts.shape[1] == self.n_dim

        P = np.zeros((n_smpls, self.n_classes))
        for i in range(self.n_classes):
            X = pnts - self.mus[i]
            ΣI = np.linalg.inv(self.sigmas[i])
            exponent = np.einsum('...i,ij,...j', X, ΣI, X)
            norm = np.sqrt(
                np.power(2.0 * np.pi, self.n_dim) *
                np.linalg.det(self.sigmas[i]))
            P[:, i] = self.ps[i] * np.exp(-0.5 * exponent) / norm

        return P

    def _do_sample_unbiased(self, n_smpls):
        # Select the gaussians from which we are sampling
        idcs = self.rng.choice(np.arange(self.n_classes), n_smpls, p=self.ps)

        # Individually sample from the Gaussians
        # TODO: Speed this up dramatically! Only call the random number
        # generator once and rotate/offset/skew the points manually
        xs = np.zeros((n_smpls, self.n_dim))
        ys = np.zeros((n_smpls, self.n_classes), dtype=np.bool)
        for i in range(n_smpls):
            cls = idcs[i]
            xs[i] = self.rng.multivariate_normal(mean=self.mus[cls],
                                                 cov=self.sigmas[cls])
            ys[i, cls] = True

        return xs, ys

    def _do_sample_biased(self, n_smpls):
        # In biased mode, we're moving along the space-filling hilbert curve
        # that was pre-computed in the constructor. For each point on the
        # hilbert curve we compute the probability of a sample emitted here.
        # We then use that probability to decide whether to actually sample a
        # point at that location or not. In this way the points are sampled
        # with a certain "order" and as such reflect online learning
        # scenarios in a better way.

        # Create the output array and a write pointer pointing at the current
        # output entry
        xs = np.zeros((n_smpls, self.n_dim))
        ys = np.zeros((n_smpls, self.n_classes), np.bool)
        ptr = 0

        # Shift the hilbert curve according to the last offset
        n_pnts = self.hilbert_curve_pnts.shape[0]
        shift = self.rng.randint(n_pnts)
        pnts = np.roll(self.hilbert_curve_pnts, -self.hilbert_curve_offs, axis=0)
        last_idx = 0

        while ptr < n_smpls:
            # Add some uniform noise to the Hilbert curve points
            noise = self.rng.uniform(-1, 1, (n_pnts, self.n_dim))
            noise *= self.hilbert_curve_scale[None, :] # Correct scale
            pnts += noise

            # Sample the PDFs at each of the points
            P = self.pdf(pnts)
            pdf = np.sum(P, axis=1) # sum accross all classes

            # Randomly select the points based on the pdf
            ps = self.rng.uniform(0, np.max(pdf), n_pnts)
            sel = pdf >= ps
            n_sel_total = np.sum(sel)
            n_sel = min(n_smpls - ptr, n_sel_total)
            xs[ptr:(ptr+n_sel)] = pnts[sel][:n_sel]

            # Randomly assign a class to each of the points. We're using the
            # "Gumbel max" trick here. See
            # https://timvieira.github.io/blog/post/2019/09/16/algorithms-for-sampling-without-replacement/
            P = P[sel][:n_sel]
            cls = np.argmax(np.log(P + 1e-16) + self.rng.gumbel(0, 1, P.shape), axis=1)
            ys[np.arange(ptr, ptr+n_sel), cls] = True

            # Remember the index of the last selected point
            sel_cumsum = np.cumsum(np.asarray(sel, np.int))
            last_idx = np.argmax(sel_cumsum == n_sel)

            # Advance the pointer
            ptr = ptr + n_sel

        # Adjust the hilbert curve offset. This ensures that we continue
        # sampling where we left off.
        self.hilbert_curve_offs = (self.hilbert_curve_offs + last_idx) % n_pnts

        return xs, ys

    def do_sample(self, n_smpls, mode):
        if (not self.biased) or (mode != "training"):
            return self._do_sample_unbiased(n_smpls)
        else:
            return self._do_sample_biased(n_smpls)

def select_gaussian_clusters_biased(task_descr):
    """
    This function selects the "biased" parameter based on the given trial
    descriptor. In particular, if the trial descriptor indicates that learning
    is supposed to be "online", then we'll switch to a "biased" dataset.
    """
    return task_descr.sequential

manifest = DatasetManifest(name="gaussian_clusters",
                           ctor=GaussianClusters,
                           params={
                            "n_classes": [2, 4, 8],
                            "n_dim": [2,],
                            "biased": select_gaussian_clusters_biased})

