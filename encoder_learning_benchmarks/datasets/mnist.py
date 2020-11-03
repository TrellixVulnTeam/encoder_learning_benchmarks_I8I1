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

import gzip
import os

from ..common import *


def read_idxgz(filename):
    with gzip.open(filename, mode="rb") as f:
        # Read the header
        z0, z1, dtype, ndim = f.read(4)
        assert z0 == 0 and z1 == 0 and dtype == 0x08 and ndim > 0

        dims = []
        for i in range(ndim):
            nit0, nit1, nit2, nit3 = f.read(4)
            dims.append(nit3 | (nit2 << 8) | (nit1 << 16) | (nit0 << 24))

        # Read the remaining data
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(*dims)


_MNIST = {}


class MNIST(ClassificationDataset):
    def do_init(self):
        def mkpath(filename):
            return os.path.join(os.path.dirname(__file__), "data", filename)

        def rescale(xs):
            return 2.0 * np.asarray(xs, dtype=np.float) / 255.0 - 1.0

        # Read the MNIST dataset -- once
        if len(_MNIST) == 0:
            _MNIST["imgs"] = rescale(
                read_idxgz(mkpath("train-images-idx3-ubyte.gz")))
            _MNIST["lbls"] = read_idxgz(mkpath("train-labels-idx1-ubyte.gz"))

            _MNIST["imgs_test"] = rescale(
                read_idxgz(mkpath("t10k-images-idx3-ubyte.gz")))
            _MNIST["lbls_test"] = read_idxgz(
                mkpath("t10k-labels-idx1-ubyte.gz"))

        # Some aliases for the individual arrays
        imgs, lbls, imgs_test, lbls_test = (_MNIST["imgs"], _MNIST["lbls"],
                                            _MNIST["imgs_test"],
                                            _MNIST["lbls_test"])
        self._imgs, self._lbls, self._imgs_test, self._lbls_test = (imgs, lbls,
                                                                    imgs_test,
                                                                    lbls_test)

        # Some sanity checks
        assert imgs.ndim == 3 and imgs_test.ndim == 3
        assert lbls.ndim == 1 and lbls_test.ndim == 1
        assert imgs.shape[0] == lbls.shape[0]
        assert imgs_test.shape[0] == lbls_test.shape[0]
        assert imgs.shape[1] == imgs_test.shape[1]
        assert imgs.shape[2] == imgs_test.shape[2]
        assert np.min(imgs) == -1.0 and np.max(imgs) == 1.0
        assert np.min(imgs_test) == -1.0 and np.max(imgs_test) == 1.0
        assert np.min(lbls) == 0 and np.max(lbls) == 9
        assert np.min(lbls_test) == 0 and np.max(lbls_test) == 9

        # Determine the number of training, validation, and test samples. Note
        # that we arbitarily declare the last 10% of the samples to be our
        # validation samples.
        n_dim_in = imgs.shape[1] * imgs.shape[2]
        n_training_total = imgs.shape[0]
        n_validation = n_training_total // 10
        n_training = n_training_total - n_validation
        n_test = imgs_test.shape[0]

        # Return the dataset descriptor
        return DatasetDescriptor(n_dim_in=n_dim_in,
                                 n_dim_out=10,
                                 n_max_smpls_training=n_training,
                                 n_max_smpls_validation=n_validation,
                                 n_max_smpls_test=n_test,
                                 is_finite=True)

    def do_sample(self, n_smpls, mode):
        # Depending on the mode, select the right dataset
        if mode == "training":
            xs = self._imgs[:self.n_max_smpls_training]
            ys = self._lbls[:self.n_max_smpls_training]
        elif mode == "validation":
            xs = self._imgs[self.n_max_smpls_training:]
            ys = self._lbls[self.n_max_smpls_training:]
        elif mode == "test":
            xs = self._imgs_test
            ys = self._lbls_test

        # Make sure we're not querying too many samples
        assert n_smpls <= xs.shape[0]

        # Reshape the images into the correct vector form and construct the
        # one-hot-coded target vector
        xs_res = xs[:n_smpls].reshape(n_smpls, self.n_dim_in)
        ys_res = np.zeros((n_smpls, 10), dtype=np.bool)
        ys_res[np.arange(n_smpls), ys[:n_smpls]] = True

        return xs_res, ys_res


manifest = DatasetManifest(name="mnist", ctor=MNIST)

