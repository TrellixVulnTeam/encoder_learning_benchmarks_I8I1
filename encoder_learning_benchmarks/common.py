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

import copy
import dataclasses
import math
import numpy as np
import scipy.signal
import scipy.linalg

from .hilbertcurve import HilbertCurve

###############################################################################
# Useful functions                                                            #
###############################################################################


def nts(T, dt=1e-3):
    return int(T / dt + 1e-9)


def mkrng(rng=np.random):
    """
    Derives a new random number generator from the given random number
    generator.
    """
    return np.random.RandomState(rng.randint(1 << 31))


###############################################################################
# Random Signal Generator                                                     #
###############################################################################


class FilteredGaussianSignal:
    """
    The FilteredGaussianSignal class generates a low-pass filtered white noise
    signal.
    """
    def __init__(self,
                 n_dim=1,
                 freq_low=None,
                 freq_high=1.0,
                 order=4,
                 dt=1e-3,
                 rng=np.random,
                 rms=0.5):
        assert (not freq_low is None) or (not freq_high is None)

        # Copy the given parameters
        self.n_dim = n_dim
        self.dt = dt
        self.rms = rms

        # Derive a new random number generator from the given rng. This ensures
        # that the signal will always be the same for a given random state,
        # independent of other
        self.rng = mkrng(rng)

        # Build the Butterworth filter
        if freq_low is None:
            btype = "lowpass"
            Wn = freq_high
        elif freq_high is None:
            btype = "highpass"
            Wn = freq_low
        else:
            btype = "bandpass"
            Wn = [freq_low, freq_high]
        self.b, self.a = scipy.signal.butter(N=order,
                                             Wn=Wn,
                                             btype=btype,
                                             analog=False,
                                             output='ba',
                                             fs=1.0 / dt)

        # Scale the output to reach the RMS
        self.b *= rms / np.sqrt(2.0 * dt * freq_high)

        # Initial state
        self.zi = np.zeros((max(len(self.a), len(self.b)) - 1, self.n_dim))

    def __call__(self, n_smpls):
        # Generate some random input
        xs = self.rng.randn(n_smpls, self.n_dim)

        # Filter each dimension independently, save the final state so multiple
        # calls to this function will create a seamless signal
        ys = np.empty((n_smpls, self.n_dim))
        for i in range(self.n_dim):
            ys[:, i], self.zi[:, i] = scipy.signal.lfilter(self.b,
                                                           self.a,
                                                           xs[:, i],
                                                           zi=self.zi[:, i])
        return ys


###############################################################################
# Hilbert Curve Points Generator                                              #
###############################################################################


def generate_hilbert_curve_points(n_dim=2, n_pnts_tar_log2=16, aabb=None):
    # Make sure the given arguments are sane
    assert n_dim > 0
    assert n_pnts_tar_log2 > 1

    # Make sure the AABB has the right shape
    if aabb is None:
        aabb = np.array((-np.ones(n_dim), np.ones(n_dim)))
    aabb = np.asarray(aabb)
    assert aabb.ndim == 2
    assert aabb.shape[0] == 2
    assert aabb.shape[1] == n_dim

    # Determine the order of the hilbert curve. We would like about
    # 100,000 points to sample from, since this is the epoch size.
    # The number of points on the Hilbert curve is given as n = 2**(p*N),
    # where p is the order and N is the number of dimensions.
    n_pnts_tar = 2**n_pnts_tar_log2
    p = int(math.ceil(math.log2(n_pnts_tar) / n_dim))
    n_pnts = 2**(n_dim * p)

    # Create the hilbert curve object and sample all points
    hc = HilbertCurve(p, n_dim)
    hilbert_curve_pnts = np.zeros((n_pnts, n_dim))
    for i in range(n_pnts):
        hilbert_curve_pnts[i] = hc.coordinates_from_distance(i)

    # Scale the hilbert curve to fit into the AABB
    range_ = ((aabb[1] - aabb[0]))
    offs = aabb[0]
    hilbert_curve_pnts = hilbert_curve_pnts / ((2**p) - 1)
    hilbert_curve_pnts *= range_[None, :]
    hilbert_curve_pnts += offs[None, :]

    # Remember the scaling factors
    hilbert_curve_scale = np.ones(n_dim) / ((2**p) - 1)
    hilbert_curve_scale *= range_

    return hilbert_curve_pnts, hilbert_curve_scale


###############################################################################
# Dataset Base Classes                                                        #
###############################################################################


@dataclasses.dataclass
class DatasetDescriptor:
    """
    Dataclass containing run-time information about a dataset.
    """
    n_dim_in: int = 1
    n_dim_out: int = 1
    n_max_smpls_training: int = -1
    n_max_smpls_validation: int = -1
    n_max_smpls_test: int = -1
    is_time_continous: bool = False
    is_finite: bool = False


class Dataset:
    def do_init(self):
        raise NotImplemented("do_init not implemented")

    def do_sample(self, n_smpls):
        raise NotImplemented("do_sample not implemented")

    def do_error(self, ys, ys_hat):
        return np.sqrt(np.mean(np.square(ys - ys_hat))) / self.rms()

    def __init__(self, dt=1e-3, rng=np.random, *args, **kwargs):
        """
        Initializes the environment.

        dt: is the timestep.
        rng: is the random number generator.
        """
        assert dt > 0.0

        # Copy the given arguments
        self._dt = dt
        self._rng = mkrng(rng)

        # Call the environment constructor, and check the returned environment
        # descriptor
        self._descr = self.do_init(*args, **kwargs)
        assert type(self._descr) is DatasetDescriptor
        if self.is_finite:
            assert ((self.n_max_smpls_training > 0)
                    and (self.n_max_smpls_validation > 0)
                    and (self.n_max_smpls_test > 0))
        else:
            assert ((self.n_max_smpls_training == -1)
                    and (self.n_max_smpls_validation == -1)
                    and (self.n_max_smpls_test == -1))

    def sample(self, n_smpls, mode="training"):
        """
        Returns n_smpls samples from the dataset. For time-continous datasets
        this 
        """

        # Make sure the number of samples is non-negative
        assert int(n_smpls) >= 0

        # Make sure the mode is either "training", "validation", or "test"
        assert mode in {"training", "validation", "test"}

        # If we're sampling the test data, make sure we're using the same RNG
        # every time. The try-finally block below ensures that we're always
        # resetting the rng, no matter what
        old_rng = self._rng
        if mode == "test":
            self._rng = np.random.RandomState(1232199609)
        try:
            # Call the actual implementation of step and destructure the return
            # value
            xs, ys = self.do_sample(int(n_smpls), mode)

            # Make sure the resulting arrays have the right dimensionality
            xs, ys = np.asarray(xs), np.asarray(ys)
            if xs.ndim != 2:
                xs = xs.reshape(n_smpls, -1)
            if ys.ndim != 2:
                ys = ys.reshape(n_smpls, -1)

            # Make sure the returned arrays have the right shape
            assert xs.shape == (n_smpls, self.n_dim_in)
            assert ys.shape == (n_smpls, self.n_dim_out)
        finally:
            self._rng = old_rng

        return xs, ys

    def rms(self, n_max_smpls=10000):
        """
        Estimates the RMS (standard deviation) of the target values. The default
        implementation queries n_max_smpls samples (if the dataset is not
        finite) or reads all samples (if the dataset is finite) and computes
        the RMS on those values. The RMS value is cached, so repeating calls to
        this function are fast.

        n_max_smpls:
            Maximum number of samples to use.
        """
        # Copy this instance so we won't influence the state of this dataset.
        this = copy.deepcopy(self)

        # If the "_rms" member variable does not exist, estimate the rms and
        # create that member variable.
        if not hasattr(self, "_rms"):
            if self.is_finite:
                # Successively read training, validation, and test samples, at
                # most n_max_smpls samples
                n_smpls = 0
                _, ys_training = this.sample(
                    min(n_max_smpls, self.n_max_smpls_training), "training")

                # Read the validation samples, taking the already read training
                # samples into account
                n_smpls = ys_training.shape[0]
                _, ys_validation = this.sample(
                    min(n_max_smpls - n_smpls, self.n_max_smpls_validation),
                    "validation")

                # Read the test samples, taking the already read training and
                # validation samples into account.
                n_smpls = n_smpls + ys_validation.shape[0]
                _, ys_test = this.sample(
                    min(n_max_smpls - n_smpls, self.n_max_smpls_test), "test")

                # Merge all read samples into a single array
                ys = np.concatenate((ys_training, ys_validation, ys_test),
                                    axis=0)
            else:
                # Just sample n_max_smpls samples
                _, ys = this.sample(n_max_smpls)

            # Compute the RMS and store it in the "_rms" member variable
            self._rms = np.sqrt(np.mean(np.square(ys)))

        return self._rms

    def error(self, ys, ys_hat):
        """
        Computes the reconstruction error. Per default this is implemented as
        the NRMSE (see do_error)
        """

        # Make sure the given arrays are numpy arrays and have the right shape
        ys, ys_hat = np.asarray(ys), np.asarray(ys_hat)
        if ys.ndim != 2:
            ys = ys.reshape(-1, self.n_dim_out)
        if ys_hat.ndim != 2:
            ys_hat = ys.reshape(-1, self.n_dim_out)
        assert ys.shape[0] == ys_hat.shape[0]
        assert ys.shape == (ys.shape[0], self.n_dim_out)
        assert ys_hat.shape == (ys_hat.shape[0], self.n_dim_out)

        # Compute the NRMSE
        return self.do_error(ys, ys_hat)

    @property
    def dt(self):
        return self._dt

    @property
    def rng(self):
        return self._rng

    @property
    def descr(self):
        return self._descr

    @property
    def n_dim_in(self):
        return self._descr.n_dim_in

    @property
    def n_dim_out(self):
        return self._descr.n_dim_out

    @property
    def n_max_smpls_training(self):
        return self._descr.n_max_smpls_training

    @property
    def n_max_smpls_validation(self):
        return self._descr.n_max_smpls_validation

    @property
    def n_max_smpls_test(self):
        return self._descr.n_max_smpls_test

    @property
    def n_dim_(self):
        return self._descr.n_control_dim

    @property
    def is_finite(self):
        return self._descr.is_finite

    @property
    def is_time_continous(self):
        return self._descr.is_time_continous

    @property
    def is_classification_dataset(self):
        return False


class ClassificationDataset(Dataset):
    """
    Implements an error measure for datasets with a one-hot coded class vector
    as a target.
    """
    def do_error(self, ys, ys_hat):
        # Make sure both arrays have a one-hot coding. Do this by finding the
        # dimension with the maximum value. This operation does nothing if
        # the array already is a boolean array.
        ys_oho = ys == np.max(ys, axis=1)[:, None]
        ys_hat_oho = ys_hat == np.max(ys_hat, axis=1)[:, None]

        # Compute the number of samples that differ in at least one place
        n_smpls = ys.shape[0]
        n_errs = np.sum(
            np.asarray(np.any(ys_oho != ys_hat_oho, axis=1), dtype=np.int))
        return n_errs / n_smpls

    @property
    def is_classification_dataset(self):
        return True


###############################################################################
# Network Base Classes                                                        #
###############################################################################


class Network:
    def do_init(self):
        raise NotImplemented("do_init not implemented")

    def do_activities(self, xs):
        raise NotImplemented("do_activities not implemented")

    def do_jacobian(self, xs, as_):
        raise NotImplemented("do_jacobian not implemented")

    def do_numerical_jacobian(self, xs, eta):
        def calc_da(k, idcs=tuple()):
            # Prepend a slice selecting all neurons at the same time to the
            # given parameter index
            idcs = (slice(None), *idcs)

            # Perform a single step in the negative direction. Restore the
            # full copy of the parameters after we're done
            self._params[k][idcs] -= 0.5 * eta  # Adapt parameters
            self.normalize_params()
            asm1 = self.activities(xs)  # Forward pass
            self._params[k][...] = params[k]  # Reset

            # Perform a single step in the positive direction.
            self._params[k][idcs] += 0.5 * eta  # Adapt parameters
            self.normalize_params()
            asp1 = self.activities(xs)  # Forward pass
            self._params[k][...] = params[k]  # Reset

            return (asp1 - asm1) / eta

        # Create a backup of the current parameters
        params = copy.deepcopy(self._params)

        # Numerically compute the derivative
        res = {}
        try:
            for k, p in params.items():
                ndim = p.ndim - 1
                if ndim >= 1:
                    da = np.zeros((xs.shape[0], *p.shape))
                    for idcs in np.indices(p.shape[1:]).T.reshape(-1, ndim):
                        da[(slice(None), slice(None),
                            *idcs)] = calc_da(k, idcs)
                else:
                    da = calc_da(k)
                res[k] = da
        finally:
            for k, p in params.items():
                self._params[k][...] = params[k]
        return res

    def __init__(self, n_dim_in, n_dim_hidden, rng=np.random, *args, **kwargs):
        # Copy the given arguments
        self._n_dim_in = n_dim_in
        self._n_dim_hidden = n_dim_hidden
        self._rng = mkrng(rng)

        # Call the implementation-specific initialisation method
        self._params = self.do_init(*args, **kwargs)
        assert type(self._params) is dict

        # The first dimension of each parameter vector must be equal to the
        # number of neurons. The assumption is that each row in the parameter
        # matrices affects the activity of exactly one neuron. This assumption
        # is particularly important for the "jacobian" function.
        assert all((type(p) is np.ndarray) and (p.ndim >= 1) and (
            p.shape[0] == self.n_dim_hidden) for p in self._params.values())

    def _condition(self, xs, n):
        xs = np.asarray(xs)
        if xs.ndim != 2:
            xs = xs.reshape(-1, n)
        assert xs.shape == (xs.shape[0], n)
        return xs

    def normalize_params(self):
        """
        To be called every time an external process is updating the network
        parameters. This method will ensure that certain invariants are met,
        such as the encoders always being unit vectors. Does nothing in the
        default implementation
        """
        pass

    def activities(self, xs):
        """
        Computes the activities of the hidden neuron layer for the given input.
        """
        xs = self._condition(xs, self.n_dim_in)
        ys = self.do_activities(xs)
        assert ys.shape == (xs.shape[0], self.n_dim_hidden)
        return ys

    def jacobian(self, xs, numerical=False, eta=1e-6):
        """
        Computes the impact of a parameter change on the activities of each
        individual neuron. Returns a dictionary just like the "params"
        dictionary. Each entry is an array, where the first two dimensions have
        the following shape:

            n_smpls x n_dim_hidden x ...

        xs:
            Input into the network used to compute the activities.
        numerical:
            When set to True computes the Jacobian numerically. This is slow and
            (relatively) imprecise, and mostly for testing purposes.
        eta:
            When computing the Jacobian numerically, defines the step-size used
            in the differential quotient.
        """

        # Make sure xs and as_ are 2D arrays and that their first dimension has
        # the same number of entries
        xs = self._condition(xs, self.n_dim_in)

        # Compute the jacobian
        if numerical:
            jacobian = self.do_numerical_jacobian(xs, eta)
        else:
            jacobian = self.do_jacobian(xs)

        # Make sure that the returned dictionary has the right entries
        assert (type(jacobian) == dict) and (tuple(jacobian.keys()) == tuple(
            self.params.keys()))

        # Make sure that each dictionary entry has the right dimensionality
        assert all(
            (p.shape[0] == xs.shape[0]) and (p.shape[1] == self.n_dim_hidden)
            and (p.shape[1:] == self.params[k].shape)
            for k, p in jacobian.items())

        return jacobian

    @property
    def params(self):
        return self._params

    @property
    def n_dim_in(self):
        return self._n_dim_in

    @property
    def n_dim_hidden(self):
        return self._n_dim_hidden

    @property
    def rng(self):
        return self._rng


###############################################################################
# Decoder Learning Rule Base Class                                            #
###############################################################################


@dataclasses.dataclass
class DecoderLearningRuleDescriptor:
    """
    Dataclass containing run-time information about a decoder learning rule
    """
    returns_gradient: bool = True


class DecoderLearningRule:
    def do_init(self):
        raise NotImplemented("do_init not implemented")

    def do_step(self, As, ys, errs, D):
        raise NotImplemented("do_step not implemented")

    def __init__(self,
                 n_dim_hidden,
                 n_dim_out,
                 rng=np.random,
                 *args,
                 **kwargs):
        # Copy the given arguments
        assert (n_dim_hidden > 0) and (n_dim_out > 0)
        self._n_dim_hidden = n_dim_hidden
        self._n_dim_out = n_dim_out
        self._rng = mkrng(rng)

        # Call the implementation-specific constructor
        self._descr = self.do_init(*args, **kwargs)
        assert isinstance(self._descr, DecoderLearningRuleDescriptor)

    def step(self, As, ys, errs, D):
        """
        Computes an update for the given decoder.

        As: Current pre-neuron activities.
        ys: Current target values.
        """
        # Make sure the input arrays have the right shape
        As, ys, errs, D = np.asarray(As), np.asarray(ys), np.asarray(
            errs), np.asarray(D)
        if As.ndim != 2:
            As = As.reshape(-1, self.n_dim_hidden)
        if ys.ndim != 2:
            ys = ys.reshape(-1, self.n_dim_out)
        if errs.ndim != 2:
            errs = errs.reshape(-1, self.n_dim_out)
        assert As.shape[0] == ys.shape[0] == errs.shape[0]
        assert As.shape == (As.shape[0], self.n_dim_hidden)
        assert ys.shape == (ys.shape[0], self.n_dim_out)
        assert errs.shape == (errs.shape[0], self.n_dim_out)
        assert D.shape == (self.n_dim_out, self.n_dim_hidden)

        # Call the actual step function and check the output
        dD = self.do_step(As, ys, errs, D)
        assert dD.shape == (self.n_dim_out, self.n_dim_hidden)
        return dD

    @property
    def rng(self):
        return rng

    @property
    def n_dim_hidden(self):
        return self._n_dim_hidden

    @property
    def n_dim_out(self):
        return self._n_dim_out

    @property
    def returns_gradient(self):
        return self._descr.returns_gradient


###############################################################################
# Encoder Learning Rule Base Class                                            #
###############################################################################


class EncoderLearningRule:
    def do_init(self):
        raise NotImplemented("do_init not implemented")

    def do_step(self, As, xs, errs, D, net):
        raise NotImplemented("do_step not implemented")

    def __init__(self, n_dim_in, n_dim_hidden, n_dim_out, rng, *args,
                 **kwargs):
        # Copy the given arguments
        assert (n_dim_in > 0) and (n_dim_hidden > 0)
        self._n_dim_in = n_dim_in
        self._n_dim_hidden = n_dim_hidden
        self._n_dim_out = n_dim_out
        self._rng = mkrng(rng)

        # Call the implementation-specific constructor
        self.do_init(*args, **kwargs)

    def step(self, As, xs, errs, D, net):
        """
        Computes an update for the given decoder.

        As: Current activities.
        xs: Current input values.
        """
        # Make sure the input arrays have the right shape
        As, xs, errs, D = np.asarray(As), np.asarray(xs), np.asarray(
            errs), np.asarray(D)
        if As.ndim != 2:
            As = As.reshape(-1, self.n_dim_hidden)
        if xs.ndim != 2:
            xs = xs.reshape(-1, self.n_dim_in)
        if errs.ndim != 2:
            errs = errs.reshape(-1, self.n_dim_out)
        if D.ndim != 2:
            D = D.reshape()
        assert As.shape[0] == xs.shape[0] == errs.shape[0]
        assert As.shape == (As.shape[0], self.n_dim_hidden)
        assert xs.shape == (xs.shape[0], self.n_dim_in)
        assert errs.shape == (errs.shape[0], self.n_dim_out)

        # Call the actual step function and check the output
        dparams = self.do_step(As, xs, errs, D, net)
        assert all(dp.shape == net.params[k].shape
                   for k, dp in dparams.items())
        return dparams

    @property
    def rng(self):
        return rng

    @property
    def n_dim_in(self):
        return self._n_dim_in

    @property
    def n_dim_hidden(self):
        return self._n_dim_hidden

    @property
    def n_dim_out(self):
        return self._n_dim_out


###############################################################################
# Optimizer Base Class                                                        #
###############################################################################


class Optimizer:
    def do_init(self):
        raise NotImplemented("do_init not implemented")

    def do_step(self, p, dp):
        raise NotImplemented("do_step not implemented")

    def __init__(self, rng=np.random, *args, **kwargs):
        self._rng = mkrng(rng)
        self.do_init(*args, **kwargs)

    def step(self, p, dp):
        self.do_step(p, dp)

    @property
    def rng(self):
        return self._rng


###############################################################################
# Manifest Classes                                                            #
###############################################################################


class Manifest:
    def __init__(self, name, ctor, params=None):
        self._name = name
        self._ctor = ctor
        self._params = {} if params is None else params

    @property
    def name(self):
        return self._name

    @property
    def ctor(self):
        return self._ctor

    @property
    def params(self):
        return self._params


class DatasetManifest(Manifest):
    pass


class NetworkManifest(Manifest):
    pass


class DecoderLearningRuleManifest(Manifest):
    pass


class EncoderLearningRuleManifest(Manifest):
    def __init__(self,
                 name,
                 ctor,
                 params=None,
                 supported_network_classes=None,
                 is_supervised=True):
        # Call the inherited constructor
        super().__init__(name, ctor, params)

        # Make sure that the "supported_network_classes" parameter has the right
        # type
        assert (supported_network_classes is None) or (isinstance(
            supported_network_classes, set) and all(
                issubclass(cls, Network) for cls in supported_network_classes))

        # Copy the given arguments
        self._supported_network_classes = supported_network_classes
        self._is_supervised = is_supervised

    @property
    def supported_network_classes(self):
        return self._supported_network_classes

    @property
    def is_supervised(self):
        return self._is_supervised


class OptimizerManifest(Manifest):
    pass


###############################################################################
# Task descriptor                                                             #
###############################################################################


@dataclasses.dataclass
class TaskDescriptor:
    """
    The trial descriptor collects information about a single trial, i.e., a
    particular combination of components and their parameters.
    """
    optimizer_name: str = ""
    optimizer_params: dict = dataclasses.field(default_factory=dict)
    dataset_name: str = ""
    dataset_params: dict = dataclasses.field(default_factory=dict)
    network_name: str = ""
    network_params: dict = dataclasses.field(default_factory=dict)
    decoder_learner_name: str = ""
    decoder_learner_params: dict = dataclasses.field(default_factory=dict)
    encoder_learner_name: str = ""
    encoder_learner_params: dict = dataclasses.field(default_factory=dict)
    seed: int = 0
    batch_size: int = 100
    n_epochs: int = 200
    n_dim_hidden: int = 100
    sequential: bool = False

