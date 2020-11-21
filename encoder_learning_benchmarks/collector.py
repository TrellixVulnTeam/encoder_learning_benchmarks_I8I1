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

import types
import numpy as np
import hashlib
import json

from . import datasets
from . import networks
from . import optimizers
from . import rules

from .common import *


def _collect_submodules(module):
    return [
        value for key, value in module.__dict__.items()
        if (isinstance(key, str) and (not key.startswith("__")) and isinstance(
            value, types.ModuleType) and hasattr(value, 'manifest'))
    ]


_MODULES = {}


def collect_modules(ignorelist=None, allowlist=None):
    """
    Returns a map containing all datasets, networks, optimizers and learning
    rules (split into encoder and decoder learning rules).
    """

    # Pre-process the ignorelist and allowlist
    def mkset(it):
        if ((it is None) or (len(it) == 0)):
            return None
        return set(s.lower() for s in it)

    ignorelist = mkset(ignorelist)
    allowlist = mkset(allowlist)
    if (not ignorelist is None) and (not allowlist is None):
        raise RuntimeError(
            "Cannot specify both an ignorelist and an allow list")

    # Use the cached modules if available
    if len(_MODULES) == 0:
        # First collect all submodules
        modules = sorted([
            *_collect_submodules(datasets), *_collect_submodules(networks),
            *_collect_submodules(optimizers), *_collect_submodules(rules)
        ],
                         key=lambda m: m.manifest.name)

        # Then sort them into the corresponding categories depending on the manifest
        # type
        def filter(manifest_type):
            return {
                m.manifest.name: m.manifest
                for m in modules if isinstance(m.manifest, manifest_type)
            }

        _MODULES["datasets"] = filter(DatasetManifest)
        _MODULES["networks"] = filter(NetworkManifest)
        _MODULES["optimizers"] = filter(OptimizerManifest)
        _MODULES["decoder_learners"] = filter(DecoderLearningRuleManifest)
        _MODULES["encoder_learners"] = filter(EncoderLearningRuleManifest)

    # Filter the modules according to the allowlist/ignorelist
    res = {}
    used_xlist_keys = set()
    for d_key, d_value in _MODULES.items():
        d_res = {}
        for key, value in d_value.items():
            if not (allowlist is None):
                if not key.lower() in allowlist:
                    continue
                else:
                    used_xlist_keys.add(key.lower())
            if not (ignorelist is None):
                if key.lower() in ignorelist:
                    used_xlist_keys.add(key.lower())
                    continue
            d_res[key] = value
        res[d_key] = d_res

    # Make sure all allowlist/ignorelist keys actually corresponded to a module
    def check_xlist_keys(lst):
        if not lst is None:
            for key in lst:
                if not key in used_xlist_keys:
                    raise RuntimeError(
                        "Allowlist or ignorelist entry \"{}\" is not a valid module!".
                        format(key))

    check_xlist_keys(ignorelist)
    check_xlist_keys(allowlist)

    return res


_FILE_HASHES = {}
_MANIFEST_HASHES = {}


def manifest_hash(manifest):
    """
    Computes a unique hash for the given manifest class describing the state of
    the code-base it depends on.
    """
    def file_hash(hasher, filename):
        if not filename in _FILE_HASHES:
            hasher = hashlib.sha1()
            with open(filename, 'rb') as f:
                while True:
                    data = f.read(65536)
                    if not data:
                        break
                    hasher.update(data)
            _FILE_HASHES[filename] = hasher.digest()
        return _FILE_HASHES[filename]

    def module_hash(module, hashes={}):
        # Do nothing if we already computed the hash for this module
        if module in hashes:
            return hashes

        # Do nothing if the module does not belong to this class
        if not (hasattr(module, '__file__') and
                ('encoder_learning_benchmarks' in module.__file__)):
            return hashes

        # Compute the hash of the file corresponding to this module
        hashes[module] = file_hash(hasher, module.__file__)

        # See what else is used by this module
        for k, v in module.__dict__.items():
            if isinstance(v, types.ModuleType):
                module_hash(v, hashes)
            elif hasattr(v, '__module__'):
                module_hash(__import__(v.__module__, fromlist=[None]), hashes)

        return hashes

    if not manifest in _MANIFEST_HASHES:
        hasher = hashlib.sha1()
        hashes = module_hash(
            __import__(manifest.ctor.__module__, fromlist=[None]))
        last = None
        for filehash in sorted(hashes.values()):
            if last == filehash:
                continue  # Only count each file once
            hasher.update(filehash)
            last = filehash
        _MANIFEST_HASHES[manifest] = hasher.digest()
    return _MANIFEST_HASHES[manifest]


def compute_task_hash(task):
    hasher = hashlib.sha1()

    # Encode the task as a JSON string
    task_dict = dataclasses.asdict(task)
    task_str = json.dumps(task_dict, sort_keys=True).encode("utf-8")
    hasher.update(task_str)

    # For each of the modules constituting the task, add the corresponding
    # manifest hash to the hasher
    modules = collect_modules()
    for key in [
            "optimizer_name", "dataset_name", "network_name",
            "decoder_learner_name", "encoder_learner_name"
    ]:
        module_key = "_".join(key.split("_")[:-1]) + "s"
        if task_dict[key] in modules[module_key]:
            module_hash = manifest_hash(modules[module_key][task_dict[key]])
            hasher.update(module_hash)

    return hasher.hexdigest()


def _collect_params(task_descr, manifest):
    if manifest is None:
        return {}

    params_dict = {}
    for k, v in sorted(manifest.params.items()):
        # Append a list of parameters for this key to the list of parameters
        params_dict[k] = []

        def append_single(value):
            if not isinstance(value, (bool, int, float, str, type(None))):
                raise RuntimeError(
                    "Error while collecting parameters for component \"{}\". Value \"{}\" for parameter key \"{}\" is not primitive."
                    .format(manifest.name, value, k))
            params_dict[k].append(value)

        # If a function is given as a value, call that function with the given
        # task descriptor. This is done because some parameters depend on the
        # current task, i.e., some learning rates depend on the dataset at hand
        # and datasets may be different depending on whether we're doing online
        # or offline learning (some datasets support being temporally biased
        # when doing online learning).
        if callable(v):
            v = v(task_descr)

        # If the value v is a simple list, simple append each entry of that list
        # to the dictionary
        if (isinstance(v, list) or isinstance(v, tuple) or isinstance(v, set)):
            for value in sorted(v):
                append_single(value)
        elif isinstance(v, dict):
            # If we have a dictionary, the value is pointing to a complex object
            # that is not serialisable to JSON. Instead we'll store the name
            # of the dictionary entry which we'll use to restore the entry
            # later.
            for k2 in sorted(v.keys()):
                params_dict[k].append({"@ref": k2})
        else:
            append_single(v)

    return params_dict


def _combine_dicts(*dicts):
    # Merge all dictionary values into a single array
    n_dicts = len(dicts)
    tuples, lens = [], []
    for i in range(n_dicts):
        for k, v in dicts[i].items():
            assert (isinstance(v, list))  # All values are assumed to be lists
            tuples.append((i, k, v))
            lens.append(len(v))

    # Iterate over all combinations in the tuples array. For each combination,
    # reconstruct the input dictionaries with a single value assigned to each
    # dictionary entry
    idcs = [0] * len(tuples)
    done = False
    while not done:
        # Emit one set of dictionaries
        res = [{} for _ in range(n_dicts)]
        for j, (i, k, v) in enumerate(tuples):
            res[i][k] = v[idcs[j]]
        yield res

        # Increment the indices
        done = True
        for i in range(len(tuples)):
            idcs[i] += 1
            if idcs[i] >= lens[i]:
                idcs[i] = 0
            else:
                done = False
                break


def _collect_tasks_for_setup(tasks, n_repeat, method, optimizer, dataset,
                             network, declrn, enclrn):
    # Pre-compute the seeds for all repetitions
    rng = np.random.RandomState(6592392)
    seeds = [rng.randint((1 << 31) - 1) for _ in range(n_repeat)]

    # Create all n_repeat versions of the task
    for i in range(n_repeat):
        task = TaskDescriptor()
        task.optimizer_name = optimizer.name
        task.dataset_name = dataset.name
        task.network_name = network.name
        task.decoder_learner_name = declrn.name
        task.encoder_learner_name = ("" if enclrn is None else enclrn.name)
        task.seed = seeds[i]  # Reproducibly select a seed for each combination
        task.sequential = (method == "online")
        task.batch_size = 1 if (method == "online") else 100

        # For each component, collect all parameters given the task descriptor
        # so far
        optimizer_params = _collect_params(task, optimizer)
        dataset_params = _collect_params(task, dataset)
        network_params = _collect_params(task, network)
        decoder_learner_params = _collect_params(task, declrn)
        encoder_learner_params = _collect_params(task, enclrn)

        # Iterate over all possible combinations of parameters
        for op, dp, ntp, dlp, elp in _combine_dicts(optimizer_params,
                                                    dataset_params,
                                                    network_params,
                                                    decoder_learner_params,
                                                    encoder_learner_params):
            # Create a copy of the already constructed task and set the
            # parameters accordingly
            t = copy.deepcopy(task)
            t.optimizer_params = op
            t.dataset_params = dp
            t.network_params = ntp
            t.decoder_learner_params = dlp
            t.encoder_learner_params = elp

            # Compute a hash for the task and append it to the tasks dictionary
            tasks[compute_task_hash(t)] = t


def collect_tasks(modules, n_repeat=1):
    """
    Creates a list of tasks combining the different datasets, networks,
    optimizers, decoder_learning_rules and encoder_learning_rules, as well as
    overall methodologies (online vs. offline learning).
    """
    tasks = {}
    for method in ["offline", "online"]:
        for dataset in modules["datasets"].values():
            for network in modules["networks"].values():
                for declrn in modules["decoder_learners"].values():
                    for enclrn in [
                            None, *modules["encoder_learners"].values()
                    ]:
                        for optimizer in modules["optimizers"].values():
                            # Only pair encoder learners with their
                            # corresponding supported network class
                            if (not enclrn is None) and (
                                    not enclrn.supported_network_classes is
                                    None):
                                if not (network.ctor.__name__
                                        in enclrn.supported_network_classes):
                                    continue

                            # XXX Temporarily turn of online learning when using
                            # MNIST (really slow)
                            if (dataset.name == "mnist") and (method
                                                              == "online"):
                                continue

                            _collect_tasks_for_setup(tasks, n_repeat, method,
                                                     optimizer, dataset,
                                                     network, declrn, enclrn)

    return tasks

