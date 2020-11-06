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
import h5py
import json
import numpy as np
import os
import sys
import tqdm
import matplotlib.cm as cm

SkipLabel = object()
"""
Map assigning a human-readable label to individual parameter dimensions.
"""
LABELS = {
    "optimizer/#name": {
        "adam": "Adam",
        "lstsq": "Least squares",
    },
    "network/#name": {
        "perceptron": "Single hidden-layer perceptron",
    },
    "decoder_learner/#name": {
        "pes": "Delta",
        "lstsq": "Least squares",
    },
    "encoder_learner/#name": {
        "": "Static encoders",
        "backprop": "Backpropagation",
        "feedback_alignment": "Feedback alignment",
        "voja": "Voja",
        "oja": "Oja",
    },
    "dataset/#name": {
        "gaussian_clusters": "Gaussian Clusters",
        "multiplication": "Multiplication",
        "mnist": "MNIST",
    },
    "dataset/skewed": {
        True: "skewed",
        False: SkipLabel,
    },
    "dataset/n_classes": lambda i: "$n_\\mathrm{{cls}}={}$".format(i),
    "dataset/n_dim": lambda i: "$n_\\mathrm{{dim}}={}$".format(i),
    "network/initialisation": {
        "normal": "Gaussian initialisation",
        "nef": "NEF initialisation",
    },
}

COLOURS = {
    "encoder_learner/#name": {
        "": "#2e3436",
        "backprop": cm.get_cmap("tab10")(0.1),
        "feedback_alignment": cm.get_cmap("tab10")(0.2),
        "voja": cm.get_cmap("tab10")(0.0),
        "oja": cm.get_cmap("tab10")(0.3),
    },
    "dataset/#name": {
        "multiplication": "#cc0000",
        "gaussian_clusters": "#3465a4",
        "mnist": "#75507b",
    },
}

MARKERS = {
    "encoder_learner/#name": {
        "": "+",
        "backprop": "^",
        "feedback_alignment": "v",
        "voja": "D",
        "oja": "d",
    }
}

DATASET_ERROR_TYPES = {
    "multiplication": "nrmse",
    "gaussian_clusters": "error_rate",
    "mnist": "error_rate",
}

ERROR_TYPES = {
    None: "Average normalised error",
    "nrmse": "NRMSE",
    "error_rate": "Classification error rate",
}


def load_benchmark_data(tar):
    """
    Loads the benchmark data from the given target directory.
    """

    # Make sure that the target directory exists
    if not os.path.isdir(tar):
        raise RuntimeError(
            "Specified target directory does not exist or is not a directory")

    # Load the manifest file and make sure that it is valid
    manifest_file = os.path.join(tar, "manifest.json")
    if not os.path.isfile(manifest_file):
        raise RuntimeError(
            "Could not find manifest file in the specified target directory")
    with open(manifest_file, 'r') as f:
        manifest = json.load(f)
    if not (("tasks" in manifest) and (isinstance(manifest["tasks"], dict))):
        raise RuntimeError("Invalid manifest file.")

    # Check what files listed in the manifest actually exist. Load the data for
    # those files and return an object containing only the stored objects.
    tasks, not_found = [], []
    n_tasks_total = len(manifest["tasks"])
    for task_idx, (task_hash,
                   task_descr) in enumerate(manifest["tasks"].items()):
        # Skip the file if it does not exist
        task_filename = os.path.join(
            tar, "encoder_learning_benchmarks_{}.h5".format(task_hash))
        if not os.path.isfile(task_filename):
            not_found.append(task_filename)
            continue

        # The file exists, now process it
        with h5py.File(task_filename, 'r') as f:
            # Make sure the task descriptor loaded from the manifest matches
            # what is stored
            task_descr_rec = json.loads(f.attrs["task"])
            if task_descr_rec != task_descr:
                raise RuntimeError("Cannot validate task descriptor")

            # For descriptor keys starting with one of these prefixes, the
            # "*_name" is merged into the "*_params" dictionary
            special_prefix = {
                "optimizer", "dataset", "network", "encoder_learner",
                "decoder_learner"
            }

            task_descr_new = {}
            for descr_key, descr_value in task_descr.items():
                # Eliminate the "@ref" objects from the descriptor
                if isinstance(descr_value, dict):
                    descr_value = {**descr_value}
                    for param_key, param_value in descr_value.items():
                        if (isinstance(param_value, dict)
                                and (len(param_value) == 1)
                                and ("@ref" in param_value)):
                            descr_value[param_key] = param_value["@ref"]

                # Handle the merging of _name into _params
                descr_key_prefix = "_".join(descr_key.split("_")[:-1])
                if descr_key_prefix in special_prefix:
                    if descr_key.endswith("_name"):
                        continue
                    descr_key = descr_key_prefix
                    descr_value["#name"] = task_descr[descr_key_prefix +
                                                      "_name"]

                task_descr_new[descr_key] = descr_value

            # Load the reconstruction errors from the h5 file
            task_data = {
                "descr": task_descr_new,
                "err_test": f["err_test"][()],
                "errs_training": f["errs_training"][()],
                "errs_validation": f["errs_validation"][()],
            }

            # Store the completely read task in the tasks dictionary. Turn the
            # task hash into a tuple; when merging datasets. We'll use this
            # to keep track of the original sources in merged datasets (i.e.,
            # those merged accross multiple seeds).
            tasks.append(((task_hash, ), task_data))

        # Print some informative messages
        if ((task_idx % 100) == 0) or (task_idx == (n_tasks_total - 1)):
            sys.stderr.write("\rLoaded {}/{} result files".format(
                len(tasks), n_tasks_total))
            sys.stderr.flush()
    sys.stderr.write("\n")
    if len(not_found) > 0:
        print("The following files were missing:\n{}".format(
            "\n".join(not_found)))

    return tasks


def merge_dicts(a, b, recursive=False):
    """
    Merges two dictionaries a and b if they have the same keys. Differing
    keys are merged into a list.
    """
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return False
    if a.keys() != b.keys():
        return False
    for k in a.keys():
        va, vb = a[k], b[k]
        if va != vb:
            if recursive and isinstance(va, dict) and isinstance(vb, dict):
                if merge_dicts(va, vb, True):
                    continue
            if not isinstance(va, list):
                va = [va]
            if not isinstance(vb, list):
                vb = [vb]
            m = sorted([*va, *vb])
            a[k] = []
            for i in range(len(m)):
                if (i == 0) or (m[i] != m[i - 1]):
                    a[k].append(m[i])
    return True


def extract_parameter_sets(data):
    """
    Extracts parameters from the dataset that are actually changing.
    """

    # Iterate over all dataset elements. Merge create lists of parameters that
    # are actually changing.
    sets = {}
    for _, task_data in data:
        descr = task_data["descr"]
        for param_key, param_value in descr.items():
            if not param_key in sets:
                sets[param_key] = []
            if not param_value in sets[param_key]:
                sets[param_key].append(copy.deepcopy(param_value))

    def cleanup(dict_):
        """
        Removes keys that do not change; merges keys accross dictionaries.
        """
        keys_to_delete = []
        did_change = False
        for k, v in dict_.items():
            if (not isinstance(v, list)) or (len(v) == 1):
                if not isinstance(v, dict):
                    keys_to_delete.append(k)
                    did_change = True
                continue
            indices_to_delete = set()
            for i1 in range(0, len(v)):
                if i1 in indices_to_delete:
                    continue
                for i2 in range(i1 + 1, len(v)):
                    if i2 in indices_to_delete:
                        continue
                    if merge_dicts(v[i1], v[i2]):
                        did_change = True
                        indices_to_delete.add(i2)
            for i, idx in enumerate(sorted(indices_to_delete)):
                del v[idx - i]
            for i in range(len(v)):
                if isinstance(v[i], dict):
                    did_change = did_change | cleanup(v[i])
            if isinstance(dict_[k], list) and (len(dict_[k]) == 1):
                did_change = True
                dict_[k] = dict_[k][0]
        for k in keys_to_delete:
            del dict_[k]
        return did_change

    while cleanup(sets):
        pass

    return sets


def filter_matches(target_dict, filter_dict):
    """
    Returns True if the entries in the target dictionary specified in the filter
    dictionary are equal. Recurses into nested dictionaries.
    """
    for k, v in filter_dict.items():
        if not k in target_dict:
            return False
        elif isinstance(v, dict) and isinstance(target_dict[k], dict):
            if not filter_matches(target_dict[k], v):
                return False
        elif (not v is None) and (v != target_dict[k]):
            return False
    return True


def filter_matches_mergeable(target_dict_1, target_dict_2, filter_dict):
    # The filter has to match both target dictionaries. Otherwise the entries
    # are not mergeable.
    if not (filter_matches(target_dict_1, filter_dict)
            and filter_matches(target_dict_2, filter_dict)):
        return False

    # The two dictionaries must have exactly the same keys, otherwise they are
    # not mergeable.
    if target_dict_1.keys() != target_dict_2.keys():
        return False
    target_keys = target_dict_1.keys()

    # All values have to be the same in the dictionary, unless a "None" key is
    # specified in the corresponding filter dictionary entry.
    for k in target_keys:
        if (k in filter_dict) and (filter_dict[k] is None):
            continue
        elif ((k in filter_dict) and isinstance(filter_dict[k], dict)
              and isinstance(target_dict_1[k], dict)
              and isinstance(target_dict_2[k], dict)):
            if not filter_matches_mergeable(target_dict_1[k], target_dict_2[k],
                                            filter_dict[k]):
                return False
        elif (target_dict_1[k] != target_dict_2[k]):
            return False
    return True


def filter_benchmark_data(data, filter_dict):
    res = []
    for task_hash, task_data in data:
        if filter_matches(task_data["descr"], filter_dict):
            res.append((task_hash, task_data))
    return res


def sort_benchmark_data(data):
    return sorted(data,
                  key=lambda o: json.dumps(o[1]["descr"], sort_keys=True))


def merge_benchmark_data(data, filter_dict):
    """
    Combines benchmark results if the results only differ accross the properties
    specified with the placeholder "None" in the filter tuple. For example,
    the following code will merge all datasets that only differ in their seed.

        merge_benchmark_data(data, {"seed": None})

    In contrast, the following example merges all samples that only differ in
    the "dataset/n_classes" property, and, additionally, only merges those
    entries that have the "sequential" property set to True.

        merge_benchmark_data(data, {"dataset": {"n_classes": None},
                                    "sequential": True})
    """
    res = []
    skip = set()
    for i1 in tqdm.tqdm(range(len(data))):
        if i1 in skip:
            continue
        key_1, value_1 = copy.deepcopy(data[i1])
        descr_1 = value_1["descr"]
        n_merged = 0
        for i2 in range(i1 + 1, len(data)):
            key_2, value_2 = data[i2]
            descr_2 = value_2["descr"]
            if filter_matches_mergeable(descr_1, descr_2, filter_dict):
                if merge_dicts(descr_1, descr_2, True):
                    # Merge was successful, skip the second dictionary
                    skip.add(i2)

                    # Store the merged descriptor in the first task
                    value_1["descr"] = descr_1

                    # Keep track of all the task hashes that constitute the
                    # merged descriptor
                    key_1 = (*key_1, *key_2)

                    # Merge all remaining measurements
                    for k, v1 in value_1.items():
                        if k == "descr":
                            continue
                        v2 = value_2[k]
                        if isinstance(v1, np.ndarray) and isinstance(
                                v2, np.ndarray):
                            if v1.ndim != 2:
                                v1 = v1.reshape(1, -1)
                            if v2.ndim != 2:
                                v2 = v2.reshape(1, -1)
                            value_1[k] = np.concatenate((v1, v2), axis=0)
                        else:
                            if not isinstance(v1, list):
                                v1 = [v1]
                            if not isinstance(v2, list):
                                v2 = [v2]
                            value_1[k] = [*v1, *v2]

                    n_merged += 1

        res.append((key_1, value_1))
    return res


def compute_benchmark_data_ylabel(data):
    error_types = []
    for _, task_data in data:
        if "error_types" in task_data:
            error_types.append(error_types)
        else:
            dset = task_data["descr"]["dataset"]["#name"]
            error_types.append(DATASET_ERROR_TYPES[dset])

    error_type = error_types[0] if len(error_types) > 0 else None
    for i, error_type in enumerate(error_types):
        if error_types[i] != error_types[0]:
            error_type = None
            break

    return ERROR_TYPES[error_type]


def compute_benchmark_data_info_sources(sets,
                                        source_dictionary,
                                        include_unlisted_sources=True):
    # Iterate over the dictionary in insertion order. Search for the keys in
    # that dictionary that actualy vary in the current dataset. Insert
    # them in this order into the label_sources array.
    sources = []
    for key, value in source_dictionary.items():
        key_parts = key.split("/")
        sets_ptr = sets
        for key_part in key_parts:
            if isinstance(sets_ptr, dict):
                if key_part in sets_ptr:
                    sets_ptr = sets_ptr[key_part]
                    continue
            sets_ptr = None
        if not sets_ptr is None:
            sources.append(key)

    # If so desired, iterate over the parameter sets again and add all the
    # varying parameters for which we did not find a specified label
    if include_unlisted_sources:

        def extract_keys(d):
            res = []
            for k, v in d.items():
                if isinstance(v, dict):
                    res += [k + "/" + k2 for k2 in extract_keys(v)]
                else:
                    res += [k]
            return res

        for key in extract_keys(sets):
            if not key in sources:
                sources.append(key)

    return sources


def compute_benchmark_data_info(data):
    import matplotlib.colors

    # Fetch the information sources from the dictionaries defined at the
    # beginning of this file
    sets = extract_parameter_sets(data)
    label_sources = compute_benchmark_data_info_sources(sets, LABELS)
    colour_sources = compute_benchmark_data_info_sources(sets, COLOURS, False)
    marker_sources = compute_benchmark_data_info_sources(sets, MARKERS, False)

    def dpath(p, d):
        for q in p.split("/"):
            if isinstance(d, dict):
                if q in d:
                    d = d[q]
                    continue
            d = None
        return d

    def eval_dictionary(descr, sources, dictionary):
        mapped_data = []
        for source in sources:
            source_value = dpath(source, descr)
            mapped_value = None
            if (not source_value is None) and (source in dictionary):
                if isinstance(dictionary[source],
                              dict) and (source_value in dictionary[source]):
                    mapped_value = dictionary[source][source_value]
                elif callable(dictionary[source]):
                    mapped_value = dictionary[source](source_value)
            mapped_data.append((source, source_value, mapped_value))
        return list(filter(lambda x: not x[2] is SkipLabel, mapped_data))

    # Iterate over each dataset in the given data array and generate the
    # corresponding label, colour, and marker
    labels = []
    colours = []
    colour_counts = {}
    markers = []
    for _, task_data in data:
        # Fetch the task descriptor
        descr = task_data["descr"]

        # Use the dictionaries to fetch the mapped data
        label = eval_dictionary(descr, label_sources, LABELS)
        colour = eval_dictionary(descr, colour_sources, COLOURS)[0][2]
        marker = eval_dictionary(descr, marker_sources, MARKERS)[0][2]

        # Assemble a label from the data extracted above
        label_str = ""
        for i in range(len(label)):
            if i == 1:
                label_str += " ("
            if i >= 2:
                label_str += ", "
            if not label[i][2] is None:
                label_str += label[i][2]
            else:
                label_str += label[i][0].split("/")[-1] + "=" + str(
                    label[i][1])
            if (i >= 1) and (i + 1 == len(label)):
                label_str += ")"

        # Count the number of times a particular colour is used. We'll use
        # different shades of the same colour if we happen to use the same
        # colour twice
        if not colour is None:
            colour = matplotlib.colors.to_rgb(colour)
        if not colour in colour_counts:
            colour_counts[colour] = 0
        colour_counts[colour] += 1

        labels.append(label_str)
        colours.append(colour)
        markers.append(marker)

    # Update the colours based on the usage count
    colour_counter = {k: 0 for k in colour_counts.keys()}
    for i, c in enumerate(colours):
        alpha = np.linspace(1.0, 0.6, colour_counts[c])[colour_counter[c]]
        colour_counter[c] += 1
        bg = np.array((1.0, 1.0, 1.0))
        colours[i] = tuple(bg * (1.0 - alpha) + alpha * np.asarray(c))

    # Compute the number of categories. This is used to select the number of
    # columns in the final plots
    max_colour_count = max(colour_counts.values())
    if max_colour_count == len(data):
        n_categories = 1
    else:
        n_categories = max_colour_count

    return {
        "labels": labels,
        "colours": colours,
        "markers": markers,
        "n_categories": n_categories
    }


def plot_benchmark_data(data,
                        ax=None,
                        figsize=(5, 2),
                        plot_training_err=True,
                        plot_validation_err=True,
                        plot_quartiles=True,
                        plot_median=True,
                        semilogy=None,
                        all_ys=None,
                        setup_axes_only=False):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Create the figure if none has been set yes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if len(data) == 0:
        for spine in ["top", "bottom", "left", "right"]:
            ax.spines[spine].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.5, "No data", va="center", ha="center")
        return

    # Compute label, colour, and marker for the given dataset
    info = compute_benchmark_data_info(data)

    # Array taking track of all the data we plotted
    if all_ys is None:
        all_ys = []

    # Plot the individual datasets
    for i, (_, task_data) in enumerate(data):
        # Fetch the important information
        label = info["labels"][i]
        colour = info["colours"][i]
        marker = info["markers"][i]

        descr = task_data["descr"]

        # Fetch the epoch data for the x-axis
        if not "epochs" in task_data:
            xs = np.arange(1, descr["n_epochs"] + 1)
        else:
            xs = task_data["epochs"]

        marker_interval = len(xs) // 10
        marker_offset = np.linspace(0,
                                    marker_interval,
                                    len(data) + 1,
                                    dtype=np.int)[i]

        # Plot the individual errors
        def plot_errs(ys,
                      colour=None,
                      linestyle=None,
                      plot_quartiles=False,
                      plot_marker=False,
                      label=None,
                      **kwargs):
            if setup_axes_only:
                return

            if ys.ndim != 2:
                ys = ys.reshape(1, -1)

            if plot_quartiles and ys.shape[0] > 1:
                perc25 = np.nanpercentile(ys, 25, axis=0)
                perc75 = np.nanpercentile(ys, 75, axis=0)
                ax.fill_between(xs, perc25, perc75, color=colour, alpha=0.25)

            ys_tar = np.nanmedian(ys, axis=0) if plot_median else np.nanmean(ys,
                                                                       axis=0)
            all_ys.append(ys_tar)
            ax.plot(xs,
                    ys_tar,
                    color=colour,
                    linestyle=linestyle,
                    label=label,
                    marker=marker if plot_marker else None,
                    markevery=(marker_offset, marker_interval),
                    markersize=5,
                    **kwargs)

        if plot_training_err and ("errs_training" in task_data):
            ys = task_data["errs_training"]
            plot_errs(ys, linestyle=(0, (1, 1)), linewidth=1.0, colour=colour)

        if plot_validation_err and ("errs_validation" in task_data):
            ys = task_data["errs_validation"]
            plot_errs(ys,
                      linestyle="-",
                      plot_quartiles=plot_quartiles,
                      plot_marker=True,
                      label=label,
                      linewidth=1.5,
                      colour=colour)

    # Set the axis scaling
    all_ys = np.asarray(all_ys)
    all_ys_min = np.nanpercentile(all_ys, 5) * 0.99
    all_ys_max = np.nanpercentile(all_ys, 95) * 1.01
    if semilogy is None:
        if np.nanpercentile(np.abs(all_ys), 1) == 0.0:
            semilogy = False
        else:
            semilogy = True

    if semilogy:
        ylim_min = np.floor(np.log10(all_ys_min) * 10) / 10
        ylim_max = np.ceil(np.log10(all_ys_max) * 10) / 10
        ax.set_yscale("log")
        ax.set_ylim(np.power(10, ylim_min), np.power(10, ylim_max))
    if not semilogy:
        ylim_max_mag = np.power(10, -np.floor(np.log10(all_ys_max)))
        print(ylim_max_mag, all_ys_max)
        ylim_max = np.ceil(
            np.max(np.asarray(all_ys)) * ylim_max_mag) / ylim_max_mag
        ax.set_ylim(0, ylim_max)

    ax.legend(ncol=info["n_categories"])  # TODO return legend data

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))

    ax.set_xlim(min(xs), max(xs))
    ax.set_xticks(np.linspace(min(xs), max(xs), 5, dtype=np.int))
    ax.set_xlabel("Epoch")
    ax.set_ylabel(compute_benchmark_data_ylabel(data))

