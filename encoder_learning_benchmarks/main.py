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

import argparse
import dataclasses
import json
import numpy as np
import os
import re
import sys
import time

import logging
logger = logging.getLogger(__name__)

from . import collector
from . import common
from . import benchmark

###############################################################################
# Parent process functionality                                                #
###############################################################################


def parent_launch_task(debug, *args):
    import copy
    import subprocess

    # Assemble the environment, disable multi-threading. We are running multiple
    # experiments in parallel with a single thread, which is generally much more
    # efficient
    env = {**os.environ}
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    # Fetch the path this package is located in and add it to the Python path
    path = os.path.abspath(os.path.dirname(os.path.join("..", __file__)))
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = path + ":" + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = path

    # Assemble the parameters used to run the python interpreter
    args = list(
        filter(
            bool,
            [
                sys.executable,  # Run the Python interpreter
                None if debug else
                "-O",  # Turn on optimizations if we're not debugging
                "-u",  # Do not buffer stdin and stdout
                "-m",
                "encoder_learning_benchmarks.main",
                *args
            ]))

    # Create a pipe to communicate with the sub process
    stderr_pipe_in, stderr_pipe_out = os.pipe()
    stdout_pipe_in, stdout_pipe_out = os.pipe()

    # Create the process and return the process descriptor
    process = subprocess.Popen(args,
                               shell=False,
                               env=env,
                               stdout=stdout_pipe_out,
                               stderr=stderr_pipe_out,
                               stdin=subprocess.DEVNULL)
    return {
        "process": process,
        "stdout_pipe_in": stdout_pipe_in,
        "stdout_pipe_out": stdout_pipe_out,
        "stderr_pipe_in": stderr_pipe_in,
        "stderr_pipe_out": stderr_pipe_out,
        "n_epochs_done": 0,
        "stdout_buf": b"",
        "stderr_buf": b"",
    }


def parent_wait_task(processes, timeout=0.1):
    import select

    # Assemble the list of pipes to read from
    rlist = []
    for process_data in processes.values():
        rlist.append(process_data["stdout_pipe_in"])
        rlist.append(process_data["stderr_pipe_in"])

    t0 = time.time()
    rem_timeout = timeout
    while rem_timeout > 0.0:
        # Wait for one of the subprocesses to become ready. Make sure the total
        # time spent in "select" does not exceed the given timeout. This
        # typically prevents starvation of the parent process by its children.
        rlist, _, _ = select.select(rlist, [], [], rem_timeout)
        rem_timeout = timeout - (time.time() - t0)

        # Find the processes that became ready and read data from that process
        # into a buffer
        for process_data in processes.values():
            # Read data from stdout if any data is available
            stdout_pipe_in = process_data["stdout_pipe_in"]
            if stdout_pipe_in in rlist:
                # Read a chunk of data
                process_data["stdout_buf"] += os.read(stdout_pipe_in, 4096)

                # Split the buffer along linebreaks and search from JSON
                # messages
                lines = process_data["stdout_buf"].split(b"\n")
                for line in lines:
                    if ((len(line) > 0) and line[0] == ord("{")
                            and line[-1] == ord("}")):
                        try:
                            msg = json.loads(str(line, "utf-8"))
                            if ("type" in msg) and (msg["type"] == "progress"):
                                process_data["n_epochs_done"] = msg["i"]
                        except json.JSONDecodeError:
                            pass

                # Only keep the last line
                process_data["stdout_buf"] = lines[-1]

            # Read data from stderr if any data is available
            stderr_pipe_in = process_data["stderr_pipe_in"]
            if stderr_pipe_in in rlist:
                process_data["stderr_buf"] += os.read(stderr_pipe_in, 4096)

    # Check whether any of the processes has terminated. If yes, remember that
    # index and remove the corresponding process from the list.
    tasks_done = []
    for task_hash, process_data in processes.items():
        process = process_data["process"]
        if not (process.poll() is None):
            # We no longer need to keep the process handle around
            process_data["process"] = None

            # Mark the process as done
            tasks_done.append(task_hash)

            # Close all open pipes
            parent_kill_child(process_data)

            # Print all messages from stderr if the process exited with a
            # non-zero error code
            if process.returncode != 0:
                raise RuntimeError(
                    "Child process exited with error code.\n\n" +
                    str(process_data["stderr_buf"], "utf-8"))
    for task_hash in tasks_done:
        del processes[task_hash]

    return tasks_done


def parent_kill_child(process_data):
    # Kill the process and wait for it to finish
    if not process_data["process"] is None:
        process_data["process"].kill()
        process_data["process"].wait()
        process_data["process"] = None

    # Make sure the pipe is closed
    if not process_data["stdout_pipe_in"] is None:
        os.close(process_data["stdout_pipe_in"])
        process_data["stdout_pipe_in"] = None
    if not process_data["stdout_pipe_out"] is None:
        os.close(process_data["stdout_pipe_out"])
        process_data["stdout_pipe_out"] = None
    if not process_data["stderr_pipe_in"] is None:
        os.close(process_data["stderr_pipe_in"])
        process_data["stderr_pipe_in"] = None
    if not process_data["stderr_pipe_out"] is None:
        os.close(process_data["stderr_pipe_out"])
        process_data["stderr_pipe_out"] = None


def parent_run_main_loop(args, tasks):
    import multiprocessing
    import tqdm

    # Convert the dictionary into a list
    task_list = list(tasks.items())

    # Variables used for keeping track of the child state
    n_tasks = len(task_list)  # Total number of tasks
    tasks_done = set()  # Hashes of the finished tasks
    task_ptr = 0  # Next task that is to be launched
    processes = {}  # List of currently running processes

    # Detect output files that already exist.
    if os.path.isdir(args.tar):
        logger.info("Collecting completed tasks...")
        pattern = re.compile(
            "^encoder_learning_benchmarks_([0-9a-fA-F]{40})\\.h5$")
        for filename in os.listdir(args.tar):
            match = pattern.match(filename)
            if match:
                task_hash = match.groups(1)[0]
                if task_hash in tasks:
                    tasks_done.add(task_hash)
        if len(tasks_done) == len(tasks):
            logger.info("Found {}/{} completed tasks. Nothing to do.".format(
                len(tasks_done), len(tasks)))
            return
        if len(tasks_done) > 0:
            logger.info(
                "Found {}/{} completed tasks. Resuming execution.".format(
                    len(tasks_done), len(tasks)))
        else:
            logger.info("No completed tasks found.")

    # Determine the maximum number of processes. On all computers but the CTN
    # GPU server half the number of concurrent processes.
    n_max_processes = multiprocessing.cpu_count()
    if "HOSTNAME" in os.environ:
        if not ("ctngpu" in os.environ["HOSTNAME"].lower()):
            n_max_processes = n_max_processes // 2

    # Compute the total number of epochs
    logger.info("Executing tasks...")
    n_epochs_done = sum(tasks[h].n_epochs for h in tasks_done)
    n_total_epochs = sum(task.n_epochs for _, task in task_list)
    try:
        with tqdm.tqdm(total=n_total_epochs,
                       unit="epochs",
                       dynamic_ncols=True,
                       smoothing=0.1,
                       miniters=1,
                       initial=n_epochs_done) as pbar:
            while len(tasks_done) < n_tasks:
                # Launch new processes
                while (len(processes) < n_max_processes) and (task_ptr <
                                                              n_tasks):
                    # Fetch information about the current task and advance the task
                    # pointer
                    task_hash, task_descr = task_list[task_ptr]
                    task_ptr += 1
                    if task_hash in tasks_done:
                        continue

                    # Launch the task as a sub-process
                    debug = args.debug if hasattr(args, "debug") else False
                    descr = {task_hash: dataclasses.asdict(task_descr)}
                    process_data = parent_launch_task(debug, "child", "--tar",
                                                      args.tar, "--descr",
                                                      json.dumps(descr))
                    processes[task_hash] = process_data

                # Read from all sub-processes. This function will timeout at
                # least every 0.1 seconds so we can update the progress bar.
                for task_done in parent_wait_task(processes):
                    tasks_done.add(task_done)

                # Compute the number of epochs that are done
                n_epochs_done = sum(tasks[h].n_epochs for h in tasks_done)

                # Add the number of epochs already processed in each sub-process
                for process_descr in processes.values():
                    n_epochs_done += process_descr["n_epochs_done"]

                # Advance the progress bar
                pbar.update(n_epochs_done - pbar.n)
    finally:
        # Kill all remaining child processes
        for process_data in processes.values():
            parent_kill_child(process_data)


###############################################################################
# Child process functionality                                                 #
###############################################################################


def child_process_task(args, task_hash, task):
    import h5py

    # Create the common random number generator
    rng = np.random.RandomState(task.seed)

    # Instantiate all task components
    modules = collector.collect_modules()
    task_dict = dataclasses.asdict(task)
    classes_dict = {}
    for key in [
            "optimizer_name", "dataset_name", "network_name",
            "decoder_learner_name", "encoder_learner_name"
    ]:
        # Fetch the manifest corresponding to 
        module_key = "_".join(key.split("_")[:-1]) + "s"
        class_key = "_".join(key.split("_")[:-1])
        params_key = class_key + "_params"

        # Ignore empty task dictionary entries
        if task_dict[key] == "":
            classes_dict[class_key] = None
            continue

        # Fetch the manifest for the specified class
        manifest = modules[module_key][task_dict[key]]

        # Assemble the final component parameters; in particular, resolve "@ref"
        # references to classes defined in the manifest parameters
        params = {
            "rng": rng
        }
        for param_key, param_value in task_dict[params_key].items():
            if isinstance(param_value, dict) and (len(param_value) == 1) and ("@ref" in param_value):
                params[param_key] = manifest.params[param_key][param_value["@ref"]]
            else:
                params[param_key] = param_value

        # Instantiate the class. We'll need to pass some dimensionalities to all
        # classes
        if class_key == "network":
            params["n_dim_in"] = classes_dict["dataset"].n_dim_in
            params["n_dim_hidden"] = task.n_dim_hidden
        elif class_key == "decoder_learner":
            params["n_dim_hidden"] = classes_dict["network"].n_dim_hidden
            params["n_dim_out"] = classes_dict["dataset"].n_dim_out
        elif class_key == "encoder_learner":
            params["n_dim_in"] = classes_dict["dataset"].n_dim_out
            params["n_dim_hidden"] = classes_dict["network"].n_dim_hidden
            params["n_dim_out"] = classes_dict["dataset"].n_dim_out
        classes_dict[class_key] = manifest.ctor(**params)

    # Execute the task. Tell the parent process about the execution progress
    # by writing a JSON message to standard out.
    benchmark_kwargs = {
        **classes_dict,
        "rng": rng,
        "batch_size": task.batch_size,
        "sequential": task.sequential,
        "n_epochs": task.n_epochs,
        "compute_test_error": True,
        "progress": benchmark.print_json_progress,
    }
    benchmark_result = benchmark.run_single_trial(**benchmark_kwargs)

    # Make sure the target directory exists
    os.makedirs(args.tar, exist_ok=True)

    # Create the output file. Write to a temporary file first, then move that
    # file to the target.
    outfile = os.path.join(
        args.tar, "encoder_learning_benchmarks_{}.h5".format(task_hash))
    tmpfile = "." + outfile + ".tmp"
    with h5py.File(tmpfile, 'w') as f:
        # Store the task configuration
        f.attrs["task"] = json.dumps(task_dict)

        # Store the benchmark result
        for key, value in benchmark_result.items():
            f.create_dataset(key, data=value)

    # This is an atomic operation on POSIX systems
    os.rename(tmpfile, outfile)


###############################################################################
# Main entry points                                                           #
###############################################################################


def main_parent():
    # Construct the parser
    parser = argparse.ArgumentParser(
        description="Runs the encoder learning benchmark")
    parser.add_argument(
        "--n_repeat",
        help="Number of repetitions for each task",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--n_partitions",
        help="Number of partitions the tasks should be divided into",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--partitions",
        help="Comma separated list of partitions (indexed starting with one)",
        type=str,
        default="*",
    )
    parser.add_argument(
        "--tar",
        help="Directory to which the output files should be written",
        type=str,
        default="out")
    parser.add_argument("--debug",
                        help="Enables assertions in the child processes.",
                        action='store_const',
                        const=True)

    # Parse the command line arguments
    args = parser.parse_args()

    # Make sure the number of repetitions is valid
    if not args.n_repeat > 0:
        logger.error("n_repeat must be larger than zero")
        return 1

    # Parse the partition information
    if not args.n_partitions > 0:
        logger.error("n_partitions must be larger than zero")
        return 1
    try:
        if args.partitions == "*":
            args.partitions = set(range(1, args.n_partitions + 1))
        else:
            args.partitions = set(map(int, args.partitions.split(",")))
            for p in args.partitions:
                if (p <= 0) or (p > args.n_partitions):
                    logger.error("Invalid partition index {}".format(p))
                    return 1
    except ValueError:
        logger.error("Error while parsing the partition descriptor")
        return 1
    logger.info("Number of partitions: {}".format(args.n_partitions))
    logger.info("Partitions processed on this node: {}".format(", ".join(
        map(str, args.partitions))))

    # Assemble all tasks
    logger.info("Collecting tasks...")
    tasks = collector.collect_tasks(n_repeat=args.n_repeat)

    # Write the task hashes to a manifest file in the target directory. This way
    # we'll always know which files belong to the most recent experiment run.
    logger.info("Writing manifest...")
    os.makedirs(args.tar, exist_ok=True)
    with open(os.path.join(args.tar, "manifest.json"), 'w') as f:
        json.dump(
            {
                "tasks": {
                    task_hash: dataclasses.asdict(task)
                    for task_hash, task in tasks.items()
                },
            },
            f,
            indent=4,
            sort_keys=True)

    # Extract the selected partitions
    part_bnds = np.linspace(0, len(tasks), args.n_partitions + 1, dtype=np.int)
    part_ptr = 0
    tasks_sel = {}
    for i, (key, value) in enumerate(tasks.items()):
        if (i >= part_bnds[part_ptr]) and (i < part_bnds[part_ptr + 1]):
            if (part_ptr + 1) in args.partitions:
                tasks_sel[key] = value
        if i == part_bnds[part_ptr + 1] - 1:
            part_ptr += 1

    # Run the main loop that takes care of executing the individual child
    # processes
    parent_run_main_loop(args, tasks_sel)

    return 0


def main_child():
    # Construct the parser
    parser = argparse.ArgumentParser(
        description="Sub-process responsible for running a single experiment")
    parser.add_argument("--tar",
                        help="Target directory",
                        type=str,
                        required=True)
    parser.add_argument("--descr",
                        help="Task descriptor",
                        type=str,
                        required=True)

    # Parse the command line arguments
    args = parser.parse_args(sys.argv[2:])

    # Iterate over all tasks
    descr = json.loads(args.descr)
    for task_hash, task_dict in descr.items():
        # Reconstruct the task descriptor
        task = common.TaskDescriptor(**task_dict)

        # Make sure the task hash can be reconstructed
        task_hash_rec = collector.compute_task_hash(task)
        if task_hash_rec != task_hash:
            raise RuntimeError(
                "Could not validate task hash. This is caused by the source "
                "code version changing while experiments are running. Please "
                "re-start the top-level task runner.")

        # Execute the actual task
        child_process_task(args, task_hash, task)

    return 0


def main():
    # Enable logging
    logging.basicConfig(level=logging.DEBUG)

    # Decide whether this is the parant process or the child task runner
    if len(sys.argv) > 1:
        if sys.argv[1] == "child":
            return main_child()
    return main_parent()


if __name__ == "__main__":
    sys.exit(main())

