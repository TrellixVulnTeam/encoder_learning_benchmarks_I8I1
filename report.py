#!/usr/bin/env python3

import json
import os
import sys
import tempfile
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import shutil
import subprocess
import time
import numpy as np

import encoder_learning_benchmarks.analyze as analyze


def parse_args(*args, **kwargs):
    import argparse

    parser = argparse.ArgumentParser(
        description='Creates a report from an experiment run')
    parser.add_argument('input', type=str, help='Input file or directory')

    return parser.parse_args(*args, **kwargs)


def make_report(dir, f, pages):
    setattr(f, "writeln", lambda s: f.write(s + "\n"))
    f.writeln(r"\documentclass[a3paper]{report}")
    f.writeln(r"\usepackage[landscape,top=0.5cm,bottom=1.5cm,left=1cm,right=1cm]{geometry}")
    f.writeln(r"\usepackage{graphicx}")
    f.writeln(r"\usepackage{array}")
    f.writeln(r"\usepackage{amsmath}")
    f.writeln(r"\setlength\parindent{0pt}")
    f.writeln(r"\begin{document}")
    fig_idx = 0
    sub_pages_per_page = 2
    for page_no, (dataset, page) in enumerate(pages.items()):
        dataset = json.loads(dataset)

        if page_no > 0:
            f.writeln(r"\newpage")

        # Iterate over all network setups
        for sub_page_no, (network, sub_page) in enumerate(page.items()):
            network = json.loads(network)
            first_table_on_page = sub_page_no == 0
            if (sub_page_no > 0) and (sub_page_no % sub_pages_per_page == 0):
                f.writeln(r"\newpage")
                first_table_on_page = True
            elif (sub_page_no % sub_pages_per_page != 0):
                f.writeln(r"\\")

            # Write a page title
            f.writeln(r"\begin{center}")
            if first_table_on_page:
                f.writeln(r"{\Large " + analyze.compute_label({"dataset": dataset}) + r"\vphantom{(yM}}\\")
            f.writeln(r"{\large " + analyze.compute_label({"network": network}) + r"\vphantom{(yM}}\\")
            if first_table_on_page:
                f.writeln(r"{\includegraphics{" + "legend_{}_{}.pdf".format(page_no, sub_page_no) + r"}}")
            f.writeln(r"\end{center}")

            # Fetch the extents of the table
            n_tbls = len(sub_page)
            n_rows = max(len(x) for x in sub_page.values())
            n_cols = max(max(len(y) for y in x.values()) for x in sub_page.values())
            tbls = [[[None for _ in range(n_cols)] for _ in range(n_rows)] for _ in range(n_tbls)]

            # Collect all legend artists and legend labels
            collected_legend_artists = []
            collected_legend_labels = []

            # Render the individual graphs
            for i, (online, tbl) in enumerate(sub_page.items()):
                online = json.loads(online)
                for j, (decoder_learner, row) in enumerate(tbl.items()):
                    decoder_learner = json.loads(decoder_learner)
                    for k, (optimizer, data) in enumerate(row.items()):
                        optimizer = json.loads(optimizer)
                        content = ""
                        fig, ax = plt.subplots(figsize=(3.125, 1.3))
                        try:
                            _, _, legend_artists, legend_labels = analyze.plot_benchmark_data(data, ax)
                            collected_legend_artists += legend_artists
                            collected_legend_labels += legend_labels
                            plt.tight_layout()
                            fn = "plt_{}_{}_{}_{}.pdf".format(fig_idx, i, j, k)
                            fig_idx += 1
                            fig.savefig(os.path.join(dir, fn), transparent=True, pad_inches=0)

                            # Include the graphics
                            content = r"\includegraphics{" + fn + r"} \footnotesize "
                        except ValueError:
                            pass
                        finally:
                            plt.close(fig)

                        # List the errors
                        info = analyze.compute_benchmark_data_info(data)
                        content += r"{\begin{align*}" + "\n"
                        for l, (task_hashes, task_data) in enumerate(data):
                            if l > 0:
                                content += r"\\[-0.15cm]" + "\n"
                            valid = ~np.isnan(np.asarray(task_data["err_test"]))
                            if not np.any(valid):
                                err_test = r"\mathrm{N/A}"
                            else:
                                err_test = "{:0.3f}".format(np.nanmedian(task_data["err_test"]))
                            n_valid = np.sum(valid)
                            content += "\t" + r"\text{" + info["labels"][l] + r": } & " + "E_\mathrm{{test}} = {}, n = {}".format(err_test, n_valid)
                        content += r"\end{align*}}" + "\n"

                        tbls[i][j][k] = (online, decoder_learner, optimizer), content

            # Render the legend
            legend_keys, legend_artists, legend_labels = set(), [], []
            for artist, label in zip(collected_legend_artists, collected_legend_labels):
                key = json.dumps((artist, label), sort_keys=True)
                if key in legend_keys:
                    continue
                legend_keys.add(key)
                legend_artists.append(mlines.Line2D([0], [0], **artist))
                legend_labels.append(label)

            fig, ax = plt.subplots(figsize=(5, 0.2))
            ax.legend(legend_artists, legend_labels, ncol=len(legend_artists))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            fn = "legend_{}_{}.pdf".format(page_no, sub_page_no)
            fig.savefig(os.path.join(dir, fn), transparent=True, bbox_inches='tight', pad_inches=0)

            # Render the table
            f.writeln(r"\begin{tabular}{" + " ".join("r " + " ".join([r"p{3.13in}"] * n_cols) for _ in range(n_tbls)) + r"}")

            if first_table_on_page:
                # Render the top-level column labels
                for i in range(n_tbls):
                    (online, _, _), _ = tbls[i][0][0]
                    f.write(r" & \multicolumn{" + str(n_cols) + "}{c}{")
                    f.write(r"\textsc{" + ("Online" if online else "Offline") + r"}")
                    f.write(r"}")
                    if i + 1 < n_tbls:
                        f.write(r" &")
                f.writeln(r" \\")

                # Render the second-level column labels
                for i in range(n_tbls):
                    f.write(" & ")
                    for k in range(n_cols):
                        (_, _, optimizer), _ = tbls[i][0][k]
                        lbl = analyze.compute_label({"optimizer": optimizer})
                        f.write(r"\textbf{" + lbl + "}")
                        if k + 1 < n_cols:
                            f.write(r" & ")
                    if i + 1 < n_tbls:
                        f.write(r" & ")
                f.writeln(r" \\")

            # Render the remaining table
            for j in range(n_rows):
                for i in range(n_tbls):
                    (_, decoder_learner, _), _ = tbls[i][j][0]
                    lbl = analyze.compute_label({"decoder_learner": decoder_learner})
                    f.write(r"\raisebox{0.75in}{\textbf{" + lbl + "}}" + r" & ")
                    for k in range(n_cols):
                        (_, _, _), content = tbls[i][j][k]
                        f.write(content)
                        if k + 1 < n_cols:
                            f.write(r" & ")
                    if i + 1 < n_tbls:
                        f.write(r" & ")
                f.writeln(r" \\")

            f.writeln(r"\end{tabular}")



    f.writeln(r"\end{document}")


def main():
    # Read the command line arguments
    args = parse_args()

    # Read the input data and merge runs of the same experiment
    data = analyze.load_benchmark_data(args.input)
    data = analyze.merge_benchmark_data(data, {"seed": None})
    data = analyze.sort_benchmark_data(data)

    # Helper function used to iterate over the datasets in a particular order
    def iterate(data, keys, values=[]):
        if len(keys) == 0:
            yield data, values
        else:
            sets = analyze.extract_parameter_sets(data, merge=False)
            sets_clean = analyze.remove_constants_from_parameter_sets(sets)
            for value in sorted(sets_clean[keys[0]], key=lambda x: json.dumps(x, sort_keys=True)):
                data_flt = analyze.filter_benchmark_data(
                    data, {keys[0]: value})
                for elem in iterate(data_flt, keys[1:], values + [value]):
                    yield elem

    def insert(dict_, keys, value):
        for i, key in enumerate(keys):
            key = json.dumps(key, sort_keys=True)
            if not key in dict_:
                dict_[key] = {} if i + 1 < len(keys) else value
            dict_ = dict_[key]

    # Sort the datasets into a set of pages
    pages = {}
    for data_flt, (sequential, dataset, network, decoder_learner,
                   optimizer) in iterate(data, [
                       "sequential", "dataset", "network", "decoder_learner",
                       "optimizer"
                   ]):
        # Sort the plots into the page by dataset,network,sequential/decoder_learner/optimizer
        insert(pages,
               [dataset, network, sequential, decoder_learner, optimizer],
               data_flt)

    # Iterate over the pages; create the individual plots and combine them into
    # PDF using LaTeX
#    os.makedirs("report_tmp", exist_ok=True)
#    dir = "report_tmp"
    with tempfile.TemporaryDirectory() as dir:
        # Build the report
        with open(os.path.join(dir, "report.tex"), "w") as f:
            make_report(dir, f, pages)

        # Render the LaTeX document
        if subprocess.run(["pdflatex", "-interaction=nonstopmode", "report.tex"], cwd=dir).returncode == 0:
            shutil.move(os.path.join(dir, "report.pdf"), "report.pdf")
        else:
            print("LaTeX subproces failed")
            sys.exit(1)


if __name__ == "__main__":
    main()
