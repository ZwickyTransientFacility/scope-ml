#!/usr/bin/env python
import pathlib
import argparse
import pandas as pd
from datetime import timedelta
import json
import matplotlib.pyplot as plt
import os
from datetime import datetime

BASE_DIR = pathlib.Path.cwd()
plt.rcParams["font.size"] = 16


def get_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--logs-path",
        type=str,
        default="generated_features_new/logs",
        help="path (from base_dir) to slurm logs",
    )
    parser.add_argument(
        "--logs-name-pattern",
        type=str,
        default="generate_features_new",
        help="common naming convention for slurm logs (e.g. generate_features)",
    )
    parser.add_argument(
        "--logs-suffix",
        type=str,
        default="out",
        help="suffix for log files",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="runtime_output",
        help="path (from base_dir) + prefix for output file",
    )
    parser.add_argument(
        "--plot-name",
        type=str,
        default="quad_runtime_hist",
        help="name of histogram plot (saved in base_dir)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2023-12-27",
        help="Earliest date to include in log search [YYYY-MM-DD]",
    )

    return parser


def analyze_logs(
    logs_path="generated_features_new/logs",
    logs_name_pattern="generate_features_new",
    logs_suffix="out",
    output_prefix="runtime_output",
    plot_name="quad_runtime_hist",
    start_date="2023-12-27",
):
    """
    analyze slurm logs from feature generation to quantify runtime
    """

    logs_path = BASE_DIR / logs_path
    log_files = logs_path.glob(f"{logs_name_pattern}_[0-9]*_[0-9]*.{logs_suffix}")
    log_files = [x for x in log_files]

    if len(log_files) == 0:
        raise ValueError("Could not find any log files.")

    start_date = datetime.strptime(start_date, '%Y-%m-%d')

    results_dct = {}
    log_count = 0
    done_count = 0
    for log_file in log_files:

        mod_time = os.path.getmtime(log_file)
        mod_datetime = datetime.utcfromtimestamp(mod_time)

        if mod_datetime > start_date:
            log_count += 1
            job_id = str(log_file).split("_")[-2]

            try:
                log_output = pd.read_table(log_file, header=None)
            except pd.errors.EmptyDataError:
                # Some logs may be empty if the instance just began
                continue

            try:
                n_sources_start = int(log_output.iloc[2].values[0].split()[1])
            except IndexError:
                # Some logs may not yet have initial results if instance just began
                continue

            try:
                n_sources_end = int(log_output.iloc[-2].values[0].split()[3])
                runtime = float(log_output.iloc[-1].values[0].split()[3])
            except IndexError:
                # Some logs may not yet have final results if the instance is still running
                continue

            delta = timedelta(
                seconds=runtime,
            )
            total_seconds = delta.total_seconds()

            results_dct[int(job_id)] = {
                "n_sources_start": n_sources_start,
                "n_sources_end": n_sources_end,
                "runtime_seconds": total_seconds,
                "seconds_per_source_start": total_seconds / n_sources_start,
            }

            done_count += 1

    print(f"Found {log_count} logs modified after {start_date}.")
    # make histogram
    sec_per_lc_start = [x['seconds_per_source_start'] for x in results_dct.values()]

    fig = plt.figure(figsize=(7, 7))
    plt.hist(sec_per_lc_start)
    plt.xlabel("Quadrant runtime [sec per lightcurve]")
    plt.ylabel("Count")
    fig.savefig(BASE_DIR / f"{plot_name}_{logs_name_pattern}.pdf", bbox_inches='tight')
    print(f"Saved plot to {BASE_DIR}/{plot_name}_{logs_name_pattern}.pdf")

    with open(BASE_DIR / f"{output_prefix}_{logs_name_pattern}.json", "w") as f:
        json.dump(results_dct, f)
    print(
        f"Wrote results for {done_count} completed jobs to {BASE_DIR}/{output_prefix}_{logs_name_pattern}.json"
    )


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()
    analyze_logs(**vars(args))
