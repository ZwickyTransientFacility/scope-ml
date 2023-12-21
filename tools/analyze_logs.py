#!/usr/bin/env python
import pathlib
import argparse
import pandas as pd
import warnings
from datetime import timedelta
import json
import matplotlib.pyplot as plt

BASE_DIR = pathlib.Path(__file__).parent.parent.absolute()
plt.rcParams["font.size"] = 16


def get_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--logs-path",
        type=str,
        help="path (from base_dir) to slurm logs",
    )
    parser.add_argument(
        "--job-ids-prefix",
        type=str,
        default="job_ids",
        help="path (from base_dir) + prefix of file containing slurm job ids",
    )
    parser.add_argument(
        "--logs-name-pattern",
        type=str,
        default="",
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
        "--workflow",
        type=str,
        default="feature_generation",
        help="name of workflow",
    )

    return parser


def main(
    logs_path,
    job_ids_prefix="job_ids",
    logs_name_pattern="",
    logs_suffix="out",
    output_prefix="runtime_output",
    plot_name="quad_runtime_hist",
    workflow="feature_generation",
):
    job_ids = pd.read_table(BASE_DIR / f"{job_ids_prefix}_{workflow}.txt", header=None)

    logs_path = BASE_DIR / logs_path

    results_dct = {}
    for id_runtime_pair in job_ids.values:
        job_id, runtime = id_runtime_pair[0].split(',')

        log_path = logs_path.glob(f"{logs_name_pattern}_{job_id}_*.{logs_suffix}")
        log_path = [x for x in log_path]

        if len(log_path) > 0:
            log_output = pd.read_table(log_path[0], header=None)

            n_sources_start = int(log_output.iloc[2].values[0].split()[1])
            n_sources_end = int(log_output.iloc[-1].values[0].split()[3])

            runtime_split = runtime.split("-")
            if len(runtime_split) == 1:
                runtime_days = 0
                runtime_hms = runtime_split[0].split(":")
            else:
                runtime_days = int(runtime_split[0])
                runtime_hms = runtime_split[1].split(":")

            runtime_hours = int(runtime_hms[0])
            runtime_minutes = int(runtime_hms[1])
            runtime_seconds = int(runtime_hms[2])

            delta = timedelta(
                days=runtime_days,
                hours=runtime_hours,
                minutes=runtime_minutes,
                seconds=runtime_seconds,
            )
            total_seconds = delta.total_seconds()

            results_dct[int(job_id)] = {
                "n_sources_start": n_sources_start,
                "n_sources_end": n_sources_end,
                "runtime_seconds": total_seconds,
                "seconds_per_source_start": total_seconds / n_sources_start,
            }

        else:
            warnings.warn(f"Could not find log for job ID {job_id}")

    # make histogram
    sec_per_lc_start = [x['seconds_per_source_start'] for x in results_dct.values()]

    fig = plt.figure(figsize=(7, 7))
    plt.hist(sec_per_lc_start)
    plt.xlabel("Quadrant runtime [sec per lightcurve]")
    plt.ylabel("Count")
    fig.savefig(BASE_DIR / f"{plot_name}_{workflow}.pdf", bbox_inches='tight')
    print(f"Saved plot to {BASE_DIR}/{plot_name}_{workflow}.pdf")

    with open(BASE_DIR / f"{output_prefix}_{workflow}.json", "w") as f:
        json.dump(results_dct, f)
    print(f"Wrote results to {BASE_DIR}/{output_prefix}_{workflow}.json")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(**vars(args))
