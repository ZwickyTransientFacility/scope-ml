#!/usr/bin/env python
import pathlib
import argparse
import pandas as pd
import warnings
from datetime import timedelta
import json

BASE_DIR = pathlib.Path(__file__).parent.parent.absolute()


def get_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--job-ids-path",
        type=str,
        help="path (from base_dir) to file containing slurm job ids",
    )
    parser.add_argument(
        "--logs-path",
        type=str,
        help="path (from base_dir) to slurm logs",
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
        "--output-path",
        type=str,
        default="runtime_output.json",
        help="path (from base_dir) to output file",
    )

    return parser


def main(
    job_ids_path,
    logs_path,
    logs_name_pattern="",
    logs_suffix="out",
    output_path="runtime_output.json",
):
    job_ids = pd.read_table(BASE_DIR / job_ids_path, header=None)

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

    with open(BASE_DIR / output_path, "w") as f:
        json.dump(results_dct, f)
    print(f"Wrote results to {BASE_DIR / output_path}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(**vars(args))
