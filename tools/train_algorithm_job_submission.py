#!/usr/bin/env python
import os
import pathlib
import time
import argparse
import yaml
from tools.train_algorithm_slurm import parse_training_script
import numpy as np

BASE_DIR = pathlib.Path(__file__).parent.parent.absolute()

# Read config file
config_path = BASE_DIR / "config.yaml"
with open(config_path) as config_yaml:
    config = yaml.load(config_yaml, Loader=yaml.FullLoader)


def parse_commandline():
    """
    Parse the options given on the command-line.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dirname",
        type=str,
        default='training',
        help="Directory name for training slurm scripts",
    )
    parser.add_argument(
        "--scriptname",
        type=str,
        default='train_script.sh',
        help="training script name",
    )
    parser.add_argument(
        "-f", "--filetype", default="slurm", help="Type of job submission file"
    )
    parser.add_argument(
        "--user",
        type=str,
        default="bhealy",
        help="HPC username",
    )
    parser.add_argument(
        "--max_instances",
        type=int,
        default=100,
        help="Max number of instances to run in parallel",
    )
    parser.add_argument(
        "--wait_time_minutes",
        type=float,
        default=5.0,
        help="Time to wait between job status checks",
    )
    parser.add_argument(
        "--sweep",
        action='store_true',
        default=False,
        help="If set, job submission runs filter_completed in different directory",
    )

    args = parser.parse_args()
    return args


def filter_completed(tags, group, algorithm, sweep=False):
    tags_remaining = tags.copy()
    for tag in tags:
        if sweep:
            searchDir = BASE_DIR / f'models_{algorithm}' / group / 'sweeps' / tag
        else:
            searchDir = BASE_DIR / f'models_{algorithm}' / group / tag
        try:
            has_files = any(searchDir.iterdir())
        except FileNotFoundError:
            has_files = False
        if has_files:
            tags_remaining.remove(tag)

    print('Models remaining: ', len(tags_remaining))
    return tags_remaining


def run_job(tag):
    sbatchstr = f"sbatch --export=TID={tag} {subfile}"
    print(sbatchstr)
    os.system(sbatchstr)


if __name__ == '__main__':
    # Parse command line
    args = parse_commandline()

    dir_path = os.path.dirname(os.path.realpath(__file__))

    scriptname = args.scriptname
    filetype = args.filetype
    dirname = args.dirname
    sweep = args.sweep

    slurmDir = str(BASE_DIR / dirname)
    scriptpath = str(BASE_DIR / scriptname)

    tags, group, algorithm, _ = parse_training_script(scriptpath)

    subDir = os.path.join(slurmDir, filetype)
    subfile = os.path.join(subDir, '%s.sub' % filetype)

    tags_remaining = filter_completed(tags, group, algorithm, sweep=sweep)
    njobs = len(tags_remaining)

    counter = 0
    status_njobs = njobs
    diff_njobs = 0
    new_max_instances = np.min([args.max_instances, njobs])
    size = new_max_instances
    final_round = False
    if size == njobs:
        final_round = True
    while njobs > 0:
        # Limit number of parallel jobs
        for tag in tags_remaining:
            if counter < new_max_instances:
                run_job(tag)
                counter += 1

        print(f"Instances available: {new_max_instances - counter}")

        if final_round:
            print('The final jobs in the run have been queued - breaking loop.')
            print('Run "squeue -u <username>" to check status of remaining jobs.')
            break
        else:
            # Wait between status checks
            os.system(f"squeue -u {args.user}")
            print(f"Waiting {args.wait_time_minutes} minutes until next check...")
            time.sleep(args.wait_time_minutes * 60)

            # Filter completed runs, redefine njobs
            tags_remaining = filter_completed(
                tags_remaining, group, algorithm, sweep=sweep
            )
            njobs = len(tags_remaining)
            print('%d jobs remaining...' % njobs)

            # Compute difference in njobs to count available instances
            diff_njobs = status_njobs - njobs
            status_njobs = njobs

            # Decrease counter if jobs have finished
            counter -= diff_njobs

            # Define size of the next quadrant_indices array
            size = np.min([new_max_instances - counter, njobs])
            # Signal to stop looping when the last set of jobs is queued
            if size == njobs:
                final_round = True
