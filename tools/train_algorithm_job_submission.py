#!/usr/bin/env python
import os
import pathlib
import time
import argparse
from tools.train_algorithm_slurm import parse_training_script
import numpy as np
import datetime
from scope.utils import parse_load_config


BASE_DIR = pathlib.Path.cwd()
config = parse_load_config()


def get_parser():
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
        "--max-instances",
        type=int,
        default=100,
        help="Max number of instances to run in parallel",
    )
    parser.add_argument(
        "--wait-time-minutes",
        type=float,
        default=5.0,
        help="Time to wait between job status checks (minutes)",
    )
    parser.add_argument(
        "--submit-interval-seconds",
        type=float,
        default=5.0,
        help="Time to wait between job submissions (seconds)",
    )
    parser.add_argument(
        "--sweep",
        action='store_true',
        default=False,
        help="If set, job submission runs filter_completed in different directory",
    )
    parser.add_argument(
        "--reset-running",
        action='store_true',
        default=False,
        help="If set, reset the 'running' status of all tags",
    )
    parser.add_argument(
        "--group-name",
        type=str,
        default=None,
        help="Group name (if different from name in training script)",
    )

    return parser


def filter_completed(tags, group, algorithm, sweep=False, reset_running=False):
    # Using two lists of tags allows us to distinguish running models from completed ones, minimizing computational waste
    # tags_remaining_to_complete informs when the counter should be decreased
    # tags_remaining_to_run informs which jobs to select from once the counter allows it
    tags_remaining_to_complete = tags.copy()
    tags_remaining_to_run = tags.copy()
    for tag in tags:
        try:
            if sweep:
                searchDir = BASE_DIR / f'models_{algorithm}' / group / 'sweeps' / tag
                if reset_running:
                    paths = [x for x in searchDir.glob('*.running')]
                    for path in paths:
                        path.unlink()
                has_files = any(searchDir.iterdir())
                running = has_files
                has_model = has_files
            else:
                searchDir = BASE_DIR / f'models_{algorithm}' / group / tag
                if reset_running:
                    paths = [x for x in searchDir.glob('*.running')]
                    for path in paths:
                        path.unlink()
                contents = [x for x in searchDir.iterdir()]

                # Check if hdf5 (DNN) or json (XGB) models have been saved
                if algorithm == 'dnn':
                    has_model = np.sum([x.suffix == '.h5' for x in contents]) > 0
                else:
                    has_model = np.sum([x.suffix == '.json' for x in contents]) > 0

                # tags_remaining_to_run is a subset of tags_remaining_to_complete
                # (A tag could be queued but not yet finished: removed from 'to_run' but not from 'to_complete')
                running = (np.sum([(x.suffix == '.running') for x in contents]) > 0) | (
                    has_model
                )

        except FileNotFoundError:
            has_model = False
            running = False

        if has_model:
            tags_remaining_to_complete.remove(tag)

        if running:
            tags_remaining_to_run.remove(tag)

    print('Models remaining to complete: ', len(tags_remaining_to_complete))
    print('Models remaining to run: ', len(tags_remaining_to_run))
    return tags_remaining_to_complete, tags_remaining_to_run


def run_job(
    tag,
    group,
    algorithm,
    subfile,
    submit_interval_seconds=5.0,
    sweep=False,
):
    # Don't hit WandB server with too many login attempts at once
    time.sleep(submit_interval_seconds)

    sbatchstr = f"sbatch --export=TID={tag} {subfile}"
    print(sbatchstr)
    os.system(sbatchstr)

    time_tag = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Make dummy file to register as completed
    if sweep:
        output_path = BASE_DIR / f'models_{algorithm}' / group / 'sweeps' / tag
        os.makedirs(output_path, exist_ok=True)
        os.system(f'touch {str(output_path)}/{tag}.{time_tag}.sweep.running')
    else:
        output_path = BASE_DIR / f'models_{algorithm}' / group / tag
        os.makedirs(output_path, exist_ok=True)
        os.system(f'touch {str(output_path)}/{tag}.{time_tag}.running')


def main():
    # Parse command line
    parser = get_parser()
    args, _ = parser.parse_known_args()

    scriptname = args.scriptname
    filetype = args.filetype
    dirname = args.dirname
    sweep = args.sweep
    reset_running = args.reset_running

    slurmDir = str(BASE_DIR / dirname)
    scriptpath = str(BASE_DIR / scriptname)

    tags, group, algorithm, _ = parse_training_script(scriptpath)
    if args.group_name is not None:
        group = args.group_name

    subDir = os.path.join(slurmDir, filetype)
    subfile = os.path.join(subDir, '%s.sub' % filetype)

    tags_remaining_to_complete, tags_remaining_to_run = filter_completed(
        tags, group, algorithm, sweep=sweep, reset_running=reset_running
    )
    njobs = len(tags_remaining_to_run)

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
        for tag in tags_remaining_to_run:
            # Only run jobs from tags_remaining_to_run list
            if counter < new_max_instances:
                run_job(
                    tag,
                    group,
                    algorithm,
                    subfile,
                    submit_interval_seconds=args.submit_interval_seconds,
                    sweep=sweep,
                )
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

            # Filter completed runs, redefine njobs using tags_remaininig_to_complete
            tags_remaining_to_complete, tags_remaining_to_run = filter_completed(
                tags_remaining_to_complete, group, algorithm, sweep=sweep
            )
            njobs = len(tags_remaining_to_complete)
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
