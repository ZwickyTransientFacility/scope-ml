#!/usr/bin/env python
import os
import pathlib
import time
import argparse
import pandas as pd
import numpy as np
import subprocess
from scope.utils import parse_load_config


BASE_DIR = pathlib.Path.cwd()
config = parse_load_config()

fields_to_run = config['feature_generation']['fields_to_run']
path_to_features = config['feature_generation']['path_to_features']
if path_to_features is not None:
    BASE_DIR = pathlib.Path(path_to_features)


def get_parser():
    """
    Parse the options given on the command-line.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dirname",
        type=str,
        default='generated_features',
        help="Directory name for generated features",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default='gen_features',
        help="Prefix for generated feature file",
    )
    parser.add_argument(
        "-f", "--filetype", default="slurm", help="Type of submission file"
    )
    parser.add_argument(
        "--doSubmit",
        action="store_true",
        default=False,
        help="If set, start jobs with limits specified by --max-instances and --wait-time-minutes",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=20,
        help="Max number of instances to run in parallel",
    )
    parser.add_argument(
        "--wait-time-minutes",
        type=float,
        default=5.0,
        help="Time to wait between job status checks",
    )
    parser.add_argument(
        "--doSubmitLoop",
        action="store_true",
        default=False,
        help="If set, loop to initiate instances until out of jobs (hard on Kowalski)",
    )
    parser.add_argument(
        "--runParallel",
        action="store_true",
        default=False,
        help="If set, run jobs in parallel using slurm. Otherwise, run in series on a single instance.",
    )
    parser.add_argument(
        "--user",
        type=str,
        default="bhealy",
        help="HPC username",
    )
    parser.add_argument(
        "--reset-running",
        action='store_true',
        default=False,
        help="If set, reset the 'running' status of all tags",
    )
    parser.add_argument(
        "--submit-interval-minutes",
        type=float,
        default=1.0,
        help="Time to wait between job submissions (minutes)",
    )

    return parser


def filter_running(user):
    command = f"squeue -u {user}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    running_jobs_count = 0
    if result.returncode == 0:
        running_jobs = result.stdout.splitlines()
        # squeue output will always have 1 line for header
        if len(running_jobs) > 1:
            running_jobs = [x.strip().split() for x in running_jobs[1:]]
            for job in running_jobs:
                job_name = job[2]
                if "ztf_fg" in job_name:
                    running_jobs_count += 1
    else:
        print("Error executing the command. Exit code:", result.returncode)
        print("Error output:", result.stderr)
        raise ValueError()

    print(f"Identified {running_jobs_count} running jobs.")

    return running_jobs_count


def filter_completed(df, resultsDir, filename, reset_running=False):

    start_time = time.time()

    jobs_remaining_to_complete = []
    jobs_remaining_to_run = []
    for ii, (_, row) in enumerate(df.iterrows()):

        field, ccd, quadrant = int(row["field"]), int(row["ccd"]), int(row["quadrant"])

        resultsDir_iter = resultsDir + f"/field_{field}"
        filename_iter = filename + f"_field_{field}_ccd_{ccd}_quad_{quadrant}"
        filename_iter += '.parquet'

        resultsPath = pathlib.Path(resultsDir_iter)
        resultsPath.mkdir(parents=True, exist_ok=True)

        if reset_running:
            paths = [x for x in resultsPath.glob('*.running')]
            for path in paths:
                path.unlink()

        contents = [x for x in resultsPath.iterdir()]

        filepath = resultsPath / filename_iter

        has_file = filepath.is_file()
        if not has_file:
            jobs_remaining_to_complete.append(ii)

        # jobs_remaining_to_run is a subset of jobs_remaining_to_complete
        # (A job could be queued but not yet finished: removed from 'to_run' but not from 'to_complete')
        running = (
            np.sum(
                [
                    (
                        x.name
                        == f'{filename}_field_{field}_ccd_{ccd}_quad_{quadrant}.running'
                    )
                    for x in contents
                ]
            )
            > 0
        ) | (has_file)
        if not running:
            jobs_remaining_to_run.append(ii)

    df_toComplete = df.iloc[jobs_remaining_to_complete]
    df_toComplete.reset_index(inplace=True, drop=True)

    df_toRun = df.iloc[jobs_remaining_to_run]
    df_toRun.reset_index(inplace=True, drop=True)

    print('Jobs remaining to complete: ', len(jobs_remaining_to_complete))
    print('Jobs remaining to run: ', len(jobs_remaining_to_run))

    end_time = time.time()
    print('Checking completed jobs took %.2f seconds' % (end_time - start_time))

    return df_toComplete, df_toRun


def run_job(
    df,
    quadrant_index,
    resultsDir,
    filename,
    qsubfile,
    jobline,
    runParallel=False,
    submit_interval_minutes=1.0,
):
    # Don't hit kowalski with too many simultaneous queries
    time.sleep(submit_interval_minutes * 60)

    row = df.iloc[quadrant_index]
    field, ccd, quadrant = int(row["field"]), int(row["ccd"]), int(row["quadrant"])

    resultsDir += f"/field_{field}"
    filename += f"_field_{field}_ccd_{ccd}_quad_{quadrant}"

    resultsPath = pathlib.Path(resultsDir)
    os.system(f'touch {str(resultsPath)}/{filename}.running')

    filename += '.parquet'
    filepath = resultsPath / filename

    if not os.path.isfile(filepath):
        if runParallel:
            sbatchstr = f"sbatch --export=QID={row['job_number']} {qsubfile}"
            print(sbatchstr)
            os.system(sbatchstr)
        else:
            jobstr = jobline.replace("$QID", "%d" % row["job_number"])
            print(jobstr)
            os.system(jobstr)


def main():
    # Start with 60s delay to allow previous submission job to conclude (esp. if running as cron job)
    time.sleep(60)

    # Parse command line
    parser = get_parser()
    args, _ = parser.parse_known_args()

    running_jobs_count = filter_running(args.user)

    filename = args.filename
    filetype = args.filetype
    dirname = args.dirname
    resultsDir = str(BASE_DIR / dirname)
    reset_running = args.reset_running

    qsubDir = os.path.join(resultsDir, filetype)
    if not os.path.isdir(qsubDir):
        os.makedirs(qsubDir)
    qsubfile = os.path.join(qsubDir, '%s.sub' % filetype)

    lines = [line.rstrip('\n') for line in open(qsubfile)]
    jobline = lines[-1]

    quadrantfile = os.path.join(qsubDir, '%s.dat' % filetype)

    names = ["job_number", "field", "ccd", "quadrant"]

    df_original = pd.read_csv(quadrantfile, header=None, delimiter=' ', names=names)
    pd.set_option('display.max_columns', None)

    if fields_to_run is not None:
        print(f"Running fields {fields_to_run}.")
        field_mask = np.isin(df_original['field'], fields_to_run)
        df_filtered = df_original[field_mask].reset_index(drop=True)
    else:
        df_filtered = df_original

    df_to_complete, df_to_run = filter_completed(
        df_filtered, resultsDir, filename, reset_running=reset_running
    )
    njobs = len(df_to_run)
    nchoice = njobs
    print('%d jobs remaining to complete...' % njobs)
    print('%d jobs remaining to queue...' % nchoice)

    if args.doSubmit:
        failure_count = 0
        counter = running_jobs_count
        status_njobs = len(df_to_complete)
        # Redefine max instances if fewer jobs remain
        new_max_instances = np.min([args.max_instances, nchoice])
        size = new_max_instances - counter
        final_round = False
        if size == nchoice:
            final_round = True
        while njobs > 0:
            # Limit number of parallel jobs for Kowalski stability
            if counter < new_max_instances:
                # Avoid choosing same index multiple times in one round of jobs
                rng = np.random.default_rng()
                quadrant_indices = rng.choice(nchoice, size=size, replace=False)

                for quadrant_index in quadrant_indices:
                    run_job(
                        df_to_run,
                        quadrant_index,
                        resultsDir,
                        filename,
                        qsubfile,
                        jobline,
                        runParallel=args.runParallel,
                        submit_interval_minutes=args.submit_interval_minutes,
                    )
                    counter += 1

                print(f"Instances available: {new_max_instances - counter}")

                if final_round:
                    print('The final jobs in the run have been queued; breaking loop.')
                    print(
                        'Run "squeue -u <username>" to check status of remaining jobs.'
                    )
                    print(f"{failure_count} jobs failed during full run.")
                    break
            else:
                # Wait between status checks
                print(f"Waiting {args.wait_time_minutes} minutes until next check...")
                time.sleep(args.wait_time_minutes * 60)

                # Filter completed runs, redefine njobs
                df_to_complete, df_to_run = filter_completed(
                    df_to_complete, resultsDir, filename
                )
                njobs = len(df_to_complete)
                nchoice = len(df_to_run)
                print('%d jobs remaining to complete...' % njobs)
                print('%d jobs remaining to queue...' % nchoice)

                running_jobs_count = filter_running(args.user)

                failed_this_round = 0
                n_jobs_diff = counter - running_jobs_count
                n_jobs_finished = status_njobs - njobs
                if n_jobs_finished != n_jobs_diff:
                    failed_this_round = np.abs(n_jobs_finished - n_jobs_diff)
                    failure_count += failed_this_round

                status_njobs = njobs
                counter = running_jobs_count
                # Note that if a job has failed, it will not be re-queued until
                # its quadrant's .running file is removed (or set --reset-running)

                # Define size of the next quadrant_indices array
                size = np.min([new_max_instances - counter, nchoice])
                # Signal to stop looping when the last set of jobs is queued
                if size == nchoice:
                    final_round = True

                print(
                    f"Detected {failed_this_round} failed jobs this round ({failure_count} total failures)."
                )

    elif args.doSubmitLoop:
        confirm = input(
            "Warning: setting --doSubmitLoop ignores limits on number of jobs to submit. Continue? (yes/no): "
        )
        if confirm in ['yes', 'Yes', 'YES']:
            for quadrant_index in range(njobs):
                run_job(
                    df_to_run,
                    quadrant_index,
                    resultsDir,
                    filename,
                    qsubfile,
                    jobline,
                    runParallel=args.runParallel,
                    submit_interval_minutes=args.submit_interval_minutes,
                )
        else:
            print('Canceled loop submission.')
