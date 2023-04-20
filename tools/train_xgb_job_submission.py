#!/usr/bin/env python
import os
import pathlib

# import time
import argparse

# import pandas as pd
# import numpy as np
import yaml
from train_xgb_slurm import parse_training_script


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
        default='xgb_training',
        help="Directory name for training slurm scripts",
    )
    parser.add_argument(
        "--scriptname",
        type=str,
        default='train_xgb.sh',
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
    # parser.add_argument(
    #     "--doSubmit",
    #     action="store_true",
    #     default=False,
    #     help="If set, start jobs with limits specified by --max_instances and --wait_time_minutes",
    # )
    # parser.add_argument(
    #     "--max_instances",
    #     type=int,
    #     default=64,
    #     help="Max number of instances to run in parallel",
    # )
    # parser.add_argument(
    #     "--wait_time_minutes",
    #     type=float,
    #     default=5.0,
    #     help="Time to wait between job status checks",
    # )
    # parser.add_argument(
    #     "--doSubmitLoop",
    #     action="store_true",
    #     default=False,
    #     help="If set, loop to initiate instances until out of jobs (hard on Kowalski)",
    # )
    # parser.add_argument(
    #     "--runParallel",
    #     action="store_true",
    #     default=False,
    #     help="If set, run jobs in parallel using slurm. Otherwise, run in series on a single instance.",
    # )

    args = parser.parse_args()
    return args


# def filter_completed(df, resultsDir, filename):

#     start_time = time.time()

#     tbd = []
#     for ii, (_, row) in enumerate(df.iterrows()):

#         field, ccd, quadrant = int(row["field"]), int(row["ccd"]), int(row["quadrant"])

#         resultsDir_iter = resultsDir + f"/field_{field}"
#         filename_iter = filename + f"_field_{field}_ccd_{ccd}_quad_{quadrant}"
#         filename_iter += '.parquet'
#         filepath = os.path.join(resultsDir_iter, filename_iter)

#         if not os.path.isfile(filepath):
#             tbd.append(ii)
#         else:
#             print(filepath)
#     df = df.iloc[tbd]
#     df.reset_index(inplace=True, drop=True)

#     end_time = time.time()
#     print('Checking completed jobs took %.2f seconds' % (end_time - start_time))

#     return df


def run_job(tag):
    sbatchstr = f"sbatch --export=QID={tag} {subfile}"
    print(sbatchstr)
    # os.system(sbatchstr)


if __name__ == '__main__':
    # Parse command line
    args = parse_commandline()

    dir_path = os.path.dirname(os.path.realpath(__file__))

    scriptname = args.scriptname
    filetype = args.filetype
    dirname = args.dirname

    slurmDir = str(BASE_DIR / dirname)
    scriptpath = str(BASE_DIR / scriptname)

    tags, group, _ = parse_training_script(scriptpath)

    subDir = os.path.join(slurmDir, filetype)
    subfile = os.path.join(subDir, '%s.sub' % filetype)

    # lines = [line.rstrip('\n') for line in open(subfile)]
    # jobline = lines[-1]
    # joblineSplit = list(filter(None, jobline.split("algorithm")[-1].split(" ")))
    # algorithm = joblineSplit[0]

    # quadrantfile = os.path.join(subDir, '%s.dat' % filetype)

    # names = ["job_number", "field", "ccd", "quadrant"]

    # df_original = pd.read_csv(quadrantfile, header=None, delimiter=' ', names=names)
    # pd.set_option('display.max_columns', None)

    # if fields_to_run is not None:
    #     print(f"Running fields {fields_to_run}.")
    #     field_mask = np.isin(df_original['field'], fields_to_run)
    #     df_filtered = df_original[field_mask].reset_index(drop=True)
    # else:
    #     df_filtered = df_original

    # df = filter_completed(df_filtered, resultsDir, filename)
    # njobs = len(df)
    # print('%d jobs remaining...' % njobs)

    # if args.doSubmit:
    #     counter = 0
    #     status_njobs = njobs
    #     diff_njobs = 0
    #     size = args.max_instances
    #     final_round = False
    #     while njobs > 0:
    #         # Limit number of parallel jobs for Kowalski stability
    #         if counter < args.max_instances:
    #             # Avoid choosing same index multiple times in one round of jobs
    #             rng = np.random.default_rng()
    #             quadrant_indices = rng.choice(njobs, size=size, replace=False)

    #             for quadrant_index in quadrant_indices:
    #                 run_job(
    #                     df,
    #                     quadrant_index,
    #                     resultsDir,
    #                     filename,
    #                     runParallel=args.runParallel,
    #                 )
    #                 counter += 1

    #             print(f"Instances available: {args.max_instances - counter}")

    #             if final_round:
    #                 print('The final jobs in the run have been queued - breaking loop.')
    #                 print(
    #                     'Run "squeue -u <username>" to check status of remaining jobs.'
    #                 )
    #                 break
    #         else:
    #             # Wait between status checks
    #             os.system(f"squeue -u {args.user}")
    #             print(f"Waiting {args.wait_time_minutes} minutes until next check...")
    #             time.sleep(args.wait_time_minutes * 60)

    #             # Filter completed runs, redefine njobs
    #             df = filter_completed(df, resultsDir, filename)
    #             njobs = len(df)
    #             print('%d jobs remaining...' % njobs)

    #             # Compute difference in njobs to count available instances
    #             diff_njobs = status_njobs - njobs
    #             status_njobs = njobs

    #             # Decrease counter if jobs have finished
    #             counter -= diff_njobs

    #             # Define size of the next quadrant_indices array
    #             size = np.min([args.max_instances - counter, njobs])
    #             # Signal to stop looping when the last set of jobs is queued
    #             if size == njobs:
    #                 final_round = True

    confirm = input(
        "Warning: there is no limit on the number of jobs submitted by this script. Continue? (yes/no): "
    )
    if confirm in ['yes', 'Yes', 'YES']:
        for tag in tags:
            run_job(tag)
    else:
        print('Canceled loop submission.')
