#!/usr/bin/env python
import os
import pathlib
import argparse
import yaml

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
        default='inference',
        help="Directory name for training slurm scripts",
    )
    parser.add_argument(
        "--scriptname",
        type=str,
        default='get_all_preds.sh',
        help="inference script name",
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
        "--group_name",
        type=str,
        default=None,
        help="Group name",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default='dnn',
        help="dnn or xgb",
    )

    args = parser.parse_args()
    return args


def filter_completed(fields, algorithm):

    for field in fields:
        searchDir = BASE_DIR / f'preds_{algorithm}' / f'field_{field}'
        searchDir.mkdir(parents=True, exist_ok=True)
        has_files = any(searchDir.iterdir())

        if has_files:
            fields.remove(field)

    print('Models remaining to run: ', len(fields))
    return fields


def run_job(field):
    sbatchstr = f"sbatch --export=FID={field} {subfile}"
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
    reset_running = args.reset_running

    slurmDir = str(BASE_DIR / dirname)
    scriptpath = str(BASE_DIR / scriptname)

    fields = config['inference']['fields_to_run']
    group = args.group_name
    algorithm = args.algorithm

    subDir = os.path.join(slurmDir, filetype)
    subfile = os.path.join(subDir, '%s.sub' % filetype)

    fields_remaining = filter_completed(
        fields,
        algorithm,
    )
    njobs = len(fields_remaining)

    for field in fields_remaining:
        # Only run jobs from tags_remaining_to_run list
        run_job(
            field,
        )

    os.system(f"squeue -u {args.user}")

    print('The final jobs in the run have been queued - breaking loop.')
    print('Run "squeue -u <username>" to check status of remaining jobs.')
