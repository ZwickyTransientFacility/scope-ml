#!/usr/bin/env python
import argparse
import pathlib
import yaml
import os

# TODO: enable GPU-accelerated training

BASE_DIR = pathlib.Path(__file__).parent.parent.absolute()

config_path = BASE_DIR / "config.yaml"
with open(config_path) as config_yaml:
    config = yaml.load(config_yaml, Loader=yaml.FullLoader)


def parse_training_script(script_path):
    with open(script_path, 'r') as f:
        lines = f.readlines()

    tags = []
    for line in lines:
        if 'scope.py train' in line:
            line_info = line.removeprefix('./scope.py train').split()
            for arg in line_info:
                if '--tag' in arg:
                    tag = arg.split('=')[1]
                    tags += [tag]
                    line_info.remove(arg)

                if '--group' in arg:
                    group = arg.split('=')[1]
                else:
                    group = 'experiment'

    return tags, group, line_info


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scriptname",
        type=str,
        default='train_xgb.sh',
        help="Training script filename within scope directory",
    )
    parser.add_argument(
        "--dirname",
        type=str,
        default='xgb_training',
        help="Directory name for slurm scripts/logs",
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default='train_xgb',
        help="job name",
    )
    parser.add_argument(
        "--cluster_name",
        type=str,
        default='Expanse',
        help="Name of HPC cluster",
    )
    parser.add_argument(
        "--partition_type",
        type=str,
        default='shared',
        help="Partition name to request for computing",
    )
    parser.add_argument(
        "--submit_partition_type",
        type=str,
        default='shared',
        help="Partition name to request for job submission",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Number of nodes to request for computing",
    )
    parser.add_argument(
        "--Ncore",
        default=10,
        type=int,
        help="number of cores to request for computing",
    )
    parser.add_argument(
        "--submit_nodes",
        type=int,
        default=1,
        help="Number of nodes to request for job submission",
    )
    parser.add_argument(
        "--submit_Ncore",
        default=1,
        type=int,
        help="number of cores to request for job submission",
    )
    # parser.add_argument(
    #     "--gpus",
    #     type=int,
    #     default=1,
    #     help="Number of GPUs to request",
    # )
    parser.add_argument(
        "--memory_GB",
        type=int,
        default=16,
        help="Memory allocation to request for computing",
    )
    parser.add_argument(
        "--submit_memory_GB",
        type=int,
        default=16,
        help="Memory allocation to request for job submission",
    )
    parser.add_argument(
        "--time",
        type=str,
        default='12:00:00',
        help="Walltime for instance",
    )
    parser.add_argument(
        "--mail_user",
        type=str,
        default='healyb@umn.edu',
        help="contact email address",
    )
    parser.add_argument(
        "--account_name",
        type=str,
        default='umn131',
        help="Name of account with current HPC allocation",
    )
    parser.add_argument(
        "--python_env_name",
        type=str,
        default='scope-env',
        help="Name of python environment to activate",
    )
    parser.add_argument(
        "--generateTrainingFile",
        action='store_true',
        default=False,
        help="if set, generate a list of labels and job numbers, save to slurm.dat",
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

    args = parser.parse_args()

    scriptname = args.scriptname
    dirname = args.dirname

    script_path = BASE_DIR / scriptname
    _, group, line_info = parse_training_script(script_path)

    dirpath = BASE_DIR / dirname
    os.makedirs(dirpath, exist_ok=True)

    slurmDir = os.path.join(dirpath, 'slurm')
    os.makedirs(slurmDir, exist_ok=True)

    logsDir = os.path.join(dirpath, 'logs')
    os.makedirs(logsDir, exist_ok=True)

    # Main script to run feature generation
    fid = open(os.path.join(slurmDir, 'slurm.sub'), 'w')
    fid.write('#!/bin/bash\n')
    fid.write(f'#SBATCH --job-name={args.job_name}.job\n')
    fid.write(f'#SBATCH --output=../logs/{args.job_name}_%A_%a.out\n')
    fid.write(f'#SBATCH --error=../logs/{args.job_name}_%A_%a.err\n')
    fid.write(f'#SBATCH -p {args.partition_type}\n')
    fid.write(f'#SBATCH --nodes {args.nodes}\n')
    fid.write(f'#SBATCH --ntasks-per-node {args.Ncore}\n')
    # fid.write(f'#SBATCH --gpus {args.gpus}\n')
    fid.write(f'#SBATCH --mem {args.memory_GB}G\n')
    fid.write(f'#SBATCH --time={args.time}\n')
    fid.write('#SBATCH --mail-type=ALL\n')
    fid.write(f'#SBATCH --mail-user={args.mail_user}\n')
    fid.write(f'#SBATCH -A {args.account_name}\n')

    if args.cluster_name in ['Expanse', 'expanse', 'EXPANSE']:
        fid.write('module purge\n')
        # fid.write('module add gpu\n')
        # fid.write('module add cuda\n')
        fid.write(f'source activate {args.python_env_name}\n')

    fid.write(
        str(BASE_DIR / 'scope.py train ') + "--tag=$TID " + " ".join(line_info) + '\n'
    )
    fid.close()

    # Secondary script to manage job submission using train_xgb_job_submission.py
    # (Python code can also be run interactively)
    fid = open(os.path.join(slurmDir, 'slurm_submission.sub'), 'w')
    fid.write('#!/bin/bash\n')
    fid.write(f'#SBATCH --job-name={args.job_name}_submit.job\n')
    fid.write(f'#SBATCH --output=../logs/{args.job_name}_submit_%A_%a.out\n')
    fid.write(f'#SBATCH --error=../logs/{args.job_name}_submit_%A_%a.err\n')
    fid.write(f'#SBATCH -p {args.submit_partition_type}\n')
    fid.write(f'#SBATCH --nodes {args.submit_nodes}\n')
    fid.write(f'#SBATCH --ntasks-per-node {args.submit_Ncore}\n')
    fid.write(f'#SBATCH --mem {args.submit_memory_GB}G\n')
    fid.write(f'#SBATCH -A {args.account_name}\n')
    fid.write(f'#SBATCH --time={args.time}\n')
    fid.write('#SBATCH --mail-type=ALL\n')
    fid.write(f'#SBATCH --mail-user={args.mail_user}\n')

    if args.cluster_name in ['Expanse', 'expanse', 'EXPANSE']:
        fid.write('module purge\n')
        fid.write('module add slurm\n')
        fid.write(f'source activate {args.python_env_name}\n')

    fid.write(
        '%s/train_xgb_job_submission.py --dirname=%s --scriptname=%s --user=%s --max_instances=%s --wait_time_minutes=%s\n'
        % (
            BASE_DIR / 'tools',
            dirname,
            scriptname,
            args.user,
            args.max_instances,
            args.wait_time_minutes,
        )
    )
    fid.close()
