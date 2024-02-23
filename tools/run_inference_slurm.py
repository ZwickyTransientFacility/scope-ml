#!/usr/bin/env python
import argparse
import pathlib
import os
from scope.utils import parse_load_config


BASE_DIR = pathlib.Path.cwd()
BASE_DIR_PREDS = BASE_DIR

config = parse_load_config()

path_to_preds = config['inference']['path_to_preds']
if path_to_preds is not None:
    BASE_DIR_PREDS = pathlib.Path(path_to_preds)


def get_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scriptname",
        type=str,
        default='get_all_preds.sh',
        help="Inference script filename within scope directory",
    )
    parser.add_argument(
        "--dirname",
        type=str,
        default='inference',
        help="Directory name for slurm scripts/logs",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default='run_inference',
        help="job name",
    )
    parser.add_argument(
        "--cluster-name",
        type=str,
        default='Expanse',
        help="Name of HPC cluster",
    )
    parser.add_argument(
        "--partition-type",
        type=str,
        default='gpu-shared',
        help="Partition name to request for computing",
    )
    parser.add_argument(
        "--submit-partition-type",
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
        default=8,
        type=int,
        help="number of cores to request for computing",
    )
    parser.add_argument(
        "--submit-nodes",
        type=int,
        default=1,
        help="Number of nodes to request for job submission",
    )
    parser.add_argument(
        "--submit-Ncore",
        default=1,
        type=int,
        help="number of cores to request for job submission",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to request",
    )
    parser.add_argument(
        "--memory-GB",
        type=int,
        default=64,
        help="Memory allocation to request for computing",
    )
    parser.add_argument(
        "--submit-memory-GB",
        type=int,
        default=16,
        help="Memory allocation to request for job submission",
    )
    parser.add_argument(
        "--time",
        type=str,
        default='24:00:00',
        help="Walltime for instance",
    )
    parser.add_argument(
        "--mail-user",
        type=str,
        default='healyb@umn.edu',
        help="contact email address",
    )
    parser.add_argument(
        "--account-name",
        type=str,
        default='umn131',
        help="Name of account with current HPC allocation",
    )
    parser.add_argument(
        "--python-env-name",
        type=str,
        default='scope-env',
        help="Name of python environment to activate",
    )
    parser.add_argument(
        "--user",
        type=str,
        default="bhealy",
        help="HPC username",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="dnn",
        help="dnn or xgb",
    )

    return parser


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()

    scriptname = args.scriptname
    script_path = BASE_DIR / scriptname

    algorithm = args.algorithm

    dirname = f"{algorithm}_{args.dirname}"
    jobname = f"{args.job_name}_{algorithm}"

    dirpath = BASE_DIR_PREDS / dirname
    os.makedirs(dirpath, exist_ok=True)

    slurmDir = os.path.join(dirpath, 'slurm')
    os.makedirs(slurmDir, exist_ok=True)

    logsDir = os.path.join(dirpath, 'logs')
    os.makedirs(logsDir, exist_ok=True)

    # Main script to run feature generation
    fid = open(os.path.join(slurmDir, 'slurm.sub'), 'w')
    fid.write('#!/bin/bash\n')
    fid.write(f'#SBATCH --job-name={jobname}.job\n')
    fid.write(f'#SBATCH --output={dirname}/logs/{jobname}_%A_%a.out\n')
    fid.write(f'#SBATCH --error={dirname}/logs/{jobname}_%A_%a.err\n')
    fid.write(f'#SBATCH -p {args.partition_type}\n')
    fid.write(f'#SBATCH --nodes {args.nodes}\n')
    fid.write(f'#SBATCH --ntasks-per-node {args.Ncore}\n')
    fid.write(f'#SBATCH --gpus {args.gpus}\n')
    fid.write(f'#SBATCH --mem {args.memory_GB}G\n')
    fid.write(f'#SBATCH --time={args.time}\n')
    fid.write('#SBATCH --mail-type=ALL\n')
    fid.write(f'#SBATCH --mail-user={args.mail_user}\n')
    fid.write(f'#SBATCH -A {args.account_name}\n')

    if args.cluster_name in ['Expanse', 'expanse', 'EXPANSE']:
        fid.write('module purge\n')
        if args.gpus > 0:
            fid.write('module add gpu/0.15.4\n')
            fid.write('module add cuda\n')
        fid.write(f'source activate {args.python_env_name}\n')

    fid.write(f"{script_path} $FID" + '\n')
    fid.close()

    # Secondary script to manage job submission using run_inference_job_submission.py
    # (Python code can also be run interactively)
    fid = open(os.path.join(slurmDir, 'slurm_submission.sub'), 'w')
    fid.write('#!/bin/bash\n')
    fid.write(f'#SBATCH --job-name={jobname}_submit.job\n')
    fid.write(f'#SBATCH --output={dirname}/logs/{jobname}_submit_%A_%a.out\n')
    fid.write(f'#SBATCH --error={dirname}/logs/{jobname}_submit_%A_%a.err\n')
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
        'run-inference-job-submission --dirname %s --scriptname %s --user %s --algorithm %s\n'
        % (
            dirname,
            scriptname,
            args.user,
            algorithm,
        )
    )
    fid.close()
