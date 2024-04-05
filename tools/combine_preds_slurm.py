#!/usr/bin/env python
import argparse
import pathlib
import os
from scope.utils import parse_load_config
from tools.combine_preds import get_parser


BASE_DIR = pathlib.Path.cwd()
config = parse_load_config()


def get_slurm_parser():

    cp_parser = get_parser(add_help=False)
    parser = argparse.ArgumentParser(parents=[cp_parser])
    parser.add_argument(
        "--dirname",
        type=str,
        default='combine_preds',
        help="Directory name for slurm scripts/logs",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default='combine_preds',
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
        default='shared',
        help="Partition name to request for computing",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Number of nodes to request for computing",
    )
    parser.add_argument(
        "--Ncore",
        default=1,
        type=int,
        help="number of cores to request for computing",
    )
    parser.add_argument(
        "--memory-GB",
        type=int,
        default=128,
        help="Memory allocation to request for computing",
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
    return parser


def main():
    parser = get_slurm_parser()
    args, _ = parser.parse_known_args()

    dirname = args.dirname
    jobname = args.job_name

    dirpath = BASE_DIR / dirname
    os.makedirs(dirpath, exist_ok=True)

    slurmDir = os.path.join(dirpath, 'slurm')
    os.makedirs(slurmDir, exist_ok=True)

    logsDir = os.path.join(dirpath, 'logs')
    os.makedirs(logsDir, exist_ok=True)

    extra_flags = []
    if args.use_config_fields:
        extra_flags.append("--use-config-fields")
    if args.merge_dnn_xgb:
        extra_flags.append("--merge-dnn-xgb")
    if args.doNotSave:
        extra_flags.append("--doNotSave")
    if args.write_csv:
        extra_flags.append("--write-csv")
    extra_flags = " ".join(extra_flags)

    # Main script to run combine-preds
    fid = open(os.path.join(slurmDir, 'slurm.sub'), 'w')
    fid.write('#!/bin/bash\n')
    fid.write(f'#SBATCH --job-name={jobname}.job\n')
    fid.write(f'#SBATCH --output={dirname}/logs/{jobname}_%A_%a.out\n')
    fid.write(f'#SBATCH --error={dirname}/logs/{jobname}_%A_%a.err\n')
    fid.write(f'#SBATCH -p {args.partition_type}\n')
    fid.write(f'#SBATCH --nodes {args.nodes}\n')
    fid.write(f'#SBATCH --ntasks-per-node {args.Ncore}\n')
    fid.write(f'#SBATCH --mem {args.memory_GB}G\n')
    fid.write(f'#SBATCH --time={args.time}\n')
    fid.write('#SBATCH --mail-type=ALL\n')
    fid.write(f'#SBATCH --mail-user={args.mail_user}\n')
    fid.write(f'#SBATCH -A {args.account_name}\n')

    if args.cluster_name in ['Expanse', 'expanse', 'EXPANSE']:
        fid.write('module purge\n')
        fid.write(f'source activate {args.python_env_name}\n')
    elif args.cluster_name in ['Delta', 'delta', 'DELTA']:
        fid.write('module purge\n')
        fid.write('module add anaconda3_cpu\n')
        fid.write(f'source activate {args.python_env_name}\n')

    fid.write(
        'combine-preds --path-to-preds %s --combined-preds-dirname %s --specific-field %s --dateobs %s --dnn-directory %s --xgb-directory %s --agg-method %s --p-threshold %s %s \n'
        % (
            args.path_to_preds,
            args.combined_preds_dirname,
            args.specific_field,
            args.dateobs,
            args.dnn_directory,
            args.xgb_directory,
            args.agg_method,
            args.p_threshold,
            extra_flags,
        )
    )
    fid.close()
