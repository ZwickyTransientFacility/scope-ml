#!/usr/bin/env python
import argparse
import pathlib
import yaml
import os
from penquins import Kowalski
import numpy as np


BASE_DIR = pathlib.Path(__file__).parent.parent.absolute()

# setup connection to Kowalski instances
config_path = pathlib.Path(__file__).parent.parent.absolute() / "config.yaml"
with open(config_path) as config_yaml:
    config = yaml.load(config_yaml, Loader=yaml.FullLoader)

# use token specified as env var (if exists)
kowalski_token_env = os.environ.get("KOWALSKI_TOKEN")
kowalski_alt_token_env = os.environ.get("KOWALSKI_ALT_TOKEN")
if (kowalski_token_env is not None) & (kowalski_alt_token_env is not None):
    config["kowalski"]["token"] = kowalski_token_env
    config["kowalski"]["alt_token"] = kowalski_alt_token_env

timeout = config['kowalski']['timeout']

gloria = Kowalski(**config['kowalski'], verbose=False)
melman = Kowalski(
    token=config['kowalski']['token'],
    protocol="https",
    host="melman.caltech.edu",
    port=443,
    timeout=timeout,
)
kowalski = Kowalski(
    token=config['kowalski']['alt_token'],
    protocol="https",
    host="kowalski.caltech.edu",
    port=443,
    timeout=timeout,
)

source_catalog = config['kowalski']['collections']['sources']
alerts_catalog = config['kowalski']['collections']['alerts']
gaia_catalog = config['kowalski']['collections']['gaia']
ext_catalog_info = config['feature_generation']['external_catalog_features']
cesium_feature_list = config['feature_generation']['cesium_features']

kowalski_instances = {'kowalski': kowalski, 'gloria': gloria, 'melman': melman}


def check_quads_for_sources(
    fields: list = np.arange(1, 2001),
    kowalski_instance_name: str = 'melman',
    catalog=source_catalog,
):
    """
    Check ZTF field/ccd/quadrant combos for any sources.

    """
    kowalski_instance = kowalski_instances[kowalski_instance_name]

    has_sources = np.zeros(len(fields), dtype=bool)
    missing_ccd_quad = np.zeros(len(fields), dtype=bool)
    field_dct = {}
    for idx, field in enumerate(fields):
        print('Running field %d' % field)
        except_count = 0
        # Run minimal query to determine if sources exist in field
        q = {
            "query_type": "find",
            "query": {
                "catalog": catalog,
                "filter": {
                    'field': {'$eq': int(field)},
                },
                "projection": {"_id": 1},
            },
            "kwargs": {"limit": 1},
        }
        rsp = kowalski_instance.query(q)
        data = rsp['data']

        if len(data) > 0:
            has_sources[idx] = True
        else:
            continue

        if has_sources[idx]:
            print(f'Field {field} has sources...')
            field_dct[field] = {}
            for ccd in range(1, 17):
                quads = []
                for quadrant in range(1, 5):

                    # Another minimal query for each ccd/quad combo
                    q = {
                        "query_type": "find",
                        "query": {
                            "catalog": catalog,
                            "filter": {
                                'field': {'$eq': int(field)},
                                'ccd': {'$eq': int(ccd)},
                                'quad': {'$eq': int(quadrant)},
                            },
                            "projection": {"_id": 1},
                        },
                        "kwargs": {"limit": 1},
                    }
                    rsp = kowalski_instance.query(q)
                    data = rsp['data']

                    if len(data) > 0:
                        quads += [quadrant]
                    else:
                        except_count += 1

                if len(quads) > 0:
                    field_dct[field].update({ccd: quads})

        print(f"{64 - except_count} ccd/quad combos")
        if except_count > 0:
            missing_ccd_quad[idx] = True

    print(f"Sources found in {np.sum(has_sources)} fields.")

    return field_dct, has_sources, missing_ccd_quad


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source_catalog",
        default=source_catalog,
        help="name of source collection on Kowalski",
    )
    parser.add_argument(
        "--alerts_catalog",
        default=alerts_catalog,
        help="name of alerts collection on Kowalski",
    )
    parser.add_argument(
        "--gaia_catalog",
        default=gaia_catalog,
        help="name of Gaia collection on Kowalski",
    )
    parser.add_argument(
        "--bright_star_query_radius_arcsec",
        type=float,
        default=300.0,
        help="size of cone search radius to search for bright stars",
    )
    parser.add_argument(
        "--xmatch_radius_arcsec",
        type=float,
        default=2.0,
        help="cone radius for all crossmatches",
    )
    parser.add_argument(
        "--query_size_limit",
        type=int,
        default=10000,
        help="sources per query limit for large Kowalski queries",
    )
    parser.add_argument(
        "--period_batch_size",
        type=int,
        default=1000,
        help="batch size for GPU-accelerated period algorithms",
    )
    parser.add_argument(
        "--doCPU",
        action='store_true',
        default=False,
        help="if set, run period-finding algorithms on CPU",
    )
    parser.add_argument(
        "--doGPU",
        action='store_true',
        default=False,
        help="if set, use GPU-accelerated period algorithms",
    )
    parser.add_argument(
        "--samples_per_peak",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--doLongPeriod",
        action='store_true',
        default=False,
        help="if set, optimize frequency grid for long periods",
    )
    parser.add_argument(
        "--doRemoveTerrestrial",
        action='store_true',
        default=False,
        help="if set, remove terrestrial frequencies from period analysis",
    )
    parser.add_argument(
        "--Ncore",
        default=10,
        type=int,
        help="number of cores for parallel period finding",
    )
    parser.add_argument(
        "--field",
        type=int,
        default=296,
        help="if not -doAllFields, ZTF field to run on",
    )
    parser.add_argument(
        "--ccd", type=int, default=1, help="if not -doAllCCDs, ZTF ccd to run on"
    )
    parser.add_argument(
        "--quad", type=int, default=1, help="if not -doAllQuads, ZTF field to run on"
    )
    parser.add_argument(
        "--min_n_lc_points",
        type=int,
        default=50,
        help="minimum number of unflagged light curve points to run feature generation",
    )
    parser.add_argument(
        "--min_cadence_minutes",
        type=float,
        default=30.0,
        help="minimum cadence (in minutes) between light curve points. For groups of points closer together than this value, only the first will be kept.",
    )
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
        help="Filename prefix for generated features",
    )
    parser.add_argument(
        "--doCesium",
        action='store_true',
        default=False,
        help="if set, use Cesium to generate additional features specified in config",
    )
    parser.add_argument(
        "--doNotSave",
        action='store_true',
        default=False,
        help="if set, do not save features",
    )
    parser.add_argument(
        "--stop_early",
        action='store_true',
        default=False,
        help="if set, stop when number of sources reaches query_size_limit. Helpful for testing on small samples.",
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default='generate_features',
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
        default='gpu-shared',
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
        help="Number of nodes to request",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to request",
    )
    parser.add_argument(
        "--memory_GB",
        type=int,
        default=180,
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
        default='48:00:00',
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
        "--kowalski_instance_name",
        type=str,
        default='melman',
        help="Name of Kowalski instance containing source collection",
    )
    parser.add_argument(
        "--generateQuadrantFile",
        action='store_true',
        default=False,
        help="if set, generate a list of fields/ccd/quads and job numbers, save to slurm.dat",
    )
    parser.add_argument("--doQuadrantFile", action="store_true", default=False)
    parser.add_argument("--quadrant_file", default="slurm.dat")
    parser.add_argument("--quadrant_index", default=0, type=int)
    parser.add_argument(
        "--max_instances",
        type=int,
        default=20,
        help="Max number of instances to run in parallel",
    )
    parser.add_argument(
        "--wait_time_minutes",
        type=float,
        default=5.0,
        help="Time to wait between job status checks",
    )
    parser.add_argument(
        "--runParallel",
        action="store_true",
        default=False,
        help="If set, run jobs in parallel using slurm. Otherwise, run in series on a single instance.",
    )

    args = parser.parse_args()

    if not (args.doCPU or args.doGPU):
        print("--doCPU or --doGPU required")
        exit(0)

    source_catalog = args.source_catalog
    alerts_catalog = args.alerts_catalog
    gaia_catalog = args.gaia_catalog
    bright_star_query_radius_arcsec = args.bright_star_query_radius_arcsec
    xmatch_radius_arcsec = args.xmatch_radius_arcsec
    query_size_limit = args.query_size_limit
    period_batch_size = args.period_batch_size
    doCPU = args.doCPU
    doGPU = args.doGPU
    samples_per_peak = args.samples_per_peak
    doLongPeriod = args.doLongPeriod
    doRemoveTerrestrial = args.doRemoveTerrestrial
    Ncore = args.Ncore
    field = args.field
    ccd = args.ccd
    quad = args.quad
    min_n_lc_points = args.min_n_lc_points
    min_cadence_minutes = args.min_cadence_minutes
    dirname = args.dirname
    filename = args.filename
    doCesium = args.doCesium
    doNotSave = args.doNotSave
    stop_early = args.stop_early

    if args.doCPU:
        cpu_gpu_flag = "--doCPU"
    else:
        cpu_gpu_flag = "--doGPU"

    extra_flags = []
    if args.doLongPeriod:
        extra_flags.append("--doLongPeriod")
    if args.doRemoveTerrestrial:
        extra_flags.append("--doRemoveTerrestrial")
    if args.doCesium:
        extra_flags.append("--doCesium")
    if args.doNotSave:
        extra_flags.append("--doNotSave")
    if args.stop_early:
        extra_flags.append("--stop_early")
    extra_flags = " ".join(extra_flags)

    dirpath = BASE_DIR / dirname
    os.makedirs(dirpath, exist_ok=True)

    slurmDir = os.path.join(dirpath, 'slurm')
    if not os.path.isdir(slurmDir):
        os.makedirs(slurmDir)

    logsDir = os.path.join(dirpath, 'logs')
    if not os.path.isdir(logsDir):
        os.makedirs(logsDir)

    quadrantfile = os.path.join(slurmDir, args.quadrant_file)
    if args.generateQuadrantFile:
        field_dct, _, _ = check_quads_for_sources(
            kowalski_instance_name=args.kowalski_instance_name, catalog=source_catalog
        )

        job_number = 0

        fid = open(quadrantfile, 'w')
        for field in field_dct.keys():
            for ccd in field_dct[field].keys():
                for quad in field_dct[field][ccd]:
                    fid.write('%d %d %d %d\n' % (job_number, field, ccd, quad))
                    job_number += 1
        fid.close()

    # Main script to run feature generation
    fid = open(os.path.join(slurmDir, 'slurm.sub'), 'w')
    fid.write('#!/bin/bash\n')
    fid.write(f'#SBATCH --job-name={args.job_name}.job\n')
    fid.write(f'#SBATCH --output=../logs/{args.job_name}_%A_%a.out\n')
    fid.write(f'#SBATCH --error=../logs/{args.job_name}_%A_%a.err\n')
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
        fid.write('module add gpu\n')
        fid.write('module add cuda\n')
        fid.write(f'source activate {args.python_env_name}\n')

    if args.doQuadrantFile:
        fid.write(
            '%s/generate_features.py --source_catalog %s --alerts_catalog %s --gaia_catalog %s --bright_star_query_radius_arcsec %s --xmatch_radius_arcsec %s --query_size_limit %s --period_batch_size %s --samples_per_peak %s --Ncore %s --min_n_lc_points %s --min_cadence_minutes %s --dirname %s --filename %s --doQuadrantFile --quadrant_file %s --quadrant_index $QID %s %s\n'
            % (
                BASE_DIR / 'tools',
                source_catalog,
                alerts_catalog,
                gaia_catalog,
                bright_star_query_radius_arcsec,
                xmatch_radius_arcsec,
                query_size_limit,
                period_batch_size,
                samples_per_peak,
                Ncore,
                min_n_lc_points,
                min_cadence_minutes,
                dirname,
                filename,
                args.quadrant_file,
                cpu_gpu_flag,
                extra_flags,
            )
        )
    else:
        fid.write(
            '%s/generate_features.py --source_catalog %s --alerts_catalog %s --gaia_catalog %s --bright_star_query_radius_arcsec %s --xmatch_radius_arcsec %s --query_size_limit %s --period_batch_size %s --samples_per_peak %s --Ncore %s --field %s --ccd %s --quad %s --min_n_lc_points %s --min_cadence_minutes %s --dirname %s --filename %s %s %s\n'
            % (
                BASE_DIR / 'tools',
                source_catalog,
                alerts_catalog,
                gaia_catalog,
                bright_star_query_radius_arcsec,
                xmatch_radius_arcsec,
                query_size_limit,
                period_batch_size,
                samples_per_peak,
                Ncore,
                field,
                ccd,
                quad,
                min_n_lc_points,
                min_cadence_minutes,
                dirname,
                filename,
                cpu_gpu_flag,
                extra_flags,
            )
        )
    fid.close()

    # Secondary script to manage job submission using generate_features_job_submission.py
    # (Python code can also be run interactively)
    fid = open(os.path.join(slurmDir, 'slurm_submission.sub'), 'w')
    fid.write('#!/bin/bash\n')
    fid.write(f'#SBATCH --job-name={args.job_name}_submit.job\n')
    fid.write(f'#SBATCH --output=../logs/{args.job_name}_submit_%A_%a.out\n')
    fid.write(f'#SBATCH --error=../logs/{args.job_name}_submit_%A_%a.err\n')
    fid.write(f'#SBATCH -p {args.submit_partition_type}\n')
    fid.write(f'#SBATCH --mem {args.submit_memory_GB}G\n')
    fid.write(f'#SBATCH -A {args.account_name}\n')
    fid.write(f'#SBATCH --time={args.time}\n')
    fid.write('#SBATCH --mail-type=ALL\n')
    fid.write(f'#SBATCH --mail-user={args.mail_user}\n')

    if args.cluster_name in ['Expanse', 'expanse', 'EXPANSE']:
        fid.write('module purge\n')
        fid.write('module add slurm\n')
        fid.write(f'source activate {args.python_env_name}\n')

    if args.runParallel:
        fid.write(
            '%s/generate_features_job_submission.py --dirname %s --filename %s --doSubmit --runParallel --max_instances %s --wait_time_minutes %s\n'
            % (
                BASE_DIR / 'tools',
                dirpath,
                filename,
                args.max_instances,
                args.wait_time_minutes,
            )
        )
    else:
        fid.write(
            '%s/generate_features_job_submission.py --dirname %s --filename %s --doSubmit --max_instances %s --wait_time_minutes %s\n'
            % (
                BASE_DIR / 'tools',
                dirpath,
                filename,
                args.max_instances,
                args.wait_time_minutes,
            )
        )
    fid.close()
