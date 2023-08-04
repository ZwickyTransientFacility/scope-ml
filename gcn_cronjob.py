#!/usr/bin/env python
# When setting up crontab, ensure that PYTHONPATH is specified in the cron environment
# This can be done by adding a line before your cron job (e.g. PYTHONPATH = /path/to/scope)
from scope.fritz import api
from datetime import datetime, timedelta
import argparse
import pathlib
import yaml
from tools.scope_download_gcn_sources import download_gcn_sources
import os
from scope.utils import read_parquet
import numpy as np
import warnings
import json


BASE_DIR = pathlib.Path(__file__).parent.absolute()
NUM_PER_PAGE = 100

config_path = BASE_DIR / "config.yaml"
with open(config_path) as config_yaml:
    config = yaml.load(config_yaml, Loader=yaml.FullLoader)


def query_gcn_events(
    daysAgo=7.0,
    query_group_ids: list = [],
    post_group_ids: list = [1544],
    days_range: float = 7.0,
    radius_arcsec: float = 0.5,
    save_filename: str = 'tools/fritzDownload/specific_ids_GCN_sources',
    taxonomy_map: str = 'tools/fritz_mapper.json',
    combined_preds_dirname: str = 'GCN_dnn_xgb',
    dateobs: str = None,
    p_threshold: float = 0.7,
    username: str = 'bhealy',
    generated_features_dirname: str = 'generated_features_gcn_sources',
    partition: str = 'gpu-debug',
    doNotPost: bool = False,
    agg_method: str = 'mean',
    dnn_preds_directory: str = 'GCN_dnn',
    xgb_preds_directory: str = 'GCN_xgb',
    path_to_python: str = '~/miniforge3/envs/scope-env/bin/python',
    checkpoint_filename: str = 'gcn_sources_checkpoint.json',
    checkpoint_refresh_days: float = 30.0,
    ignore_checkpoint: bool = False,
):

    currentDate = datetime.utcnow()
    current_dt = currentDate.strftime("%Y-%m-%dT%H:%M:%S")

    checkpoint_path = BASE_DIR / checkpoint_filename

    chk_deleted = False
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            checkpoint_dict = json.load(f)
        chk_startDate = datetime.strptime(
            checkpoint_dict['start_dt'], "%Y-%m-%dT%H:%M:%S"
        )
        # Delete checkpoint file if start date is too far in past
        # (Avoids endlessly growing id list)
        chk_diff = currentDate - chk_startDate
        if chk_diff.seconds / 86400.0 > checkpoint_refresh_days:
            checkpoint_path.unlink()
            chk_deleted = True
    if (not checkpoint_path.exists()) | (chk_deleted):
        checkpoint_dict = {'start_dt': current_dt, 'sources': {'gw': [], 'grb': []}}

    if dateobs is None:
        utcnow = datetime.utcnow()
        start_dt = utcnow - timedelta(days=daysAgo)
        startDate = start_dt.strftime("%Y-%m-%dT%H:%M:%S")
        print(f"Querying for GCN events occurring after {startDate}...")

        nPerPage = NUM_PER_PAGE

        params = {'startDate': startDate, 'numPerPage': nPerPage}
        response = api('GET', '/api/gcn_event', data=params).json()

        if response.get('status', 'error') == 'success':
            data = response.get('data')
            allMatches = data.get('totalMatches')
            pages = int(np.ceil(allMatches / nPerPage))

            print(f'Found {allMatches} events.')

            gcn_events = []

            # iterate over all pages in results
            if (allMatches is not None) & (allMatches > 0):
                print(f'Downloading {allMatches} events...')
                for pageNum in range(1, pages + 1):
                    print(f'Page {pageNum} of {pages}...')
                    page_response = api(
                        "GET",
                        '/api/gcn_event',
                        {
                            "startDate": startDate,
                            'numPerPage': NUM_PER_PAGE,
                            'pageNumber': pageNum,
                        },  # page numbers start at 1
                    ).json()
                    page_data = page_response.get('data')
                    events = page_data.get('events')
                    gcn_events.extend([x for x in events])

        else:
            raise ValueError('Query error - no data returned.')

    else:
        response = api('GET', f'/api/gcn_event/{dateobs}').json()
        if response.get('status', 'error') == 'success':
            event = response.get('data')
            gcn_events = [event]
        else:
            warnings.warn("Unsuccessful query.")
            return

    for event in gcn_events:
        dateobs = event["dateobs"]
        tags = event["tags"]

        chk_dict_keys = []
        # Set group(s) for classifications
        if ("GRB" in tags) | ("Fermi" in tags):
            chk_dict_keys.append('grb')
            # Gamma Ray Bursts group on Fritz
            post_group_ids = [48]
        if "GW" in tags:
            chk_dict_keys.append('gw')
            if 1544 not in post_group_ids:
                # EM+GW group on Fritz
                post_group_ids.append(1544)

        post_group_ids_str = "".join([f"{x} " for x in post_group_ids]).strip()
        print(f'Running for event {dateobs}...')

        # Colons can confuse the file system; replace them for saving
        save_dateobs = dateobs.replace(':', '-')

        # Check for existing sources file
        filepath = (
            BASE_DIR
            / f'tools/fritzDownload/specific_ids_GCN_sources.{save_dateobs}.parquet'
        )
        if filepath.exists():
            existing_sources = read_parquet(filepath)
            existing_ids = existing_sources['ztf_id'].values
        else:
            existing_ids = []

        print(f'Downloading GCN sources for {dateobs}...')
        ids = download_gcn_sources(
            dateobs=dateobs,
            group_ids=query_group_ids,
            days_range=days_range,
            radius_arcsec=radius_arcsec,
            save_filename=save_filename,
        )

        sources_to_run = False
        if ids is not None:
            for key in chk_dict_keys:
                for id in ids:
                    if (id not in checkpoint_dict['sources'][key]) | ignore_checkpoint:
                        sources_to_run = True
                        checkpoint_dict['sources'][key].append(id)

        try:
            current_sources = read_parquet(filepath)
            new_sources = current_sources.copy().set_index('ztf_id')

            for id in existing_ids:
                try:
                    new_sources = new_sources.drop(id)
                except KeyError:
                    continue
            new_sources.reset_index(inplace=True)
        except FileNotFoundError:
            new_sources = []

        if len(new_sources) > 0:
            has_new_sources = True
            print(f"Event {dateobs} has {len(new_sources)} new sources.")
        else:
            has_new_sources = False
            print(f"Event {dateobs} has no new sources.")

        if sources_to_run:
            if ignore_checkpoint:
                print('Processing all new sources, ignoring checkpoint list...')
            else:
                print('Processing sources missing from checkpoint list...')
            features_file = (
                BASE_DIR
                / f"{generated_features_dirname}/specific_ids/gen_gcn_features_{save_dateobs}_specific_ids.parquet"
            )
            if (not features_file.exists()) | (has_new_sources):
                print("Generating features on Expanse...")
                os.system(
                    f"scp {filepath} {username}@login.expanse.sdsc.edu:/home/{username}/scope/tools/fritzDownload/."
                )
                os.system(
                    f'ssh -tt {username}@login.expanse.sdsc.edu \
                    "source .bash_profile && \
                    cd scope/{generated_features_dirname}/slurm && \
                    sbatch --wait --export=DOBS={save_dateobs},DS={filepath.name} {partition}_slurm.sub"'
                )
                print("Finished generating features on Expanse.")

                os.system(
                    f"rsync -avh {username}@login.expanse.sdsc.edu:/home/{username}/scope/{generated_features_dirname} {BASE_DIR}/."
                )

            if features_file.exists():
                features = read_parquet(features_file)

                if len(features) > 0:
                    preds_dnn_file = (
                        BASE_DIR
                        / f"preds_dnn/field_{save_dateobs}_specific_ids/field_{save_dateobs}_specific_ids.parquet"
                    )
                    preds_xgb_file = (
                        BASE_DIR
                        / f"preds_xgb/field_{save_dateobs}_specific_ids/field_{save_dateobs}_specific_ids.parquet"
                    )
                    preds_dnn_xgb_file = (
                        BASE_DIR
                        / f"{combined_preds_dirname}/{save_dateobs}/merged_GCN_sources_{save_dateobs}.parquet"
                    )
                    if (
                        (not preds_dnn_xgb_file.exists())
                        | (not preds_dnn_file.exists())
                        | (not preds_xgb_file.exists())
                        | has_new_sources
                    ):
                        print("Running DNN and XGB inference...")
                        # DNN: use nobalance_DR16_DNN models
                        os.system(
                            f"{BASE_DIR}/get_all_preds_dnn_GCN.sh {save_dateobs}_specific_ids"
                        )

                        # XGB: use DR16_importance models
                        os.system(
                            f"{BASE_DIR}/get_all_preds_xgb_GCN.sh {save_dateobs}_specific_ids"
                        )

                        print(
                            "Consolidating DNN and XGB classification results for Fritz..."
                        )
                        os.system(
                            f"{path_to_python} {BASE_DIR}/scope.py select_fritz_sample --fields='{save_dateobs}_specific_ids' --group='DR16' --algorithm='xgb' \
                                --probability_threshold=0 --consol_filename='inference_results_{save_dateobs}' --al_directory='GCN' \
                                --al_filename='GCN_sources_{save_dateobs}' --write_consolidation_results --select_top_n --doAllSources --write_csv"
                        )

                        os.system(
                            f"{path_to_python} {BASE_DIR}/scope.py select_fritz_sample --fields='{save_dateobs}_specific_ids' --group='nobalance_DR16_DNN' --algorithm='dnn' \
                                --probability_threshold=0 --consol_filename='inference_results_{save_dateobs}' --al_directory='GCN' \
                                --al_filename='GCN_sources_{save_dateobs}' --write_consolidation_results --select_top_n --doAllSources --write_csv"
                        )

                        print("Combining DNN and XGB preds...")
                        os.system(
                            f"{path_to_python} {BASE_DIR}/tools/combine_preds.py --dateobs {save_dateobs} --combined_preds_dirname {combined_preds_dirname}/{save_dateobs} \
                                  --merge_dnn_xgb --write_csv --p_threshold {p_threshold} --agg_method {agg_method} --dnn_directory {dnn_preds_directory} \
                                  --xgb_directory {xgb_preds_directory}"
                        )

                    if not doNotPost:
                        print(
                            f"Uploading classifications with p > {p_threshold}. Posting light curves as comments."
                        )
                        os.system(
                            f"{path_to_python} {BASE_DIR}/tools/scope_upload_classification.py --file {BASE_DIR}/{combined_preds_dirname}/{save_dateobs}/merged_GCN_sources_{save_dateobs}.parquet \
                                --classification read --taxonomy_map {BASE_DIR}/{taxonomy_map} --skip_phot --use_existing_obj_id --group_ids {post_group_ids_str} --radius_arcsec {radius_arcsec} \
                                --p_threshold {p_threshold} --post_phot_as_comment --post_phasefolded_phot"
                        )

                    print(f"Finished for {dateobs}.")

                else:
                    warnings.warn("No features returned.")

            else:
                warnings.warn("Features file does not exist.")

        else:
            print('No unclassified sources to run.')
        print()

    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--daysAgo",
        default=7.0,
        type=float,
        help="Number of days before today to query GCN events",
    )
    parser.add_argument(
        "--query_group_ids",
        type=int,
        nargs='+',
        default=[],
        help="group ids to query sources (all if not specified)",
    )
    parser.add_argument(
        "--post_group_ids",
        type=int,
        nargs='+',
        default=[1544],
        help="group ids to post source classifications (EM+GW group if not specified)",
    )
    parser.add_argument(
        "--days_range",
        type=float,
        default=7.0,
        help="max days past event to search for sources",
    )
    parser.add_argument(
        "--radius_arcsec",
        type=float,
        default=0.5,
        help="radius around new sources to search for existing ZTF sources",
    )
    parser.add_argument(
        "--save_filename",
        type=str,
        default='tools/fritzDownload/specific_ids_GCN_sources',
        help="filename to save source ids/coordinates",
    )
    parser.add_argument(
        "--taxonomy_map",
        type=str,
        default='tools/fritz_mapper.json',
        help="path to taxonomy map for uploading classifications to Fritz",
    )
    parser.add_argument(
        "--combined_preds_dirname",
        type=str,
        default='GCN_dnn_xgb',
        help="dirname in which to save combined preds files",
    )
    parser.add_argument(
        "--dateobs",
        type=str,
        default=None,
        help="If querying specific dateobs, specify here to override daysAgo.",
    )
    parser.add_argument(
        "--p_threshold",
        type=float,
        default=0.7,
        help="minimum classification probability to post to Fritz",
    )
    parser.add_argument(
        "--username",
        type=str,
        default='bhealy',
        help="Username for compute resources (e.g. Expanse)",
    )
    parser.add_argument(
        "--generated_features_dirname",
        type=str,
        default='generated_features_GCN_sources',
        help="dirname containing generated GCN source features",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default='gpu-debug',
        help="name of compute partition on which to run feature generation",
    )
    parser.add_argument(
        "--doNotPost",
        action='store_true',
        help="If set, run analysis but do not post classifications. Useful for testing",
    )
    parser.add_argument(
        "--agg_method",
        type=str,
        default='mean',
        help="Aggregation method for classification probabilities (mean or max)",
    )
    parser.add_argument(
        "--dnn_preds_directory",
        type=str,
        default='GCN_dnn',
        help="dirname in which dnn preds are saved",
    )
    parser.add_argument(
        "--xgb_preds_directory",
        type=str,
        default='GCN_xgb',
        help="dirname in which xgb preds preds are saved",
    )
    parser.add_argument(
        "--path_to_python",
        type=str,
        default='~/miniforge3/envs/scope-env/bin/python',
        help="path to python within scope environment (run 'which python' while your scope environment is active to find)",
    )
    parser.add_argument(
        "--checkpoint_filename",
        type=str,
        default='gcn_sources_checkpoint.json',
        help="filename containing source ids already classified",
    )
    parser.add_argument(
        "--checkpoint_refresh_days",
        type=float,
        default=30.0,
        help="days after checkpoint start_date to delete json file and re-generate",
    )
    parser.add_argument(
        "--ignore_checkpoint",
        action='store_true',
        help="If set, ignore current classified sources listed in checkpoint file (bool)",
    )

    args = parser.parse_args()

    query_gcn_events(
        daysAgo=args.daysAgo,
        query_group_ids=args.query_group_ids,
        post_group_ids=args.post_group_ids,
        days_range=args.days_range,
        radius_arcsec=args.radius_arcsec,
        save_filename=args.save_filename,
        taxonomy_map=args.taxonomy_map,
        combined_preds_dirname=args.combined_preds_dirname,
        dateobs=args.dateobs,
        p_threshold=args.p_threshold,
        username=args.username,
        generated_features_dirname=args.generated_features_dirname,
        partition=args.partition,
        doNotPost=args.doNotPost,
        agg_method=args.agg_method,
        dnn_preds_directory=args.dnn_preds_directory,
        xgb_preds_directory=args.xgb_preds_directory,
        path_to_python=args.path_to_python,
        checkpoint_filename=args.checkpoint_filename,
        checkpoint_refresh_days=args.checkpoint_refresh_days,
        ignore_checkpoint=args.ignore_checkpoint,
    )
