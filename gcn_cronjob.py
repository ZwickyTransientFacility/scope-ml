#!/usr/bin/env python
# When setting up crontab, ensure that PYTHONPATH is specified in the cron environment
# This can be done by adding a line before your cron job (e.g. PYTHONPATH = /path/to/scope)
from scope.fritz import api
from datetime import datetime, timedelta
import argparse
import pathlib
from tools.scope_download_gcn_sources import download_gcn_sources
import os
from scope.utils import read_parquet, parse_load_config
import numpy as np
import warnings
import json
from scope.scope_class import Scope
from tools.combine_preds import combine_preds
from tools.scope_upload_classification import upload_classification


NUM_PER_PAGE = 100
BASE_DIR = pathlib.Path.cwd()
config = parse_load_config()


def query_gcn_events(
    daysAgo=7.0,
    query_group_ids: list = [],
    post_group_ids: list = [1544],
    days_range: float = 7.0,
    radius_arcsec: float = 0.5,
    save_filename: str = 'fritzDownload/specific_ids_GCN_sources',
    taxonomy_map: str = 'tools/fritz_mapper.json',
    combined_preds_dirname: str = 'GCN_dnn_xgb',
    dateobs: str = None,
    p_threshold: float = 0.7,
    username: str = 'bhealy',
    generated_features_dirname: str = 'generated_features_GCN_sources',
    partition: str = 'gpu-debug',
    doNotPost: bool = False,
    agg_method: str = 'mean',
    dnn_preds_directory: str = 'GCN_dnn',
    xgb_preds_directory: str = 'GCN_xgb',
    checkpoint_filename: str = 'gcn_sources_checkpoint.json',
    checkpoint_refresh_days: float = 180.0,
    ignore_checkpoint: bool = False,
):
    scope = Scope()
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

        print(f'Running for event {dateobs}...')

        # Colons can confuse the file system; replace them for saving
        save_dateobs = dateobs.replace(':', '-')

        # Check for existing sources file
        filepath = (
            BASE_DIR / f'fritzDownload/specific_ids_GCN_sources.{save_dateobs}.parquet'
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
                    if id not in checkpoint_dict['sources'][key]:
                        sources_to_run = True
                        checkpoint_dict['sources'][key].append(id)
                    elif ignore_checkpoint:
                        sources_to_run = True

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
                    f"scp {filepath} {username}@login.expanse.sdsc.edu:/expanse/lustre/projects/umn131/{username}/{generated_features_dirname}/fg_sources/."
                )
                os.system(
                    f'ssh -tt {username}@login.expanse.sdsc.edu \
                    "source .bash_profile && \
                    cd /expanse/lustre/projects/umn131/{username} && \
                    sbatch --wait --export=DOBS={save_dateobs},DS={filepath.name} {generated_features_dirname}/slurm/{partition}_slurm.sub"'
                )
                print("Finished generating features on Expanse.")

                os.system(
                    f"rsync -avh {username}@login.expanse.sdsc.edu:/expanse/lustre/projects/umn131/{username}/{generated_features_dirname} {BASE_DIR}/."
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

                        try:
                            generator = scope.select_fritz_sample(
                                fields=[f"{save_dateobs}_specific_ids"],
                                group="trained_xgb_models",
                                algorithm="xgb",
                                probability_threshold=0.0,
                                consol_filename=f"inference_results_{save_dateobs}",
                                al_directory="GCN",
                                al_filename=f"GCN_sources_{save_dateobs}",
                                write_consolidation_results=True,
                                select_top_n=True,
                                doAllSources=True,
                                write_csv=True,
                            )
                            [x for x in generator]

                            generator = scope.select_fritz_sample(
                                fields=[f"{save_dateobs}_specific_ids"],
                                group="trained_dnn_models",
                                algorithm="dnn",
                                probability_threshold=0.0,
                                consol_filename=f"inference_results_{save_dateobs}",
                                al_directory="GCN",
                                al_filename=f"GCN_sources_{save_dateobs}",
                                write_consolidation_results=True,
                                select_top_n=True,
                                doAllSources=True,
                                write_csv=True,
                            )
                            [x for x in generator]

                        except Exception as e:
                            print(f"Exception raised during select_fritz_sample: {e}")

                        print("Combining DNN and XGB preds...")

                        try:
                            combine_preds(
                                dateobs=save_dateobs,
                                combined_preds_dirname=f"{combined_preds_dirname}/{save_dateobs}",
                                merge_dnn_xgb=True,
                                write_csv=True,
                                p_threshold=p_threshold,
                                agg_method=agg_method,
                                dnn_directory=dnn_preds_directory,
                                xgb_directory=xgb_preds_directory,
                            )
                        except Exception as e:
                            print(f"Exception raised during combine_preds: {e}")

                    if not doNotPost:
                        print(
                            f"Uploading classifications with p > {p_threshold}. Posting light curves as comments."
                        )

                        try:
                            upload_classification(
                                file=f"{BASE_DIR}/{combined_preds_dirname}/{save_dateobs}/merged_GCN_sources_{save_dateobs}.parquet",
                                classification=["read"],
                                taxonomy_map=f"{BASE_DIR}/{taxonomy_map}",
                                skip_phot=True,
                                use_existing_obj_id=True,
                                group_ids=post_group_ids,
                                radius_arcsec=radius_arcsec,
                                p_threshold=p_threshold,
                                post_phot_as_comment=True,
                                post_phasefolded_phot=True,
                            )
                        except Exception as e:
                            print(f"Exception raised during upload_classification: {e}")

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


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--daysAgo",
        default=7.0,
        type=float,
        help="Number of days before today to query GCN events",
    )
    parser.add_argument(
        "--query-group-ids",
        type=int,
        nargs='+',
        default=[],
        help="group ids to query sources (all if not specified)",
    )
    parser.add_argument(
        "--post-group-ids",
        type=int,
        nargs='+',
        default=[1544],
        help="group ids to post source classifications (EM+GW group if not specified)",
    )
    parser.add_argument(
        "--days-range",
        type=float,
        default=7.0,
        help="max days past event to search for sources",
    )
    parser.add_argument(
        "--radius-arcsec",
        type=float,
        default=0.5,
        help="radius around new sources to search for existing ZTF sources",
    )
    parser.add_argument(
        "--save-filename",
        type=str,
        default='fritzDownload/specific_ids_GCN_sources',
        help="filename to save source ids/coordinates",
    )
    parser.add_argument(
        "--taxonomy-map",
        type=str,
        default='tools/fritz_mapper.json',
        help="path to taxonomy map for uploading classifications to Fritz",
    )
    parser.add_argument(
        "--combined-preds-dirname",
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
        "--p-threshold",
        type=float,
        default=0.7,
        help="minimum classification probability to post to Fritz",
    )
    parser.add_argument(
        "--username",
        type=str,
        default='dwarshofsky',
        help="Username for compute resources (e.g. Expanse)",
    )
    parser.add_argument(
        "--generated-features-dirname",
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
        "--agg-method",
        type=str,
        default='mean',
        help="Aggregation method for classification probabilities (mean or max)",
    )
    parser.add_argument(
        "--dnn-preds-directory",
        type=str,
        default='GCN_dnn',
        help="dirname in which dnn preds are saved",
    )
    parser.add_argument(
        "--xgb-preds-directory",
        type=str,
        default='GCN_xgb',
        help="dirname in which xgb preds preds are saved",
    )
    parser.add_argument(
        "--checkpoint-filename",
        type=str,
        default='gcn_sources_checkpoint.json',
        help="filename containing source ids already classified",
    )
    parser.add_argument(
        "--checkpoint-refresh-days",
        type=float,
        default=180.0,
        help="days after checkpoint start_date to delete json file and re-generate",
    )
    parser.add_argument(
        "--ignore-checkpoint",
        action='store_true',
        help="If set, ignore current classified sources listed in checkpoint file (bool)",
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args, _ = parser.parse_known_args()

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
        checkpoint_filename=args.checkpoint_filename,
        checkpoint_refresh_days=args.checkpoint_refresh_days,
        ignore_checkpoint=args.ignore_checkpoint,
    )
