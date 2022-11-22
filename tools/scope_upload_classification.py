#!/usr/bin/env python
import argparse
import json as JSON
import pandas as pd
from penquins import Kowalski
from scope.fritz import save_newsource, api
import math
import warnings
import yaml
import pathlib
from tools import scope_manage_annotation
from datetime import datetime

RADIUS_ARCSEC = 2
UPLOAD_BATCHSIZE = 10


def upload_classification(
    file: str,
    gloria,
    group_ids: list,
    taxonomy_id: int,
    classification: list,
    taxonomy_map: str,
    comment: str,
    start: int,
    stop: int,
    ztf_origin: str,
    skip_phot: bool = False,
    p_threshold: float = 0.0,
    no_match_ids: bool = False,
):
    """
    Upload labels to Fritz
    :param file: path to CSV file containing labels (str)
    :param gloria: Gloria object
    :param group_ids: list of group ids on Fritz for upload target location [int, int, ...]
    :param taxonomy_id: scope taxonomy id (int)
    :param classification: list of classifications [str, str, ...]
    :param taxonomy_map: if classification is ['read'], path to JSON file containing taxonomy mapping (str)
    :param comment: single comment to post (str)
    :param start: index in CSV file to start upload (int)
    :param stop: index in CSV file to stop upload (inclusive) (int)
    :ztf_origin: origin of uploaded ZTF data; if set, posts ztf_id to annotation (str)
    :skip_phot: if True, only upload groups and classifications (no photometry) (bool)
    :p_threshold: classification probabilties must be >= this number to post (float)
    :no_match_ids: if True, skip matching ZTF source ids when searching existing sources
    """

    # read in file to csv
    sources = pd.read_csv(file)
    columns = sources.columns

    if start is not None:
        sources = sources.loc[start:]
    else:
        start = sources.index[0]
    if stop is not None:
        sources = sources.loc[:stop]
    else:
        stop = sources.index[-1]

    # for classification "read" mode, load taxonomy map
    read_classes = False
    if classification is not None:
        if (
            (classification[0] == "read")
            | (classification[0] == 'Read')
            | (classification[0] == 'READ')
        ):
            read_classes = True
            with open(taxonomy_map, 'r') as f:
                tax_map = JSON.load(f)

            classes = [
                key for key in tax_map.keys()
            ]  # define list of columns to examine

    dict_list = []
    for index, row in sources.iterrows():
        probs = {}
        cls_list = []
        tax_dict = {}
        existing_classes = []

        if read_classes:
            row_classes = row[classes]  # limit current row to specific columns
            if p_threshold > 0.0:
                threshold_keys = row_classes.keys()[
                    row_classes >= p_threshold
                ]  # determine which dataset classifications are nonzero
            # Do not post classifications if p = 0
            else:
                threshold_keys = row_classes.keys()[row_classes > 0]

            for val in threshold_keys:
                cls = tax_map[val]['fritz_label']
                tax_id = tax_map[val]['taxonomy_id']
                if cls != 'None':  # if Fritz taxonomy value exists, add to class list
                    probs[cls] = row[val]
                    cls_list += [cls]
                    tax_dict[cls] = tax_id

        else:
            # for manual i classifications, use last i columns for probability
            for i in range(len(classification)):
                cls = classification[i]
                cls_list += [cls]
                probs[cls] = row.iloc[-1 * len(classification) + i]

        ra, dec = float(row.ra), float(row.dec)

        # Check for and assign period
        period = None  # default
        if 'period' in columns:
            period = float(row.period)
            if math.isnan(period):
                period = None
                warnings.warn('period is NaN - skipping period upload.')
        else:
            warnings.warn('period column is missing - skipping period upload.')

        # get object id
        response = api(
            "GET", f"/api/sources?&ra={ra}&dec={dec}&radius={RADIUS_ARCSEC/3600}"
        )
        data = response.json().get('data')

        existing_source = []
        src_dict = {}
        create_time_dict = {}
        obj_id = None

        if ztf_origin is not None:
            ztfid = int(row['ztf_id'])
        elif not no_match_ids:
            raise ValueError('ZTF ID/origin must be provided for IDs to be matched.')

        if data["totalMatches"] > 0:
            # get most recent ZTFJ source
            id_found = 0
            for src in data['sources']:
                if id_found:
                    break
                src_id = src['id']
                src_dict[src_id] = src

                if not no_match_ids:
                    annotations = src['annotations']
                    for annot in annotations:
                        if annot['origin'] == ztf_origin:
                            src_ztf_ids = [int(x) for x in annot['data'].values()]
                            if ztfid in src_ztf_ids:
                                id_found = 1
                                break
                else:
                    create_time = src['created_at']
                    dt_create_time = datetime.strptime(
                        create_time, '%Y-%m-%dT%H:%M:%S.%f'
                    )
                    create_time_dict[dt_create_time] = src_id

            if id_found:
                existing_source = src_dict[src_id]
                obj_id = src_id
            elif no_match_ids:
                create_time_list = [x for x in create_time_dict.keys()]
                create_time_list.sort(reverse=True)

                for t in create_time_list:
                    src_id = create_time_dict[t]
                    if src_id[:4] == 'ZTFJ':
                        existing_source = src_dict[src_id]
                        obj_id = src_id
                        break

        print(f"object {index} id:", obj_id)

        # save_newsource can only be skipped if source exists
        if (len(existing_source) == 0) | (not skip_phot):
            if (len(existing_source) == 0) & (skip_phot):
                warnings.warn('Cannot skip new source - saving.')
            obj_id = save_newsource(
                gloria,
                group_ids,
                ra,
                dec,
                period=period,
                return_id=True,
                radius=RADIUS_ARCSEC,
            )

        data_groups = []
        data_classes = []

        # check which groups source is already in
        add_group_ids = group_ids.copy()
        if len(existing_source) > 0:
            data_groups = existing_source['groups']
            data_classes = existing_source['classifications']

        # remove existing groups from list of groups
        for entry in data_groups:
            existing_group_id = entry['id']
            if existing_group_id in add_group_ids:
                add_group_ids.remove(existing_group_id)

        if len(add_group_ids) > 0:
            # save to new group_ids
            json = {"objId": obj_id, "inviteGroupIds": add_group_ids}
            response = api("POST", "/api/source_groups", json)

        # check for existing classifications
        for entry in data_classes:
            existing_classes += [entry['classification']]
        # allow classification assignment to be skipped
        if classification is not None:
            for cls in cls_list:
                if cls not in existing_classes:
                    tax = tax_dict[cls]
                    prob = probs[cls]
                    # post all non-duplicate classifications
                    json = {
                        "obj_id": obj_id,
                        "classification": cls,
                        "taxonomy_id": tax,
                        "probability": prob,
                        "group_ids": group_ids,
                    }
                    dict_list += [json]

        if comment is not None:
            # get comment text
            response_comments = api("GET", f"/api/sources/{obj_id}/comments")
            data_comments = response_comments.json().get("data")

            # check for existing comments
            existing_comments = []
            for entry in data_comments:
                existing_comments += [entry['text']]

            # post all non-duplicate comments
            if comment not in existing_comments:
                json = {
                    "text": comment,
                }
                response = api("POST", f"/api/sources/{obj_id}/comments", json)

        # Post ZTF ID as annotation
        if ztf_origin is not None:
            if obj_id not in [x for x in src_dict.keys()]:
                scope_manage_annotation.manage_annotation(
                    'POST', obj_id, group_ids, ztf_origin, 'ztf_id', str(ztfid)
                )
            else:
                source_to_update = src_dict[obj_id]
                existing_ztf_ids = [
                    x['data']
                    for x in source_to_update['annotations']
                    if x['origin'] == ztf_origin
                ]
                if len(existing_ztf_ids) == 0:
                    scope_manage_annotation.manage_annotation(
                        'POST', obj_id, group_ids, ztf_origin, 'ztf_id', str(ztfid)
                    )
                else:
                    n_existing_ztf_ids = len(existing_ztf_ids[0].keys())
                    if ztfid not in existing_ztf_ids[0].values():
                        scope_manage_annotation.manage_annotation(
                            'UPDATE',
                            obj_id,
                            group_ids,
                            ztf_origin,
                            f'ztf_id_{n_existing_ztf_ids+1}',
                            str(ztfid),
                        )

        # batch upload classifications
        if len(dict_list) != 0:
            if (((index - start) + 1) % UPLOAD_BATCHSIZE == 0) | (index == stop):
                print('uploading classifications...')
                json_classes = {'classifications': dict_list}
                api("POST", "/api/classification", json_classes)
                dict_list = []


if __name__ == "__main__":
    # setup connection to gloria to get the lightcurves
    config_path = pathlib.Path(__file__).parent.parent.absolute() / "config.yaml"
    with open(config_path) as config_yaml:
        config = yaml.load(config_yaml, Loader=yaml.FullLoader)
    gloria = Kowalski(**config['kowalski'], verbose=False)

    parser = argparse.ArgumentParser()
    parser.add_argument("-file", help="dataset")
    parser.add_argument("-group_ids", type=int, nargs='+', help="list of group ids")
    parser.add_argument(
        "-taxonomy_id",
        type=int,
        help="Fritz scope taxonomy id",
    )
    # parser.add_argument("-classification", type=str, help="name of object class")
    parser.add_argument(
        "-classification", type=str, nargs='+', help="list of object classes"
    )
    parser.add_argument(
        "-taxonomy_map",
        type=str,
        help="JSON file mapping between origin labels and Fritz taxonomy",
    )
    parser.add_argument(
        "-comment",
        type=str,
        help="Post specified string to comments for sources in file",
    )
    parser.add_argument("-start", type=int, help="Zero-based index to start uploading")
    parser.add_argument(
        "-stop",
        type=int,
        help="Index to stop uploading (inclusive)",
    )
    parser.add_argument(
        "-skip_phot",
        type=bool,
        nargs='?',
        default=False,
        const=True,
        help="Skip photometry upload, only post groups and classifications.",
    )

    parser.add_argument("-ztf_origin", type=str, help="Origin of uploaded ZTF data")

    parser.add_argument(
        "-p_threshold",
        type=float,
        default=0.0,
        help="Classification probability >= this number to upload",
    )

    parser.add_argument(
        "-no_match_ids",
        action='store_true',
        default=False,
        help="If set, match input and existing sources without using ZTF source IDs.",
    )

    args = parser.parse_args()

    # upload classification objects
    upload_classification(
        args.file,
        gloria,
        args.group_ids,
        args.taxonomy_id,
        args.classification,
        args.taxonomy_map,
        args.comment,
        args.start,
        args.stop,
        args.ztf_origin,
        args.skip_phot,
        args.p_threshold,
        args.no_match_ids,
    )
