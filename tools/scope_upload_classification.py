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

MAX_ATTEMPTS = 10
RADIUS_ARCSEC = 2
UPLOAD_BATCHSIZE = 10


def upload_classification(
    file: str,
    gloria,
    group_ids: list,
    taxonomy_id: int,
    classification: list,
    token: str,
    taxonomy_map: str,
    comment: str,
    start: int,
    stop: int,
    origin: str,
    skip_phot: bool = False,
):
    """
    Upload labels to Fritz
    :param file: path to CSV file containing labels (str)
    :param gloria: Gloria object
    :param group_ids: list of group ids on Fritz for upload target location [int, int, ...]
    :param taxonomy_id: scope taxonomy id (int)
    :param classification: list of classifications [str, str, ...]
    :param token: Fritz token (str)
    :param taxonomy_map: if classification is ['read'], path to JSON file containing taxonomy mapping (str)
    :param comment: single comment to post (str)
    :param start: index in CSV file to start upload (int)
    :param stop: index in CSV file to stop upload (inclusive) (int)
    :origin: origin of uploaded data, posted to id annotation (str)
    :skip_phot: if True, only upload groups and classifications (no photometry) (bool)
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
            nonzero_keys = row_classes.keys()[
                row_classes > 0
            ]  # determine which dataset classifications are nonzero

            for val in nonzero_keys:
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
            "GET",
            f"/api/sources?&ra={ra}&dec={dec}&radius={RADIUS_ARCSEC/3600}",
            token,
        )
        data = response.get('data')

        existing_source = []
        obj_id = None
        if data["totalMatches"] > 0:
            # get most recent source
            for src in data['sources']:
                src_id = src['id']
                if src_id[:4] == 'ZTFJ':
                    existing_source = src
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
                token,
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
            response = api("POST", "/api/source_groups", token, json)

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
            response_comments = api("GET", f"/api/sources/{obj_id}/comments", token)
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
                response = api("POST", f"/api/sources/{obj_id}/comments", token, json)

        # Post ZTF ID as annotation
        if origin is not None:
            ztfid = row['ztf_id']
            scope_manage_annotation.manage_annotation(
                'POST', obj_id, group_ids, token, origin, 'ztf_id', ztfid
            )

        # batch upload classifications
        if len(dict_list) != 0:
            if (((index - start) + 1) % UPLOAD_BATCHSIZE == 0) | (index == stop):
                print('uploading classifications...')
                json_classes = {'classifications': dict_list}
                api("POST", "/api/classification", token, json_classes)
                dict_list = []


if __name__ == "__main__":
    # setup connection to gloria to get the lightcurves
    config_path = pathlib.Path(__file__).parent.parent.absolute() / "config.yaml"
    with open(config_path) as config_yaml:
        config = yaml.load(config_yaml, Loader=yaml.FullLoader)
    gloria = Kowalski(**config['kowalski'], verbose=False)

    # pass Fritz token as command line argument
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
        "-token",
        type=str,
        help="put your Fritz token here. You can get it from your Fritz profile page",
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

    parser.add_argument("-origin", type=str, help="Origin of uploaded data")

    args = parser.parse_args()

    # upload classification objects
    upload_classification(
        args.file,
        gloria,
        args.group_ids,
        args.taxonomy_id,
        args.classification,
        args.token,
        args.taxonomy_map,
        args.comment,
        args.start,
        args.stop,
        args.origin,
        args.skip_phot,
    )
