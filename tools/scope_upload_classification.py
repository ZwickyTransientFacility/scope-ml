#!/usr/bin/env python
import argparse
import json as JSON
import pandas as pd
from penquins import Kowalski
from time import sleep
from scope.fritz import save_newsource, api
import math
import warnings


def upload_classification(
    file: str,
    gloria,
    group_ids: list,
    taxonomy_id: int,
    classification: list,
    token: str,
    taxonomy_map: str,
    comment: str,
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
    """

    # read in file to csv
    sources = pd.read_csv(file)
    columns = sources.columns

    for index, row in sources.iterrows():
        probs = {}
        cls_list = []
        existing_classes = []

        # for classification "read" mode, load taxonomy map
        if classification is not None:
            if (
                (classification[0] == "read")
                | (classification[0] == 'Read')
                | (classification[0] == 'READ')
            ):
                with open(taxonomy_map, 'r') as f:
                    tax_map = JSON.load(f)

                classes = [
                    key for key in tax_map.keys()
                ]  # define list of columns to examine
                row_classes = row[classes]  # limit current row to specific columns
                nonzero_keys = row_classes.keys()[
                    row_classes > 0
                ]  # determine which dataset classifications are nonzero

                for val in nonzero_keys:
                    cls = tax_map[val]
                    if (
                        cls != 'None'
                    ):  # if Fritz taxonomy value exists, add to class list
                        probs[cls] = row[val]
                        cls_list += [cls]

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
        response = api("GET", f"/api/sources?&ra={ra}&dec={dec}&radius={2/3600}", token)
        sleep(0.9)
        data = response.json().get("data")
        obj_id = None
        if data["totalMatches"] > 0:
            obj_id = data["sources"][0]["id"]
        print(f"object {index} id:", obj_id)

        # save new source
        if obj_id is None:
            obj_id = save_newsource(
                gloria, group_ids, ra, dec, token, period=period, return_id=True
            )
            data_groups = []
            data_classes = []
        # save existing source
        else:
            # check which groups source is already in
            add_group_ids = group_ids.copy()
            response = api("GET", f"/api/sources/{obj_id}", token)
            data = response.json().get("data")

            data_groups = data['groups']
            data_classes = data['classifications']

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
                    prob = probs[cls]
                    # post all non-duplicate classifications
                    json = {
                        "obj_id": obj_id,
                        "classification": cls,
                        "taxonomy_id": taxonomy_id,
                        "probability": prob,
                        "group_ids": group_ids,
                    }
                    response = api("POST", "/api/classification", token, json)

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


if __name__ == "__main__":
    # setup connection to gloria to get the lightcurves
    # secrets file requires Kowalski username/password or token, host, port, and protocol
    with open('secrets.json', 'r') as f:
        secrets = JSON.load(f)
    gloria = Kowalski(**secrets['gloria'], verbose=False)

    # pass Fritz token as command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-file", help="dataset")
    parser.add_argument("-group_ids", type=int, nargs='+', help="list of group ids")
    parser.add_argument(
        "-taxonomy_id",
        type=int,
        nargs='?',
        default=9,
        const=9,
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
    )
