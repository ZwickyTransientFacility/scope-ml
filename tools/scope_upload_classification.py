#!/usr/bin/env python
import sys
import argparse
import json
import pandas as pd
from penquins import Kowalski
from time import sleep
from scope.fritz import save_newsource, api

sys.path.append('../../scope')


def upload_classification(
    file, gloria, group_ids, taxonomy_id: int, classification: str
):
    """
    Upload labels to Fritz
    :param file: file containing labels (csv)
    :param gloria: Gloria object
    :param group_ids: group id on Fritz for upload target location (int int int ...)
    :param taxonomy_id: scope taxonomy id (int)
    :param classification: classified object (str)
    """

    for index, row in file.iterrows():
        # get information from objects
        prob = row.iloc[-1]
        ra, dec, period = float(row.ra), float(row.dec), float(row.period)

        # get object id
        response = api(
            "GET", f"/api/sources?&ra={ra}&dec={dec}&radius={2/3600}", args.token
        )
        sleep(0.9)
        data = response.json().get("data")
        obj_id = None
        if data["totalMatches"] > 0:
            obj_id = data["sources"][0]["id"]
        print(f"object {index} id:", obj_id)

        # save new source
        if obj_id is None:
            obj_id = save_newsource(
                gloria, group_ids, ra, dec, args.token, period=period, return_id=True
            )

        # save existing source
        else:
            # check which groups source is already in
            add_group_ids = group_ids.copy()
            response = api("GET", f"/api/sources/{obj_id}/groups", args.token)
            data = response.json().get("data")

            # remove existing groups from list of groups
            for entry in data:
                existing_group_id = entry['id']
                if existing_group_id in add_group_ids:
                    add_group_ids.remove(existing_group_id)

            if len(add_group_ids) > 0:
                # save to new group_ids
                json = {"objId": obj_id, "inviteGroupIds": add_group_ids}
                response = api("POST", "/api/source_groups", args.token, json)

        # upload classification
        # allow for multiple classifications at once?
        json = {
            "obj_id": obj_id,
            "classification": classification,
            "taxonomy_id": taxonomy_id,
            "probability": prob,
            "group_ids": group_ids,
        }
        response = api("POST", "/api/classification", args.token, json)


if __name__ == "__main__":
    # setup connection to gloria to get the lightcurves
    # secrets file requires Kowalski username/password or token, host, port, and protocol
    with open('secrets.json', 'r') as f:
        secrets = json.load(f)
    gloria = Kowalski(**secrets['gloria'], verbose=False)

    # pass Fritz token as command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-file", help="dataset")
    parser.add_argument("-group_ids", type=int, nargs='+', help="list of group ids")
    parser.add_argument("-taxonomy_id", type=int, help="Fritz scope taxonomy id")
    parser.add_argument("-classification", type=str, help="name of object class")
    parser.add_argument(
        "-token",
        type=str,
        help="put your Fritz token here. You can get it from your Fritz profile page",
    )
    args = parser.parse_args()

    # read in file to csv
    sample = pd.read_csv(args.file)

    # upload classification objects
    upload_classification(
        sample, gloria, args.group_ids, args.taxonomy_id, args.classification
    )
