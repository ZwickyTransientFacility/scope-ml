import argparse
import csv
import json
import pandas as pd
from penquins import Kowalski
from time import sleep
from scope.fritz import api


def download_classification(file, gloria, group_ids):
    """
    Upload labels to Fritz
    :param file: file containing objects without labels (csv)
    :param gloria: Gloria object
    :param group_ids: group id on Fritz for upload target location (list)
    """
    # create new empty classification column
    file["classification"] = pd.NaT

    for index, row in file.iterrows():
        # get information from objects
        ra, dec = float(row.ra), float(row.dec)

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

        # no labels if new source
        if obj_id is None:
            file.at[index, 'classification'] = None

        # save existing source
        else:
            # save to group_id
            json = {"objId": obj_id, "inviteGroupIds": group_ids}
            response = api("POST", "/api/source_groups", args.token, json)

        # get classificaiton
        response = api("GET", f"/api/sources/{obj_id}/classifications", args.token)
        data = response.json().get("data")

        # store to new column
        file.at[index, 'classification'] = data[0]["classification"]

    return file


if __name__ == "__main__":
    # setup connection to gloria to get the lightcurves
    with open('secrets.json', 'r') as f:
        secrets = json.load(f)
    gloria = Kowalski(**secrets['gloria'], verbose=False)

    # pass Fritz token as command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-file", help="dataset")
    parser.add_argument("-group_ids", help="list of group ids")
    parser.add_argument(
        "-token",
        type=str,
        help="put your Fritz token here. You can get it from your Fritz profile page",
    )
    args = parser.parse_args()

    # read in file to csv
    with open(args.file) as f:
        data = csv.reader(f)
        sample = pd.DataFrame(data)

    # download object classifications in the file
    download_classification(sample, gloria, args.group_ids)
