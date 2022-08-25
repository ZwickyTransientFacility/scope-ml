#!/usr/bin/env python
import argparse
import json as JSON
import pandas as pd
from penquins import Kowalski
from time import sleep
from scope.fritz import api
import warnings


def download_classification(file: str, gloria, group_ids: list, token: str):
    """
    Download labels from Fritz
    :param file: CSV file containing obj_id column or "all" (str)
    :param gloria: Gloria object
    :param group_ids: group id on Fritz for download target location (list)
    """

    # read in CSV file
    sources = pd.read_csv(file)
    columns = sources.columns

    # create new empty columns
    sources["classification"] = None
    sources["probability"] = None
    sources["period_origin"] = None
    sources["period"] = None
    # add obj_id column if not passed in
    if 'obj_id' not in columns:
        sources["obj_id"] = None
        search_by_obj_id = False
    else:
        search_by_obj_id = True

    for index, row in sources.iterrows():
        # query objects, starting with obj_id
        data = []
        if search_by_obj_id:
            obj_id = row.obj_id
            response = api("GET", '/api/sources/%s' % obj_id, token)
            sleep(0.9)
            data = response.json().get("data")
            if len(data) == 0:
                warnings.warn('No results from obj_id search - querying by ra/dec.')

        # continue with coordinate search if obj_id unsuccsessful
        if len(data) == 0:
            if ('ra' in columns) & ('dec' in columns):
                # query by ra/dec to get object id
                ra, dec = row.ra, row.dec
                response = api(
                    "GET", f"/api/sources?&ra={ra}&dec={dec}&radius={2/3600}", token
                )
                sleep(0.9)
                data = response.json().get("data")
                obj_id = None
                if data["totalMatches"] > 0:
                    obj_id = data["sources"][0]["id"]
                    sources.at[index, 'obj_id'] = obj_id
            else:
                raise KeyError(
                    'Attemped to search by coordinates, but unable to find ra and dec columns.'
                )

        print(f"object {index} id:", obj_id)

        # if successful search, get and save labels/probabilities/period annotations to sources dataframe
        if obj_id is not None:
            response = api("GET", f"/api/sources/{obj_id}/classifications", token)
            data = response.json().get("data")

            annot_response = api(
                "GET", f'/api/sources/{obj_id}/annotations', token
            ).json()
            data_annot = annot_response.get('data')

            cls_list = ''
            prb_list = ''
            for entry in data:
                cls = entry['classification']
                prb = entry['probability']
                cls_list += cls + ';'  # same format as download from Fritz frontend
                prb_list += str(prb) + ';'
            cls_list = cls_list[:-1]  # remove trailing semicolon
            prb_list = prb_list[:-1]

            # loop through annotations, checking for periods
            origin_list = ''
            period_list = ''
            for entry in data_annot:
                annot_origin = entry['origin']
                annot_data = entry['data']

                annot_name = [x for x in annot_data.keys()][0]
                annot_value = [x for x in annot_data.values()][0]

                # if period is found, add to list
                if annot_name == 'period':
                    origin_list += annot_origin + ';'
                    period_list += str(annot_value) + ';'
            origin_list = origin_list[:-1]
            period_list = period_list[:-1]

            # store to new columns
            sources.at[index, 'classification'] = cls_list
            sources.at[index, 'probability'] = prb_list
            sources.at[index, 'period_origin'] = origin_list
            sources.at[index, 'period'] = period_list
            sources.to_csv(file, index=False)

        else:
            warnings.warn(f'Unable to find source {index} on Fritz.')

    return sources


if __name__ == "__main__":
    # setup connection to gloria to get the lightcurves
    with open('secrets.json', 'r') as f:
        secrets = JSON.load(f)
    gloria = Kowalski(**secrets['gloria'], verbose=False)

    # pass Fritz token as command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-file", help="dataset")
    parser.add_argument("-group_ids", type=int, nargs='+', help="list of group ids")
    parser.add_argument(
        "-token",
        type=str,
        help="put your Fritz token here. You can get it from your Fritz profile page",
    )
    args = parser.parse_args()

    # download object classifications in the file
    download_classification(args.file, gloria, args.group_ids, args.token)
