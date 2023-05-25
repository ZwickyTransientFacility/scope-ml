#!/usr/bin/env python
import argparse
import pathlib
import yaml
import os
from penquins import Kowalski
from datetime import datetime, timedelta
import numpy as np
import json
from scope.fritz import api
from tools.get_quad_ids import get_cone_ids

BASE_DIR = pathlib.Path(__file__).parent.parent.absolute()
NUM_PER_PAGE = 100

# EM+GW is group 1544

config_path = BASE_DIR / "config.yaml"
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

kowalski_instances = {'kowalski': kowalski, 'gloria': gloria, 'melman': melman}


def download_gcn_sources(
    dateobs: str,
    group_ids: list = [],
    days_range: float = 7.0,
    radius_arcsec: float = 2.0,
    save_filename: str = 'tools/fritzDownload/specific_ids_GCN_sources.json',
):

    dateobs_datetime = datetime.strptime(dateobs, '%Y-%m-%dT%H:%M:%S')
    endDate_datetime = dateobs_datetime + timedelta(days=days_range)
    endDate = endDate_datetime.strftime("%Y-%m-%dT%H:%M:%S")

    # response = api('GET', f'/api/gcn_event/{dateobs}')
    # data = response.json().get('data')
    # if len(data) > 0:
    #     gcn_event = data
    #     most_recent_localization = gcn_event['localizations'][0]
    #     #loc_id = most_recent_localization['properties']['localization_id']
    #     print(most_recent_localization)
    #     print(type(most_recent_localization))
    #     print(len(most_recent_localization))

    data = {
        'group_ids': group_ids,
        'localizationDateobs': dateobs,
        'startDate': dateobs,
        'endDate': endDate,
        'numPerPage': NUM_PER_PAGE,
    }
    response = api('GET', '/api/sources', data=data)
    data = response.json().get('data')
    status = response.json().get('status')

    # Determine number of pages
    allMatches = data.get('totalMatches')
    nPerPage = data.get('numPerPage')
    pages = int(np.ceil(allMatches / nPerPage))

    gcn_source_dct = {}

    # iterate over all pages in results
    if (allMatches is not None) & (allMatches > 0):
        print(f'Downloading {allMatches} sources...')
        for pageNum in range(1, pages + 1):
            print(f'Page {pageNum} of {pages}...')
            page_response = api(
                "GET",
                '/api/sources',
                {
                    "group_ids": group_ids,
                    'numPerPage': NUM_PER_PAGE,
                    'pageNumber': pageNum,
                },  # page numbers start at 1
            )
            page_data = page_response.json().get('data')

            for src in page_data['sources']:
                id = src['id']
                ra = src['ra']
                dec = src['dec']
                gcn_source_dct[id] = {'gcn_source_coordinates': [ra, dec]}

    else:
        if allMatches is None:
            print('Check if query successfully completed. Status: ', status)
        elif allMatches == 0:
            print('No sources found.')

    print('Finding ZTF sources around GCN source coords...')

    obj_id_list = [x for x in gcn_source_dct.keys()]
    ra_list = [x['gcn_source_coordinates'][0] for x in gcn_source_dct.values()]
    dec_list = [x['gcn_source_coordinates'][1] for x in gcn_source_dct.values()]

    ids = get_cone_ids(
        obj_id_list=obj_id_list,
        ra_list=ra_list,
        dec_list=dec_list,
        catalog=source_catalog,
        kowalski_instance=kowalski_instances['melman'],
        max_distance=radius_arcsec,
        distance_units='arcsec',
        get_coords=True,
    )

    ids = ids.set_index('_id').to_dict(orient='index')
    with open(str(BASE_DIR / save_filename), 'w') as f:
        json.dump(ids, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dateobs", type=str, help="dataset")
    parser.add_argument("--group_ids", type=int, nargs='+', help="dataset")
    parser.add_argument("--days_range", type=float, default=7.0, help="dataset")
    parser.add_argument("--radius_arcsec", type=float, default=2.0, help="dataset")

    args = parser.parse_args()

    download_gcn_sources(
        dateobs=args.dateobs,
        group_ids=args.group_ids,
        days_range=args.days_range,
        radius_arcsec=args.radius_arcsec,
    )
