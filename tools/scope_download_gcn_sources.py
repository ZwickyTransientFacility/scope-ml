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
from scope.utils import write_parquet

BASE_DIR = pathlib.Path(__file__).parent.parent.absolute()
NUM_PER_PAGE = 100

# EM+GW is group 1544
# Recommendation: query all groups when downloading, then upload all classifications to single group (e.g. EM+GW)

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
    save_filename: str = 'tools/fritzDownload/specific_ids_GCN_sources',
):
    """
    Download sources for a GCN event from Fritz (with intent to generate features/run inference on these sources)

    :param dateobs: unique dateObs of GCN event (str)
    :param group_ids: group ids to query sources [all if not specified] (list)
    :param days_range: max days past event to search for sources (float)
    :param radius_arcsec: radius [arcsec] around new sources to search for existing ZTF sources (float)
    :param save_filename: filename to save source ids/coordinates (str)

    """

    dateobs_datetime = datetime.strptime(dateobs, '%Y-%m-%dT%H:%M:%S')
    endDate_datetime = dateobs_datetime + timedelta(days=days_range)
    endDate = endDate_datetime.strftime("%Y-%m-%dT%H:%M:%S")

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
                    'localizationDateobs': dateobs,
                    'startDate': dateobs,
                    'endDate': endDate,
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

    ids = ids.drop_duplicates('_id').reset_index(drop=True)
    ids.rename({'_id': 'ztf_id', 'obj_id': 'fritz_name'}, axis=1, inplace=True)

    write_parquet(ids, str(BASE_DIR / f"{save_filename}.parquet"))

    coord_col = ids['coordinates']
    ids['radec_geojson'] = [row['radec_geojson'] for row in coord_col]
    ids.drop('coordinates', axis=1, inplace=True)

    ids = ids.set_index('ztf_id').to_dict(orient='index')
    with open(str(BASE_DIR / f"{save_filename}.json"), 'w') as f:
        json.dump(ids, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dateobs", type=str, help="unique dateObs of GCN event")
    parser.add_argument(
        "--group_ids",
        type=int,
        nargs='+',
        help="group ids to query sources (all if not specified)",
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
        default=2.0,
        help="radius around new sources to search for existing ZTF sources",
    )
    parser.add_argument(
        "--save_filename",
        type=str,
        default='tools/fritzDownload/specific_ids_GCN_sources',
        help="filename to save source ids/coordinates",
    )

    args = parser.parse_args()

    download_gcn_sources(
        dateobs=args.dateobs,
        group_ids=args.group_ids,
        days_range=args.days_range,
        radius_arcsec=args.radius_arcsec,
        save_filename=args.save_filename,
    )
