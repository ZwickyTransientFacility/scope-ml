import argparse
from penquins import Kowalski
import pandas as pd
import json
import sys


def get_field_ids(catalog, field=301, ccd=4, quad=3, minobs=20, skip=0, limit=10):
    '''Get ids for a specific quad of a CCD for a particular ZTF field.
    Parameters
    ----------
    catalog : str
        Catalog containing ids, CCD, quad, and light curves
    field : int
        ZTF field number
    ccd : int
        CCD number [1,16] (not checked)
    quad : int
        CCD quad number [1,4] (not checked)
    minobs : int
        Minimum points in the light curve for the object to be selected
    skip : int
        How many of the selected rows to skip
        Along with limit this can be used to loop over a quad in chunks
    limit : int
        How many of the selected rows to return. Default is 10
    Returns
    -------
    ids : list
        A list of ids

    USAGE: data = get_field_ids('ZTF_sources_20210401',field=301,ccd=2,quad=3,\
        minobs=5,skip=0,limit=20)
    '''

    if limit == 0:
        limit = 10000000000

    q = {
        'query_type': 'find',
        'query': {
            'catalog': catalog,
            'filter': {
                "field": {"$eq": field},
                "ccd": {"$eq": ccd},
                "quad": {"$eq": quad},
                "nobs": {"$gt": minobs},
            },
            "projection": {
                "_id": 1,
            },
        },
        "kwargs": {"limit": limit, "skip": skip},
    }

    r = gloria.query(q)
    data = r.get('data')
    return [data[i]['_id'] for i in range(len(data))]


if __name__ == "__main__":

    # pass Fritz token through secrets.json or as a command line argument
    with open('secrets.json', 'r') as f:
        secrets = json.load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-catalog",
        help="catalog (default: ZTF_sources_20210401)",
        default='ZTF_sources_20210401',
    )
    parser.add_argument(
        "output",
        action='store',
        type=argparse.FileType('w'),
        help="file to write output to",
    )

    token_help = "put your Fritz token here or in the secrets file.\
                    You can get it from your Fritz profile page. This becomes\
                    an optional parameter if put in secrets file."
    if 'token' not in secrets['gloria'] or secrets['gloria']['token'] == "":
        if '-token' not in sys.argv:
            parser.add_argument("token", type=str, help=token_help)
    parser.add_argument("-token", type=str, help=token_help)
    parser.add_argument(
        "-field", type=int, default=301, help="field number (default 301)"
    )
    parser.add_argument("-ccd", type=int, default=4, help="ccd number (default 4)")
    parser.add_argument("-quad", type=int, default=3, help="quad number (default 3)")
    parser.add_argument(
        "-minobs",
        type=int,
        default=20,
        help="minimum number of points in lc (default 20)",
    )
    parser.add_argument(
        "-skip", type=int, default=0, help="number of rows to skip (default 0)"
    )
    parser.add_argument(
        "-limit", type=int, default=10, help="number of rows to return (default 10)"
    )

    args = parser.parse_args()

    # setup connection to gloria to get the lightcurves
    if args.token:
        secrets['gloria']['token'] = args.token
    gloria = Kowalski(**secrets['gloria'], verbose=False)

    #    data = get_field_ids(catalog='ZTF_sources_20210401',limit=args.limit)
    # print(data)
    data = get_field_ids(catalog=args.catalog, limit=args.limit)
    pd.DataFrame(data).to_csv(args.output, index=False, header=False)
