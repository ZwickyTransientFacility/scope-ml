import argparse
from penquins import Kowalski
import json


def get_field_ids(catalog, field=301, ccd=2, quad=3, minobs=5, skip=0, limit=10):
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

    # pass Fritz token as command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-catalog", help="catalog")
    parser.add_argument("-field", type=int, default=301, help="field number")
    parser.add_argument("-ccd", type=int, default=4, help="ccd number")
    parser.add_argument("-quad", type=int, default=3, help="quad number")
    parser.add_argument(
        "-minobs", type=int, default=20, help="minimum number of points in lc"
    )
    parser.add_argument("-skip", type=int, default=0, help="number of rows to skip")
    parser.add_argument("-limit", type=int, default=10, help="number of rows to return")
    parser.add_argument(
        "-token",
        type=str,
        help="put your Fritz token here. You can get it from your Fritz profile page",
    )
    args = parser.parse_args()

    # setup connection to gloria to get the lightcurves
    with open('secrets.json', 'r') as f:
        secrets = json.load(f)
    if args.token:
        secrets['gloria']['token'] = args.token
    gloria = Kowalski(**secrets['gloria'], verbose=False)

    #    data = get_field_ids(catalog='ZTF_sources_20210401',limit=args.limit)
    data = get_field_ids(catalog=args.catalog, limit=args.limit)
    print(data)
