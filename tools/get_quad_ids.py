from penquins import Kowalski
import json


def get_field_ids(catalog, field=301, ccd=2, quad=3, minobs=5, n=10):
    '''Get ids for a specific quad of a CCD for a particular ZTF field.

    USAGE: data = get_field_ids('ZTF_sources_20210401',field=301,ccd=2,quad=3,minobs=5,n=20)
    It is assumed that you are connected to the database already.
    ccd range: [1,16] (not checked)
    quad range: p1,4] (not checked)
    Defualt minobs is 5
    Default number of ids to return is 10 (use 0 for all rows)
    Output: list of ids
    '''

    if n == 0:
        limit = 10000000000
    else:
        limit = n

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
        "kwargs": {"limit": limit},
    }

    r = gloria.query(q)
    data = r.get('data')
    return [data[i]['_id'] for i in range(len(data))]


if __name__ == "__main__":
    # setup connection to gloria to get the lightcurves
    with open('secrets.json', 'r') as f:
        secrets = json.load(f)
    gloria = Kowalski(**secrets['gloria'], verbose=False)

    data = get_field_ids(
        'ZTF_sources_20210401', field=301, ccd=2, quad=3, minobs=5, n=20
    )
