import argparse
from penquins import Kowalski
import pandas as pd
import numpy as np
import json
import sys
import os
import h5py

BASE_DIR = os.path.dirname(__file__)


def get_all_ids(
    func,
    catalog,
    field=301,
    ccd_range=16,
    quad_range=4,
    minobs=20,
    limit=0,
    verbose=2,
    output_dir=os.path.join(os.path.dirname(__file__), "output/"),
):
    '''
    Function wrapper for getting all field ids in a particular ccd and quad range

    Parameters
    ==========
    func : function
        Function for getting ids for a specific quad of a CCD for a particular ZTF field.
    catalog : str
        Catalog containing ids, CCD, quad, and light curves
    field : int
        ZTF field number
    ccd_range : int
        Range of CCD numbers starting from 1 to get the ids. Takes values from [1,16]
    quad_range : int
        Range of CCD quad numbers starting from 1. Takes values from [1,4]
    minobs : int
        Minimum points in the light curve for the object to be selected
    limit : int
        How many of the selected rows to return. Default is 10000
    output_dir : str
        Relative directory path to save output files to
    Returns
    =======
    Stores separate hdf5 files for each quad in the specified range in a directory

    USAGE: get_all_ids(get_field_ids, 'ZTF_sources_20210401',field=301,ccd_range=2,quad_range=4,\
        minobs=5,limit=2000)
    '''
    os.makedirs(output_dir, exist_ok=True)

    dct = {}
    if verbose > 0:
        dct["catalog"] = catalog
        dct["minobs"] = minobs
        dct["field"] = field
        dct["ccd_range"] = ccd_range
        dct["quad_range"] = quad_range
        dct["ccd"] = {}
        count = 0
    for ccd in range(1, ccd_range + 1):
        dct["ccd"][ccd] = {}
        dct["ccd"][ccd]["quad"] = {}
        for quad in range(1, quad_range + 1):

            if verbose > 1:
                hf = h5py.File(
                    output_dir
                    + 'data_ccd_'
                    + str(ccd).zfill(2)
                    + '_quad_'
                    + str(quad)
                    + '.h5',
                    'w',
                )
            i = 0
            ser = pd.Series(np.array([]))
            while 1:
                data = func(
                    catalog,
                    field=field,
                    ccd=ccd,
                    quad=quad,
                    minobs=minobs,
                    skip=(i * limit),
                    limit=limit,
                )
                # concat data to series containing all data
                if verbose > 1:
                    ser = pd.concat([ser, pd.Series(data)], axis=0)
                if len(data) < limit:
                    if verbose > 0:
                        length = len(data) + (i * limit)
                        count += length
                        dct["ccd"][ccd]["quad"][quad] = length
                    hf.create_dataset(
                        "dataset_ccd_"
                        + str(ccd)
                        + "_quad_"
                        + str(quad)
                        + "_field_"
                        + str(field),
                        data=ser,
                    )
                    hf.close()
                    break
                i += 1
    dct["total"] = count
    # Write metadata in this file
    f = output_dir + "meta.json"
    os.makedirs(os.path.dirname(f), exist_ok=True)
    with open(f, "w") as outfile:
        try:
            json.dump(dct, outfile)  # dump dictionary to a json file
        except Exception as e:
            print("error dumping to json, message: ", e)


# get all field ids in one file
def get_all_field_ids(
    func,
    catalog,
    field=301,
    ccd_range=16,
    quad_range=4,
    minobs=20,
    limit=0,
    verbose=2,
    output_dir=os.path.join(os.path.dirname(__file__), "output/"),
):
    '''
    Function wrapper for getting all field ids in a particular ccd and quad range
    Parameters
    ----------
    func : function
        Function for getting ids for a specific quad of a CCD for a particular ZTF field.
    catalog : str
        Catalog containing ids, CCD, quad, and light curves
    field : int
        ZTF field number
    ccd_range : int
        Range of CCD numbers starting from 1 to get the ids. Takes values from [1,16]
    quad_range : int
        Range of CCD quad numbers starting from 1. Takes values from [1,4]
    minobs : int
        Minimum points in the light curve for the object to be selected
    limit : int
        How many of the selected rows to return. Default is 10000
    output_dir : str
        Relative directory path to save output files to

    Returns
    -------
    Single hdf5 file (field_<field_number>.h5) for all the quads in the specified range. 

    USAGE: get_all_ids(get_field_ids, 'ZTF_sources_20210401',field=301,ccd_range=2,quad_range=4,\
        minobs=5,limit=2000)
    '''
    os.makedirs(output_dir, exist_ok=True)

    dct = {}
    if verbose > 0:
        dct["catalog"] = catalog
        dct["minobs"] = minobs
        dct["field"] = field
        dct["ccd_range"] = ccd_range
        dct["quad_range"] = quad_range
        dct["ccd"] = {}
        count = 0

    ser = pd.Series(np.array([]))
    for ccd in range(1, ccd_range + 1):
        dct["ccd"][ccd] = {}
        dct["ccd"][ccd]["quad"] = {}
        for quad in range(1, quad_range + 1):
            # dct["ccd"][ccd]["quad"][quad] = {}
            i = 0
            while 1:
                data = func(
                    catalog,
                    field=field,
                    ccd=ccd,
                    quad=quad,
                    minobs=minobs,
                    skip=(i * limit),
                    limit=limit,
                )
                # concat data to series containing all data
                if verbose > 1:
                    ser = pd.concat([ser, pd.Series(data)], axis=0)
                if len(data) < limit:
                    if verbose > 0:
                        length = len(data) + (i * limit)
                        count += length
                        dct["ccd"][ccd]["quad"][quad] = length
                    break
                i += 1
    if verbose > 1:
        hf = h5py.File(
            output_dir + "field_" + str(field) + '.h5',
            'w',
        )
        hf.create_dataset('dataset_field_' + str(field), data=ser)
        hf.close()

    dct["total"] = count
    # Write metadata in this file
    f = output_dir + "meta.json"
    os.makedirs(os.path.dirname(f), exist_ok=True)
    with open(f, "w") as outfile:
        try:
            json.dump(dct, outfile)  # dump dictionary to a json file
        except Exception as e:
            print("error dumping to json, message: ", e)


def get_field_ids(catalog, field=301, ccd=4, quad=3, minobs=20, skip=0, limit=10000):
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
        How many of the selected rows to return. Default is 10000
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
                "n": {"$gt": minobs},
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
    # return list(map(lambda d: d['_id'], data))


if __name__ == "__main__":
    DEFAULT_FIELD = 301
    DEFAULT_CCD = 4
    DEFAULT_QUAD = 3
    DEFAULT_MINOBS = 20
    DEFAULT_LIMIT = 10
    DEFAULT_SKIP = 0

    # pass Fritz token through secrets.json or as a command line argument
    with open(os.path.join(BASE_DIR, 'secrets.json'), 'r') as f:
        secrets = json.load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog",
        help="catalog (default: ZTF_source_features_DR5)",
        default='ZTF_source_features_DR5',
    )
    parser.add_argument(
        "--output",
        action='store',
        default='output.txt',
        type=argparse.FileType('w'),
        help="file to write output to",
    )
    parser.add_argument(
        "--output-dir",
        action='store',
        default=None,
        help="relative directory path to save output files to",
    )

    token_help = "put your Fritz token here or in the secrets file.\
                    You can get it from your Fritz profile page. This becomes\
                    an optional parameter if put in secrets file."
    if 'token' not in secrets['gloria'] or secrets['gloria']['token'] == "":
        if '-token' not in sys.argv:
            parser.add_argument("token", type=str, help=token_help)
    parser.add_argument("--token", type=str, help=token_help)
    parser.add_argument(
        "--field", type=int, default=DEFAULT_FIELD, help="field number (default 301)"
    )
    parser.add_argument(
        "--ccd", type=int, default=DEFAULT_CCD, help="ccd number (default 4)"
    )
    parser.add_argument(
        "--quad", type=int, default=DEFAULT_QUAD, help="quad number (default 3)"
    )
    parser.add_argument(
        "--ccd-range",
        type=int,
        default=16,
        help="ccd range from 1 to ccd_range (default 16, i.e. default range is [1,16])",
    )
    parser.add_argument(
        "--quad-range",
        type=int,
        default=4,
        help="quad range from 1 to quad_range (default 4, i.e. default range is [1,4])",
    )
    parser.add_argument(
        "--minobs",
        type=int,
        default=DEFAULT_MINOBS,
        help="minimum number of points in light curve (default 20)",
    )
    parser.add_argument(
        "--skip", type=int, default=0, help="number of rows to skip (default 0)"
    )
    parser.add_argument(
        "--limit", type=int, default=10000, help="number of rows to return (default 10000)"
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=2,
        help="verbose level: 0=silent, 1=basic, 2=full",
    )
    parser.add_argument(
        "--all-quads",
        action="store_true",
        default=True,
        help="if passed as argument, get ids from all 64 quads for a particular field",
    )
    parser.add_argument(
        "--whole-field",
        action="store_true",
        default=True,
        help="if passed as argument, store all ids of the field in one file",
    )

    args = parser.parse_args()

    # Set default output directory
    if args.output_dir is None:
        output_dir = "../ids/field_" + str(args.field) + "/"
    else:
        output_dir = args.output_dir + "/ids/field_" + str(args.field) + "/"

    # setup connection to gloria to get the lightcurves
    if args.token:
        secrets['gloria']['token'] = args.token
    gloria = Kowalski(**secrets['gloria'], verbose=False)

    #    data = get_field_ids(catalog='ZTF_sources_20210401',limit=args.limit)
    # print(data)

    if args.all_quads:
        if not args.whole_field:
            get_all_ids(
                get_field_ids,
                catalog=args.catalog,
                field=args.field,
                ccd_range=args.ccd_range,
                quad_range=args.quad_range,
                minobs=args.minobs,
                limit=args.limit,
                verbose=args.verbose,
                output_dir=os.path.join(os.path.dirname(__file__), output_dir),
            )
        else:
            get_all_field_ids(
                get_field_ids,
                catalog=args.catalog,
                field=args.field,
                ccd_range=args.ccd_range,
                quad_range=args.quad_range,
                minobs=args.minobs,
                limit=args.limit,
                verbose=args.verbose,
                output_dir=os.path.join(os.path.dirname(__file__), output_dir),
            )

    else:
        data, _ = get_field_ids(
            catalog=args.catalog,
            field=args.field,
            ccd=args.ccd,
            quad=args.quad,
            minobs=args.minobs,
            skip=args.skip,
            limit=args.limit,
        )
        pd.DataFrame(data).to_csv(args.output, index=False, header=False)
