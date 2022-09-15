import argparse
from penquins import Kowalski
import pandas as pd
import json
import sys
import os
import h5py
from tqdm import tqdm
import time


def get_all_ids(
    func,
    catalog,
    field=301,
    ccd_range=16,
    quad_range=4,
    minobs=20,
    limit=10000,
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
    Directory containing hdf5 files for each quad in the specified range

    USAGE: get_all_ids(get_field_ids, 'ZTF_sources_20210401',field=301,ccd_range=2,quad_range=4,\
        minobs=5,limit=2000)
    '''
    os.makedirs(output_dir, exist_ok=True)

    # Write metadata in this file
    f = open(output_dir + "data.txt", "w")
    if verbose > 0:
        string = (
            "Catalog: "
            + catalog
            + "\nMin points: "
            + str(minobs)
            + "\nField: "
            + str(field)
            + "\nCCD Range: [1,"
            + str(ccd_range)
            + "] "
            + "\nQuad Range: [1,"
            + str(quad_range)
            + "]\n.\n.\n.\n\n"
        )
        f.writelines(string)

    for ccd in tqdm(range(1, ccd_range + 1), disable=(not (verbose > 1))):
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
            total_time = 0
            while 1:
                data, time_taken = func(
                    catalog,
                    field=field,
                    ccd=ccd,
                    quad=quad,
                    minobs=minobs,
                    skip=(i * limit),
                    limit=limit,
                )
                # Write to hdf5
                if verbose > 1:
                    dset = hf.create_dataset('dataset_' + str(i).zfill(3), data=data)
                    dset.attrs['exec_time'] = time_taken  # add attribute for time taken
                total_time += time_taken
                if len(data) < limit:
                    if verbose > 0:
                        length = len(data) + (i * limit)
                        string = (
                            "\nCCD: "
                            + str(ccd)
                            + " Quad: "
                            + str(quad)
                            + "\nNumber of ids: "
                            + str(length)
                            + "\nExecution Time: "
                            + str(round(total_time * 1000, 4))
                            + " ms\n"
                        )
                        f.writelines(string)  # Write metadata for each quad
                    break
                i += 1
            if verbose > 1:
                hf.close()
    f.close()


def gettime(func):
    '''
    Wrapper function that reports the execution time of func.
    '''

    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        return result, end - start

    return wrap


@gettime
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
    time_taken : float
        Execution time

    USAGE: data, time_taken = get_field_ids('ZTF_sources_20210401',field=301,ccd=2,quad=3,\
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
        "-output",
        action='store',
        default='output.txt',
        type=argparse.FileType('w'),
        help="file to write output to",
    )
    parser.add_argument(
        "-output_dir",
        action='store',
        default="output/",
        help="relative directory path to save output files to",
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
        "-ccd_range",
        type=int,
        default=16,
        help="ccd range from 1 to ccd_range (default 16 -> default ccd range [1,16])",
    )
    parser.add_argument(
        "-quad_range",
        type=int,
        default=4,
        help="quad range from 1 to quad_range (default 4 -> default quad range [1,4])",
    )
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
        "-limit", type=int, default=10000, help="number of rows to return (default 10)"
    )
    parser.add_argument("-verbose", type=int, default=2, help="how much data to store")
    parser.add_argument(
        "--all-quads",
        action="store_true",
        default=False,
        help="if passed as argument, get all quads for a particular field",
    )

    args = parser.parse_args()

    # setup connection to gloria to get the lightcurves
    if args.token:
        secrets['gloria']['token'] = args.token
    gloria = Kowalski(**secrets['gloria'], verbose=False)

    #    data = get_field_ids(catalog='ZTF_sources_20210401',limit=args.limit)
    # print(data)

    if args.all_quads:
        get_all_ids(
            get_field_ids,
            catalog=args.catalog,
            field=args.field,
            ccd_range=args.ccd_range,
            quad_range=args.quad_range,
            minobs=args.minobs,
            limit=args.limit,
            verbose=args.verbose,
            output_dir=os.path.join(os.path.dirname(__file__), args.output_dir),
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
