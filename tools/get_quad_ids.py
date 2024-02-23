#!/usr/bin/env python
import argparse
from penquins import Kowalski
import pandas as pd
import numpy as np
import json
import os
import h5py
import pathlib
from scope.utils import parse_load_config

BASE_DIR = pathlib.Path.cwd()
DEFAULT_FIELD = 301
DEFAULT_CCD = 4
DEFAULT_QUAD = 3
DEFAULT_CCD_RANGE = [1, 16]
DEFAULT_QUAD_RANGE = [1, 4]
DEFAULT_MINOBS = 20
DEFAULT_LIMIT = 10000
DEFAULT_SKIP = 0
DEFAULT_VERBOSE = 2
config = parse_load_config()

# use tokens specified as env vars (if exist)
kowalski_token_env = os.environ.get("KOWALSKI_INSTANCE_TOKEN")
gloria_token_env = os.environ.get("GLORIA_INSTANCE_TOKEN")
melman_token_env = os.environ.get("MELMAN_INSTANCE_TOKEN")

# Set up Kowalski instance connection
if kowalski_token_env is not None:
    config["kowalski"]["hosts"]["kowalski"]["token"] = kowalski_token_env
if gloria_token_env is not None:
    config["kowalski"]["hosts"]["gloria"]["token"] = gloria_token_env
if melman_token_env is not None:
    config["kowalski"]["hosts"]["melman"]["token"] = melman_token_env

timeout = config['kowalski']['timeout']

hosts = [
    x
    for x in config['kowalski']['hosts']
    if config['kowalski']['hosts'][x]['token'] is not None
]
instances = {
    host: {
        'protocol': config['kowalski']['protocol'],
        'port': config['kowalski']['port'],
        'host': f'{host}.caltech.edu',
        'token': config['kowalski']['hosts'][host]['token'],
    }
    for host in hosts
}

kowalski_instances = Kowalski(timeout=timeout, instances=instances)


def get_ids_loop(
    func,
    catalog,
    kowalski_instances=kowalski_instances,
    field=296,
    ccd_range=[1, 16],
    quad_range=[1, 4],
    minobs=20,
    limit=10000,
    verbose=2,
    output_dir=None,
    whole_field=False,
    save=True,
    get_coords=False,
    stop_early=False,
):
    '''
        Function wrapper for getting ids in a particular ccd and quad range

        Parameters
        ----------
        func : function
            Function for getting ids for a specific quad of a CCD for a particular ZTF field.
        catalog : str
            Catalog containing ids, CCD, quad, and light curves
        kowalski_instances:
            Authenticated instances of kowalski. Defaults to config-specified info.
        field : int
            ZTF field number
        ccd_range : int
            CCD number or range of numbers, starting from 1 to get the ids. Takes values from [1,16]
        quad_range : int
            CCD quad number or range of numbers, starting from 1. Takes values from [1,4]
        minobs : int
            Minimum points in the light curve for the object to be selected
        limit : int
            How many of the selected rows to return. Default is 10000
        output_dir : str
            Relative directory path to save output files to
        whole_field: bool
            If True, save one file containing all field ids. Otherwise, save files for each ccd/quad pair
        save: bool
            If True, save results (either by ccd/quad or whole field)
        get_coords: bool
            If True, return dictionary linking ids and object geojson coordinates
        stop_early: bool
            If True, stop loop when number of sources reaches limit

        Returns
        -------
        Single or separate hdf5 files (field_<field_number>.h5 or data_<ccd_number>_quad_<quad_number>.h5)
        for all the quads in the specified range.

        USAGE: get_ids_loop(get_field_ids, 'ZTF_sources_20210401',field=301,ccd_range=[1,2],quad_range=[2,4],\
            minobs=5,limit=2000, whole_field=False)
                get_ids_loop(get_field_ids, 'ZTF_sources_20210401',field=301,ccd_range=1,quad_range=2,\
            minobs=5,limit=2000, whole_field=False)
        '''
    if output_dir is None:
        output_dir = os.path.join(str(BASE_DIR), "ids/field_" + str(field) + "/")

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
    lst = []
    save_individual = (save) & (not whole_field)

    if isinstance(ccd_range, int):
        ccd_range = [ccd_range, ccd_range]
    if isinstance(quad_range, int):
        quad_range = [quad_range, quad_range]

    for ccd in range(ccd_range[0], ccd_range[1] + 1):
        dct["ccd"][ccd] = {}
        dct["ccd"][ccd]["quad"] = {}
        for quad in range(quad_range[0], quad_range[1] + 1):

            i = 0
            if get_coords:
                quaddata = {}
                quaddata_keys = []
                quaddata_values = []
            else:
                quaddata = []

            while True:
                data = func(
                    catalog,
                    kowalski_instances=kowalski_instances,
                    field=field,
                    ccd=ccd,
                    quad=quad,
                    minobs=minobs,
                    skip=(i * limit),
                    limit=limit,
                    save=save_individual,
                    output_dir=output_dir,
                    get_coords=get_coords,
                )
                if get_coords:
                    quaddata_keys += [x for x in data.keys()]
                    quaddata_values += [x for x in data.values()]
                    quaddata.update(dict(zip(quaddata_keys, quaddata_values)))
                else:
                    quaddata += [x for x in data]

                # concat data to series containing all data
                if verbose > 1:
                    ser = pd.concat([ser, pd.Series(data)], axis=0)
                if (len(data) < limit) | ((len(data) == limit) & stop_early):
                    if verbose > 0:
                        length = len(data) + (i * limit)
                        count += length
                        dct["ccd"][ccd]["quad"][quad] = length
                        lst += [quaddata]
                    break
                i += 1
    if (verbose > 1) & (whole_field) & (save):
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

    print(f'Found {len(ser)} total IDs.')
    return ser, lst


def get_cone_ids(
    obj_id_list: list,
    ra_list: list,
    dec_list: list,
    catalog: str = 'ZTF_source_features_DR16',
    kowalski_instances=kowalski_instances,
    max_distance: float = 2.0,
    distance_units: str = "arcsec",
    limit_per_query: int = 1000,
    get_coords: bool = False,
) -> pd.DataFrame:
    """Cone search ZTF ID for a set of given positions

    :param obj_id_list: unique object identifiers (list of str)
    :param ra_list: RA in deg (list of float)
    :param dec_list: Dec in deg (list of float)
    :param catalog: catalog to query
    :kowalski_instances: Authenticated instances of kowalski. Defaults to config-specified info.
    :param max_distance: float
    :param distance_units: arcsec | arcmin | deg | rad
    :param limit_per_query: max number of sources in a query (int)
    :param get_coords: flag to get radec_geojson coordinates from Kowalski (bool)

    :return: DataFrame with ZTF ids paired with input obj_ids
    """

    if limit_per_query == 0:
        limit_per_query = 10000000000

    id = 0
    data = {}

    while True:
        selected_obj_id = obj_id_list[
            id * limit_per_query : min(len(obj_id_list), (id + 1) * limit_per_query)
        ]
        selected_ra = ra_list[
            id * limit_per_query : min(len(obj_id_list), (id + 1) * limit_per_query)
        ]
        selected_dec = dec_list[
            id * limit_per_query : min(len(obj_id_list), (id + 1) * limit_per_query)
        ]

        radec = [(selected_ra[i], selected_dec[i]) for i in range(len(selected_obj_id))]

        projection = {"_id": 1}
        if get_coords:
            projection.update({"coordinates.radec_geojson.coordinates": 1})

        query = {
            "query_type": "cone_search",
            "query": {
                "object_coordinates": {
                    "radec": dict(zip(selected_obj_id, radec)),
                    "cone_search_radius": max_distance,
                    "cone_search_unit": distance_units,
                },
                "catalogs": {
                    catalog: {
                        "filter": {},
                        "projection": projection,
                    }
                },
            },
        }
        responses = kowalski_instances.query(query=query)

        for name in responses.keys():
            if len(responses[name]) > 0:
                response = responses[name]
                if response.get("status", "error") == "success":
                    temp_data = response.get("data").get(catalog)
                    if temp_data is not None:
                        data.update(temp_data)
                    else:
                        print(response)
                        raise ValueError(f"No data found for obj_ids {selected_obj_id}")

        if ((id + 1) * limit_per_query) >= len(obj_id_list):
            print(f'{len(obj_id_list)} done')
            break
        id += 1
        if (id * limit_per_query) % limit_per_query == 0:
            print(id * limit_per_query, "done")

    for obj in data.keys():
        vals = data[obj]
        for v in vals:
            v['obj_id'] = obj.replace('_', '.')

    features_all = [v for k, v in data.items() if len(v) > 0]

    df = pd.DataFrame.from_records([f for x in features_all for f in x])

    return df


def get_field_ids(
    catalog,
    kowalski_instances=kowalski_instances,
    field=301,
    ccd=4,
    quad=3,
    minobs=20,
    skip=0,
    limit=10000,
    save=False,
    output_dir=None,
    get_coords=False,
):
    '''Get ids for a specific quad of a CCD for a particular ZTF field.
    Parameters
    ----------
    catalog : str
        Catalog containing ids, CCD, quad, and light curves
    kowalski_instances:
        Authenticated instances of kowalski. Defaults to config-specified info.
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
    get_coords: bool
            If True, return dictionary linking ids and object geojson coordinates
    Returns
    -------
    ids : list
        A list of ids

    USAGE: data = get_field_ids('ZTF_sources_20210401',field=301,ccd=2,quad=3,\
        minobs=5,skip=0,limit=20)
    '''

    if limit == 0:
        limit = 10000000000

    filter = {
        "field": {"$eq": field},
        "ccd": {"$eq": ccd},
        "quad": {"$eq": quad},
    }
    if minobs > 0:
        if 'sources' in catalog:
            filter["nobs"] = {"$gte": minobs}
        elif 'features' in catalog:
            filter["n"] = {"$gte": minobs}

    projection = {"_id": 1}
    if get_coords:
        projection['coordinates.radec_geojson.coordinates'] = 1

    q = {
        'query_type': 'find',
        'query': {
            'catalog': catalog,
            'filter': filter,
            "projection": projection,
        },
        "kwargs": {"limit": limit, "skip": skip},
    }

    responses = kowalski_instances.query(q)

    for name in responses.keys():
        if len(responses[name]) > 0:
            response = responses[name]
            if response.get("status", "error") == "success":
                data = response.get("data")
                if data is not None:
                    ids = [data[i]['_id'] for i in range(len(data))]

    if get_coords:
        coords = [data[i]['coordinates'] for i in range(len(data))]

    if save:
        print(f"Found {len(ids)} results to save.")

        pd.DataFrame(ids).to_csv(
            os.path.join(
                output_dir,
                "data_ccd_"
                + str(ccd)
                + "_quad_"
                + str(quad)
                + "_field_"
                + str(field)
                + ".csv",
            ),
            index=False,
            header=False,
        )

        hf = h5py.File(
            output_dir + 'data_ccd_' + str(ccd).zfill(2) + '_quad_' + str(quad) + '.h5',
            'w',
        )
        hf.create_dataset(
            "dataset_ccd_" + str(ccd) + "_quad_" + str(quad) + "_field_" + str(field),
            data=ids,
        )
        hf.close()

    if get_coords:
        return dict(zip(ids, coords))
    else:
        return ids


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog",
        help="catalog (default: ZTF_source_features_DR16)",
        default='ZTF_source_features_DR16',
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
    parser.add_argument(
        "--field", type=int, default=DEFAULT_FIELD, help="field number (default 301)"
    )
    parser.add_argument(
        "--ccd-range",
        type=int,
        nargs='+',
        default=DEFAULT_CCD_RANGE,
        help="ccd range; single int or list of two ints between 1 and 16 (default range is [1,16])",
    )
    parser.add_argument(
        "--quad-range",
        type=int,
        nargs='+',
        default=DEFAULT_QUAD_RANGE,
        help="quad range; single int or list of two ints between 1 and 4 (default range is [1,4])",
    )
    parser.add_argument(
        "--minobs",
        type=int,
        default=DEFAULT_MINOBS,
        help="minimum number of points in light curve (default 20)",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=DEFAULT_SKIP,
        help="number of rows to skip (default 0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="number of rows to return (default 10000)",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=DEFAULT_VERBOSE,
        help="verbose level: 0=silent, 1=basic, 2=full",
    )
    parser.add_argument(
        "--multi-quads",
        action="store_true",
        help="if passed as argument, get ids from multiple quads for a particular field and save in separate files",
    )
    parser.add_argument(
        "--whole-field",
        action="store_true",
        help="if passed as argument, store all ids of the field in one file",
    )

    return parser()


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()

    # Set default output directory
    if args.output_dir is None:
        output_dir = os.path.join(str(BASE_DIR), "ids/field_" + str(args.field) + "/")
    else:
        output_dir = args.output_dir + "/ids/field_" + str(args.field) + "/"
    os.makedirs(output_dir, exist_ok=True)

    if (args.multi_quads) | (args.whole_field):
        if args.whole_field:
            print(
                f'Saving single file for entire field ({args.field}) across ccd/quadrant range.'
            )
        else:
            print(
                f'Saving multiple files for ccd/quadrant pairs in range {args.ccd_range}, {args.quad_range}.'
            )
        get_ids_loop(
            get_field_ids,
            catalog=args.catalog,
            kowalski_instances=kowalski_instances,
            field=args.field,
            ccd_range=args.ccd_range,
            quad_range=args.quad_range,
            minobs=args.minobs,
            limit=args.limit,
            verbose=args.verbose,
            output_dir=os.path.join(str(BASE_DIR), output_dir),
            whole_field=args.whole_field,
            save=True,
        )

    else:
        # Handle different types of input for ccd/quad_range
        if isinstance(args.ccd_range, list):
            ccd = args.ccd_range[0]
        else:
            ccd = args.ccd_range
        if isinstance(args.quad_range, list):
            quad = args.quad_range[0]
        else:
            quad = args.quad_range
        print(
            f'Saving up to {args.limit} results for single ccd/quadrant pair ({ccd},{quad}), skipping {args.skip} rows...'
        )

        _ = get_field_ids(
            catalog=args.catalog,
            kowalski_instances=kowalski_instances,
            field=args.field,
            ccd=ccd,
            quad=quad,
            minobs=args.minobs,
            skip=args.skip,
            limit=args.limit,
            save=True,
            output_dir=os.path.join(str(BASE_DIR), output_dir),
        )
