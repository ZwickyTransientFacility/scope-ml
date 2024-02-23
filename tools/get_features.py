#!/usr/bin/env python
import numpy as np
import pandas as pd
import pathlib
from penquins import Kowalski
from typing import List
import os
import time
import h5py
from scope.utils import write_parquet, impute_features, parse_load_config
from datetime import datetime
import pyarrow.dataset as ds
import argparse

BASE_DIR = pathlib.Path.cwd()
config = parse_load_config()

JUST = 50
DEFAULT_FIELD = 291
DEFAULT_CCD_RANGE = [1, 16]
DEFAULT_QUAD_RANGE = [1, 4]
DEFAULT_LIMIT = 1000
DEFAULT_SAVE_BATCHSIZE = 100000
features_catalog = config['kowalski']['collections']['features']
DEFAULT_CATALOG = features_catalog

# Access datatypes in config file
all_feature_names_config = config["features"]["ontological"]
dtype_dict = {
    key: all_feature_names_config[key]['dtype'] for key in all_feature_names_config
}

# Only features listed in config (regardless of include:) will be downloaded
projection_dict = {key: 1 for key in all_feature_names_config}

period_suffix = config['features']['info']['period_suffix']
# Rename periodic feature columns if suffix provided in config (features: info: period_suffix:)
if not ((period_suffix is None) | (period_suffix == 'None')):
    all_feature_names = [x for x in all_feature_names_config.keys()]
    periodic_bool = [all_feature_names_config[x]['periodic'] for x in all_feature_names]
    for j, name in enumerate(all_feature_names):
        if periodic_bool[j]:
            all_feature_names[j] = f'{name}_{period_suffix}'

    dtype_values = [x for x in dtype_dict.values()]
    projection_values = [x for x in projection_dict.values()]

    dtype_dict = {
        all_feature_names[i]: dtype_values[i] for i in range(len(dtype_values))
    }
    projection_dict = {
        all_feature_names[i]: projection_values[i]
        for i in range(len(projection_values))
    }

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


def get_features_loop(
    func,
    source_ids: List[int],
    features_catalog: str = "ZTF_source_features_DR16",
    verbose: bool = False,
    whole_field: bool = True,
    field: int = 291,
    ccd: int = 1,
    quad: int = 1,
    limit_per_query: int = 1000,
    max_sources: int = 100000,
    impute_missing_features: bool = False,
    self_impute: bool = True,
    restart: bool = True,
    write_csv: bool = False,
    projection: dict = {},
    suffix: str = None,
    save: bool = True,
):
    """
    Get the features of all sources in a field.

    Parameters
    ==========
    func: function
        function over which to loop (get_features)
    source_ids: list
        list of source ids for feature queries
    features_catalog: str
        Name of Kowalski collection to query for features
    verbose: bool
        verbose if set
    whole_field: bool
        If True, get features of all sources in the field, else get features of a particular quad
    field: int
        Field number.
    ccd: int
        CCD (between 1 and 16)
    quad: int
        Quadrand; (between 1 and 4)
    limit_per_query: int
        Number of sources to query at a time.
    max_sources: int
        Number of sources to save in single file.
    impute_missing_features: bool
        if True, impute missing features using strategies in config file.
    self_impute: bool
        if True, impute features using self-imputation; otherwise use training set in config file.
    restart: bool
        if True, restart the querying of features even if file exists.
    write_csv: bool
        if True, writes results as csv file in addition to parquet.
    projection: dict
        mongoDB projection of columns to return.
    suffix: str
        Suffix to add to saved feature file.
    save: bool
        if True, save results

    Returns
    =======
    df: pandas.DataFrame
        DataFrame containing features
    outfile: str
        filepath to saved features, if applicable

    Stores the features in a file at the following location:
        features/field_<field>/field_<field>.parquet
    or  features/field_<field>/field_<field>.csv
    """

    if not whole_field:
        outfile = (
            os.path.dirname(__file__)
            + "/../features/field_"
            + str(field)
            + "/ccd_"
            + str(ccd).zfill(2)
            + "_quad_"
            + str(quad)
        )

    else:
        outfile = (
            os.path.dirname(__file__)
            + "/../features/field_"
            + str(field)
            + "/"
            + "field_"
            + str(field)
        )

    if suffix is not None:
        outfile += f'_{suffix}'
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    DS = ds.dataset(os.path.dirname(outfile), format='parquet')
    indiv_files = DS.files
    files_exist = len(indiv_files) > 0
    existing_ids = []

    # Set source_ids
    if (not restart) & (files_exist):
        generator = DS.to_batches(columns=['_id'])
        for batch in generator:
            existing_ids += batch['_id'].to_pylist()
        # Remove existing source_ids from list
        todo_source_ids = list(set(source_ids) - set(existing_ids))
        if len(todo_source_ids) == 0:
            print('Dataset is already complete.')
            return

    n_sources = len(source_ids)
    if n_sources % max_sources != 0:
        n_iterations = n_sources // max_sources + 1
    else:
        n_iterations = n_sources // max_sources
    start_iteration = len(existing_ids) // max_sources

    for i in range(start_iteration, n_iterations):
        print(f"Iteration {i+1} of {n_iterations}...")
        select_source_ids = source_ids[
            i * max_sources : min(n_sources, (i + 1) * max_sources)
        ]

        df, _ = func(
            source_ids=select_source_ids,
            features_catalog=features_catalog,
            verbose=verbose,
            limit_per_query=limit_per_query,
            impute_missing_features=impute_missing_features,
            self_impute=self_impute,
            projection=projection,
        )

        if save:
            write_parquet(df, f'{outfile}_iter_{i}.parquet')
            files_exist = True
            if write_csv:
                df.to_csv(f'{outfile}_iter_{i}.csv', index=False)

    return df, outfile


def get_features(
    source_ids: List[int],
    features_catalog: str = "ZTF_source_features_DR16",
    verbose: bool = False,
    limit_per_query: int = 1000,
    impute_missing_features: bool = False,
    self_impute: bool = True,
    dtypes: dict = dtype_dict,
    projection: dict = projection_dict,
):
    '''
    Get features of all ids present in the field in one file.

    Parameters
    ==========
    source_ids: list
        list of source ids for feature queries
    features_catalog: str
        Name of Kowalski collection to query for features
    verbose: bool
        verbose if set
    limit_per_query: int
        Number of sources to query at a time.
    impute_missing_features: bool
        if True, impute missing features using strategies in config file.
    self_impute: bool
        if True, impute features using self-imputation; otherwise use training set in config file.
    dtypes: dict
        dictionary containing dtypes for each feature (see features: ontological: in config.yaml)
    projection: dict
        mongoDB projection of columns to return.

    Returns
    =======
    df: pandas.DataFrame
        DataFrame containing features
    dmdt: numpy.Array
        dmdt histograms (also included in df)
    '''

    id = 0
    df_collection = []
    dmdt_temp = []
    dmdt_collection = []

    while True:
        query = {
            "query_type": "find",
            "query": {
                "catalog": features_catalog,
                "filter": {
                    "_id": {
                        "$in": source_ids[
                            id * limit_per_query : (id + 1) * limit_per_query
                        ]
                    }
                },
                "projection": projection,
            },
        }
        responses = kowalski_instances.query(query=query)

        for name in responses.keys():
            if len(responses[name]) > 0:
                response = responses[name]
                if response.get("status", "error") == "success":
                    source_data = response.get("data")
                    if source_data is None:
                        print(response)
                        raise ValueError(f"No data found for source ids {source_ids}")

        df_temp = pd.DataFrame(source_data)

        if len(df_temp) > 0:
            if (projection == {}) | ("dmdt" in projection):
                df_temp = df_temp.astype(dtype=dtypes)

            df_collection += [df_temp]
            try:
                dmdt_temp = np.expand_dims(
                    np.array([d for d in df_temp['dmdt'].values]), axis=-1
                )
            except Exception as e:
                # Print dmdt error if using the default projection or user requests the feature
                if (projection == {}) | ("dmdt" in projection):
                    print("Error", e)
                    print(df_temp)
            dmdt_collection += [dmdt_temp]

        if ((id + 1) * limit_per_query) >= len(source_ids):
            print(f'{len(source_ids)} done')
            break
        id += 1
        if (id * limit_per_query) % limit_per_query == 0:
            print(id * limit_per_query, "done")

    df = pd.concat(df_collection, axis=0)
    df.reset_index(drop=True, inplace=True)
    dmdt = np.vstack(dmdt_collection)

    if impute_missing_features:
        df = impute_features(df, self_impute=self_impute)

    # Add metadata
    utcnow = datetime.utcnow()
    start_dt = utcnow.strftime("%Y-%m-%d %H:%M:%S")
    features_ztf_dr = features_catalog.split('_')[-1]
    df.attrs['features_download_dateTime_utc'] = start_dt
    df.attrs['features_ztf_dataRelease'] = features_ztf_dr
    df.attrs['features_imputed'] = impute_missing_features

    if verbose:
        print("Features dataframe: ", df)
        print("dmdt shape: ", dmdt.shape)

    return df, dmdt


def get_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--field",
        type=int,
        help="field number",
        default=DEFAULT_FIELD,
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
        "--limit-per-query",
        type=int,
        default=DEFAULT_LIMIT,
        help="number of rows to return (default 10000)",
    )
    parser.add_argument(
        "--max-sources",
        type=int,
        default=DEFAULT_SAVE_BATCHSIZE,
        help="Number of sources to save in single file",
    )
    parser.add_argument(
        "--features-catalog",
        type=str,
        help="features catalog (default: ZTF_source_features_DR16)",
        default=DEFAULT_CATALOG,
    )
    parser.add_argument(
        "--whole-field",
        action="store_true",
        help="if passed as argument, store all features in one file",
    )
    parser.add_argument(
        "--start",
        type=int,
        help="Start index of the sources to query (to be used with --whole-field)",
        default=None,
    )
    parser.add_argument(
        "--end",
        type=int,
        help="End index of the sources to query. (to be used with --whole-field)",
        default=None,
    )
    parser.add_argument(
        "--restart",
        action='store_true',
        help="if set, restart the querying of features even if file exists",
    )
    parser.add_argument(
        "--no-write-results",
        action='store_true',
        help="if set, do not write results or make necessary directories",
    )
    parser.add_argument(
        "--write-csv",
        action='store_true',
        help="if set, writes results as csv file in addition to parquet",
    )
    parser.add_argument(
        "--column-list",
        type=str,
        nargs='+',
        help="List of column names to return from Kowalski collection",
        default=None,
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Suffix to add to saved feature file",
    )
    parser.add_argument(
        "--impute-missing-features",
        action='store_true',
        help="if set, impute missing features using strategies in config file",
    )
    parser.add_argument(
        "--no-self-impute",
        action='store_true',
        help="if set, impute features using training set in config file, rather than self-imputation",
    )
    parser.add_argument(
        "--tm",
        action='store_true',
        help="if set, report timing of different parts of the code to run",
    )
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="verbose",
    )

    return parser


def main():

    parser = get_parser()
    args, _ = parser.parse_known_args()

    field = args.field
    ccd_range = args.ccd_range
    quad_range = args.quad_range
    limit_per_query = args.limit_per_query
    max_sources = args.max_sources
    features_catalog = args.features_catalog
    whole_field = args.whole_field
    start = args.start
    end = args.end
    restart = args.restart
    write_results = not args.no_write_results
    write_csv = args.write_csv
    column_list = args.column_list
    suffix = args.suffix
    impute_missing_features = args.impute_missing_features
    self_impute = not args.no_self_impute
    tm = args.tm
    verbose = args.verbose

    projection = {}
    if column_list is not None:
        keys = [name for name in column_list]
        projection = {k: 1 for k in keys}

    if (len(ccd_range) not in [1, 2]) | (len(quad_range) not in [1, 2]):
        raise ValueError(
            "Please specify 1 or 2 integers for --ccd-range and --quad-range"
        )

    if len(ccd_range) == 1:
        ccd_range = [ccd_range[0], ccd_range[0]]
    if len(quad_range) == 1:
        quad_range = [quad_range[0], quad_range[0]]

    iter_dct = {}

    if not whole_field:
        for ccd in range(np.min(ccd_range), np.max(ccd_range) + 1):
            for quad in range(np.min(quad_range), np.max(quad_range) + 1):
                default_file = (
                    "../ids/field_"
                    + str(field)
                    + "/data_ccd_"
                    + str(ccd).zfill(2)
                    + "_quad_"
                    + str(quad)
                    + ".h5"
                )
                iter_dct[(ccd, quad)] = default_file
    else:
        default_file = "../ids/field_" + str(field) + "/field_" + str(field) + ".h5"
        iter_dct[field] = default_file

    for k, v in iter_dct.items():
        if isinstance(k, tuple):
            ccd_quad = k
            print(f'Getting features for ccd {ccd_quad[0]} quad {ccd_quad[1]}...')
        else:
            ccd_quad = (0, 0)
            print(f'Getting features for field {field}...')
        source_ids_filename = v

        filename = os.path.join(BASE_DIR, source_ids_filename)

        ts = time.time()
        source_ids = np.array([])
        with h5py.File(filename, "r") as f:
            ids = np.array(f[list(f.keys())[0]])
            source_ids = list(map(int, np.concatenate((source_ids, ids), axis=0)))
        te = time.time()
        if tm:
            print(
                "read source_ids from .h5".ljust(JUST)
                + "\t --> \t"
                + str(round(te - ts, 4))
                + " s"
            )

        if verbose:
            print(f"{len(source_ids)} total source ids")

        if write_results:
            get_features_loop(
                get_features,
                source_ids=source_ids[start:end],
                features_catalog=features_catalog,
                verbose=verbose,
                whole_field=whole_field,
                field=field,
                ccd=ccd_quad[0],
                quad=ccd_quad[1],
                limit_per_query=limit_per_query,
                max_sources=max_sources,
                impute_missing_features=impute_missing_features,
                self_impute=self_impute,
                restart=restart,
                write_csv=write_csv,
                projection=projection,
                suffix=suffix,
                save=True,
            )

        else:
            # get raw features
            get_features(
                source_ids=source_ids[start:end],
                features_catalog=features_catalog,
                verbose=verbose,
                limit_per_query=limit_per_query,
                impute_missing_features=impute_missing_features,
                self_impute=self_impute,
                projection=projection,
            )
