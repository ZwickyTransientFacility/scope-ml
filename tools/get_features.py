#!/usr/bin/env python
import fire
import numpy as np
import pandas as pd
import pathlib
from penquins import Kowalski
from typing import List
import yaml
import os
import time
import h5py
from scope.utils import write_parquet
from datetime import datetime
import pyarrow.dataset as ds

BASE_DIR = os.path.dirname(__file__)
JUST = 50
TIMEOUT = 300


config_path = pathlib.Path(__file__).parent.parent.absolute() / "config.yaml"
with open(config_path) as config_yaml:
    config = yaml.load(config_yaml, Loader=yaml.FullLoader)
# Access datatypes in config file
feature_names = config['features']['ontological']
dtype_dict = {key: feature_names[key]['dtype'] for key in feature_names}

# Use new penquins KowalskiInstances class here once approved
kowalski = Kowalski(
    token=config["kowalski"]["token"],
    protocol=config["kowalski"]["protocol"],
    host=config["kowalski"]["host"],
    port=config["kowalski"]["port"],
    timeout=TIMEOUT,
)


def get_features_loop(
    func,
    source_ids: List[int],
    features_catalog: str = "ZTF_source_features_DR5",
    verbose: bool = False,
    whole_field: bool = True,
    field: int = 291,
    ccd: int = 1,
    quad: int = 1,
    limit_per_query: int = 1000,
    max_sources: int = 100000,
    restart: bool = True,
    write_csv: bool = False,
    projection: dict = {},
    suffix: str = None,
):
    '''
    Loop over get_features.py to save at specified checkpoints.
    '''

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
    n_iterations = n_sources // max_sources + 1
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
            projection=projection,
        )

        write_parquet(df, f'{outfile}_iter_{i}.parquet')
        files_exist = True
        if write_csv:
            df.to_csv(f'{outfile}_iter_{i}.csv', index=False)


def get_features(
    source_ids: List[int],
    features_catalog: str = "ZTF_source_features_DR5",
    verbose: bool = False,
    limit_per_query: int = 1000,
    projection: dict = {},
):
    '''
    Get features of all ids present in the field in one file.
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
        response = kowalski.query(query=query)
        source_data = response.get("data")

        if source_data is None:
            print(response)
            raise ValueError(f"No data found for source ids {source_ids}")

        df_temp = pd.DataFrame(source_data)
        if projection == {}:
            df_temp = df_temp.astype(dtype=dtype_dict)
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

    # Add metadata
    utcnow = datetime.utcnow()
    start_dt = utcnow.strftime("%Y-%m-%d %H:%M:%S")
    features_ztf_dr = features_catalog.split('_')[-1]
    df.attrs['features_download_dateTime_utc'] = start_dt
    df.attrs['features_ztf_dataRelease'] = features_ztf_dr

    if verbose:
        print("Features dataframe: ", df)
        print("dmdt shape: ", dmdt.shape)

    # if not write_results:
    return df, dmdt


def run(**kwargs):
    """
    Get the features of all sources in a field.

    Parameters
    ==========
    field: int
        Field number.
    ccd: int
        CCD number (required if whole_field is False).
    quad: int
        Quad number (required if whole_field is False).
    limit_per_query: int
        Number of sources to query at a time.
    max_sources: int
        Number of sources to save in single file.
    features_catalog: str
        Name of Kowalski collection to query for features
    whole_field: bool
        If True, get features of all sources in the field, else get features of a particular quad.
    start: int
        Start index of the sources to query. (to be used with whole_field)
    end: int
        End index of the sources to query. (to be used with whole_field)
    restart: bool
        if True, restart the querying of features even if file exists.
    write_results: bool
        if True, write results and make necessary directories.
    write_csv: bool
        if True, writes results as csv file in addition to parquet.
    column_list: list
        List of strings for each column to return from Kowalski colleciton.
    suffix: str
        Suffix to add to saved feature file.
    Returns
    =======
    Stores the features in a file at the following location:
        features/field_<field>/field_<field>.parquet
    or  features/field_<field>/field_<field>.csv
    """

    DEFAULT_FIELD = 291
    DEFAULT_CCD = 1
    DEFAULT_QUAD = 1
    DEFAULT_LIMIT = 1000
    DEFAULT_SAVE_BATCHSIZE = 100000
    DEFAULT_CATALOG = "ZTF_source_features_DR5"

    field = kwargs.get("field", DEFAULT_FIELD)
    ccd = kwargs.get("ccd", DEFAULT_CCD)
    quad = kwargs.get("quad", DEFAULT_QUAD)
    limit_per_query = kwargs.get("limit_per_query", DEFAULT_LIMIT)
    max_sources = kwargs.get("max_sources", DEFAULT_SAVE_BATCHSIZE)
    features_catalog = kwargs.get("features_catalog", DEFAULT_CATALOG)
    whole_field = kwargs.get("whole_field", False)
    start = kwargs.get("start", None)
    end = kwargs.get("end", None)
    restart = kwargs.get("restart", False)
    write_results = kwargs.get("write_results", True)
    write_csv = kwargs.get("write_csv", False)
    column_list = kwargs.get("column_list", [])
    suffix = kwargs.get("suffix", None)

    if column_list != []:
        keys = [name for name in column_list]
        projection = {k: 1 for k in keys}

    if not whole_field:
        default_file = (
            "../ids/field_"
            + str(field)
            + "/data_ccd_"
            + str(ccd).zfill(2)
            + "_quad_"
            + str(quad)
            + ".h5"
        )
    else:
        default_file = "../ids/field_" + str(field) + "/field_" + str(field) + ".h5"

    source_ids_filename = kwargs.get("source_ids_filename", default_file)

    tm = kwargs.get("time", False)
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

    verbose = kwargs.get("verbose", False)
    if verbose:
        print(f"{len(source_ids)} total source ids")

    if not write_results:
        # get raw features
        get_features(
            source_ids=source_ids[start:end],
            features_catalog=features_catalog,
            verbose=verbose,
            limit_per_query=limit_per_query,
            projection=projection,
        )

    else:
        get_features_loop(
            get_features,
            source_ids=source_ids[start:end],
            features_catalog=features_catalog,
            verbose=verbose,
            whole_field=whole_field,
            field=field,
            ccd=ccd,
            quad=quad,
            limit_per_query=limit_per_query,
            max_sources=max_sources,
            restart=restart,
            write_csv=write_csv,
            projection=projection,
            suffix=suffix,
        )


if __name__ == "__main__":
    fire.Fire(run)
