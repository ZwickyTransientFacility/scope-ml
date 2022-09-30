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

BASE_DIR = os.path.dirname(__file__)
JUST = 50
TIMEOUT = 300


config_path = pathlib.Path(__file__).parent.parent.absolute() / "config.yaml"
with open(config_path) as config_yaml:
    config = yaml.load(config_yaml, Loader=yaml.FullLoader)

kowalski = Kowalski(
    token=config["kowalski"]["token"],
    protocol=config["kowalski"]["protocol"],
    host=config["kowalski"]["host"],
    port=config["kowalski"]["port"],
    timeout=TIMEOUT,
)


def get_features(
    source_ids: List[int],
    features_catalog: str = "ZTF_source_features_DR5",
    verbose: bool = False,
    restart: bool = True,
    **kwargs,
):

    ccd = kwargs.get("ccd", 1)
    quad = kwargs.get("quad", 1)
    field = kwargs.get("field", 302)
    outfile = (
        os.path.dirname(__file__)
        + "/../features/field_"
        + str(field)
        + "/ccd_"
        + str(ccd).zfill(2)
        + "_quad_"
        + str(quad)
    )

    limit = kwargs.get("limit", 1000)

    if not hasattr(source_ids, "__iter__"):
        source_ids = (source_ids,)

    id = 0
    df_collection = []
    dmdt_collection = []
    while 1:
        skip = id * limit
        query = {
            "query_type": "find",
            "query": {
                "catalog": features_catalog,
                "filter": {"_id": {"$in": source_ids}},
            },
            "kwargs": {"limit": limit, "skip": skip},
        }
        response = kowalski.query(query=query)
        source_data = response.get("data")

        if source_data is None:
            print(response)
            raise ValueError(f"No data found for source ids {source_ids}")

        # , index=np.arange(id*query_length,min((id+1)*query_length, len(source_data))))
        df_temp = pd.DataFrame.from_records(source_data)
        df_collection += [df_temp]
        dmdt_temp = np.expand_dims(
            np.array([d for d in df_temp['dmdt'].values]), axis=-1
        )
        dmdt_collection += [dmdt_temp]

        if ((id + 1) * limit) > len(source_ids):
            break
        id += 1

    df = pd.concat(df_collection, axis=0)
    dmdt = np.vstack(dmdt_collection)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    if (
        restart is False
        and os.path.exists(outfile + ".pkl")
        and os.path.exists(outfile + ".csv")
    ):
        df1 = pd.read_pickle(outfile + ".pkl")
        df2 = pd.concat([df1, df], axis=0)
        df2.to_pickle(outfile + ".pkl")
        df1 = pd.read_csv(outfile + ".csv")
        df2 = pd.concat([df1, df], axis=0)
        df2.to_csv(outfile + ".csv", index=False)
    else:
        df.to_pickle(outfile + ".pkl")
        df.to_csv(outfile + ".csv", index=False)

    if verbose:
        print(df)
        print("dmdt shape", dmdt.shape)

    return df, dmdt


# get features of all ids in a field
def get_field_features(
    source_ids: List[int],
    features_catalog: str = "ZTF_source_features_DR5",
    verbose: bool = False,
    restart: bool = True,
    **kwargs,
):
    '''
    Get features of all ids present in the field in one file.
    '''
    field = kwargs.get("field", 302)
    outfile = (
        os.path.dirname(__file__)
        + "/../features/field_"
        + str(field)
        + "/"
        + "field_"
        + str(field)
    )
    limit = kwargs.get("limit", 1000)

    id = 0
    df_collection = []
    dmdt_collection = []
    while 1:
        query = {
            "query_type": "find",
            "query": {
                "catalog": features_catalog,
                "filter": {"_id": {"$in": source_ids[id * limit : (id + 1) * limit]}},
            },
        }
        response = kowalski.query(query=query)
        source_data = response.get("data")

        if source_data is None:
            print(response)
            raise ValueError(f"No data found for source ids {source_ids}")

        df_temp = pd.DataFrame.from_records(source_data)
        df_collection += [df_temp]
        try:
            dmdt_temp = np.expand_dims(
                np.array([d for d in df_temp['dmdt'].values]), axis=-1
            )
        except Exception as e:
            print("Error", e)
            print(df_temp)
        dmdt_collection += [dmdt_temp]

        if ((id + 1) * limit) >= len(source_ids):
            print(f'{len(source_ids)} done')
            break
        id += 1
        if (id * limit) % 5000 == 0:
            print(id * limit, "done")

    df = pd.concat(df_collection, axis=0)
    dmdt = np.vstack(dmdt_collection)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    if (
        restart is False
        and os.path.exists(outfile + ".pkl")
        and os.path.exists(outfile + ".csv")
    ):
        df1 = pd.read_pickle(outfile + ".pkl")
        df2 = pd.concat([df1, df], axis=0)
        df2.to_pickle(outfile + ".pkl")
        df1 = pd.read_csv(outfile + ".csv")
        df2 = pd.concat([df1, df], axis=0)
        df2.to_csv(outfile + ".csv", index=False)
    else:
        df.to_pickle(outfile + ".pkl")
        df.to_csv(outfile + ".csv", index=False)

    if verbose:
        print(df)
        print("dmdt shape", dmdt.shape)

    return df, dmdt


def run(**kwargs):
    """
    Get the features of all sources in a field.

    Parameters
    ==========
    field: int
        Field number.
    limit: int
        Number of sources to query at a time.
    whole_field: bool
        If True, get features of all sources in the field, else get features of a particular quad.
    ccd: int
        CCD number (required if whole_field is False).
    quad: int
        Quad number (required if whole_field is False).
    start: int
        Start index of the sources to query. (to be used with whole_field)
    end: int
        End index of the sources to query. (to be used with whole_field)
    restart: bool
        if True, restart the querying of features even if file exists.
    Returns
    =======
    Stores the features in a file at the following location:
        features/field_<field>/field_<field>.csv
    and features/field_<field>/field_<field>.pkl
    """

    DEFAULT_FIELD = 291
    DEFAULT_CCD = 1
    DEFAULT_QUAD = 1
    DEFAULT_LIMIT = 1000
    DEFAULT_CATALOG = "ZTF_source_features_DR5"

    field = kwargs.get("field", DEFAULT_FIELD)
    ccd = kwargs.get("ccd", DEFAULT_CCD)
    quad = kwargs.get("quad", DEFAULT_QUAD)
    limit = kwargs.get("limit", DEFAULT_LIMIT)
    whole_field = kwargs.get("whole_field", True)
    start = kwargs.get("start", None)
    end = kwargs.get("end", None)
    restart = kwargs.get("restart", True)

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
        # print("are equal?",source_ids == source_ids1)

    verbose = kwargs.get("verbose", False)
    if verbose:
        print(len(source_ids))

    # get raw features

    if not whole_field:
        get_features(
            source_ids=source_ids[start:end],
            features_catalog=DEFAULT_CATALOG,
            verbose=verbose,
            field=field,
            ccd=ccd,
            quad=quad,
            limit=limit,
            restart=restart,
        )
    else:
        get_field_features(
            source_ids=source_ids[start:end],
            features_catalog=DEFAULT_CATALOG,
            verbose=verbose,
            field=field,
            limit=limit,
            restart=restart,
        )


if __name__ == "__main__":
    fire.Fire(run)
