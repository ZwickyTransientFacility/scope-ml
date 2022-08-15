import urllib
import requests
import pathlib
import yaml
from typing import Optional, Mapping
import numpy as np
import pandas as pd


def get_highscoring_objects(
    G,
    otype='vnv',
    condition="$or",
    limit=0.9,
    limit_dnn=None,
    limit_xgb=None,
):

    if limit_dnn is None:
        limit_dnn = limit
    if limit_xgb is None:
        limit_xgb = limit

    # example
    q = {
        "query_type": "find",
        "query": {
            "catalog": "ZTF_source_classifications_DR5",
            "filter": {
                condition: [
                    {'%s_xgb' % otype: {'$gt': limit_xgb}},
                    {'%s_dnn' % otype: {'$gt': limit_dnn}},
                ],
            },
            "projection": {},
        },
        "kwargs": {},
    }

    r = G.query(q)

    return pd.DataFrame(r['data'])


def get_stats(G, ids):
    qs = [
        {
            "query_type": "find",
            "query": {
                "catalog": "ZTF_source_features_DR5",
                "filter": {'_id': i},
                "projection": {},
            },
            "kwargs": {},
        }
        for i in ids
    ]
    rs = G.batch_query(qs, n_treads=32)

    return pd.DataFrame([s['data'][0] for s in rs])


# define the baseurl and set the token to connect
config_path = pathlib.Path(__file__).parent.parent.absolute() / "fritz.yaml"
with open(config_path) as config_yaml:
    config = yaml.load(config_yaml, Loader=yaml.FullLoader)
BASE_URL = "https://fritz.science/"


def api(
    method: str,
    endpoint: str,
    token: str,
    data: Optional[Mapping] = None,
    base_url: str = BASE_URL,
):
    method = method.upper()
    headers = {"Authorization": f"token {token}"}
    kwargs = {
        "method": method,
        "url": urllib.parse.urljoin(base_url, endpoint),
        "headers": headers,
    }
    if method not in ("GET", "HEAD"):
        kwargs["json"] = data
    elif method == "GET":
        kwargs["params"] = data
    response = requests.request(**kwargs)

    return response


def radec_to_iau_name(ra: float, dec: float, prefix: str = "ZTFJ"):
    """Transform R.A./Decl. in degrees to IAU-style hexadecimal designations."""
    if not 0.0 <= ra < 360.0:
        raise ValueError("Bad RA value in degrees")
    if not -90.0 <= dec <= 90.0:
        raise ValueError("Bad Dec value in degrees")

    ra_h = np.floor(ra * 12.0 / 180.0)
    ra_m = np.floor((ra * 12.0 / 180.0 - ra_h) * 60.0)
    ra_s = ((ra * 12.0 / 180.0 - ra_h) * 60.0 - ra_m) * 60.0

    dec_d = np.floor(abs(dec)) * np.sign(dec)
    dec_m = np.floor(np.abs(dec - dec_d) * 60.0)
    dec_s = np.abs(np.abs(dec - dec_d) * 60.0 - dec_m) * 60.0

    hms = f"{ra_h:02.0f}{ra_m:02.0f}{ra_s:05.2f}"
    dms = f"{dec_d:+03.0f}{dec_m:02.0f}{dec_s:04.1f}"

    return prefix + hms + dms


def get_lightcurves(gloria, ra, dec, radius=2.0):

    query = {"query_type": "info", "query": {"command": "catalog_names"}}
    available_catalog_names = gloria.query(query=query).get("data")
    # expose only the ZTF light curves for now
    available_catalogs = [
        catalog for catalog in available_catalog_names if "ZTF_sources" in catalog
    ]

    # simply select last catalog
    catalog = available_catalogs[-1]

    # allow access to all data by default
    program_id_selector = list([1, 2, 3])

    # executing a cone search
    # grab id's first
    query = {
        "query_type": "near",
        "query": {
            "max_distance": radius,
            "distance_units": "arcsec",
            "radec": {"query_coords": [ra, dec]},
            "catalogs": {
                catalog: {
                    "filter": {},
                    "projection": {"_id": 1},
                }
            },
        },
        "kwargs": {
            "max_time_ms": 10000,
            "limit": 1000,
        },
    }

    response = gloria.query(query=query)
    if response.get("status", "error") == "success":
        light_curve_ids = [
            item["_id"] for item in response.get("data")[catalog]["query_coords"]
        ]
        if len(light_curve_ids) == 0:
            print("No lightcurves found")
            return None

        query = {
            "query_type": "aggregate",
            "query": {
                "catalog": catalog,
                "pipeline": [
                    {"$match": {"_id": {"$in": light_curve_ids}}},
                    {
                        "$project": {
                            "_id": 1,
                            "ra": 1,
                            "dec": 1,
                            "filter": 1,
                            "meanmag": 1,
                            "vonneumannratio": 1,
                            "refchi": 1,
                            "refmag": 1,
                            "refmagerr": 1,
                            "iqr": 1,
                            "data": {
                                "$filter": {
                                    "input": "$data",
                                    "as": "item",
                                    "cond": {
                                        "$in": [
                                            "$$item.programid",
                                            program_id_selector,
                                        ]
                                    },
                                }
                            },
                        }
                    },
                ],
            },
        }
        response = gloria.query(query=query)
        if response.get("status", "error") == "success":
            light_curves = response.get("data")

    return light_curves


def make_photometry(light_curves: list, drop_flagged: bool = False):
    """
    Make a pandas.DataFrame with photometry
    :param light_curves: list of photometric time series
    :param drop_flagged: drop data points with catflags!=0
    :return:
    """
    dfs = []
    for light_curve in light_curves:
        if len(light_curve["data"]):
            df = pd.DataFrame.from_records(light_curve["data"])
            df["fid"] = light_curve["filter"]
            dfs.append(df)

    df_light_curve = pd.concat(dfs, ignore_index=True, sort=False)

    ztf_filters = {1: "ztfg", 2: "ztfr", 3: "ztfi"}
    df_light_curve["ztf_filter"] = df_light_curve["fid"].apply(lambda x: ztf_filters[x])
    df_light_curve["magsys"] = "ab"
    df_light_curve["zp"] = 23.9
    df_light_curve["mjd"] = df_light_curve["hjd"] - 2400000.5

    df_light_curve["mjd"] = df_light_curve["mjd"].apply(lambda x: np.float64(x))
    df_light_curve["mag"] = df_light_curve["mag"].apply(lambda x: np.float32(x))
    df_light_curve["magerr"] = df_light_curve["magerr"].apply(lambda x: np.float32(x))

    # filter out flagged data:
    if drop_flagged:
        mask_not_flagged = df_light_curve["catflags"] == 0
        df_light_curve = df_light_curve.loc[mask_not_flagged]

    return df_light_curve


def save_newsource(
    gloria,
    group_ids,
    ra,
    dec,
    token,
    radius=2.0,
    obj_id=None,
    dryrun=False,
    period=None,
    return_id=False,
):

    # get the lightcurves
    light_curves = get_lightcurves(gloria, ra, dec, 2.0)
    if len(light_curves) < 1:
        print('No lightcurves found for this objects!')
        return None

    # generate position-based name if obj_id not set
    if obj_id is None:
        ra_mean = float(
            np.mean(
                [
                    light_curve["ra"]
                    for light_curve in light_curves
                    if light_curve.get("ra") is not None
                ]
            )
        )
        dec_mean = float(
            np.mean(
                [
                    light_curve["dec"]
                    for light_curve in light_curves
                    if light_curve.get("dec") is not None
                ]
            )
        )
        obj_id = radec_to_iau_name(ra_mean, dec_mean, prefix="ZTFJ")

        # a source exists on F already?
        response = api(
            "GET", f"/api/sources?&ra={ra}&dec={dec}&radius={radius/3600}", token
        )
        data = response.json().get("data")
        if data["totalMatches"] > 0:
            # print(data)
            # save source to the groupids if it is already on Fritz
            print("%s already exists on Fritz!" % data["sources"][0]['id'])
            return None

        # post new source to Fritz
        if not dryrun:
            post_source_data = {
                "id": obj_id,
                "ra": ra_mean,
                "dec": dec_mean,
                "group_ids": group_ids,
                "origin": "Fritz",
            }
            response = api("POST", "/api/sources", token, post_source_data)
            if response.json()["status"] == "error":
                print(f"Failed to save {obj_id} as a Source")
                return None

        print("Found %d lightcurves" % len(light_curves))

        # post photometry to obj_id; drop flagged data
        df_photometry = make_photometry(light_curves, drop_flagged=True)

        # hardcoded this because it is easier, but if Fritz ever changes
        # this number will change
        instrument_id = 1

        photometry = {
            "obj_id": obj_id,
            "instrument_id": instrument_id,
            "mjd": df_photometry["mjd"].tolist(),
            "mag": df_photometry["mag"].tolist(),
            "magerr": df_photometry["magerr"].tolist(),
            "limiting_mag": df_photometry["zp"].tolist(),
            "magsys": df_photometry["magsys"].tolist(),
            "filter": df_photometry["ztf_filter"].tolist(),
            "ra": df_photometry["ra"].tolist(),
            "dec": df_photometry["dec"].tolist(),
        }

        if (len(photometry.get("mag", ())) > 0) & (not dryrun):
            print("Attempting to upload as %s" % obj_id)
            response = api("PUT", "/api/photometry", token, photometry)
            if response.json()["status"] == "error":
                print('Failed to post to Fritz')
                return None

    if period is not None:
        # upload the period if it is provided
        data = {
            "origin": "uploadedperiod",
            "group_ids": group_ids,
            "data": {'period': period},
        }
        response = api("POST", "api/sources/%s/annotations" % obj_id, token, data=data)

    if return_id is True:
        return obj_id
    else:
        return None
