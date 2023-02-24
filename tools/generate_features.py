#!/usr/bin/env python
import scope
import argparse
import pathlib
import yaml
import os
from tools.get_quad_ids import get_ids_loop, get_field_ids
from scope.fritz import get_lightcurves_via_ids
from scope.utils import (
    TychoBVfromGaia,
    exclude_radius,
    removeHighCadence,
    write_parquet,
)
import numpy as np
from penquins import Kowalski
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from datetime import datetime
from tools.featureGeneration import alertstats

# import time
# from tools.featureGeneration import lcstats, periodsearch
# import periodfind
# from numba import jit
# from cesium.featurize import time_series, featurize_single_ts, featurize_time_series, featurize_ts_files
# import cesium.features as fts


BASE_DIR = pathlib.Path(__file__).parent.parent.absolute()

# setup connection to Kowalski instances
config_path = pathlib.Path(__file__).parent.parent.absolute() / "config.yaml"
with open(config_path) as config_yaml:
    config = yaml.load(config_yaml, Loader=yaml.FullLoader)

timeout = config['kowalski']['timeout']

gloria = Kowalski(**config['kowalski'], verbose=False)
melman = Kowalski(
    token=config['kowalski']['token'],
    protocol="https",
    host="melman.caltech.edu",
    port=443,
    timeout=timeout,
)
kowalski = Kowalski(
    token=config['kowalski']['alt_token'],
    protocol="https",
    host="kowalski.caltech.edu",
    port=443,
    timeout=timeout,
)

source_catalog = config['kowalski']['collections']['sources']
alerts_catalog = config['kowalski']['collections']['alerts']
gaia_catalog = config['kowalski']['collections']['gaia']

kowalski_instances = {'kowalski': kowalski, 'gloria': gloria, 'melman': melman}


def drop_close_bright_stars(
    id_dct: dict,
    kowalski_instance: Kowalski,
    catalog: str = gaia_catalog,
    query_radius_arcsec: float = 300.0,
    xmatch_radius_arcsec: float = 2.0,
    limit: int = 10000,
):
    """
    Use Gaia to identify and drop sources that are too close to bright stars

    :param id_dct: one quadrant's worth of id-coordinate pairs (dict)
    :param kowalski_instance: authenticated instance of a Kowalski database
    :param catalog: name of catalog to use [currently only supports Gaia catalogs] (str)
    :param query_radius_arcsec: size of cone search radius to search for bright stars.
        Default is 300 corresponding with approximate maximum from A. Drake's exclusion radius (float)
    :param xmatch_radius_arcsec: size of cone within which to match a queried source with an input source.
        If any sources from the query fall within this cone, the closest one will be matched to the input source and dropped from further bright-star considerations (float)
    :param limit: batch size of kowalski_instance queries (int)

    :return id_dct_keep: dictionary containing subset of input sources far enough away from bright stars
    """

    ids = [x for x in id_dct]
    id_dct_keep = id_dct.copy()

    n_sources = len(id_dct)
    n_iterations = n_sources // limit + 1
    gaia_results_dct = {}

    print(f'Querying {catalog} catalog in batches...')
    for i in range(0, n_iterations):
        print(f"Iteration {i+1} of {n_iterations}...")
        id_slice = [x for x in id_dct.keys()][
            i * limit : min(n_sources, (i + 1) * limit)
        ]

        radec_geojson = np.array(
            [id_dct[x]['radec_geojson']['coordinates'] for x in id_slice]
        ).transpose()

        # Need to add 180 -> no negative RAs
        radec_geojson[0, :] += 180.0

        # Get Gaia EDR3 ID, G mag, BP-RP, and coordinates
        query = {
            "query_type": "cone_search",
            "query": {
                "object_coordinates": {
                    "radec": dict(zip(id_slice, radec_geojson.transpose().tolist())),
                    "cone_search_radius": query_radius_arcsec,
                    "cone_search_unit": 'arcsec',
                },
                "catalogs": {
                    catalog: {
                        # Select sources brighter than G magnitude 13:
                        # -Conversion to Tycho mags only good for G < 13
                        # -Need for exclusion radius only for stars with B <~ 13
                        # -For most stars, if G > 13, B > 13
                        "filter": {"phot_g_mean_mag": {"$lt": 13.0}},
                        "projection": {
                            "phot_g_mean_mag": 1,
                            "bp_rp": 1,
                            "coordinates.radec_geojson.coordinates": 1,
                        },
                    }
                },
                "filter": {},
            },
        }

        q = kowalski_instance.query(query)
        gaia_results = q['data'][catalog]
        gaia_results_dct.update(gaia_results)

    print('Identifying sources too close to bright stars...')
    # Loop over each id to compare with query results
    count = 0
    for id in ids:
        count += 1
        if count % limit == 0:
            print(f"{count} done")
        if count == len(ids):
            print(f"{count} done")

        val = id_dct[id]

        ra_geojson, dec_geojson = val['radec_geojson']['coordinates']

        single_result = gaia_results_dct[str(id)]
        if len(single_result) > 0:
            coords = np.array(
                [
                    x['coordinates']['radec_geojson']['coordinates']
                    for x in single_result
                ]
            )
            coords[:, 0] += 180.0

            # SkyCoord object for query results
            Coords = SkyCoord(coords, unit=['deg', 'deg'])
            # SkyCoord object for input source
            coord = SkyCoord(ra_geojson + 180.0, dec_geojson, unit=['deg', 'deg'])

            all_separations = Coords.separation(coord)
            # Identify closest source to input
            drop_source = np.argmin(all_separations)

            # If closest source is within specified radius, treat it as the input source and drop it from further consideration
            xmatch_source = {}
            if all_separations[drop_source] < xmatch_radius_arcsec * u.arcsec:
                xmatch_source = single_result.pop(drop_source)
                Coords = np.delete(Coords, drop_source)

            # If possible, use all-Gaia coordinates for next step
            if len(xmatch_source) > 0:
                xmatch_ra, xmatch_dec = xmatch_source['coordinates']['radec_geojson'][
                    'coordinates'
                ]
                xmatch_ra += 180.0
                xmatch_coord = SkyCoord(xmatch_ra, xmatch_dec, unit=['deg', 'deg'])
            else:
                xmatch_coord = SkyCoord(
                    ra_geojson + 180.0, dec_geojson, unit=['deg', 'deg']
                )

            # Use mapping from Gaia -> Tycho to set exclusion radius for each source
            for idx, source in enumerate(single_result):
                try:
                    B, V = TychoBVfromGaia(source['phot_g_mean_mag'], source['bp_rp'])
                    excl_radius = exclude_radius(B, V)
                except KeyError:
                    # Not all Gaia sources have BP-RP
                    excl_radius = 0.0
                if excl_radius > 0.0:
                    sep = xmatch_coord.separation(Coords[idx])
                    if excl_radius * u.arcsec > sep.to(u.arcsec):
                        # If there is a bright star that's too close, drop from returned dict
                        id_dct_keep.pop(id)
                        break

    print(f"Dropped {len(id_dct) - len(id_dct_keep)} sources.")
    return id_dct_keep


# def generate_features(source_catalog=source_catalog, alerts_catalog=alerts_catalog, gaia_catalog=gaia_catalog, bright_star_query_radius_arcsec=300.,
# xmatch_radius_arcsec=2., kowalski_instances = kowalski_instances, limit=10000, doAllFields=False, field=296, doAllCCDs=False,
# ccd=1, doAllQuads=False, quad=1, min_n_lc_points=50, min_cadence_minutes=30., dirname='generated_features', filename='features', doNotSave=False):
def generate_features(
    source_catalog: str = source_catalog,
    alerts_catalog: str = alerts_catalog,
    gaia_catalog: str = gaia_catalog,
    bright_star_query_radius_arcsec: float = 300.0,
    xmatch_radius_arcsec: float = 2.0,
    kowalski_instances: dict = kowalski_instances,
    limit: int = 10000,
    field: int = 296,
    ccd: int = 1,
    quad: int = 1,
    min_n_lc_points: int = 50,
    min_cadence_minutes: float = 30.0,
    dirname: str = 'generated_features',
    filename: str = 'features',
    doNotSave: bool = False,
):

    # Get code version and current date/time for metadata
    code_version = scope.__version__
    utcnow = datetime.utcnow()
    start_dt = utcnow.strftime("%Y-%m-%d %H:%M:%S")

    print('Getting IDs...')
    _, lst = get_ids_loop(
        get_field_ids,
        catalog=source_catalog,
        kowalski_instance=kowalski_instances['melman'],
        limit=limit,
        field=field,
        ccd_range=ccd,
        quad_range=quad,
        minobs=0,
        save=False,
        get_coords=True,
    )

    # Each index of lst corresponds to a different ccd/quad combo
    feature_gen_source_list = drop_close_bright_stars(
        lst[0],
        kowalski_instance=kowalski_instances['gloria'],
        catalog=gaia_catalog,
        query_radius_arcsec=bright_star_query_radius_arcsec,
        xmatch_radius_arcsec=xmatch_radius_arcsec,
        limit=limit,
    )

    print('Getting lightcurves...')
    lcs = get_lightcurves_via_ids(
        kowalski_instance=kowalski_instances['melman'],
        ids=[x for x in feature_gen_source_list.keys()],
        catalog=source_catalog,
        limit_per_query=limit,
    )

    feature_dict = feature_gen_source_list.copy()
    print('Analyzing lightcuves and computing features...')
    # Start by dropping flagged points
    count = 0
    for idx, lc in enumerate(lcs):
        count += 1
        if (idx + 1) % limit == 0:
            print(f"{count} done")
        if count == len(lcs):
            print(f"{count} done")
        _id = lc['_id']
        lc_unflagged = [x for x in lc['data'] if x['catflags'] == 0]
        flt = lc['filter']

        tme = [[x['hjd'], x['mag'], x['magerr']] for x in lc_unflagged]
        try:
            tme_arr = np.array(tme)
            t, m, e = tme_arr.transpose()

            # Remove all but the first of each group of high-cadence points
            tt, mm, ee = removeHighCadence(t, m, e, cadence_minutes=min_cadence_minutes)

            # Discard sources lacking minimum number of points
            if len(tt) < min_n_lc_points:
                feature_dict.pop(_id)
            else:
                # Add basic info
                feature_dict[_id]['ra'] = (
                    feature_gen_source_list[_id]['radec_geojson']['coordinates'][0]
                    + 180.0
                )
                feature_dict[_id]['dec'] = feature_gen_source_list[_id][
                    'radec_geojson'
                ]['coordinates'][1]
                feature_dict[_id]['field'] = field
                feature_dict[_id]['ccd'] = ccd
                feature_dict[_id]['quad'] = quad
                feature_dict[_id]['filter'] = flt

                # Begin generating features - below code to be replaced by lcstats, periodfind, etc.
                mean_mag = np.mean(mm)
                median_mag = np.median(mm)
                mean_err = np.mean(ee)

                feature_dict[_id]['mean'] = mean_mag
                feature_dict[_id]['median'] = median_mag
                feature_dict[_id]['mean_err'] = mean_err
                #
        except ValueError:
            feature_dict.pop(_id)

    # Get ZTF alert stats
    alert_stats_dct = alertstats.get_ztf_alert_stats(
        feature_dict,
        kowalski_instance=kowalski_instances['kowalski'],
        radius_arcsec=xmatch_radius_arcsec,
        limit=limit,
    )
    for _id in feature_dict.keys():
        feature_dict[_id]['n_ztf_alerts'] = alert_stats_dct[_id]['n_ztf_alerts']
        feature_dict[_id]['mean_ztf_alert_braai'] = alert_stats_dct[_id][
            'mean_ztf_alert_braai'
        ]

    feature_df = pd.DataFrame.from_dict(feature_dict, orient='index')
    utcnow = datetime.utcnow()
    end_dt = utcnow.strftime("%Y-%m-%d %H:%M:%S")

    # Add metadata
    feature_df.attrs['scope_code_version'] = code_version
    feature_df.attrs['feature_generation_start_dateTime_utc'] = start_dt
    feature_df.attrs['feature_generation_end_dateTime_utc'] = end_dt
    feature_df.attrs['ZTF_source_catalog'] = source_catalog
    feature_df.attrs['ZTF_alerts_catalog'] = alerts_catalog
    feature_df.attrs['Gaia_catalog'] = gaia_catalog

    # Write results
    if not doNotSave:
        filename += f"_field_{field}_ccd_{ccd}_quad_{quad}"
        filename += '.parquet'
        dirpath = BASE_DIR / dirname / f"field_{field}"
        os.makedirs(dirpath, exist_ok=True)

        filepath = dirpath / filename
        write_parquet(feature_df, str(filepath))
        print(f"Wrote features for {len(feature_df)} sources.")
    else:
        print(f"Generated features for {len(feature_df)} sources.")

    return feature_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-source_catalog",
        default=source_catalog,
        help="name of source collection on Kowalski",
    )
    parser.add_argument(
        "-alerts_catalog",
        default=alerts_catalog,
        help="name of alerts collection on Kowalski",
    )
    parser.add_argument(
        "-gaia_catalog",
        default=gaia_catalog,
        help="name of Gaia collection on Kowalski",
    )
    parser.add_argument(
        "-bright_star_query_radius_arcsec",
        type=float,
        default=300.0,
        help="size of cone search radius to search for bright stars",
    )
    parser.add_argument(
        "-xmatch_radius_arcsec",
        type=float,
        default=2.0,
        help="cone radius for all crossmatches",
    )
    parser.add_argument(
        "-query_size_limit",
        type=int,
        default=10000,
        help="sources per query limit for large Kowalski queries",
    )

    # Loop over field/ccd/quad functionality coming soon
    # parser.add_argument("-doAllFields", action='store_true', default=False, help="if True, run on all fields")
    parser.add_argument(
        "-field", type=int, default=296, help="if not -doAllFields, ZTF field to run on"
    )
    # parser.add_argument("-doAllCCDs", action='store_true', default=False, help="if True, run on all ccds for given field")
    parser.add_argument(
        "-ccd", type=int, default=1, help="if not -doAllCCDs, ZTF ccd to run on"
    )
    # parser.add_argument("-doAllQuads", action='store_true', default=False, help="if True, run on all quads for specified field/ccds")
    parser.add_argument(
        "-quad", type=int, default=1, help="if not -doAllQuads, ZTF field to run on"
    )
    parser.add_argument(
        "-min_n_lc_points",
        type=int,
        default=50,
        help="minimum number of unflagged light curve points to run feature generation",
    )
    parser.add_argument(
        "-min_cadence_minutes",
        type=float,
        default=30.0,
        help="minimum cadence (in minutes) between light curve points. For groups of points closer together than this value, only the first will be kept.",
    )
    parser.add_argument(
        "-dirname",
        type=str,
        default='generated_features',
        help="if True, run on all quads for specified field/ccds",
    )
    parser.add_argument(
        "-filename",
        type=str,
        default='gen_features',
        help="if True, run on all quads for specified field/ccds",
    )
    parser.add_argument(
        "-doNotSave",
        action='store_true',
        default=False,
        help="if True, do not save features",
    )

    args = parser.parse_args()

    # call generate_features
    generate_features(
        source_catalog=args.source_catalog,
        alerts_catalog=args.alerts_catalog,
        gaia_catalog=args.gaia_catalog,
        bright_star_query_radius_arcsec=args.bright_star_query_radius_arcsec,
        xmatch_radius_arcsec=args.xmatch_radius_arcsec,
        limit=args.query_size_limit,
        # args.doAllFields,
        field=args.field,
        # args.doAllCCDs,
        ccd=args.ccd,
        # args.doAllQuads,
        quad=args.quad,
        min_n_lc_points=args.min_n_lc_points,
        min_cadence_minutes=args.min_cadence_minutes,
        dirname=args.dirname,
        filename=args.filename,
        doNotSave=args.doNotSave,
    )
