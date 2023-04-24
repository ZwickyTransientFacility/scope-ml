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
    sort_lightcurve,
    read_parquet,
    read_hdf,
    split_dict,
)
import numpy as np
from penquins import Kowalski
import pandas as pd
from astropy.coordinates import SkyCoord, angular_separation
import astropy.units as u
from datetime import datetime
from tools.featureGeneration import lcstats, periodsearch, alertstats, external_xmatch
import warnings
from cesium.featurize import time_series, featurize_single_ts
import json
from joblib import Parallel, delayed

BASE_DIR = pathlib.Path(__file__).parent.parent.absolute()

# setup connection to Kowalski instances
config_path = pathlib.Path(__file__).parent.parent.absolute() / "config.yaml"
with open(config_path) as config_yaml:
    config = yaml.load(config_yaml, Loader=yaml.FullLoader)

# use token specified as env var (if exists)
kowalski_token_env = os.environ.get("KOWALSKI_TOKEN")
kowalski_alt_token_env = os.environ.get("KOWALSKI_ALT_TOKEN")
if (kowalski_token_env is not None) & (kowalski_alt_token_env is not None):
    config["kowalski"]["token"] = kowalski_token_env
    config["kowalski"]["alt_token"] = kowalski_alt_token_env

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
dmdt_ints = config['feature_generation']['dmdt_ints']
ext_catalog_info = config['feature_generation']['external_catalog_features']
cesium_feature_list = config['feature_generation']['cesium_features']
period_algorithms = config['feature_generation']['period_algorithms']

kowalski_instances = {'kowalski': kowalski, 'gloria': gloria, 'melman': melman}


def drop_close_bright_stars(
    id_dct: dict,
    kowalski_instance: Kowalski,
    catalog: str = gaia_catalog,
    query_radius_arcsec: float = 300.0,
    xmatch_radius_arcsec: float = 2.0,
    doSpecificIDs: bool = False,
    limit: int = 10000,
    Ncore: int = 8,
    save: bool = False,
    save_filename: str = 'tools/fritzDownload/merged_classifications_features_dropCloseSources.parquet',
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
    :param doSpecificIDs: if set, query for specific ZTF IDs instead of a single quadrant (bool)
    :param limit: if doSpecificIDs is set, max number of sources to be queries in one batch (int)
    :param Ncore: if doSpecificIDs is set, number of cores over which to parallelize queries (int)
    :param save: if set, save sources passing the close source analysis (bool)
    :param save_filename: path/name from BASE_DIR to save sources (str)

    :return id_dct_keep: dictionary containing subset of input sources far enough away from bright stars
    """

    ids = [x for x in id_dct]
    coords = np.array(
        [x['radec_geojson']['coordinates'] for x in id_dct.values()]
    ).transpose()
    sources_ra = coords[0] + 180.0
    sources_dec = coords[1]

    min_ra = np.min(sources_ra) - query_radius_arcsec / 3600.0
    max_ra = np.max(sources_ra) + query_radius_arcsec / 3600.0
    min_dec = np.min(sources_dec) - query_radius_arcsec / 3600.0
    max_dec = np.max(sources_dec) + query_radius_arcsec / 3600.0

    ctr_ra = np.mean([min_ra, max_ra])
    ctr_dec = np.mean([min_dec, max_dec])

    max_cone_radius = np.max([max_dec - ctr_dec, max_ra - ctr_ra]) * 3600 * np.sqrt(2)

    id_dct_keep = id_dct.copy()

    gaia_results_dct = {}

    if not doSpecificIDs:
        query = {
            "query_type": "cone_search",
            "query": {
                "object_coordinates": {
                    "radec": {'quad': [ctr_ra, ctr_dec]},
                    "cone_search_radius": max_cone_radius,
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

        gaia_results = q.get('data')
        gaia_results_dct.update(gaia_results[catalog])

        print('Identifying sources too close to bright stars...')

        # Loop over each id to compare with query results
        query_result = np.array(gaia_results_dct['quad'])

    else:
        n_sources = len(id_dct)
        if n_sources % limit != 0:
            n_iterations = n_sources // limit + 1
        else:
            n_iterations = n_sources // limit

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
            radec_dict = dict(zip(id_slice, radec_geojson.transpose().tolist()))

            if Ncore > 1:
                # Split dictionary for parallel querying
                radec_split_list = [lst for lst in split_dict(radec_dict, Ncore)]
                queries = [
                    {
                        "query_type": "cone_search",
                        "query": {
                            "object_coordinates": {
                                "radec": dct,
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
                    for dct in radec_split_list
                ]

                q = kowalski_instance.batch_query(queries, n_treads=8)
                for batch_result in q:
                    gaia_results = batch_result['data'][catalog]
                    gaia_results_dct.update(gaia_results)
            else:
                # Get Gaia EDR3 ID, G mag, BP-RP, and coordinates
                query = {
                    "query_type": "cone_search",
                    "query": {
                        "object_coordinates": {
                            # "radec": dict(zip(id_slice, radec_geojson.transpose().tolist())),
                            "radec": radec_dict,
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
        query_result = gaia_results_dct

    if len(query_result) > 0:
        if not doSpecificIDs:
            source_coords = np.array(
                [x['radec_geojson']['coordinates'] for x in id_dct.values()]
            )
            source_coords[:, 0] += 180.0
            Source_Coords = SkyCoord(source_coords, unit=[u.deg, u.deg])
            lon1 = Source_Coords.spherical.lon.to(u.rad).value
            lat1 = Source_Coords.spherical.lat.to(u.rad).value

            query_coords = np.array(
                [x['coordinates']['radec_geojson']['coordinates'] for x in query_result]
            )
            query_coords[:, 0] += 180.0
            Query_Coords = SkyCoord(query_coords, unit=[u.deg, u.deg])
            lon2 = Query_Coords.spherical.lon.to(u.rad).value
            lat2 = Query_Coords.spherical.lat.to(u.rad).value

            for i, id in enumerate(ids):
                # Reset bright source dictionary for each iteration
                single_result = query_result.copy()

                Coords = np.copy(Query_Coords)

                val = id_dct[id]
                ra_geojson, dec_geojson = val['radec_geojson']['coordinates']

                # Compute separations in radians and convert, using 1 rad = 206265 arcsec
                # ~10x faster than SkyCoord.separation() if lon/lat is calculated out of loop
                all_separations = (
                    angular_separation(lon1[i], lat1[i], lon2, lat2) * 206265
                )
                within_range = all_separations < query_radius_arcsec

                if np.sum(within_range) > 0:
                    all_separations = all_separations[within_range]
                    Coords = Coords[within_range]
                    single_result = query_result[within_range].tolist()

                    # Identify closest source to input
                    drop_source = np.argmin(all_separations)

                    # If closest source is within specified radius, treat it as the input source and drop it from further consideration
                    xmatch_source = {}
                    if all_separations[drop_source] < xmatch_radius_arcsec:
                        xmatch_source = single_result.pop(drop_source)
                        Coords = np.delete(Coords, drop_source)

                    update_id_dict(
                        id_dct_keep,
                        id,
                        xmatch_source,
                        ra_geojson,
                        dec_geojson,
                        single_result,
                        Coords,
                    )
        else:
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
                    coord = SkyCoord(
                        ra_geojson + 180.0, dec_geojson, unit=['deg', 'deg']
                    )

                    all_separations = Coords.separation(coord)
                    # Identify closest source to input
                    drop_source = np.argmin(all_separations)

                    # If closest source is within specified radius, treat it as the input source and drop it from further consideration
                    xmatch_source = {}
                    if all_separations[drop_source] < xmatch_radius_arcsec * u.arcsec:
                        xmatch_source = single_result.pop(drop_source)
                        Coords = np.delete(Coords, drop_source)

                    update_id_dict(
                        id_dct_keep,
                        id,
                        xmatch_source,
                        ra_geojson,
                        dec_geojson,
                        single_result,
                        Coords,
                    )
    else:
        id_dct_keep = id_dct

    if save:
        save_df = pd.DataFrame.from_records(id_dct_keep)
        write_parquet(save_df, str(BASE_DIR / save_filename))

    print(f"Dropped {len(id_dct) - len(id_dct_keep)} sources.")
    return id_dct_keep


def update_id_dict(
    id_dct_keep, id, xmatch_source, ra_geojson, dec_geojson, single_result, Coords
):
    """
    Helper function called by drop_close_bright_stars
    """
    # If possible, use all-Gaia coordinates for next step
    if len(xmatch_source) > 0:
        xmatch_ra, xmatch_dec = xmatch_source['coordinates']['radec_geojson'][
            'coordinates'
        ]
        xmatch_ra += 180.0
        xmatch_coord = SkyCoord(xmatch_ra, xmatch_dec, unit=['deg', 'deg'])
    else:
        xmatch_coord = SkyCoord(ra_geojson + 180.0, dec_geojson, unit=['deg', 'deg'])

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


def generate_features(
    source_catalog: str = source_catalog,
    alerts_catalog: str = alerts_catalog,
    gaia_catalog: str = gaia_catalog,
    bright_star_query_radius_arcsec: float = 300.0,
    xmatch_radius_arcsec: float = 2.0,
    kowalski_instances: dict = kowalski_instances,
    limit: int = 10000,
    period_algorithms: dict = period_algorithms,
    period_batch_size: int = 1000,
    doCPU: bool = False,
    doGPU: bool = False,
    samples_per_peak: int = 10,
    doScaleMinPeriod: bool = False,
    doRemoveTerrestrial: bool = False,
    Ncore: int = 8,
    field: int = 296,
    ccd: int = 1,
    quad: int = 1,
    min_n_lc_points: int = 50,
    min_cadence_minutes: float = 30.0,
    dirname: str = 'generated_features',
    filename: str = 'gen_features',
    doCesium: bool = False,
    doNotSave: bool = False,
    stop_early: bool = False,
    doQuadrantFile: bool = False,
    quadrant_file: str = 'slurm.dat',
    quadrant_index: int = 0,
    doSpecificIDs: bool = False,
    doNotRemoveCloseSources: bool = False,
):
    """
    Generate features for ZTF light curves

    :param source_catalog*: name of Kowalski catalog containing ZTF sources (str)
    :param alerts_catalog*: name of Kowalski catalog containing ZTF alerts (str)
    :param gaia_catalog*: name of Kowalski catalog containing Gaia data (str)
    :param bright_star_query_radius_arcsec: maximum angular distance from ZTF sources to query nearby bright stars in Gaia (float)
    :param xmatch_radius_arcsec: maximum angular distance from ZTF sources to match external catalog sources (float)
    :param kowalski_instances*: dictionary containing {names of Kowalski instances : authenticated penquins.Kowalski objects} (dict)
    :param limit: maximum number of sources to process in batch queries / statistics calculations (int)
    :param period_algorithms*: dictionary containing names of period algorithms to run. Normally specified in config - if specified here, should be a (list)
    :param period_batch_size: maximum number of sources to simultaneously perform period finding (int)
    :param doCPU: flag to run config-specified CPU period algorithms (bool)
    :param doGPU: flag to run config-specified GPU period algorithms (bool)
    :param samples_per_peak: number of samples per periodogram peak (int)
    :param doScaleMinPeriod: for period finding, scale min period based on min_cadence_minutes [otherwise, min P = 3 min] (bool)
    :param doRemoveTerrestrial: remove terrestrial frequencies from period-finding analysis (bool)
    :param Ncore: number of CPU cores to parallelize queries (int)
    :param field: ZTF field to run (int)
    :param ccd: ZTF ccd to run (int)
    :param quad: ZTF quadrant to run (int)
    :param min_n_lc_points: minimum number of points required to generate features for a light curve (int)
    :param min_cadence_minutes: minimum cadence between light curve points. Higher-cadence data are dropped except for the first point in the sequence (float)
    :param dirname: name of generated feature directory (str)
    :param filename: prefix of each feature filename (str)
    :param doCesium: flag to compute config-specified cesium features in addition to default list (bool)
    :param doNotSave: flag to avoid saving generated features (bool)
    :param stop_early: flag to stop feature generation before entire quadrant is run. Pair with --limit to run small-scale tests (bool)
    :param doQuadrantFile: flag to use a generated file containing [jobID, field, ccd, quad] columns instead of specifying --field, --ccd and --quad (bool)
    :param quadrant_file: name of quadrant file in the generated_features/slurm directory or equivalent (str)
    :param quadrant_index: number of job in quadrant file to run (int)
    :param doSpecificIDs: flag to perform feature generation for ztf_id column in config-specified file (bool)
    :param doNotRemoveCloseSources: flag to skip removal of sources too close to bright stars via Gaia (bool)

    :return feature_df: dataframe containing generated features

    * - specified in config.yaml
    """

    # Get code version and current date/time for metadata
    code_version = scope.__version__
    utcnow = datetime.utcnow()
    start_dt = utcnow.strftime("%Y-%m-%d %H:%M:%S")

    # Select period algorithms from config based on CPU or GPU specification
    if type(period_algorithms) == dict:
        if doCPU:
            period_algorithms = period_algorithms['CPU']
        elif doGPU:
            period_algorithms = period_algorithms['GPU']

    if not doSpecificIDs:
        # Code supporting parallelization across fields/ccds/quads
        slurmDir = os.path.join(str(BASE_DIR / dirname), 'slurm')
        if doQuadrantFile:
            names = ["job_number", "field", "ccd", "quadrant"]
            df_original = pd.read_csv(
                os.path.join(slurmDir, quadrant_file),
                header=None,
                delimiter=' ',
                names=names,
            )
            row = df_original.iloc[quadrant_index, :]
            field, ccd, quad = int(row["field"]), int(row["ccd"]), int(row["quadrant"])

        print(f'Running field {field}, CCD {ccd}, Quadrant {quad}...')
        print('Getting IDs...')
        _, lst = get_ids_loop(
            get_field_ids,
            catalog=source_catalog,
            kowalski_instance=kowalski_instances['melman'],
            limit=limit,
            field=field,
            ccd_range=ccd,
            quad_range=quad,
            minobs=min_n_lc_points,
            save=False,
            get_coords=True,
            stop_early=stop_early,
        )

        if not doNotRemoveCloseSources:
            # Each index of lst corresponds to a different ccd/quad combo
            feature_gen_source_list = drop_close_bright_stars(
                lst[0],
                kowalski_instance=kowalski_instances['gloria'],
                catalog=gaia_catalog,
                query_radius_arcsec=bright_star_query_radius_arcsec,
                xmatch_radius_arcsec=xmatch_radius_arcsec,
            )
    else:
        if not doNotRemoveCloseSources:
            # Read ztf_id column from csv, hdf5 or parquet file specified in config entry
            fg_sources_config = config['feature_generation']['dataset']
        else:
            # Load pre-saved dataset if Gaia analysis already complete
            fg_sources_config = config['feature_generation']['dataset_skipGaia']

        fg_sources_path = str(BASE_DIR / fg_sources_config)

        if fg_sources_path.endswith('.parquet'):
            fg_sources = read_parquet(fg_sources_path)
        elif fg_sources_path.endswith('.h5'):
            fg_sources = read_hdf(fg_sources_path)
        elif fg_sources_path.endswith('.csv'):
            fg_sources = pd.read_csv(fg_sources_path)
        else:
            raise ValueError("Sources must be stored in .parquet, .h5 or .csv format.")

        try:
            ztf_ids = fg_sources['ztf_id'].values.tolist()
            coordinates = fg_sources['coordinates'].values.tolist()
        except KeyError:
            raise KeyError(
                'Columns "ztf_id" and "coordinates" must be included in source dataset.'
            )

        n_fg_sources = len(fg_sources)
        if stop_early:
            n_fg_sources = limit

        # Create list with same structure as query results
        lst = [
            {
                ztf_ids[i]: {
                    'radec_geojson': {
                        'coordinates': coordinates[i]['radec_geojson']['coordinates']
                    }
                }
                for i in range(n_fg_sources)
            }
        ]
        print(f'Loaded ZTF IDs for {len(lst[0])} sources.')

        if not doNotRemoveCloseSources:
            # Each index of lst corresponds to a different ccd/quad combo
            feature_gen_source_list = drop_close_bright_stars(
                lst[0],
                kowalski_instance=kowalski_instances['gloria'],
                catalog=gaia_catalog,
                query_radius_arcsec=bright_star_query_radius_arcsec,
                xmatch_radius_arcsec=xmatch_radius_arcsec,
                doSpecificIDs=True,
                limit=limit,
                Ncore=Ncore,
                save=not doNotSave,
            )

    print('Getting lightcurves...')
    # For small source lists, shrink LC query limit until batching occurs
    lc_limit = limit
    if Ncore > 1:
        lc_limit = int(np.ceil(len(feature_gen_source_list) / Ncore))

    lcs = get_lightcurves_via_ids(
        kowalski_instance=kowalski_instances['melman'],
        ids=[x for x in feature_gen_source_list.keys()],
        catalog=source_catalog,
        limit_per_query=lc_limit,
        Ncore=Ncore,
    )

    feature_dict = feature_gen_source_list.copy()
    print('Analyzing lightcuves and computing basic features...')
    # Start by dropping flagged points
    count = 0
    baseline = 0
    keep_id_list = []
    tme_collection = []
    tme_dict = {}
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

            # Ensure light curves are monotonically increasing in time
            t, m, e = sort_lightcurve(t, m, e)

            # Remove all but the first of each group of high-cadence points
            tt, mm, ee = removeHighCadence(t, m, e, cadence_minutes=min_cadence_minutes)

            # Discard sources lacking minimum number of points
            if len(tt) < min_n_lc_points:
                feature_dict.pop(_id)
            else:
                keep_id_list += [_id]
                if doCesium:
                    cesium_TS = time_series.TimeSeries(tt, mm, ee)

                # Determine largest time baseline over loop
                new_baseline = max(tt) - min(tt)
                if new_baseline > baseline:
                    baseline = new_baseline

                new_tme_arr = np.array([tt, mm, ee])
                tme_collection += [new_tme_arr]
                tme_dict[_id] = {}
                tme_dict[_id]['tme'] = new_tme_arr

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

                if doCesium:
                    cesium_features = featurize_single_ts(
                        cesium_TS, cesium_feature_list
                    )
                    cesium_features_dict = (
                        cesium_features.reset_index()
                        .drop('channel', axis=1)
                        .set_index('feature')
                        .to_dict()[0]
                    )
                    feature_dict[_id].update(cesium_features_dict)

        except ValueError:
            feature_dict.pop(_id)

    basicStats = Parallel(n_jobs=Ncore)(
        delayed(lcstats.calc_basic_stats)(id, vals['tme'])
        for id, vals in tme_dict.items()
    )

    for statline in basicStats:
        _id = [x for x in statline.keys()][0]
        statvals = [x for x in statline.values()][0]

        feature_dict[_id]['n'] = statvals[0]
        feature_dict[_id]['median'] = statvals[1]
        feature_dict[_id]['wmean'] = statvals[2]
        feature_dict[_id]['chi2red'] = statvals[3]
        feature_dict[_id]['roms'] = statvals[4]
        feature_dict[_id]['wstd'] = statvals[5]
        feature_dict[_id]['norm_peak_to_peak_amp'] = statvals[6]
        feature_dict[_id]['norm_excess_var'] = statvals[7]
        feature_dict[_id]['median_abs_dev'] = statvals[8]
        feature_dict[_id]['iqr'] = statvals[9]
        feature_dict[_id]['i60r'] = statvals[10]
        feature_dict[_id]['i70r'] = statvals[11]
        feature_dict[_id]['i80r'] = statvals[12]
        feature_dict[_id]['i90r'] = statvals[13]
        feature_dict[_id]['skew'] = statvals[14]
        feature_dict[_id]['smallkurt'] = statvals[15]
        feature_dict[_id]['inv_vonneumannratio'] = statvals[16]
        feature_dict[_id]['welch_i'] = statvals[17]
        feature_dict[_id]['stetson_j'] = statvals[18]
        feature_dict[_id]['stetson_k'] = statvals[19]
        feature_dict[_id]['ad'] = statvals[20]
        feature_dict[_id]['sw'] = statvals[21]

    if baseline > 0:
        # Define frequency grid using largest LC time baseline
        if doScaleMinPeriod:
            fmin, fmax = 2 / baseline, 1 / (
                2 * min_cadence_minutes / 1440
            )  # Nyquist frequency given minimum cadence converted to days
        else:
            fmin, fmax = 2 / baseline, 480

        df = 1.0 / (samples_per_peak * baseline)
        nf = int(np.ceil((fmax - fmin) / df))
        freqs = fmin + df * np.arange(nf)

        # Define terrestrial frequencies to remove
        if doRemoveTerrestrial:
            freqs_to_remove = [
                [3e-2, 4e-2],
                [3.95, 4.05],
                [2.95, 3.05],
                [1.95, 2.05],
                [0.95, 1.05],
                [0.48, 0.52],
                [0.32, 0.34],
            ]
        else:
            freqs_to_remove = None

        # Continue with periodsearch/periodfind
        period_dict = {}
        significance_dict = {}
        pdot_dict = {}
        if doCPU or doGPU:
            if doCPU and doGPU:
                raise KeyError('Please set only one of -doCPU or -doGPU.')

            n_sources = len(feature_dict)
            if n_sources % period_batch_size != 0:
                n_iterations = n_sources // period_batch_size + 1
            else:
                n_iterations = n_sources // period_batch_size

            for algorithm in period_algorithms:
                # Iterate over period batches and algorithms
                all_periods = np.array([])
                all_significances = np.array([])
                all_pdots = np.array([])

                print(
                    f'Running period algorithms for {len(feature_dict)} sources in batches of {period_batch_size}...'
                )
                print(f'Running {algorithm} algorithm:')

                for i in range(0, n_iterations):
                    print(f"Iteration {i+1} of {n_iterations}...")
                    periods, significances, pdots = periodsearch.find_periods(
                        algorithm,
                        tme_collection[
                            i
                            * period_batch_size : min(
                                n_sources, (i + 1) * period_batch_size
                            )
                        ],
                        freqs,
                        batch_size=period_batch_size,
                        doGPU=doGPU,
                        doCPU=doCPU,
                        doRemoveTerrestrial=doRemoveTerrestrial,
                        doUsePDot=False,
                        doSingleTimeSegment=False,
                        freqs_to_remove=freqs_to_remove,
                        phase_bins=20,
                        mag_bins=10,
                        # Ncore=Ncore, # CPU parallelization to be added
                    )

                    all_periods = np.concatenate([all_periods, periods])
                    all_significances = np.concatenate(
                        [all_significances, significances]
                    )
                    all_pdots = np.concatenate([all_pdots, pdots])

                period_dict[algorithm] = all_periods
                significance_dict[algorithm] = all_significances
                pdot_dict[algorithm] = all_pdots
        else:
            warnings.warn("Skipping period finding; setting all periods to 1.0 d.")
            # Default periods 1.0 d
            period_algorithms = ['Ones']
            period_dict['Ones'] = np.ones(len(tme_collection))
            significance_dict['Ones'] = np.ones(len(tme_collection))
            pdot_dict['Ones'] = np.ones(len(tme_collection))

        for algorithm in period_algorithms:
            for idx, _id in enumerate(keep_id_list):

                period = period_dict[algorithm][idx]
                significance = significance_dict[algorithm][idx]
                pdot = pdot_dict[algorithm][idx]

                tme_dict[_id][f'period_{algorithm}'] = period
                tme_dict[_id][f'significance_{algorithm}'] = significance
                tme_dict[_id][f'pdot_{algorithm}'] = pdot

                feature_dict[_id][f'period_{algorithm}'] = period
                feature_dict[_id][f'significance_{algorithm}'] = significance
                feature_dict[_id][f'pdot_{algorithm}'] = pdot

        print(f'Computing Fourier stats for {len(period_dict)} algorithms...')
        for algorithm in period_algorithms:
            print(f'- Algorithm: {algorithm}')
            fourierStats = Parallel(n_jobs=Ncore)(
                delayed(lcstats.calc_fourier_stats)(
                    id, vals['tme'], vals[f'period_{algorithm}']
                )
                for id, vals in tme_dict.items()
            )

            for statline in fourierStats:
                _id = [x for x in statline.keys()][0]
                statvals = [x for x in statline.values()][0]

                feature_dict[_id][f'f1_power_{algorithm}'] = statvals[0]
                feature_dict[_id][f'f1_BIC_{algorithm}'] = statvals[1]
                feature_dict[_id][f'f1_a_{algorithm}'] = statvals[2]
                feature_dict[_id][f'f1_b_{algorithm}'] = statvals[3]
                feature_dict[_id][f'f1_amp_{algorithm}'] = statvals[4]
                feature_dict[_id][f'f1_phi0_{algorithm}'] = statvals[5]
                feature_dict[_id][f'f1_relamp1_{algorithm}'] = statvals[6]
                feature_dict[_id][f'f1_relphi1_{algorithm}'] = statvals[7]
                feature_dict[_id][f'f1_relamp2_{algorithm}'] = statvals[8]
                feature_dict[_id][f'f1_relphi2_{algorithm}'] = statvals[9]
                feature_dict[_id][f'f1_relamp3_{algorithm}'] = statvals[10]
                feature_dict[_id][f'f1_relphi3_{algorithm}'] = statvals[11]
                feature_dict[_id][f'f1_relamp4_{algorithm}'] = statvals[12]
                feature_dict[_id][f'f1_relphi4_{algorithm}'] = statvals[13]

        print('Computing dmdt histograms...')
        dmdt = Parallel(n_jobs=Ncore)(
            delayed(lcstats.compute_dmdt)(id, vals['tme'], dmdt_ints)
            for id, vals in tme_dict.items()
        )

        for dmdtline in dmdt:
            _id = [x for x in dmdtline.keys()][0]
            dmdtvals = [x for x in dmdtline.values()][0]
            feature_dict[_id]['dmdt'] = dmdtvals.tolist()

        # Get ZTF alert stats
        alert_stats_dct = alertstats.get_ztf_alert_stats(
            feature_dict,
            kowalski_instance=kowalski_instances['kowalski'],
            radius_arcsec=xmatch_radius_arcsec,
            limit=limit,
            Ncore=Ncore,
        )
        for _id in feature_dict.keys():
            feature_dict[_id]['n_ztf_alerts'] = alert_stats_dct[_id]['n_ztf_alerts']
            feature_dict[_id]['mean_ztf_alert_braai'] = alert_stats_dct[_id][
                'mean_ztf_alert_braai'
            ]

        # Add crossmatches to Gaia, AllWISE and PS1 (by default, see config.yaml)
        feature_dict = external_xmatch.xmatch(
            feature_dict,
            kowalski_instances['gloria'],
            catalog_info=ext_catalog_info,
            radius_arcsec=xmatch_radius_arcsec,
            limit=limit,
            Ncore=Ncore,
        )
        feature_df = pd.DataFrame.from_dict(feature_dict, orient='index')

        # Convert various _id datatypes to Int64
        colnames = [x for x in feature_df.columns]
        for col in colnames:
            if '_id' in col:
                feature_df[col] = feature_df[col].astype("Int64")

    else:
        # If baseline is still zero, then no light curves met the selection criteria
        # Generate an empty DF instead so this field/ccd/quad is treated as done
        print('No light curves meet selection criteria.')
        feature_df = pd.DataFrame()

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
        if not doSpecificIDs:
            filename += f"_field_{field}_ccd_{ccd}_quad_{quad}"
            filename += '.parquet'
            dirpath = BASE_DIR / dirname / f"field_{field}"
            os.makedirs(dirpath, exist_ok=True)

            source_count = len(feature_df)
            meta_dct = {}
            meta_dct["(field, ccd, quad)"] = {
                f"({field}, {ccd}, {quad})": {
                    "minobs": min_n_lc_points,
                    "start_time_utc": start_dt,
                    "end_time_utc": end_dt,
                    "ZTF_source_catalog": source_catalog,
                    "ZTF_alerts_catalog": alerts_catalog,
                    "Gaia_catalog": gaia_catalog,
                    "total": source_count,
                }
            }
        else:
            filename += "_specific_ids"
            filename += '.parquet'
            dirpath = BASE_DIR / dirname / "specific_ids"
            os.makedirs(dirpath, exist_ok=True)

            source_count = len(feature_df)
            meta_dct = {}
            meta_dct["specific_ids"] = {
                "specific_ids": {
                    "minobs": min_n_lc_points,
                    "start_time_utc": start_dt,
                    "end_time_utc": end_dt,
                    "ZTF_source_catalog": source_catalog,
                    "ZTF_alerts_catalog": alerts_catalog,
                    "Gaia_catalog": gaia_catalog,
                    "total": source_count,
                }
            }

        meta_filename = BASE_DIR / dirname / "meta.json"

        if os.path.exists(meta_filename):
            with open(meta_filename, 'r') as f:
                dct = json.load(f)
                if not doSpecificIDs:
                    dct["(field, ccd, quad)"].update(meta_dct["(field, ccd, quad)"])
                else:
                    dct["specific_ids"] = meta_dct["specific_ids"]
                meta_dct = dct

        with open(meta_filename, 'w') as f:
            try:
                json.dump(meta_dct, f)
            except Exception as e:
                print("error dumping to json, message: ", e)

        filepath = dirpath / filename
        write_parquet(feature_df, str(filepath))
        print(f"Wrote features for {source_count} sources.")

    else:
        print(f"Generated features for {len(feature_df)} sources.")

    return feature_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source_catalog",
        default=source_catalog,
        help="name of source collection on Kowalski",
    )
    parser.add_argument(
        "--alerts_catalog",
        default=alerts_catalog,
        help="name of alerts collection on Kowalski",
    )
    parser.add_argument(
        "--gaia_catalog",
        default=gaia_catalog,
        help="name of Gaia collection on Kowalski",
    )
    parser.add_argument(
        "--bright_star_query_radius_arcsec",
        type=float,
        default=300.0,
        help="size of cone search radius to search for bright stars",
    )
    parser.add_argument(
        "--xmatch_radius_arcsec",
        type=float,
        default=2.0,
        help="cone radius for all crossmatches",
    )
    parser.add_argument(
        "--query_size_limit",
        type=int,
        default=10000,
        help="sources per query limit for large Kowalski queries",
    )
    parser.add_argument(
        "--period_algorithms",
        nargs='+',
        default=period_algorithms,
        help="to override config, list algorithms to use for period-finding with periodsearch.py",
    )
    parser.add_argument(
        "--period_batch_size",
        type=int,
        default=1000,
        help="batch size for GPU-accelerated period algorithms",
    )
    parser.add_argument(
        "--doCPU",
        action='store_true',
        default=False,
        help="if set, run period-finding algorithm on CPU",
    )
    parser.add_argument(
        "--doGPU",
        action='store_true',
        default=False,
        help="if set, use GPU-accelerated period algorithm",
    )
    parser.add_argument(
        "--samples_per_peak",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--doScaleMinPeriod",
        action='store_true',
        default=False,
        help="if set, scale min period using min_cadence_minutes",
    )
    parser.add_argument(
        "--doRemoveTerrestrial",
        action='store_true',
        default=False,
        help="if set, remove terrestrial frequencies from period analysis",
    )
    parser.add_argument(
        "--Ncore",
        default=8,
        type=int,
        help="number of cores for parallel period finding",
    )
    parser.add_argument(
        "--field",
        type=int,
        default=296,
        help="if not -doAllFields, ZTF field to run on",
    )
    parser.add_argument(
        "--ccd", type=int, default=1, help="if not -doAllCCDs, ZTF ccd to run on"
    )
    parser.add_argument(
        "--quad", type=int, default=1, help="if not -doAllQuads, ZTF field to run on"
    )
    parser.add_argument(
        "--min_n_lc_points",
        type=int,
        default=50,
        help="minimum number of unflagged light curve points to run feature generation",
    )
    parser.add_argument(
        "--min_cadence_minutes",
        type=float,
        default=30.0,
        help="minimum cadence (in minutes) between light curve points. For groups of points closer together than this value, only the first will be kept.",
    )
    parser.add_argument(
        "--dirname",
        type=str,
        default='generated_features',
        help="Directory name for generated features",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default='gen_features',
        help="Prefix for generated feature file",
    )
    parser.add_argument(
        "--doCesium",
        action='store_true',
        default=False,
        help="if set, use Cesium to generate additional features specified in config",
    )
    parser.add_argument(
        "--doNotSave",
        action='store_true',
        default=False,
        help="if set, do not save features",
    )
    parser.add_argument(
        "--stop_early",
        action='store_true',
        default=False,
        help="if set, stop when number of sources reaches query_size_limit. Helpful for testing on small samples.",
    )
    parser.add_argument("--doQuadrantFile", action="store_true", default=False)
    parser.add_argument("--quadrant_file", default="slurm.dat")
    parser.add_argument("--quadrant_index", default=0, type=int)
    parser.add_argument(
        "--doSpecificIDs",
        action='store_true',
        default=False,
        help="if set, perform feature generation for ztf_id column in config-specified file",
    )
    parser.add_argument(
        "--doNotRemoveCloseSources",
        action='store_true',
        default=False,
        help="if set, skip removal of sources too close to bright stars via Gaia. May be useful if input data has previously been analyzed in this way.",
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
        period_algorithms=args.period_algorithms,
        period_batch_size=args.period_batch_size,
        doCPU=args.doCPU,
        doGPU=args.doGPU,
        samples_per_peak=args.samples_per_peak,
        doScaleMinPeriod=args.doScaleMinPeriod,
        doRemoveTerrestrial=args.doRemoveTerrestrial,
        Ncore=args.Ncore,
        field=args.field,
        ccd=args.ccd,
        quad=args.quad,
        min_n_lc_points=args.min_n_lc_points,
        min_cadence_minutes=args.min_cadence_minutes,
        dirname=args.dirname,
        filename=args.filename,
        doCesium=args.doCesium,
        doNotSave=args.doNotSave,
        stop_early=args.stop_early,
        doQuadrantFile=args.doQuadrantFile,
        quadrant_file=args.quadrant_file,
        quadrant_index=args.quadrant_index,
        doSpecificIDs=args.doSpecificIDs,
        doNotRemoveCloseSources=args.doNotRemoveCloseSources,
    )
