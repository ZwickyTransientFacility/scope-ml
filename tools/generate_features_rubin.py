#!/usr/bin/env python
"""
Feature generation pipeline for Rubin LSST lightcurves.

Parallel to tools/generate_features.py, this script fetches Rubin forced
photometry via TAP, computes the same statistical features, and outputs
a parquet file compatible with the scope-ml training pipeline.

Key differences from generate_features.py:
- Source selection via cone search or objectId list (no field/CCD/quad)
- Uses RubinTAPClient.get_lightcurves() instead of get_lightcurves_via_ids()
- Skips ZTF alert stats (n_ztf_alerts, mean_ztf_alert_braai → set to 0)
- External crossmatches use Kowalski if available, otherwise skipped
- Stores coord_ra/coord_dec directly (no GeoJSON 180-degree offset)
"""

import scope
import argparse
import pathlib
import os
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
import json
import time

from scope.utils import (
    removeHighCadence,
    write_parquet,
    sort_lightcurve,
    parse_load_config,
)
from scope.rubin import RubinTAPClient
from tools.get_rubin_ids import get_rubin_objects_by_cone, get_rubin_objects_from_file
from tools.featureGeneration import lcstats, periodsearch
from joblib import Parallel, delayed

BASE_DIR = pathlib.Path.cwd()
config = parse_load_config()

# Rubin TAP token from environment variable
rubin_token_env = os.environ.get("RUBIN_TAP_TOKEN")
if rubin_token_env is not None:
    config["rubin"]["token"] = rubin_token_env

# Config values
dmdt_ints = config['feature_generation']['dmdt_ints']
period_algorithms = config['feature_generation']['period_algorithms']
path_to_features = config['feature_generation']['path_to_features']
period_search_config = config['feature_generation'].get('period_search', {})

if path_to_features is not None:
    BASE_DIR = pathlib.Path(path_to_features)

# Optional: Kowalski for external crossmatches
kowalski_instances = None
try:
    from penquins import Kowalski

    kowalski_token_env = os.environ.get("KOWALSKI_INSTANCE_TOKEN")
    gloria_token_env = os.environ.get("GLORIA_INSTANCE_TOKEN")
    melman_token_env = os.environ.get("MELMAN_INSTANCE_TOKEN")

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
    if len(hosts) > 0:
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
except Exception:
    pass


def generate_features_rubin(
    ra=None,
    dec=None,
    radius_arcsec=30.0,
    objectid_file=None,
    bands=None,
    period_algorithms=period_algorithms,
    period_batch_size=period_search_config.get('period_batch_size', 1000),
    doCPU=False,
    doGPU=False,
    samples_per_peak=period_search_config.get('samples_per_peak', 10),
    doScaleMinPeriod=False,
    doRemoveTerrestrial=False,
    Ncore=8,
    min_n_lc_points=period_search_config.get('min_n_lc_points', 50),
    min_cadence_minutes=period_search_config.get('min_cadence_minutes', 5.0),
    dirname='generated_features_rubin',
    filename='gen_features_rubin',
    doNotSave=False,
    stop_early=False,
    limit=10000,
    top_n_periods=50,
    max_freq=period_search_config.get('max_freq', 288.0),
    xmatch_radius_arcsec=2.0,
    phase_bins=period_search_config.get('phase_bins', 20),
    mag_bins=period_search_config.get('mag_bins', 10),
):
    """
    Generate features for Rubin LSST light curves.

    Parameters
    ----------
    ra : float, optional
        Right ascension in degrees (for cone search).
    dec : float, optional
        Declination in degrees (for cone search).
    radius_arcsec : float, optional
        Cone search radius in arcseconds.
    objectid_file : str, optional
        Path to CSV file with objectId column.
    bands : list of str, optional
        Bands to use (e.g., ['g', 'r']). None = all bands.
    period_algorithms : dict or list
        Period-finding algorithms to use.
    period_batch_size : int
        Batch size for period finding.
    doCPU : bool
        Run period finding on CPU.
    doGPU : bool
        Run period finding on GPU.
    samples_per_peak : int
        Number of frequency samples per peak.
    doScaleMinPeriod : bool
        Scale minimum period from cadence.
    doRemoveTerrestrial : bool
        Remove terrestrial frequencies.
    Ncore : int
        Number of parallel cores.
    min_n_lc_points : int
        Minimum lightcurve points required.
    min_cadence_minutes : float
        Minimum cadence between points.
    dirname : str
        Output directory name.
    filename : str
        Output filename prefix.
    doNotSave : bool
        Skip saving output.
    stop_early : bool
        Stop after limit sources.
    limit : int
        Maximum sources for cone search / batch queries.
    top_n_periods : int
        Number of top periods for nested algorithms.
    max_freq : float
        Maximum frequency for period finding (cycles/day).
        288.0 = 5-minute periods, 48.0 = 30-minute periods.
    xmatch_radius_arcsec : float
        Cross-match radius in arcseconds.
    phase_bins : int
        Number of phase bins for CE/AOV/FPW.
    mag_bins : int
        Number of magnitude bins for CE.

    Returns
    -------
    pd.DataFrame
        DataFrame with generated features.
    """
    t0 = time.time()

    # Get code version and current date/time for metadata
    code_version = scope.__version__
    utcnow = datetime.utcnow()
    start_dt = utcnow.strftime("%Y-%m-%d %H:%M:%S")

    # --- 1. Source Discovery ---
    print("Discovering Rubin sources...")
    if objectid_file is not None:
        objects = get_rubin_objects_from_file(objectid_file)
    elif ra is not None and dec is not None:
        objects = get_rubin_objects_by_cone(ra, dec, radius_arcsec, limit=limit)
    else:
        raise ValueError("Must specify either --ra/--dec or --objectid-file")

    if stop_early and len(objects) > limit:
        objectids = list(objects.keys())[:limit]
        objects = {oid: objects[oid] for oid in objectids}

    if len(objects) == 0:
        print("No objects found.")
        return pd.DataFrame()

    # --- 2. Lightcurve Retrieval ---
    print(f"Fetching lightcurves for {len(objects)} objects...")
    rubin_config = config.get("rubin", {})
    client = RubinTAPClient(config=rubin_config)
    objectids = list(objects.keys())
    lcs = client.get_lightcurves(objectids, bands=bands)

    if len(lcs) == 0:
        print("No lightcurves retrieved.")
        return pd.DataFrame()

    # Build mapping from objectId -> object metadata
    # One lc entry per (objectId, band) pair
    lc_ids = set(lc['_id'] for lc in lcs)
    feature_gen_source_dict = {oid: objects[oid] for oid in objects if oid in lc_ids}

    # Select period algorithms from config based on CPU or GPU specification
    pa = period_algorithms
    if isinstance(pa, dict):
        if doCPU:
            pa = pa['CPU']
        elif doGPU:
            pa = pa['GPU']

    # --- 3. Lightcurve Preprocessing ---
    print("Analyzing lightcurves and computing basic features...")
    feature_dict = {}
    keep_id_list = []
    tme_collection = []
    tme_dict = {}
    baseline = 0

    # Group lightcurves by objectId: combine all bands
    lc_by_object = {}
    for lc in lcs:
        oid = lc['_id']
        if oid not in lc_by_object:
            lc_by_object[oid] = []
        lc_by_object[oid].append(lc)

    count = 0
    for oid, lc_list in lc_by_object.items():
        count += 1
        if count % 1000 == 0:
            print(f"{count} done")

        # Combine all data points across bands for this object
        all_data = []
        filters_used = set()
        for lc in lc_list:
            filters_used.add(lc['filter'])
            # Filter on catflags == 0 (unflagged points)
            unflagged = [x for x in lc['data'] if x['catflags'] == 0]
            all_data.extend(unflagged)

        if len(all_data) == 0:
            continue

        tme = [[x['hjd'], x['mag'], x['magerr']] for x in all_data]
        try:
            tme_arr = np.array(tme)
            t, m, e = tme_arr.transpose()

            # Sort by time
            t, m, e = sort_lightcurve(t, m, e)

            # Remove high-cadence duplicates
            tt, mm, ee = removeHighCadence(t, m, e, cadence_minutes=min_cadence_minutes)

            # Check minimum points
            if len(tt) < min_n_lc_points:
                continue

            keep_id_list.append(oid)

            new_baseline = max(tt) - min(tt)
            if new_baseline > baseline:
                baseline = new_baseline

            new_tme_arr = np.array([tt, mm, ee])
            tme_collection.append(new_tme_arr)
            tme_dict[oid] = {'tme': new_tme_arr}

            # Store feature info
            feature_dict[oid] = {}
            obj_info = feature_gen_source_dict.get(oid, {})
            feature_dict[oid]['ra'] = obj_info.get('coord_ra', 0.0)
            feature_dict[oid]['dec'] = obj_info.get('coord_dec', 0.0)
            feature_dict[oid]['filter'] = list(filters_used)
            feature_dict[oid]['survey'] = 'rubin_dp1'

        except ValueError:
            continue

    print(f"{count} done")
    print(f"Kept {len(keep_id_list)} objects with >= {min_n_lc_points} points.")

    if len(keep_id_list) == 0 or baseline == 0:
        print("No light curves meet selection criteria.")
        feature_df = pd.DataFrame()
        _save_results(
            feature_df, start_dt, code_version, dirname, filename, doNotSave, BASE_DIR
        )
        return feature_df

    # --- 4. Basic Statistics ---
    print("Computing basic statistics...")
    id_list_bs = list(tme_dict.keys())
    lightcurves_bs = [tme_dict[_id]['tme'] for _id in id_list_bs]
    basic_stats_arr = periodsearch.compute_basic_stats(lightcurves_bs)

    stat_names = [
        'n',
        'median',
        'wmean',
        'chi2red',
        'roms',
        'wstd',
        'norm_peak_to_peak_amp',
        'norm_excess_var',
        'median_abs_dev',
        'iqr',
        'i60r',
        'i70r',
        'i80r',
        'i90r',
        'skew',
        'smallkurt',
        'inv_vonneumannratio',
        'welch_i',
        'stetson_j',
        'stetson_k',
        'ad',
        'sw',
    ]

    for idx, _id in enumerate(id_list_bs):
        for si, name in enumerate(stat_names):
            feature_dict[_id][name] = float(basic_stats_arr[idx, si])

    # --- 5. Period Finding ---
    if doScaleMinPeriod:
        fmin, fmax = 2 / baseline, 1 / (2 * min_cadence_minutes / 1440)
    else:
        fmin, fmax = 2 / baseline, max_freq

    df = 1.0 / (samples_per_peak * baseline)
    nf = int(np.ceil((fmax - fmin) / df))
    freqs = fmin + df * np.arange(nf)

    # Terrestrial frequencies
    if doRemoveTerrestrial:
        freqs_to_remove = [
            [0.0025, 0.003],
            [0.00125, 0.0015],
            [0.000833, 0.001],
            [0.000625, 0.00075],
            [0.0005, 0.0006],
            [0.005, 0.006],
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

    freqs_copy = freqs.copy()
    if freqs_to_remove is not None:
        for pair in freqs_to_remove:
            idx = np.where((freqs_copy < pair[0]) | (freqs_copy > pair[1]))[0]
            freqs_copy = freqs_copy[idx]
    freqs_no_terrestrial = freqs_copy

    period_dict = {}
    significance_dict = {}
    pdot_dict = {}
    do_nested_algorithms = False

    if doCPU or doGPU:
        if doCPU and doGPU:
            raise ValueError('Please set only one of --doCPU or --doGPU.')

        if 'ELS_ECE_EAOV' in pa:
            pa = ['ELS_periodogram', 'ECE_periodogram', 'EAOV_periodogram']
            do_nested_algorithms = True
            warnings.warn(
                'Performing nested ELS/ECE -> EAOV period search. '
                'Other algorithms in config will be ignored.'
            )
        elif 'LS_CE_AOV' in pa:
            pa = ['LS_periodogram', 'CE_periodogram', 'AOV_periodogram']
            do_nested_algorithms = True
            warnings.warn(
                'Performing nested LS/CE -> AOV period search. '
                'Other algorithms in config will be ignored.'
            )

        n_sources = len(feature_dict)
        if n_sources % period_batch_size != 0:
            n_iterations = n_sources // period_batch_size + 1
        else:
            n_iterations = n_sources // period_batch_size

        all_periods = {algorithm: [] for algorithm in pa}
        all_significances = {algorithm: [] for algorithm in pa}
        all_pdots = {algorithm: [] for algorithm in pa}

        if do_nested_algorithms:
            nested_key = 'ELS_ECE_EAOV'
            all_periods[nested_key] = []
            all_significances[nested_key] = []
            all_pdots[nested_key] = []

        print(
            f'Running {len(pa)} period algorithms for {n_sources} sources '
            f'in batches of {period_batch_size}...'
        )
        for i in range(n_iterations):
            print(f"Iteration {i+1} of {n_iterations}...")

            for algorithm in pa:
                print(f'Running {algorithm} algorithm:')
                periods, significances, pdots = periodsearch.find_periods(
                    algorithm,
                    tme_collection[
                        i
                        * period_batch_size : min(
                            n_sources, (i + 1) * period_batch_size
                        )
                    ],
                    freqs,
                    doGPU=doGPU,
                    doCPU=doCPU,
                    doRemoveTerrestrial=doRemoveTerrestrial,
                    doUsePDot=False,
                    doSingleTimeSegment=False,
                    freqs_to_remove=freqs_to_remove,
                    phase_bins=phase_bins,
                    mag_bins=mag_bins,
                    Ncore=Ncore,
                )

                if not do_nested_algorithms:
                    all_periods[algorithm] = np.concatenate(
                        [all_periods[algorithm], periods]
                    )
                else:
                    p_vals = [p['period'] for p in periods]
                    p_stats = [p['data'] for p in periods]
                    all_periods[algorithm] = np.concatenate(
                        [all_periods[algorithm], p_vals]
                    )
                    if algorithm in ('ELS_periodogram', 'LS_periodogram'):
                        topN_significance_indices_ELS = [
                            np.argsort(ps.flatten())[::-1][:top_n_periods]
                            for ps in p_stats
                        ]
                    elif algorithm in ('ECE_periodogram', 'CE_periodogram'):
                        topN_significance_indices_ECE = [
                            np.argsort(ps.flatten())[:top_n_periods] for ps in p_stats
                        ]
                    elif algorithm in ('EAOV_periodogram', 'AOV_periodogram'):
                        ELS_ECE_top_indices = np.concatenate(
                            [
                                topN_significance_indices_ELS,
                                topN_significance_indices_ECE,
                            ],
                            axis=1,
                        )
                        ELS_ECE_top_indices = [
                            np.unique(x) for x in ELS_ECE_top_indices
                        ]
                        best_index_of_indices = [
                            np.argmax(p_stats[i][ELS_ECE_top_indices[i]])
                            for i in range(len(p_stats))
                        ]
                        best_indices = [
                            ELS_ECE_top_indices[i][best_index_of_indices[i]]
                            for i in range(len(ELS_ECE_top_indices))
                        ]
                        all_periods[nested_key] = np.concatenate(
                            [
                                all_periods[nested_key],
                                1 / freqs_no_terrestrial[best_indices],
                            ]
                        )
                        all_significances[nested_key] = np.concatenate(
                            [
                                all_significances[nested_key],
                                [
                                    p_stats[i].flatten()[best_indices[i]]
                                    for i in range(len(best_indices))
                                ],
                            ]
                        )
                        all_pdots[nested_key] = np.concatenate(
                            [all_pdots[nested_key], pdots]
                        )

                all_significances[algorithm] = np.concatenate(
                    [all_significances[algorithm], significances]
                )
                all_pdots[algorithm] = np.concatenate([all_pdots[algorithm], pdots])

        period_dict = all_periods
        significance_dict = all_significances
        pdot_dict = all_pdots

        if do_nested_algorithms:
            pa = pa + [nested_key]

    else:
        warnings.warn("Skipping period finding; setting all periods to 1.0 d.")
        pa = ['Ones']
        period_dict['Ones'] = np.ones(len(tme_collection))
        significance_dict['Ones'] = np.ones(len(tme_collection))
        pdot_dict['Ones'] = np.ones(len(tme_collection))

    for algorithm in pa:
        if algorithm not in ["ELS_ECE_EAOV", "LS_CE_AOV"]:
            algorithm_name = algorithm.split('_')[0]
        else:
            algorithm_name = algorithm

        for idx, _id in enumerate(keep_id_list):
            period = period_dict[algorithm][idx]
            significance = significance_dict[algorithm][idx]
            pdot = pdot_dict[algorithm][idx]

            tme_dict[_id][f'period_{algorithm_name}'] = period
            tme_dict[_id][f'significance_{algorithm_name}'] = significance
            tme_dict[_id][f'pdot_{algorithm_name}'] = pdot

            feature_dict[_id][f'period_{algorithm_name}'] = period
            feature_dict[_id][f'significance_{algorithm_name}'] = significance
            feature_dict[_id][f'pdot_{algorithm_name}'] = pdot

    # --- 6. Fourier Statistics ---
    print(f'Computing Fourier stats for {len(period_dict)} algorithms...')
    id_list = list(tme_dict.keys())
    lightcurves_ordered = [tme_dict[_id]['tme'] for _id in id_list]

    fourier_names = [
        'f1_power',
        'f1_BIC',
        'f1_a',
        'f1_b',
        'f1_amp',
        'f1_phi0',
        'f1_relamp1',
        'f1_relphi1',
        'f1_relamp2',
        'f1_relphi2',
        'f1_relamp3',
        'f1_relphi3',
        'f1_relamp4',
        'f1_relphi4',
    ]

    for algorithm in pa:
        if algorithm not in ["ELS_ECE_EAOV", "LS_CE_AOV"]:
            algorithm_name = algorithm.split('_')[0]
        else:
            algorithm_name = algorithm
        print(f'- Algorithm: {algorithm}')

        periods_for_algo = np.array(
            [tme_dict[_id][f'period_{algorithm_name}'] for _id in id_list],
            dtype=np.float32,
        )
        fourier_features = periodsearch.compute_fourier_features(
            lightcurves_ordered, periods_for_algo
        )

        for idx, _id in enumerate(id_list):
            for i, name in enumerate(fourier_names):
                feature_dict[_id][f'{name}_{algorithm_name}'] = float(
                    fourier_features[idx, i]
                )

    # --- 7. dmdt Histograms ---
    print('Computing dmdt histograms...')
    id_list_dmdt = list(tme_dict.keys())
    lightcurves_dmdt = [tme_dict[_id]['tme'] for _id in id_list_dmdt]
    dmdt_arr = periodsearch.compute_dmdt_features(lightcurves_dmdt, dmdt_ints)

    for idx, _id in enumerate(id_list_dmdt):
        feature_dict[_id]['dmdt'] = dmdt_arr[idx].tolist()

    # --- 8. Alert Stats (skipped for Rubin) ---
    print('Skipping ZTF alert stats for Rubin data (setting to 0)...')
    for _id in feature_dict.keys():
        feature_dict[_id]['n_ztf_alerts'] = 0
        feature_dict[_id]['mean_ztf_alert_braai'] = 0.0

    # --- 9. External Cross-matches (if Kowalski available) ---
    if kowalski_instances is not None:
        try:
            from tools.featureGeneration import external_xmatch

            ext_catalog_info = config['feature_generation']['external_catalog_features']
            print('Computing external cross-matches via Kowalski...')
            feature_dict = external_xmatch.xmatch(
                feature_dict,
                kowalski_instances,
                ext_catalog_info,
                radius_arcsec=xmatch_radius_arcsec,
                limit=limit,
                Ncore=Ncore,
            )
        except Exception as e:
            warnings.warn(f"External cross-match failed: {e}")
    else:
        print("Skipping external cross-matches (Kowalski not available).")

    # --- 10. Build DataFrame ---
    feature_df = pd.DataFrame.from_dict(feature_dict, orient='index')
    feature_df.index.set_names('_id', inplace=True)
    feature_df.reset_index(inplace=True)

    # Convert _id columns to Int64
    colnames = [x for x in feature_df.columns]
    for col in colnames:
        if '_id' in col:
            try:
                feature_df[col] = feature_df[col].astype("Int64")
            except (ValueError, TypeError):
                pass

    # --- 11. Save ---
    _save_results(
        feature_df, start_dt, code_version, dirname, filename, doNotSave, BASE_DIR
    )

    t1 = time.time()
    print(f"Finished running in {t1 - t0} seconds.")

    return feature_df


def _save_results(
    feature_df, start_dt, code_version, dirname, filename, doNotSave, base_dir
):
    """Save feature DataFrame and metadata."""
    utcnow = datetime.utcnow()
    end_dt = utcnow.strftime("%Y-%m-%d %H:%M:%S")

    feature_df.attrs['scope_code_version'] = code_version
    feature_df.attrs['feature_generation_start_dateTime_utc'] = start_dt
    feature_df.attrs['feature_generation_end_dateTime_utc'] = end_dt
    feature_df.attrs['survey'] = 'rubin_dp1'

    if not doNotSave:
        filename += '.parquet'
        dirpath = base_dir / dirname
        os.makedirs(dirpath, exist_ok=True)

        source_count = len(feature_df)
        meta_dct = {
            "rubin": {
                "start_time_utc": start_dt,
                "end_time_utc": end_dt,
                "survey": "rubin_dp1",
                "total": source_count,
            }
        }

        meta_filename = dirpath / "meta.json"
        if os.path.exists(meta_filename):
            with open(meta_filename, 'r') as f:
                dct = json.load(f)
                dct.update(meta_dct)
                meta_dct = dct

        with open(meta_filename, 'w') as f:
            try:
                json.dump(meta_dct, f)
            except Exception as e:
                print("error dumping to json, message: ", e)

        filepath = dirpath / filename
        write_parquet(feature_df, str(filepath))
        print(f"Wrote features for {source_count} sources to {filepath}.")
    else:
        print(f"Generated features for {len(feature_df)} sources.")


def get_parser(**kwargs):
    add_help = kwargs.get("add_help", True)
    parser = argparse.ArgumentParser(
        description="Generate features from Rubin LSST lightcurves.",
        add_help=add_help,
    )

    parser.add_argument(
        "--ra",
        type=float,
        default=None,
        help="Right ascension in degrees (for cone search)",
    )
    parser.add_argument(
        "--dec",
        type=float,
        default=None,
        help="Declination in degrees (for cone search)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=30.0,
        help="Cone search radius in arcseconds (default 30)",
    )
    parser.add_argument(
        "--objectid-file",
        type=str,
        default=None,
        help="Path to CSV file with objectId column",
    )
    parser.add_argument(
        "--bands",
        nargs='+',
        default=None,
        help="Bands to use (e.g., g r i). Default: all bands.",
    )
    parser.add_argument(
        "--period-algorithms",
        nargs='+',
        default=period_algorithms,
        help="Period-finding algorithms to use",
    )
    parser.add_argument(
        "--period-batch-size",
        type=int,
        default=period_search_config.get('period_batch_size', 1000),
        help="Batch size for period algorithms",
    )
    parser.add_argument(
        "--doCPU",
        action='store_true',
        default=False,
        help="Run period finding on CPU",
    )
    parser.add_argument(
        "--doGPU",
        action='store_true',
        default=False,
        help="Run period finding on GPU",
    )
    parser.add_argument(
        "--samples-per-peak",
        default=period_search_config.get('samples_per_peak', 10),
        type=int,
    )
    parser.add_argument(
        "--doScaleMinPeriod",
        action='store_true',
        default=False,
        help="Scale min period using --min-cadence-minutes",
    )
    parser.add_argument(
        "--doRemoveTerrestrial",
        action='store_true',
        default=False,
        help="Remove terrestrial frequencies from period analysis",
    )
    parser.add_argument(
        "--Ncore",
        default=8,
        type=int,
        help="Number of cores for parallel processing",
    )
    parser.add_argument(
        "--min-n-lc-points",
        type=int,
        default=period_search_config.get('min_n_lc_points', 50),
        help="Minimum lightcurve points to generate features",
    )
    parser.add_argument(
        "--min-cadence-minutes",
        type=float,
        default=period_search_config.get('min_cadence_minutes', 5.0),
        help="Minimum cadence between lightcurve points (minutes)",
    )
    parser.add_argument(
        "--phase-bins",
        type=int,
        default=period_search_config.get('phase_bins', 20),
        help="Number of phase bins for CE/AOV/FPW",
    )
    parser.add_argument(
        "--mag-bins",
        type=int,
        default=period_search_config.get('mag_bins', 10),
        help="Number of magnitude bins for CE",
    )
    parser.add_argument(
        "--dirname",
        type=str,
        default='generated_features_rubin',
        help="Directory name for generated features",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default='gen_features_rubin',
        help="Prefix for generated feature file",
    )
    parser.add_argument(
        "--doNotSave",
        action='store_true',
        default=False,
        help="Do not save features to disk",
    )
    parser.add_argument(
        "--stop-early",
        action='store_true',
        default=False,
        help="Stop after --limit sources",
    )
    parser.add_argument(
        "--query-size-limit",
        type=int,
        default=10000,
        help="Maximum sources for cone search / batch queries",
    )
    parser.add_argument(
        "--top-n-periods",
        type=int,
        default=50,
        help="Number of top periods for nested algorithms",
    )
    parser.add_argument(
        "--max-freq",
        type=float,
        default=period_search_config.get('max_freq', 288.0),
        help="Maximum frequency [1/days] for period finding (288=5min, 48=30min)",
    )
    parser.add_argument(
        "--xmatch-radius-arcsec",
        type=float,
        default=2.0,
        help="Cross-match radius in arcseconds",
    )
    return parser


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()

    generate_features_rubin(
        ra=args.ra,
        dec=args.dec,
        radius_arcsec=args.radius,
        objectid_file=args.objectid_file,
        bands=args.bands,
        period_algorithms=args.period_algorithms,
        period_batch_size=args.period_batch_size,
        doCPU=args.doCPU,
        doGPU=args.doGPU,
        samples_per_peak=args.samples_per_peak,
        doScaleMinPeriod=args.doScaleMinPeriod,
        doRemoveTerrestrial=args.doRemoveTerrestrial,
        Ncore=args.Ncore,
        min_n_lc_points=args.min_n_lc_points,
        min_cadence_minutes=args.min_cadence_minutes,
        dirname=args.dirname,
        filename=args.filename,
        doNotSave=args.doNotSave,
        stop_early=args.stop_early,
        limit=args.query_size_limit,
        top_n_periods=args.top_n_periods,
        max_freq=args.max_freq,
        xmatch_radius_arcsec=args.xmatch_radius_arcsec,
        phase_bins=args.phase_bins,
        mag_bins=args.mag_bins,
    )


if __name__ == "__main__":
    main()
