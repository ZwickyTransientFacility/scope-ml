#!/usr/bin/env python
"""
Regenerate ZTF training set with the latest period-finding features.

This script:
1. Reads the old training set parquet to extract source IDs and labels
2. Calls generate_features() with top-N periods, FPW, and agreement scoring
3. Merges the new features with old class labels
4. Saves the result as a new parquet

Usage:
    # Small test run (100 sources, CPU)
    python tools/regenerate_training_set.py \
        --old-training-set tools/fritzDownload/merged_classifications_features.parquet \
        --doCPU --stop-early --limit 100 --output new_training_set.parquet

    # Full run (GPU)
    python tools/regenerate_training_set.py \
        --old-training-set tools/fritzDownload/merged_classifications_features.parquet \
        --doGPU --output new_training_set.parquet
"""

import argparse
import json
import os
import pathlib
import pickle
import sys
import tempfile

import numpy as np
import pandas as pd

from scope.utils import read_parquet, write_parquet, parse_load_config

BASE_DIR = pathlib.Path.cwd()
config = parse_load_config()


def _load_label_columns():
    """Return the set of classification column names from the golden dataset mapper."""
    mapper_path = BASE_DIR / 'tools' / 'golden_dataset_mapper.json'
    if mapper_path.exists():
        with open(mapper_path, 'r') as f:
            mapper = json.load(f)
        return set(mapper.keys())
    return set()


# Columns that are identifiers/metadata rather than features or labels
_ID_COLUMNS = {
    '_id',
    'ztf_id',
    'obj_id',
    'fritz_name',
    'ra',
    'dec',
    'radec_geojson',
    'coordinates',
    'field',
    'ccd',
    'quad',
    'filter',
}


def extract_source_list(old_df):
    """Build a source-list DataFrame for generate_features --doSpecificIDs.

    Parameters
    ----------
    old_df : pd.DataFrame
        Old training set with at least 'ztf_id' (or '_id') and coordinate columns.

    Returns
    -------
    source_list_df : pd.DataFrame
        DataFrame with 'ztf_id' and 'coordinates' columns.
    """
    # Determine the ZTF ID column
    if 'ztf_id' in old_df.columns:
        id_col = 'ztf_id'
    elif '_id' in old_df.columns:
        id_col = '_id'
    else:
        raise KeyError("Old training set must have 'ztf_id' or '_id' column.")

    ztf_ids = old_df[id_col].values

    # Build coordinates in the format expected by generate_features
    coords = []
    if 'ra' in old_df.columns and 'dec' in old_df.columns:
        for _, row in old_df.iterrows():
            # ZTF GeoJSON convention: lon = ra - 180
            ra_geojson = row['ra'] - 180.0
            dec_geojson = row['dec']
            coords.append({'radec_geojson': {'coordinates': [ra_geojson, dec_geojson]}})
    elif 'coordinates' in old_df.columns:
        coords = old_df['coordinates'].values.tolist()
    elif 'radec_geojson' in old_df.columns:
        for _, row in old_df.iterrows():
            coords.append({'radec_geojson': row['radec_geojson']})
    else:
        raise KeyError(
            "Old training set must have 'ra'/'dec' or 'coordinates' columns."
        )

    return pd.DataFrame({'ztf_id': ztf_ids, 'coordinates': coords})


def extract_labels(old_df, label_columns):
    """Extract label columns from the old training set.

    Parameters
    ----------
    old_df : pd.DataFrame
        Old training set.
    label_columns : set of str
        Names of classification columns.

    Returns
    -------
    labels_df : pd.DataFrame
        DataFrame with ZTF ID as index and label columns.
    """
    id_col = 'ztf_id' if 'ztf_id' in old_df.columns else '_id'

    # Find label columns that actually exist in the old DataFrame
    present_labels = [c for c in old_df.columns if c in label_columns]

    # Also include obj_id if present (useful for Fritz lookups)
    extra = [c for c in ['obj_id', 'fritz_name'] if c in old_df.columns]

    cols = [id_col] + extra + present_labels
    labels_df = old_df[cols].copy()
    labels_df = labels_df.rename(columns={id_col: 'ztf_id'})
    labels_df = labels_df.set_index('ztf_id')

    return labels_df


def regenerate(
    old_training_set_path,
    output_path='new_training_set.parquet',
    doCPU=False,
    doGPU=False,
    stop_early=False,
    limit=10000,
    top_n_periods=10,
    max_freq=48.0,
    period_batch_size=1000,
    Ncore=8,
    doRemoveTerrestrial=False,
    doCesium=False,
    xmatch_radius_arcsec=2.0,
    period_algorithms=None,
    kowalski_cache=None,
    chunk_index=None,
    n_chunks=None,
    checkpoint_dir=None,
):
    """Regenerate the ZTF training set with updated features.

    Parameters
    ----------
    old_training_set_path : str
        Path to the existing training set parquet.
    output_path : str
        Path for the new training set parquet.
    doCPU, doGPU : bool
        Period-finding device selection.
    stop_early : bool
        If True, limit to ``limit`` sources.
    limit : int
        Maximum sources when stop_early is set.
    top_n_periods : int
        Number of top periods to extract per algorithm.
    max_freq : float
        Maximum frequency for period finding (1/days).
    period_batch_size : int
        Batch size for period finding.
    Ncore : int
        Number of cores for parallel queries.
    doRemoveTerrestrial : bool
        Remove terrestrial frequencies.
    doCesium : bool
        Compute cesium features.
    xmatch_radius_arcsec : float
        Cross-match radius in arcseconds.
    period_algorithms : list or None
        Override period algorithms from config.
    kowalski_cache : str or None
        Path to directory containing pre-downloaded Kowalski data
        (lightcurves.pkl, alert_stats.parquet, xmatch.parquet).
    """
    # Lazy import to avoid loading Kowalski etc at module level
    from tools.generate_features import generate_features

    # --- 0. Load Kowalski cache if provided ---
    local_lightcurves = None
    local_alert_stats = None
    local_xmatch = None
    if kowalski_cache is not None:
        cache_dir = pathlib.Path(kowalski_cache)
        print(f"Loading Kowalski cache from {cache_dir}...")

        lc_path = cache_dir / 'lightcurves.parquet'
        lc_pkl_path = cache_dir / 'lightcurves.pkl'
        if lc_path.exists():
            from tools.download_kowalski_cache import load_lcs_parquet

            local_lightcurves = load_lcs_parquet(lc_path)
            print(f"  Loaded {len(local_lightcurves)} light curves from cache")
        elif lc_pkl_path.exists():
            print(f"  WARNING: Using legacy pickle {lc_pkl_path} (high memory).")
            print("  Run download_kowalski_cache.py to migrate to parquet.")
            with open(lc_pkl_path, 'rb') as f:
                local_lightcurves = pickle.load(f)
            print(f"  Loaded {len(local_lightcurves)} light curves from cache")
        else:
            raise FileNotFoundError(f"Expected {lc_path}")

        alert_path = cache_dir / 'alert_stats.parquet'
        if alert_path.exists():
            alert_df = read_parquet(str(alert_path))
            local_alert_stats = {}
            for _, row in alert_df.iterrows():
                local_alert_stats[row['_id']] = {
                    'n_ztf_alerts': row['n_ztf_alerts'],
                    'mean_ztf_alert_braai': row['mean_ztf_alert_braai'],
                }
            print(
                f"  Loaded alert stats for {len(local_alert_stats)} sources from cache"
            )
        else:
            raise FileNotFoundError(f"Expected {alert_path}")

        xmatch_path = cache_dir / 'xmatch.parquet'
        if xmatch_path.exists():
            xmatch_df = read_parquet(str(xmatch_path))
            local_xmatch = {}
            id_col = '_id' if '_id' in xmatch_df.columns else xmatch_df.columns[0]
            xmatch_cols = [c for c in xmatch_df.columns if c != id_col]
            for _, row in xmatch_df.iterrows():
                local_xmatch[row[id_col]] = {c: row[c] for c in xmatch_cols}
            print(f"  Loaded xmatch data for {len(local_xmatch)} sources from cache")
        else:
            raise FileNotFoundError(f"Expected {xmatch_path}")

    print(f"Loading old training set from {old_training_set_path}...")
    old_df = read_parquet(str(old_training_set_path))
    print(f"  {len(old_df)} sources, {len(old_df.columns)} columns")

    # --- 1. Extract labels ---
    label_columns = _load_label_columns()
    # Also detect any label columns not in mapper (columns with float values in [0,1])
    for col in old_df.columns:
        if col in _ID_COLUMNS or col in label_columns:
            continue
        if old_df[col].dtype in ('float64', 'float32'):
            vals = old_df[col].dropna()
            if len(vals) > 0 and vals.min() >= 0.0 and vals.max() <= 1.0:
                # Check if this looks like a probability column (many 0s and 1s)
                frac_binary = ((vals == 0.0) | (vals == 1.0)).mean()
                if frac_binary > 0.5:
                    label_columns.add(col)

    labels_df = extract_labels(old_df, label_columns)
    present_labels = [c for c in labels_df.columns if c in label_columns]
    print(f"  Extracted {len(present_labels)} label columns")

    # --- 2. Build source list ---
    source_list_df = extract_source_list(old_df)

    # --- 2b. Chunk slicing ---
    if chunk_index is not None and n_chunks is not None:
        total = len(source_list_df)
        chunk_size = int(np.ceil(total / n_chunks))
        start = chunk_index * chunk_size
        end = min(start + chunk_size, total)
        source_list_df = source_list_df.iloc[start:end].reset_index(drop=True)
        print(
            f"  Chunk {chunk_index}/{n_chunks}: sources {start}-{end} ({len(source_list_df)} sources)"
        )

        # Filter local caches to only this chunk's sources
        chunk_ids = set(source_list_df['ztf_id'])
        if local_lightcurves is not None:
            local_lightcurves = [
                lc for lc in local_lightcurves if lc['_id'] in chunk_ids
            ]
            print(f"  Filtered light curves to {len(local_lightcurves)} for this chunk")
        if local_alert_stats is not None:
            local_alert_stats = {
                k: v for k, v in local_alert_stats.items() if k in chunk_ids
            }
        if local_xmatch is not None:
            local_xmatch = {k: v for k, v in local_xmatch.items() if k in chunk_ids}

    # Save source list to temp parquet for generate_features
    with tempfile.TemporaryDirectory() as tmpdir:
        source_list_path = os.path.join(tmpdir, 'source_list.parquet')
        write_parquet(source_list_df, source_list_path)
        print(f"  Wrote source list for {len(source_list_df)} sources")

        # --- 3. Run feature generation ---
        print("\nRunning feature generation...")
        # Set up checkpoint directory
        ckpt_dir = checkpoint_dir
        if ckpt_dir is None and chunk_index is not None:
            # Auto-create per-chunk checkpoint dir
            ckpt_dir = str(
                pathlib.Path(output_path).parent / f'checkpoints/chunk_{chunk_index}'
            )
        elif ckpt_dir is None:
            ckpt_dir = str(pathlib.Path(output_path).parent / 'checkpoints')

        kwargs = dict(
            doSpecificIDs=True,
            skipCloseSources=True,
            fg_dataset=source_list_path,
            doCPU=doCPU,
            doGPU=doGPU,
            doCesium=doCesium,
            top_n_periods=top_n_periods,
            max_freq=max_freq,
            period_batch_size=period_batch_size,
            Ncore=Ncore,
            doRemoveTerrestrial=doRemoveTerrestrial,
            xmatch_radius_arcsec=xmatch_radius_arcsec,
            doNotSave=True,
            stop_early=stop_early,
            limit=limit,
            local_lightcurves=local_lightcurves,
            local_alert_stats=local_alert_stats,
            local_xmatch=local_xmatch,
            checkpoint_dir=ckpt_dir,
        )
        if period_algorithms is not None:
            kwargs['period_algorithms'] = period_algorithms

        new_feature_df = generate_features(**kwargs)

    if len(new_feature_df) == 0:
        print("ERROR: Feature generation returned no sources.")
        sys.exit(1)

    # --- 4. Merge new features with old labels ---
    print(f"\nMerging {len(new_feature_df)} new feature rows with labels...")

    # Ensure consistent ID column
    if '_id' in new_feature_df.columns:
        new_feature_df = new_feature_df.rename(columns={'_id': 'ztf_id'})
    new_feature_df = new_feature_df.set_index('ztf_id')

    merged_df = new_feature_df.join(labels_df, how='inner')

    # --- 5. Validate ---
    n_merged = len(merged_df)
    n_old = len(old_df)
    n_new = len(new_feature_df)
    print("\nValidation:")
    print(f"  Old training set:  {n_old} sources")
    print(f"  New features:      {n_new} sources")
    print(f"  Merged (inner):    {n_merged} sources")

    if n_merged < n_new * 0.9:
        print(f"  WARNING: Lost >10% of sources in merge ({n_new - n_merged} lost)")

    # Check label preservation
    label_cols_in_merged = [c for c in merged_df.columns if c in label_columns]
    print(f"  Label columns:     {len(label_cols_in_merged)}")

    # Check for new feature columns
    new_feature_cols = [
        c for c in merged_df.columns if 'agree' in c or 'consensus' in c
    ]
    topn_cols = [
        c for c in merged_df.columns if c.startswith('period_') and '_' in c[7:]
    ]
    print(f"  Agreement features: {len(new_feature_cols)}")
    print(f"  Top-N period cols:  {len(topn_cols)}")

    # Check that old cesium features are absent (if doCesium=False)
    if not doCesium:
        cesium_cols = [
            c
            for c in merged_df.columns
            if c in config.get('feature_generation', {}).get('cesium_features', [])
        ]
        if cesium_cols:
            print(
                f"  NOTE: {len(cesium_cols)} cesium columns still present (from labels join)"
            )

    # --- 6. Save ---
    merged_df = merged_df.reset_index()
    write_parquet(merged_df, str(output_path))
    print(f"\nSaved new training set to {output_path}")
    print(f"  {len(merged_df)} sources, {len(merged_df.columns)} columns")

    return merged_df


def get_parser():
    parser = argparse.ArgumentParser(
        description="Regenerate ZTF training set with latest period-finding features."
    )
    parser.add_argument(
        "--old-training-set",
        type=str,
        required=True,
        help="Path to the existing training set parquet",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="new_training_set.parquet",
        help="Output path for the new training set",
    )
    parser.add_argument("--doCPU", action='store_true', default=False)
    parser.add_argument("--doGPU", action='store_true', default=False)
    parser.add_argument("--stop-early", action='store_true', default=False)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--top-n-periods", type=int, default=10)
    parser.add_argument("--max-freq", type=float, default=48.0)
    parser.add_argument("--period-batch-size", type=int, default=1000)
    parser.add_argument("--Ncore", type=int, default=8)
    parser.add_argument("--doRemoveTerrestrial", action='store_true', default=False)
    parser.add_argument("--doCesium", action='store_true', default=False)
    parser.add_argument("--xmatch-radius-arcsec", type=float, default=2.0)
    parser.add_argument(
        "--period-algorithms",
        nargs='+',
        default=None,
        help="Override period algorithms from config",
    )
    parser.add_argument(
        "--kowalski-cache",
        type=str,
        default=None,
        help="Path to directory with pre-downloaded Kowalski data (lightcurves.pkl, alert_stats.parquet, xmatch.parquet)",
    )
    parser.add_argument(
        "--chunk-index",
        type=int,
        default=None,
        help="Chunk index for split-job parallelism (0-based)",
    )
    parser.add_argument(
        "--n-chunks",
        type=int,
        default=None,
        help="Total number of chunks for split-job parallelism",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory for period-finding checkpoints (auto-created if not set)",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    regenerate(
        old_training_set_path=args.old_training_set,
        output_path=args.output,
        doCPU=args.doCPU,
        doGPU=args.doGPU,
        stop_early=args.stop_early,
        limit=args.limit,
        top_n_periods=args.top_n_periods,
        max_freq=args.max_freq,
        period_batch_size=args.period_batch_size,
        Ncore=args.Ncore,
        doRemoveTerrestrial=args.doRemoveTerrestrial,
        doCesium=args.doCesium,
        xmatch_radius_arcsec=args.xmatch_radius_arcsec,
        period_algorithms=args.period_algorithms,
        kowalski_cache=args.kowalski_cache,
        chunk_index=args.chunk_index,
        n_chunks=args.n_chunks,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
