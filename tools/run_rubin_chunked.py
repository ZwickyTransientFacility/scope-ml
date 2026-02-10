#!/usr/bin/env python
"""
Chunked runner for Rubin DP1 feature generation with resume capability.

Splits the full object ID list into chunks, runs the feature generation
pipeline on each chunk independently, and saves per-chunk parquet files.
If the session dies, restarting picks up where it left off (completed
chunks are detected by their output files).

Usage:
    python tools/run_rubin_chunked.py --objectid-file rubin_dp1_all_ids.csv \
        --doGPU --chunk-size 10000

Final merged output: generated_features_rubin/gen_features_rubin_full.parquet
"""

import argparse
import os
import sys
import time
import pathlib
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tools.generate_features_rubin import generate_features_rubin


def run_chunked(
    objectid_file,
    chunk_size=10000,
    dirname="generated_features_rubin",
    doGPU=False,
    doCPU=False,
    min_n_lc_points=50,
    max_freq=288.0,
    samples_per_peak=10,
    period_batch_size=1000,
    phase_bins=20,
    mag_bins=10,
    Ncore=8,
    top_n_periods=1,
    chunk_subdir="chunks",
    output_filename="gen_features_rubin_full",
):
    """Run feature generation in resumable chunks."""
    t0 = time.time()

    # Load all object IDs
    all_ids_df = pd.read_csv(objectid_file)
    all_ids = all_ids_df["objectId"].values
    n_total = len(all_ids)
    n_chunks = (n_total + chunk_size - 1) // chunk_size

    print(f"Total objects: {n_total:,}")
    print(f"Chunk size: {chunk_size:,}")
    print(f"Number of chunks: {n_chunks}")
    print(f"GPU: {doGPU}, CPU: {doCPU}")
    print()

    dirpath = pathlib.Path(dirname)
    os.makedirs(dirpath, exist_ok=True)
    chunk_dir = dirpath / chunk_subdir
    os.makedirs(chunk_dir, exist_ok=True)

    completed = 0
    skipped = 0

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, n_total)
        chunk_file = chunk_dir / f"chunk_{chunk_idx:04d}.parquet"

        # Resume: skip completed chunks
        if chunk_file.exists():
            skipped += 1
            continue

        print(f"\n{'='*60}")
        print(f"CHUNK {chunk_idx + 1}/{n_chunks} "
              f"(objects {start:,}–{end - 1:,}, {end - start:,} objects)")
        print(f"Completed so far: {completed + skipped}/{n_chunks} "
              f"({skipped} resumed)")
        print(f"{'='*60}")

        # Write temporary chunk CSV
        chunk_ids_df = all_ids_df.iloc[start:end]
        tmp_csv = chunk_dir / f"tmp_chunk_{chunk_idx:04d}.csv"
        chunk_ids_df.to_csv(tmp_csv, index=False)

        try:
            chunk_t0 = time.time()
            result_df = generate_features_rubin(
                objectid_file=str(tmp_csv),
                doGPU=doGPU,
                doCPU=doCPU,
                min_n_lc_points=min_n_lc_points,
                max_freq=max_freq,
                samples_per_peak=samples_per_peak,
                period_batch_size=period_batch_size,
                phase_bins=phase_bins,
                mag_bins=mag_bins,
                Ncore=Ncore,
                top_n_periods=top_n_periods,
                dirname=str(chunk_dir),
                filename=f"chunk_{chunk_idx:04d}",
                doNotSave=True,  # We save manually below
            )
            chunk_elapsed = time.time() - chunk_t0

            if result_df is not None and len(result_df) > 0:
                result_df.to_parquet(str(chunk_file))
                print(f"  Saved {len(result_df)} sources to {chunk_file} "
                      f"({chunk_elapsed:.0f}s)")
                completed += 1
            else:
                # Write empty marker so we don't re-process
                pd.DataFrame().to_parquet(str(chunk_file))
                print(f"  No sources passed filters in this chunk "
                      f"({chunk_elapsed:.0f}s)")
                completed += 1

        except Exception as e:
            print(f"  ERROR on chunk {chunk_idx}: {e}")
            import traceback
            traceback.print_exc()
            # Don't write output — this chunk will be retried on resume
            continue
        finally:
            # Clean up temp CSV
            if tmp_csv.exists():
                os.remove(tmp_csv)

    # --- Merge all chunks ---
    print(f"\n{'='*60}")
    print("MERGING CHUNKS")
    print(f"{'='*60}")

    chunk_files = sorted(chunk_dir.glob("chunk_*.parquet"))
    dfs = []
    for cf in chunk_files:
        df = pd.read_parquet(cf)
        if len(df) > 0:
            dfs.append(df)

    if len(dfs) > 0:
        merged = pd.concat(dfs, ignore_index=True)
        output_path = dirpath / f"{output_filename}.parquet"
        merged.to_parquet(str(output_path))
        elapsed = time.time() - t0
        print(f"\nMerged {len(merged):,} sources from {len(dfs)} chunks "
              f"into {output_path}")
        print(f"Total wall time: {elapsed / 3600:.1f} hours")
    else:
        print("No data to merge.")


def main():
    parser = argparse.ArgumentParser(
        description="Chunked Rubin feature generation with resume"
    )
    parser.add_argument("--objectid-file", required=True,
                        help="CSV with objectId column")
    parser.add_argument("--chunk-size", type=int, default=10000,
                        help="Objects per chunk (default 10000)")
    parser.add_argument("--dirname", default="generated_features_rubin",
                        help="Output directory")
    parser.add_argument("--doGPU", action="store_true", default=False)
    parser.add_argument("--doCPU", action="store_true", default=False)
    parser.add_argument("--min-n-lc-points", type=int, default=50)
    parser.add_argument("--max-freq", type=float, default=288.0)
    parser.add_argument("--samples-per-peak", type=int, default=10)
    parser.add_argument("--period-batch-size", type=int, default=1000)
    parser.add_argument("--phase-bins", type=int, default=20)
    parser.add_argument("--mag-bins", type=int, default=10)
    parser.add_argument("--Ncore", type=int, default=8)
    parser.add_argument("--top-n-periods", type=int, default=1,
                        help="Save top N periods per algorithm (default 1)")
    parser.add_argument("--chunk-subdir", default="chunks",
                        help="Subdirectory for chunk files (default 'chunks')")
    parser.add_argument("--output-filename", default="gen_features_rubin_full",
                        help="Merged output filename (without .parquet)")
    args = parser.parse_args()

    run_chunked(
        objectid_file=args.objectid_file,
        chunk_size=args.chunk_size,
        dirname=args.dirname,
        doGPU=args.doGPU,
        doCPU=args.doCPU,
        min_n_lc_points=args.min_n_lc_points,
        max_freq=args.max_freq,
        samples_per_peak=args.samples_per_peak,
        period_batch_size=args.period_batch_size,
        phase_bins=args.phase_bins,
        mag_bins=args.mag_bins,
        Ncore=args.Ncore,
        top_n_periods=args.top_n_periods,
        chunk_subdir=args.chunk_subdir,
        output_filename=args.output_filename,
    )


if __name__ == "__main__":
    main()
