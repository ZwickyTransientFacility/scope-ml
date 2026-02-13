#!/usr/bin/env python
"""
Merge chunk parquet files from parallel feature generation into a single training set.

Usage:
    python tools/merge_training_chunks.py \
        --chunks-dir regenerate_training \
        --output regenerate_training/new_training_set.parquet \
        --n-chunks 8
"""
import argparse
import pathlib
import sys

import pandas as pd


def merge_chunks(chunks_dir, output_path, n_chunks=None, chunk_pattern='chunk_*.parquet'):
    chunks_dir = pathlib.Path(chunks_dir)

    if n_chunks is not None:
        # Look for specific chunk files
        chunk_files = []
        missing = []
        for i in range(n_chunks):
            p = chunks_dir / f'chunk_{i}.parquet'
            if p.exists():
                chunk_files.append(p)
            else:
                missing.append(i)
        if missing:
            print(f"WARNING: Missing chunks: {missing}")
            print("Merging available chunks only.")
    else:
        chunk_files = sorted(chunks_dir.glob(chunk_pattern))

    if not chunk_files:
        print(f"ERROR: No chunk files found in {chunks_dir}")
        sys.exit(1)

    print(f"Found {len(chunk_files)} chunk files:")
    dfs = []
    for f in chunk_files:
        df = pd.read_parquet(str(f))
        print(f"  {f.name}: {len(df)} sources, {len(df.columns)} columns")
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)

    # Check for duplicates
    id_col = 'ztf_id' if 'ztf_id' in merged.columns else '_id'
    n_dupes = merged[id_col].duplicated().sum()
    if n_dupes > 0:
        print(f"WARNING: {n_dupes} duplicate IDs found — removing duplicates (keeping first)")
        merged = merged.drop_duplicates(subset=id_col, keep='first')

    merged.to_parquet(str(output_path), index=False)
    print(f"\nMerged training set: {len(merged)} sources, {len(merged.columns)} columns")
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge chunk parquets into final training set")
    parser.add_argument('--chunks-dir', type=str, default='regenerate_training',
                        help='Directory containing chunk_*.parquet files')
    parser.add_argument('--output', type=str, default='regenerate_training/new_training_set.parquet',
                        help='Output path for merged parquet')
    parser.add_argument('--n-chunks', type=int, default=None,
                        help='Expected number of chunks (checks for missing)')
    args = parser.parse_args()
    merge_chunks(args.chunks_dir, args.output, args.n_chunks)


if __name__ == '__main__':
    main()
