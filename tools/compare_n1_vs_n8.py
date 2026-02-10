#!/usr/bin/env python
"""
Compare N=1 (single best period) vs N=8 (top-8 periods) feature generation.

Reads the two parquet files and produces diagnostic tables + optional plots
showing how cross-algorithm agreement improves with multiple period peaks.

Usage:
    python tools/compare_n1_vs_n8.py \
        --n1 generated_features_rubin/gen_features_rubin_n1.parquet \
        --n8 generated_features_rubin/gen_features_rubin_top8.parquet \
        [--plots]
"""

import argparse
import sys
import numpy as np
import pandas as pd


# ── helpers ──────────────────────────────────────────────────────────────────

ALGOS = ["ELS", "ECE", "EAOV", "EFPW"]
HARMONIC_RATIOS = [1.0, 0.5, 2.0]
TOLERANCE = 0.05  # 5% fractional tolerance


def _period_match(pa, pb, tol=TOLERANCE, harmonics=HARMONIC_RATIOS):
    """Check whether two periods match within tolerance, allowing harmonics."""
    if np.isnan(pa) or np.isnan(pb) or pa <= 0 or pb <= 0:
        return False
    for h in harmonics:
        if abs(pa / (pb * h) - 1.0) < tol:
            return True
    return False


def n1_agreement(row):
    """Count pairwise agreement using only the single best period per algo."""
    periods = {}
    for algo in ALGOS:
        col = f"period_{algo}"
        if col in row.index and not np.isnan(row[col]):
            periods[algo] = row[col]

    n_agree = 0
    n_total = 0
    keys = list(periods.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            n_total += 1
            if _period_match(periods[keys[i]], periods[keys[j]]):
                n_agree += 1
    return n_agree, n_total


def spurious_fraction(series, threshold=0.007):
    """Fraction of periods below the spurious threshold."""
    valid = series.dropna()
    if len(valid) == 0:
        return np.nan
    return (valid < threshold).sum() / len(valid)


# ── main comparison ─────────────────────────────────────────────────────────


def compare(n1_path, n8_path, do_plots=False):
    print("Loading N=1 features...")
    df1 = pd.read_parquet(n1_path)
    print(f"  {len(df1):,} sources, {len(df1.columns)} columns")

    print("Loading N=8 features...")
    df8 = pd.read_parquet(n8_path)
    print(f"  {len(df8):,} sources, {len(df8.columns)} columns")

    # ── 1. Basic column comparison ───────────────────────────────────────
    extra_cols = set(df8.columns) - set(df1.columns)
    print(f"\nNew columns in N=8 ({len(extra_cols)}):")
    agree_cols = sorted(c for c in extra_cols if "agree" in c)
    topn_cols = sorted(c for c in extra_cols if c.startswith("period_") or c.startswith("significance_"))
    print(f"  Agreement: {agree_cols}")
    print(f"  Top-N period/sig columns: {len(topn_cols)}")

    # ── 2. N=1 pairwise agreement (baseline) ────────────────────────────
    print("\n" + "=" * 60)
    print("N=1 CROSS-ALGORITHM AGREEMENT (single best period)")
    print("=" * 60)

    # Merge on _id to compare the same sources
    common_ids = set(df1["_id"]) & set(df8["_id"])
    print(f"Common sources: {len(common_ids):,}")

    df1c = df1[df1["_id"].isin(common_ids)].set_index("_id").sort_index()
    df8c = df8[df8["_id"].isin(common_ids)].set_index("_id").sort_index()

    # Compute N=1 agreement from scratch (for the common sources)
    n1_pairs = df1c.apply(n1_agreement, axis=1)
    df1c["n1_agree"] = [x[0] for x in n1_pairs]
    df1c["n1_total"] = [x[1] for x in n1_pairs]
    df1c["n1_agree_score"] = df1c["n1_agree"] / df1c["n1_total"].replace(0, np.nan)

    print(f"\nN=1 agreement score distribution:")
    print(df1c["n1_agree_score"].describe().to_string())
    n1_any = (df1c["n1_agree"] > 0).sum()
    n1_all = (df1c["n1_agree"] == df1c["n1_total"]).sum()
    print(f"\n  Any agreement (>=1 pair): {n1_any:,} ({100*n1_any/len(df1c):.1f}%)")
    print(f"  Full agreement (6/6 pairs): {n1_all:,} ({100*n1_all/len(df1c):.1f}%)")

    # ── 3. N=8 agreement scores ──────────────────────────────────────────
    if "agree_score" in df8c.columns:
        print("\n" + "=" * 60)
        print("N=8 CROSS-ALGORITHM AGREEMENT (top-8 periods with harmonics)")
        print("=" * 60)
        print(f"\nN=8 agreement score distribution:")
        print(df8c["agree_score"].describe().to_string())

        n8_any = (df8c["n_agree_pairs"] > 0).sum()
        n8_all = (df8c["agree_score"] == 1.0).sum()
        print(f"\n  Any agreement (>=1 pair): {n8_any:,} ({100*n8_any/len(df8c):.1f}%)")
        print(f"  Full agreement (6/6 pairs): {n8_all:,} ({100*n8_all/len(df8c):.1f}%)")

        # ── 4. Improvement from N=1 to N=8 ──────────────────────────────
        print("\n" + "=" * 60)
        print("IMPROVEMENT: N=1 → N=8")
        print("=" * 60)

        improved = (df8c["agree_score"] > df1c["n1_agree_score"]).sum()
        same = (df8c["agree_score"] == df1c["n1_agree_score"]).sum()
        worse = (df8c["agree_score"] < df1c["n1_agree_score"]).sum()
        print(f"  Improved: {improved:,} ({100*improved/len(df1c):.1f}%)")
        print(f"  Same:     {same:,} ({100*same/len(df1c):.1f}%)")
        print(f"  Worse:    {worse:,} ({100*worse/len(df1c):.1f}%)")

        delta = df8c["agree_score"] - df1c["n1_agree_score"]
        print(f"\n  Mean improvement: {delta.mean():.4f}")
        print(f"  Median improvement: {delta.median():.4f}")

        # Sources that went from 0 agreement to >0
        gained = ((df1c["n1_agree"] == 0) & (df8c["n_agree_pairs"] > 0)).sum()
        n1_zero = (df1c["n1_agree"] == 0).sum()
        print(f"\n  Sources with N=1 agree=0: {n1_zero:,}")
        print(f"  Of those, gained agreement with N=8: {gained:,} ({100*gained/max(n1_zero,1):.1f}%)")

    # ── 5. Period quality comparison ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("PERIOD QUALITY: SPURIOUS FRACTION (<0.007d)")
    print("=" * 60)

    for algo in ALGOS:
        col1 = f"period_{algo}"
        col8_best = f"period_1_{algo.lstrip('E')}" if algo.startswith("E") else f"period_1_{algo}"
        # Also check the agreed-upon period
        frac1 = spurious_fraction(df1c[col1]) if col1 in df1c.columns else np.nan
        print(f"  {algo:6s} N=1 best: {100*frac1:.1f}% spurious", end="")
        if col8_best in df8c.columns:
            frac8 = spurious_fraction(df8c[col8_best])
            print(f"  |  N=8 rank-1: {100*frac8:.1f}% spurious")
        else:
            print()

    if "best_agree_period" in df8c.columns:
        agree_p = df8c["best_agree_period"].dropna()
        if len(agree_p) > 0:
            frac_agree = (agree_p < 0.007).sum() / len(agree_p)
            print(f"\n  Best-agree period: {100*frac_agree:.1f}% spurious "
                  f"({len(agree_p):,} sources with agreement)")

    # ── 6. Best candidates from N=8 ─────────────────────────────────────
    if "agree_score" in df8c.columns and "best_agree_period" in df8c.columns:
        print("\n" + "=" * 60)
        print("TOP CANDIDATES (N=8 agreement)")
        print("=" * 60)

        # Filter: agree_score >= 0.5, best_agree_period > 0.04d, significance > 10
        candidates = df8c[
            (df8c["agree_score"] >= 0.5)
            & (df8c["best_agree_period"] > 0.04)
            & (df8c["best_agree_period"] < 10.0)
        ].copy()

        # Add max significance across algos
        sig_cols = [f"significance_{a}" for a in ALGOS if f"significance_{a}" in df8c.columns]
        if sig_cols:
            candidates["max_sig"] = candidates[sig_cols].max(axis=1)
            candidates = candidates.sort_values("max_sig", ascending=False)

        print(f"\n  Candidates with agree>=0.5, 0.04d < P < 10d: {len(candidates):,}")

        if len(candidates) > 0:
            print(f"\n  Top 20 by significance:")
            show_cols = ["best_agree_period", "agree_score", "n_agree_pairs"]
            if "max_sig" in candidates.columns:
                show_cols.append("max_sig")
            show_cols.extend(sig_cols[:2])
            available = [c for c in show_cols if c in candidates.columns]
            top20 = candidates.head(20)[available]
            print(top20.to_string())

    # ── 7. Period distribution by rank (N=8) ─────────────────────────────
    if any(f"period_1_{a.lstrip('E')}" in df8c.columns for a in ALGOS):
        print("\n" + "=" * 60)
        print("PERIOD DISTRIBUTION BY RANK (N=8, ELS algorithm)")
        print("=" * 60)

        algo_short = "LS"
        for rank in range(1, 9):
            col = f"period_{rank}_{algo_short}"
            if col in df8c.columns:
                vals = df8c[col].dropna()
                spur = (vals < 0.007).sum() / len(vals) * 100 if len(vals) > 0 else 0
                med = vals.median()
                reasonable = ((vals > 0.04) & (vals < 10)).sum()
                print(f"  Rank {rank}: median={med:.5f}d, "
                      f"spurious(<0.007d)={spur:.0f}%, "
                      f"reasonable(0.04-10d)={reasonable:,}")

    # ── plots ────────────────────────────────────────────────────────────
    if do_plots and "agree_score" in df8c.columns:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Panel 1: Agreement score histograms
            ax = axes[0, 0]
            bins = np.linspace(0, 1, 21)
            ax.hist(df1c["n1_agree_score"].dropna(), bins=bins, alpha=0.6, label="N=1", density=True)
            ax.hist(df8c["agree_score"].dropna(), bins=bins, alpha=0.6, label="N=8", density=True)
            ax.set_xlabel("Agreement Score")
            ax.set_ylabel("Density")
            ax.set_title("Cross-Algorithm Agreement: N=1 vs N=8")
            ax.legend()

            # Panel 2: Best-agree period distribution
            ax = axes[0, 1]
            agree_p = df8c["best_agree_period"].dropna()
            agree_p_log = np.log10(agree_p[agree_p > 0])
            ax.hist(agree_p_log, bins=50, color="steelblue", alpha=0.7)
            ax.axvline(np.log10(0.007), color="red", ls="--", label="Spurious edge (0.007d)")
            ax.axvline(np.log10(0.04), color="green", ls="--", label="Reasonable start (0.04d)")
            ax.set_xlabel("log10(Best Agree Period [days])")
            ax.set_ylabel("Count")
            ax.set_title("Period Distribution (N=8 agreed period)")
            ax.legend(fontsize=8)

            # Panel 3: Improvement scatter
            ax = axes[1, 0]
            ax.scatter(
                df1c["n1_agree_score"].values,
                df8c["agree_score"].values,
                s=1, alpha=0.1, rasterized=True,
            )
            ax.plot([0, 1], [0, 1], "r--", alpha=0.5)
            ax.set_xlabel("N=1 Agreement Score")
            ax.set_ylabel("N=8 Agreement Score")
            ax.set_title("Agreement Improvement")

            # Panel 4: Significance vs agreement (N=8)
            ax = axes[1, 1]
            sig_cols = [f"significance_{a}" for a in ALGOS if f"significance_{a}" in df8c.columns]
            if sig_cols:
                max_sig = df8c[sig_cols].max(axis=1)
                ax.scatter(df8c["agree_score"].values, max_sig.values,
                           s=1, alpha=0.1, rasterized=True)
                ax.set_xlabel("N=8 Agreement Score")
                ax.set_ylabel("Max Significance (any algo)")
                ax.set_title("Significance vs Agreement")

            plt.tight_layout()
            out_path = str(pd.io.common.stringify_path(
                "generated_features_rubin/compare_n1_vs_n8.png"
            ))
            plt.savefig(out_path, dpi=150)
            print(f"\nSaved comparison plot to {out_path}")
            plt.close()
        except ImportError:
            print("\nmatplotlib not available — skipping plots.")

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="Compare N=1 vs N=8 period features")
    parser.add_argument("--n1", required=True, help="N=1 parquet path")
    parser.add_argument("--n8", required=True, help="N=8 parquet path")
    parser.add_argument("--plots", action="store_true", help="Generate comparison plots")
    args = parser.parse_args()
    compare(args.n1, args.n8, do_plots=args.plots)


if __name__ == "__main__":
    main()
