"""
Field-dependent cadence alias detection and detrending for period finding.

DP1 observations have field-specific cadence patterns that create spectral
window function aliases — periods that dominate the periodogram for many sources
in a spatial region because of the observation sampling, not real variability.

This module provides:
  - assign_field(ra): map RA to a DP1 field name
  - compute_field_window_function(): compute W(f) from visit times
  - identify_alias_frequencies(): find alias peaks in W(f)
  - build_field_alias_map(): orchestrate per-field alias identification
  - detrend_periodogram(): divide LS periodogram by W(f) for top-N extraction
  - compute_neighbour_alias_fraction(): spatial consistency check
"""

import numpy as np

# ── DP1 field definitions ────────────────────────────────────────────────
# Each field is defined by an RA range (all at similar Dec ~ -28 to -44).
# These are well-separated so there is no overlap concern.

DP1_FIELDS = {
    'RA38': (36.0, 40.0),
    'RA53': (51.0, 55.0),
    'RA59': (57.0, 61.0),
    'RA95': (93.0, 97.0),
}


def assign_field(ra):
    """Map a right ascension value to a DP1 field name.

    Parameters
    ----------
    ra : float
        Right ascension in degrees.

    Returns
    -------
    str or None
        Field name (e.g. 'RA95') or None if not in any defined field.
    """
    if ra is None or (isinstance(ra, float) and np.isnan(ra)):
        return None
    for name, (lo, hi) in DP1_FIELDS.items():
        if lo <= ra <= hi:
            return name
    return None


# ── Spectral window function ─────────────────────────────────────────────


def compute_field_window_function(visit_times, freqs):
    """Compute the spectral window function W(f) for a set of observation times.

    The window function is the Lomb-Scargle periodogram of a *constant* signal
    sampled at the given observation times.  Peaks in W(f) reveal frequencies
    where the cadence itself creates power, independent of any real variability.

    Uses astropy's LombScargle for correctness (this is a one-time computation
    per field, not performance-critical).

    Parameters
    ----------
    visit_times : array-like
        Observation MJD times (need not be sorted).
    freqs : array-like
        Frequency grid in cycles/day.

    Returns
    -------
    window_power : ndarray
        Power spectrum of the window function at each frequency.
        Normalised so that the maximum is 1.0.
    """
    from astropy.timeseries import LombScargle

    times = np.asarray(visit_times, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)

    # Constant signal -> window function
    y = np.ones_like(times)
    ls = LombScargle(times, y, fit_mean=False, center_data=False)
    power = ls.power(freqs)

    # Normalise to [0, 1]
    pmax = np.max(power)
    if pmax > 0:
        power = power / pmax

    return power


def identify_alias_frequencies(window_power, freqs, threshold=0.5, expand_bins=3):
    """Identify frequency ranges where the window function exceeds threshold.

    Parameters
    ----------
    window_power : ndarray
        Window function power (normalised to max=1).
    freqs : ndarray
        Frequency grid matching window_power.
    threshold : float
        Fraction of peak power above which a frequency is flagged as an alias.
        Default 0.5 means peaks must exceed 50% of the strongest window peak.
    expand_bins : int
        Number of bins to expand each alias zone on each side, to account
        for finite frequency resolution.

    Returns
    -------
    alias_zones : list of [float, float]
        List of [freq_lo, freq_hi] pairs in the same format as the existing
        ``freqs_to_remove`` in ``generate_features_rubin.py``.
    """
    mask = window_power >= threshold
    if not np.any(mask):
        return []

    # Find contiguous runs of above-threshold bins
    indices = np.where(mask)[0]
    zones = []
    run_start = indices[0]
    prev = indices[0]

    for idx in indices[1:]:
        if idx != prev + 1:
            # End of a contiguous run
            lo = max(0, run_start - expand_bins)
            hi = min(len(freqs) - 1, prev + expand_bins)
            zones.append([float(freqs[lo]), float(freqs[hi])])
            run_start = idx
        prev = idx

    # Close the last run
    lo = max(0, run_start - expand_bins)
    hi = min(len(freqs) - 1, prev + expand_bins)
    zones.append([float(freqs[lo]), float(freqs[hi])])

    return zones


def build_field_alias_map(
    visit_df, freqs, threshold=0.5, expand_bins=3, ra_column='fieldRA'
):
    """Compute per-field cadence alias zones from the Visit table.

    Parameters
    ----------
    visit_df : pandas.DataFrame
        Visit table with at least 'expMidptMJD' and a field RA column.
        If *ra_column* is not present, all visits are treated as one group.
    freqs : ndarray
        Frequency grid in cycles/day.
    threshold : float
        Window function threshold for alias identification.
    expand_bins : int
        Number of bins to expand alias zones.
    ra_column : str
        Column name containing the field RA (used to assign visits to fields).
        If the column doesn't exist, visits are grouped using the median RA
        of each visit's pointing if 'fieldRA' is available, otherwise all
        visits are used as a single group.

    Returns
    -------
    field_alias_map : dict
        ``{field_name: [[freq_lo, freq_hi], ...]}`` mapping from field name
        to alias frequency zones.
    field_window : dict
        ``{field_name: ndarray}`` mapping from field name to window function.
    """
    field_alias_map = {}
    field_window = {}

    # Try to group visits by field
    if ra_column in visit_df.columns:
        # Assign each visit to a field based on RA
        visit_df = visit_df.copy()
        visit_df['_dp1_field'] = visit_df[ra_column].apply(assign_field)

        for field_name in DP1_FIELDS:
            field_visits = visit_df[visit_df['_dp1_field'] == field_name]
            if len(field_visits) < 10:
                continue
            times = field_visits['expMidptMJD'].values
            wp = compute_field_window_function(times, freqs)
            field_window[field_name] = wp
            field_alias_map[field_name] = identify_alias_frequencies(
                wp,
                freqs,
                threshold=threshold,
                expand_bins=expand_bins,
            )
    else:
        # No field RA info — treat all visits as one group
        times = visit_df['expMidptMJD'].values
        if len(times) >= 10:
            wp = compute_field_window_function(times, freqs)
            field_window['ALL'] = wp
            field_alias_map['ALL'] = identify_alias_frequencies(
                wp,
                freqs,
                threshold=threshold,
                expand_bins=expand_bins,
            )

    return field_alias_map, field_window


def merge_alias_zones(field_alias_map):
    """Merge all per-field alias zones into a single list.

    Since the 4 DP1 fields are well-separated in RA, cross-contamination is
    minimal — a cadence alias from RA95 won't create false power in RA53
    sources because those sources have different observation times.  Merging
    globally is safe and avoids having to sub-batch the period search by field.

    Parameters
    ----------
    field_alias_map : dict
        ``{field_name: [[freq_lo, freq_hi], ...]}``

    Returns
    -------
    merged : list of [float, float]
        Union of all alias zones.
    """
    merged = []
    for zones in field_alias_map.values():
        merged.extend(zones)
    return merged


# ── Periodogram detrending ───────────────────────────────────────────────


def detrend_periodogram(periodogram_data, window_power, floor=0.01):
    """Divide a periodogram by the window function to suppress cadence aliases.

    Used for LS periodogram top-N extraction: divides each source's raw
    periodogram by W(f) so that alias peaks are suppressed before selecting
    the top-N peaks.

    Parameters
    ----------
    periodogram_data : ndarray
        Raw periodogram power array (1D, same length as window_power).
    window_power : ndarray
        Window function power at the same frequencies, normalised to max=1.
    floor : float
        Minimum window power value to prevent division by very small numbers.
        Default 0.01.

    Returns
    -------
    detrended : ndarray
        Detrended periodogram (same shape as input).
    """
    wp = np.maximum(window_power, floor)
    return periodogram_data / wp


def detrend_periodogram_results(periodogram_results, window_power, floor=0.01):
    """Apply detrending to a list of periodogram result dicts in-place.

    Each dict in the list has 'period' and 'data' keys (as returned by
    ``find_periods`` with a ``_periodogram`` suffix).  This modifies the
    'data' array and recomputes 'period' based on the detrended peak.

    Parameters
    ----------
    periodogram_results : list of dict
        Periodogram results from ``find_periods``.
    window_power : ndarray
        Window function power array.
    floor : float
        Minimum window power for detrending.

    Returns
    -------
    periodogram_results : list of dict
        Same list, modified in place for convenience.
    """
    for res in periodogram_results:
        data = res['data'].flatten()
        detrended = detrend_periodogram(data, window_power[: len(data)], floor)
        res['data'] = detrended.reshape(res['data'].shape)

    return periodogram_results


# ── Spatial neighbour alias fraction ─────────────────────────────────────


def compute_neighbour_alias_fraction(
    ra, dec, periods, radius_arcmin=2.0, tolerance=0.05
):
    """Compute fraction of spatial neighbours sharing the same period.

    For each source, counts the fraction of neighbours within *radius_arcmin*
    that have a period matching within *tolerance* fractional difference.
    High fractions indicate the period is likely a cadence alias rather than
    real variability.

    Parameters
    ----------
    ra : array-like
        Right ascension in degrees, shape (N,).
    dec : array-like
        Declination in degrees, shape (N,).
    periods : array-like
        Best-fit periods in days, shape (N,).
    radius_arcmin : float
        Search radius in arcminutes.
    tolerance : float
        Fractional period match tolerance.

    Returns
    -------
    fractions : ndarray, shape (N,)
        Fraction of neighbours sharing the same period.
        0.0 if the source has no neighbours within radius.
    """
    from scipy.spatial import cKDTree

    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    periods = np.asarray(periods, dtype=np.float64)
    n = len(ra)

    if n == 0:
        return np.array([], dtype=np.float64)

    # Convert to Cartesian for cKDTree (approximate, fine for small scales)
    cos_dec = np.cos(np.deg2rad(dec))
    x = ra * cos_dec
    y = dec
    coords = np.column_stack([x, y])

    # radius in degrees (arcmin -> deg)
    radius_deg = radius_arcmin / 60.0

    tree = cKDTree(coords)
    fractions = np.zeros(n, dtype=np.float64)

    for i in range(n):
        p_i = periods[i]
        if np.isnan(p_i) or p_i <= 0:
            fractions[i] = 0.0
            continue

        neighbours = tree.query_ball_point(coords[i], radius_deg)
        # Remove self
        neighbours = [j for j in neighbours if j != i]
        if len(neighbours) == 0:
            fractions[i] = 0.0
            continue

        n_match = 0
        for j in neighbours:
            p_j = periods[j]
            if np.isnan(p_j) or p_j <= 0:
                continue
            if abs(p_i / p_j - 1.0) < tolerance:
                n_match += 1

        fractions[i] = n_match / len(neighbours)

    return fractions
