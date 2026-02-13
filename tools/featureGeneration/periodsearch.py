import numpy as np
import warnings

import periodfind

# ---------------------------------------------------------------------------
# Algorithm name mapping
# ---------------------------------------------------------------------------

_ALGO_MAP = {"CE", "AOV", "LS", "FPW", "BLS"}


def _normalize_algorithm(algorithm):
    """Parse algorithm string into (base_name, is_periodogram).

    Handles names like 'ECE', 'CE', 'EAOV_periodogram', 'LS', etc.
    Returns the canonical base name and whether periodogram output is requested.
    """
    algo_name = algorithm
    is_periodogram = "_periodogram" in algo_name
    algo_name = algo_name.replace("_periodogram", "")

    # Strip legacy "E" prefix (ECE -> CE, EAOV -> AOV, ELS -> LS)
    if algo_name.startswith("E") and algo_name[1:] in _ALGO_MAP:
        algo_name = algo_name[1:]

    return algo_name, is_periodogram


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------


def _prepare_lightcurves(lightcurves, doSingleTimeSegment, return_errs=False):
    """Sort by time, subtract tmin, normalize mags, and cast to float32.

    Parameters
    ----------
    lightcurves : list of tuple
        Each element is (times, mags, magerrs).
    doSingleTimeSegment : bool
        If True, build a common time grid across all lightcurves.
    return_errs : bool, default=False
        If True, also return the magnitude error arrays (for FPW).

    Returns
    -------
    time_stack : list of ndarray (float32)
    mag_stack : list of ndarray (float32)
    err_stack : list of ndarray (float32), only if return_errs=True
    """
    if doSingleTimeSegment:
        tt = np.empty((0, 1))
        for lightcurve in lightcurves:
            tt = np.unique(np.append(tt, lightcurve[0]))

    time_stack, mag_stack, err_stack = [], [], []
    for lightcurve in lightcurves:
        if doSingleTimeSegment:
            _, x_ind, y_ind = np.intersect1d(tt, lightcurve[0], return_indices=True)
            mag_array = 999 * np.ones(tt.shape)
            magerr_array = 999 * np.ones(tt.shape)
            mag_array[x_ind] = lightcurve[1][y_ind]
            magerr_array[x_ind] = lightcurve[2][y_ind]
            lightcurve = (tt, mag_array, magerr_array)
        else:
            idx = np.argsort(lightcurve[0])
            tmin = np.min(lightcurve[0])
            lightcurve = (
                lightcurve[0][idx] - tmin,
                lightcurve[1][idx],
                lightcurve[2][idx],
            )

        time_stack.append(np.asarray(lightcurve[0], dtype=np.float32))
        lc = np.asarray(lightcurve[1], dtype=np.float64)
        lc = (lc - np.min(lc)) / (np.max(lc) - np.min(lc))
        mag_stack.append(np.asarray(lc, dtype=np.float32))
        if return_errs:
            err_stack.append(np.asarray(lightcurve[2], dtype=np.float32))

    if return_errs:
        return time_stack, mag_stack, err_stack
    return time_stack, mag_stack


def _build_pdots(doUsePDot):
    """Build the array of period time-derivatives to test.

    Parameters
    ----------
    doUsePDot : bool
        If True, return a logarithmic grid of negative pdots plus zero.
        If False, return [0.0].

    Returns
    -------
    pdots_to_test : ndarray (float32)
    """
    if doUsePDot:
        num_pdots = 10
        max_pdot = 1e-10
        min_pdot = 1e-12
        pdots_to_test = -np.logspace(np.log10(min_pdot), np.log10(max_pdot), num_pdots)
        pdots_to_test = np.append(0, pdots_to_test)
    else:
        pdots_to_test = np.array([0.0])
    return pdots_to_test.astype(np.float32)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def find_periods(
    algorithm,
    lightcurves,
    freqs,
    doGPU=False,
    doCPU=False,
    doRemoveTerrestrial=False,
    doUsePDot=False,
    doSingleTimeSegment=False,
    freqs_to_remove=None,
    phase_bins=20,
    mag_bins=10,
    Ncore=8,
    qmin=0.01,
    qmax=0.5,
):

    if freqs_to_remove is not None:
        for pair in freqs_to_remove:
            idx = np.where((freqs < pair[0]) | (freqs > pair[1]))[0]
            freqs = freqs[idx]

    periods_best, significances = [], []
    pdots = np.zeros((len(lightcurves),))
    print('Period finding lightcurves...')

    # Normalize algorithm name
    algo_name, is_periodogram = _normalize_algorithm(algorithm)

    # -----------------------------------------------------------------------
    # Unified periodfind path for CE / AOV / LS (works on both CPU and GPU)
    # -----------------------------------------------------------------------
    if algo_name in _ALGO_MAP:
        device = 'gpu' if doGPU else 'cpu'
        needs_errs = algo_name in ("FPW", "BLS")

        # Create algorithm via unified factory
        if algo_name == "CE":
            algo = periodfind.ConditionalEntropy(
                n_phase=phase_bins, n_mag=mag_bins, device=device
            )
        elif algo_name == "AOV":
            algo = periodfind.AOV(n_phase=phase_bins, device=device)
        elif algo_name == "LS":
            algo = periodfind.LombScargle(device=device)
        elif algo_name == "FPW":
            algo = periodfind.FPW(n_bins=phase_bins, device=device)
        elif algo_name == "BLS":
            algo = periodfind.BoxLeastSquares(
                n_bins=50, qmin=qmin, qmax=qmax, device=device
            )

        # Prepare data
        if needs_errs:
            time_stack, mag_stack, err_stack = _prepare_lightcurves(
                lightcurves, doSingleTimeSegment, return_errs=True
            )
        else:
            time_stack, mag_stack = _prepare_lightcurves(
                lightcurves, doSingleTimeSegment
            )
        periods = (1.0 / freqs).astype(np.float32)
        pdots_to_test = _build_pdots(doUsePDot)

        print("Number of lightcurves: %d" % len(time_stack))
        print("Number of frequency bins: %d" % len(freqs))
        print("Number of phase bins: %d" % phase_bins)
        print("Number of magnitude bins: %d" % mag_bins)

        output_mode = 'periodogram' if is_periodogram else 'stats'
        extra_kwargs = {"errs": err_stack} if needs_errs else {}
        data_out = algo.calc(
            time_stack,
            mag_stack,
            periods,
            pdots_to_test,
            output=output_mode,
            **extra_kwargs,
        )

        # Process results
        if not is_periodogram:
            # Stats output -> flat arrays
            periods_best = np.zeros((len(lightcurves), 1))
            significances = np.zeros((len(lightcurves), 1))
            pdots = np.zeros((len(lightcurves), 1))

            for ii, stat in enumerate(data_out):
                if np.isnan(stat.significance):
                    warnings.warn(
                        "Oops... significance  is nan... something went wrong"
                    )
                periods_best[ii] = stat.params[0]
                pdots[ii] = stat.params[1]
                significances[ii] = stat.significance

            pdots = pdots.flatten()
            periods_best = periods_best.flatten()
            significances = significances.flatten()
        else:
            # Periodogram output -> list of dicts
            periods_best = []
            significances = np.zeros((len(lightcurves), 1))
            pdots = np.zeros((len(lightcurves), 1))

            for ii, stat in enumerate(data_out):
                if algo_name == "CE":
                    significance = np.abs(
                        np.mean(stat.data) - np.min(stat.data)
                    ) / np.std(stat.data)
                    period = periods[np.argmin(stat.data)]
                else:
                    # AOV and LS use maxima
                    significance = np.abs(
                        np.mean(stat.data) - np.max(stat.data)
                    ) / np.std(stat.data)
                    period = periods[np.argmax(stat.data)]

                if np.isnan(significance):
                    warnings.warn(
                        "Oops... significance  is nan... something went wrong"
                    )

                periods_best.append({'period': period, 'data': stat.data})
                pdots[ii] = pdots_to_test[0]
                significances[ii] = significance

            pdots = pdots.flatten()
            significances = significances.flatten()

    return np.array(periods_best), np.array(significances), np.array(pdots)


def extract_top_n_periods(periodogram_results, freqs, n_top=8, n_chunks_multiplier=3):
    """Extract top N periods from periodogram using hybrid chunking.

    The frequency grid is divided into ``n_chunks_multiplier * n_top``
    equal-width chunks.  The single best peak from each chunk is recorded,
    then all chunk-winners are ranked by significance and the top ``n_top``
    are returned.  This guarantees that the returned periods come from
    distinct regions of frequency space, avoiding the adjacent-bin
    duplication problem of the previous sort-and-take approach.

    Parameters
    ----------
    periodogram_results : list of dict
        Output from ``find_periods`` when called with a ``_periodogram``
        algorithm suffix.  Each dict has ``'period'`` (best float) and
        ``'data'`` (full periodogram array).
    freqs : ndarray
        The frequency grid used for period finding.
    n_top : int
        Number of top periods to return per source.
    n_chunks_multiplier : int
        The periodogram is split into ``n_chunks_multiplier * n_top``
        chunks.  Higher values give finer coverage but may split real
        peaks across chunk boundaries.  Default 3.

    Returns
    -------
    top_periods : ndarray, shape (n_sources, n_top)
        Periods in descending order of significance.
    top_significances : ndarray, shape (n_sources, n_top)
        Corresponding significance values.
    """
    periods_grid = 1.0 / freqs
    n_sources = len(periodogram_results)
    top_periods = np.full((n_sources, n_top), np.nan, dtype=np.float64)
    top_significances = np.full((n_sources, n_top), np.nan, dtype=np.float64)

    n_chunks = n_chunks_multiplier * n_top

    for i, res in enumerate(periodogram_results):
        data = res['data'].flatten()
        n_freqs = len(data)
        if n_freqs == 0:
            continue

        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            continue

        # CE uses minima; others use maxima
        best_idx_min = np.argmin(data)
        best_period = res['period']
        use_minima = np.isclose(periods_grid[best_idx_min], best_period, rtol=1e-4)

        chunk_size = n_freqs // n_chunks
        if chunk_size < 1:
            # Fewer frequency bins than chunks — fall back to simple sort
            if use_minima:
                sorted_indices = np.argsort(data)
            else:
                sorted_indices = np.argsort(data)[::-1]
            sigs = np.abs(data[sorted_indices] - mean_val) / std_val
            n_fill = min(n_top, len(sorted_indices))
            top_periods[i, :n_fill] = periods_grid[sorted_indices[:n_fill]]
            top_significances[i, :n_fill] = sigs[:n_fill]
            continue

        # Find best peak in each chunk
        chunk_periods = np.empty(n_chunks, dtype=np.float64)
        chunk_sigs = np.empty(n_chunks, dtype=np.float64)

        for c in range(n_chunks):
            lo = c * chunk_size
            hi = lo + chunk_size if c < n_chunks - 1 else n_freqs
            chunk_data = data[lo:hi]

            if use_minima:
                best_local = np.argmin(chunk_data)
            else:
                best_local = np.argmax(chunk_data)

            best_global = lo + best_local
            chunk_periods[c] = periods_grid[best_global]
            chunk_sigs[c] = abs(data[best_global] - mean_val) / std_val

        # Rank by significance and keep top n_top
        order = np.argsort(chunk_sigs)[::-1]
        n_fill = min(n_top, len(order))
        top_periods[i, :n_fill] = chunk_periods[order[:n_fill]]
        top_significances[i, :n_fill] = chunk_sigs[order[:n_fill]]

    return top_periods, top_significances


def _period_match(pa_val, pb_val, tolerance=0.05, harmonics=None):
    """Check if two periods match within tolerance, allowing harmonics.

    Parameters
    ----------
    pa_val, pb_val : float
        Period values to compare.
    tolerance : float
        Fractional tolerance for matching (default 0.05 = 5%).
    harmonics : list of float or None
        Harmonic ratios to check.  ``None`` uses [1, 0.5, 2, 1/3, 3].

    Returns
    -------
    bool
    """
    if harmonics is None:
        harmonics = [1.0, 0.5, 2.0, 1.0 / 3, 3.0]
    if np.isnan(pa_val) or np.isnan(pb_val) or pa_val <= 0 or pb_val <= 0:
        return False
    for h in harmonics:
        if abs(pa_val / (pb_val * h) - 1.0) < tolerance:
            return True
    return False


def compute_agreement_scores(
    period_dict,
    top_n_periods_dict,
    keep_id_list,
    period_algorithms,
    min_agree_period=0.007,
    harmonics=None,
    tolerance=0.05,
):
    """Compute cross-algorithm agreement scores for period finding.

    Multi-tier scoring with spurious-period filtering and expanded
    harmonics.  This was originally inlined in
    ``generate_features_rubin.py`` and is now shared across pipelines.

    Parameters
    ----------
    period_dict : dict
        ``{algorithm_key: 1-D array}`` of best periods per source,
        indexed in the same order as *keep_id_list*.
    top_n_periods_dict : dict
        ``{algorithm_key: 2-D array (n_sources, n_top)}`` of ranked
        periods per source.
    keep_id_list : list
        Source identifiers, one per row in the arrays above.
    period_algorithms : list of str
        Algorithm keys present in *period_dict* / *top_n_periods_dict*.
        Names like ``"ELS_ECE_EAOV"`` or ``"LS_CE_AOV"`` are kept
        verbatim; all others are shortened with ``algo.split('_')[0]``.
    min_agree_period : float
        Periods shorter than this (days) are treated as spurious and
        excluded from agreement checks.
    harmonics : list of float or None
        Harmonic ratios passed to :func:`_period_match`.
    tolerance : float
        Fractional tolerance passed to :func:`_period_match`.

    Returns
    -------
    dict
        ``{source_id: {feature_name: value, ...}}`` with keys:
        ``n_agree_pairs``, ``n_total_pairs``, ``agree_score``,
        ``agree_strict``, ``agree_weighted``, ``best_agree_period``,
        ``best_consensus_period``.
    """
    if harmonics is None:
        harmonics = [1.0, 0.5, 2.0, 1.0 / 3, 3.0]

    NESTED_NAMES = {"ELS_ECE_EAOV", "LS_CE_AOV"}

    def _algo_display(algo):
        return algo if algo in NESTED_NAMES else algo.split('_')[0]

    # Build ordered list of (algo_key, display_name) for algorithms that
    # have top-N data.
    algo_pairs = []
    for algo in period_algorithms:
        if algo in top_n_periods_dict:
            algo_pairs.append((algo, _algo_display(algo)))

    results = {}
    for idx, _id in enumerate(keep_id_list):
        # Per-source lookup tables
        top_lists = {}
        best_period_per_algo = {}
        for algo_key, aname in algo_pairs:
            top_lists[aname] = top_n_periods_dict[algo_key][idx]
            best_period_per_algo[aname] = period_dict[algo_key][idx]

        anames = list(top_lists.keys())
        total_pairs = len(anames) * (len(anames) - 1) // 2

        # --- agree_strict: top-1 only, filtered, expanded harmonics ---
        n_strict = 0
        for ii in range(len(anames)):
            for jj in range(ii + 1, len(anames)):
                p_a = best_period_per_algo[anames[ii]]
                p_b = best_period_per_algo[anames[jj]]
                if p_a >= min_agree_period and p_b >= min_agree_period:
                    if _period_match(p_a, p_b, tolerance, harmonics):
                        n_strict += 1

        # --- agree_score: top-N with filtering ---
        n_agree_pairs = 0
        best_agree_period = np.nan
        for ii in range(len(anames)):
            for jj in range(ii + 1, len(anames)):
                ps_a = top_lists[anames[ii]]
                ps_b = top_lists[anames[jj]]
                matched = False
                for pa_val in ps_a:
                    if np.isnan(pa_val) or pa_val < min_agree_period:
                        continue
                    if matched:
                        break
                    for pb_val in ps_b:
                        if np.isnan(pb_val) or pb_val < min_agree_period:
                            continue
                        if _period_match(pa_val, pb_val, tolerance, harmonics):
                            matched = True
                            if np.isnan(best_agree_period):
                                best_agree_period = pa_val
                            break
                if matched:
                    n_agree_pairs += 1

        # --- agree_weighted: rank-weighted, weight = 1/(rank_i * rank_j) ---
        weighted_sum = 0.0
        weight_total = 0.0
        for ii in range(len(anames)):
            for jj in range(ii + 1, len(anames)):
                ps_a = top_lists[anames[ii]]
                ps_b = top_lists[anames[jj]]
                pair_weight = 0.0
                for ri, pa_val in enumerate(ps_a):
                    if np.isnan(pa_val) or pa_val < min_agree_period:
                        continue
                    for rj, pb_val in enumerate(ps_b):
                        if np.isnan(pb_val) or pb_val < min_agree_period:
                            continue
                        if _period_match(pa_val, pb_val, tolerance, harmonics):
                            w = 1.0 / ((ri + 1) * (rj + 1))
                            if w > pair_weight:
                                pair_weight = w
                weighted_sum += pair_weight
                weight_total += 1.0

        # --- best_consensus_period: period with most weighted matches ---
        period_votes = {}
        for aname in anames:
            for rank, p in enumerate(top_lists[aname]):
                if np.isnan(p) or p < min_agree_period:
                    continue
                vote_weight = 0.0
                for other in anames:
                    if other == aname:
                        continue
                    for rj, p_other in enumerate(top_lists[other]):
                        if np.isnan(p_other) or p_other < min_agree_period:
                            continue
                        if _period_match(p, p_other, tolerance, harmonics):
                            vote_weight += 1.0 / ((rank + 1) * (rj + 1))
                            break
                matched_key = None
                for existing_p in period_votes:
                    if _period_match(p, existing_p, tolerance, harmonics):
                        matched_key = existing_p
                        break
                if matched_key is not None:
                    period_votes[matched_key] += vote_weight
                else:
                    period_votes[p] = vote_weight

        best_consensus = np.nan
        if period_votes:
            best_consensus = max(period_votes, key=period_votes.get)
            if period_votes[best_consensus] == 0:
                best_consensus = np.nan

        results[_id] = {
            'n_agree_pairs': n_agree_pairs,
            'n_total_pairs': total_pairs,
            'agree_score': (n_agree_pairs / total_pairs if total_pairs > 0 else 0.0),
            'agree_strict': (n_strict / total_pairs if total_pairs > 0 else 0.0),
            'agree_weighted': (
                weighted_sum / weight_total if weight_total > 0 else 0.0
            ),
            'best_agree_period': best_agree_period,
            'best_consensus_period': best_consensus,
        }

    return results


def compute_fourier_features(lightcurves, periods):
    """Compute Fourier decomposition features for a batch of light curves.

    Uses the periodfind Rust backend (weighted linear least-squares with
    BIC model selection) instead of the per-source scipy curve_fit loop.

    Parameters
    ----------
    lightcurves : list of tuple
        Each element is (times, mags, magerrs).  Raw magnitudes are used
        (no normalization to [0,1]).
    periods : array-like, shape (n_curves,)
        Best-fit period for each light curve.

    Returns
    -------
    features : ndarray, shape (n_curves, 14)
        Columns: [power, BIC, offset, slope, A1, B1, A2, B2, A3, B3,
                  A4, B4, A5, B5]
    """
    time_stack, mag_stack, err_stack = [], [], []
    for lightcurve in lightcurves:
        idx = np.argsort(lightcurve[0])
        time_stack.append(np.asarray(lightcurve[0][idx], dtype=np.float32))
        mag_stack.append(np.asarray(lightcurve[1][idx], dtype=np.float32))
        err_stack.append(np.asarray(lightcurve[2][idx], dtype=np.float32))

    periods_arr = np.asarray(periods, dtype=np.float32)

    fd = periodfind.FourierDecomposition(device='cpu')
    return fd.calc(time_stack, mag_stack, err_stack, periods_arr)


def compute_dmdt_features(lightcurves, dmdt_ints):
    """Compute dm-dt histograms via Rust for a batch of light curves.

    Parameters
    ----------
    lightcurves : list of tuple
        Each element is (times, mags, magerrs).
    dmdt_ints : dict
        Must contain 'dtints' and 'dmints' keys with bin edge arrays.

    Returns
    -------
    ndarray, shape (n_curves, n_dm_bins, n_dt_bins)
        L2-normalised dm-dt histograms.
    """
    time_stack, mag_stack = [], []
    for lc in lightcurves:
        time_stack.append(np.asarray(lc[0], dtype=np.float32))
        mag_stack.append(np.asarray(lc[1], dtype=np.float32))

    dt_edges = np.asarray(dmdt_ints['dtints'], dtype=np.float32)
    dm_edges = np.asarray(dmdt_ints['dmints'], dtype=np.float32)

    dd = periodfind.DmDt()
    return dd.calc(time_stack, mag_stack, dt_edges, dm_edges)


def compute_basic_stats(lightcurves):
    """Compute 22 basic statistics via Rust for a batch of light curves.

    Parameters
    ----------
    lightcurves : list of tuple
        Each element is (times, mags, magerrs).

    Returns
    -------
    ndarray, shape (n_curves, 22)
    """
    time_stack, mag_stack, err_stack = [], [], []
    for lc in lightcurves:
        time_stack.append(np.asarray(lc[0], dtype=np.float32))
        mag_stack.append(np.asarray(lc[1], dtype=np.float32))
        err_stack.append(np.asarray(lc[2], dtype=np.float32))

    bs = periodfind.BasicStats()
    return bs.calc(time_stack, mag_stack, err_stack)


def remove_high_cadence_batch(tme_list, cadence_minutes=30.0):
    """Batch high-cadence removal via Rust.

    Parameters
    ----------
    tme_list : list of (times, mags, errs)
        Each element is a tuple of 1D arrays.
    cadence_minutes : float, default=30.0
        Minimum cadence in minutes.

    Returns
    -------
    list of (ndarray, ndarray, ndarray)
        Filtered (times, mags, errs) tuples.
    """
    times = [np.asarray(tme[0], dtype=np.float32) for tme in tme_list]
    mags = [np.asarray(tme[1], dtype=np.float32) for tme in tme_list]
    errs = [np.asarray(tme[2], dtype=np.float32) for tme in tme_list]

    return periodfind.remove_high_cadence(times, mags, errs, cadence_minutes)
