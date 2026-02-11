import numpy as np
import fast_histogram
import warnings

import periodfind

# ---------------------------------------------------------------------------
# Algorithm name mapping
# ---------------------------------------------------------------------------

_ALGO_MAP = {"CE", "AOV", "LS", "FPW"}


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
):

    if doRemoveTerrestrial and (freqs_to_remove is not None):
        indexes = []
        for pair in freqs_to_remove:
            idx = np.where((freqs < pair[0]) | (freqs > pair[1]))[0]
            indexes += [idx]
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
        needs_errs = algo_name == "FPW"

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

    # -----------------------------------------------------------------------
    # AOV_cython (separate Cython library, not periodfind)
    # -----------------------------------------------------------------------
    elif algorithm == "AOV_cython":
        from AOV_cython import aov as pyaov

        for ii, data in enumerate(lightcurves):
            if (np.mod(ii, 10) == 0) | ((ii + 1) == len(lightcurves)):
                print("%d/%d" % (ii + 1, len(lightcurves)))

            copy = np.ma.copy(data).T
            copy[:, 1] = (copy[:, 1] - np.min(copy[:, 1])) / (
                np.max(copy[:, 1]) - np.min(copy[:, 1])
            )

            aov = pyaov(
                freqs,
                copy[:, 0],
                copy[:, 1],
                np.mean(copy[:, 1]),
                len(copy[:, 0]),
                10,
                len(freqs),
            )

            significance = np.abs(np.mean(aov) - np.max(aov)) / np.std(aov)
            freq = freqs[np.argmax(aov)]
            period = 1.0 / freq

            periods_best.append(period)
            significances.append(significance)

    return np.array(periods_best), np.array(significances), np.array(pdots)


def extract_top_n_periods(periodogram_results, freqs, n_top=8):
    """Extract top N periods from periodogram output of find_periods.

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

    for i, res in enumerate(periodogram_results):
        data = res['data'].flatten()
        if len(data) == 0:
            continue

        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            continue

        # CE uses minima; others use maxima
        best_idx_min = np.argmin(data)
        best_period = res['period']

        # Determine if this is a CE algorithm (minima = best) by checking
        # which extreme matches the reported best period
        use_minima = np.isclose(periods_grid[best_idx_min], best_period, rtol=1e-4)

        if use_minima:
            sorted_indices = np.argsort(data)  # ascending = best CE first
            sigs = np.abs(mean_val - data[sorted_indices]) / std_val
        else:
            sorted_indices = np.argsort(data)[::-1]  # descending = best LS/AOV first
            sigs = np.abs(data[sorted_indices] - mean_val) / std_val

        n_fill = min(n_top, len(sorted_indices))
        top_periods[i, :n_fill] = periods_grid[sorted_indices[:n_fill]]
        top_significances[i, :n_fill] = sigs[:n_fill]

    return top_periods, top_significances


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


def calc_AOV(amhw, data, freqs_to_keep, df):
    copy = np.ma.copy(data).T
    copy[:, 1] = (copy[:, 1] - np.min(copy[:, 1])) / (
        np.max(copy[:, 1]) - np.min(copy[:, 1])
    )

    freqs, aovs = np.empty((0, 1)), np.empty((0, 1))
    for _, fr0 in enumerate(freqs_to_keep):
        err = copy[:, 2]
        aov, frtmp, _ = amhw(
            copy[:, 0],
            copy[:, 1],
            err,
            fr0=fr0 - 50 * df,
            fstop=fr0 + 50 * df,
            fstep=df / 2.0,
            nh2=4,
        )
        idx = np.where(frtmp > 0)[0]

        aovs = np.append(aovs, aov[idx])
        freqs = np.append(freqs, frtmp[idx])

    significance = np.abs(np.mean(aovs) - np.max(aovs)) / np.std(aovs)
    periods = 1.0 / freqs
    significance = np.max(aovs)
    period = periods[np.argmax(aovs)]

    return [period, significance]


def CE(period, data, xbins=10, ybins=5):
    """
    Returns the conditional entropy of *data* rephased with *period*.

    **Parameters**

    period : number
        The period to rephase *data* by.
    data : array-like, shape = [n_samples, 2] or [n_samples, 3]
        Array containing columns *time*, *mag*, and (optional) *error*.
    xbins : int, optional
        Number of phase bins (default 10).
    ybins : int, optional
        Number of magnitude bins (default 5).
    """
    if period <= 0:
        return np.PINF

    r = np.ma.array(data, copy=True)
    r[:, 0] = np.mod(r[:, 0], period) / period

    bins = fast_histogram.histogram2d(
        r[:, 0], r[:, 1], range=[[0, 1], [0, 1]], bins=[xbins, ybins]
    )
    size = r.shape[0]

    if size > 0:
        divided_bins = bins / size

        # indices where that is positive to avoid division by zero
        arg_positive = divided_bins > 0

        # array containing the sums of each column in the bins array
        column_sums = np.sum(divided_bins, axis=1)  # changed 0 by 1

        # array is repeated row-wise, so that it can be sliced by arg_positive
        column_sums = np.repeat(np.atleast_2d(column_sums).T, ybins, axis=1)

        # select only the elements in both arrays which correspond to a positive bin
        select_divided_bins = divided_bins[arg_positive]
        select_column_sums = column_sums[arg_positive]

        # initialize the result array
        A = np.empty((xbins, ybins), dtype=float)

        # store at every index [i,j] in A which corresponds to a positive bin:
        A[arg_positive] = select_divided_bins * np.log(
            select_column_sums / select_divided_bins
        )

        # store 0 at every index in A which corresponds to a non-positive bin
        A[~arg_positive] = 0

        # return the summation
        return np.sum(A)

    else:
        return np.PINF


def amhw(time, amplitude, error, fstop, fstep, nh2=3, fr0=0.0):
    '''
    th,fr,frmax=pyaov.amhw(time, valin, error, fstop, fstep, nh2=3, fr0=0.)

    Purpose: Returns multiharmonic AOV periodogram, obtained by fitting data
        with a series of trigonometric polynomials. For default nh2=3 this
        is Lomb-Scargle periodogram corrected for constant shift.
    Input:
        time, amplitude, error : numpy arrays of size (n*1)
        fstop: frequency to stop calculation at, float
        fstep: size of frequency steps, float
    Optional input:
        nh2[=3]: no. of model parms. (number of harmonics=nh2/2)
        fr0[=0.]: start frequency

    Output:
        th,fr: periodogram values & frequencies: numpy arrays of size (m*1)
              where m = (fstop-fr0)/fstep+1
        frmax: frequency of maximum

    Method:
        General method involving projection onto orthogonal trigonometric
        polynomials is due to Schwarzenberg-Czerny, 1996. For nh2=2 or 3 it reduces
        Ferraz-Mello (1991), i.e. to Lomb-Scargle periodogram improved by constant
        shift of values. Advantage of the shift is vividly illustrated by Foster (1995).
    Please quote:
        A.Schwarzenberg-Czerny, 1996, Astrophys. J.,460, L107.
    Other references:
        Foster, G., 1995, AJ v.109, p.1889 (his Fig.1).
        Ferraz-Mello, S., 1981, AJ v.86, p.619.
        Lomb, N. R., 1976, Ap&SS v.39, p.447.
        Scargle, J. D., 1982, ApJ v.263, p.835.
    '''
    #
    # Python wrapper for period search routines
    # (C) Alex Schwarzenberg-Czerny, 2011                alex@camk.edu.pl
    # Based on the wrapper scheme contributed by Ewald Zietsman <ewald.zietsman@gmail.com>
    import aov as _aov

    # check the arrays here, make sure they are all the same size
    try:
        assert time.size == amplitude.size == error.size
    except AssertionError:
        print('Input arrays must be the same dimensions')
        return 0

    # check the other input values
    try:
        assert fstop > 0
        assert fstep > 0
    except AssertionError:
        print('Frequency stop and step values must be greater than 0')
        return 0

    # maybe something else can go wrong?
    try:
        th, frmax = _aov.aov.aovmhw(
            time,
            amplitude,
            error,
            fstep,
            int((fstop - fr0) / fstep + 1),
            fr0=fr0,
            nh2=nh2,
        )

        # make an array that contains the frequencies too
        freqs = np.linspace(fr0, fstop, int((fstop - fr0) / fstep + 1))
        return th, freqs, frmax

    except Exception as e:
        print(e)
        print("Something unexpected went wrong!!")
        return 0
