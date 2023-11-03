import numpy as np
import fast_histogram
import warnings
from joblib import Parallel, delayed


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

    fr0 = np.min(freqs)
    fstep = np.diff(freqs)[0]
    fstop = np.max(freqs) + fstep

    if doRemoveTerrestrial and (freqs_to_remove is not None):
        indexes = []
        for pair in freqs_to_remove:
            idx = np.where((freqs < pair[0]) | (freqs > pair[1]))[0]
            indexes += [idx]
            freqs = freqs[idx]

    periods_best, significances = [], []
    pdots = np.zeros((len(lightcurves),))
    print('Period finding lightcurves...')
    if doGPU:

        if (algorithm == "ECE") or (algorithm == "EAOV") or (algorithm == "ELS"):
            if algorithm == "ECE":
                from periodfind.ce import ConditionalEntropy

                ce = ConditionalEntropy(phase_bins, mag_bins)
            elif algorithm == "EAOV":
                from periodfind.aov import AOV

                aov = AOV(phase_bins)
            elif algorithm == "ELS":
                from periodfind.ls import LombScargle

                ls = LombScargle()

            if doUsePDot:
                num_pdots = 10
                max_pdot = 1e-10
                min_pdot = 1e-12
                pdots_to_test = -np.logspace(
                    np.log10(min_pdot), np.log10(max_pdot), num_pdots
                )
                pdots_to_test = np.append(0, pdots_to_test)
            else:
                pdots_to_test = np.array([0.0])

            if doSingleTimeSegment:
                tt = np.empty((0, 1))
                for lightcurve in lightcurves:
                    tt = np.unique(np.append(tt, lightcurve[0]))

            maxn = -np.inf
            time_stack, mag_stack = [], []
            for lightcurve in lightcurves:
                if doSingleTimeSegment:
                    xy, x_ind, y_ind = np.intersect1d(
                        tt, lightcurve[0], return_indices=True
                    )
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

                time_stack.append(lightcurve[0].astype(np.float32))
                lc = lightcurve[1]
                lc = (lc - np.min(lc)) / (np.max(lc) - np.min(lc))
                mag_stack.append(lc.astype(np.float32))

                if len(idx) > maxn:
                    maxn = len(idx)

            print("Number of lightcurves: %d" % len(time_stack))
            print("Max length of lightcurves: %d" % maxn)
            print("Number of frequency bins: %d" % len(freqs))
            print("Number of phase bins: %d" % phase_bins)
            print("Number of magnitude bins: %d" % mag_bins)

            periods = (1.0 / freqs).astype(np.float32)
            pdots_to_test = pdots_to_test.astype(np.float32)

            periods_best = np.zeros((len(lightcurves), 1))
            significances = np.zeros((len(lightcurves), 1))
            pdots = np.zeros((len(lightcurves), 1))

            if algorithm == "ECE":
                data_out = ce.calc(time_stack, mag_stack, periods, pdots_to_test)
            elif algorithm == "EAOV":
                data_out = aov.calc(time_stack, mag_stack, periods, pdots_to_test)
            elif algorithm == "ELS":
                data_out = ls.calc(time_stack, mag_stack, periods, pdots_to_test)

            for ii, stat in enumerate(data_out):
                if np.isnan(stat.significance):
                    warnings.warn(
                        "Oops... significance  is nan... something went wrong"
                    )

                periods_best[ii] = stat.params[0]
                pdots[ii] = stat.params[1]
                significances[ii] = stat.significance
            pdots, periods_best, significances = (
                pdots.flatten(),
                periods_best.flatten(),
                significances.flatten(),
            )

        elif (
            (algorithm == "ECE_periodogram")
            or (algorithm == "EAOV_periodogram")
            or (algorithm == "ELS_periodogram")
        ):
            if algorithm.split("_")[0] == "ECE":
                from periodfind.ce import ConditionalEntropy

                ce = ConditionalEntropy(phase_bins, mag_bins)
            elif algorithm.split("_")[0] == "EAOV":
                from periodfind.aov import AOV

                aov = AOV(phase_bins)
            elif algorithm.split("_")[0] == "ELS":
                from periodfind.ls import LombScargle

                ls = LombScargle()

            if doUsePDot:
                num_pdots = 10
                max_pdot = 1e-10
                min_pdot = 1e-12
                pdots_to_test = -np.logspace(
                    np.log10(min_pdot), np.log10(max_pdot), num_pdots
                )
                pdots_to_test = np.append(0, pdots_to_test)
            else:
                pdots_to_test = np.array([0.0])

            if doSingleTimeSegment:
                tt = np.empty((0, 1))
                for lightcurve in lightcurves:
                    tt = np.unique(np.append(tt, lightcurve[0]))

            maxn = -np.inf
            time_stack, mag_stack = [], []
            for lightcurve in lightcurves:
                if doSingleTimeSegment:
                    _, x_ind, y_ind = np.intersect1d(
                        tt, lightcurve[0], return_indices=True
                    )
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

                time_stack.append(lightcurve[0].astype(np.float32))
                lc = lightcurve[1]
                lc = (lc - np.min(lc)) / (np.max(lc) - np.min(lc))
                mag_stack.append(lc.astype(np.float32))

                if len(idx) > maxn:
                    maxn = len(idx)

            print("Number of lightcurves: %d" % len(time_stack))
            print("Max length of lightcurves: %d" % maxn)
            print("Number of frequency bins: %d" % len(freqs))
            print("Number of phase bins: %d" % phase_bins)
            print("Number of magnitude bins: %d" % mag_bins)

            periods = (1.0 / freqs).astype(np.float32)
            pdots_to_test = pdots_to_test.astype(np.float32)

            periods_best = []
            significances = np.zeros((len(lightcurves), 1))
            pdots = np.zeros((len(lightcurves), 1))

            if algorithm.split("_")[0] == "ECE":
                data_out = ce.calc(
                    time_stack, mag_stack, periods, pdots_to_test, output='periodogram'
                )
            elif algorithm.split("_")[0] == "EAOV":
                data_out = aov.calc(
                    time_stack, mag_stack, periods, pdots_to_test, output='periodogram'
                )
            elif algorithm.split("_")[0] == "ELS":
                data_out = ls.calc(
                    time_stack, mag_stack, periods, pdots_to_test, output='periodogram'
                )

            for ii, stat in enumerate(data_out):
                if algorithm.split("_")[0] == "ECE":
                    significance = np.abs(
                        np.mean(stat.data) - np.min(stat.data)
                    ) / np.std(stat.data)
                    period = periods[np.argmin(stat.data)]
                elif algorithm.split("_")[0] == "EAOV":
                    significance = np.abs(
                        np.mean(stat.data) - np.max(stat.data)
                    ) / np.std(stat.data)
                    period = periods[np.argmax(stat.data)]
                elif algorithm.split("_")[0] == "ELS":
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
            pdots, significances = pdots.flatten(), significances.flatten()

    elif doCPU:

        periods = 1 / freqs

        if "LS" in algorithm:
            from astropy.timeseries import LombScargle

            for ii, data in enumerate(lightcurves):
                if (np.mod(ii, 10) == 0) | ((ii + 1) == len(lightcurves)):
                    print("%d/%d" % (ii + 1, len(lightcurves)))
                copy = np.ma.copy(data).T
                nrows, _ = copy.shape

                if nrows == 1:
                    periods_best.append(-1)
                    significances.append(-1)
                    continue

                ls = LombScargle(copy[:, 0], copy[:, 1], copy[:, 2])
                power = ls.power(freqs)
                fap = ls.false_alarm_probability(power, maximum_frequency=np.max(freqs))

                idx = np.argmin(fap)
                significance = 1.0 / fap[idx]
                period = 1.0 / freqs[idx]

                if "periodogram" in algorithm:
                    periods_best.append({'period': period, 'data': 1.0 / fap})
                else:
                    periods_best.append(period)

                significances.append(significance)

        elif "CE" in algorithm:
            for ii, data in enumerate(lightcurves):
                if (np.mod(ii, 10) == 0) | ((ii + 1) == len(lightcurves)):
                    print("%d/%d" % (ii + 1, len(lightcurves)))

                copy = np.ma.copy(data).T
                copy[:, 1] = (copy[:, 1] - np.min(copy[:, 1])) / (
                    np.max(copy[:, 1]) - np.min(copy[:, 1])
                )
                entropies = Parallel(n_jobs=Ncore)(
                    delayed(CE)(period, copy, phase_bins, mag_bins)
                    for period in periods
                )
                significance = np.abs(np.mean(entropies) - np.min(entropies)) / np.std(
                    entropies
                )
                period = periods[np.argmin(entropies)]

                if "periodogram" in algorithm:
                    periods_best.append(
                        {
                            'period': period,
                            'data': np.abs(np.mean(entropies) - entropies)
                            / np.std(entropies),
                        }
                    )
                else:
                    periods_best.append(period)

                significances.append(significance)

        elif "AOV" in algorithm:
            for ii, data in enumerate(lightcurves):
                if (np.mod(ii, 10) == 0) | ((ii + 1) == len(lightcurves)):
                    print("%d/%d" % (ii + 1, len(lightcurves)))

                copy = np.ma.copy(data).T
                copy[:, 1] = (copy[:, 1] - np.min(copy[:, 1])) / (
                    np.max(copy[:, 1]) - np.min(copy[:, 1])
                )

                aov, _, _ = amhw(
                    copy[:, 0],
                    copy[:, 1],
                    copy[:, 2],
                    fstop=fstop,
                    fstep=fstep,
                    fr0=fr0,
                )

                if doRemoveTerrestrial and (freqs_to_remove is not None):
                    for idx in indexes:
                        aov = aov[idx]

                significance = np.abs(np.mean(aov) - np.max(aov)) / np.std(aov)
                period = periods[np.argmax(aov)]

                if "periodogram" in algorithm:
                    periods_best.append(
                        {
                            'period': period,
                            'data': np.abs(np.mean(aov) - aov) / np.std(aov),
                        }
                    )
                else:
                    periods_best.append(period)

                significances.append(significance)

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
