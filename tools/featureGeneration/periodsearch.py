import numpy as np
from scope import utils
import fast_histogram


def find_periods(
    algorithm,
    lightcurves,
    freqs,
    batch_size=1,
    doGPU=False,
    doCPU=False,
    doSaveMemory=False,
    doRemoveTerrestrial=False,
    doUsePDot=False,
    doSingleTimeSegment=False,
    freqs_to_remove=None,
    phase_bins=20,
    mag_bins=10,
    doParallel=False,
    Ncore=4,
):

    if (
        doRemoveTerrestrial
        and (freqs_to_remove is not None)
        and not (
            algorithm == "LS"
            or algorithm == "GCE_LS_AOV"
            or algorithm == "GCE_LS"
            or algorithm == "GCE_LS_AOV_x3"
        )
    ):
        for pair in freqs_to_remove:
            idx = np.where((freqs < pair[0]) | (freqs > pair[1]))[0]
            freqs = freqs[idx]

    periods_best, significances = [], []
    pdots = np.zeros((len(lightcurves),))
    print('Period finding lightcurves...')
    if doGPU:

        if algorithm == "CE":
            from cuvarbase.ce import ConditionalEntropyAsyncProcess

            proc = ConditionalEntropyAsyncProcess(
                use_double=True,
                use_fast=True,
                phase_bins=phase_bins,
                mag_bins=mag_bins,
                phase_overlap=1,
                mag_overlap=1,
                only_keep_best_freq=True,
            )

            if doSaveMemory:
                periods_best, significances = proc.batched_run_const_nfreq(
                    lightcurves,
                    batch_size=batch_size,
                    freqs=freqs,
                    only_keep_best_freq=True,
                    show_progress=True,
                    returnBestFreq=True,
                )
            else:

                results = proc.batched_run_const_nfreq(
                    lightcurves,
                    batch_size=batch_size,
                    freqs=freqs,
                    only_keep_best_freq=True,
                    show_progress=True,
                    returnBestFreq=False,
                )
                for lightcurve, out in zip(lightcurves, results):
                    periods = 1.0 / out[0]
                    entropies = out[1]

                    significance = np.abs(
                        np.mean(entropies) - np.min(entropies)
                    ) / np.std(entropies)
                    period = periods[np.argmin(entropies)]

                    periods_best.append(period)
                    significances.append(significance)

        elif algorithm == "BLS":
            from cuvarbase.bls import eebls_gpu_fast

            for ii, data in enumerate(lightcurves):
                if np.mod(ii, 10) == 0:
                    print("%d/%d" % (ii, len(lightcurves)))
                copy = np.ma.copy(data).T
                powers = eebls_gpu_fast(
                    copy[:, 0],
                    copy[:, 1],
                    copy[:, 2],
                    freq_batch_size=batch_size,
                    freqs=freqs,
                )

                significance = np.abs(np.mean(powers) - np.max(powers)) / np.std(powers)
                freq = freqs[np.argmax(powers)]
                period = 1.0 / freq

                periods_best.append(period)
                significances.append(significance)

        elif algorithm == "LS":
            from cuvarbase.lombscargle import LombScargleAsyncProcess, fap_baluev

            nfft_sigma, spp = 10, 10

            ls_proc = LombScargleAsyncProcess(use_double=True, sigma=nfft_sigma)

            if doSaveMemory:
                periods_best, significances = ls_proc.batched_run_const_nfreq(
                    lightcurves,
                    batch_size=batch_size,
                    use_fft=True,
                    samples_per_peak=spp,
                    returnBestFreq=True,
                    freqs=freqs,
                    doRemoveTerrestrial=doRemoveTerrestrial,
                    freqs_to_remove=freqs_to_remove,
                )
            else:
                results = ls_proc.batched_run_const_nfreq(
                    lightcurves,
                    batch_size=batch_size,
                    use_fft=True,
                    samples_per_peak=spp,
                    returnBestFreq=False,
                    freqs=freqs,
                )

                for data, out in zip(lightcurves, results):
                    freqs, powers = out
                    if doRemoveTerrestrial and (freqs_to_remove is not None):
                        for pair in freqs_to_remove:
                            idx = np.where((freqs < pair[0]) | (freqs > pair[1]))[0]
                            freqs = freqs[idx]
                            powers = powers[idx]

                    copy = np.ma.copy(data).T
                    fap = fap_baluev(copy[:, 0], copy[:, 2], powers, np.max(freqs))
                    idx = np.argmin(fap)

                    period = 1.0 / freqs[idx]
                    significance = 1.0 / fap[idx]

                    periods_best.append(period)
                    significances.append(significance)

            ls_proc.finish()

        elif algorithm == "PDM":
            from cuvarbase.pdm import PDMAsyncProcess

            kind, nbins = 'binned_linterp', 10

            pdm_proc = PDMAsyncProcess()
            for lightcurve in lightcurves:
                results = pdm_proc.run([lightcurve], kind=kind, nbins=nbins)
                pdm_proc.finish()
                powers = results[0]

                significance = np.abs(np.mean(powers) - np.max(powers)) / np.std(powers)
                freq = freqs[np.argmax(powers)]
                period = 1.0 / freq

                periods_best.append(period)
                significances.append(significance)

        elif algorithm == "GCE":
            from gcex.gce import ConditionalEntropy

            ce = ConditionalEntropy(phase_bins=phase_bins, mag_bins=mag_bins)

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
            lightcurves_stack = []
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

                lightcurve_stack = np.vstack(
                    (lightcurve[0], lightcurve[1], lightcurve[2])
                ).T
                lightcurves_stack.append(lightcurve_stack)

                if len(idx) > maxn:
                    maxn = len(idx)

            periods_best = np.zeros((len(lightcurves), 1))
            significances = np.zeros((len(lightcurves), 1))
            pdots = np.zeros((len(lightcurves), 1))

            pdots_split = np.array_split(pdots_to_test, len(pdots_to_test))
            for ii, pdot in enumerate(pdots_split):
                print("Running pdot %d / %d" % (ii + 1, len(pdots_split)))

                print("Number of lightcurves: %d" % len(lightcurves_stack))
                print("Max length of lightcurves: %d" % maxn)
                print("Batch size: %d" % batch_size)
                print("Number of frequency bins: %d" % len(freqs))
                print("Number of phase bins: %d" % phase_bins)
                print("Number of magnitude bins: %d" % mag_bins)

                results = ce.batched_run_const_nfreq(
                    lightcurves_stack, batch_size, freqs, pdot, show_progress=False
                )
                periods = 1.0 / freqs

                for jj, (lightcurve, entropies2) in enumerate(
                    zip(lightcurves, results)
                ):
                    for kk, entropies in enumerate(entropies2):
                        significance = np.abs(
                            np.mean(entropies) - np.min(entropies)
                        ) / np.std(entropies)

                        period = periods[np.argmin(entropies)]
                        if significance > significances[jj]:
                            periods_best[jj] = period
                            significances[jj] = significance
                            pdots[jj] = pdot[kk] * 1.0
            pdots, periods_best, significances = (
                pdots.flatten(),
                periods_best.flatten(),
                significances.flatten(),
            )

        elif (algorithm == "ECE") or (algorithm == "EAOV") or (algorithm == "ELS"):
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

                lightcurve_stack = np.vstack(
                    (lightcurve[0], lightcurve[1], lightcurve[2])
                ).T
                time_stack.append(lightcurve[0].astype(np.float32))
                lc = lightcurve[1]
                lc = (lc - np.min(lc)) / (np.max(lc) - np.min(lc))
                mag_stack.append(lc.astype(np.float32))

                if len(idx) > maxn:
                    maxn = len(idx)

            print("Number of lightcurves: %d" % len(time_stack))
            print("Max length of lightcurves: %d" % maxn)
            print("Batch size: %d" % batch_size)
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
                    raise ValueError(
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

                lightcurve_stack = np.vstack(
                    (lightcurve[0], lightcurve[1], lightcurve[2])
                ).T
                time_stack.append(lightcurve[0].astype(np.float32))
                lc = lightcurve[1]
                lc = (lc - np.min(lc)) / (np.max(lc) - np.min(lc))
                mag_stack.append(lc.astype(np.float32))

                if len(idx) > maxn:
                    maxn = len(idx)

            print("Number of lightcurves: %d" % len(time_stack))
            print("Max length of lightcurves: %d" % maxn)
            print("Batch size: %d" % batch_size)
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
                        np.mean(stat.data) - np.min(stat.data)
                    ) / np.std(stat.data)
                    period = periods[np.argmin(stat.data)]
                elif algorithm.split("_")[0] == "ELS":
                    significance = np.abs(
                        np.mean(stat.data) - np.max(stat.data)
                    ) / np.std(stat.data)
                    period = periods[np.argmax(stat.data)]

                if np.isnan(significance):
                    raise ValueError(
                        "Oops... significance  is nan... something went wrong"
                    )

                periods_best.append({'period': period, 'data': stat.data})
                pdots[ii] = pdots_to_test[0]
                significances[ii] = significance
            pdots, significances = pdots.flatten(), significances.flatten()

        elif algorithm == "GCE_LS_AOV":
            nfreqs_to_keep = 100
            nfreqs_to_keep = 50
            df = freqs[1] - freqs[0]

            from cuvarbase.lombscargle import LombScargleAsyncProcess, fap_baluev
            from gcex.gce import ConditionalEntropy

            ce = ConditionalEntropy(phase_bins=phase_bins, mag_bins=mag_bins)

            freqs_to_keep = {}
            for jj in range(len(lightcurves)):
                freqs_to_keep[jj] = np.empty((0, 1))

            pdot = np.array([0.0])
            if doSingleTimeSegment:
                tt = np.empty((0, 1))
                for lightcurve in lightcurves:
                    tt = np.unique(np.append(tt, lightcurve[0]))

            maxn = -np.inf
            lightcurves_stack = []
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
                    tmin = np.min(lightcurve[0])
                    idx = np.argsort(lightcurve[0])
                    lightcurve = (
                        lightcurve[0][idx] - tmin,
                        lightcurve[1][idx],
                        lightcurve[2][idx],
                    )

                lightcurve_stack = np.vstack(
                    (lightcurve[0], lightcurve[1], lightcurve[2])
                ).T
                lightcurves_stack.append(lightcurve_stack)

                if len(idx) > maxn:
                    maxn = len(idx)

            print("Number of lightcurves: %d" % len(lightcurves_stack))
            print("Max length of lightcurves: %d" % maxn)
            print("Batch size: %d" % batch_size)
            print("Number of frequency bins: %d" % len(freqs))
            print("Number of phase bins: %d" % phase_bins)
            print("Number of magnitude bins: %d" % mag_bins)

            results = ce.batched_run_const_nfreq(
                lightcurves_stack, batch_size, freqs, pdot, show_progress=False
            )

            for jj, (lightcurve, entropies2) in enumerate(zip(lightcurves, results)):
                for kk, entropies in enumerate(entropies2):
                    freqs_tmp = np.copy(freqs)
                    if doRemoveTerrestrial and (freqs_to_remove is not None):
                        for pair in freqs_to_remove:
                            idx = np.where(
                                (freqs_tmp < pair[0]) | (freqs_tmp > pair[1])
                            )[0]
                            freqs_tmp = freqs_tmp[idx]
                            entropies = entropies[idx]
                    significance = np.abs(np.mean(entropies) - entropies) / np.std(
                        entropies
                    )
                    idx = np.argsort(significance)[::-1]

                    freqs_to_keep[jj] = np.append(
                        freqs_to_keep[jj], freqs_tmp[idx[:nfreqs_to_keep]]
                    )

            nfft_sigma, spp = 10, 10

            ls_proc = LombScargleAsyncProcess(use_double=True, sigma=nfft_sigma)
            results = ls_proc.batched_run_const_nfreq(
                lightcurves,
                batch_size=batch_size,
                use_fft=True,
                samples_per_peak=spp,
                returnBestFreq=False,
                freqs=freqs,
            )

            for jj, (data, out) in enumerate(zip(lightcurves, results)):
                freqs, powers = out
                if doRemoveTerrestrial and (freqs_to_remove is not None):
                    for pair in freqs_to_remove:
                        idx = np.where((freqs < pair[0]) | (freqs > pair[1]))[0]
                        freqs = freqs[idx]
                        powers = powers[idx]

                copy = np.ma.copy(data).T
                fap = fap_baluev(copy[:, 0], copy[:, 2], powers, np.max(freqs))
                idx = np.argsort(fap)
                freqs_to_keep[jj] = np.append(
                    freqs_to_keep[jj], freqs[idx[:nfreqs_to_keep]]
                )
            ls_proc.finish()

            if doParallel:
                from joblib import Parallel, delayed

                res = Parallel(n_jobs=Ncore)(
                    delayed(calc_AOV)(amhw, data, freqs_to_keep[jj], df)
                    for jj, data in enumerate(lightcurves)
                )
                periods_best = [x[0] for x in res]
                significances = [x[1] for x in res]
            else:
                for jj, data in enumerate(lightcurves):
                    if np.mod(jj, 10) == 0:
                        print("%d/%d" % (jj, len(lightcurves)))

                    period, significance = calc_AOV(amhw, data, freqs_to_keep[jj], df)
                    periods_best.append(period)
                    significances.append(significance)

        elif algorithm == "GCE_LS_AOV_x3":
            nfreqs_to_keep = 100
            nfreqs_to_keep = 50
            df = freqs[1] - freqs[0]

            niter = 3

            from cuvarbase.lombscargle import LombScargleAsyncProcess, fap_baluev
            from gcex.gce import ConditionalEntropy

            ce = ConditionalEntropy(phase_bins=phase_bins, mag_bins=mag_bins)

            freqs_to_keep = {}
            for jj in range(len(lightcurves)):
                freqs_to_keep[jj] = np.empty((0, 1))

            pdot = np.array([0.0])
            if doSingleTimeSegment:
                tt = np.empty((0, 1))
                for lightcurve in lightcurves:
                    tt = np.unique(np.append(tt, lightcurve[0]))

            maxn = -np.inf
            lightcurves_stack = []
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
                    tmin = np.min(lightcurve[0])
                    idx = np.argsort(lightcurve[0])
                    lightcurve = (
                        lightcurve[0][idx] - tmin,
                        lightcurve[1][idx],
                        lightcurve[2][idx],
                    )

                lightcurve_stack = np.vstack(
                    (lightcurve[0], lightcurve[1], lightcurve[2])
                ).T
                lightcurves_stack.append(lightcurve_stack)

                if len(idx) > maxn:
                    maxn = len(idx)

            print("Number of lightcurves: %d" % len(lightcurves_stack))
            print("Max length of lightcurves: %d" % maxn)
            print("Batch size: %d" % batch_size)
            print("Number of frequency bins: %d" % len(freqs))
            print("Number of phase bins: %d" % phase_bins)
            print("Number of magnitude bins: %d" % mag_bins)

            entropies_all = {}
            for nn in range(niter):
                results = ce.batched_run_const_nfreq(
                    lightcurves_stack, batch_size, freqs, pdot, show_progress=False
                )

                for jj, (lightcurve, entropies2) in enumerate(
                    zip(lightcurves, results)
                ):
                    if nn == 0:
                        entropies_all[jj] = np.empty((0, len(freqs)))

                    for kk, entropies in enumerate(entropies2):
                        entropies_all[jj] = np.append(
                            entropies_all[jj], np.atleast_2d(entropies), axis=0
                        )

            for jj in entropies_all.keys():
                entropies = np.median(entropies_all[jj], axis=0)
                freqs_tmp = np.copy(freqs)
                if doRemoveTerrestrial and (freqs_to_remove is not None):
                    for pair in freqs_to_remove:
                        idx = np.where((freqs_tmp < pair[0]) | (freqs_tmp > pair[1]))[0]
                        freqs_tmp = freqs_tmp[idx]
                        entropies = entropies[idx]
                significance = np.abs(np.mean(entropies) - entropies) / np.std(
                    entropies
                )
                idx = np.argsort(significance)[::-1]

                freqs_to_keep[jj] = np.append(
                    freqs_to_keep[jj], freqs_tmp[idx[:nfreqs_to_keep]]
                )

            nfft_sigma, spp = 10, 10

            freqs_all = {}
            powers_all = {}
            for nn in range(niter):
                ls_proc = LombScargleAsyncProcess(use_double=True, sigma=nfft_sigma)
                results = ls_proc.batched_run_const_nfreq(
                    lightcurves,
                    batch_size=batch_size,
                    use_fft=True,
                    samples_per_peak=spp,
                    returnBestFreq=False,
                    freqs=freqs,
                )

                for jj, (data, out) in enumerate(zip(lightcurves, results)):
                    freqs, powers = out
                    if nn == 0:
                        freqs_all[jj] = freqs
                        powers_all[jj] = np.empty((0, len(freqs)))
                    powers_all[jj] = np.append(
                        powers_all[jj], np.atleast_2d(powers), axis=0
                    )
                ls_proc.finish()

            for jj in powers_all.keys():
                data = lightcurves[jj]
                freqs = freqs_all[jj]
                powers = np.median(powers_all[jj], axis=0)
                if doRemoveTerrestrial and (freqs_to_remove is not None):
                    for pair in freqs_to_remove:
                        idx = np.where((freqs < pair[0]) | (freqs > pair[1]))[0]
                        freqs = freqs[idx]
                        powers = powers[idx]

                copy = np.ma.copy(data).T
                fap = fap_baluev(copy[:, 0], copy[:, 2], powers, np.max(freqs))
                idx = np.argsort(fap)
                freqs_to_keep[jj] = np.append(
                    freqs_to_keep[jj], freqs[idx[:nfreqs_to_keep]]
                )

            for jj, data in enumerate(lightcurves):
                if np.mod(jj, 10) == 0:
                    print("%d/%d" % (jj, len(lightcurves)))

                copy = np.ma.copy(data).T
                copy[:, 1] = (copy[:, 1] - np.min(copy[:, 1])) / (
                    np.max(copy[:, 1]) - np.min(copy[:, 1])
                )

                freqs, aovs = np.empty((0, 1)), np.empty((0, 1))
                for ii, fr0 in enumerate(freqs_to_keep[jj]):
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

                periods_best.append(period)
                significances.append(significance)

        elif algorithm == "GCE_LS":
            from cuvarbase.lombscargle import LombScargleAsyncProcess, fap_baluev
            from gcex.gce import ConditionalEntropy

            ce = ConditionalEntropy(phase_bins=phase_bins, mag_bins=mag_bins)

            pdot = np.array([0.0])
            if doSingleTimeSegment:
                tt = np.empty((0, 1))
                for lightcurve in lightcurves:
                    tt = np.unique(np.append(tt, lightcurve[0]))

            maxn = -np.inf
            lightcurves_stack = []
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
                    tmin = np.min(lightcurve[0])
                    idx = np.argsort(lightcurve[0])
                    lightcurve = (
                        lightcurve[0][idx] - tmin,
                        lightcurve[1][idx],
                        lightcurve[2][idx],
                    )

                lightcurve_stack = np.vstack((lightcurve[0], lightcurve[1])).T
                lightcurves_stack.append(lightcurve_stack)

                if len(idx) > maxn:
                    maxn = len(idx)

            print("Number of lightcurves: %d" % len(lightcurves_stack))
            print("Max length of lightcurves: %d" % maxn)
            print("Batch size: %d" % batch_size)
            print("Number of frequency bins: %d" % len(freqs))
            print("Number of phase bins: %d" % phase_bins)
            print("Number of magnitude bins: %d" % mag_bins)

            results2 = ce.batched_run_const_nfreq(
                lightcurves_stack, batch_size, freqs, pdot, show_progress=False
            )

            nfft_sigma, spp = 10, 10

            ls_proc = LombScargleAsyncProcess(use_double=True, sigma=nfft_sigma)
            results1 = ls_proc.batched_run_const_nfreq(
                lightcurves,
                batch_size=batch_size,
                use_fft=True,
                samples_per_peak=spp,
                returnBestFreq=False,
                freqs=freqs,
            )

            for jj, (data, out, entropies2) in enumerate(
                zip(lightcurves, results1, results2)
            ):
                entropies = entropies2[0]
                freqs1, powers = out
                if doRemoveTerrestrial and (freqs_to_remove is not None):
                    for pair in freqs_to_remove:
                        idx = np.where((freqs1 < pair[0]) | (freqs1 > pair[1]))[0]
                        freqs1 = freqs1[idx]
                        powers = powers[idx]
                        entropies = entropies[idx]

                significance1 = np.abs(powers - np.mean(powers)) / np.std(powers)
                significance2 = np.abs(np.mean(entropies) - entropies) / np.std(
                    entropies
                )
                significance = significance1 * significance2
                idx = np.argmax(significance)

                period = 1.0 / freqs[idx]
                significance = significance[idx]

                periods_best.append(period)
                significances.append(significance)
            ls_proc.finish()

        elif algorithm == "FFT":
            from cuvarbase.lombscargle import fap_baluev
            from reikna import cluda
            from reikna.fft.fft import FFT

            T = 30.0 / 86400.0

            api = cluda.get_api('cuda')
            dev = api.get_platforms()[0].get_devices()[0]
            thr = api.Thread(dev)

            x = np.arange(0.0, 12.0 / 24.0, T).astype(np.complex128)
            fft = FFT(x, axes=(0,))
            fftc = fft.compile(thr, fast_math=True)

            lightcurves_stack = []

            period_min, period_max = 60.0 / 86400.0, 12.0 * 3600.0 / 86400.0
            freq_min, freq_max = 1 / period_max, 1 / period_min

            for ii, lightcurve in enumerate(lightcurves):
                bins_tmp = np.arange(
                    np.min(lightcurve[0]), np.max(lightcurve[1]), 1 / 24.0
                )
                bins = np.vstack((bins_tmp[:-12], bins_tmp[12:]))
                n, bins_tmp = utils.overlapping_histogram(lightcurve[0], bins)
                idx = np.argmax(n)
                bins_max = bins[:, idx]

                x = np.arange(bins_max[0], bins_max[0] + 12.0 / 24.0, T)
                y = np.interp(x, lightcurve[0], lightcurve[1]).astype(np.complex128)
                if len(y) == 0:
                    periods_best.append(-1)
                    significances.append(-1)
                    continue
                y = y - np.median(y)
                y = y * np.hanning(len(y))

                dev = thr.to_device(y)
                fftc(dev, dev)

                Y = dev.get()
                powers = np.abs(Y)
                N = len(y)
                freqs = np.linspace(0.0, 1.0 / (2.0 * T), N / 2)

                powers = powers[: int(N / 2)]
                idx = np.where((freqs >= freq_min) & (freqs <= freq_max))[0]

                freqs, powers = freqs[idx], powers[idx]
                powers = powers * freqs**2

                significance = np.abs(np.median(powers) - np.max(powers)) / np.std(
                    powers
                )
                freq = freqs[np.argmax(powers)]
                period = 1.0 / freq

                periods_best.append(period)
                significances.append(significance)

    elif doCPU:

        periods = 1 / freqs

        if algorithm == "LS":
            from astropy.timeseries import LombScargle

            for ii, data in enumerate(lightcurves):
                if np.mod(ii, 1) == 0:
                    print("%d/%d" % (ii, len(lightcurves)))
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
                periods_best.append(period)
                significances.append(significance)

        elif algorithm == "CE":
            for ii, data in enumerate(lightcurves):
                if np.mod(ii, 1) == 0:
                    print("%d/%d" % (ii, len(lightcurves)))

                copy = np.ma.copy(data).T
                copy[:, 1] = (copy[:, 1] - np.min(copy[:, 1])) / (
                    np.max(copy[:, 1]) - np.min(copy[:, 1])
                )
                entropies = []
                for period in periods:
                    entropy = CE(period, data=copy, xbins=phase_bins, ybins=mag_bins)
                    entropies.append(entropy)
                significance = np.abs(np.mean(entropies) - np.min(entropies)) / np.std(
                    entropies
                )
                period = periods[np.argmin(entropies)]

                periods_best.append(period)
                significances.append(significance)

        elif algorithm == "AOV":
            for ii, data in enumerate(lightcurves):
                if np.mod(ii, 10) == 0:
                    print("%d/%d" % (ii, len(lightcurves)))

                copy = np.ma.copy(data).T
                copy[:, 1] = (copy[:, 1] - np.min(copy[:, 1])) / (
                    np.max(copy[:, 1]) - np.min(copy[:, 1])
                )

                aov, _, _ = amhw(
                    copy[:, 0],
                    copy[:, 1],
                    copy[:, 2],
                    fstop=np.max(1.0 / periods),
                    fstep=1 / periods[0],
                )

                significance = np.abs(np.mean(aov) - np.max(aov)) / np.std(aov)
                period = periods[np.argmax(aov)]

                periods_best.append(period)
                significances.append(significance)

        elif algorithm == "AOV_cython":
            from AOV_cython import aov as pyaov

            for ii, data in enumerate(lightcurves):
                if np.mod(ii, 10) == 0:
                    print("%d/%d" % (ii, len(lightcurves)))

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
