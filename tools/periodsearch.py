
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import ztfperiodic.utils


def find_periods(algorithm, lightcurves, freqs, batch_size=1,
                 doGPU=False, doCPU=False, doSaveMemory=False,
                 doRemoveTerrestrial=False,
                 doRemoveWindow=False,
                 doUsePDot=False, doSingleTimeSegment=False,
                 freqs_to_remove=None,
                 phase_bins=20, mag_bins=10,
                 doParallel=False,
                 Ncore=4):

    if doRemoveTerrestrial and (freqs_to_remove is not None) and not (algorithm=="LS" or algorithm=="GCE_LS_AOV" or algorithm=="GCE_LS" or algorithm=="GCE_LS_AOV_x3"):
        for pair in freqs_to_remove:
            idx = np.where((freqs < pair[0]) | (freqs > pair[1]))[0]
            freqs = freqs[idx]

    periods_best, significances = [], []
    pdots = np.zeros((len(lightcurves),))
    print('Period finding lightcurves...')
    if doGPU:
    
        if algorithm == "CE":
            from cuvarbase.ce import ConditionalEntropyAsyncProcess
    
            proc = ConditionalEntropyAsyncProcess(use_double=True, use_fast=True, phase_bins=phase_bins, mag_bins=mag_bins, phase_overlap=1, mag_overlap=1, only_keep_best_freq=True)
    
            if doSaveMemory:
                periods_best, significances = proc.batched_run_const_nfreq(lightcurves, batch_size=batch_size, freqs = freqs, only_keep_best_freq=True,show_progress=True,returnBestFreq=True)
            else:

                results = proc.batched_run_const_nfreq(lightcurves, batch_size=batch_size, freqs = freqs, only_keep_best_freq=True,show_progress=True,returnBestFreq=False)
                cnt = 0
                for lightcurve, out in zip(lightcurves,results):
                    periods = 1./out[0]
                    entropies = out[1]

                    significance = np.abs(np.mean(entropies)-np.min(entropies))/np.std(entropies)
                    period = periods[np.argmin(entropies)]
    
                    periods_best.append(period)
                    significances.append(significance)
   
        elif algorithm == "BLS":
            from cuvarbase.bls import eebls_gpu_fast
            for ii,data in enumerate(lightcurves):
                if np.mod(ii,10) == 0:
                    print("%d/%d"%(ii,len(lightcurves)))
                copy = np.ma.copy(data).T
                powers = eebls_gpu_fast(copy[:,0],copy[:,1], copy[:,2],
                                        freq_batch_size=batch_size,
                                        freqs = freqs)
    
                significance = np.abs(np.mean(powers)-np.max(powers))/np.std(powers)
                freq = freqs[np.argmax(powers)]
                period = 1.0/freq
    
                periods_best.append(period)
                significances.append(significance)
    
        elif algorithm == "LS":
            from cuvarbase.lombscargle import LombScargleAsyncProcess, fap_baluev
    
            nfft_sigma, spp = 10, 10
    
            ls_proc = LombScargleAsyncProcess(use_double=True,
                                                  sigma=nfft_sigma)
    
            if doSaveMemory:
                periods_best, significances = ls_proc.batched_run_const_nfreq(lightcurves, batch_size=batch_size, use_fft=True, samples_per_peak=spp, returnBestFreq=True, freqs = freqs, doRemoveTerrestrial=doRemoveTerrestrial, freqs_to_remove=freqs_to_remove)
            else:
                results = ls_proc.batched_run_const_nfreq(lightcurves,
                                                          batch_size=batch_size,
                                                          use_fft=True,
                                                          samples_per_peak=spp,
                                                          returnBestFreq=False,
                                                          freqs = freqs)
    
                for data, out in zip(lightcurves,results):
                    freqs, powers = out
                    if doRemoveTerrestrial and (freqs_to_remove is not None):
                        for pair in freqs_to_remove:
                            idx = np.where((freqs < pair[0]) | (freqs > pair[1]))[0]
                            freqs = freqs[idx]
                            powers = powers[idx]

                    copy = np.ma.copy(data).T
                    fap = fap_baluev(copy[:,0], copy[:,2], powers, np.max(freqs))
                    idx = np.argmin(fap)
    
                    period = 1./freqs[idx]
                    significance = 1./fap[idx]

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
    
                significance = np.abs(np.mean(powers)-np.max(powers))/np.std(powers)
                freq = freqs[np.argmax(powers)]
                period = 1.0/freq
    
                periods_best.append(period)
                significances.append(significance)


        elif algorithm == "GCE":
            from gcex.gce import ConditionalEntropy
            ce = ConditionalEntropy(phase_bins=phase_bins, mag_bins=mag_bins)

            if doUsePDot:
                num_pdots = 10
                max_pdot = 1e-10
                min_pdot = 1e-12
                pdots_to_test = -np.logspace(np.log10(min_pdot), np.log10(max_pdot), num_pdots)
                pdots_to_test = np.append(0,pdots_to_test)
                #pdots_to_test = np.array([-2.365e-11])
            else:
                pdots_to_test = np.array([0.0])

            if doSingleTimeSegment:
                tt = np.empty((0,1))
                for lightcurve in lightcurves:
                    tt = np.unique(np.append(tt, lightcurve[0]))

            maxn = -np.inf
            lightcurves_stack = []
            for lightcurve in lightcurves:
                if doSingleTimeSegment:
                    xy, x_ind, y_ind = np.intersect1d(tt, lightcurve[0], 
                                                      return_indices=True)
                    mag_array = 999*np.ones(tt.shape)
                    magerr_array = 999*np.ones(tt.shape)
                    mag_array[x_ind] = lightcurve[1][y_ind]
                    magerr_array[x_ind] = lightcurve[2][y_ind]
                    lightcurve = (tt, mag_array, magerr_array)
                else:
                    idx = np.argsort(lightcurve[0])
                    tmin = np.min(lightcurve[0])
                    lightcurve = (lightcurve[0][idx]-tmin,
                                  lightcurve[1][idx],
                                  lightcurve[2][idx])

                lightcurve_stack = np.vstack((lightcurve[0],
                                              lightcurve[1],
                                              lightcurve[2])).T
                lightcurves_stack.append(lightcurve_stack)

                if len(idx) > maxn:
                    maxn = len(idx)

            periods_best = np.zeros((len(lightcurves),1))
            significances = np.zeros((len(lightcurves),1))
            pdots = np.zeros((len(lightcurves),1))

            pdots_split = np.array_split(pdots_to_test,len(pdots_to_test))
            for ii, pdot in enumerate(pdots_split):
                print("Running pdot %d / %d" % (ii+1, len(pdots_split)))

                print("Number of lightcurves: %d" % len(lightcurves_stack))
                print("Max length of lightcurves: %d" % maxn)
                print("Batch size: %d" % batch_size)
                print("Number of frequency bins: %d" % len(freqs))
                print("Number of phase bins: %d" % phase_bins)
                print("Number of magnitude bins: %d" % mag_bins)

                results = ce.batched_run_const_nfreq(lightcurves_stack, batch_size, freqs, pdot, show_progress=False)
                periods = 1./freqs
           
                for jj, (lightcurve, entropies2) in enumerate(zip(lightcurves,results)):
                    for kk, entropies in enumerate(entropies2):
                        significance = np.abs(np.mean(entropies)-np.min(entropies))/np.std(entropies)

                        period = periods[np.argmin(entropies)]
                        if significance > significances[jj]:
                            periods_best[jj] = period
                            significances[jj] = significance
                            pdots[jj] = pdot[kk]*1.0 
            pdots, periods_best, significances = pdots.flatten(), periods_best.flatten(), significances.flatten()

        elif (algorithm == "ECE") or (algorithm == "EAOV") or (algorithm == "ELS"):
            if algorithm == "ECE":
                from periodfind.ce import ConditionalEntropy
                #ce = ConditionalEntropy(phase_bins=phase_bins, mag_bins=mag_bins)
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
                pdots_to_test = -np.logspace(np.log10(min_pdot), np.log10(max_pdot), num_pdots)
                pdots_to_test = np.append(0,pdots_to_test)
                #pdots_to_test = np.array([-2.365e-11])
            else:
                pdots_to_test = np.array([0.0])

            if doSingleTimeSegment:
                tt = np.empty((0,1))
                for lightcurve in lightcurves:
                    tt = np.unique(np.append(tt, lightcurve[0]))

            maxn = -np.inf
            time_stack, mag_stack = [], []
            for lightcurve in lightcurves:
                if doSingleTimeSegment:
                    xy, x_ind, y_ind = np.intersect1d(tt, lightcurve[0],
                                                      return_indices=True)
                    mag_array = 999*np.ones(tt.shape)
                    magerr_array = 999*np.ones(tt.shape)
                    mag_array[x_ind] = lightcurve[1][y_ind]
                    magerr_array[x_ind] = lightcurve[2][y_ind]
                    lightcurve = (tt, mag_array, magerr_array)
                else:
                    idx = np.argsort(lightcurve[0])
                    tmin = np.min(lightcurve[0])
                    lightcurve = (lightcurve[0][idx]-tmin,
                                  lightcurve[1][idx],
                                  lightcurve[2][idx])

                lightcurve_stack = np.vstack((lightcurve[0],
                                              lightcurve[1],
                                              lightcurve[2])).T
                time_stack.append(lightcurve[0].astype(np.float32))
                lc = lightcurve[1]
                lc = (lc - np.min(lc))/(np.max(lc)-np.min(lc))
                mag_stack.append(lc.astype(np.float32))

                if len(idx) > maxn:
                    maxn = len(idx)

            print("Number of lightcurves: %d" % len(time_stack))
            print("Max length of lightcurves: %d" % maxn)
            print("Batch size: %d" % batch_size)
            print("Number of frequency bins: %d" % len(freqs))
            print("Number of phase bins: %d" % phase_bins)
            print("Number of magnitude bins: %d" % mag_bins)

            periods = (1.0/freqs).astype(np.float32)
            pdots_to_test = pdots_to_test.astype(np.float32)

            periods_best = np.zeros((len(lightcurves),1))
            significances = np.zeros((len(lightcurves),1))
            pdots = np.zeros((len(lightcurves),1))

            if algorithm == "ECE":
                data_out = ce.calc(time_stack, mag_stack, periods, pdots_to_test)
            elif algorithm == "EAOV":
                data_out = aov.calc(time_stack, mag_stack, periods, pdots_to_test)
            elif algorithm == "ELS":
                data_out = ls.calc(time_stack, mag_stack, periods, pdots_to_test)

            for ii, stat in enumerate(data_out):
                if np.isnan(stat.significance):
                    raise ValueError("Oops... significance  is nan... something went wrong")

                periods_best[ii] = stat.params[0]
                pdots[ii] = stat.params[1]
                significances[ii] = stat.significance
            pdots, periods_best, significances = pdots.flatten(), periods_best.flatten(), significances.flatten()

        elif (algorithm == "ECE_periodogram") or (algorithm == "EAOV_periodogram") or (algorithm == "ELS_periodogram"):
            if algorithm.split("_")[0] == "ECE":
                from periodfind.ce import ConditionalEntropy
                #ce = ConditionalEntropy(phase_bins=phase_bins, mag_bins=mag_bins)
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
                pdots_to_test = -np.logspace(np.log10(min_pdot), np.log10(max_pdot), num_pdots)
                pdots_to_test = np.append(0,pdots_to_test)
                #pdots_to_test = np.array([-2.365e-11])
            else:
                pdots_to_test = np.array([0.0])

            if doSingleTimeSegment:
                tt = np.empty((0,1))
                for lightcurve in lightcurves:
                    tt = np.unique(np.append(tt, lightcurve[0]))

            maxn = -np.inf
            time_stack, mag_stack = [], []
            for lightcurve in lightcurves:
                if doSingleTimeSegment:
                    xy, x_ind, y_ind = np.intersect1d(tt, lightcurve[0],
                                                      return_indices=True)
                    mag_array = 999*np.ones(tt.shape)
                    magerr_array = 999*np.ones(tt.shape)
                    mag_array[x_ind] = lightcurve[1][y_ind]
                    magerr_array[x_ind] = lightcurve[2][y_ind]
                    lightcurve = (tt, mag_array, magerr_array)
                else:
                    idx = np.argsort(lightcurve[0])
                    tmin = np.min(lightcurve[0])
                    lightcurve = (lightcurve[0][idx]-tmin,
                                  lightcurve[1][idx],
                                  lightcurve[2][idx])

                lightcurve_stack = np.vstack((lightcurve[0],
                                              lightcurve[1],
                                              lightcurve[2])).T
                time_stack.append(lightcurve[0].astype(np.float32))
                lc = lightcurve[1]
                lc = (lc - np.min(lc))/(np.max(lc)-np.min(lc))
                mag_stack.append(lc.astype(np.float32))

                if len(idx) > maxn:
                    maxn = len(idx)

            print("Number of lightcurves: %d" % len(time_stack))
            print("Max length of lightcurves: %d" % maxn)
            print("Batch size: %d" % batch_size)
            print("Number of frequency bins: %d" % len(freqs))
            print("Number of phase bins: %d" % phase_bins)
            print("Number of magnitude bins: %d" % mag_bins)

            periods = (1.0/freqs).astype(np.float32)
            pdots_to_test = pdots_to_test.astype(np.float32)

            periods_best = []
            significances = np.zeros((len(lightcurves),1))
            pdots = np.zeros((len(lightcurves),1))

            if algorithm.split("_")[0] == "ECE":
                data_out = ce.calc(time_stack, mag_stack, periods, pdots_to_test, output='periodogram')
            elif algorithm.split("_")[0] == "EAOV":
                data_out = aov.calc(time_stack, mag_stack, periods, pdots_to_test, output='periodogram')
            elif algorithm.split("_")[0] == "ELS":
                data_out = ls.calc(time_stack, mag_stack, periods, pdots_to_test, output='periodogram')

            for ii, stat in enumerate(data_out):
                if algorithm.split("_")[0] == "ECE":
                    significance = np.abs(np.mean(stat.data)-np.min(stat.data))/np.std(stat.data)
                    period = periods[np.argmin(stat.data)]
                elif algorithm.split("_")[0] == "EAOV":
                    significance = np.abs(np.mean(stat.data)-np.min(stat.data))/np.std(stat.data)
                    period = periods[np.argmin(stat.data)]
                elif algorithm.split("_")[0] == "ELS":
                    significance = np.abs(np.mean(stat.data)-np.max(stat.data))/np.std(stat.data)
                    period = periods[np.argmax(stat.data)]

                if np.isnan(significance):
                    raise ValueError("Oops... significance  is nan... something went wrong")

                periods_best.append({'period': period, 'data': stat.data})
                pdots[ii] = pdots_to_test[0]
                significances[ii] = significance
            pdots, significances = pdots.flatten(), significances.flatten()

        elif algorithm == "GCE_LS_AOV":
            nfreqs_to_keep = 100
            nfreqs_to_keep = 50
            df = freqs[1]-freqs[0]

            from cuvarbase.lombscargle import LombScargleAsyncProcess, fap_baluev
            from gcex.gce import ConditionalEntropy
            from ztfperiodic.pyaov.pyaov import aovw, amhw

            ce = ConditionalEntropy(phase_bins=phase_bins, mag_bins=mag_bins)

            freqs_to_keep = {}
            for jj in range(len(lightcurves)):
                freqs_to_keep[jj] = np.empty((0,1))

            pdot = np.array([0.0])
            if doSingleTimeSegment:
                tt = np.empty((0,1))
                for lightcurve in lightcurves:
                    tt = np.unique(np.append(tt, lightcurve[0]))

            maxn = -np.inf
            lightcurves_stack = []
            for lightcurve in lightcurves:
                if doSingleTimeSegment:
                    xy, x_ind, y_ind = np.intersect1d(tt, lightcurve[0],
                                                      return_indices=True)
                    mag_array = 999*np.ones(tt.shape)
                    magerr_array = 999*np.ones(tt.shape)
                    mag_array[x_ind] = lightcurve[1][y_ind]
                    magerr_array[x_ind] = lightcurve[2][y_ind]
                    lightcurve = (tt, mag_array, magerr_array)
                else:
                    tmin = np.min(lightcurve[0])
                    idx = np.argsort(lightcurve[0])
                    lightcurve = (lightcurve[0][idx]-tmin,
                                  lightcurve[1][idx],
                                  lightcurve[2][idx])

                lightcurve_stack = np.vstack((lightcurve[0],
                                              lightcurve[1],
                                              lightcurve[2])).T
                lightcurves_stack.append(lightcurve_stack)

                if len(idx) > maxn:
                    maxn = len(idx)

            print("Number of lightcurves: %d" % len(lightcurves_stack))
            print("Max length of lightcurves: %d" % maxn)
            print("Batch size: %d" % batch_size)
            print("Number of frequency bins: %d" % len(freqs))
            print("Number of phase bins: %d" % phase_bins)
            print("Number of magnitude bins: %d" % mag_bins)

            results = ce.batched_run_const_nfreq(lightcurves_stack, batch_size, freqs, pdot, show_progress=False)

            for jj, (lightcurve, entropies2) in enumerate(zip(lightcurves,results)):
                for kk, entropies in enumerate(entropies2):
                    freqs_tmp = np.copy(freqs)
                    if doRemoveTerrestrial and (freqs_to_remove is not None):
                        for pair in freqs_to_remove:
                            idx = np.where((freqs_tmp < pair[0]) | (freqs_tmp > pair[1]))[0]
                            freqs_tmp = freqs_tmp[idx]
                            entropies = entropies[idx]
                    significance = np.abs(np.mean(entropies)-entropies)/np.std(entropies)
                    idx = np.argsort(significance)[::-1]

                    freqs_to_keep[jj] = np.append(freqs_to_keep[jj],
                                                  freqs_tmp[idx[:nfreqs_to_keep]])

            nfft_sigma, spp = 10, 10

            ls_proc = LombScargleAsyncProcess(use_double=True,
                                                  sigma=nfft_sigma)
            results = ls_proc.batched_run_const_nfreq(lightcurves,
                                                      batch_size=batch_size,
                                                      use_fft=True,
                                                      samples_per_peak=spp,
                                                      returnBestFreq=False,
                                                      freqs = freqs)

            for jj, (data, out) in enumerate(zip(lightcurves,results)):
                freqs, powers = out
                if doRemoveTerrestrial and (freqs_to_remove is not None):
                    for pair in freqs_to_remove:
                        idx = np.where((freqs < pair[0]) | (freqs > pair[1]))[0]
                        freqs = freqs[idx]
                        powers = powers[idx]

                copy = np.ma.copy(data).T
                fap = fap_baluev(copy[:,0], copy[:,2], powers, np.max(freqs))
                idx = np.argsort(fap)
                freqs_to_keep[jj] = np.append(freqs_to_keep[jj],
                                              freqs[idx[:nfreqs_to_keep]])
            ls_proc.finish()

            if doParallel:
                from joblib import Parallel, delayed
                res = Parallel(n_jobs=Ncore)(delayed(calc_AOV)(data, freqs_to_keep[jj], df) for jj, data in enumerate(lightcurves))
                periods_best = [x[0] for x in res]
                significances = [x[1] for x in res]
            else:
                for jj, data in enumerate(lightcurves):
                    if np.mod(jj,10) == 0:
                        print("%d/%d"%(jj,len(lightcurves)))

                    period, significance = calc_AOV(data, freqs_to_keep[jj], df)
                    periods_best.append(period)
                    significances.append(significance)

        elif algorithm == "GCE_LS_AOV_x3":
            nfreqs_to_keep = 100
            nfreqs_to_keep = 50
            df = freqs[1]-freqs[0]

            niter = 3

            from cuvarbase.lombscargle import LombScargleAsyncProcess, fap_baluev
            from gcex.gce import ConditionalEntropy
            from ztfperiodic.pyaov.pyaov import aovw, amhw

            ce = ConditionalEntropy(phase_bins=phase_bins, mag_bins=mag_bins)

            freqs_to_keep = {}
            for jj in range(len(lightcurves)):
                freqs_to_keep[jj] = np.empty((0,1))

            pdot = np.array([0.0])
            if doSingleTimeSegment:
                tt = np.empty((0,1))
                for lightcurve in lightcurves:
                    tt = np.unique(np.append(tt, lightcurve[0]))

            maxn = -np.inf
            lightcurves_stack = []
            for lightcurve in lightcurves:
                if doSingleTimeSegment:
                    xy, x_ind, y_ind = np.intersect1d(tt, lightcurve[0],
                                                      return_indices=True)
                    mag_array = 999*np.ones(tt.shape)
                    magerr_array = 999*np.ones(tt.shape)
                    mag_array[x_ind] = lightcurve[1][y_ind]
                    magerr_array[x_ind] = lightcurve[2][y_ind]
                    lightcurve = (tt, mag_array, magerr_array)
                else:
                    tmin = np.min(lightcurve[0])
                    idx = np.argsort(lightcurve[0])
                    lightcurve = (lightcurve[0][idx]-tmin,
                                  lightcurve[1][idx],
                                  lightcurve[2][idx])

                lightcurve_stack = np.vstack((lightcurve[0],
                                              lightcurve[1],
                                              lightcurve[2])).T
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
                results = ce.batched_run_const_nfreq(lightcurves_stack, batch_size, freqs, pdot, show_progress=False)

                for jj, (lightcurve, entropies2) in enumerate(zip(lightcurves,results)):
                    if nn == 0:
                        entropies_all[jj] = np.empty((0,len(freqs))) 

                    for kk, entropies in enumerate(entropies2):
                        entropies_all[jj] = np.append(entropies_all[jj],
                                                      np.atleast_2d(entropies),
                                                      axis=0)
            
            for jj in entropies_all.keys():
                entropies = np.median(entropies_all[jj], axis=0)
                freqs_tmp = np.copy(freqs)
                if doRemoveTerrestrial and (freqs_to_remove is not None):
                    for pair in freqs_to_remove:
                        idx = np.where((freqs_tmp < pair[0]) | (freqs_tmp > pair[1]))[0]
                        freqs_tmp = freqs_tmp[idx]
                        entropies = entropies[idx]
                significance = np.abs(np.mean(entropies)-entropies)/np.std(entropies)
                idx = np.argsort(significance)[::-1]

                freqs_to_keep[jj] = np.append(freqs_to_keep[jj],
                                              freqs_tmp[idx[:nfreqs_to_keep]])

            nfft_sigma, spp = 10, 10

            freqs_all = {}
            powers_all = {}
            for nn in range(niter):
                ls_proc = LombScargleAsyncProcess(use_double=True,
                                                  sigma=nfft_sigma)
                results = ls_proc.batched_run_const_nfreq(lightcurves,
                                                          batch_size=batch_size,
                                                          use_fft=True,
                                                          samples_per_peak=spp,
                                                          returnBestFreq=False,
                                                          freqs = freqs)

                for jj, (data, out) in enumerate(zip(lightcurves,results)):
                    freqs, powers = out
                    if nn == 0:
                        freqs_all[jj] = freqs
                        powers_all[jj] = np.empty((0,len(freqs)))
                    powers_all[jj] = np.append(powers_all[jj],
                                               np.atleast_2d(powers),
                                               axis=0)
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
                fap = fap_baluev(copy[:,0], copy[:,2], powers, np.max(freqs))
                idx = np.argsort(fap)
                freqs_to_keep[jj] = np.append(freqs_to_keep[jj],
                                              freqs[idx[:nfreqs_to_keep]])

            for jj, data in enumerate(lightcurves):
                if np.mod(jj,10) == 0:
                    print("%d/%d"%(jj,len(lightcurves)))

                copy = np.ma.copy(data).T
                copy[:,1] = (copy[:,1]  - np.min(copy[:,1])) \
                   / (np.max(copy[:,1]) - np.min(copy[:,1]))

                freqs, aovs = np.empty((0,1)), np.empty((0,1))
                for ii, fr0 in enumerate(freqs_to_keep[jj]):
                    err = copy[:,2]
                    aov, frtmp, _ = amhw(copy[:,0], copy[:,1], err,
                                         fr0=fr0-50*df,
                                         fstop=fr0+50*df,
                                         fstep=df/2.0,
                                         nh2=4)
                    idx = np.where(frtmp > 0)[0]

                    aovs = np.append(aovs,aov[idx])
                    freqs = np.append(freqs,frtmp[idx])

                significance = np.abs(np.mean(aovs)-np.max(aovs))/np.std(aovs)
                periods = 1./freqs
                significance = np.max(aovs)
                period = periods[np.argmax(aovs)]

                periods_best.append(period)
                significances.append(significance)

        elif algorithm == "GCE_LS":
            from cuvarbase.lombscargle import LombScargleAsyncProcess, fap_baluev
            from gcex.gce import ConditionalEntropy
            from ztfperiodic.pyaov.pyaov import aovw, amhw

            ce = ConditionalEntropy(phase_bins=phase_bins, mag_bins=mag_bins)

            pdot = np.array([0.0])
            if doSingleTimeSegment:
                tt = np.empty((0,1))
                for lightcurve in lightcurves:
                    tt = np.unique(np.append(tt, lightcurve[0]))

            maxn = -np.inf
            lightcurves_stack = []
            for lightcurve in lightcurves:
                if doSingleTimeSegment:
                    xy, x_ind, y_ind = np.intersect1d(tt, lightcurve[0],
                                                      return_indices=True)
                    mag_array = 999*np.ones(tt.shape)
                    magerr_array = 999*np.ones(tt.shape)
                    mag_array[x_ind] = lightcurve[1][y_ind]
                    magerr_array[x_ind] = lightcurve[2][y_ind]
                    lightcurve = (tt, mag_array, magerr_array)
                else:
                    tmin = np.min(lightcurve[0])
                    idx = np.argsort(lightcurve[0])
                    lightcurve = (lightcurve[0][idx]-tmin,
                                  lightcurve[1][idx],
                                  lightcurve[2][idx])

                lightcurve_stack = np.vstack((lightcurve[0],
                                              lightcurve[1])).T
                lightcurves_stack.append(lightcurve_stack)

                if len(idx) > maxn:
                    maxn = len(idx)

            print("Number of lightcurves: %d" % len(lightcurves_stack))
            print("Max length of lightcurves: %d" % maxn)
            print("Batch size: %d" % batch_size)
            print("Number of frequency bins: %d" % len(freqs))
            print("Number of phase bins: %d" % phase_bins)
            print("Number of magnitude bins: %d" % mag_bins)

            results2 = ce.batched_run_const_nfreq(lightcurves_stack, batch_size, freqs, pdot, show_progress=False)

            nfft_sigma, spp = 10, 10

            ls_proc = LombScargleAsyncProcess(use_double=True,
                                                  sigma=nfft_sigma)
            results1 = ls_proc.batched_run_const_nfreq(lightcurves,
                                                       batch_size=batch_size,
                                                       use_fft=True,
                                                       samples_per_peak=spp,
                                                       returnBestFreq=False,
                                                       freqs = freqs)

            for jj, (data, out, entropies2) in enumerate(zip(lightcurves, results1, results2)):
                entropies = entropies2[0]
                freqs1, powers = out
                if doRemoveTerrestrial and (freqs_to_remove is not None):
                    for pair in freqs_to_remove:
                        idx = np.where((freqs1 < pair[0]) | (freqs1 > pair[1]))[0]
                        freqs1 = freqs1[idx]
                        powers = powers[idx]
                        entropies = entropies[idx]

                #copy = np.ma.copy(data).T
                #fap = fap_baluev(copy[:,0], copy[:,2], powers, np.max(freqs))
                #significance1 = 1./fap
                significance1 = np.abs(powers-np.mean(powers))/np.std(powers)
                significance2 = np.abs(np.mean(entropies)-entropies)/np.std(entropies)
                significance = significance1*significance2
                #significance = significance2
                idx = np.argmax(significance)

                period = 1./freqs[idx]
                significance = significance[idx]

                periods_best.append(period)
                significances.append(significance)
            ls_proc.finish()

        elif algorithm == "FFT":
            from cuvarbase.lombscargle import fap_baluev
            from reikna import cluda
            from reikna.fft.fft import FFT

            T = 30.0/86400.0
            fs = 1.0/T

            api = cluda.get_api('cuda')
            dev = api.get_platforms()[0].get_devices()[0]
            thr = api.Thread(dev)

            x = np.arange(0.0, 12.0/24.0, T).astype(np.complex128)
            fft  = FFT(x, axes=(0,))
            fftc = fft.compile(thr, fast_math=True)

            lightcurves_stack = []

            period_min, period_max = 60.0/86400.0, 12.0*3600.0/86400.0
            freq_min, freq_max = 1/period_max, 1/period_min

            for ii, lightcurve in enumerate(lightcurves):
                bins_tmp = np.arange(np.min(lightcurve[0]), np.max(lightcurve[1]), 1/24.0)
                bins = np.vstack((bins_tmp[:-12],bins_tmp[12:]))
                n, bins_tmp = ztfperiodic.utils.overlapping_histogram(lightcurve[0], bins)
                idx = np.argmax(n)
                bins_max = bins[:,idx]

                x = np.arange(bins_max[0], bins_max[0] + 12.0/24.0, T)
                y = np.interp(x, lightcurve[0], lightcurve[1]).astype(np.complex128)
                yerr = np.interp(x, lightcurve[0], lightcurve[2])
                if len(y) == 0:
                    periods_best.append(-1)
                    significances.append(-1)
                    continue                    
                y = y - np.median(y)
                y = y * np.hanning(len(y))

                dev   = thr.to_device(y)
                fftc(dev, dev)

                Y = dev.get()
                powers = np.abs(Y)
                N = len(y)
                freqs = np.linspace(0.0, 1.0/(2.0*T), N/2)

                powers = powers[:int(N/2)]
                idx = np.where((freqs >= freq_min) & (freqs <= freq_max))[0]

                freqs, powers = freqs[idx], powers[idx]
                powers = powers * freqs**2

                significance = np.abs(np.median(powers)-np.max(powers))/np.std(powers)   
                freq = freqs[np.argmax(powers)]
                period = 1.0/freq

                periods_best.append(period)
                significances.append(significance)

    elif doCPU:
    
        periods = 1/freqs
        period_jobs=1
    
        if algorithm == "LS":
            from astropy.stats import LombScargle
            for ii,data in enumerate(lightcurves):
                if np.mod(ii,1) == 0:
                    print("%d/%d"%(ii,len(lightcurves)))
                copy = np.ma.copy(data).T
                nrows, ncols = copy.shape
    
                if nrows == 1:
                    periods_best.append(-1)
                    significances.append(-1)
                    continue
    
                ls = LombScargle(copy[:,0], copy[:,1], copy[:,2])
                power = ls.power(freqs)
                fap = ls.false_alarm_probability(power,maximum_frequency=np.max(freqs))
    
                idx = np.argmin(fap)
                significance = 1./fap[idx]
                period = 1./freqs[idx]
                periods_best.append(period)
                significances.append(significance)
    
        elif algorithm == "CE":
            from ztfperiodic.period import CE
            for ii,data in enumerate(lightcurves):
                if np.mod(ii,1) == 0:
                    print("%d/%d"%(ii,len(lightcurves)))
    
                copy = np.ma.copy(data).T
                copy[:,1] = (copy[:,1]  - np.min(copy[:,1])) \
                   / (np.max(copy[:,1]) - np.min(copy[:,1]))
                entropies = []
                for period in periods:
                    entropy = CE(period, data=copy, xbins=phase_bins, ybins=mag_bins)
                    entropies.append(entropy)
                significance = np.abs(np.mean(entropies)-np.min(entropies))/np.std(entropies)
                period = periods[np.argmin(entropies)]
    
                periods_best.append(period)
                significances.append(significance)
    
        elif algorithm == "AOV":
            from ztfperiodic.pyaov.pyaov import aovw, amhw
            for ii,data in enumerate(lightcurves):
                if np.mod(ii,10) == 0:
                    print("%d/%d"%(ii,len(lightcurves)))
    
                copy = np.ma.copy(data).T
                copy[:,1] = (copy[:,1]  - np.min(copy[:,1])) \
                   / (np.max(copy[:,1]) - np.min(copy[:,1]))
    
                aov, fr, _ = amhw(copy[:,0], copy[:,1], copy[:,2],
                                  fstop=np.max(1.0/periods),
                                  fstep=1/periods[0])
    
                significance = np.abs(np.mean(aov)-np.max(aov))/np.std(aov)
                period = periods[np.argmax(aov)]
    
                periods_best.append(period)
                significances.append(significance)
   
        elif algorithm == "AOV_cython":
            from AOV_cython import aov as pyaov
            for ii,data in enumerate(lightcurves):
                if np.mod(ii,10) == 0:
                    print("%d/%d"%(ii,len(lightcurves)))

                copy = np.ma.copy(data).T
                copy[:,1] = (copy[:,1]  - np.min(copy[:,1])) \
                   / (np.max(copy[:,1]) - np.min(copy[:,1]))

                aov = pyaov(freqs, copy[:,0], copy[:,1],
                            np.mean(copy[:,1]),
                            len(copy[:,0]), 10, len(freqs))

                significance = np.abs(np.mean(aov)-np.max(aov))/np.std(aov)
                freq = freqs[np.argmax(aov)]
                period = 1.0/freq

                periods_best.append(period)
                significances.append(significance)
 
    return np.array(periods_best), np.array(significances), np.array(pdots)

def calc_AOV(data, freqs_to_keep, df):
    copy = np.ma.copy(data).T
    copy[:,1] = (copy[:,1]  - np.min(copy[:,1])) \
       / (np.max(copy[:,1]) - np.min(copy[:,1]))

    freqs, aovs = np.empty((0,1)), np.empty((0,1))
    for ii, fr0 in enumerate(freqs_to_keep):
        err = copy[:,2]
        aov, frtmp, _ = amhw(copy[:,0], copy[:,1], err,
                             fr0=fr0-50*df,
                             fstop=fr0+50*df,
                             fstep=df/2.0,
                             nh2=4)
        idx = np.where(frtmp > 0)[0]

        aovs = np.append(aovs,aov[idx])
        freqs = np.append(freqs,frtmp[idx])

    significance = np.abs(np.mean(aovs)-np.max(aovs))/np.std(aovs)
    periods = 1./freqs
    significance = np.max(aovs)
    period = periods[np.argmax(aovs)]

    return [period, significance]
