"""
eBOSC (extended Better Oscillation Detection) function library
Rewritten from MATLAB to Python by Julian Q. Kosciessa

The original license information follows:
---
This file is part of the extended Better OSCillation detection (eBOSC) library.

The eBOSC library is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The eBOSC library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Foobar.  If not, see <http://www.gnu.org/licenses/>.

Copyright 2020 Julian Q. Kosciessa, Thomas H. Grandy, Douglas D. Garrett & Markus Werkle-Bergner
---
"""

import numpy.matlib
import numpy as np
# import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats.distributions import chi2
from ebosc.helpers import find_nearest_value
from ebosc.BOSC import BOSC_tf

def eBOSC_getThresholds(cfg_eBOSC, TFR, eBOSC):
    """This function estimates the static duration and power thresholds and
    saves information regarding the overall spectrum and background.
    Inputs: 
               cfg | config structure with cfg.eBOSC field
               TFR | time-frequency matrix
               eBOSC | main eBOSC output structure; will be updated
    
    Outputs: 
               eBOSC   | updated w.r.t. background info (see below)
                       | bg_pow: overall power spectrum
                       | bg_log10_pow: overall power spectrum (log10)
                       | pv: intercept and slope of fit
                       | mp: linear background power
                       | pt: power threshold
               pt | empirical power threshold
               dt | duration threshold
    """

    # concatenate power estimates in time across trials of interest
    
    trial2extract = cfg_eBOSC['trial_background']
    # remove BGpad at beginning and end to avoid edge artifacts
    time2extract = np.arange(cfg_eBOSC['pad.background_sample']+1, TFR.shape[2]-cfg_eBOSC['pad.background_sample']+1,1)
    # index both trial and time dimension simultaneously
    TFR = TFR[np.ix_(trial2extract,range(TFR.shape[1]),time2extract)]
    # concatenate trials in time dimension: permute dimensions, then reshape
    TFR_t = np.transpose(TFR, [1,2,0])
    BG = TFR_t.reshape(TFR_t.shape[0],TFR_t.shape[1]*TFR_t.shape[2])
    del TFR_t, trial2extract, time2extract    
    # plt.imshow(BG[:,0:100], extent=[0, 1, 0, 1])
    
    # if frequency ranges should be exluded to reduce the influence of
    # rhythmic peaks on the estimation of the linear background, the
    # following section removes these specified ranges
    freqKeep = np.ones(cfg_eBOSC['F'].shape, dtype=bool)
    # allow for no peak removal
    if cfg_eBOSC['threshold.excludePeak'].size == 0:
        print("NOT removing frequency peaks from the background")
    else:
        print("Removing frequency peaks from the background")
        # n-dimensional arrays allow for the removal of multiple peaks
        for indExFreq in range(cfg_eBOSC['threshold.excludePeak'].shape[0]):
            # find empirical peak in specified range
            freqInd1 = np.where(cfg_eBOSC['F'] >= cfg_eBOSC['threshold.excludePeak'][indExFreq,0])[0][0]
            freqInd2 = np.where(cfg_eBOSC['F'] <= cfg_eBOSC['threshold.excludePeak'][indExFreq,1])[-1][-1]
            freqidx = np.arange(freqInd1,freqInd2+1)
            meanbg_within_range = list(BG[freqidx,:].mean(1))
            indPos = meanbg_within_range.index(max(meanbg_within_range))
            indPos = freqidx[indPos]
            # approximate wavelet extension in frequency domain
            # note: we do not remove the specified range, but the FWHM
            # around the empirical peak
            LowFreq = cfg_eBOSC['F'][indPos]-(((2/cfg_eBOSC['wavenumber'])*cfg_eBOSC['F'][indPos])/2)
            UpFreq = cfg_eBOSC['F'][indPos]+(((2/cfg_eBOSC['wavenumber'])*cfg_eBOSC['F'][indPos])/2)
            # index power estimates within the above range to remove from BG fit
            freqKeep[np.logical_and(cfg_eBOSC['F'] >= LowFreq, cfg_eBOSC['F'] <= UpFreq)] = False

    fitInput = {}
    fitInput['f_'] = cfg_eBOSC['F'][freqKeep]
    fitInput['BG_'] = BG[freqKeep, :]
   
    dataForBG = np.log10(fitInput['BG_']).mean(1)
    
    # perform the robust linear fit, only including putatively aperiodic components (i.e., peak exclusion)
    # replicate TukeyBiweight from MATLABs robustfit function
    exog = np.log10(fitInput['f_'])
    exog = sm.add_constant(exog)
    endog = dataForBG
    rlm_model = sm.RLM(endog, exog, M=sm.robust.norms.TukeyBiweight())
    rlm_results = rlm_model.fit()
    # MATLAB: b = robustfit(np.log10(fitInput['f_']),dataForBG)
    pv = np.zeros(2)
    pv[0] = rlm_results.params[1]
    pv[1] = rlm_results.params[0]
    mp = 10**(np.polyval(pv,np.log10(cfg_eBOSC['F'])))

    # compute eBOSC power (pt) and duration (dt) thresholds: 
    # power threshold is based on a chi-square distribution with df=2 and mean as estimated above
    pt=chi2.ppf(cfg_eBOSC['threshold.percentile'],2)*mp/2
    # duration threshold is the specified number of cycles, so it scales with frequency
    dt=(cfg_eBOSC['threshold.duration']*cfg_eBOSC['fsample']/cfg_eBOSC['F'])
    dt=np.transpose(dt, [1,0])

    # save multiple time-invariant estimates that could be of interest:
    # overall wavelet power spectrum (NOT only background)
    time2encode = np.arange(cfg_eBOSC['pad.total_sample']+1, BG.shape[1]-cfg_eBOSC['pad.total_sample']+1,1)
    eBOSC['static.bg_pow'].loc[cfg_eBOSC['tmp_channel'],:] = BG[:,time2encode].mean(1)
    # eBOSC[cfg_eBOSC['tmp_channelID']] = {'static.bg_pow': BG[:,time2encode].mean(1)}
    # log10-transformed wavelet power spectrum (NOT only background)
    eBOSC['static.bg_log10_pow'].loc[cfg_eBOSC['tmp_channel'],:] = np.log10(BG[:,time2encode]).mean(1)
    # intercept and slope parameters of the robust linear 1/f fit (log-log)
    eBOSC['static.pv'].loc[cfg_eBOSC['tmp_channel'],:] = pv
    # linear background power at each estimated frequency
    eBOSC['static.mp'].loc[cfg_eBOSC['tmp_channel'],:] = mp
    # statistical power threshold
    eBOSC['static.pt'].loc[cfg_eBOSC['tmp_channel'],:] = pt

    return eBOSC, pt, dt

def eBOSC_episode_sparsefreq(cfg_eBOSC, detected, TFR):
    """Sparsen the detected matrix along the frequency dimension
    """    
    print('Creating sparse detected matrix ...')
    
    freqWidth = (2/cfg_eBOSC['wavenumber'])*cfg_eBOSC['F']
    lowFreq = cfg_eBOSC['F']-(freqWidth/2)
    highFreq = cfg_eBOSC['F']+(freqWidth/2)
    # %% define range for each frequency across which max. is detected
    fmat = np.zeros([cfg_eBOSC['F'].shape[0],3])
    for [indF,valF] in enumerate(cfg_eBOSC['F']):
        #print(indF)
        lastVal = np.where(cfg_eBOSC['F']<=lowFreq[indF])[0]
        if len(lastVal)>0:
            # first freq falling into range
            fmat[indF,0] = lastVal[-1]+1
        else: fmat[indF,0] = 0
        firstVal = np.where(cfg_eBOSC['F']>=highFreq[indF])[0]
        if len(firstVal)>0:
            # last freq falling into range
            fmat[indF,2] = firstVal[0]-1
        else: fmat[indF,2] = cfg_eBOSC['F'].shape[0]-1
    fmat[:,1] = np.arange(0, cfg_eBOSC['F'].shape[0],1)
    del indF
    range_cur = np.diff(fmat, axis=1)
    range_cur = [int(np.max(range_cur[:,0])), int(np.max(range_cur[:,1]))]
    # %% perform the actual search
    # initialize variables
    # append frequency search space (i.e. range at both ends. first index refers to lower range
    c1 = np.zeros([int(range_cur[0]),TFR.shape[1]])
    c2 = TFR*detected
    c3 = np.zeros([int(range_cur[1]),TFR.shape[1]])
    tmp_B = np.concatenate([c1, c2, c3])
    del c1,c2,c3
    # preallocate matrix (incl. padding , which will be removed)
    detected = np.zeros(tmp_B.shape)
    # loop across frequencies. note that indexing respects the appended segments
    freqs_to_search = np.arange(int(range_cur[0]), int(tmp_B.shape[0]-range_cur[1]),1)
    for f in freqs_to_search:
        # encode detected positions where power is higher than in LOWER and HIGHER ranges
        range1 = [f+np.arange(1,int(range_cur[1])+1)][0]
        range2 = [f-np.arange(1,int(range_cur[0])+1)][0]
        ranges = np.concatenate([range1,range2])
        detected[f,:] = np.logical_and(tmp_B[f,:] != 0, np.min(tmp_B[f,:] >= tmp_B[ranges,:],axis=0))
    # only retain data without padded zeros
    detected = detected[freqs_to_search,:]
    return detected

def eBOSC_episode_postproc_fwhm(cfg_eBOSC, episodes, TFR):
    """
    % This function performs post-processing of input episodes by checking
    % whether 'detected' time points can trivially be explained by the FWHM of
    % the wavelet used in the time-frequency transform.
    %
    % Inputs: 
    %           cfg | config structure with cfg.eBOSC field
    %           episodes | table of episodes
    %           TFR | time-frequency matrix
    %
    % Outputs: 
    %           episodes_new | updated table of episodes
    %           detected_new | updated binary detected matrix
    """
    
    print("Applying FWHM post-processing ...")
    
    # re-initialize detected_new (for post-proc results)
    detected_new = np.zeros(TFR.shape)
    # initialize new dictionary to save results in
    episodesTable = {}
    for entry in episodes:
        episodesTable[entry] = []

    for e in range(len(episodes['Trial'])):
        # get temporary frequency vector
        f_ = episodes['Frequency'][e]
        f_unique = np.unique(f_)           
        # find index within minor tolerance (float arrays)
        f_ind_unique = np.where(np.abs(cfg_eBOSC['F'][:,None] - f_unique) < 1e-5)
        f_ind_unique = f_ind_unique[0]
        # get temporary amplitude vector
        a_ = episodes['Power'][e]
        # location in time with reference to matrix TFR
        t_ind = np.int_(np.arange(episodes['ColID'][e][0], episodes['ColID'][e][-1]+1))
        # initiate bias matrix (only requires to encode frequencies occuring within episode)
        biasMat = np.zeros([len(f_unique),len(a_)])

        for tp in range(len(a_)):
            # The FWHM correction is done independently at each
            # frequency. To accomplish this, we actually reference
            # to the original data in the TF matrix.
            # search within frequencies that occur within the episode
            for f in range(len(f_unique)):
                # create wavelet with center frequency and amplitude at time point
                st=1/(2*np.pi*(f_unique/cfg_eBOSC['wavenumber']))
                step_size = 1/cfg_eBOSC['fsample']
                t=np.arange(-3.6*st[f],3.6*st[f]+step_size,step_size)
                wave = np.exp(-t**2/(2*st[f]**2))*np.exp(1j*2*np.pi*f_unique[f]*t)                
                if cfg_eBOSC['postproc.effSignal'] == 'all':
                    # Morlet wavelet with amplitude-power threshold modulation
                    m = TFR[f_ind_unique[f], int(t_ind[tp])]*wave
                elif cfg_eBOSC['postproc.effSignal'] == 'PT':
                    m = (TFR[f_ind_unique[f], int(t_ind[tp])]-
                         cfg_eBOSC['tmp.pt'][f_ind_unique[f]])*wave
                # amplitude of wavelet
                wl_a = abs(m)
                maxval = max(wl_a)
                maxloc = np.where(np.abs(wl_a[:,None] - maxval) < 1e-5)[0][0]
                index_fwhm = np.where(wl_a>= maxval/2)[0][0]
                # amplitude at fwhm, freq
                fwhm_a = wl_a[index_fwhm]
                if cfg_eBOSC['postproc.effSignal'] =='PT':
                    # re-add power threshold
                    fwhm_a = fwhm_a+cfg_eBOSC['tmp.pt'][f_ind_unique[f]]
                correctionDist = maxloc-index_fwhm
                # extract FWHM amplitude of frequency- and amplitude-specific wavelet
                # check that lower fwhm is part of signal 
                if tp-correctionDist >= 0:
                    # and that existing value is lower than update
                    if biasMat[f,tp-correctionDist] < fwhm_a:
                        biasMat[f,tp-correctionDist] = fwhm_a
                # check that upper fwhm is part of signal 
                if tp+correctionDist+1 <= biasMat.shape[1]:
                    # and that existing value is lower than update
                    if biasMat[f,tp+correctionDist] < fwhm_a:
                        biasMat[f,tp+correctionDist] = fwhm_a

        # plt.imshow(biasMat, extent=[0, 1, 0, 1])

        # retain only those points that are larger than the FWHM
        aMat_retain = np.zeros(biasMat.shape)
        indFreqs = np.where(np.abs(f_[:,None] - f_unique) < 1e-5)
        indFreqs = indFreqs[1]
        for indF in range(len(f_unique)):
            aMat_retain[indF,np.where(indFreqs == indF)[0]] = np.transpose(a_[indFreqs == indF])
        # anything that is lower than the convolved wavelet is removed
        aMat_retain[aMat_retain <= biasMat] = 0

        # identify which time points to retain and discard
        # Options: only correct at signal edge; correct within entire signal
        keep = aMat_retain.mean(0)>0
        keep = keep>0
        if cfg_eBOSC['postproc.edgeOnly'] == 'yes':
            keepEdgeRemovalOnly = np.zeros([len(keep)],dtype=bool)
            keepEdgeRemovalOnly[np.arange(np.where(keep==1)[0][0],np.where(keep==1)[0][-1]+1)] = True
            keep = keepEdgeRemovalOnly
            del keepEdgeRemovalOnly
            
        # get new episodes
        keep = np.concatenate(([0], keep, [0]))
        d_keep = np.diff(keep.astype(float))
    
        if max(d_keep) == 1 and min(d_keep) == -1:
            # start and end indices
            ind_epsd_begin = np.where(d_keep == 1)[0]
            ind_epsd_end = np.where(d_keep == -1)[0]-1
            for i in range(len(ind_epsd_begin)):
                # check for passing the duration requirement
                # get average frequency
                tmp_col = np.arange(ind_epsd_begin[i],ind_epsd_end[i]+1)
                avg_frq = np.mean(f_[tmp_col])
                # match to closest frequency
                [tmp_a, indF] = find_nearest_value(cfg_eBOSC['F'], avg_frq)
                # check number of data points to fulfill number of cycles criterion
                num_pnt = np.floor((cfg_eBOSC['fsample']/ avg_frq) * int(np.reshape(cfg_eBOSC['threshold.duration'],[-1,1])[indF]))
                # if duration criterion remains fulfilled, encode in table
                if len(tmp_col) >= num_pnt:
                    # update all data in table with new episode limits
                    episodesTable['RowID'].append(episodes['RowID'][e][tmp_col])
                    episodesTable['ColID'].append([t_ind[tmp_col[0]], t_ind[tmp_col[-1]]])
                    episodesTable['Frequency'].append(f_[tmp_col])
                    episodesTable['FrequencyMean'].append(np.mean(episodesTable['Frequency'][-1]))
                    episodesTable['Power'].append(a_[tmp_col])
                    episodesTable['PowerMean'].append(np.mean(episodesTable['Power'][-1]))
                    episodesTable['DurationS'].append(np.diff(episodesTable['ColID'][-1])[0] / cfg_eBOSC['fsample'])
                    episodesTable['DurationC'].append(episodesTable['DurationS'][-1] * episodesTable['FrequencyMean'][-1])
                    episodesTable['Trial'].append(cfg_eBOSC['tmp_trial'])
                    episodesTable['Channel'].append(cfg_eBOSC['tmp_channel']) 
                    episodesTable['Onset'].append(cfg_eBOSC['time.time_tfr'][int(episodesTable['ColID'][-1][0])])
                    episodesTable['Offset'].append(cfg_eBOSC['time.time_tfr'][int(episodesTable['ColID'][-1][-1])])
                    episodesTable['SNR'].append(episodes['SNR'][e][tmp_col])
                    episodesTable['SNRMean'].append(np.mean(episodesTable['SNR'][-1]))
                    # set all detected points to one in binary detected matrix
                    detected_new[episodesTable['RowID'][-1],t_ind[tmp_col]] = 1
                    
    # plt.imshow(detected_new, extent=[0, 1, 0, 1])
    # return post-processed episode dictionary and updated binary detected matrix
    return episodesTable, detected_new

def eBOSC_episode_postproc_maxbias(cfg_eBOSC, episodes, TFR):
    """
    % This function performs post-processing of input episodes by checking
    % whether 'detected' time points can be explained by the simulated extension of
    % the wavelet used in the time-frequency transform.
    %
    % Inputs: 
    %           cfg | config structure with cfg.eBOSC field
    %           episodes | table of episodes
    %           TFR | time-frequency matrix
    %
    % Outputs: 
    %           episodes_new | updated table of episodes
    %           detected_new | updated binary detected matrix
    
    % This method works as follows: we estimate the bias introduced by
    % wavelet convolution. The bias is represented by the amplitudes
    % estimated for the zero-shouldered signal (i.e. for which no real 
    % data was initially available). The influence of episodic
    % amplitudes on neighboring time points is assessed by scaling each
    % time point's amplitude with the last 'rhythmic simulated time
    % point', i.e. the first time wavelet amplitude in the simulated
    % rhythmic time points. At this time point the 'bias' is maximal,
    % although more precisely, this amplitude does not represent a
    % bias per se.
    """
    
    print("Applying maxbias post-processing ...")
    
    # re-initialize detected_new (for post-proc results)
    N_freq = TFR.shape[0]
    N_tp = TFR.shape[1]
    detected_new = np.zeros([N_freq, N_tp]);
    # initialize new dictionary to save results in
    # this is required as episodes may split, thus needing novel entries
    episodesTable = {}
    for entry in episodes:
        episodesTable[entry] = []
    
    # generate "bias" matrix
    # the logic here is as follows: we take a sinusoid, zero-pad it, and get the TFR
    # the bias is the tfr power produced for the padding (where power should be zero)
    B_bias = np.zeros([len(cfg_eBOSC['F']),len(cfg_eBOSC['F']),2*N_tp+1])
    amp_max = np.zeros([len(cfg_eBOSC['F']), len(cfg_eBOSC['F'])])
    for f in range(len(cfg_eBOSC['F'])):
        # temporary time vector and signal
        step_size = 1/cfg_eBOSC['fsample']
        time = np.arange(step_size, 1/cfg_eBOSC['F'][f]+step_size,step_size)
        tmp_sig = np.cos(time*2*np.pi*cfg_eBOSC['F'][f])*-1+1
        # signal for time-frequency analysis
        signal = np.concatenate((np.zeros([N_tp]), tmp_sig, np.zeros([N_tp])))
        [tmp_bias_mat, tmp_time, tmp_freq] = BOSC_tf(signal,cfg_eBOSC['F'],cfg_eBOSC['fsample'],cfg_eBOSC['wavenumber'])
        # bias matrix
        points_begin = np.arange(0,N_tp+1)
        points_end = np.arange(N_tp,B_bias.shape[2]+1)
        # for some reason, we have to transpose the matrix here, as the submatrix dimension order changes???
        B_bias[f,:,points_begin] = np.transpose(tmp_bias_mat[:,points_begin])
        B_bias[f,:,points_end] = np.transpose(np.fliplr(tmp_bias_mat[:,points_begin]))
        # maximum amplitude
        amp_max[f,:] = B_bias[f,:,:].max(1)
        # plt.imshow(amp_max, extent=[0, 1, 0, 1])

    # midpoint index
    ind_mid = N_tp+1
    # loop episodes
    for e in range(len(episodes['Trial'])):
        # get temporary frequency vector
        f_ = episodes['Frequency'][e]
        # get temporary amplitude vector
        a_ = episodes['Power'][e]
        m_ = np.zeros([len(a_),len(a_)])
        # location in time with reference to matrix TFR
        t_ind = np.arange(int(episodes['ColID'][e][0]),int(episodes['ColID'][e][-1]+1))
        # indices of time points' frequencies within "bias" matrix
        f_vec = episodes['RowID'][e]
        # figure; hold on;
        for tp in range(len(a_)):
            # index of current point's frequency within "bias" matrix
            ind_f = f_vec[tp]
            # get bias vector that varies with frequency of the
            # timepoints in the episode
            temporalBiasIndices = np.arange(ind_mid+1-tp,ind_mid+len(a_)-tp+1)
            ind1 = numpy.matlib.repmat(ind_f,len(f_vec),1)
            ind2 = np.reshape(f_vec,[-1,1])
            ind3 = np.reshape(temporalBiasIndices,[-1,1])
            indices = np.ravel_multi_index([ind1, ind2, ind3], 
                                           dims = B_bias.shape, order = 'C')
            tmp_biasVec = B_bias.flatten('C')[indices]
            # temporary "bias" vector (frequency-varying)
            if cfg_eBOSC['postproc.effSignal'] == 'all':
                tmp_bias = ((tmp_biasVec/np.reshape(amp_max[ind_f,f_vec],[-1,1]))*a_[tp])
            elif cfg_eBOSC['postproc.effSignal'] == 'PT':
                tmp_bias = ((tmp_biasVec/np.reshape(amp_max[ind_f,f_vec],[-1,1]))*
                            (a_[tp]-cfg_eBOSC['tmp.pt'][ind_f])) + cfg_eBOSC['tmp.pt'][ind_f]
            # compare to data
            m_[tp,:] = np.transpose(a_ >= tmp_bias)
            #plot(a_', 'k'); hold on; plot(tmp_bias, 'r');

        # identify which time points to retain and discard
        # Options: only correct at signal edge; correct within entire signal
        keep = m_.sum(0) == len(a_)
        if cfg_eBOSC['postproc.edgeOnly'] == 'yes':
            # keep everything that would be kept within the vector,
            # no removal within episode except for edges possible
            keepEdgeRemovalOnly = np.zeros([len(keep)],dtype=bool)
            keepEdgeRemovalOnly[np.arange(np.where(keep==1)[0][0],np.where(keep==1)[0][-1]+1)] = True
            keep = keepEdgeRemovalOnly
            del keepEdgeRemovalOnly

        # get new episodes
        keep = np.concatenate(([0], keep, [0]))
        d_keep = np.diff(keep.astype(float))
    
        if max(d_keep) == 1 and min(d_keep) == -1:
            # start and end indices
            ind_epsd_begin = np.where(d_keep == 1)[0]
            ind_epsd_end = np.where(d_keep == -1)[0]-1
            for i in range(len(ind_epsd_begin)):
                # check for passing the duration requirement
                # get average frequency
                tmp_col = np.arange(ind_epsd_begin[i],ind_epsd_end[i]+1)
                avg_frq = np.mean(f_[tmp_col])
                # match to closest frequency
                [tmp_a, indF] = find_nearest_value(cfg_eBOSC['F'], avg_frq)
                # check number of data points to fulfill number of cycles criterion
                num_pnt = np.floor((cfg_eBOSC['fsample']/ avg_frq) * int(np.reshape(cfg_eBOSC['threshold.duration'],[-1,1])[indF]))
                # if duration criterion remains fulfilled, encode in table
                if len(tmp_col) >= num_pnt:
                    # update all data in table with new episode limits
                    episodesTable['RowID'].append(episodes['RowID'][e][tmp_col])
                    episodesTable['ColID'].append([t_ind[tmp_col[0]], t_ind[tmp_col[-1]]])
                    episodesTable['Frequency'].append(f_[tmp_col])
                    episodesTable['FrequencyMean'].append(np.mean(episodesTable['Frequency'][-1]))
                    episodesTable['Power'].append(a_[tmp_col])
                    episodesTable['PowerMean'].append(np.mean(episodesTable['Power'][-1]))
                    episodesTable['DurationS'].append(np.diff(episodesTable['ColID'][-1])[0] / cfg_eBOSC['fsample'])
                    episodesTable['DurationC'].append(episodesTable['DurationS'][-1] * episodesTable['FrequencyMean'][-1])
                    episodesTable['Trial'].append(cfg_eBOSC['tmp_trial'])
                    episodesTable['Channel'].append(cfg_eBOSC['tmp_channel']) 
                    episodesTable['Onset'].append(cfg_eBOSC['time.time_tfr'][int(episodesTable['ColID'][-1][0])])
                    episodesTable['Offset'].append(cfg_eBOSC['time.time_tfr'][int(episodesTable['ColID'][-1][-1])])
                    episodesTable['SNR'].append(episodes['SNR'][e][tmp_col])
                    episodesTable['SNRMean'].append(np.mean(episodesTable['SNR'][-1]))
                    # set all detected points to one in binary detected matrix
                    detected_new[episodesTable['RowID'][-1],t_ind[tmp_col]] = 1
    # return post-processed episode dictionary and updated binary detected matrix
    return episodesTable, detected_new

def eBOSC_episode_rm_shoulder(cfg_eBOSC,detected1,episodes):
    """ Remove parts of the episode that fall into the 'shoulder' of individual
    trials. There is no check for adherence to a given duration criterion necessary,
    as the point of the padding of the detected matrix is exactly to account
    for allowing the presence of a few cycles.
    """

    print("Removing padding from detected episodes")

    ind1 = cfg_eBOSC['pad.detection_sample']
    ind2 = detected1.shape[1] - cfg_eBOSC['pad.detection_sample']
    rmv = []
    for j in range(len(episodes['Trial'])):
        # get time points of current episode
        tmp_col = np.arange(episodes['ColID'][j][0],episodes['ColID'][j][1]+1)
        # find time points that fall inside the padding (i.e. on- and offset)
        ex = np.where(np.logical_or(tmp_col < ind1, tmp_col >= ind2))[0]
        # remove padded time points from episodes
        tmp_col = np.delete(tmp_col, ex)
        episodes['RowID'][j] = np.delete(episodes['RowID'][j], ex)
        episodes['Power'][j] = np.delete(episodes['Power'][j], ex)
        episodes['Frequency'][j] = np.delete(episodes['Frequency'][j], ex)
        episodes['SNR'][j] = np.delete(episodes['SNR'][j], ex)
        # if nothing remains of episode: retain for later deletion
        if len(tmp_col)==0:
            rmv.append(j)
        else:
            # shift onset according to padding
            # Important: new col index is indexing w.r.t. to matrix AFTER
            # detected padding is removed!
            tmp_col = tmp_col - ind1
            episodes['ColID'][j] = [tmp_col[0], tmp_col[-1]]
            # re-compute mean frequency
            episodes['FrequencyMean'][j] = np.mean(episodes['Frequency'][j])
            # re-compute mean amplitude
            episodes['PowerMean'][j] = np.mean(episodes['Power'][j])
            # re-compute mean SNR
            episodes['SNRMean'][j] = np.mean(episodes['SNR'][j])
            # re-compute duration
            episodes['DurationS'][j] = np.diff(episodes['ColID'][j])[0] / cfg_eBOSC['fsample']
            episodes['DurationC'][j] = episodes['DurationS'][j] * episodes['FrequencyMean'][j]
            # update absolute on-/offsets (should remain the same)
            episodes['Onset'][j] = cfg_eBOSC['time.time_det'][int(episodes['ColID'][j][0])]
            episodes['Offset'][j] = cfg_eBOSC['time.time_det'][int(episodes['ColID'][j][-1])]
    # remove now empty episodes from table    
    for entry in episodes:
        # https://stackoverflow.com/questions/21032034/deleting-multiple-indexes-from-a-list-at-once-python
        episodes[entry] = [v for i, v in enumerate(episodes[entry]) if i not in rmv]
    return episodes

def eBOSC_episode_create(cfg_eBOSC,TFR,detected,eBOSC):
    """This function creates continuous rhythmic "episodes" and attempts to control for the impact of wavelet parameters.
      Time-frequency points that best represent neural rhythms are identified by
      heuristically removing temporal and frequency leakage. 
    
     Frequency leakage: at each frequency x time point, power has to exceed neighboring frequencies.
      Then it is checked whether the detected time-frequency points belong to
      a continuous episode for which (1) the frequency maximally changes by 
      +/- n steps (cfg.eBOSC.fstp) from on time point to the next and (2) that is at 
      least as long as n number of cycles (cfg.eBOSC.threshold.duration) of the average freqency
      of that episode (a priori duration threshold).
    
     Temporal leakage: The impact of the amplitude at each time point within a rhythmic episode on previous
      and following time points is tested with the goal to exclude supra-threshold time
      points that are due to the wavelet extension in time. 
    
    Input:   
               cfg         | config structure with cfg.eBOSC field
               TFR         | time-frequency matrix (excl. WLpadding)
               detected    | detected oscillations in TFR (based on power and duration threshold)
               eBOSC       | main eBOSC output structure; necessary to read in
                               prior eBOSC.episodes if they exist in a loop
    
    Output:  
               detected_new    | new detected matrix with frequency leakage removed
               episodesTable   | table with specific episode information:
                     Trial: trial index (corresponds to cfg.eBOSC.trial)
                     Channel: channel index
                     FrequencyMean: mean frequency of episode (Hz)
                     DurationS: episode duration (in sec)
                     DurationC: episode duration (in cycles, based on mean frequency)
                     PowerMean: mean amplitude of amplitude
                     Onset: episode onset in s
                     Offset: episode onset in s
                     Power: (cell) time-resolved wavelet-based amplitude estimates during episode
                     Frequency: (cell) time-resolved wavelet-based frequency
                     RowID: (cell) row index (frequency dimension): following eBOSC_episode_rm_shoulder relative to data excl. detection padding
                     ColID: (cell) column index (time dimension)
                     SNR: (cell) time-resolved signal-to-noise ratio: momentary amplitude/static background estimate at channel*frequency
                     SNRMean: mean signal-to-noise ratio
    """

    # initialize dictionary to save results in
    episodesTable = {}
    episodesTable['RowID'] = []
    episodesTable['ColID'] = []
    episodesTable['Frequency'] = []
    episodesTable['FrequencyMean'] = []
    episodesTable['Power'] = []
    episodesTable['PowerMean'] = []
    episodesTable['DurationS'] = []
    episodesTable['DurationC'] = []
    episodesTable['Trial'] = []
    episodesTable['Channel'] = []
    episodesTable['Onset'] = []
    episodesTable['Offset'] = []
    episodesTable['SNR'] = []
    episodesTable['SNRMean'] = []
    
    # %% Accounting for the frequency spread of the wavelet
    
    # Here, we compute the bandpass response as given by the wavelet
    # formula and apply half of the BP repsonse on top of the center frequency.
    # Because of log-scaling, the widths are not the same on both sides.
    
    detected = eBOSC_episode_sparsefreq(cfg_eBOSC, detected, TFR)    
    
    # %%  Create continuous rhythmic episodes
    
    # define step size in adjacency matrix
    cfg_eBOSC['fstp'] = 1
        
    # add zeros
    padding = np.zeros([cfg_eBOSC['fstp'],detected.shape[1]])
    detected_remaining = np.vstack([padding, detected, padding])
    detected_remaining[:,0] = 0
    detected_remaining[:,-1] = 0
    # detected_remaining serves as a dummy matrix; unless all entries from detected_remaining are
    # removed, we will continue extracting episodes
    tmp_B1 = np.vstack([padding, TFR*detected, padding])
    tmp_B1[:,0] = 0
    tmp_B1[:,-1] = 0
    detected_new = np.zeros(detected.shape)

    while sum(sum(detected_remaining)) > 0:
        # sampling point counter
        x = []
        y = []
        # find seed (remember that numpy uses row-first format!)
        # we need increasing x-axis sorting here
        [tmp_y,tmp_x] = np.where(np.matrix.transpose(detected_remaining)==1)
        x.append(tmp_x[0])
        y.append(tmp_y[0])
        # check next sampling point
        chck = 0
        while chck == 0:
            # next sampling point
            next_point = y[-1]+1
            next_freqs = np.arange(x[-1]-cfg_eBOSC['fstp'],
                          x[-1]+cfg_eBOSC['fstp']+1)
            tmp = np.where(detected_remaining[next_freqs,next_point]==1)[0]
            if tmp.size > 0:
                y.append(next_point)
                if tmp.size > 1:
                    # JQK 161017: It is possible that an episode is branching 
                    # two ways, hence we follow the 'strongest' branch; 
                    # Note that there is no correction for 1/f here, but 
                    # practically, it leads to satisfying results 
                    # (i.e. following the longer episodes).
                    tmp_data = tmp_B1[next_freqs,next_point]
                    tmp = np.where(tmp_data == max(tmp_data))[0]
                x.append(next_freqs[tmp[0]])
            else:
                chck = 1
            
        # check for passing the duration requirement
        # get average frequency
        avg_frq = np.mean(cfg_eBOSC['F'][np.array(x)-cfg_eBOSC['fstp']])
        # match to closest frequency
        [tmp_a, indF] = find_nearest_value(cfg_eBOSC['F'], avg_frq)
        # check number of data points to fulfill number of cycles criterion
        num_pnt = np.floor((cfg_eBOSC['fsample']/ avg_frq) * int(np.reshape(cfg_eBOSC['threshold.duration'],[-1,1])[indF]))
        if len(y) >= num_pnt:
            # %% encode episode that crosses duration threshold
            episodesTable['RowID'].append(np.array(x)-cfg_eBOSC['fstp'])
            episodesTable['ColID'].append([np.single(y[0]), np.single(y[-1])])
            episodesTable['Frequency'].append(np.single(cfg_eBOSC['F'][episodesTable['RowID'][-1]]))
            episodesTable['FrequencyMean'].append(np.single(avg_frq))
            tmp_x = episodesTable['RowID'][-1]
            tmp_y = np.arange(int(episodesTable['ColID'][-1][0]),int(episodesTable['ColID'][-1][1])+1)
            linIdx = np.ravel_multi_index([np.reshape(tmp_x,[-1,1]),
                                  np.reshape(tmp_y,[-1,1])], 
                                 dims=TFR.shape, order='C')
            episodesTable['Power'].append(np.single(TFR.flatten('C')[linIdx]))
            episodesTable['PowerMean'].append(np.mean(episodesTable['Power'][-1]))
            episodesTable['DurationS'].append(np.single(len(y)/cfg_eBOSC['fsample']))
            episodesTable['DurationC'].append(episodesTable['DurationS'][-1]*episodesTable['FrequencyMean'][-1])
            episodesTable['Trial'].append(cfg_eBOSC['tmp_trial']) # Note that the trial is non-zero-based
            episodesTable['Channel'].append(cfg_eBOSC['tmp_channel']) 
            # episode onset in absolute time
            episodesTable['Onset'].append(cfg_eBOSC['time.time_tfr'][int(episodesTable['ColID'][-1][0])]) 
            # episode offset in absolute time
            episodesTable['Offset'].append(cfg_eBOSC['time.time_tfr'][int(episodesTable['ColID'][-1][-1])]) 
            # extract (static) background power at frequencies
            episodesTable['SNR'].append(episodesTable['Power'][-1]/
                                 eBOSC['static.pt'].iloc[cfg_eBOSC['tmp_channelID'],
                                                         episodesTable['RowID'][-1]].values)
            episodesTable['SNRMean'].append(np.mean(episodesTable['SNR'][-1]))
            
            # remove processed segment from detected matrix
            detected_remaining[x,y] = 0
            # set all detected points to one in binary detected matrix
            rows = episodesTable['RowID'][-1]
            cols = np.arange(int(episodesTable['ColID'][-1][0]),
                                  int(episodesTable['ColID'][-1][1])+1)
            detected_new[rows,cols] = 1
        else:
            # %% remove episode from consideration due to being lower than duration
            detected_remaining[x,y] = 0
        
        # some sanity checks that episode selection was sensible
        #plt.imshow(detected, extent=[0, 1, 0, 1])
        #plt.imshow(detected_new, extent=[0, 1, 0, 1])
    
    # %%  Exclude temporal amplitude "leakage" due to wavelet smearing
    # temporarily pass on power threshold for easier access
    cfg_eBOSC['tmp.pt'] = eBOSC['static.pt'].loc[cfg_eBOSC['tmp_channel']].values
    
    # only do this if there are any episodes to fine-tune
    if cfg_eBOSC['postproc.use'] == 'yes' and len(episodesTable['Trial']) > 0:
        if cfg_eBOSC['postproc.method'] == 'FWHM':
            [episodesTable, detected_new] = eBOSC_episode_postproc_fwhm(cfg_eBOSC, episodesTable, TFR)
        elif cfg_eBOSC['postproc.method'] == 'MaxBias':
            [episodesTable, detected_new] = eBOSC_episode_postproc_maxbias(cfg_eBOSC, episodesTable, TFR)
        
    # %% remove episodes and part of episodes that fall into 'shoulder'
    
    if len(episodesTable['Trial']) > 0 and cfg_eBOSC['pad.detection_sample']>0:
        episodesTable = eBOSC_episode_rm_shoulder(cfg_eBOSC,detected_new,episodesTable)
    
    # %% if an episode list already exists, append results
    
    if 'episodes' in eBOSC:
        # initialize dictionary entries if not existing
        if not len(eBOSC['episodes']):
            for entry in episodesTable:
                eBOSC['episodes'][entry] = [] 
        # append current results
        for entry in episodesTable:
            episodesTable[entry] = eBOSC['episodes'][entry] + episodesTable[entry]
        
    return episodesTable, detected_new