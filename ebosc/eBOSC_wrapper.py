# -*- coding: utf-8 -*-

def eBOSC_wrapper(cfg_eBOSC, data):
    """Main eBOSC wrapper function. Executes eBOSC subfunctions.
    Inputs: 
              cfg | config structure with cfg.eBOSC field:
                        cfg.eBOSC.F                     | frequency sampling
                        cfg.eBOSC.wavenumber            | wavelet family parameter (time-frequency tradeoff)
                        cfg.eBOSC.fsample               | current sampling frequency of EEG data
                        cfg.eBOSC.pad.tfr_s             | padding following wavelet transform to avoid edge artifacts in seconds (bi-lateral)
                        cfg.eBOSC.pad.detection_s       | padding following rhythm detection in seconds (bi-lateral); 'shoulder' for BOSC eBOSC.detected matrix to account for duration threshold
                        cfg.eBOSC.pad.total_s           | complete padding (WL + shoulder)
                        cfg.eBOSC.pad.background_s      | padding of segments for BG (only avoiding edge artifacts)
                        cfg.eBOSC.threshold.excludePeak | lower and upper bound of frequencies to be excluded during background fit (Hz) (previously: LowFreqExcludeBG HighFreqExcludeBG)
                        cfg.eBOSC.threshold.duration    | vector of duration thresholds at each frequency (previously: ncyc)
                        cfg.eBOSC.threshold.percentile  | percentile of background fit for power threshold
                        cfg.eBOSC.postproc.use          | Post-processing of rhythmic eBOSC.episodes, i.e., wavelet 'deconvolution' (default = 'no')
                        cfg.eBOSC.postproc.method       | Deconvolution method (default = 'MaxBias', FWHM: 'FWHM')
                        cfg.eBOSC.postproc.edgeOnly     | Deconvolution only at on- and offsets of eBOSC.episodes? (default = 'yes')
                        cfg.eBOSC.postproc.effSignal	 | Power deconvolution on whole signal or signal above power threshold? (default = 'PT')
                        cfg.eBOSC.channel               | Subset of channels? (default: [] = all)
                        cfg.eBOSC.trial                 | Subset of trials? (default: [] = all)
                        cfg.eBOSC.trial_background      | Subset of trials for background? (default: [] = all)
              data | input time series data in FieldTrip format with:
                   | .trial field: {trial}(channel x time)
                   | .time field: {trial}(channel x time)
                   | .label field: {channelName}

    Outputs: 
              eBOSC | main eBOSC output structure
                  eBOSC.episodes | table of individual rhythmic episodes (see eBOSC_episode_create)
                  eBOSC.detected | binary matrix of detected time-frequency points (prior to episode creation)
                  eBOSC.pepisode | temporal average of detected rhythms (prior to episode creation)
                  eBOSC.detected_ep | binary matrix of detected time-frequency points (following episode creation)
                  eBOSC.abundance_ep | temporal average of detected rhythms (following episode creation)
              cfg | config structure
    """

    import pandas as pd
    import numpy as np
    
    # %% get list of channel names (very manual solution, replace if possible)

    channelNames = list(data.columns.values)
    channelNames.remove('time')
    channelNames.remove('condition')
    channelNames.remove('epoch')

    # %% define some defaults for included channels and trials, if not specified
    
    if not cfg_eBOSC['channel']:
        cfg_eBOSC['channel'] = channelNames # list of channel names
    
    if not cfg_eBOSC['trial']:
        # remember to count trial 1 as zero
        cfg_eBOSC['trial'] = list(np.arange(0,len(pd.unique(data['epoch']))))
    else: # this ensures the zero count
        cfg_eBOSC['trial'] = list(np.array(cfg_eBOSC['trial']) - 1)
        
    if not cfg_eBOSC['trial_background']:
        cfg_eBOSC['trial_background'] = list(np.arange(0,len(pd.unique(data['epoch']))))
    else: # this ensures the zero count
        cfg_eBOSC['trial_background'] = list(np.array(cfg_eBOSC['trial_background']) - 1)

    # %% calculate the sample points for paddding
    
    cfg_eBOSC['pad.tfr_sample'] = int(cfg_eBOSC['pad.tfr_s'] * cfg_eBOSC['fsample'])
    cfg_eBOSC['pad.detection_sample'] = int(cfg_eBOSC['pad.detection_s'] * cfg_eBOSC['fsample'])
    cfg_eBOSC['pad.total_s'] = cfg_eBOSC['pad.tfr_s'] + cfg_eBOSC['pad.detection_s']
    cfg_eBOSC['pad.total_sample'] = int(cfg_eBOSC['pad.tfr_sample'] + cfg_eBOSC['pad.detection_sample'])
    cfg_eBOSC['pad.background_sample'] = int(cfg_eBOSC['pad.tfr_sample'])
    
    # %% calculate time vectors (necessary for preallocating data frames)
    
    n_trial = len(cfg_eBOSC['trial'])
    n_freq = len(cfg_eBOSC['F'])
    n_time_total = len(pd.unique(data.loc[data['epoch']==0, ('time')]))
    # create a new time vector that is non-continuous and starts at zero
    cfg_eBOSC['time.time_total'] = np.arange(0, 1/cfg_eBOSC['fsample']*(n_time_total) , 1/cfg_eBOSC['fsample'])
    # get timing and info for post-TFR padding removal
    tfr_time2extract = np.arange(cfg_eBOSC['pad.tfr_sample']+1, n_time_total-cfg_eBOSC['pad.tfr_sample']+1,1)
    cfg_eBOSC['time.time_tfr'] = cfg_eBOSC['time.time_total'][tfr_time2extract]
    n_time_tfr = len(cfg_eBOSC['time.time_tfr'])
    # get timing and info for post-detected padding removal
    det_time2extract = np.arange(cfg_eBOSC['pad.detection_sample']+1, n_time_tfr-cfg_eBOSC['pad.detection_sample']+1,1)
    cfg_eBOSC['time.time_det'] = cfg_eBOSC['time.time_total'][det_time2extract]
    n_time_det = len(cfg_eBOSC['time.time_det'])
        
    # %% preallocate data frames

    eBOSC = {};
    eBOSC['static.bg_pow'] = pd.DataFrame(columns=cfg_eBOSC['F'])
    eBOSC['static.bg_log10_pow'] = pd.DataFrame(columns=cfg_eBOSC['F'])    
    eBOSC['static.pv'] = pd.DataFrame(columns=['slope', 'intercept'])
    eBOSC['static.mp'] = pd.DataFrame(columns=cfg_eBOSC['F'])    
    eBOSC['static.pt'] = pd.DataFrame(columns=cfg_eBOSC['F'])   
    
    # Multiindex for channel x trial x frequency x time
    arrays = np.array([cfg_eBOSC['channel'],cfg_eBOSC['trial'],cfg_eBOSC['F'], cfg_eBOSC['time.time_det']],dtype=object)
    #tuples = list(zip(*arrays))
    names=["channel", "trial", "frequency", "time"]
    index=pd.MultiIndex.from_product(arrays,names=names)
    nullData=np.zeros(len(arrays[0]) * len(arrays[1]) * len(arrays[2]) * len(arrays[3]) )
    eBOSC['detected'] = pd.DataFrame(data=nullData, index=index)
    eBOSC['detected_ep'] = pd.DataFrame(data=nullData, index=index)
    del nullData, index
    
    arrays = np.array([cfg_eBOSC['channel'],cfg_eBOSC['trial'],cfg_eBOSC['F']],dtype=object)
    names=["channel", "trial", "frequency"]
    index=pd.MultiIndex.from_product(arrays,names=names)
    nullData=np.zeros(len(arrays[0]) * len(arrays[1]) * len(arrays[2]))
    eBOSC['pepisode'] = pd.DataFrame(data=nullData, index=index)
    eBOSC['abundance_ep'] = pd.DataFrame(data=nullData, index=index)
    
    eBOSC['episodes'] = {}

    # %% main eBOSC loop
    
    for channel in cfg_eBOSC['channel']:
        print('Channel: ' + channel + '; Nr. ' + str(cfg_eBOSC['channel'].index(channel)+1) + '/' + str(len(cfg_eBOSC['channel'])))
        cfg_eBOSC['tmp_channelID'] = cfg_eBOSC['channel'].index(channel)
        cfg_eBOSC['tmp_channel'] = channel
                
        # %% Step 1: time-frequency wavelet decomposition for whole signal to prepare background fit

        from BOSC import BOSC_tf 

        n_trial = len(cfg_eBOSC['trial'])
        n_freq = len(cfg_eBOSC['F'])
        n_time = len(pd.unique(data.loc[data['epoch']==0, ('time')]))
        TFR = np.zeros((n_trial, n_freq, n_time))
        TFR[:] = np.nan
        for trial in cfg_eBOSC['trial']:
            eegsignal = data.loc[data['epoch']==trial, (channel)]
            F = cfg_eBOSC['F']
            Fsample = cfg_eBOSC['fsample']
            wavenumber = cfg_eBOSC['wavenumber']
            [TFR[trial,:,:], tmp, tmp] = BOSC_tf(eegsignal,F,Fsample,wavenumber)
            del eegsignal,F,Fsample,wavenumber,tmp
            
        # %% plot example time-frequency spectrograms (only for intuition/debugging) 
        # assumes that multiple trials are present
        
        # import matplotlib.pyplot as plt
        # plt.imshow(TFR[0,:,:], extent=[0, 1, 0, 1])
        # plt.imshow(TFR[:,:,:].mean(axis=0), extent=[0, 1, 0, 1])
        # plt.imshow(TFR[:,:,:].mean(axis=1), extent=[0, 1, 0, 1])
        # plt.imshow(TFR[:,:,:].mean(axis=2), extent=[0, 1, 0, 1])
                
        # %% Step 2: robust background power fit (see 2020 NeuroImage paper)
       
        from eBOSC import eBOSC_getThresholds  
       
        if cfg_eBOSC['threshold.IRASA'] == 'yes':
            # alternative background fit via IRASA (not implemented yet)
            print('IRASA has not been implemented yet (= FieldTrip MATLAB code)')
            print('Using non-IRASA default')
            # [frac, cfg] = getAperiodicSpectrum_IRASA(data, cfg);
            # [eBOSC, pt, dt] = eBOSC_getThresholds(cfg, TFR, eBOSC, frac); 
            [eBOSC, pt, dt] = eBOSC_getThresholds(cfg_eBOSC, TFR, eBOSC)
        else:
            [eBOSC, pt, dt] = eBOSC_getThresholds(cfg_eBOSC, TFR, eBOSC)
         
        # %% application of thresholds to single trials

        for trial in cfg_eBOSC['trial']:
            print('Trial Nr. ' + str(trial+1) + '/' + str(len(cfg_eBOSC['trial'])))
            # encode current trial ID for later
            cfg_eBOSC['tmp_trialID'] = trial
            # trial ID in the intuitive convention
            cfg_eBOSC['tmp_trial'] = cfg_eBOSC['trial'].index(trial)+1

            # get wavelet transform for single trial
            # tfr padding is removed to avoid edge artifacts from the wavelet
            # transform. Note that a padding fpr detection remains attached so that there
            # is no problems with too few sample points at the edges to
            # fulfill the duration criterion.         
            time2extract = np.arange(cfg_eBOSC['pad.tfr_sample']+1, TFR.shape[2]-cfg_eBOSC['pad.tfr_sample']+1,1)
            TFR_ = np.transpose(TFR[trial,:,time2extract],[1,0])
            
            # %% Step 3: detect rhythms and calculate Pepisode

            from BOSC import BOSC_detect
            # The next section applies both the power and the duration
            # threshold to detect individual rhythmic segments in the continuous signals.
            detected = np.zeros((TFR_.shape))
            for f in range(len(cfg_eBOSC['F'])):
                detected[f,:] = BOSC_detect(TFR_[f,:],pt[f],dt[f][0],cfg_eBOSC['fsample']);

            # remove padding for detection (matrix with padding required for refinement)
            time2encode = np.arange(cfg_eBOSC['pad.detection_sample'], detected.shape[1]-cfg_eBOSC['pad.detection_sample'],1)
            eBOSC['detected'].loc[(channel, trial)] = np.reshape(detected[:,time2encode],[-1,1])

            # encode pepisode of detected rhythms (optional)
            eBOSC['pepisode'].loc[(channel, trial)] = eBOSC['detected'].loc[(channel, trial)].mean(level=['frequency']).values
            
            # %% Step 4 (optional): create table of separate rhythmic episodes

            from eBOSC import eBOSC_episode_create
            [episodes, detected_ep] = eBOSC_episode_create(cfg_eBOSC,TFR_,detected,eBOSC)
            # insert detected episodes into episode structure
            eBOSC['episodes'] = episodes
            
            # remove padding for detection (already done for eBOSC.episodes)
            time2encode = np.arange(cfg_eBOSC['pad.detection_sample'], detected_ep.shape[1]-cfg_eBOSC['pad.detection_sample'],1)
            eBOSC['detected_ep'].loc[(channel, trial)] = np.reshape(detected_ep[:,time2encode],[-1,1])

            # encode abundance of eBOSC.episodes (optional)
            eBOSC['abundance_ep'].loc[(channel, trial)] = eBOSC['detected_ep'].loc[(channel, trial)].mean(level=['frequency']).values

            # Supplementary Plot: original eBOSC.detected vs. sparse episode power
            detected_cur = eBOSC['detected_ep'].loc[(channel, trial)]
            detected_cur = detected_cur.pivot_table(index=['frequency'], columns='time')
            
# =============================================================================
#             fig, axes = plt.subplots(nrows=2, ncols=1)
#             detected_cur = eBOSC['detected_ep'].loc[(channel, trial)]
#             detected_cur = detected_cur.pivot_table(index=['frequency'], columns='time')
#             curPlot = detected_cur*TFR_[:,time2encode]
#             axes[0].imshow(curPlot, aspect='auto')
#             detected_cur = eBOSC['detected'].loc[(channel, trial)]
#             detected_cur = detected_cur.pivot_table(index=['frequency'], columns='time')
#             curPlot = detected_cur*TFR_[:,time2encode]
#             axes[1].imshow(curPlot, aspect='auto')
# =============================================================================

    # %% return dictionaries back to caller script
    return eBOSC, cfg_eBOSC