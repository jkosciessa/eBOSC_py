#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various helper functions

Created on Fri Feb 19 11:07:23 2021

@author: kosciessa
"""

def find_nearest_value(array, value):
    import numpy as np
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def getTimeFromFTmat(fname, var_name='data'):
    """
    Get original timing from FieldTrip structure
    Solution based on https://github.com/mne-tools/mne-python/issues/2476
    """
    import scipy.io as sio
    import numpy as np
    # load Matlab/Fieldtrip data
    mat = sio.loadmat(fname, squeeze_me=True, struct_as_record=False)
    ft_data = mat[var_name]
    # convert to mne
    n_trial = len(ft_data.trial)
    n_chans, n_time = ft_data.trial[0].shape
    #data = np.zeros((n_trial, n_chans, n_time))
    time = np.zeros((n_trial, n_time))
    for trial in range(n_trial):
        # data[trial, :, :] = ft_data.trial[trial]
        time[trial, :] = ft_data.time[trial]
    return time
