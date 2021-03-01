""" Various helper functions
"""

import numpy as np
import scipy.io as sio
from pathlib import Path

def find_nearest_value(array, value):
    """Find nearest value and index of float in array
    Parameters:
    array : Array of values [1d array]
    value : Value of interest [float]
    Returns:
    array[idx] : Nearest value [1d float]
    idx : Nearest index [1d float]
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def getTimeFromFTmat(fname, var_name='data'):
    """
    Get original timing from FieldTrip structure
    Solution based on https://github.com/mne-tools/mne-python/issues/2476
    """
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
        # Note that this indexes time_orig in the adapted structure
        time[trial, :] = ft_data.time_orig[trial]
    return time

def get_project_root() -> Path:
    return Path(__file__).parent