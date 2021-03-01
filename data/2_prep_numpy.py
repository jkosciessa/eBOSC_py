""" Convert the provided example from Fieldtrip to numpy array """

# Note: these packages may not be automatically installed!
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mne import read_evokeds, read_epochs_fieldtrip
from ebosc.helpers import get_project_root
from ebosc.helpers import getTimeFromFTmat

pn = dict()
pn['root']  = os.path.abspath(os.path.dirname(get_project_root()))

# read in evoked structure in native .fif format to get the MNE info class
evoked_path = os.path.join(pn['root'],'data','1160_rest_EEG_Rlm_Fhl_rdSeg_Art_EC.fif')
evoked = read_evokeds(evoked_path)

# import the .mat data, this does not respect variations in timing across epochs
epoch_path = os.path.join(pn['root'],'data','1160_rest_EEG_Rlm_Fhl_rdSeg_Art_EC.mat')
dataMNE = read_epochs_fieldtrip(fname=epoch_path, info=evoked[0].info, data_name='data',trialinfo_column=0)

# retrieve data from MNE structure into pandas data frame
data = dataMNE.to_data_frame(time_format=None, scalings=dict(eeg=1, mag=1, grad=1))

# %% plot data by time (note that all trials falsely share the same epoch time

curEpoch = data[data['epoch']==0]
curEpoch.plot(x ='time', y='Fp1', kind = 'line')
plt.show()

# %% correct the timing to have separate timings for each epoch
original_time = getTimeFromFTmat(epoch_path, var_name='data')

# replace data in dataframe with correct timings
n_trials = len(pd.unique(data['epoch'])) 
for trial in range(n_trials):
    data.loc[data['epoch']==trial, ('time')] = original_time[trial, :]

# %% plot data from all 'trials' (now correct continuous time)

data.plot(x ='time', y='Fp1', kind = 'line')
plt.show()

# %% plot heatmap of time-trials

n_trials = len(pd.unique(data['epoch']))
n_time = len(pd.unique(data.loc[data['epoch']==0, ('time')]))
plot_data = np.zeros((n_trials, n_time))
for trial in range(n_trials):
    plot_data[trial, :] = data.loc[data['epoch']==trial, ('Oz')]
plt.imshow(plot_data, extent=[0, 1, 0, 1])
plt.title('Trial-wise signals at channel Oz')
plt.show()

# %% remove unnecessary channels

del data['A1']
del data['IOR']
del data['LHEOG']
del data['RHEOG']

# %% save example data as csv

numpy_path = os.path.join(pn['root'],'data','1160_rest_EEG_Rlm_Fhl_rdSeg_Art_EC.csv')
data.to_csv(numpy_path)