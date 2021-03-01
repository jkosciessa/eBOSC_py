% Prepare files for MNE read-in

addpath('/Users/kosciessa/OneDrive/Dev/eBOSCRepos/eBOSC/external/fieldtrip')
ft_defaults

pn.data = '/Users/kosciessa/OneDrive/Dev/eBOSCRepos/eBOSC_py/data/';
load([pn.data, '1160_rest_EEG_Rlm_Fhl_rdSeg_Art_EC.mat'])

% create timelock data to export
cfg = [];
avg = ft_timelockanalysis(cfg, data);

% export to fif file
fiff_file  = [pn.data, '1160_rest_EEG_Rlm_Fhl_rdSeg_Art_EC.fif'];
fieldtrip2fiff(fiff_file, avg)

% append sampleinfo to .mat (no conditions here)
data.trialinfo = ones(size(data.sampleinfo,1),1);
% change time to be identical across trials
data.time_orig = data.time;
for indTrial = 1:numel(data.time)
    data.time{indTrial} = data.time{1}-data.time{1}(1)
end
save([pn.data, '1160_rest_EEG_Rlm_Fhl_rdSeg_Art_EC.mat'], 'data')