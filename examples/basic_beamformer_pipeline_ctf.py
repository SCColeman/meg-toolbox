#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A quick and easy route to beamformer power maps and parcellated data.

@author: Sebastian C. Coleman
"""

import mne
from glob import glob
import os.path as op
import os
import sys
from meg_toolbox import preprocess, inverse, forward
import pandas as pd

# paths
root = '/path/to/your/project/root'
data_path = op.join(root, 'data')   # I keep my raw data in this pseudo-BIDS format e.g. /data/sub-01/sub-01.ds
deriv_path = op.join(root, 'derivatives')    # here is where I save anything processed e.g. /deriv/sub-01/sub-01_preproc-raw.fif
subjects_dir = op.join(root, 'subjects_dir')   # FreeSurfer subjects_dir, e.g. /subjects_dir/sub-01   (consistent naming with above helps)

# load raw, unprocessed data (CTF in this case)
subject = 'sub-01'
fname = op.join(data_path, subject, subject + '.ds')
raw = mne.io.read_raw_ctf(fname, preload=True)

#%% preprocessing

raw = preprocess.basic_preprocess(raw, 
                                  grad_comp=3, # gradient compensation - CTF only
                                  picks='mag',   # throw away all other channels
                                  l_freq=1,    
                                  h_freq=100, 
                                  notch=60,
                                  resample_sfreq=250,
                                  )
raw = preprocess.remove_noisy_channels(raw, f_band=(20,100), z_threshold=10)
raw = preprocess.annotate_bad_segments(raw, f_band=(20,100), z_threshold=8)

#%% manual coreg - this varies hugely by system so no attempt to automate this.
# typically involves marking fiducials on mri to define the coordinate space,
# then registering the sensors to the mri based on either fiducial coils or 
# headshape points or both.

coreg = mne.gui.coregistration(inst=fname, subject=subject, 
                               subjects_dir=subjects_dir, block=True)
trans = coreg.coreg.trans

#%% forward model

bem = forward.make_conduction_model(subject, subjects_dir)
src = forward.make_volume_source_space(subject, subjects_dir, grid_spacing=8)
fwd = forward.make_forward_model(raw, trans, bem, src)

#%% beamform induced power

stc, stc_img = inverse.contrast_beamformer(raw,
                                           src, 
                                           fwd,
                                           f_lims=(13,30),
                                           active_event_id='Button',   # the name/ID of the trigger - not it's value
                                           baseline_event_id='Button',
                                           epoch_window=(-0.5, 1),
                                           active_window=(0.3, 0.8),   # beta rebound
                                           baseline_window=(-0.25, 0.25)   # beta desync
                                           )
stc.plot(src, subject, subjects_dir, clim={'type': 'percent', 'pos_lims': [95, 97, 100]})

#%% beamform to parcellation
# The resulting "source_raw" object is ready to perform atlas-based power
# or connectivity analyses

# parcellation - any 3D atlas nifti and corresponding list of names
parcellation_fname = '/d/gmi/1/sebastiancoleman/atlases/giles38_3D.nii.gz'
parcel_names = list(pd.read_csv('/d/gmi/1/sebastiancoleman/atlases/giles38_names.csv').to_numpy()[:,0])

filters = inverse.calculate_beamformer_weights(raw, fwd, reg=0.05)
source_raw, sub_atlas = inverse.parcel_beamformer(raw,
                                        filters,
                                        src,
                                        subject,
                                        subjects_dir,
                                        parcellation_fname,
                                        parcel_names,
                                        method='pca',
                                        orthogonalise=True,
                                        )
