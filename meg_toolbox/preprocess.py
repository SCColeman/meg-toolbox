#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing MEG data.

Author: Sebastian Coleman
"""

import numpy as np
import mne
from datetime import datetime

def is_outlier(data, threshold=3.5, only_positive=True):
    """
    Identify outliers using a median absolute deviation (MAD) criterion.
    """
    data = np.asarray(data)
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    if mad == 0:
        outliers = np.zeros_like(data, dtype=bool)
    else:
        modified_z = 0.6745 * (data - median) / mad
        if only_positive==True:
            outliers = modified_z > threshold
        else:
            outliers = np.abs(modified_z) > threshold

    return outliers

def basic_preprocess(
    raw,
    grad_comp=13,
    picks="mag",
    l_freq=1.0,
    h_freq=100.0,
    notch=60.0,
    notch_width=1.0,
    resample_sfreq=250.0
):
    """
    Apply basic preprocessing steps to a Raw object.
    """
    
    print('Preprocessing...')
    raw = raw.copy()

    if grad_comp is not None:
        raw.apply_gradient_compensation(grad_comp, verbose=False)

    if picks is not None:
        raw.pick(picks, verbose=False)

    if l_freq is not None or h_freq is not None:
        raw.filter(l_freq, h_freq, verbose=False)

    if notch is not None:
        raw.notch_filter(notch, notch_widths=notch_width, verbose=False)

    if resample_sfreq is not None:
        raw.resample(resample_sfreq, verbose=False)

    print('Finished!')

    return raw

def remove_noisy_channels(
    raw, 
    f_band=(20, 100),
    z_threshold=5.0,
    drop=True
    ):
    """
    Detect and mark bad channels using signal variance.
    """
    
    print('Detecting bad channels...')
    raw = raw.copy()
    noise = raw.copy().filter(*f_band, verbose=False).get_data(reject_by_annotation='omit')
    
    variances = np.var(noise, 1)
    outliers = is_outlier(variances, z_threshold)
    bad_chans = [raw.ch_names[i] for i in range(len(raw.ch_names)) if outliers[i]]
    raw.info['bads'] += bad_chans
    raw.info['bads'] = [str(inst) for inst in raw.info['bads']]
    
    if drop==True:
        raw.drop_channels(raw.info['bads'])
    
    print('Finished!')
    print(f'Removed {len(bad_chans)} channels')
    
    return raw

def remove_noisy_channels_with_hfc(
        raw,
        f_band=(1, 100),
        z_threshold=5.0,
        ):
    """
    Apply HFC and remove noisy channels in one step - useful for OPM data.
    These steps are combined as it is important that bad channels are
    removed prior to HFC, but HFC is often required to "see" the bad channels.
    """
    
    print('Removing noisy channels and applying HFC...')
    raw = raw.copy()
    
    # apply HFC to temporary raw to find bad channels
    raw_temp = raw.copy()
    projs = mne.preprocessing.compute_proj_hfc(raw_temp.info, order=1)
    raw_temp.add_proj(projs).apply_proj(verbose="error")
    noise = raw_temp.copy().filter(*f_band, verbose=False).get_data(reject_by_annotation='omit')
    variances = np.var(noise, 1)
    outliers = is_outlier(variances, z_threshold)
    bad_chans = [raw_temp.ch_names[i] for i in range(len(raw_temp.ch_names)) if outliers[i]]
    
    # remove bad channels and apply HFC
    raw.drop_channels(bad_chans)
    projs = mne.preprocessing.compute_proj_hfc(raw.info, order=1, verbose=False)
    raw.add_proj(projs).apply_proj(verbose="error")
    print('Finished!')
    
    return raw  
    
def annotate_bad_segments(
    raw,
    f_band=(20, 100),
    segment_length=0.5,
    step=0.25,
    z_threshold=5.0,
    description="BAD_segment"
    ):
    """
    Detect and annotate noisy segments using mean signal variance.
    """
    
    print('Detecting bad segments...')
    raw = raw.copy()

    sfreq = raw.info["sfreq"]
    seg_len = int(segment_length * sfreq)
    step_len = int(step * sfreq)
    noise = raw.copy().filter(*f_band, verbose=False, picks='all').get_data()

    variances = [
        np.mean(np.var(noise[:, i:i + seg_len], axis=1))
        for i in range(0, noise.shape[1] - seg_len + 1, step_len)
    ]
    variances = np.asarray(variances)
    outliers = is_outlier(variances, z_threshold)

    annotations = raw.annotations
    times = raw.times

    for idx, start in enumerate(range(0, noise.shape[1] - seg_len, step_len)):
        if outliers[idx]:
            onset = times[start]
            duration = segment_length
            annotations += mne.Annotations(
                onset=onset,
                duration=duration,
                description=description,
                orig_time=annotations.orig_time,
            )

    raw.set_annotations(annotations, verbose=False)

    print('Finished!')

    return raw

def annotate_high_amplitude_segments(
    raw,
    segment_length=0.5,
    step=0.25,
    threshold=1e-12,
    description="BAD_segment"
    ):
    """
    Detect and annotate noisy segments using maximum amplitude.
    """
    
    print('Detecting bad segments...')
    raw = raw.copy()

    sfreq = raw.info["sfreq"]
    seg_len = int(segment_length * sfreq)
    step_len = int(step * sfreq)
    noise = raw.get_data()

    amplitudes = [
        np.max(np.abs(noise[:, i:i + seg_len]))
        for i in range(0, noise.shape[1] - seg_len + 1, step_len)
    ]
    amplitudes = np.asarray(amplitudes)
    outliers = amplitudes > threshold

    annotations = raw.annotations
    times = raw.times

    for idx, start in enumerate(range(0, noise.shape[1] - seg_len, step_len)):
        if outliers[idx]:
            onset = times[start]
            duration = segment_length
            annotations += mne.Annotations(
                onset=onset,
                duration=duration,
                description=description,
                orig_time=annotations.orig_time,
            )

    raw.set_annotations(annotations, verbose=False)

    print('Finished!')

    return raw


def fit_ica_manual(raw, n_components=20, random_state=42, method="fastica"):
    """
    Fit ICA decomposition fully manually. User selects artefactual components interactively.

    """
    
    print('Fitting ICA...')
    raw = raw.copy()
    
    # Fit ICA
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        random_state=random_state,
        method=method
    )
    ica.fit(raw, verbose=False)

    # Show interactive plots, pausing until closed
    print("Inspect ICA components. Close the figures when done.")
    ica.plot_components(nrows=int(np.floor(np.sqrt(n_components))), ncols='auto')
    ica.plot_sources(raw, block=True)

    # Apply ICA
    ica.apply(raw)

    return raw, ica
