#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis to be applied on raw objects (sensor or source).

@author: sebastiancoleman
"""

import mne
import numpy as np

def calculate_evoked(
    raw,
    events, 
    ids,
    event_name,
    tmin=-0.5,
    tmax=1.0,
    baseline=(-0.5, -0.1),
    ):
    """
    Calculate the evoked response for a given event from raw data.

    """
    
    if isinstance(event_name, str):
        event_name = [event_name]

    epochs = mne.Epochs(
        raw,
        events,
        event_id=[ids[name] for name in event_name],
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        picks='all',
        preload=True
    )
    evoked = epochs.average(picks='all')
    return evoked


def calculate_tfr(
    raw,
    events, 
    ids,
    event_name,
    tmin=-0.5,
    tmax=1.0,
    freqs=np.arange(2, 40, 2),
    n_cycles=None,
    baseline=(-0.5, -0.1),
    padding=0.5,
    n_jobs=1,
    ):
    """
    Calculate the time-frequency representation (TFR) for a given event.

    """

    if n_cycles is None:
        n_cycles = freqs / 2

    if isinstance(event_name, str):
        event_name = [event_name]

    epochs = mne.Epochs(
        raw,
        events,
        event_id=[ids[name] for name in event_name],
        tmin=tmin - padding,
        tmax=tmax + padding,
        baseline=None,
        picks='all',
        preload=True
    )

    tfr = epochs.compute_tfr(
        method='multitaper',
        freqs=freqs,
        n_cycles=n_cycles,
        average=True,
        picks='all',
        n_jobs=n_jobs,
    )

    tfr.crop(tmin, tmax)
    tfr.apply_baseline(baseline, mode='percent')

    return tfr


def zscore_excluding_bads(raw):
    
    raw = raw.copy()
    
    # extract data
    data = raw.get_data(picks='all')
    clean_data = raw.get_data(picks='all', reject_by_annotation='omit')
    
    # calculate clean z-score
    mean = np.mean(clean_data, axis=-1)
    std = np.std(clean_data, axis=-1)
    
    # apply z-score to full data
    clean_z = ((data.T - mean) / std).T
    
    # make into raw
    raw_z = mne.io.RawArray(clean_z, raw.info)
    raw_z.set_meas_date(raw.info['meas_date'])
    raw_z.set_annotations(raw.annotations)
    
    return raw_z



