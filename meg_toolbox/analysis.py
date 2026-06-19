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
    event_name,
    tmin=-0.5,
    tmax=1.0,
    baseline=(-0.5, -0.1),
    picks=None
    ):
    """
    Calculate the evoked response for a given event from raw data.

    """
    if picks is None:
        picks = 'all'

    events, event_ids = mne.events_from_annotations(raw)
    if event_name not in event_ids:
        raise ValueError(f"Event '{event_name}' not found in annotations.")

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_ids[event_name],
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        picks=picks,
        preload=True
    )
    evoked = epochs.average(picks=picks)
    return evoked


def calculate_tfr(
    raw,
    event_name,
    tmin=-0.5,
    tmax=1.0,
    freqs=np.arange(2, 40, 2),
    n_cycles=None,
    baseline=(-0.5, -0.1),
    padding=0.5,
    picks=None
    ):
    """
    Calculate the time-frequency representation (TFR) for a given event.

    """
    if picks is None:
        picks = 'all'
    if n_cycles is None:
        n_cycles = freqs / 2

    events, event_ids = mne.events_from_annotations(raw)
    if event_name not in event_ids:
        raise ValueError(f"Event '{event_name}' not found in annotations.")

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_ids[event_name],
        tmin=tmin - padding,
        tmax=tmax + padding,
        baseline=None,
        picks=picks,
        preload=True
    )

    tfr = epochs.compute_tfr(
        method='multitaper',
        freqs=freqs,
        n_cycles=n_cycles,
        average=True,
        picks=picks
    )

    tfr.crop(tmin, tmax)
    tfr.apply_baseline(baseline, mode='percent')

    return tfr

