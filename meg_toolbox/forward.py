#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forward modelling for MEG data.

@author: sebastiancoleman
"""
    
import os.path as op
import mne
from nilearn import image


def mark_mri_fiducials(fs_subject, subjects_dir):
    """
    Launch the MNE coregistration GUI to mark MRI fiducials.

    This is an interactive step. In the GUI:
    - Mark NAS, LPA, RPA on the MRI
    - Click "Lock fiducials"
    - Click "Save MRI fiducials"

    """
    print(
        "Launching MNE coregistration GUI.\n"
        "Mark MRI fiducials, then select 'Lock fiducials' and 'Save MRI fids'."
    )
    mne.gui.coregistration(
        subject=fs_subject,
        subjects_dir=subjects_dir,
    )


def view_mri_to_find_fiducials(fs_subject, subjects_dir):
    """
    Open an interactive orthographic MRI viewer to help locate fiducials.

    """
    mri_fname = op.join(
        subjects_dir, fs_subject, "mri", "orig", "001.mgz"
    )
    img = image.load_img(mri_fname)
    viewer = img.orthoview()
    return viewer


def compute_coreg_trans(raw, fs_subject, subjects_dir):
    """
    Compute MEG–MRI coregistration using fiducials only.

    Assumes MRI fiducials have already been saved using the MNE GUI.

    """
    coreg = mne.coreg.Coregistration(
        info=raw.info,
        subject=fs_subject,
        subjects_dir=subjects_dir,
    )
    coreg.fit_fiducials()
    return coreg.trans


def plot_coregistration(
    raw,
    trans,
    fs_subject,
    subjects_dir,
    meg=True,
    dig=True,
    mri_fiducials=True,
    ):
    """
    Visualise MEG–MRI alignment.

    """
    view_kwargs = dict(azimuth=45, elevation=90, distance=0.8, focalpoint=(0.0, 0.0, 0.05))
    fig = mne.viz.plot_alignment(
        info=raw.info,
        trans=trans,
        subject=fs_subject,
        subjects_dir=subjects_dir,
        meg=meg,
        dig=dig,
        coord_frame='head',
        mri_fiducials=mri_fiducials,
        surfaces='head-dense',
    )
    mne.viz.set_3d_view(fig, **view_kwargs)
    return fig
    
def make_conduction_model(fs_subject, subjects_dir):
    """
    Make a single shell conduction model.

    """
    conductivity = (0.3,)
    model = mne.make_bem_model(
            subject=fs_subject, ico=4,
            conductivity=conductivity,
            subjects_dir=subjects_dir,
            verbose=False)
    bem = mne.make_bem_solution(model, verbose=False)
    
    return bem

def make_volume_source_space(fs_subject, subjects_dir, grid_spacing=5):
    """
    Make a volumetric (grid) source space.

    """
    surface = op.join(subjects_dir, fs_subject, 'bem', 'inner_skull.surf')
    src = mne.setup_volume_source_space(
        fs_subject,
        pos=grid_spacing,
        surface=surface,
        subjects_dir=subjects_dir,
        verbose=False,
        )
    
    return src

def make_forward_model(raw, trans, bem, src):
    """
    Calculate the forward model.

    """
    fwd = mne.make_forward_solution(
        raw.info,
        trans=trans,
        src=src,
        bem=bem,
        meg=True,
        eeg=False,
        verbose=False,
        )
    
    return fwd

    
