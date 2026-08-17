#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 11:10:02 2026

@author: sebastiancoleman
"""

import os.path as op
import os
from nilearn import image, datasets, plotting
import mne
import numpy as np
from nibabel.affines import apply_affine
from mne.transforms import compute_volume_registration, apply_volume_registration
import pickle
import nibabel as nib


def compute_subject_mni_transforms(fs_subject, subjects_dir, res=2):
    
    # make new directory for transforms
    outdir = op.join(subjects_dir, fs_subject, 'subject_mni_transforms')
    os.makedirs(outdir, exist_ok=True)
    
    # calculate forward transform (subject->MNI)
    moving = image.load_img(op.join(subjects_dir, fs_subject, 'mri', 'brainmask.mgz'))
    static = datasets.load_mni152_template()
    affine_forward, sdr_forward = compute_volume_registration(moving, static, zooms=res)
    
    mri2mni = apply_volume_registration(moving, static, affine_forward, sdr_forward)
    
    # calculate inverse transform (MNI->subject)
    moving = datasets.load_mni152_template()
    static = image.load_img(op.join(subjects_dir, fs_subject, 'mri', 'brainmask.mgz'))
    affine_inverse, sdr_inverse = compute_volume_registration(moving, static, zooms=res)
    
    mni2mri = apply_volume_registration(moving, static, affine_inverse, sdr_inverse)
    
    # save
    np.savetxt(op.join(outdir, 'mri2mni_aff.txt'), affine_forward, fmt='%.18e')
    np.save(op.join(outdir, 'mri2mni_aff.npy'), affine_forward)
    with open(op.join(outdir, 'mri2mni_sdr.pkl'), "wb") as output_file:
        pickle.dump(sdr_forward, output_file)
    nib.save(mri2mni, op.join(outdir, 'mri2mni.nii.gz'))
    
    np.savetxt(op.join(outdir, 'mni2mri_aff.txt'), affine_inverse, fmt='%.18e')
    np.save(op.join(outdir, 'mni2mri_aff.npy'), affine_inverse)
    with open(op.join(outdir, 'mni2mri_sdr.pkl'), "wb") as output_file:
        pickle.dump(sdr_inverse, output_file)
    nib.save(mni2mri, op.join(outdir, 'mni2mri.nii.gz'))
    
    
def apply_transform(img, static, reg_affine, sdr_morph=None):
    
    transformed = apply_volume_registration(img, static, reg_affine, sdr_morph, verbose=False)
    return transformed