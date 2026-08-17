#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inverse modelling for MEG data.

@author: sebastiancoleman
"""

import os.path as op
from nilearn import image, datasets, plotting
import mne
from sklearn.decomposition import PCA
import numpy as np
from nibabel.affines import apply_affine
from tqdm import tqdm
from scipy.stats import zscore, kurtosis
from mne.transforms import compute_volume_registration, apply_volume_registration
from mne_connectivity import symmetric_orth
import pickle


def calculate_beamformer_weights(raw, fwd, reg=0.05):
    
    """
    Calculate beamformer weights from data and forward model.
    Arguments suitable for most cases.
    """
    
    cov = mne.compute_raw_covariance(raw, reject_by_annotation=True)
    filters = mne.beamformer.make_lcmv(
            raw.info,
            fwd,
            cov,
            reg=reg,
            noise_cov=None,
            pick_ori='max-power',
            weight_norm='nai',
            rank=None,
            reduce_rank=True,
            verbose=False,
            )
    return filters


def parcel_beamformer(raw, 
                      filters, 
                      fwd, 
                      fs_subject, 
                      subjects_dir, 
                      parcellation_fname, 
                      parcel_names, 
                      method='centroid',
                      orthogonalise=True,
                      ):
    """
    Use calculated beamformer weights to construct parcel-level raw object.

    """
    
    # load subject mri and template
    mri = image.load_img(op.join(subjects_dir, fs_subject, 'mri', 'brainmask.mgz'))
    template = datasets.load_mni152_template()
    
    # load transform from MNI space to subject MRI space
    transforms_path = op.join(subjects_dir, fs_subject, 'subject_mni_transforms')
    if not op.exists(transforms_path):
        raise Exception("MNI->MRI transforms do not exist. Run transform.compute_subject_mni_transforms.")
    reg_affine = np.load(op.join(transforms_path, 'mni2mri_aff.npy'))
    with open(op.join(transforms_path, 'mni2mri_sdr.pkl'), 'rb') as file:
        sdr_morph = pickle.load(file)
        
    # convert atlas to subject MRI space
    atlas_img = image.load_img(parcellation_fname)
    atlas_mri = apply_volume_registration(atlas_img, mri, reg_affine, sdr_morph, interpolation='nearest')
    indices = np.unique(atlas_mri.get_fdata())[1:]

    # get src coords IN MRI (surface RAS) SPACE 
    src_coords = fwd['src'][0]['rr'][fwd['src'][0]['inuse']==1]
    head_mri_t = mne.transforms.invert_transform(fwd['mri_head_t'])
    src_coords = mne.transforms.apply_trans(head_mri_t, src_coords)
    src_coords *= 1000   # needs to be in mm
    
    # convert coords from surface RAS to scanner RAS
    torig = mri.header.get_vox2ras_tkr()
    surf2scanner = mri.affine @ np.linalg.inv(torig)
    coords_scanner = apply_affine(surf2scanner, src_coords)
    scanner_vox = np.round(apply_affine(np.linalg.inv(mri.affine), coords_scanner)).astype(int)

    # for each source, get a corresponding parcel label
    atlas_data = atlas_mri.get_fdata()
    src_atlas = []
    for vox in scanner_vox:
        src_ind = atlas_data[vox[0], vox[1], vox[2]]
        src_atlas.append(src_ind)
    src_atlas = np.array(src_atlas)

    # Take 1st PC of sources within each parcel
    data = raw.get_data()
    weights = filters['weights']
    parcel_data = np.zeros((len(parcel_names), data.shape[1]))
    for parcel in tqdm(range(len(parcel_names)), desc='parcellating...'):
        
        if method=='pca':

            # beamform to all voxels in parcel
            parcel_ind = src_atlas==indices[parcel]
            parcel_weights = weights[parcel_ind,:]
            parcel_stc = parcel_weights @ data
            
            # take first pc 
            pca = PCA(1).fit(parcel_stc.T)
            parcel_stc = pca.transform(parcel_stc.T).T
            parcel_data[parcel,:] = parcel_stc.copy()
            
        elif method=='centroid':
            
            # find centroid of parcel
            parcel_ind = src_atlas==indices[parcel]
            parcel_coords = src_coords[parcel_ind]
            centroid_coord = np.mean(parcel_coords, 0)
            
            # find closest source index
            src_dists = np.linalg.norm(src_coords - centroid_coord, axis=-1)
            src_centroid = np.argmin(src_dists)
            
            # beamform to centroid
            centroid_weights = weights[src_centroid,:]
            centroid_stc = centroid_weights @ data
            parcel_data[parcel,:] = centroid_stc
        
    if orthogonalise:
        parcel_data = symmetric_orth(parcel_data)
    
    # make into source-level raw
    info = mne.create_info(parcel_names, raw.info['sfreq'], 'misc')
    atlas_raw = mne.io.RawArray(parcel_data, info)
    atlas_raw.set_meas_date(raw.info['meas_date'])
    atlas_raw.set_annotations(raw.annotations)
    if orthogonalise:
        atlas_raw.apply_function(zscore, picks='all')
    
    return atlas_raw, atlas_mri


def extract_parcel_peak_VEs(raw, 
                      filters, 
                      fwd, 
                      fs_subject, 
                      subjects_dir, 
                      parcellation_fname, 
                      parcel_names, 
                      stat_img,
                      mode='abs',
                      ):
    """
    Use calculated beamformer weights to construct parcel-level raw object.

    """
    
    # load subject mri and template
    mri = image.load_img(op.join(subjects_dir, fs_subject, 'mri', 'brainmask.mgz'))
    template = datasets.load_mni152_template()
    
    # load transform from MNI space to subject MRI space
    transforms_path = op.join(subjects_dir, fs_subject, 'subject_mni_transforms')
    if not op.exists(transforms_path):
        raise Exception("MNI->MRI transforms do not exist. Run transform.compute_subject_mni_transforms.")
    reg_affine = np.load(op.join(transforms_path, 'mni2mri_aff.npy'))
    with open(op.join(transforms_path, 'mni2mri_sdr.pkl'), 'rb') as file:
        sdr_morph = pickle.load(file)
        
    # convert atlas to subject MRI space
    atlas_img = image.load_img(parcellation_fname)
    atlas_mri = apply_volume_registration(atlas_img, mri, reg_affine, sdr_morph, interpolation='nearest')
    indices = np.unique(atlas_mri.get_fdata())[1:]

    # get src coords IN MRI (surface RAS) SPACE 
    src_coords = fwd['src'][0]['rr'][fwd['src'][0]['inuse']==1]
    head_mri_t = mne.transforms.invert_transform(fwd['mri_head_t'])
    src_coords = mne.transforms.apply_trans(head_mri_t, src_coords)
    src_coords *= 1000   # needs to be in mm

    # convert coords from surface RAS to scanner RAS
    torig = mri.header.get_vox2ras_tkr()
    surf2scanner = mri.affine @ np.linalg.inv(torig)
    coords_scanner = apply_affine(surf2scanner, src_coords)
    scanner_vox = np.round(apply_affine(np.linalg.inv(mri.affine), coords_scanner)).astype(int)

    # resample statistical map to mri (/ source space)
    atlas_data = atlas_mri.get_fdata()
    stat_resampled = image.resample_to_img(stat_img, mri)
    if len(stat_resampled.shape)==4:
        stat_data = stat_resampled.get_fdata()[:,:,:,0]
    else:
        stat_data = stat_resampled.get_fdata()
      
    # correct sign depending on mode
    if mode=='abs':
        stat_data = np.abs(stat_data)
    elif mode=='neg':
        stat_data = -stat_data
    elif mode=='pos':
        pass
    else:
        raise Exception("Mode must be either 'pos', 'neg', or 'abs'.")
    
    src_atlas = []
    src_stat = []
    for vox in scanner_vox:
        src_ind = atlas_data[vox[0], vox[1], vox[2]]
        src_atlas.append(src_ind)
        src_val = stat_data[vox[0], vox[1], vox[2]]
        src_stat.append(src_val)
    src_atlas = np.array(src_atlas)
    src_stat = np.array(src_stat)

    # Take statistical peak of each source in atlas region
    data = raw.get_data()
    weights = filters['weights']
    parcel_data = np.zeros((len(parcel_names), data.shape[1]))
    for parcel in tqdm(range(len(parcel_names)), desc='parcellating...'):

        # beamform to all voxels in parcel
        parcel_ind = src_atlas==indices[parcel]
        parcel_max = np.argmax(src_stat * parcel_ind)
        parcel_weights = weights[parcel_max,:]
        parcel_stc = parcel_weights @ data
        parcel_data[parcel,:] = parcel_stc.copy()
        
    # make into source-level raw
    info = mne.create_info(parcel_names, raw.info['sfreq'], 'misc')
    atlas_raw = mne.io.RawArray(parcel_data, info)
    atlas_raw.set_meas_date(raw.info['meas_date'])
    atlas_raw.set_annotations(raw.annotations)
    atlas_raw.apply_function(zscore, picks='all')
    
    return atlas_raw, atlas_mri

def orthogonalise_source_raw(source_raw):
    
    source_raw_orth = mne.io.RawArray(zscore(symmetric_orth(source_raw.get_data()), -1), source_raw.info)
    source_raw_orth.set_annotations(source_raw.annotations)
    
    return source_raw_orth


def kurtosis_beamformer(raw,
                        fwd,
                        src
                        ):
    
    """
    Calculate kurtosis map.
    """
    
    # filter into spikey range
    raw_filt = raw.copy().filter(20, 70, verbose=False).resample(250)
    
    # beamform
    cov = mne.compute_raw_covariance(raw_filt, reject_by_annotation=True)
    filters = calculate_beamformer_weights(raw_filt, fwd, reg=0.02)
    
    stc = mne.beamformer.apply_lcmv_raw(raw_filt, filters, start=0, stop=1)
    data = raw_filt.get_data(reject_by_annotation='omit')
    weights = filters['weights']
    k_map = np.zeros(stc.shape[0])
    for source in tqdm(range(stc.shape[0])):
        source_data = weights[source,:] @ data
        k_map[source] = kurtosis(source_data)
    kurtosis_stc = mne.VolSourceEstimate(np.expand_dims(k_map, 1), stc.vertices, 0, 1)
    kurtosis_img = kurtosis_stc.as_volume(src)
    
    return kurtosis_stc, kurtosis_img, filters


def contrast_beamformer(raw, 
                        events,
                        ids,
                        fwd,
                        f_lims,
                        active_event_id, 
                        baseline_event_id, 
                        epoch_window=(-0.5, 0.5),
                        active_window=(0, 0.5),
                        baseline_window=(-0.5, 0),
                        ):
    
    """
    Beamformer imaging based on an event-related contrast.
    """
    
    # filter data
    raw_filt = raw.copy().filter(f_lims[0], f_lims[1])
    
    # epoch
    if type(active_event_id)==str:
        active_event_id = [active_event_id]
    if type(baseline_event_id)==str:
        baseline_event_id = [baseline_event_id]
    epoch_ids = [ev for ev in active_event_id] + [ev for ev in baseline_event_id]
    epoch_ids = list(np.unique(epoch_ids))
    epochs = mne.Epochs(raw_filt, events, [ids[ev] for ev in epoch_ids], 
                        tmin=epoch_window[0], tmax=epoch_window[1], 
                        preload=True, reject_by_annotation=True,
                        on_missing='ignore')
    
    # calculate common beamformer
    cov = mne.compute_covariance(epochs)
    filters = mne.beamformer.make_lcmv(epochs.info,
                                       fwd,
                                       cov,
                                       reg=0.05,
                                       pick_ori='max-power',
                                       weight_norm='nai',
                                       reduce_rank=True,
                                       rank=None,
                                       )
    
    # make pseudo-T
    cov_act = mne.compute_covariance(epochs[[str(ids[ev]) for ev in active_event_id]], 
                                     tmin=active_window[0], tmax=active_window[1])
    cov_base = mne.compute_covariance(epochs[[str(ids[ev]) for ev in baseline_event_id]], 
                                     tmin=baseline_window[0], tmax=baseline_window[1])
    stc_act = mne.beamformer.apply_lcmv_cov(cov_act, filters)
    stc_base = mne.beamformer.apply_lcmv_cov(cov_base, filters)
    stc = (stc_act - stc_base) / (stc_act + stc_base)
    stc_img = stc.as_volume(fwd['src'])
    
    return stc, stc_img, filters
    