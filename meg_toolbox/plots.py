#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting functions.

@author: sebastiancoleman
"""

from nilearn import image, datasets, plotting, surface
import nibabel as nib
import matplotlib as mpl
import pyvista as pv
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import statsmodels.api as sm
from scipy.stats import pearsonr
import numpy as np
from matplotlib import pyplot as plt
import os.path as op
import mne

def make_atlas_nifti(atlas, values):
    
    """
    Place a vector of region values into a deterministic 3D atlas for plotting.
    """
    
    atlas_img = image.load_img(atlas)
    mni = datasets.load_mni152_template()
    atlas_data = atlas_img.get_fdata()
    atlas_new = np.zeros(np.shape(atlas_data))
    indices = np.unique(atlas_data[atlas_data>0])
    for reg in range(len(values)):
        reg_mask = atlas_data == indices[reg]
        atlas_new[reg_mask] = values[reg]
    new_img = nib.Nifti1Image(atlas_new, atlas_img.affine, atlas_img.header)
    img = image.resample_img(new_img, mni.affine, mni.shape)
    img = image.new_img_like(img, (mni.get_fdata()>0)*img.get_fdata())
    
    return img

def make_4d_atlas_nifti(atlas, values):
    
    """
    Equivalent of make_atlas_nifti but for 4D probabilistic atlases.
    """
    
    atlas_img = image.load_img(atlas)

    # load fsaverage and atlas   
    mni = datasets.load_mni152_template()
    atlas_data = atlas_img.get_fdata()

    # place values in each parcel region
    regs = []
    for reg in range(atlas_data.shape[-1]):
        atlas_reg = atlas_data[:,:,:,reg]
        atlas_reg /= np.max(atlas_reg)
        regs.append(atlas_reg * values[reg])
    atlas_new = np.sum(regs, 0)

    # make image from new atlas data
    new_img = nib.Nifti1Image(atlas_new, atlas_img.affine, atlas_img.header)
    
    # interpolate image
    img_interp = image.resample_img(new_img, mni.affine)
    
    return img_interp

def surface_brain_plot(img, subjects_dir, surf='inflated', cmap='cold_hot', symmetric=True, 
                       threshold=0, fade=True, cbar_label=None, figsize=(10,7)):
    
    """
    Take a 3D nifti image in MNI space and project onto fsaverage surface
    for a cool looking brain plot.
    """
    
    # make MNE stc out of nifti
    lh_surf = op.join(subjects_dir, 'fsaverage', 'surf', 'lh.pial')
    lh = surface.vol_to_surf(img, lh_surf)
    rh_surf = op.join(subjects_dir, 'fsaverage', 'surf', 'rh.pial')
    rh = surface.vol_to_surf(img, rh_surf)
    data = np.hstack([lh, rh])
    vertices = [np.arange(len(lh)), np.arange(len(rh))]
    stc = mne.SourceEstimate(data, vertices, tmin=0, tstep=1)

    # set up axes
    fig = plt.figure(figsize=(5.2, 4))
    ax1 = fig.add_axes([0.05, 0.58, 0.45, 0.40])  # top-left
    ax2 = fig.add_axes([0.5, 0.58, 0.45, 0.40])  # top-right
    ax3 = fig.add_axes([0.05, 0.17, 0.45, 0.38])  # bottom-left
    ax4 = fig.add_axes([0.5, 0.17, 0.45, 0.38])  # bottom-right
    cax = fig.add_axes([0.25, 0.12, 0.5, 0.03]) # colorbar ax
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor('none')
        ax.axis(False)
        
    # set up threshold
    if symmetric:
        vmax = np.max(np.abs(data))
        vmin = -vmax
        mid = threshold + ((vmax-threshold)/3)
        if fade:
            clim = {'kind': 'value', 'pos_lims':(threshold, mid, vmax)}
        else:
            clim = {'kind': 'value', 'pos_lims':(threshold, threshold, vmax)}
    else:
        vmax = np.max(data)
        vmin = np.min(data)
        mid = threshold + ((vmax-threshold)/3)
        if fade:
            clim = {'kind': 'value', 'lims':(threshold, mid, vmax)}
        else:
            clim = {'kind': 'value', 'lims':(threshold, threshold, vmax)}
        
    if surf=='inflated':
        cortex='low_contrast'
    elif surf=='pial':
        cortex=(0.7, 0.7, 0.7)
    else:
        cortex=(0.6, 0.6, 0.6)
    plot_kwargs = dict(subject='fsaverage',
                       subjects_dir=subjects_dir,
                       surface=surf,
                       cortex=cortex,
                       background='white',
                       colorbar=False,
                       time_label=None,
                       time_viewer=False,
                       transparent=True,
                       alpha=0.8,
                       clim=clim,
                       colormap=cmap,
                       )
    
    def remove_white_space(imdata):
        nonwhite_pix = (imdata != 255).any(-1)
        nonwhite_row = nonwhite_pix.any(1)
        nonwhite_col = nonwhite_pix.any(0)
        imdata_cropped = imdata[nonwhite_row][:, nonwhite_col]
        return imdata_cropped

    # top left
    views = ['lat']
    hemi = 'lh'
    brain = stc.plot(views=views, hemi=hemi, **plot_kwargs)
    screenshot = brain.screenshot()
    brain.close()
    screenshot = remove_white_space(screenshot)
    ax1.imshow(screenshot)

    # top right
    views = ['lat']
    hemi = 'rh'
    brain = stc.plot(views=views, hemi=hemi, **plot_kwargs)
    screenshot = brain.screenshot()
    brain.close()
    screenshot = remove_white_space(screenshot)
    ax2.imshow(screenshot)

    # bottom left
    views = ['med']
    hemi = 'lh'
    brain = stc.plot(views=views, hemi=hemi, **plot_kwargs)
    screenshot = brain.screenshot()
    brain.close()
    screenshot = remove_white_space(screenshot)
    ax3.imshow(screenshot)

    # bottom right
    views = ['med']
    hemi = 'rh'
    brain = stc.plot(views=views, hemi=hemi, **plot_kwargs)
    screenshot = brain.screenshot()
    brain.close()
    screenshot = remove_white_space(screenshot)
    ax4.imshow(screenshot)

    # colorbar
    mne.viz.plot_brain_colorbar(cax, clim, cmap, orientation='horizontal', label=cbar_label)
    
    return fig

def glass_brain_plot(adjacency, atlas_coords, threshold, cbar_label):
    
    """
    Make a glass brain connectivity plot using an adjacency matrix and 
    corresponding 3D coordinates for each atlas region.
    """
    
    def add_spheres(plotter, points, radius=2, color="gray"):
        for point in points:
            sphere = pv.Sphere(radius=radius, center=point)
            plotter.add_mesh(sphere, color=color)
    
    def remove_white_space(imdata, decim=None):
        nonwhite_pix = (imdata != 255).any(-1)
        nonwhite_row = nonwhite_pix.any(1)
        if decim:
            nonwhite_row[::decim] = True
        nonwhite_col = nonwhite_pix.any(0)
        if decim:
            nonwhite_col[::decim] = True
        imdata_cropped = imdata[nonwhite_row][:, nonwhite_col]
        return imdata_cropped
    
    # get upper triangle only
    triu = np.triu_indices(adjacency.shape[0],1)
    mask = np.zeros((adjacency.shape[0],adjacency.shape[0]))
    mask[triu] = 1
    adjacency = adjacency.copy()
    adjacency *= mask
    
    # get indices
    threshold = threshold
    indices = np.argwhere(np.abs(adjacency) > threshold)
    values = adjacency[np.abs(adjacency) > threshold]
    
    # get coordinates of indices
    lines = []
    for ind in range(indices.shape[0]):
        line = np.array([atlas_coords[indices[ind,0],:], atlas_coords[indices[ind,1],:]])
        lines.append(line)
    lines = np.concatenate(lines, 0)
    
    # get colors
    vmin, vmax = -np.max(np.abs(adjacency[triu])), np.max(np.abs(adjacency[triu]))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.RdBu_r
    rgb_values = cmap(norm(values))
    
    # get fsaverage meshes
    fsaverage = datasets.fetch_surf_fsaverage()
    lh, rh = surface.load_surf_mesh(fsaverage.pial_left), surface.load_surf_mesh(fsaverage.pial_right)
    
    # lh mesh
    coords, faces = lh.coordinates, lh.faces
    faces_vtk = np.column_stack((np.full(faces.shape[0], 3), faces)).ravel()
    lh_mesh = pv.PolyData(coords, faces_vtk)

    # rh mesh
    coords, faces = rh.coordinates, rh.faces
    faces_vtk = np.column_stack((np.full(faces.shape[0], 3), faces)).ravel()
    rh_mesh = pv.PolyData(coords, faces_vtk)
    
    # plot brain
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(lh_mesh, color="gray", opacity=0.06)
    plotter.add_mesh(rh_mesh, color="gray", opacity=0.06)
    
    # add line
    for l, ll in enumerate(np.arange(0, lines.shape[0], 2)):
        color = rgb_values[l,:]
        color[-1] = 0.6
        plotter.add_lines(lines[ll:ll+2,:], color=rgb_values[l,:], width=7)
    
    # add coordinates
    add_spheres(plotter, atlas_coords, radius=1.5, color="gray")
    
    # get up view
    plotter.view_xy() # up view
    img_up = plotter.screenshot()
    img_up = remove_white_space(img_up)
    
    # get side view
    plotter.view_yz() # side view
    img_side = plotter.screenshot()
    img_side = remove_white_space(img_side)
    
    # get back view
    plotter.view_xz() # back view
    img_back = plotter.screenshot()
    img_back = remove_white_space(img_back)
    
    # plot 
    fig = plt.figure(figsize=(9,5.5))
    ax1 = fig.add_axes([0.15, 0.2, 0.4, 0.7])  # top-left
    ax2 = fig.add_axes([0.55, 0.55, 0.3, 0.35])  # top-right
    ax3 = fig.add_axes([0.55, 0.2, 0.3, 0.35])  # bottom-left
    cax = fig.add_axes([0.32, 0.12, 0.4, 0.03]) # colorbar ax
    
    # insert images
    ax1.imshow(img_up)
    ax1.axis(False)
    ax2.imshow(img_side)
    ax2.axis(False)
    ax3.imshow(img_back)
    ax3.axis(False)
    
    # add cbar
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=cax, orientation='horizontal', label=cbar_label)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(cbar_label, fontsize=16, labelpad=0)
    
    return fig

def regression_plot(x, y, ax, color='purple', label=None):
    
    """
    Simple regression plot with shaded confidence interval.
    """
    
    # regression
    X = sm.add_constant(x)
    model = sm.OLS(y, X)
    results = model.fit()
    slope = results.params[1]
    intercept = results.params[0]
    y_pred = intercept + slope*x
    predictions = results.get_prediction(X)
    ci = predictions.conf_int()
    sort_i = np.argsort(x)
    stat, p = pearsonr(x,y)
    
    # plot
    ax.scatter(x,y, color=color, alpha=0.5)
    ax.fill_between(x[sort_i], ci[sort_i, 0], ci[sort_i, 1], color=color, alpha=0.2)
    ax.plot(x[sort_i],y_pred[sort_i], color=color, label=label, linewidth=2.5)
    
    return stat, p, slope


def line_with_error(ax, data, times, color, legend):
    
    """
    plot line with error.
    """
    values = np.mean(data, 0)
    err = np.std(data, 0) / np.sqrt(len(data))
    ax.plot(times, values, color=color, label=legend)
    ax.fill_between(times, values-err, values+err, alpha=0.2, color=color)