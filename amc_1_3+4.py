#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 10:46:10 2026
    ACM V3
    Rewrite
    step: 1.2
@author: Bryn King, Emmanuel Vu
"""

import sys
import os
import warnings
import numpy as np
from os import listdir
from george_module import list_to_num_dict
from os.path import isfile, join
from numpy.polynomial.polynomial import Polynomial
from scipy import ndimage
import matplotlib.pyplot as plt

'''     Configurations      '''
# ===== Distances (AU) =====
ceres_AU = 2.925908893634523 # Date: 06/13/15 @ JPL Horizon
vesta_AU = 2.253290241358010 # Date: 08/20/11 @ JPL Horizon

# ===== Directories =====
ceres_dir = '/rapid/inst/vir/data/ceres/combined_cycles' 
vesta_dir = '/rapid/inst/vir/data/vesta/combined_cycles'
calib_dir = '/rapid/inst/vir/DWNCLVIR_I1B/CALIB' # wl & solar spectrum dir
version_dir = '/rapid/inst/vir/acm_v5'
prev_step_dir = version_dir + '/1_2'
step_save_dir = version_dir + '/1_3+4'
os.makedirs(step_save_dir, exist_ok=True)

# ===== Files =====
solar_fname = 'DAWN_VIR_IR_SOLAR_SPECTRUM_V2.TAB'
wl_fname = 'DAWN_VIR_IR_HIGHRES_SPECAL_V2.TAB'

# ===== Constants =====
num_bands = 432
num_samples = 256

# ===== Loading Wavelengths + Solar Spectrum =====
solar_fpath = join(calib_dir,solar_fname)
solar = np.loadtxt(solar_fpath)
try:
    solar.shape[0] == num_bands
except:
    warnings.warn('Incorrect solar spectrum file found')
    sys.exit()

wl_fpath = join(calib_dir,wl_fname)
wl = np.loadtxt(wl_fpath)
try:
    wl.shape[0] == num_bands
except:
    warnings.warn('Incorrect wavelength file found')
    sys.exit()

# ===== Miscellaneous =====
# setting up pho.cub headers
pcol_list = ['phase','emission','incid','lat','lon']
pcol = list_to_num_dict(pcol_list) 

# find inds for filter junctions
ind142, ind156, ind241, ind261, ind374, ind383, ind435, ind454 =\
    np.argmin(abs(wl-1.42)), np.argmin(abs(wl-1.56)),\
        np.argmin(abs(wl-2.41)), np.argmin(abs(wl-2.61)),\
            np.argmin(abs(wl-3.74)), np.argmin(abs(wl-3.83)),\
                np.argmin(abs(wl-4.35)), np.argmin(abs(wl-4.54))

filter_junction_indices = np.array([[ind142, ind156], [ind241, ind261],\
                                  [ind374, ind383], [ind435, ind454]])
    
ind259, ind407 = np.argmin(abs(wl-2.59)), np.argmin(abs(wl-4.07))

# ===== Functions ======
def weighted_avg(arr, w=2):
    """
    calculates the weighted average across a 2d image in the y (x,y) dimension
    a value n = [(n-1) + 2n + (n+1)] / 3
    edges are n = [(n-1) + n] /2
    Parameters
    ----------
    arr : numpy array
        2d array
    w : weight of the center pixel, optional
        The default is 2.

    Returns
    -------
    output array
    """
    weight = w
    kernel = np.array([1, weight, 1])
    output = np.copy(arr)
    for i, ival in enumerate(arr[0, :]):
        output[:, i] = ndimage.correlate(arr[:, i], kernel, mode='constant', cval=0.0)
    output[1:-1, :] /= np.sum(kernel)
    # edges
    output[0, :] /= (np.sum(kernel) - kernel[0])
    output[-1, :] /= (np.sum(kernel) - kernel[-1])
    return output
    
# ===== Step =====
print('Loading S_med')
S_med = np.load(prev_step_dir + '/S_med.npy')
S_med_new = np.copy(S_med)

# ===== 1.3 + 1.4 =====
print("Performing 1.3")
S_med_weighted = np.zeros(np.shape(S_med))
S_med_weighted[:ind259] = weighted_avg(S_med[:ind259], 2)
S_med_weighted[ind259:ind407] = weighted_avg(S_med[ind259:ind407], 2)
S_med_weighted[ind407:] = weighted_avg(S_med[ind407:], 2)

print("Performing 1.4")
S_med_avg = np.zeros(np.shape(S_med))
S_med_avg[:ind259] = weighted_avg(S_med_weighted[:ind259], 1)
S_med_avg[ind259:ind407] = weighted_avg(S_med_weighted[ind259:ind407], 1)
S_med_avg[ind407:] = weighted_avg(S_med_weighted[ind407:], 1)

S_med_ratio = S_med_weighted / S_med_avg

# calculating mean and std of ratio for every sample 
ratio_mean = np.nanmean(S_med_ratio, axis = 0)
ratio_std = np.nanstd(S_med_ratio, axis = 0)

bad_vals = []
S_med_mask = np.zeros(S_med.shape)

# looking for outliers to replace with point from fitted polynomial
for si in range(S_med.shape[1]):
    for wi in range(S_med.shape[0]):
        # Ignore the wavelength values in the stitching range
        if (164 <= wi <= 166) or (321 <= wi <= 323): #lazy
            continue # do nothing
        # checking to see if ratio is outside 3 stds
        if S_med_ratio[wi, si] >= (ratio_mean[si] + (3 * ratio_std[si])) or \
            S_med_ratio[wi, si] <= (ratio_mean[si] - (3 * ratio_std[si])):
                bad_vals.append((wi, si))
                S_med_new[wi, si] = np.nan
                

poly_domains = []
poly_ranges = []

for i, val in enumerate(bad_vals):
    #getting the domains and ranges for the polynomial fitting
    wi = val[0]
    si = val[1]
    if wi < 10: # for wl on lower end
        data = S_med_new[:wi + 21, si]
        idx = np.isfinite(data)
        wl_subset = wl[:wi + 21]
        neighbor_wl = wl_subset[idx]
        neighbor_wl_data = data[idx]
    elif wi > 421: # for wl on higher end
        data = S_med_new[wi - 20:, si]
        idx = np.isfinite(data)
        wl_subset = wl[wi - 20:]
        neighbor_wl = wl_subset[idx]
        neighbor_wl_data = data[idx]
    else: # typical wl
        data = S_med_new[wi - 10: wi + 11, si]
        idx = np.isfinite(data)
        wl_subset = wl[wi - 10: wi + 11]
        neighbor_wl = wl_subset[idx]
        neighbor_wl_data = data[idx]
    poly_domains.append(neighbor_wl)
    poly_ranges.append(neighbor_wl_data)    
                
for i, val in enumerate(bad_vals):
    wi = val[0]
    si = val[1]
    # constructing fitted polynomial and applying to outlier
    poly = Polynomial.fit(poly_domains[i], poly_ranges[i], 2)
    S_med_new[wi, si] = poly(wl[wi])
    S_med_mask[wi, si] = poly(wl[wi])
print('Saving...')
np.save(step_save_dir + '/S_med', S_med_new)

print('Done.')

# Testing

"""
S_med_old = np.load('/rapid2/inst/vir/artifact_correction/python_data_p1_34/S_med.npy')

S_med_diff = S_med_old - S_med_new

S_med[S_med>1] = 1
S_med_weighted[S_med_weighted>1] = 1

fig = plt.figure(dpi=400)
# plt.imshow(S_med_weighted)
plt.imshow(S_med_old)
plt.title("S_med_old")
plt.show()
fig = plt.figure(dpi=400)
plt.imshow(S_med_new)
plt.title("S_med_new")
plt.show()
# fig = plt.figure(dpi=400)
# plt.imshow(S_med_ratio)
# plt.title("S_med_ratio")
# plt.show()
# fig = plt.figure(dpi=400)
# plt.imshow(S_med_mask)
# plt.title("Mask")
# plt.show()
# fig = plt.figure(dpi=400)
# plt.imshow(S_med_diff)
# plt.title("S_med diff")
# plt.colorbar()
# plt.show()
# fig = plt.figure(dpi=400)
# plt.plot(S_med_new[50])
# plt.show()
"""