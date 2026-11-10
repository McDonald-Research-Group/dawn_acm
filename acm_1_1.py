#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 10:46:10 2026
    ACM V3
    Rewrite
    step: 1.1
@author: Bryn King, Emmanuel Vu
"""

import sys
import os
import warnings
import numpy as np
from os import listdir
from george_module import list_to_num_dict
from os.path import isfile, join

'''     Configurations      '''
# ===== Distances (AU) =====
ceres_AU = 2.925908893634523 # Date: 06/13/15 @ JPL Horizon
vesta_AU = 2.253290241358010 # Date: 08/20/11 @ JPL Horizon

# ===== Directories =====
ceres_dir = '/rapid/inst/vir/data/ceres/combined_cycles' 
vesta_dir = '/rapid/inst/vir/data/vesta/combined_cycles'
calib_dir = '/rapid/inst/vir/DWNCLVIR_I1B/CALIB' # wl & solar spectrum dir
version_dir = '/rapid/inst/vir/acm_v5'
prev_step_dir = version_dir + '/pre'
step_save_dir = version_dir + '/1_1'
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
    
# ===== Functions ======

# ===== Step =====
'''     Loading Previous Step Files     '''
print('Loading Ceres Spectra Array...')
ceres_spectra = np.load(version_dir + '/pre' + '/ceres_spectra.npy')
# print('Loading Ceres Cube zip...')
# ceres_cubs = np.load(pre_dir + '/ceres_cubs.npz')
print('Loading Vesta Spectra Array...')
vesta_spectra = np.load(version_dir + '/pre' + '/vesta_spectra.npy')

'''     Loading Cubes       '''
# ===== 1.1 =====
print("Performing 1.1...")
# Note: incidence angle filtering was done in load_spectra function

# finding indices to stitch Vesta and Ceres data
ind259, ind407 = np.argmin(abs(wl-2.59)), np.argmin(abs(wl-4.07))

# finding lesser of the two line lengths to properly concatenate
line_ind = min(ceres_spectra.shape[1], vesta_spectra.shape[1]) - 1

# combining vesta and ceres data at boundaries
stitched_spectra = np.concatenate((ceres_spectra[:ind259, :line_ind], \
                       vesta_spectra[ind259:ind407, :line_ind], \
                           ceres_spectra[ind407:, :line_ind]), axis = 0)

print('Saving...')
np.save(step_save_dir + '/stitched_spectra', stitched_spectra)

print('Done.')
