#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 10:46:10 2026
    ACM V3
    Rewrite
    step: Pre
@author: Bryn King, Emmanuel Vu
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
from osgeo import gdal
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
prev_step_dir = version_dir + '/X'
step_save_dir = version_dir + '/pre'
os.makedirs(step_save_dir, exist_ok=True)

# ===== Files =====
solar_fname = 'DAWN_VIR_IR_SOLAR_SPECTRUM_V2.TAB'
wl_fname = 'DAWN_VIR_IR_HIGHRES_SPECAL_V2.TAB'
bad_pixel_file = "/rapid/inst/vir/misc/BadPixelsSurvey.csv"

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
    
# Dead pixles
dead_pixels = pd.read_csv(bad_pixel_file)[["Band", "Sample"]].to_numpy()

# ===== Functions =====
def get_cub_prefix(direct):
    '''
    A function that gets the file name of any .cub files 
    (excluding .pho.cub files) in a given directory.
    Input:
        direct: directory containing the .cub files.
    Output:
        fprefix_sort: a sorted list of the .cub file names.
    '''
    temp_list = [file for file in listdir(direct) if isfile(join(direct, file))]
    fprefix_list = []
    for file in temp_list:
        if '.cub' in file:
            if '.pho' not in file:
                if '.aux' not in file:
                    fprefix_list.append(file.split('.')[0])
    fprefix_sort = np.sort(fprefix_list)
    return fprefix_sort

def load_spectra(fprefix_sort, direct, distance, keep_data = False):
    '''
    A function that converts a list of .cub files to I/F. Additionally, it can 
    replace data from spectra with incidence angles greater than 60 degrees 
    with NaNs.
    Inputs:
        fprefix_sort: a sorted list of .cub file names.
        direct: directory of the .cub file.
        distance: distance (AU) of the dwarf planet to the sun.
        keep_data: if True, additionally returns a list of the converted I/F 
        data with no incidence angle filtering
    Output:
        filt_spectra: filtered array of the converted .cub data
        (optional) cub_list: unfiltered and separated list of the .cub data
    '''
    
    filtered_spectra = np.zeros((num_bands, 0, num_samples)) # return array
    cub_list = [] # return list
    # running_time = []
    for fi, fprefix in enumerate(fprefix_sort):
        # st = time.time()
        
        # open spectral cube
        cub_fname = fprefix +'.cub'
        cub_fpath = join(direct, cub_fname)
        cub_cf = gdal.Open(cub_fpath)
        
        cub = cub_cf.ReadAsArray()
        
        try: 
            cub.shape[0] == num_bands
        except:
            warnings.warn('Non high-res observation found')
            sys.exit()
            
        # replacing special pixel values w/NaNs
        bad_ind = np.where(cub < 0)
        cub[bad_ind] = np.nan
        cub[dead_pixels[:, 0], :, dead_pixels[:, 1]] = np.nan #dead pixel list from VIR CALIBRATION L1B document
        
        # open cube backplanes
        pho_fname = fprefix + '_ba1.pho.cub'
        pho_fpath = join(direct, pho_fname)
        pho_cf = gdal.Open(pho_fpath)
        pho = pho_cf.ReadAsArray()
        
        # converting to I/F
        for wi in range(cub.shape[0]):
            cub[wi] = (distance ** 2 * cub[wi] * np.pi) / solar[wi]
        
        # storing unfiltered cub
        if keep_data:
            cub_list.append(cub)
    
        # replacing solar incidence angles > 60 degrees w/NaNs
        filtered_cub = cub.copy()
        bad_incid_ind = np.where(pho[pcol['incid']] > 60)
        filtered_cub[:, *bad_incid_ind] = np.nan
        
        # filtering out all-NaN lines
        lines_to_delete = []
        for line in range(cub.shape[1]):
            if np.where(np.isnan(filtered_cub[:, line, :]))[0].size ==\
                num_bands * num_samples:
                lines_to_delete.append(line)
        filtered_cub = np.delete(filtered_cub, lines_to_delete, 1)

        # adding to spectra pool
        filtered_spectra = np.concatenate((filtered_spectra, filtered_cub), axis = 1)
        
        # print progress
        # nd = time.time()
        # time_per_loop = nd - st
        # running_time.append(time_per_loop)
        # print_inline_progress(fi, fprefix_sort, running_time, 1)
    
    if keep_data:
        return filtered_spectra, cub_list
    else:
        return filtered_spectra
    
# ===== Step =====

'''     Loading Cubes       '''
# ===== Retrieving Cub File Names =====
ceres_prefix_sort = get_cub_prefix(ceres_dir)
vesta_prefix_sort = get_cub_prefix(vesta_dir)

# ===== Loading Cub Spectra =====
print("Ceres Spectra Progress")

ceres_spectra, ceres_cubs = load_spectra(ceres_prefix_sort, ceres_dir,\
                                         ceres_AU, keep_data = True)

np.save(step_save_dir + '/ceres_spectra', ceres_spectra)
np.savez(step_save_dir + '/ceres_cubs', *ceres_cubs)


print("\nVesta Spectra Progress")
vesta_spectra = load_spectra(vesta_prefix_sort, vesta_dir, vesta_AU)
print('\n')

np.save(step_save_dir + '/vesta_spectra', vesta_spectra)