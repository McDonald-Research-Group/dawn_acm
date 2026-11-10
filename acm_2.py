#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 10:46:10 2026
    ACM V3
    Rewrite
    step: 2.1 + 2.2
@author: Bryn King, Emmanuel Vu
"""

import sys
import os
import warnings
import numpy as np
from george_module import list_to_num_dict
from os.path import isfile, join
from scipy import ndimage
from osgeo import gdal
import matplotlib.pyplot as plt
import pdr

'''     Configurations      '''
# ===== Distances (AU) =====
ceres_AU = 2.925908893634523 # Date: 06/13/15 @ JPL Horizon
vesta_AU = 2.253290241358010 # Date: 08/20/11 @ JPL Horizon

# ===== Directories =====
ceres_dir = '/rapid/inst/vir/data/ceres/combined_cycles' 
vesta_dir = '/rapid/inst/vir/data/vesta/combined_cycles'
calib_dir = '/rapid/inst/vir/DWNCLVIR_I1B/CALIB' # wl & solar spectrum dir
version_dir = '/rapid/inst/vir/acm_v4'
prev_step_dir = version_dir + '/'
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
    
ind259, ind407 = np.argmin(abs(wl-2.59)), np.argmin(abs(wl-4.07))
    
# ===== Functions ======
def weighted_avg(arr, w=2):
    """
    calculates the weighted average across a 3d image in the x of (x,y,z) dimension
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
    for i, ival in enumerate(arr[0, :, 0]):
        for j, jval in enumerate(arr[0, 0, :]):
            output[:, i, j] = ndimage.correlate(arr[:, i, j], kernel, mode='constant', cval=0.0)

    # Normalize body
    output[1:-1, :, :] /= np.sum(kernel)
    # edges
    output[0, :, :] /= (np.sum(kernel) - kernel[0])
    output[-1, :, :] /= (np.sum(kernel) - kernel[-1])
    return output

def cub_to_array(cube_fp):
    """
    Loads a filepath to a .cub into a numpy array, converting to I/F.

    Parameters
    ----------
    gdal_fp : string
        string pointing to a .cub file

    Returns
    -------
    np.array
        Array of the pixel data from the .cub converted to I/F
    """
    arr = gdal.Open(cube_fp).ReadAsArray()
    # nan bad values
    arr[arr < 0] = np.nan
    # Convert to I/F
    for wi in range(arr.shape[0]):
        arr[wi] = (ceres_AU ** 2 * arr[wi] * np.pi) / solar[wi]
        
    return arr

def lbl_to_array(lbl_fp):
    """
    Converts a pointer to a .LBL file to a numpy array in I/F.

    Parameters
    ----------
    lbl_fp : string
        points to a .LBL file

    Returns
    -------
    np.array
        array in I/F with bad values removed.
    """
    arr = pdr.read(lbl_fp)["QUBE"]
    # nan bad values
    arr[arr < 0] = np.nan
    # Convert to I/F
    for wi in range(arr.shape[0]):
        arr[wi] = (ceres_AU ** 2 * arr[wi] * np.pi) / solar[wi]
        
    return arr
    
def apply_acm(cube_arr, A):
    """
    Applies the ACM to a single cube

    Parameters
    ----------
    cube_arr : np.array of the cube to apply the correction to
    A: Artifact correction matrix

    Returns
    -------
    3d numpy array
        corrected cube
    """
    R = weighted_avg(cube_arr, 2)
    R_prime = np.copy(R)

    for line_index, val in enumerate(R[0, :, 0]):
        R_prime[:, line_index, :] = R[:, line_index, :] / (A + 1)
    
    return R_prime

# ===== Step =====
print('Loading ACM')
A_fp = version_dir + '/1_7/A.npy'
A = np.load(A_fp)


# Importing single .cub?
single_cub = 0
single_cub_fp = 'rapid2/inst/vir/data/ceres/corrazzo_cube_v3/VIR_IR_1B_1_498058497.cub'
single_cub_save_as = 'rapid2/inst/vir/data/ceres/corrazzo_cube_v3/VIR_IR_1B_1_498058497_corrected'
# Importing a single .LBL?
single_lbl = 0
single_lbl_fp = 'rapid2/inst/vir/data/ceres/corrazzo_cube_v3/VIR_IR_1B_1_498058497.LBL'
single_lbl_save_as = 'rapid2/inst/vir/data/ceres/corrazzo_cube_v3/VIR_IR_1B_1_498058497_corrected'

# Importing from directory full of .cubs?
dir_of_cub = 0
cub_dir = '/rapid2/inst/vir/data/ceres/combined_cycles'
# Importing from directory full of .LBLs?
dir_of_lbl = 1
lbl_dir = '/rapid/inst/vir/data/ceres/combined_cycles'
# Save corrected file/s?
saving = 1
# Directory to save file/s to (automatically names them with the suffix '_corrected')
save_dir = '/rapid/inst/vir/data/ceres/combined_cycles_v4'

if single_cub:
    # Single .cub
    print("Loading cube")
    cube = cub_to_array(single_cub_fp)
    print("Applying ACM")
    cube_fixed = apply_acm(cube, A)
    if saving:
        print("Saving")
        np.save(single_cub_save_as, cube_fixed)
elif single_lbl:
    # Single LBL
    print("Loading cube")
    cube = lbl_to_array(single_lbl_fp)
    print("Applying ACM")
    cube_fixed = apply_acm(cube, A)
    if saving:
        print("Saving")
        np.save(single_lbl_save_as, cube_fixed)
elif dir_of_cub:
    # Directory of cubes
    dir_list = os.listdir(cub_dir)
    arr_list = []
    if dir_of_cub:
        print("Finding .cub files")
        cub_list = []
        for i in dir_list:
            if i.endswith(".cub") and not i.endswith('.pho.cub'):
                cub_list.append(i)
                name_list = [s[:-4] for s in cub_list]
                size = len(cub_list)
        for num, i in enumerate(cub_list):
            print(f"Applying ACM to cube #{num} of {size}")
            arr = cub_to_array(cub_dir + '/' + i)
            corrected_arr = apply_acm(arr, A)
            if saving:
                print("Saving")
                np.save(save_dir + '/' + name_list[num] + '_corrected.npy', corrected_arr)
elif dir_of_lbl:
    # Directory of .LBLs
    lbl_list = []
    dir_list = os.listdir(lbl_dir)
    for i in dir_list:
        if i.endswith(".LBL") and not i.endswith('HK.LBL'):
            lbl_list.append(i)
            name_list = [s[:-4] for s in lbl_list]
            size = len(lbl_list)
    for num, i in enumerate(lbl_list):
        print(f"Applying ACM to cube #{num} of {size}")
        arr = lbl_to_array(lbl_dir + '/' + i)
        corrected_arr = apply_acm(arr, A)
        if saving:
            print("Saving")
            np.save(save_dir + '/' + name_list[num] + '_corrected.npy', corrected_arr)

print("Done!")