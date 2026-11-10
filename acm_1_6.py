#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 10:46:10 2026
    ACM V3
    Rewrite
    step: 1.6
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

'''     Configurations      '''
# ===== Distances (AU) =====
ceres_AU = 2.925908893634523 # Date: 06/13/15 @ JPL Horizon
vesta_AU = 2.253290241358010 # Date: 08/20/11 @ JPL Horizon

calc_degree = False
degree_fit = 4 #original value 203 

# ===== Directories =====
ceres_dir = '/rapid/inst/vir/data/ceres/combined_cycles' 
vesta_dir = '/rapid/inst/vir/data/vesta/combined_cycles'
calib_dir = '/rapid/inst/vir/DWNCLVIR_I1B/CALIB' # wl & solar spectrum dir
version_dir = '/rapid/inst/vir/acm_v5'
prev_step_dir = version_dir + '/1_5'
step_save_dir = version_dir + '/1_6'
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

# ===== Step =====
print('Loading U_med')
U_med = np.load(prev_step_dir + '/U_med.npy')

print("Performing 1.6")
# calculating best degree fit 
if calc_degree:
    resid_1_list = []
    resid_2_list = []
    resid_3_list = []

    for i in range(7):
        poly1 = Polynomial.fit(wl[:ind259], U_med[:ind259], i)
        resid = np.sqrt(np.mean((poly1(wl[:ind259]) - U_med[:ind259]) ** 2))
        resid_1_list.append(resid)
    for i in range(5):
        poly2 = Polynomial.fit(wl[ind259:ind407], U_med[ind259:ind407], i)
        resid = np.sqrt(np.mean((poly2(wl[ind259:ind407]) - U_med[ind259:ind407]) ** 2))
        resid_2_list.append(resid)
    for i in range(5):
        poly3 = Polynomial.fit(wl[ind407:], U_med[ind407:], i)
        resid = np.sqrt(np.mean((poly3(wl[ind407:]) - U_med[ind407:]) ** 2))
        resid_3_list.append(resid)
       #  poly_test = np.concatenate((poly1(wl[:ind259]),\
       #                        poly2(wl[ind259:ind407]), poly3(wl[ind407:])))
    degree_1_fit = np.argmin(resid_1_list)
    degree_2_fit = np.argmin(resid_2_list)
    degree_3_fit = np.argmin(resid_3_list)
    print(f"Degree 1 {degree_1_fit} was chosen")
    print(f"Degree 2 {degree_2_fit} was chosen")
    print(f"Degree 3 {degree_3_fit} was chosen")
    
    poly1 = Polynomial.fit(wl[:ind259], U_med[:ind259], degree_1_fit)
    poly2 = Polynomial.fit(wl[ind259:ind407], U_med[ind259:ind407], degree_2_fit)
    poly3 = Polynomial.fit(wl[ind407:], U_med[ind407:], degree_3_fit)

# fitting polynomials at partitions between stitches
if not calc_degree:
    poly1 = Polynomial.fit(wl[:ind259], U_med[:ind259], degree_fit)
    poly2 = Polynomial.fit(wl[ind259:ind407], U_med[ind259:ind407], degree_fit)
    poly3 = Polynomial.fit(wl[ind407:], U_med[ind407:], degree_fit)

# stitching polynomial
P_U = np.concatenate((poly1(wl[:ind259]), poly2(wl[ind259:ind407]), poly3(wl[ind407:])))

print('Saving...')
np.save(step_save_dir + '/P_U', P_U)

print('Done.')