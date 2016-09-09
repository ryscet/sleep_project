#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 16:47:32 2016

@author: ryszardcetnarski
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import explore_eeg as ee
from scipy import signal



if 'psg_slices' not in globals():
    print('-----loading------')
    
    psg, psg_slices = ee.parse_psg()
    neuroon, neuroon_slices = ee.parse_neuroon()
    
def cross_corr_frequency():
    f_psg_slices, Pxx_psg_slices = signal.welch(psg_slices, 200, nperseg=256, noverlap = 128) 
    f_no_slices, Pxx_no_slices = signal.welch(neuroon_slices, 125, nperseg=256, noverlap = 128)
    plt.style.use('ggplot')
   # for idx, (hz_psg, hz_no) in enumerate(zip(Pxx_psg_slices.T, Pxx_no_slices.T)):
        #CrossCorrelate(hz_psg, hz_no, 10, 1, f_psg_slices[idx])
        
    alpha_psg = Pxx_psg_slices[:, 13]
    alpha_non = Pxx_no_slices[:, 22]
    
    CrossCorrelate(alpha_psg, alpha_non, 10, 1, 'alpha')
    CrossCorrelate(alpha_psg, alpha_non, 10, 1, 'alpha')
    
    #CrossCorrelate(psg, neuroon, 10, 1, 'raw_signal')
    
    
    

def CrossCorrelate(x,y, absolute_lag, binsize, info):
    #x = np.sin(np.arange(0, 1440 * pi/180, 0.1) + 45* pi/180 )
    #y = np.sin(np.arange(0, 1440 * pi/180, 0.1))

    
    print(info)
    f, axes = plt.subplots(3)
    f.suptitle(str(info) + '_hz' )
    axes[0].plot(x,'r', label = 'psg')
    axes[0].plot(y, 'b', label = 'neuroon')

  #  print(len(x))

    lags = np.arange(-absolute_lag, absolute_lag+1,  binsize, dtype = 'int')
   # print(len(lags))
    all_coeffs = []
    all_p = []
    for lag in lags:
  #      print(lag)
        if(lag <= 0):
            tmp_x = x[ 0 : len(x) + lag]
            tmp_y = y[0 + abs(lag) ::]
        if(lag > 0 ):
            tmp_x = x[0 + abs(lag) ::]
            tmp_y = y[0 : len(x) - lag]

    #    print(len(tmp_x))
     #   print(len(tmp_y))

        coeff, p = stats.pearsonr(tmp_x , tmp_y)
        all_coeffs.append(coeff)
        all_p.append(p)

    axes[1].plot(lags,all_coeffs)
    axes[2].plot(lags,all_p)
    
    axes[1].set_ylabel('corr coefficient')
    axes[2].set_ylabel('p value')
    
    axes[1].set_xlabel('time lag')
    axes[2].set_xlabel('time_lag')
    plt.legend()
    
    f.savefig('figures/freq_cross_' + str(info) +'.pdf')
    
    
    return all_coeffs,all_p, lags