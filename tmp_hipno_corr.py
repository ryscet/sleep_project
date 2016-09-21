#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 15:58:47 2016

@author: user
"""
from parse_signal import load_psg, load_neuroon

import parse_hipnogram as ph
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal, fftpack
from numpy.fft import fft, ifft, fft2, ifft2, fftshift
from itertools import tee
from obspy.signal.cross_correlation import xcorr

try: 
    psg_signal
    print('loaded')
except:
    print('loading')
    psg_signal =  load_psg('F3-A2')
    neuroon_signal =  load_neuroon()
    
plt.style.use('ggplot')

def cross_correlation_using_fft(x, y):
    f1 = fft(x)
    f2 = fft(np.flipud(y))
    cc = np.real(ifft(f1 * f2))
    return fftshift(cc)
 
# shift < 0 means that y starts 'shift' time steps before x # shift > 0 means that y starts 'shift' time steps after x
def compute_shift(x, y):
    assert len(x) == len(y)
    c = cross_correlation_using_fft(x, y)
    assert len(c) == len(x)
    zero_index = int(len(x) / 2) - 1
    shift = zero_index - np.argmax(c)
    return shift,c
    

def iterate_hours():
    
    psg_10 = psg_signal.resample('10ms').mean()
    neuroon_10 = neuroon_signal.resample('10ms').mean()
    psg_cut = psg_10.between_time('00:00', '01:00')
    neuroon_cut = neuroon_10.between_time('00:00', '01:00')

    x = np.array([0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0, 0, 0, 0]*10) 
    y = np.array([0, 0, 0, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0]*10)
    
    a, b = xcorr(x, y, 10)

    a, b = xcorr(neuroon_cut, psg_cut, 20000)
    
    shift, c = compute_shift(neuroon_cut, psg_cut)
    print(shift)
    
    
#    for start, end in pairwise([ '23:00','00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00']):
#        plt.close('all')
#        print('start: %s'%start)
        
        



#def pandas_crosscorr(datax, datay, min_lag = 0,  max_lag=0):
#
#    lags = np.arange(min_lag, max_lag, 1)
#    
#    corrs = np.zeros(len(lags))
#    for idx,  lag in enumerate(lags):
#      #  print(idx)
#        c = datax.corr(datay.shift(lag), method = 'pearson')
#        corrs[idx] = c
#        
#    return np.array(corrs) 
#psg_hipno = ph.parse_psg_stages()
#noo_hipno = ph.parse_neuroon_stages()
#
#psg_30 = psg_hipno['stage_num'].resample('30s').mean().fillna(method = 'ffill')
#neuroon_30 = noo_hipno['stage_num'].resample('30s').mean().fillna(method = 'ffill')
#
##corr1 = signal.correlate(neuroon_10,psg_10, mode = 'same')
#corr1 = pandas_crosscorr( psg_30,neuroon_30, min_lag = - 200,  max_lag=200)
#fig,axes = plt.subplots()
#axes.plot(corr1)
#axes.set_xlim(0, len(corr1))
#axes.axvline(len(corr1)/2, color='k', linestyle='--')
#raise_window()
##for start, end in pairwise([ '23:00','00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00']):
##    plt.close('all')
##    print('start: %s'%start)
##    psg_cut = psg_10.between_time(start, end)
##    neuroon_cut = neuroon_10.between_time(start, end)
##    
##    
#
#
#
#def pairwise(iterable):
#    """Used to iterate over a list elements organized in pairs"""
#    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
#    a, b = tee(iterable)
#    next(b, None)
#    return zip(a, b)