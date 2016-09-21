#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 16:47:32 2016

@author: ryszardcetnarski
"""
from parse_signal import load_psg, load_neuroon


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import explore_eeg as ee
from scipy import signal
from itertools import tee
import pandas as pd
import parse_hipnogram as ph
import seaborn as sns


from numpy.fft import fft, ifft, fft2, ifft2, fftshift

#try: 
#    psg_signal
#    print('loaded')
#except:
#    print('loading')
#    psg_signal =  load_psg('F3-A2')
#    neuroon_signal =  load_neuroon()


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



def cross_correlate():
    psg_hipno = ph.parse_psg_stages()
    neuroon_hipno = ph.parse_neuroon_stages()
    
    
    psg_signal =  load_psg('F3-A2')
    neuroon_signal =  load_neuroon()
    
    psg_10 = psg_signal.resample('10ms').mean()
    neuroon_10 = neuroon_signal.resample('10ms').mean()
    # Create ten minutes intervals 
    dates_range = pd.date_range(psg_hipno.head(2).index.get_values()[0], neuroon_hipno.tail(1).index.get_values()[0], freq="10min")
    # Convert them to string with only hours, minutes and seconds
    dates_range = [d.strftime('%H:%M:%S') for d in dates_range]
    
                   
    all_coefs = []
    i = 0
    
    for start, end in pairwise(dates_range):
        print(start)
        neuroon_cut = neuroon_10.between_time(start, end).rolling(window = 30).mean().dropna()
        psg_cut = psg_10.between_time(start, end).rolling(window = 30).mean().dropna()

        i = i+1
        print(i)

        shift, coeffs = compute_shift(neuroon_cut, psg_cut)

        all_coefs.append((coeffs - coeffs.mean()) / coeffs.std())#normalize the coefficients because they will be shown on the same heatmap and need a common color scale

        
    all_coefs = np.array(all_coefs)
    return all_coefs

def plot_corr(all_coeffs):
    fig, axes = plt.subplots(1)
    length = len(all_coeffs[0,:])
    zero_index = int(length/ 2.0) - 1

    times = np.arange((-length/2), (length/2), 1) * 10#[zero_index - 20000: zero_index + 1000]
    widnow_size = 1000
    # Get the index of the max  
    max_corr = np.argmax(all_coeffs.mean(axis = 0))
    
    coeffs_roi = all_coeffs[:, max_corr - widnow_size : max_corr + widnow_size]
    times = times[max_corr - widnow_size : max_corr + widnow_size]
    for coefs in coeffs_roi:
        axes.plot(times, coefs, alpha = 0.1, color = 'b')
        #axes[1].plot(ps, alpha = 0.01, color = 'r')
    
    axes.plot(times, coeffs_roi.mean(axis = 0), alpha = 1, color = 'b')
    
    axes.set_xlim(times[0], times[-1])
    
    fig, axes = plt.subplots()
    fig.suptitle('Neuroon-psg cross-correlation in time domain')
    g = sns.heatmap(coeffs_roi, yticklabels = 5, ax = axes)
    g.set(xticks = np.arange(0, len(coeffs_roi[0,:]), 100))
    #g.set(xticklabels = times[::100])
    
    times = times[::100]/1000
    times_tup = [divmod(np.abs(seconds), 60) for seconds in times]
    times_str = ["-%02d:%02d" % (t[0], t[1]) for t in times_tup]
    #m, s = 
    g.set_xticklabels(times_str, rotation = 45)

    axes.set_xlabel('neuroon time shift from psg in minutes:seconds')
    axes.set_ylabel('consecutive 10 minutes windows')
    
    raise_window()
   # xticks = np.arange(-zero_index +1, zero_index + 1, 1)
    #axes.set_xticks(xticks)
    #axes.set_xticklabels([ms/10 for ms in xticks])
    #axes[1].plot(all_p.mean(axis = 0), alpha = 1, color = 'b')
    
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
    
        
def calc_p_coeff(sig_dict, absolute_lag, binsize):
    x = sig_dict['neuroon']
    y = sig_dict['psg']

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

        coeff, p = stats.pearsonr(tmp_x , tmp_y)
        #coeff, p = stats.spearmanr(tmp_x , tmp_y)
        all_coeffs.append(coeff)
        all_p.append(p)

    
    
    return np.array(all_coeffs), np.array(all_p)
    

def save_slices():
    """Run to prepare all slices for cross correlation analysis"""
    channel_name = 'F3-A2'
    psg = load_channel(channel_name)
    neuroon = pd.Series.from_csv('parsed_data/neuroon_parsed.csv')
    
    for slice_len in [60]:
        make_time_slices('%i_minute_psg_%s' %(slice_len, channel_name), slice_len, psg)
        make_time_slices('%i_minute_neuroon'%slice_len, slice_len , neuroon)

def make_time_slices(description, minute_length, sig):
    """Make slices of equal length and duration for cross correlation between psg and neuroon recording.
        Slices are z-scored"""
    psg_hipno = ph.parse_psg_stages()
    neuroon_hipno = ph.parse_neuroon_stages()

   # channel = hdf_to_series('F4-A1')
   # channel = pd.Series.from_csv('parsed_data/neuroon_parsed.csv')
   
    # Check if the slice will occur in both recordings
    channel = sig.loc[psg_hipno.head(1).index.values[0]:neuroon_hipno.tail(1).index.get_values()[0]]
    # Resample to make neuroon and psg have the same size for cross correlation
    channel = channel.resample('10ms').mean()
    sig_duration = np.array([channel.tail(1).index.get_values()[0] - channel.head(1).index.get_values()[0]], dtype='timedelta64[ms]').astype(int)[0]
    
    # in milliseconds * seconds * minutes
    slice_time_ms = 1000 * 60 * minute_length #*2 * 60 # uncomment to get 1h duration, otherwise it's 30 sec in ms
    # number of slices to divide the signal into bins of length specified by slice_time_ms   
    n_periods = round(sig_duration / slice_time_ms,1)
    
    date_rng = pd.date_range(channel.head(2).index.get_values()[0], periods= n_periods, freq= str(int(slice_time_ms)) + 'ms')
    
    slices = []
    for start, end in pairwise(date_rng):
        # Z-score the slice 
        slices.append((channel.loc[start:end] - channel.loc[start:end].mean()) / channel.loc[start:end].std())
        
    slices = np.array(slices)
  
    np.save('parsed_data/numpy_slices/' + description + '_slices', slices)
        
    return 

    
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


    
    
    
    
    
    
#if 'psg_slices' not in globals():
#    print('-----loading------')
#    
#    psg, psg_slices = ee.parse_psg()
#    neuroon, neuroon_slices = ee.parse_neuroon()
    
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