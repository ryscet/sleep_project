#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 16:47:32 2016

@author: ryszardcetnarski
"""
from parse_signal import load_psg, load_neuroon


import numpy as np
import matplotlib.pyplot as plt
from itertools import tee
import pandas as pd
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
    # Load the signal from hdf database and parse it to pandas series with datetime index
    psg_signal =  load_psg('F3-A2')
    neuroon_signal =  load_neuroon()
    # Resample the signal to 100hz, to have the same length for cross correlation
    psg_10 = psg_signal.resample('10ms').mean()
    neuroon_10 = neuroon_signal.resample('10ms').mean()
    
    # Create ten minute intervals
    dates_range = pd.date_range(psg_signal.head(1).index.get_values()[0], neuroon_signal.tail(1).index.get_values()[0], freq="10min")
    
    # Convert datetime interval boundaries to string with only hours, minutes and seconds
    dates_range = [d.strftime('%H:%M:%S') for d in dates_range]
    
    
    all_coefs = []
    
    #  iterate over overlapping pairs of 10 minutes boundaries 
    for start, end in pairwise(dates_range):
        # cut 10 minutes piece of signal
        neuroon_cut = neuroon_10.between_time(start, end)
        psg_cut = psg_10.between_time(start, end)
        # Compute the correlation using fft convolution
        shift, coeffs = compute_shift(neuroon_cut, psg_cut)
        #normalize the coefficients because they will be shown on the same heatmap and need a common color scale
        all_coefs.append((coeffs - coeffs.mean()) / coeffs.std())
        print(start)
        print(shift)

        
    all_coefs = np.array(all_coefs)
    return all_coefs

def plot_corr(all_coeffs, dates_range):
    fig, axes = plt.subplots(1)
    length = len(all_coeffs[0,:])
    zero_index = int(length/ 2.0) - 1

    # The resampled signal is 100 hz (every sample increments by 10ms) - thus dividing index by 100 results in seconds
    # we reverse the time array because right side from zero index is neuroon lag (psg starts earlier by - n steps) and left side is psg lag (psg starts later by n steps)
    times_in_sec = (np.arange((-length/2), (length/2), 1)[::-1] / 100.0) -1
    
    # number of samples to show the cofficient at around the max correlation
    widnow_size = 500
    # Get the index of the max  of the average of correlation
    max_corr = np.argmax(all_coeffs.mean(axis = 0))

    # Select the part of correlation array around max correlation    
    coeffs_roi = all_coeffs[:, max_corr - widnow_size : max_corr + widnow_size]
    # Do the same for time label
    times_in_sec = times_in_sec [max_corr - widnow_size : max_corr + widnow_size]


    for coefs in coeffs_roi:
        axes.plot(times_in_sec , coefs, alpha = 0.1, color = 'b')
        #axes[1].plot(ps, alpha = 0.01, color = 'r')
    
    axes.plot(times_in_sec , coeffs_roi.mean(axis = 0), alpha = 1, color = 'b')
    
    axes.set_xlim(times_in_sec [0], times_in_sec [-1])
    
    #Heatmap
    fig, axes = plt.subplots()
    fig.suptitle('Crosscorrelation max at: %.2f seconds'%((zero_index - max_corr) /100) )
    
    #flip the array of coeffs upside down, beacuse the imshow (behind seaborn.heatmap) draws images uppside down
    g = sns.heatmap(coeffs_roi, ax = axes)
    
    xtick_res = 100
    ytick_res = 7
    
    g.set(xticks = np.arange(0, len(coeffs_roi[0,:]), xtick_res))
    g.set(yticks = np.arange(0, len(coeffs_roi[:,0]), ytick_res))
    
    

    # get only every 100th sample time, so it is readable on the plot
    times_tup = [divmod(np.abs(seconds), 60) for seconds in times_in_sec[::xtick_res]]
    times_str = ["-%02d:%02d" % (t[0], t[1]) for t in times_tup]
    g.set_xticklabels(times_str, rotation = 45)
   
    
    g.set_yticklabels(dates_range[::ytick_res][::-1], rotation = 0)

    axes.set_xlabel('neuroon time shift from psg in mm:ss')
    axes.set_ylabel('consecutive 10 minutes windows')
    
    plt.tight_layout()
    raise_window()


    
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


    
    
