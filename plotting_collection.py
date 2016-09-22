#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 18:58:03 2016

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_crosscorrelation_heatmap(all_coeffs, dates_range):
    plt.style.use('ggplot')
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

# Time-series like coeffs plot
#    fig, axes = plt.subplots(1)

#    for coefs in coeffs_roi:
#        axes.plot(times_in_sec , coefs, alpha = 0.1, color = 'b')
#        #axes[1].plot(ps, alpha = 0.01, color = 'r')
#    
#    axes.plot(times_in_sec , coeffs_roi.mean(axis = 0), alpha = 1, color = 'b')
#    
#    axes.set_xlim(times_in_sec [0], times_in_sec [-1])
    
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
    

