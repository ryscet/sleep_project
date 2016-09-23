#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 18:58:03 2016

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_crosscorrelation_heatmap(all_coeffs, dates_range, window = 500):
    """Plots a patrix of correlation coefficients from neuroon and psg cross correlation. Rows represent a crosscorrelation
        result from a single time window, lasting 10 minutes. Each next row is the next 10 minutes of the recording. Columns represent correlation coefiicients 
        n samples back and forth from max correlation (n samples defined by 'window')

        Parameters
        ----------
        all_coeffs: 2d numpy.array
            signal crosscorrelation (columns) in consecutive time epochs (rows)
        
        dates_range: list of strings
            time info for the rows of 'all_coeffs'

        window: int
            n_samples back and forth from average max correlation to show on the plot.



    """
    plt.style.use('ggplot')

    length = len(all_coeffs[0,:])
    
    # index where the coefficient represents the correlation for non-shifted signals
    zero_index = int(length/ 2.0) - 1

    # The resampled signal is 100 hz (every sample increments by 10ms) - thus dividing index by 100 results in seconds
    # we reverse the time array because right side from zero index is neuroon lag (psg starts earlier by - n steps) and left side is psg lag (psg starts later by n steps)
    times_in_sec = (np.arange((-length/2), (length/2), 1)[::-1] / 100.0) 
    
    # number of samples to show the cofficient at around the max correlation
    widnow_size = window
    # Get the index of the max  of the average of correlation
    max_corr = np.argmax(all_coeffs.mean(axis = 0))

    # Select the part of correlation array around max correlation    
    coeffs_roi = all_coeffs[:, max_corr - widnow_size : max_corr + widnow_size]
    # Do the same for time label
    times_in_sec = times_in_sec [max_corr - widnow_size : max_corr + widnow_size]
    
    #Heatmap
    fig, axes = plt.subplots()
    axes.set_title('Crosscorrelation max at: %.2f seconds'%((zero_index - max_corr) /100), fontsize = 12)
    
    # Plot the coeff matrix using seaborn heatmap
    g = sns.heatmap(coeffs_roi,yticklabels = 6, xticklabels = 100, ax = axes)
    
    # Arrange the ticks, they will be converted from numerical array to time
    xtick_res = 100 # every second
    ytick_res = 6 # every hour
    
    #g.set(xticks = np.arange(xtick_res, len(coeffs_roi[0,:]), xtick_res))
    #g.set(yticks = np.arange(ytick_res, len(coeffs_roi[:,0]), ytick_res))
    
    # Convert coeff column indices to time
    # get only every xtick_res and ytickres time, so it is readable on the plot
    times_tup = [divmod(np.abs(seconds), 60) for seconds in times_in_sec[::xtick_res]]
    times_str = ["-%02d:%02d" % (t[0], t[1]) for t in times_tup]
    
    #Plot the ticks
    g.set_xticklabels(times_str, rotation = 45)    
    g.set_yticklabels(dates_range[::ytick_res][::-1], rotation = 0)

    axes.set_xlabel('neuroon time shift from psg in "mm:ss"')
    axes.set_ylabel('consecutive 10 minutes epochs')
    
    plt.tight_layout()

    # Time-series like coeffs plot
#    fig_ts, axes_ts = plt.subplots(1)

#    for coefs in coeffs_roi:
#        axes_ts.plot(times_in_sec , coefs, alpha = 0.1, color = 'b')
#    
#    axes_ts.plot(times_in_sec , coeffs_roi.mean(axis = 0), alpha = 1, color = 'b')
#    
#    axes_ts.set_xlim(times_in_sec [0], times_in_sec [-1])
    

