#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 18:58:03 2016

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict

    # Create a dict to have always the same colors for stages
stage_color_dict = {'N1' : 'royalblue', 'N2' :'forestgreen', 'N3' : 'coral', 'rem' : 'plum', 'wake' : 'lightgrey', 'stages_sum': 'dodgerblue'}


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
    axes.set_title('Crosscorrelation max at: %.1f seconds'%((zero_index - max_corr) /100.0), fontsize = 12)
    
    # show plot labels for x,y axis every tickres step
    xtick_res = 100 # every second
    ytick_res = 6 # every hour
    
    # Plot the coeff matrix using seaborn heatmap
    g = sns.heatmap(coeffs_roi,yticklabels = ytick_res, xticklabels = xtick_res, ax = axes)
    
    # Arrange the ticks, they will be converted from numerical array to time

    
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
    
def plot_intersection(intersection, shift_range):
    # Intersection durations are saved in seconds. Divide by 60 to get minutes

    # Shift-overlap plot
    
    fig,axes = plt.subplots(2)
    # Get at which time shift the overlap was the biggest
    max_overlap = shift_range[np.argmax(intersection['stages_sum'])]
    fig.suptitle('max overlap at %i seconds offset'%max_overlap)

    # Plot the sum of interections from all stages on y axis against the shift on x axis
    axes[0].plot(shift_range, np.array(intersection['stages_sum']) / 60.0, label = 'stages sum', color = 'dodgerblue')
    
    # mark max overlap in respect to time shift
    axes[0].axvline(max_overlap, color='k', linestyle='--')

    # There will be two scales on the axis: overlap in minutes for total overlap and z-score overlap for overlap split by sleep stage
    zscore_ax = axes[0].twinx()
    
    # iterate over stages and z score the overlap across different time shifts
    for stage in ['rem', 'N2', 'N3', 'wake']:
        intersect_sum = np.array(intersection[stage])
        z_scored = (intersect_sum - intersect_sum.mean()) / intersect_sum.std()
        zscore_ax.plot(shift_range, z_scored, color = stage_color_dict[stage], label = stage, alpha = 0.5, linestyle = '--')
        

    axes[0].set_ylabel('minutes in the same stage')
    axes[0].set_xlabel('offset in seconds')
    
    axes[0].legend(loc = 'center left')
    
    zscore_ax.grid(b=False)
    zscore_ax.legend()
    # label gets plotted over ticks
    #zscore_ax.set_ylabel('z-scored overlap per stage', rotation = 270)
    fig.tight_layout()    
    # Barplots
    
    #Plot two bars next to each other, left is the overlap in 0 time shift, righis the overlap at max overlap time shift
    
    # Get the overlap durations for 0 time shift
    sums0 = OrderedDict()
    for key, value in intersection.items():
        # divide by 60.0 to convert seconds to minutes and again to convert to hours (remainding fraction are not minutes )
        sums0[key] = np.array(value[np.where(shift_range == 0)[0][0]])/ 60.0 / 60.0
        
#
    width = 0.35 
    ind = np.arange(5)
    colors_inorder = ['dodgerblue', 'coral', 'forestgreen',  'plum', 'lightgrey',]
    #Plot the non shifted overlaps 
    axes[1].bar(left = ind, height = list(sums0.values())[::-1],width = width, alpha = 0.8, 
                tick_label =list(sums0.keys())[::-1], edgecolor = 'black', color= colors_inorder)

    # Get the overlap durations for max overlap time shift
    sumsMax = OrderedDict()
    for key, value in intersection.items():
        sumsMax[key] = np.array(value[np.argmax(intersection['stages_sum'])]) / 60.0 / 60.0
    
    # Plot the shifted overlaps
    axes[1].bar(left = ind +width, height = list(sumsMax.values())[::-1],width = width, alpha = 0.8,
                 tick_label =list(sumsMax.keys())[::-1], edgecolor = 'black', color = colors_inorder)
    
    axes[1].set_xticks(ind + width)
    
    axes[1].set_ylabel('hours in the same stage')
    
    
    plt.tight_layout()
    
def plot_roc(all_confusion_matrix, all_cl_params, plot_tables = False):
    
        roc_fig, roc_axes = plt.subplots()
        
        for stage in list(all_confusion_matrix.keys()):
           
            confusion_matrix= all_confusion_matrix[stage]
            cl_params = all_cl_params[stage]
            
            roc_axes.plot(confusion_matrix['false_positive'], confusion_matrix['true_positive'], marker= 'o', color = stage_color_dict[stage], label = stage)
            
            roc_axes.plot([0,1], [0,1], color = 'black', linestyle = '--', alpha = 0.5)
            roc_axes.set_xlabel('false_positive')
            roc_axes.set_ylabel('true_positive')
            roc_axes.legend()
            roc_axes.set_xlim(0,1)
            roc_fig.savefig('figures/roc/roc.pdf')
            
               
            
            if plot_tables:
                # Confusion matrix
                
                fig, ax= plt.subplots() 
                fig.suptitle('classification performance for %s'%stage, fontweight = 'bold')
                #plt.plot([10,10,14,14,10],[2,4,4,2,2],'r')
                col_labels=['psg positive','psg negative']
                row_labels=['neuroon\n positive','neuroon\n negative']
                
                table_vals=[['TP: %.2f'%confusion_matrix['true_positive'],'FP: %.2f'%confusion_matrix['false_positive']],\
                            ['FN: %.2f'%confusion_matrix['false_negative'], 'TN: %.2f'%confusion_matrix['true_negative']]]
                
                # the rectangle is where I want to place the table
                the_table = ax.table(cellText=table_vals,
                                  colWidths = [0.2]*2,
                                  rowLabels=row_labels,
                                  colLabels=col_labels,
                                  loc='center')
                ax.text(0.75,0.60,'accuracy: %.2f'%cl_params['accuracy'] ,size=14)
                ax.text(0.75, 0.50,'precision: %.2f'%cl_params['precision'],size=14)
                ax.text(0.75,0.40,'recall: %.2f'%cl_params['recall'],size=14)
                the_table.scale(1, 2.5)
                ax.xaxis.set_visible(False) 
                ax.yaxis.set_visible(False)
                fig.savefig('figures/roc/cm_%s.pdf'%stage)

def plot_spectra_by_stage(spectra, frequency,psg_or_noo, min_freq = 1, max_freq =3):

    fig, axes = plt.subplots(2)
    fig.suptitle('neuroon', fontweight = ('bold'))
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Log power spectral density')
    
    axes[1].set_xlabel('sleep stage')
    axes[1].set_ylabel('Delta band power')
    
    
    for stage_name, spectrum in spectra.items():
        #if(stage_name != 'wake'):
            # limit the plot to 0hz - 50hz range
        max_idx = np.argmax(frequency[stage_name][0]  > 50)
    #    for pxx, f in zip(spectrum, frequency):
          #  axes.plot(f, np.log(pxx), color = color_dict[stage_name], alpha = 0.1)
        grand_avg = np.log(spectrum).mean(axis = 0)
        std =np.log(spectrum).std(axis = 0)
        # Select only part of frequencies
        grand_avg = grand_avg[0:max_idx]
#            print(len(grand_avg))

        std = std[0:max_idx]
        #frequency = frequency[stage_name][0]
#           
        lim_freq = frequency[stage_name][0][0:max_idx]
        #print(len(frequency))


        axes[0].plot(lim_freq, grand_avg ,color = stage_color_dict[stage_name], label = stage_name)

        axes[0].fill_between(lim_freq, grand_avg - std, grand_avg + std, color = stage_color_dict[stage_name], alpha = 0.2)
        axes[0].legend()
        
    band = select_band(spectra, frequency['N2'][0],min_freq, max_freq)

    ax = sns.boxplot(data =list(band.values()), ax = axes[1])


    for box, stage in zip(ax.artists, list(band.keys())):
        box.set_facecolor(stage_color_dict[stage])
        box.set_alpha(0.8)
    axes[1].set_xticklabels(list(band.keys()))

    return band

def select_band(spectrum, frequencies, min_freq, max_freq):


    min_freq = np.argmax(frequencies > min_freq)
    max_freq = np.argmax(frequencies > max_freq)

    band = OrderedDict()
    for stage_name, stage_spectrum in spectrum.items():
        
    # Define band as the sum of the power in the bins falling between min and max frequency
        band[stage_name] = np.log(stage_spectrum[:, min_freq : max_freq]).sum(axis = 1)

    return band
    
    


