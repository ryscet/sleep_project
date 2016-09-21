#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 16:07:39 2016

@author: user
"""

import parse_hipnogram as ph

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import OrderedDict 
from datetime import timedelta


stage_color_dict = {'N1' : 'royalblue', 'N2' :'forestgreen', 'N3' : 'coral', 'rem' : 'plum', 'wake' : 'lightgrey', 'stages_sum': 'dodgerblue'}

    
def intersect_shift():
    psg_hipno = ph.parse_psg_stages()
    noo_hipno = ph.parse_neuroon_stages()
    
    intersection = OrderedDict([('wake', []), ('rem',[]), ('N1',[]), ('N2',[]), ('N3', []), ('stages_sum', [])])

    shift_range = np.arange(-500, 100, 10)
    for shift in shift_range:
        sums, _, _ = get_hipnogram_intersection(noo_hipno.copy(), psg_hipno.copy(), shift)
        for stage, intersect_dur in sums.items():
            intersection[stage].append(intersect_dur)
    
    plot_intersection(intersection, shift_range)
    


def plot_intersection(intersection, shift_range):
    
    fig, axes = plt.subplots(2)
    fig.suptitle('Neuroon-Psg overlap with time offset')
    zscore_ax = axes[0].twinx()
    
    for stage in ['rem', 'N2', 'N3', 'wake']:
        intersect_sum = np.array(intersection[stage])
        z_scored = (intersect_sum - intersect_sum.mean()) / intersect_sum.std()
        zscore_ax.plot(shift_range, z_scored, color = stage_color_dict[stage], label = stage, alpha = 0.5, linestyle = '--')
        

    max_overlap = shift_range[np.argmax(intersection['stages_sum'])]
        
    axes[0].plot(shift_range, intersection['stages_sum'], label = 'stages sum', color = 'dodgerblue')
    axes[0].axvline(max_overlap, color='k', linestyle='--')

    axes[0].set_ylabel('minutes in the same sleep stage')
    axes[0].set_xlabel('offset in seconds')
    
    axes[0].legend(loc = 'center right')
    zscore_ax.grid(b=False)
    zscore_ax.legend()
        
    sums0, means0, stds0  = get_hipnogram_intersection(noo_hipno.copy(), psg_hipno.copy(), 0)
#
    width = 0.35 
    ind = np.arange(5)
    colors_inorder = ['dodgerblue', 'lightgrey', 'forestgreen', 'coral',  'plum']
    #Plot the non shifted overlaps 
    axes[1].bar(left = ind, height = list(sums0.values()),width = width, alpha = 0.8, 
                tick_label =list(sums0.keys()), edgecolor = 'black', color= colors_inorder)

    sumsMax, meansMax, stdsMax  = get_hipnogram_intersection(noo_hipno.copy(), psg_hipno.copy(),  max_overlap)
    # Plot the shifted overlaps
    axes[1].bar(left = ind +width, height = list(sumsMax.values()),width = width, alpha = 0.8,
                 tick_label =list(sumsMax.keys()), edgecolor = 'black', color = colors_inorder)
    
    axes[1].set_xticks(ind + width)
    
#    
#    (_, caps, _) = axes[1].errorbar([1,2,3], list(means.values()),  yerr= list(stds.values()), fmt='none', ecolor = 'black', alpha = 0.8, elinewidth = 0.8, linestyle = '-.')
    
                


    raise_window()

    return intersection
    

        

def get_hipnogram_intersection(noo_hipno, psg_hipno, time_shift):

    # Weird behavior with python 3, says TypeError: unsupported type for timedelta seconds component: numpy.int64. Casting to int solves it
    noo_hipno.index = noo_hipno.index + timedelta(seconds = int(time_shift))

    
    combined = psg_hipno.join(noo_hipno, how = 'outer', lsuffix = '_psg', rsuffix = '_neuro')
    
    combined.loc[:, ['stage_num_psg', 'stage_name_psg', 'stage_num_neuro', 'stage_name_neuro', 'event_number_psg', 'event_number_neuro']] = combined.loc[:, ['stage_num_psg', 'stage_name_psg', 'stage_num_neuro', 'stage_name_neuro', 'event_number_psg', 'event_number_neuro']].fillna( method = 'bfill')        
    
    combined.loc[:, ['stage_shift_psg', 'stage_shift_neuro']] = combined.loc[:, ['stage_shift_psg', 'stage_shift_neuro']].fillna( value = 'inside')      
    
    # From the occupied room number subtract the room occupied by another mouse.
    combined['overlap'] = combined['stage_num_psg'] - combined['stage_num_neuro']
    
    same_stage = combined.loc[combined['overlap'] == 0]
    same_stage.loc[:, 'event_union'] = same_stage['event_number_psg'] + same_stage['event_number_neuro']


#    common_window = np.array([noo_hipno.tail(1).index.get_values()[0] - psg_hipno.head(1).index.get_values()[0]],dtype='timedelta64[m]').astype(int)[0]

    all_durations = OrderedDict()

    for stage_name, intersection in same_stage.groupby('event_union'):
            # Subtract the first row timestamp from the last to get the duration. Store as the duration in milliseconds.
            duration = (intersection.index.to_series().iloc[-1]- intersection.index.to_series().iloc[0]).total_seconds()
                                 
            stage_id = intersection.iloc[0, intersection.columns.get_loc('stage_name_neuro')] 
            # Keep appending results to a list stored in a dict. Check if the list exists, if not create it.
            if stage_id not in all_durations.keys():
                all_durations[stage_id] = [duration]
                
            else:   
                all_durations[stage_id].append(duration)
            

    
    means = OrderedDict()
    stds = OrderedDict()
    sums = OrderedDict()
    stages_sum = 0
    #Adding it here so its first in ordered dict and leftmost on the plot
    sums['stages_sum'] = 0
    for key, value in all_durations.items():
        #if key != 'wake':
        means[key] = np.array(value).mean()
        stds[key] = np.array(value).std()
        sums[key] = np.array(value).sum()
        
        stages_sum += np.array(value).sum()   
    
    sums['stages_sum'] = stages_sum
    # Divide total seconds by 60 to get minutes 
    #return stages_sum
    return sums, means, stds
 
    #  plot_hipnogram_intersection(means, stds, sums, common_window)
   

            
#def plot_hipnogram_intersection(means, stds, sums, common_window):
        
    #results = preapre_results(all_durations)

    fig, axes = plt.subplots(1,2, figsize = (20,10))
    fig.suptitle('Neuroon-psg hinogram comparison', fontweight = 'bold')
    
    axes[1].bar(left = [1,2,3], height = list(means.values()),width = 0.3, alpha = 0.8, align = 'center', color = [stage_color_dict['N2'], stage_color_dict['N3'], stage_color_dict['rem']], tick_label = list(means.keys()), edgecolor = 'black')
    
    (_, caps, _) = axes[1].errorbar([1,2,3], list(means.values()),  yerr= list(stds.values()), fmt='none', ecolor = 'black', alpha = 0.8, elinewidth = 0.8, linestyle = '-.')
    
    axes[1].set_xlim(0.49, 3.51)
    axes[1].set_ylabel('Average hipnogram overlap in minutes')
    axes[1].set_xlabel('sleep stage')
    

    axes[0].bar(left = [1,2,3], height = list(sums.values()),width = 0.3, alpha = 0.8, align = 'center', color = [stage_color_dict['N2'], stage_color_dict['N3'], stage_color_dict['rem']], tick_label = list(means.keys()), edgecolor = 'black')


    y_min, y_max =axes[0].get_ylim()
    percent_max = int(y_max / common_window * 100)
    ax2=axes[0].twinx()
    ax2.set_yticks(np.linspace(0, percent_max , 3))
    ax2.set_yticklabels(['%i%%' %i for i in np.linspace(0, percent_max , 3)])
    ax2.grid(b = False)
    print(percent_max)
    #ax2.set_yticks(range(0, percent_max , 5))
    
    axes[0].set_ylabel('Total hipnogram overlap in minutes')
    

    
    for cap in caps:
        cap.set_color('black')
        cap.set_markeredgewidth(1)
        
    raise_window()

    fig.savefig('figures/hipnogram_comparison/hipno_overlap.pdf')