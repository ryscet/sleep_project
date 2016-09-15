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

stage_color_dict = {'N1' : 'royalblue', 'N2' :'forestgreen', 'N3' : 'coral', 'rem' : 'plum', 'wake' : 'y' }

    
def intersect_shift():
    intersection = OrderedDict()
    Psg_Hipno = ph.parse_psg_stages()
    Noo_Hipno = ph.parse_neuroon_stages()
    

    
    fig, axes = plt.subplots()
    fig.suptitle('Neuroon-Psg overlap with time offset')
    for shift in np.arange(-600, 600, 10):
        print(shift)
        intersection[shift] = get_hipnogram_intersection(Noo_Hipno.copy(), Psg_Hipno.copy(), shift)
        
    axes.plot(list(intersection.keys()), list(intersection.values() ))
    axes.set_ylabel('minutes in the same sleep stage')
    axes.set_xlabel('offset in seconds')
    raise_window()
    return intersection
        

def get_hipnogram_intersection(noo_hipno, psg_hipno, time_shift):

    noo_hipno.index = noo_hipno.index + timedelta(seconds = time_shift)
    #noo_hipno = noo_hipno.loc[noo_hipno['stage_shift'] != 'short', :]
    
  #  psg_hipno['event_number'] = np.array([[i]*2 for i in range(len(psg_hipno) /2)]).flatten()
   # noo_hipno['event_number'] = np.array([[i]*2 for i in range(len(noo_hipno) /2 )]).flatten()
    
    combined = psg_hipno.join(noo_hipno, how = 'outer', lsuffix = '_psg', rsuffix = '_neuro')
    
    combined.loc[:, ['stage_num_psg', 'stage_name_psg', 'stage_num_neuro', 'stage_name_neuro', 'event_number_psg', 'event_number_neuro']] = combined.loc[:, ['stage_num_psg', 'stage_name_psg', 'stage_num_neuro', 'stage_name_neuro', 'event_number_psg', 'event_number_neuro']].fillna( method = 'bfill')        
    
    combined.loc[:, ['stage_shift_psg', 'stage_shift_neuro']] = combined.loc[:, ['stage_shift_psg', 'stage_shift_neuro']].fillna( value = 'inside')      
    
    # From the occupied room number subtract the room occupied by another mouse.
    combined['overlap'] = combined['stage_num_psg'] - combined['stage_num_neuro']
    
    same_stage = combined.loc[combined['overlap'] == 0]
    same_stage['event_union'] = same_stage['event_number_psg'] + same_stage['event_number_neuro']

#    # Plot intersection example
#    fig, axes = plt.subplots(2, sharex = True)
#    axes[0].plot(combined.index, combined['stage_num_psg'], 'r', alpha = 0.5, label = 'psg')
#    axes[0].plot(combined.index, combined['stage_num_neuro'], 'b', alpha = 0.5, label = 'neuroon') 
#    axes[1].plot(combined.index, combined['stage_num_psg'] - combined['stage_num_neuro'], 'g', alpha = 0.5, label = 'stage intersection')
#    axes[0].set_ylabel('room id')
#    axes[1].set_ylabel('room a - room b')
#    plt.legend()
#    raise_window()
    # Get the time where overlap was possible, i.e. exclude time when only one device was recording
    # The precision is rounded down to the last full minute
    
    
    #common_window = np.array([noo_hipno.tail(1).index.get_values()[0] - psg_hipno.head(1).index.get_values()[0]],dtype='timedelta64[m]').astype(int)[0]

    all_durations = OrderedDict()
    for stage_name, intersection in same_stage.groupby('event_union'):
            # Subtract the first row timestamp from the last to get the duration. Store as the duration in milliseconds.
            duration = np.array([intersection.tail(1).index.get_values()[0] - intersection.head(1).index.get_values()[0]], dtype='timedelta64[ms]').astype(int)[0]
    
            stage_id = intersection.iloc[0, intersection.columns.get_loc('stage_name_neuro')] 
            # Keep appending results to a list stored in a dict. Check if the list exists, if not create it.
            if stage_id not in all_durations.keys():
                # Divide to get minutes
                all_durations[stage_id] = [duration / 1000.0 / 60.0]
                
            all_durations[stage_id].append(duration / 1000.0 / 60.0)
    
    means = OrderedDict()
    stds = OrderedDict()
    sums = OrderedDict()
    grand_sum = 0
    for key, value in all_durations.items():
        if key != 'wake':
            means[key] = np.array(value).mean()
            stds[key] = np.array(value).std()
            sums[key] = np.array(value).sum()
            grand_sum = grand_sum + np.array(value).sum()   
   # print(grand_sum)
    
    #plot_hipnogram_intersection(means, stds, sums, common_window)
    return grand_sum, sums
            
  #  plot_hipnogram_intersection(means, stds, sums, common_window)
   
   # return sums

            
def plot_hipnogram_intersection(means, stds, sums, common_window):
        
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