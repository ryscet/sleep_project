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

stage_color_dict = {'N1' : 'royalblue', 'N2' :'forestgreen', 'N3' : 'coral', 'rem' : 'plum', 'wake' : 'y' }

def get_hipnogram_intersection():
    psg_hipno = ph.parse_psg_stages()
    noo_hipno = ph.parse_neuroon_stages()
    
    noo_hipno = noo_hipno.loc[noo_hipno['stage_shift'] != 'short', :]
    
    psg_hipno['event_number'] = np.array([[i]*2 for i in range(len(psg_hipno) /2)]).flatten()
    noo_hipno['event_number'] = np.array([[i]*2 for i in range(len(noo_hipno) /2 )]).flatten()
    
    combined = psg_hipno.join(noo_hipno, how = 'outer', lsuffix = '_psg', rsuffix = '_neuro')
    
    combined.loc[:, ['stage_num_psg', 'stage_name_psg', 'stage_num_neuro', 'stage_name_neuro', 'event_number_psg', 'event_number_neuro']] = combined.loc[:, ['stage_num_psg', 'stage_name_psg', 'stage_num_neuro', 'stage_name_neuro', 'event_number_psg', 'event_number_neuro']].fillna( method = 'bfill')        
    
    combined.loc[:, ['stage_shift_psg', 'stage_shift_neuro']] = combined.loc[:, ['stage_shift_psg', 'stage_shift_neuro']].fillna( value = 'inside')      
    
    # From the occupied room number subtract the room occupied by another mouse.
    combined['overlap'] = combined['stage_num_psg'] - combined['stage_num_neuro']
    
    same_stage = combined.loc[combined['overlap'] == 0]
    same_stage['event_union'] = same_stage['event_number_psg'] + same_stage['event_number_neuro']
    
    
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
    for key, value in all_durations.items():
        if key != 'wake':
            means[key] = np.array(value).mean()
            stds[key] = np.array(value).std()
            sums[key] = np.array(value).sum()
        
    #results = preapre_results(all_durations)

    fig, axes = plt.subplots(1,2)
    fig.suptitle('Neuroon-psg hinogram comparison', fontweight = 'bold')
    
    axes[1].bar(left = [1,2,3], height = list(means.values()),width = 0.3, alpha = 0.8, align = 'center', color = [stage_color_dict['N2'], stage_color_dict['N3'], stage_color_dict['rem']], tick_label = list(means.keys()), edgecolor = 'black')
    
    (_, caps, _) = axes[1].errorbar([1,2,3], list(means.values()),  yerr= list(stds.values()), fmt='none', ecolor = 'black', alpha = 0.8, elinewidth = 0.8, linestyle = '-.')
    
    axes[1].set_xlim(0.49, 3.51)
    axes[1].set_ylabel('Average hipnogram overlap in minutes')
    axes[1].set_xlabel('sleep stage')
    
    axes[0].bar(left = [1,2,3], height = list(sums.values()),width = 0.3, alpha = 0.8, align = 'center', color = [stage_color_dict['N2'], stage_color_dict['N3'], stage_color_dict['rem']], tick_label = list(means.keys()), edgecolor = 'black')
    axes[0].set_ylabel('Total hipnogram overlap in minutes')
    
    for cap in caps:
        cap.set_color('black')
        cap.set_markeredgewidth(1)
        
    raise_window()

    fig.savefig('figures/hipnogram_comparison/hipno_overlap.pdf')