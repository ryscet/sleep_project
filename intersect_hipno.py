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
                all_durations[stage_id] = [int(duration)]
                
            all_durations[stage_id].append(int(duration))
    
    for key, value in all_durations.items():
        all_durations[key] = np.array(value)
        
    results = preapre_results(all_durations)


        # Plot total time per stage
    #sns.barplot(x = list(all_durations.keys()), y =list(all_durations.values()))
    
    return  results

        
def preapre_results(all_durations):

    # Prepare the dataframe for storing the results
    parsed_durations = pd.DataFrame(columns = ['total_intersect_duration', 'number_of_intersects', 'average_intersect_duration'])
   # Iterate over dict containing list with all meetings durations
    for key, value in all_durations.items():
        # Convert the list to numpy array 
        durations =np.array(value)
        # Store results for the room as a row in a dataframe
        parsed_durations.loc[key, :] =  [durations.sum() / 1000.0,  len(durations), durations.mean() / 1000.0]  
    return parsed_durations