# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 12:47:44 2016

@author: user
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


stage_to_num = {'W':0, 'R':1, 'N1':2 , 'N2':3, 'N3':4 }
num_to_stage = {0: 'wake', 1 : 'rem', 2 :'N1', 3 : 'N2', 4: 'N4'}

def parse_neuroon_stages():
    neuroon_stages = pd.read_csv('neuroon_signals/night_01/neuroon_stages.csv', index_col = 0)

    # add two hours because time was saved in a different timezone
    neuroon_stages['timestamp'] = pd.to_datetime(neuroon_stages['timestamp'].astype(int), unit='ms', utc=True) + pd.Timedelta(hours = 2)

    # Change from negative to positive stages coding
    neuroon_stages.loc[:, 'stage_num'] = np.abs( neuroon_stages['stage'])

    #Mark the row where a new stage startes
    neuroon_stages['stage_start'] = neuroon_stages['stage_num'] - neuroon_stages['stage_num'].shift(1)
    neuroon_stages['stage_end'] = neuroon_stages['stage_num'] - neuroon_stages['stage_num'].shift(-1)

    neuroon_stages.loc[neuroon_stages['stage_start'] != 0, 'stage_shift'] = 'start'
    neuroon_stages.loc[neuroon_stages['stage_end'] != 0, 'stage_shift'] = 'end'

    # Find stages that lasted for one sampling interval, 30 sec
    neuroon_stages.loc[(neuroon_stages.loc[:,'stage_start'] != 0) & (neuroon_stages.loc[:,'stage_end'] != 0), 'stage_shift'] = 'short'

    # Leave only the rows where the stage shifted    
    neuroon_stages = neuroon_stages[pd.notnull(neuroon_stages['stage_shift'])]
    
    # Add the column with string names for stages
    neuroon_stages['stage_name'] = neuroon_stages['stage_num'].replace(num_to_stage)

    neuroon_stages.set_index(neuroon_stages['timestamp'], inplace = True, drop = True)
        
    # Drop the columns used for stage_shift calculation and the timestamp since it's the index now
    neuroon_stages.drop(['stage_start', 'stage_end', 'stage'], axis = 1, inplace = True)
    
    neuroon_stages.to_csv('parsed_data/' + 'neuroon_hipnogram.csv', index = False)

    return neuroon_stages


def parse_psg_stages():
    psg_stages = pd.read_csv('neuroon_signals/night_01/psg_stages.csv', \
                             header = None, names = ['timestamp', 'stage'])

    # Select only the rows describing the sleep stage
    psg_stages = psg_stages.loc[psg_stages['stage'].str.contains('Stage'), :]
    # Subtract unused parts of the string
    psg_stages.loc[:, 'stage' ]  = psg_stages['stage'].str.replace('Stage - ', '')
    psg_stages = psg_stages.loc[psg_stages['stage'] != 'No Stage', :]

    # Parse the time info from the string timestamp to the datetime object
    # We need to add the month and day info. Note that it changes after midnight 00:00:00
    #Get rid of empty spaces
    psg_stages.loc[:, 'timestamp']  = psg_stages['timestamp'].str.replace(' ', '')

    # Get only the hours to find a change of date index
    psg_stages['hour'] = pd.to_numeric(psg_stages['timestamp'].str[0:2], errors = 'coerce')
    # Find the index where the day changes, i.e. the first time the hour is greater than 00:00:00
    new_date = np.where(psg_stages['hour'] == 0)[0][0]
    # Add the day info to the datetime, accounting for the change after midnight
    psg_stages.iloc[0 : new_date, 0] = '2016-06-20 ' + psg_stages.iloc[0 : new_date, 0]
    psg_stages.iloc[new_date::, 0] = '2016-06-21 ' + psg_stages.iloc[new_date ::, 0]

    # Convert the string timestamp to datetime object
    psg_stages['parsed_timestamp'] =  pd.to_datetime(psg_stages['timestamp'],format = '%Y-%m-%d %H:%M:%S.%f')

    # Create numeric column with stage info
    psg_stages['stage_num'] = psg_stages['stage'].replace(stage_to_num)
    # Change the naming convention so it is the same between neuroon and psg hipnograms
    psg_stages['stage_name'] = psg_stages['stage_num'].replace(num_to_stage)
    
    # Mark the starting and ending time of each stage
    psg_stages['stage_shift']  ='start'
    
    # Stage start will always be followed by an end event.
    # Stage starts will have even indices, ends will have odd indices
    psg_stages['order'] = range(0, len(psg_stages) *2, 2)
    
    # Create the copy of start events and assign to them timestamp of the nex row, then they become end events.
    psg_copy = psg_stages.copy()
    psg_copy['stage_shift'] = 'end'
    psg_copy['order']= range(1, len(psg_stages) * 2, 2)
    # Here we assign the next row timestamp, and subtract one millisecond from it - pandas does not comply with duplicate indices, 
    # and subtracting one millisecond from an end event wil make it have a different timestamp from the next start event.
    psg_copy['parsed_timestamp'] = psg_copy['parsed_timestamp'].shift(-1) - pd.Timedelta(milliseconds = 1)

    # Combine starts and ends and sort them by order column, to have start,end,start,end,start,end, etc... order.
    psg_stages = psg_stages.append(psg_copy).sort('order')
    
    # Deal with the last timestamp which is Nan because of .shift() function
    psg_stages.iloc[-1, psg_stages.columns.get_loc('parsed_timestamp')] = psg_stages.iloc[-2, psg_stages.columns.get_loc('parsed_timestamp')] - pd.Timedelta(milliseconds = 1)

    
    psg_stages.set_index(psg_stages['parsed_timestamp'], inplace = True, drop = True)

    psg_stages.drop(['hour', 'order', 'timestamp', 'stage'], axis = 1, inplace = True)

    psg_stages.to_csv('parsed_data/' +'psg_hipnogram.csv', index = False)
    return psg_stages


def plot_hipnograms():

    plt.style.use('ggplot')
    psg = parse_psg_stages()
    neur = parse_neuroon_stages()

    fig, axes = plt.subplots(2, sharex = True, sharey = True)
    axes[0].plot(neur['timestamp'] , neur['stage'], color = 'blue', alpha = 0.7, label = 'NeuroOn')
    axes[1].plot(psg['parsed_timestamp'], psg['stage_num'], color = 'green', alpha = 0.7, label = 'PSG')

    axes[0].set_ylabel('NeuroOn hipnogram')
    axes[1].set_ylabel('PSG hipnogram')

    axes[1].set_xlabel('sleep time')

    axes[0].set_ylim(-0.1, 4.1)
    plt.yticks(range(0, 5), ['awake', 'REM', 'stage 1', 'stage 2', 'stage 3'])
    plt.legend()

   # fig2, axes2 = plt.subplots()

    #axes2.plot(neur['stage'].reindex(neur['timestamp']) - psg['stage'].reindex(psg['parsed_timestamp']))



