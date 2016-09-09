#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 21:04:24 2016

@author: ryszardcetnarski

Converts the psg recordigng saved as edf to hdf5 database. HDF5 allows to load parts of the data, so it will be quicker than edf, where you have to load all signals from polysomnograph.

"""

import pyedflib
import pandas as pd
import h5py
import numpy as np

def edf_to_hdf():
    """Opens the psg edf file and parses it into a hdf5 database - this way a single channel can be loaded."""
    
    path = 'neuroon_signals/night_01/psg_signal.edf'
    f = pyedflib.EdfReader(path)

    signal_labels = f.getSignalLabels()
        
    signal_dict = {}
    #print('Channels:')
    for idx, name in enumerate(signal_labels):
        print(name.decode("utf-8"))
        signal_dict[name.decode("utf-8")] = f.readSignal(idx)
    
    # Create a timestamp array which will be common to all psg recorded signals
    # All the sampling frequencies are the same, for eeg and non-eeg_signals
    sig_freq = f.getSampleFrequency(0)
    # Get the time of the first sample, also common to all signals
    first_sample_time = f.getStartdatetime()

    # Divide 1000 by signal frequency to get sampling interval
    # Construct an array of datetimes starting at first sample time and incrementing by sampling interval
    sig_timestamp = pd.date_range(first_sample_time, periods= len(signal_dict[next(signal_dict.__iter__())]), freq= str(int(1000/sig_freq )) + 'ms')
    
    
    # Store time info in the dict to be converted to hdf    
    # convert datetime array to unix_time
    signal_dict['timestamp'] = sig_timestamp.astype(np.int64)

    # Close edf reader to avoid errors on consecutive runs
    f._close()
    
    dict_to_hdf(signal_dict)
    
    return signal_dict

def dict_to_hdf(psg):
    path = 'neuroon_signals/night_01/psg_signal3.h5'
 
    with h5py.File(path, 'w') as hf:

        for key, value in psg.items():
            print(key)
            hf.create_dataset(key, data = value)