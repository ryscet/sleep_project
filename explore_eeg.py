# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 11:05:04 2016

@author: ryszardcetnarski
"""

import parse_hipnogram as ph
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import tee
import pyedflib
import deepdish as dd
import h5py

psg_hipno = ph.parse_psg_stages()
noo_hipno = ph.parse_neuroon_stages()

def parse_neuroon():


    neuroon = pd.read_csv('neuroon-signals/night_01/neuroon_signal.csv', dtype = {'signal': np.int32})
    
    neuroon['timestamp'] = pd.to_datetime(neuroon['timestamp'].astype(int), unit='ms', utc=True) + pd.Timedelta(hours = 2)
    neuroon.set_index(neuroon['timestamp'], inplace = True, drop = True)
    
    # Neuroon starts later and ends earlier than psg. 
    # To get window of time covered by both recordings, select the first time recorded of psg and last time recorded by neuroon
    
    #neuroon = neuroon.loc[psg_hipno.head(1).index.values[0] : noo_hipno.tail(1).index.values[0], 'signal']
    neuroon = neuroon.loc[psg_hipno.head(1).index.values[0] : psg_hipno.head(2).index.values[1]]
    
    # get the signal duration time
    sig_duration = np.array(neuroon.tail(1).index.values[0] - neuroon.head(1).index.values[0],dtype='timedelta64[ms]').astype(int)

    #sampling_freq_hz = int(round(1000.0 / (sig_duration / len(neuroon)), 1))
    
    slices = create_slices(neuroon, sig_duration )
    
    return slices

   


    
def create_slices(signal, sig_duration):
    

    # in milliseconds
    slice_time_ms = 30000
    # number of slices to divide the signal into bins of length specified by slice_time_ms   
    n_periods = round(sig_duration / slice_time_ms,1)
    
    
    date_rng = pd.date_range(psg_hipno.head(1).index.values[0], periods= n_periods, freq= str(slice_time_ms) + 'ms')
    
    slices = []
    for start, end in pairwise(date_rng):
        #slices.append(signal.loc[start:end, 'signal'].as_matrix())
        slices.append(signal.loc[start:end, 'signal'])
    
    #return np.array(slices)
    return np.array(slices)

    
    
def edf_to_hdf():
    """Opens the psg edf file and parses it into a hdf5 database - this way a single channel can be loaded."""
    
    path = '/Users/ryszardcetnarski/Desktop/sleep_project/neuroon-signals/night_01/psg_signal.edf'
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
    path = '/Users/ryszardcetnarski/Desktop/sleep_project/neuroon-signals/night_01/psg_signal3.h5'
 
    with h5py.File(path, 'w') as hf:

        for key, value in psg.items():
            print(key)
            hf.create_dataset(key, data = value)
            
            
def load_psg_hdf(channel = 'F3-A2'):
    path = '/Users/ryszardcetnarski/Desktop/sleep_project/neuroon-signals/night_01/psg_signal3.h5'
    
    with h5py.File(path,'r') as hf:
        signal = np.array(hf.get(channel))
        timestamp = pd.datetime(np.array(hf.get('timestamp')))
        
    return timestamp[0:100]
    
    
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
    
def Find_Closest_Sample(signal, event_time):
    #Subtract event time from an array containing all signal timestamps. Then the index at which the result of this subtraciton will be closest to 0 will be an index of event in the signal
    return np.argmin(np.abs(signal.index - event_time))