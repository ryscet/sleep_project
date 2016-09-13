# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 11:05:04 2016

@author: ryszardcetnarski

Open the raw data and parse it into pandas series and a numpy array of slices. Save the results in parsed_data folder.
analyze.py and cross_correlate.py use the data from parsed_data folder.

"""

import parse_hipnogram as ph
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import tee
import h5py
from joblib import Parallel, delayed
import multiprocessing


# Parse hipnograms
psg_hipno = ph.parse_psg_stages()
noo_hipno = ph.parse_neuroon_stages()
# Parse neuroon
#neuroon, neuroon_slices = parse_neuroon()   


def RunAll(name):
    print(name)
    parse_psg(name)

def parallel_parse():
 
    
    #Parse all psg channels using
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(RunAll)(name) for name in ['O2-A1', 'O1-A2', 'F4-A1','F3-A2', 'C4-A1', 'C3-A2'])
    parse_neuroon()
        



        

def parse_neuroon(cut_to_match = False):


    neuroon = pd.Series.from_csv('neuroon_signals/night_01/neuroon_signal.csv',header = 0, infer_datetime_format=True)
    neuroon.index = pd.to_datetime(neuroon.index.astype(int), unit='ms') + pd.Timedelta(hours = 2)
    
    # now in nanoVolts, scale to microVolts
    neuroon = neuroon / 1000.0

        
    # Cut the common window of time from neuroon and psg signals
    # Neuroon starts later and ends earlier than psg. 
    # To get window of time covered by both recordings, select the first time recorded of psg and last time recorded by neuroon
    
    if cut_to_match:
        neuroon = neuroon.loc[psg_hipno.head(1).index.values[0] : noo_hipno.tail(1).index.values[0]]
    
    #neuroon = neuroon.loc[psg_hipno.head(1).index.values[0] : psg_hipno.head(2).index.values[1]]
    
    # get the signal duration time
    sig_duration = np.array(neuroon.tail(1).index.values[0] - neuroon.head(1).index.values[0],dtype='timedelta64[ms]').astype(int)

    # Create slices for analysis: frequency analysis, cross-correlation, pca    
    slices =create_slices(neuroon, sig_duration, 'neuroon')
    
    neuroon.to_csv('parsed_data/neuroon_parsed.csv')
    
    return neuroon,slices
    
def parse_psg(channel = 'F3-A2', cut_to_match = False):
    path = 'neuroon_signals/night_01/psg_signal3.h5'
    
    with h5py.File(path,'r') as hf:
        signal = np.array(hf.get(channel))
        timestamp = pd.to_datetime(np.array(hf.get('timestamp')))
        
    psg_channel = pd.Series(data = signal, index = timestamp, name = channel, dtype = np.int32)
        
    if cut_to_match :
    # Cut the common window of time from neuroon and psg signals
        psg_channel = psg_channel.loc[psg_hipno.head(1).index.values[0] : noo_hipno.tail(1).index.values[0]]
    
                                  
        # get the signal duration time
    sig_duration = np.array(psg_channel.tail(1).index.values[0] - psg_channel.head(1).index.values[0],dtype='timedelta64[ms]').astype(int)


    # Create slices for analysis: frequency analysis, cross-correlation, pca    
    slices = create_slices(psg_channel, sig_duration, 'psg_' + channel)

    psg_channel.to_csv('parsed_data/psg_' + channel +'.csv')


    return psg_channel, slices

   

def create_slices(signal, sig_duration, psg_or_neuroon):
    

    # in milliseconds
    slice_time_ms = 30000 #*2 * 60 # uncomment to get 1h duration, otherwise it's 30 sec in ms
    # number of slices to divide the signal into bins of length specified by slice_time_ms   
    n_periods = round(sig_duration / slice_time_ms,1)
    
    
    date_rng = pd.date_range(psg_hipno.head(1).index.values[0], periods= n_periods, freq= str(slice_time_ms) + 'ms')
    
    slices = []
    for start, end in pairwise(date_rng):
        slices.append(signal.loc[start:end])
        
    slices = np.array(slices)
    np.save('parsed_data/' + psg_or_neuroon + '_slices', slices)
   
    #return slices
    return np.array(slices)

    
    
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
    
def Find_Closest_Sample(signal, event_time):
    #Subtract event time from an array containing all signal timestamps. Then the index at which the result of this subtraciton will be closest to 0 will be an index of event in the signal
    return np.argmin(np.abs(signal.index - event_time))
    
