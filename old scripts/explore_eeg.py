# -*- coding: utf-8 -*-
"""
Open the raw data and parse it into pandas series and a numpy array of slices. 
Optionally change the timestamp of the neuroon and/or trim the signal to the common time window.
Save the results in parsed_data folder.
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



# Parse neuroon
#neuroon, neuroon_slices = parse_neuroon()   



def parallel_parse():
    """Run to save all the parsed psg channels and neuroon in parsed_data folder.""" 
    # Not sure if its actualy faster than serially
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(parse_psg)(name) for name in ['O2-A1', 'O1-A2', 'F4-A1','F3-A2', 'C4-A1', 'C3-A2'])
    parse_neuroon()
        

def parse_neuroon(cut_to_match = False, shift_time = None):
    """Open the raw csv file and convert it to pandas series with datetime index"""
    neuroon = pd.Series.from_csv('neuroon_signals/night_01/neuroon_signal.csv',header = 0, infer_datetime_format=True)
    # Add the twp hours differnece based on time-zone mismatch between recording devices
    neuroon.index = pd.to_datetime(neuroon.index.astype(int), unit='ms') + pd.Timedelta(hours = 2)
    
    # Correct the timestamp based on cross correlation function
    if(shift_time):
        neuroon.index = neuroon.index + pd.Timedelta(milliseconds = shift_time)
        
    # now in nanoVolts, scale to microVolts
    neuroon = neuroon / 1000.0

#TODO cut_to_match used for which analysis?
    # Cut the common window of time from neuroon and psg signals.
    # Neuroon starts later and ends earlier than psg. 
    # To get window of time covered by both recordings, select the first time recorded of psg and last time recorded by neuroon
    
    if cut_to_match:
        neuroon = neuroon.loc[psg_hipno.head(1).index.values[0] : noo_hipno.tail(1).index.values[0]]
    
    # get the signal duration time
    sig_duration = np.array(neuroon.tail(1).index.values[0] - neuroon.head(1).index.values[0],dtype='timedelta64[ms]').astype(int)

    # Create slices for future analysis: frequency analysis, pca    
#TODO writhe which scripts use slices
    slices =create_slices(neuroon, sig_duration, 'neuroon')
    
    neuroon.to_csv('parsed_data/neuroon_parsed.csv')
    
    return neuroon,slices
    
def make_psg_slices(channel = 'F3-A2', cut_to_match = False):
    """Before running this a hdf database must be made from psg_signal.edf. Use psg_edf_2_hdf.py for createing the hdf database."""
    
                              
        # get the signal duration time
    sig_duration = np.array(psg_channel.tail(1).index.values[0] - psg_channel.head(1).index.values[0],dtype='timedelta64[ms]').astype(int)


    # Create slices for analysis: frequency analysis, cross-correlation, pca    
    slices = create_slices(psg_channel, sig_duration, 'psg_' + channel)


    return slices

   

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
    """Used to iterate over a list elements organized in pairs"""
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
    
    
