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
import scipy.signal as sig
import parse_hipnogram as ph

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
    
    sig_to_hdf(signal_dict, 'parsed_data/hdf/psg_database.h5')
    
    return signal_dict
    
def stages_to_hdf():
    
    psg_hipno = ph.prep_for_phases(ph.parse_psg_stages())
    neuroon_hipno = ph.prep_for_phases(ph.parse_neuroon_stages())

    path = 'parsed_data/hdf/spectrum_database.h5'
    
    for channel in ['O2-A1', 'O1-A2', 'F4-A1','F3-A2', 'C4-A1', 'C3-A2']:
        psg_signal =  hdf_to_series(channel)
        psg_slices, psg_spectra, psg_frequency =  make_stage_slices(200, psg_hipno, psg_signal, channel)
        spectra_to_hdf(psg_spectra, channel, path, psg_frequency['N1'][0,:])
        
    neuroon_signal =  pd.Series.from_csv('parsed_data/neuroon_parsed.csv')
    neuroon_slices, neuroon_spectra, neuroon_frequency =  make_stage_slices(125, neuroon_hipno, neuroon_signal, 'neuroon')
    
    spectra_to_hdf(neuroon_spectra, 'neuroon', path, neuroon_frequency['N2'][0,:])
    

            
    
def make_stage_slices(sampling_rate, hipno, signal, channel):


    
    slices = {}
    spectra = {}
    frequency = {}
    for name, sleep_stage in hipno.groupby('stage_name'):
        stage_spectra = []
        stage_slices = []
        stage_freqs = []

        for idx, phase_event in sleep_stage.iterrows():
            _slice = np.array(signal.loc[phase_event['starts'] : phase_event['ends']])
            if(_slice.size !=  0):
                stage_slices.append(_slice)
                   
                freqs, pxx = sig.welch(_slice, sampling_rate, nperseg=1024, noverlap = 256) 
                stage_spectra.append(pxx)
                stage_freqs.append(freqs)
            
        slices[name] = stage_slices
        spectra[name] = np.array(stage_spectra)
        frequency[name] = np.array(stage_freqs)
        
    return slices, spectra, frequency
    


def sig_to_hdf(_dict,channel, path):
 
    with h5py.File(path, 'w') as hf:
        for key, value in _dict.items():
            print(key)
            hf.create_dataset(key, data = value)
            
def spectra_to_hdf(_dict,channel, path, frequency):
 
    with h5py.File(path, 'a') as hf:
        g1 = hf.create_group(channel)
        g1.create_dataset('frequency', data = frequency)
        for key, value in _dict.items():
            g1.create_dataset(key, data = value)



def hdf_to_series(channel):
    path = 'parsed_data/hdf/psg_database.h5'
    
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get(channel))
        timestamp = pd.to_datetime(np.array(hf.get('timestamp')))
        
    series = pd.Series(data = data, index = timestamp)
    
    return series

def hdf_to_spectrum_dict(channel):
    path = 'parsed_data/hdf/spectrum_database.h5'
    spectrum_dict = {}
    with h5py.File(path,'r') as hf:
        gp1 = hf.get(channel)
        frequency = np.array(gp1.get('frequency'))
        
        for dataset in gp1.itervalues():
            if(dataset.name.rpartition('/')[-1] != 'frequency'):
                spectrum_dict[dataset.name.rpartition('/')[-1]] = np.array(dataset)
            
    return spectrum_dict, frequency


        
        
    